"""Raw data loading, streaming statistics, and the physical cut.

This module is the bottom of the pipeline: it turns the on-disk parameter
(.txt) and data-vector (.npy) dumps into the in-memory "source" dicts the
rest of the package consumes, without ever loading the (memmap-sized) dv
file whole. The functions stream_chunks, stream_stats, and param_stats
compute per-column normalization stats over selected rows; stage_source
materializes a row subset in RAM if it fits (else keeps the memmap);
phys_cut_idx keeps the rows with omega_b h^2 < cut; and read_param_names
reads the parameter names off a covmat header. load_source is the
orchestrator: it memmaps, cuts, sizes, and stages one source into a
{C, dv, idx, (+ means)} dict.

PS: a memmap (memory-mapped array) is a NumPy array backed by the file on
disk and read in slices, so an array larger than RAM is never loaded
whole.
"""

import os

import numpy as np
import psutil
import torch


def stream_chunks(idx, chunk):
  """
  Yield the row indices in sorted blocks of `chunk` rows.

  A generator (it `yield`s, so blocks are produced lazily,
  one at a time, never all at once). Each block is sorted so
  that when the caller uses it to index a memmap, the reads
  walk the file in increasing order -- sequential disk
  access instead of random seeks.

  Arguments:
    idx   = 1D array of row indices (any order).
    chunk = number of indices per block.

  Yields:
    a sorted sub-array of up to `chunk` indices.
  """
  # step through idx in windows of `chunk`; the last window
  # may be short. np.sort puts each window's indices in
  # increasing order, so reading those rows walks the file
  # front-to-back (sequential) rather than jumping around.
  for a in range(0, len(idx), chunk):
    yield np.sort(idx[a:a+chunk])


def stream_stats(mm, idx, method=1, CHUNK=10000):
  """
  Per-column normalization stats over selected rows.

  Streams the chosen rows CHUNK at a time and accumulates
  the statistics, so `mm` (which may be a memmap larger than
  RAM) is never fully loaded. `method` picks the scheme:
    1 = z-score  -> returns (mean, std)
    2 = min-max  -> returns (min,  max - min)
  The caller then normalizes a row as (x - offset) / scale.

  Arguments:
    mm     = 2D array indexable by row (in-RAM or memmap);
             columns are the quantities to summarize.
    idx    = row indices to include in the statistics.
    method = 1 for z-score, 2 for min-max.
    CHUNK  = rows read per streamed block.

  Returns:
    (offset, scale) as float32 torch tensors, one entry
    per column.
  """
  n = len(idx)               # total rows summarized
  ncols = mm.shape[1]        # one stat per column

  if method == 1:
    # one-pass mean/variance via running sums. float64
    # accumulators so summing many rows does not lose
    # precision or overflow.
    s1 = np.zeros(ncols, dtype="float64")   # sum of x
    s2 = np.zeros(ncols, dtype="float64")   # sum of x^2
    for rows in stream_chunks(idx=idx, chunk=CHUNK):
      # read this block of rows and upcast to float64.
      x = np.asarray(mm[rows], dtype="float64")
      s1 += x.sum(axis=0)            # accumulate sum
      s2 += (x * x).sum(axis=0)      # accumulate sum of sq

    mean = s1 / n
    # variance = (sum_sq - sum^2/n) / (n-1): the one-pass
    # computational form of the unbiased sample variance;
    # sqrt gives the per-column std.
    std  = np.sqrt((s2 - s1 * s1 / n) / (n - 1))
    offset, scale = mean, std
  elif method == 2:
    # running min/max, started at +inf / -inf so the first
    # block always replaces them.
    mn = np.full(ncols,  np.inf, dtype="float64")
    mx = np.full(ncols, -np.inf, dtype="float64")
    for rows in stream_chunks(idx=idx, chunk=CHUNK):
      x = np.asarray(mm[rows], dtype="float64")
      mn = np.minimum(mn, x.min(axis=0))   # tighten the min
      mx = np.maximum(mx, x.max(axis=0))   # tighten the max
    offset, scale = mn, mx - mn

  # hand back float32 torch tensors (the model's dtype).
  off = torch.from_numpy(offset.astype("float32"))
  scl = torch.from_numpy(scale.astype("float32"))
  return off, scl


def param_stats(arr, idx, method=1):
  # Per-column normalization stats for the cosmo params.
  #   1 = z-score  -> returns (mean, std)
  #   2 = min-max  -> returns (min,  max - min)
  # Caller normalizes as: (x - offset) / scale.
  # float64 for accurate totals, then hand back float32.
  a = np.asarray(arr[idx], dtype="float64")
  if method == 1:
    offset = a.mean(axis=0)
    scale  = a.std(axis=0, ddof=1)
  elif method == 2:
    offset = a.min(axis=0)
    scale  = a.max(axis=0) - offset
  off = torch.from_numpy(offset.astype("float32"))
  scl = torch.from_numpy(scale.astype("float32"))
  return off, scl


def stage_source(C, dv, idx, ram_frac=0.7):
  """
  Stage a source's used rows in RAM if they fit, else leave
  them on disk.

  The full dv dump may be a memmap too big for RAM, but the
  used subset (idx -- e.g. a 1/10 training cut) is far smaller
  and usually fits. If the subset's bytes are below ram_frac
  of the available RAM, materialize the compact subset (C and
  dv restricted to the used rows, held in RAM) and reindex
  locally; otherwise return the inputs unchanged so the
  loaders stream dv from the memmap by global index. Either
  way idx matches its own C/dv, so the rest of the pipeline is
  identical.

  Arguments:
    C        = full parameter dump, (N, Ncosmo).
    dv       = full dv dump, (N, Ndv); ndarray or np.memmap.
    idx      = global row indices to use.
    ram_frac = fraction of available RAM the materialized
               subset may occupy (default 0.7).

  Returns:
    C_src, dv_src, idx_src = compact in-RAM subset with
      idx_src = arange(n_used) when it fits; otherwise
      (C, dv, idx) unchanged (dv still the memmap, idx still
      global).
  """
  rows   = np.sort(np.unique(idx))      # sorted -> sequential
  nbytes = rows.size * dv.shape[1] * dv.dtype.itemsize
  avail  = psutil.virtual_memory().available
  if nbytes < ram_frac * avail:
    # materialize the subset into RAM and reindex locally.
    return (np.asarray(C[rows]),
            np.asarray(dv[rows]),
            np.arange(rows.size))
  # too big for RAM: keep full arrays + global index; the
  # loaders stream dv from disk.
  return C, dv, idx


def phys_cut_idx(C, idx, names, cut):
  """
  Keep only rows below a physical-baryon-density cut:
  omega_b h^2 = Omega_b * (H0/100)^2 < cut.

  Restricts a row selection to the physically relevant
  region. The high-omega_b h^2 cosmologies (a sparse, ~2x
  Planck corner) fail catastrophically and no real posterior
  visits them, so they are dropped from both the training
  data and the reported metric.

  Arguments:
    C     = full parameter dump, (N, n_param), physical
            units, column order given by `names`.
    idx   = candidate row indices into C (e.g. a shuffle).
    names = parameter column names in C's column order; used
            to locate the omegab and H0 columns by name.
    cut   = upper bound on omega_b h^2 (rows >= cut dropped).

  Returns:
    the subset of idx with omega_b h^2 < cut, in idx's order.
  """
  i_ob = names.index("omegab")     # baryon density column
  i_h0 = names.index("H0")         # Hubble column (km/s/Mpc)
  obh2 = C[idx, i_ob] * (C[idx, i_h0] / 100.0) ** 2
  return idx[obh2 < cut]


def read_param_names(covmat_path, comment="#"):
  """
  Parameter column names from a covmat header line.

  Reads only the first line of the covmat file, strips the leading
  comment marker, and splits on whitespace -- the column order the
  parameter arrays (and ParamGeometry) use.

  Arguments:
    covmat_path = path to the covmat file; its first line lists
                  the column names, prefixed by `comment`.
    comment     = the leading marker to strip (default "#").

  Returns:
    a list of parameter-name strings, in column order.
  """
  with open(covmat_path) as f:
    return f.readline().lstrip(comment).split()


def load_source(dv_path, params_path, names, cut, divisor=None,
                gen=None, ram_frac=0.7, with_means=False,
                param_cols=slice(2, -1), verbose=True, n_keep=None):
  """
  Load, physically cut, and stage one dv/param source.

  Memmaps the dv dump (never reading it whole), loads the params
  and keeps only the modeled columns, applies the omega_b h^2
  cut, keeps the first N // divisor cut rows of a fixed shuffle,
  stages that subset in RAM if it fits (else leaves the memmap),
  and -- when with_means -- computes the centering means the
  geometry needs. Orchestrates phys_cut_idx / stage_source /
  stream_stats / param_stats above.

  Arguments:
    dv_path     = .npy data-vector dump (memmapped).
    params_path = parameter text file; param_cols selects the
                  modeled columns out of it (the file also carries
                  weight / lnp / chi2 bookkeeping columns).
    names       = parameter column names (covmat order -- the
                  order of the KEPT columns), used by phys_cut_idx
                  to find the omegab / H0 columns.
    cut         = upper bound on omega_b h^2 (rows >= cut dropped).
    divisor     = keep N // divisor rows (10 -> ~1/10 for train); pass
                  this OR n_keep (exactly one).
    gen         = torch.Generator seeding the cut+shuffle (required).
    ram_frac    = fraction of available RAM stage_source may fill
                  (default 0.7).
    with_means  = if True, also compute C_mean / dv_mean (the
                  training source needs them; val does not).
    param_cols  = column selector applied to the loaded params
                  (default slice(2, -1): drop the leading weight /
                  lnp and the trailing chi2 column).
    verbose     = if True (default), print a one-line summary of
                  the staged source (shapes, rows, in-RAM).
    n_keep      = absolute number of rows to keep (overrides divisor;
                  for a learning-curve sweep at explicit sizes). Pass
                  this OR divisor (exactly one).

  Returns:
    a source dict {"C", "dv", "idx"} (plus "C_mean" / "dv_mean"
    when with_means), ready for build_loaders / run_emulator.
  """
  # require a generator and exactly one sizing rule.
  if gen is None:
    raise ValueError("load_source needs a torch.Generator (gen=)")
  if (divisor is None) == (n_keep is None):
    raise ValueError(
      "pass exactly one of divisor (keep N // divisor rows) or "
      "n_keep (an absolute row count)")
  dv = np.load(dv_path, mmap_mode="r", allow_pickle=False)
  # keep only the modeled parameter columns (param_cols drops the
  # weight / lnp / chi2 bookkeeping columns by default).
  C  = np.loadtxt(params_path, dtype="float32")[:, param_cols]
  if C.shape[0] != dv.shape[0]:
    raise ValueError(
      f"incompatible files: {params_path} has {C.shape[0]} rows, "
      f"{dv_path} has {dv.shape[0]}")

  n     = C.shape[0]
  order = torch.randperm(n, generator=gen).numpy()
  phys  = phys_cut_idx(C=C, idx=order, names=names, cut=cut)
  # rows to keep: an absolute n_keep, or N // divisor.
  keep  = int(n_keep) if n_keep is not None else int(n // divisor)
  if len(phys) < keep:
    raise ValueError(
      f"physical pool too small: {len(phys)} < {keep}")
  idx = phys[:keep]

  # stage the cut rows in RAM if they fit; else keep the memmap +
  # the global index (the rest of the pipeline is identical).
  C_src, dv_src, idx_src = stage_source(
    C=C, dv=dv, idx=idx, ram_frac=ram_frac)
  src = {"C": C_src, "dv": dv_src, "idx": idx_src}
  if with_means:
    # the per-column std (2nd return) is unused: the whitening
    # comes from the covmat, only the means center the targets.
    dv_mean, _ = stream_stats(mm=dv_src, idx=idx_src, method=1)
    c_mean,  _ = param_stats(arr=C_src, idx=idx_src, method=1)
    src["C_mean"]  = c_mean
    src["dv_mean"] = dv_mean

  if verbose:
    in_ram = not isinstance(dv_src, np.memmap)
    print(f"  {os.path.basename(dv_path)}: C {tuple(C_src.shape)} "
          f"dv {tuple(dv_src.shape)} used {idx_src.shape[0]} rows "
          f"| in RAM: {in_ram}")
  return src
