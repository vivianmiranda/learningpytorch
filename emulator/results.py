"""Plain-text I/O for sweep / learning-curve results.

save_learning_curves writes a whitespace-delimited table (one row per
N_train, one column per curve, with a "#"-comment header carrying the
config) that np.loadtxt reads back. It is the format the N_train sweep
and the activation bake-off drivers save, so several runs can be overlaid
later.
"""


def save_learning_curves(path, sizes, curves, meta=None):
  """
  Write learning curve(s) as a whitespace-delimited text table.

  One row per N_train, one column per curve, loadable with np.loadtxt
  (every header line is a "#" comment, which loadtxt skips). A single
  config writes a one-entry `curves`; a bake-off writes all its curves to
  one file. Layout:

    # learning curve: f(delta-chi2 > threshold) vs N_train
    # model=ResMLP  rescale=none  threshold=0.2  pool=82000
    # columns: N_train, H, power, multigate, gated_power
    2000     0.401234  0.410512  0.395001  0.402310
    4203     ...

  Arguments:
    path   = output text-file path.
    sizes  = the N_train values, one per row (any sequence; cast to int).
    curves = mapping label -> the per-size fractions, a list aligned with
             `sizes` (so curves[label][i] is the value at sizes[i]). One
             curve -> a one-entry dict. The labels become the data
             columns (in dict order), documented on the "# columns:" line.
    meta   = optional mapping written as a "# key=val  key=val" line
             (model / rescale / threshold / pool / ...); None to omit it.
  """
  sizes  = list(sizes)
  labels = list(curves)
  lines  = ["# learning curve: f(delta-chi2 > threshold) vs N_train"]
  if meta:
    # one "# key=val  key=val ..." line (insertion order preserved).
    lines.append("# " + "  ".join(f"{k}={v}" for k, v in meta.items()))
  # the column header is a comment too (so np.loadtxt skips it); labels
  # are comma-separated so a label with spaces stays unambiguous.
  lines.append("# columns: " + ", ".join(["N_train"] + [str(l)
                                                        for l in labels]))
  for i, n in enumerate(sizes):
    row = [f"{int(n):d}"] + [f"{curves[l][i]:.6f}" for l in labels]
    lines.append("  ".join(row))
  with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")
