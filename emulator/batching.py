"""Memory sizing and the regime-aware data loaders.

This module decides where each source's data lives and hands the training
loop two closures (rows -> whitened param inputs, rows -> encoded targets)
that hide it. The helpers compute_batch_size_bytes,
compute_model_size_bytes, and batches_per_load estimate the per-batch and
resident memory. _build_loaders_one picks one of three regimes against a
VRAM budget (pre-encode the whole target set on the GPU, stream it from
RAM, or stream it from a disk memmap) and reports the bytes it made
resident. build_loaders runs it once per source (train, then val against
the reduced budget) and returns the data dict the loop consumes.

PS: whitened = rotated into the covariance eigenbasis and scaled to unit
variance, so the components are decorrelated (the form the network sees).
encoded = a data vector put through the geometry's encode (keep the
unmasked entries, subtract the training mean, whiten), the form trained
against. resident = held in GPU memory for the whole run, not re-loaded
each batch.
"""

import numpy as np
import torch


def compute_batch_size_bytes(model, bs, sample_dims, dv_len=3000):
  # sample_dims = shape of one model input (no batch axis).
  # In the emulator the input is the cosmo param vector
  # so sample_dims = (Ncosmo,). The dv is the model output.
  # dv_len = full data-vector length the chi2 un-squeezes
  # to (a conservative ~3000, no cosmolike query needed).

  # model.parameters() is an iterator over the weight tensors
  # next() grabs the first one. Its .device is where that
  # tensor lives; we put x there too so they match.
  dev = next(model.parameters()).device

  # dummy input batch of shape (bs, *sample_dims). Only
  # the shapes matter for memory, not the values -> zeros.
  x = torch.zeros(bs, *sample_dims, device=dev)

  total = 0  # running byte count of saved tensors
  def pack(t):
    # pack: the first callback. autograd calls it the
    # moment a tensor is saved during forward, and stores
    # whatever pack returns. We record the size and return
    # t unchanged. (+= alone would make total a new local;
    # nonlocal points it at the outer total.)
    nonlocal total
    total += t.numel() * t.element_size()
    return t
  def unpack(stored):
    # unpack: the 2nd callback. autograd calls it in
    # backward to rebuild the tensor pack stored. We never
    # call backward, but saved_tensors_hooks requires it,
    # so we hand the stored tensor straight back.
    return stored

  # saved_tensors_hooks defines a custom way to store
  # activations between forward and backward (to save
  # memory): pack transforms each tensor on the way in
  # (e.g. compress), unpack reverses it on the way out.
  # The pair must round-trip: unpack(pack(t)) == t. Here
  # the no-op pair just spies on the sizes.
  hooks = torch.autograd.graph.saved_tensors_hooks  # alias
  with hooks(pack, unpack):
    # saving happens during forward, so the instant
    # model(x) returns, total is complete -- no backward.
    out = model(x)

  # device buffers tied to this batch: the input x, the
  # model output, and the target the loss compares it to.
  # The target has the same shape and dtype as the output,
  # so it adds another out_bytes. element_size() = that
  # tensor's bytes per element (float32 -> 4, float64 -> 8).
  in_bytes  = x.numel() * x.element_size()
  out_bytes = out.numel() * out.element_size()
  io = in_bytes + 2 * out_bytes

  # the chi2 runs OUTSIDE model(x), so the hook above never
  # sees it. Per batch it builds a few full-length float64
  # buffers: the unsqueezed residual, the r @ Cinv product,
  # and the copy autograd saves for backward. Budget three
  # (bs, dv_len) doubles.
  chi2 = 3 * bs * dv_len * 8

  return total + io + chi2


def compute_model_size_bytes(model):
  # Memory resident for the whole run: the weights, 
  # their grads, and the optimizer's per-param state
  #
  # opt_state = number of state tensors the optimizer 
  # keeps per param (each the same size as the params):
  #   SGD (plain)                 0
  #   SGD+momentum, Adagrad,      1
  #     RMSprop (default)
  #   Adam, AdamW, Adamax, NAdam  2
  #   Adam(amsgrad), RMSprop      3   <- worst typical
  #     (centered + momentum)
  opt_state = 3
  p = sum(t.numel() for t in model.parameters())
  esize = next(model.parameters()).element_size()  # bytes
  # weights(1) + grads(1) + opt_state buffers
  return p * esize * (2 + opt_state)

def batches_per_load(model, 
                     bs, 
                     sample_shape, 
                     budget,
                     dv_len=3000):
  # rows per streamed chunk whose per-batch activation cost
  # fits within `budget`. resident = model (weights + grads
  # + optimizer state) + the chi2 precision matrix Cinv; the
  # chunk gets what is left. budget is now an explicit arg
  # (real free VRAM in research, emulated GPU_MEM in class).
  cinv     = dv_len * dv_len * 8
  resident = compute_model_size_bytes(model) + cinv
  free = 0.8 * budget - resident
  per  = compute_batch_size_bytes(model=model,
                                  bs=bs,
                                  sample_dims=sample_shape,
                                  dv_len=dv_len)
  return max(1, int(free // per))


def _build_loaders_one(device, C, dv, idx,
                       param_geometry, chi2fn,
                       model, bs, budget,
                       dv_len=3000, CHUNK=1000):
  """
  Build the two data loaders for ONE source -- a train file or
  a val file -- and decide where that source's data lives.
  Returns two closures, load_C (rows -> whitened param inputs)
  and load_dv (rows -> encoded targets), plus the per-chunk row
  count and the GPU bytes this source made resident. Both
  closures take GLOBAL row indices (positions in the full C/dv
  dump); a `slots` helper maps those to local positions in the
  compact resident subset, so the rest of the pipeline is
  identical no matter where the data ended up.
  The PARAMETERS are always encoded once and kept on the GPU --
  they are tiny (n_used x Ncosmo) -- so only the DATA VECTORS
  (the large array) change placement, picked by a memory ladder
  against `budget`:
    Regime 1 (resident gather): the whole encoded target set
      fits, so pre-encode every target once and hold it on the
      GPU; a batch is then pure on-device indexing -- no
      host->device copy and no re-encoding per step.
    Regime 2 (RAM stream): the encoded set does not fit the GPU
      but the dvs are an in-RAM ndarray; stream them RAM->GPU a
      chunk at a time and encode on the fly (pinned memory on
      CUDA for a faster copy).
    Regime 3 (disk stream): the dvs exceed RAM (a np.memmap),
      so the same per-chunk path reads them from disk.
  Resident memory = the model (weights + grads + optimizer
  state) + the chi2's Cinv + the encoded params; the dvs get
  whatever is left (0.8 * budget - resident), and `fits` decides
  regime 1 versus streaming. The returned `used` (bytes this
  source made resident) lets the orchestrator subtract it from
  the budget before sizing the NEXT source, so two sequential
  builds against one GPU do not overrun it.
  Works for the plain CosmolikeChi2 (encode takes the dv alone)
  and for the param-aware losses (RescaledChi2 / ResidualBase /
  PCEResidualChi2 / PCERatioChi2), whose encode also takes this
  block's whitened params -- the resident C_used rows -- to
  build R or the PCE base; the `rescaled` flag branches encode.
  A loss may also stage a target WIDER than the model output by
  declaring a target_dim attribute (PCERatioChi2 packs the PCE
  base together with the truth, so the chi2 never recomputes the
  base in the training loop); it defaults to out_dim.

  Arguments:
    device     = target device for the staged tensors.
    C          = this source's full param dump, (N, Ncosmo).
    dv         = this source's full dv dump, (N, Ndv); ndarray
                 -> regime 2, np.memmap -> regime 3.
    idx        = global row indices into C/dv to make loadable.
    param_geometry = ParamGeometry; .encode whitens raw params.
    chi2fn     = CosmolikeChi2 or RescaledChi2 (output geom).
    model      = network; read only to size resident memory.
    bs         = minibatch size; the chunk is a multiple of it.
    budget     = VRAM bytes to plan against.
    dv_len     = full dv length the chi2 unsqueezes to.
    CHUNK      = rows per block when pre-encoding (regime 1).
  Returns:
    load_C  = callable: global rows -> whitened inputs.
    load_dv = callable: global rows -> whitened targets.
    load    = rows per chunk chosen for this regime.
    used    = GPU bytes this source made resident.
  """
  ncosmo    = C.shape[1]
  # out_dim = the model's output width = the number of unmasked
  # data-vector entries the network predicts (dest_idx holds the
  # kept positions; .numel() counts them).
  out_dim   = chi2fn.dest_idx.numel()

  # tgt_dim = the width of the target tensor this loader stages
  # per row. Normally that is just the encoded truth, one value
  # per kept entry, so tgt_dim == out_dim. One loss needs more
  # room: PCERatioChi2 forms its physical prediction as
  #   pred = base * (1 + net_output)
  # where net_output is the model's output (a fractional
  # correction) and base is a fixed reference data vector (the
  # frozen PCE). Rather than recompute base every batch, it
  # precomputes it once here and stages [base ; truth] as a
  # single 2*n_keep-wide target, then unpacks both inside the
  # chi2.
  #
  # A loss requests that wider target by defining a `target_dim`
  # attribute. getattr(obj, "name", default) returns obj.name if
  # it exists and `default` otherwise (it never raises), so a
  # loss without target_dim falls back to out_dim and stages the
  # plain truth. This is the same opt-in pattern as the
  # needs_params flag a few lines below.
  tgt_dim   = getattr(chi2fn, "target_dim", out_dim)

  # used_rows = the distinct rows this source loads. np.unique
  # both removes duplicates (idx may name a row more than once)
  # and returns them sorted, so each row is staged once and in
  # file order: the order slots() assumes, and the one that makes
  # a memmap read sequential rather than random. n_used counts them.
  used_rows = np.unique(idx)
  n_used    = len(used_rows)

  assert dv.shape[1] == chi2fn.total_size, (
    f"dv width {dv.shape[1]} != "
    f"total_size {chi2fn.total_size}")

  # encode the params once, resident on the GPU. For the
  # rescaled geometry these whitened params are also what
  # encode needs to build R.
  C_used = param_geometry.encode(
    torch.from_numpy(C[used_rows]).float().to(device))

  rescaled = getattr(chi2fn, "needs_params", False)

  model_bytes = compute_model_size_bytes(model)
  cinv        = dv_len * dv_len * 8
  enc_params  = n_used * ncosmo * 4
  resident    = (model_bytes + cinv + enc_params)

  def slots(rows):
    # map GLOBAL row numbers to their LOCAL positions in the
    # compact resident subset. used_rows is sorted and every
    # query row is one of its entries, so np.searchsorted (which
    # returns the insertion index into a sorted array) gives, for
    # each global row, exactly where it sits in used_rows -- i.e.
    # its row index in C_used / dv_used.
    local_pos = np.searchsorted(used_rows, rows)
    return torch.from_numpy(local_pos).to(device)

  def load_C(rows):
    return C_used[slots(rows)]

  enc_dvs = n_used * tgt_dim * 4
  fits    = enc_dvs + resident < 0.8 * budget

  if fits:
    # Regime 1: pre-encode every target, hold it on the GPU.
    dv_used = torch.empty(n_used, tgt_dim, device=device)
    for start in range(0, n_used, CHUNK):
      block = used_rows[start:start + CHUNK]
      # raw dvs for this block, on the device.
      dv_t = torch.from_numpy(dv[block]).float().to(device)
      if rescaled:
        # rescaled target: encode also needs this block's
        # params. C_used holds the whitened params in
        # used_rows order, so this block is the local slice
        # start : start + len(block).
        params = C_used[start:start + len(block)]
        enc = chi2fn.encode(dv=dv_t, params_whitened=params)
      else:
        enc = chi2fn.encode(dv_t)
      dv_used[start:start + len(block)] = enc

    def load_dv(rows):
      return dv_used[slots(rows)]

    bytes_per_row = (tgt_dim + ncosmo) * 4
    vram_left     = 0.8 * budget - resident - enc_dvs
    fit_rows = max(bs, int(vram_left // bytes_per_row))
    load = min(len(idx), fit_rows)

  elif not isinstance(dv, np.memmap):
    # Regime 2: dvs live in CPU RAM.
    def load_dv(rows):
      cpu = torch.from_numpy(dv[rows]).float()
      if device.type == "cuda":
        cpu = cpu.pin_memory()
      gpu = cpu.to(device)
      if rescaled:
        return chi2fn.encode(dv=gpu, params_whitened=load_C(rows))
      return chi2fn.encode(gpu)

    load = bs * batches_per_load(model=model,
                                 bs=bs,
                                 sample_shape=C.shape[1:],
                                 budget=budget,
                                 dv_len=dv_len)
  else:
    # Regime 3: dvs exceed RAM, read from the memmap.
    def load_dv(rows):
      host = torch.from_numpy(dv[rows]).float()
      gpu  = host.to(device)
      if rescaled:
        return chi2fn.encode(dv=gpu, params_whitened=load_C(rows))
      return chi2fn.encode(gpu)

    load = bs * batches_per_load(model=model,
                                 bs=bs,
                                 sample_shape=C.shape[1:],
                                 budget=budget,
                                 dv_len=dv_len)

  used = enc_params + (enc_dvs if fits else 0)
  return load_C, load_dv, load, used


def build_loaders(device, train_set, val_set, param_geometry, 
                  chi2fn, model, bs, budget,
                  dv_len=3000, CHUNK=1000):
  """
  Build the train and val loaders, return the data dict the
  training loop and eval_val consume. Train and val live in
  SEPARATE files (T and T/2); each is passed as a source
  dict and gets its own loaders via _build_loaders_one (no
  shared rows, no leakage). The same (training-built)
  param_geometry / chi2fn whiten both sources.

  Arguments:
    device     = target device.
    train_set  = training source dict:
                   "C"   full param dump (T file),
                   "dv"  full dv dump (T file),
                   "idx" global rows to train on (into C/dv).
    val_set    = validation source dict, same three keys.
    param_geometry, chi2fn, model, bs, budget, dv_len, CHUNK
               = forwarded to _build_loaders_one (see there);
                 the same geometry for both sources.
  Returns:
    data = nested dict, one sub-dict per source, both with
      the same keys:
        data["train"] = {load_C, load_dv, idx, load}
        data["val"]   = {load_C, load_dv, idx, load}
  """
  (load_C, load_dv, load,
   used_tr) = _build_loaders_one(device=device, 
                              C=train_set["C"], 
                              dv=train_set["dv"], 
                              idx=train_set["idx"],
                              param_geometry=param_geometry, 
                              chi2fn=chi2fn, 
                              model=model, 
                              bs=bs, 
                              budget=budget, 
                              dv_len=dv_len, 
                              CHUNK=CHUNK)

  # the train set is now resident on the GPU, so the val
  # call plans against a budget reduced by what train took.
  # (model + Cinv are shared and counted by each call.)
  (load_C_val, load_dv_val, load_val, 
   _) = _build_loaders_one(device=device, 
                           C=val_set["C"], 
                           dv=val_set["dv"], 
                           idx=val_set["idx"],
                           param_geometry=param_geometry, 
                           chi2fn=chi2fn,
                           model=model, 
                           bs=bs, 
                           budget=budget - used_tr,
                           dv_len=dv_len, 
                           CHUNK=CHUNK)

  return {
    "train": {
      "load_C": load_C,
      "load_dv": load_dv,
      "idx": train_set["idx"],
      "load": load,
    },
    "val": {
      "load_C": load_C_val,
      "load_dv": load_dv_val,
      "idx": val_set["idx"],
      "load": load_val,
    }
  }
