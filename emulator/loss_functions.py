"""Chi2 losses and the robustness annealing schedule."""

import numpy as np
import torch

from .analytics import _analytic_R


def anneal_value(epoch, opts):
  """Value of an annealed robustness knob at a given epoch.

  Holds opts["start"] for the first hold_epochs, then ramps
  toward opts["end"] over the next anneal_epochs, then stays
  at end. shape picks the schedule:
    "const"  -- fixed at start forever (no annealing); the
                fixed-trim baseline. end / hold_epochs /
                anneal_epochs are ignored.
    "linear" -- straight ramp start -> end.
    "cosine" -- smooth ease with zero slope at both ends;
                avoids the abrupt loss jumps a discrete
                schedule causes (those can mislead a reactive
                ReduceLROnPlateau).
    "step"   -- the linear ramp floored to a 0.01 grid, the
                literal 5% -> 4% -> 3% drop.

  Arguments:
    epoch = current epoch (1-based, as in the loop).
    opts  = dict with start, end, hold_epochs,
            anneal_epochs, shape (end / hold_epochs /
            anneal_epochs are unused when shape is "const").

  Returns:
    the knob value at this epoch (float).
  """
  shape = opts["shape"]
  start = opts["start"]
  # constant schedule: hold `start` every epoch (baseline).
  if shape == "const":
    return float(start)
  end  = opts["end"]
  hold = opts["hold_epochs"]
  span = max(1, opts["anneal_epochs"])
  # before the ramp begins: hold the start value.
  if epoch <= hold:
    return float(start)
  # fraction along the ramp, clamped to [0, 1].
  t = min(1.0, (epoch - hold) / span)
  if shape == "cosine":
    # cosine ease: runs 1 -> 0, so value goes start -> end
    # smoothly (no abrupt change at either end).
    ease = 0.5 * (1.0 + np.cos(np.pi * t))
    return float(end + (start - end) * ease)
  # linear value (also the base for the stepped grid).
  val = start + (end - start) * t
  if shape == "step":
    # floor to a 0.01 grid (5% -> 4% -> ...); `end` is the
    # floor it never drops below.
    val = max(end, np.floor(val * 100.0) / 100.0)
  return float(val)


class CosmolikeChi2:
  """
  Adds the chi2 and the training loss to a geometry.

  Composition, not inheritance: a CosmolikeChi2 HOLDS a
  DataVectorGeometry (self.geom) rather than being one, so
  one geometry (built once -- one cosmolike read, one
  eigendecomposition) is shared by several loss variants
  (plain, rescaled, element-weighted) without rebuilding.
  self.geom owns the masked-dv geometry and transforms; this
  class adds only the chi2 (the Mahalanobis distance
  r^T Cinv r, per sample) and the training loss built on it.
  The loss ranks the batch and trims the worst `trim`
  fraction before averaging, so a few contaminated or
  pathological data vectors cannot dominate the gradient.
  Any per-sample transform (e.g. sqrt) is applied after the
  trim, per sample.
  """

  def __init__(self, geom):
    """Hold the geometry the chi2 contracts against.

    Arguments:
      geom = DataVectorGeometry for this probe; owns the
             whitening basis, Cinv / Cinv_sq, dest_idx,
             total_size, center, and every dv transform.
    """
    self.geom = geom

  # --- thin delegation to the geometry ---
  # The pipeline (loaders, run_emulator) reads a few
  # geometry quantities off the chi2 object. Forward them to
  # self.geom so those call sites are unchanged: only how
  # chi2fn is BUILT differs, not how it is used.
  @property
  def dest_idx(self):
    return self.geom.dest_idx

  @property
  def total_size(self):
    return self.geom.total_size

  def encode(self, dv):
    return self.geom.encode(dv)

  def decode(self, whitened_sq):
    return self.geom.decode(whitened_sq)

  def chi2(self, pred, target, full=False):
    """
    Per-sample chi2 = r^T Cinv r, two equal ways.

    `full` selects which path (the chi2 sanity test proves
    they give the same numbers; pass 0/1 or False/True):
      full=False (0, default) -- contract the squeezed
        residual with the masked sub-block Cinv_sq
        (out_dim x out_dim). Fast: no unsqueeze, small
        einsum. Needs geom.Cinv_sq (built in the geometry).
      full=True  (1) -- unsqueeze the residual to the full
        vector and contract with the full Cinv (total_size).
        The slower reference; masked entries contribute 0.
    """
    # geo = self.geom: the geometry this chi2 contracts
    # against (unwhiten / unsqueeze / Cinv live there now).
    geo = self.geom
    if full:
      # reference path: full-length residual + full Cinv.
      r = geo.unsqueeze(geo.unwhiten(pred - target))
      # einsum operands are positional (no keywords): the
      # subscripts "bi,ij,bj->b" name them in order --
      # r (b,i) = residual, geo.Cinv (i,j) = full precision,
      # r (b,j) = the same residual; contracts to chi2 (b,).
      return torch.einsum("bi,ij,bj->b", r, geo.Cinv, r)
    # fast path: squeezed residual + masked sub-block.
    r = geo.unwhiten(pred - target)
    # operands in subscript order: r (b,i) = residual,
    # geo.Cinv_sq (i,j) = masked-block precision, r (b,j).
    return torch.einsum("bi,ij,bj->b", r, geo.Cinv_sq, r)

  def loss(self, pred, target, mode="sqrt", trim=0.05,
           focus=0.0, focus_scale=1.0):
    """
    Scalar training loss from the per-sample chi2.

    Computes the per-sample chi2, optionally trims the worst
    `trim` fraction of the batch (a hard reject -- robust to
    contamination, but it hides genuinely hard regions, so
    evaluation never trims), then averages a per-sample
    transform chosen by `mode`.

    Arguments:
      pred   = network outputs, whitened space (B, out_dim).
      target = whitened targets, same shape.
      mode   = per-sample transform before the mean:
               "chi2"       -> c
               "sqrt"       -> sqrt(c)
               "sqrt_dchi2" -> sqrt(1+2c)-1 (pseudo-Huber)
      trim   = fraction of the worst (largest-chi2) samples
               to drop before averaging; 0 disables trimming.
      focus  = focal weight exponent gamma. <= 0 -> plain
               mean (no weighting); > 0 -> weight each
               sample by (c/(c+focus_scale))**focus
               (detached), up-weighting hard points so the
               optimizer keeps chasing the tail. Annealed.
      focus_scale = chi2 scale where the focal weight turns
                    on; hardness h = c/(c+focus_scale)
                    crosses 0.5 at c = focus_scale.
    Returns:
      a scalar loss tensor (the batch mean of the transform).
    """
    c = self.chi2(pred=pred, target=target)   # per-sample chi2, (B,)
    if trim > 0.0:
      # how many samples to keep after trimming:
      #   c.numel() = element count of c = batch size B
      #     (reads the shape, not the data -> no GPU sync).
      #   (1.0 - trim) = fraction kept (0.95 at trim=0.05);
      #   * numel -> fractional keep-count (e.g. 121.6).
      #   round(...) = Python builtin, nearest integer;
      #   int(...) makes the result a plain int; and
      #   max(1, ...) floors the count at 1, so a tiny
      #   batch can never keep zero samples (topk(c, 0)
      #   or the mean of an empty tensor would break).
      k = max(1, int(round((1.0 - trim) * c.numel())))
      # torch.topk(c, k, largest=False) returns the k
      # smallest chi2 values (largest=False flips it from
      # the default k-largest): the best-fit kept samples,
      # dropping the worst `trim` fraction. It returns a
      # (values, indices) named tuple; `c, _ = ...` unpacks
      # it -- values become c, indices go to `_` (Python's
      # throwaway name). Gradients still flow through the
      # kept values; only which samples are kept is non-
      # differentiable.
      c, _ = torch.topk(c, k, largest=False)
    
    # per-sample transformed loss (not yet averaged)
    if mode == "chi2":
      v = c
    elif mode == "sqrt":
      v = torch.sqrt(c)
    elif mode == "sqrt_dchi2":
      v = torch.sqrt(1.0 + 2.0 * c) - 1.0
    else:
      raise ValueError(f"unknown loss mode: {mode}")

    # focal hardness weight: increasing in chi2 and capped.
    #   h = c/(c+focus_scale) in [0,1) -- a soft "is this
    #   point hard?" (0 for c<<scale, ->1 for c>>scale);
    #   h**gamma sharpens it. detach() freezes the weight as
    #   a priority, so the optimizer cannot lower the loss by
    #   shrinking a point's weight instead of fitting it.
    # gamma = max(focus, 0): a negative focus (the "off"
    # sentinel) clamps to 0, and h**0 = 1 everywhere, so the
    # weighted mean collapses to the plain mean -- no fragile
    # "focus == 0" test, no special-case return.
    gamma = max(focus, 0.0)
    h = (c / (c + focus_scale)).detach()
    w = h ** gamma
    # normalized weighted mean (stable scale as w anneals).
    return (w * v).sum() / (w.sum() + 1e-12)


class RescaledChi2(CosmolikeChi2):
  """
  CosmolikeChi2 with an analytic per-element rescaling R of
  the target: the network emulates the reshaped dv (dv*R,
  flatter across cosmologies), but the chi2 stays on the
  ORIGINAL physical dv -- R is divided back out of the
  residual, so the covariance and the reported chi2 are
  unchanged. R = 1 recovers the base class exactly.

  R is never stored. It is a deterministic function of the
  cosmological params, so it is recomputed on-device from the
  whitened model-input params this class is handed (decoded
  back to physical via the param geometry). Two consumers, one
  source: encode builds the target with R, chi2 undoes R --
  both call _R on the same whitened params, so they use a
  bit-identical R and no (N_rows, n_keep) array ever exists.

  A subclass so the base (no-reshape) path is untouched and the
  two are A/B-swappable. Build it by wrapping a geometry --
  RescaledChi2(geom); then call build_shear_angle_map(geom) and
  configure_rescaling to attach the rescale state before
  training.
  """

  # per-batch stash so the inherited loss reduction (which
  # calls self.chi2(pred=pred, target=target) without params) finds them.
  _params  = None
  # kept-element geometry tensors, built lazily in _R.
  _theta_t = None
  _zeff_t  = None
  # capability flag: this loss's encode/decode/chi2/loss take
  # the whitened params (to build R). The pipeline branches on
  # getattr(chi2fn, "needs_params", False) instead of
  # isinstance, so a future param-aware loss only has to set
  # this True -- it need not subclass RescaledChi2.
  needs_params = True
    
  def configure_rescaling(self, param_geometry, cosmo_mid,
                          names, include_amp=True,
                          u_star=0.5):
    """
    Attach the analytic-rescaling state (call once, after
    wrapping the geom and build_shear_angle_map).

    Arguments:
      param_geometry = ParamGeometry whose decode maps the
                       whitened model inputs back to physical
                       params (what R reads); the same object
                       passed to run_emulator.
      cosmo_mid      = (n_param,) reference cosmology, R = 1
                       there; typically the training-cloud mean
                       train_set["C"][train_set["idx"]].mean(0).
      names          = parameter column names (pgeom.names).
      include_amp    = pass the (Om h^2)^ns/h amplitude factor
                       to _analytic_R (standard run: True).
      u_star         = kernel-peak lens position (~0.5).
    """
    # build_shear_angle_map(geom) must have run first -- _R
    # reads these off geom. Fail loudly if the order is wrong.
    for a in ("theta_kept", "zsrc_i", "zsrc_j"):
      assert hasattr(self.geom, a), (
        "call build_shear_angle_map(geom) before "
        f"configure_rescaling (missing {a})")
        
    self.param_geometry = param_geometry
    self.cosmo_mid   = cosmo_mid
    self.names       = list(names)
    self.include_amp = include_amp
    self.u_star      = u_star
    # drop any stale geometry cache (rebuilt on next _R).
    self._theta_t = None
    self._zeff_t  = None
    return self

  def _R(self, params_whitened):
    """
    Per-(row, element) rescaling R for whitened model inputs.

    Decodes the whitened params back to physical (the form
    _analytic_R reads), then evaluates the analytic R on the
    kept-element geometry (theta_kept and the cross-pair
    z_eff = min(z_i, z_j)) on the params' device. The geometry
    tensors are built once and cached.

    Arguments:
      params_whitened = (B, n_param) whitened model inputs
                        (the same tensor the model consumes).
    Returns:
      R = (B, n_keep) float tensor on the params' device.
    """
    geo  = self.geom
    phys = self.param_geometry.decode(params_whitened)
    dev  = phys.device
    # build the device geometry tensors once (build_shear_
    # angle_map must have run to set theta_kept / zsrc_*).
    if (self._theta_t is None
        or self._theta_t.device != dev):
      zeff = np.minimum(geo.zsrc_i, geo.zsrc_j)
      self._theta_t = torch.as_tensor(
        geo.theta_kept, dtype=torch.float32, device=dev)
      self._zeff_t = torch.as_tensor(
        zeff, dtype=torch.float32, device=dev)
    return _analytic_R(theta_arcmin=self._theta_t,
                       z_eff=self._zeff_t,
                       cosmo=phys,
                       cosmo_mid=self.cosmo_mid,
                       names=self.names,
                       u_star=self.u_star,
                       include_amp=self.include_amp)

  def encode(self, dv, params_whitened):
    # squeeze -> apply the analytic rescaling -> center +
    # whiten. params_whitened gives R per row and element.
    geo = self.geom
    R = self._R(params_whitened)
    return geo.whiten(geo.squeeze(dv) * R - geo.center)

  def decode(self, y, params_whitened):
    # network output -> reshaped dv -> physical dv (/ R).
    geo = self.geom
    R = self._R(params_whitened)
    return (geo.unwhiten(y).float() + geo.center) / R

  def chi2(self, pred, target, params_whitened=None,
               full=False):
    # residual in the reshaped-whitened space, divided by R
    # back to a PHYSICAL squeezed residual, then the usual
    # masked Mahalanobis. The center cancels in pred - target.
    # params_whitened=None -> use the stash set by loss().
    if params_whitened is None:
      params_whitened = self._params
    if params_whitened is None:
      raise RuntimeError(
        "RescaledChi2.chi2 needs the whitened params: pass "
        "them, or call via loss() which stashes them")
    geo = self.geom
    R = self._R(params_whitened)
    r = geo.unwhiten(pred - target) / R
    if full:
      r = geo.unsqueeze(r)
      # operands in subscript order: r (b,i) = residual / R,
      # geo.Cinv (i,j) = full precision, r (b,j).
      return torch.einsum("bi,ij,bj->b", r, geo.Cinv, r)
    # operands in subscript order: r (b,i) = residual / R,
    # geo.Cinv_sq (i,j) = masked-block precision, r (b,j).
    return torch.einsum("bi,ij,bj->b", r, geo.Cinv_sq, r)

  def loss(self, pred, target, params_whitened,
           *args, **kwargs):
    # stash the params for the inherited reduction (trim /
    # transform / focal), which calls self.chi2(pred=pred, target=target)
    # -> picks up self._params. No copy of the base loss body.
    self._params = params_whitened
    return super().loss(pred, target, *args, **kwargs)


class ResidualBaseChi2(RescaledChi2):
  """
  Analytic baseline as a residual base (the "B" form), to test
  the conditioning question against RescaledChi2 (the "A" form)
  with everything else held fixed.

  Both use the SAME analytic R. The difference is WHERE R
  enters the network's reconstruction d_pred (u = unwhiten of
  the net output, c = center):
    A (RescaledChi2):  d_pred = (u + c) / R  -- R divides the
        NET OUTPUT, so the chi2 gradient carries diag(1/R), a
        per-cosmology conditioning factor.
    B (this class):    d_pred =  u + c / R   -- R moves only
        the constant baseline c -> c/R; the net output enters
        at unit gain, so the chi2 is PLAIN (no /R, no
        conditioning factor).
  So B subtracts the analytic-moved baseline in the TARGET and
  leaves the chi2 R-free: it overrides encode (c -> c/R) and
  decode, but does NOT override chi2 -- it inherits the plain
  CosmolikeChi2 chi2. R lives in the target, never in the loss.

  Reuses RescaledChi2's R machinery (_R, configure_rescaling,
  the _params stash, loss). Build and configure exactly like
  RescaledChi2: wrap a geom, build_shear_angle_map(geom), then
  configure_rescaling(...).
  """

  def encode(self, dv, params_whitened):
    # target = whiten(squeeze(dv) - center/R): the plain encode
    # with the constant baseline center swapped for the
    # analytic-moved baseline center/R. R is baked into the
    # target here, so the chi2 below needs no R.
    geo = self.geom
    R = self._R(params_whitened)
    return geo.whiten(geo.squeeze(dv) - geo.center / R)

  def decode(self, y, params_whitened):
    # physical dv = unwhiten(y) + center/R (the baseline added
    # back). The net output enters at unit gain -- no /R.
    geo = self.geom
    R = self._R(params_whitened)
    return geo.unwhiten(y).float() + geo.center / R

  def chi2(self, pred, target, params_whitened=None,
           full=False):
    # PLAIN chi2: R is already in the target, so do NOT divide
    # it out of the residual -- that absent /R is the whole
    # point vs the A form. params_whitened is accepted (the
    # loader and eval pass it, since this is a RescaledChi2
    # subclass) but ignored: the plain chi2 needs no R.
    return CosmolikeChi2.chi2(self, pred=pred, target=target, full=full)


class ElementWeightedChi2(CosmolikeChi2):
  """
  CosmolikeChi2 with a per-ELEMENT focal weight in the TRAINING
  loss (no rescaling -- isolates the per-element weight from the
  analytic R, to test one thing at a time).

  Each dv element's residual is scaled by a detached factor
  >= 1 before the chi2 sums over elements, so the network
  spends accuracy on the elements it currently fits worst in
  error-bar units -- the tight-covariance, most-constraining
  block. Mirrors the per-sample focal but over elements:
    hardness e_i = batch-mean marginal chi2 of element i,
    scale_i      = sqrt(1 + beta * (e/(e+kappa))**gamma).
  Easy elements keep scale 1 (never zeroed -- they sit near
  budget too). The inherited chi2 is unchanged, so EVAL reports
  the true (unweighted) chi2; only the training loss is shaped.
  """

  _elem_kappa  = 0.01
  _elem_gamma  = 1.0
  _elem_beta   = 4.0
  _sigma_cache = None

  def set_elem_weight(self, kappa=0.01, gamma=1.0, beta=4.0):
    """
    Set the per-element focal knobs (call once before training).

    Arguments:
      kappa = marginal-chi2 scale where an element counts as
              hard; e/(e+kappa) crosses 0.5 at e = kappa. e is
              in (residual/sigma)**2 units, so kappa ~ 0.01 is
              an element off by ~0.1 sigma.
      gamma = hardness sharpness (the focal exponent).
      beta  = boost strength; the hardest elements get a chi2
              weight up to 1 + beta.
    """
    self._elem_kappa = kappa
    self._elem_gamma = gamma
    self._elem_beta  = beta
    return self

  def _elem_sigma(self):
    # per-element marginal error bar sqrt(diag(cov)), cached.
    # cov = U diag(ev) U^T -> diag_i = sum_k (U_ik sqrt(ev_k))^2.
    if self._sigma_cache is None:
      self._sigma_cache = torch.sqrt(
        ((self.geom.evecs * self.geom.sqrt_ev) ** 2).sum(1))
    return self._sigma_cache

  def loss(self, pred, target, mode="sqrt", trim=0.05,
           focus=0.0, focus_scale=1.0):
    """
    Training loss with a per-element focal weight on the chi2.

    Same shape as CosmolikeChi2.loss (trim, mode transform,
    per-sample focal), but the per-sample chi2 is built from a
    per-element-weighted residual (hard elements scaled up).
    EVAL calls the inherited self.chi2, so the reported metric
    is the true unweighted chi2.

    Arguments:
      pred, target = whitened outputs / targets (B, out_dim).
      mode   = "chi2" / "sqrt" / "sqrt_dchi2".
      trim   = fraction of worst samples dropped; 0 off.
      focus  = per-sample focal exponent (<=0 -> plain mean).
      focus_scale = per-sample focal turn-on scale.
    Returns:
      a scalar loss tensor.
    """
    # per-ELEMENT focal: scale each element's residual by a
    # detached factor >= 1, set by the element's batch-mean
    # marginal chi2 (self-targets the tight-error-bar block).
    # No rescaling -- residual = unwhiten(pred - target).
    r = self.geom.unwhiten(pred - target)       # (B, n_keep)
    z = r / self._elem_sigma()                  # marginal resid
    e = (z * z).mean(0).detach()                # element hardness
    hard  = e / (e + self._elem_kappa)          # in [0,1)
    scale = torch.sqrt(
      1.0 + self._elem_beta * hard ** self._elem_gamma)
    rs = r * scale
    # operands in subscript order: rs (b,i) = element-weighted
    # residual, self.geom.Cinv_sq (i,j) = masked-block
    # precision, rs (b,j); contracts to per-sample chi2 (b,).
    c = torch.einsum("bi,ij,bj->b", rs, self.geom.Cinv_sq, rs)

    if trim > 0.0:
      k = max(1, int(round((1.0 - trim) * c.numel())))
      c, _ = torch.topk(c, k, largest=False)

    if mode == "chi2":
      v = c
    elif mode == "sqrt":
      v = torch.sqrt(c)
    elif mode == "sqrt_dchi2":
      v = torch.sqrt(1.0 + 2.0 * c) - 1.0
    else:
      raise ValueError(f"unknown loss mode: {mode}")

    gamma = max(focus, 0.0)
    h = (c / (c + focus_scale)).detach()
    w = h ** gamma
    return (w * v).sum() / (w.sum() + 1e-12)


def make_chi2(geom, rescale="none", param_geometry=None,
              cosmo_mid=None, data_dir="lsst_y1",
              dataset="lsst_y1_M1_GGL0.05.dataset",
              include_amp=True):
  """
  Build the chi2fn (loss + geometry wrapper), optionally rescaled.

  The analytic rescaling divides out a fast linear reference R
  (E&H zero-baryon, single-plane Limber) so the network emulates a
  flatter target; the chi2 is ALWAYS reported on the original
  physical dv. The two rescaling variants use the SAME R and differ
  only in WHERE R enters the reconstruction d_pred -- and so whether
  R lands in the loss gradient:

    rescale = "none"     -> plain CosmolikeChi2 (no R).
              "rescaled" -> RescaledChi2 (v1, the "A" form):
                            d_pred = (unwhiten(net) + center) / R,
                            so R divides the NET OUTPUT and the chi2
                            gradient carries a per-cosmology
                            diag(1/R) conditioning factor.
              "residual" -> ResidualBaseChi2 (v2, the "B" form):
                            d_pred = unwhiten(net) + center / R, so
                            R moves ONLY the baseline (center ->
                            center/R); the net enters at unit gain
                            and the chi2 is plain (no /R) -- the
                            clean isolation of the analytic prior.

  Both rescaling variants need the per-element angle/tomography map
  on the geometry (build_shear_angle_map, imported lazily so a plain
  build does not pull in the cosmolike-importing geometry module)
  and the analytic config (configure_rescaling).

  Arguments:
    geom           = DataVectorGeometry for the probe (e.g. xi).
    rescale        = "none" / "rescaled" / "residual" (see above).
    param_geometry = ParamGeometry whose decode maps the whitened
                     model inputs back to physical params (what R
                     reads); required when rescale != "none".
    cosmo_mid      = (n_param,) reference cosmology where R = 1,
                     typically the training-cloud mean; required
                     when rescale != "none".
    data_dir       = cosmolike data folder for the angle map.
    dataset        = .dataset ini for the angle map.
    include_amp    = pass the (Om h^2)^ns/h amplitude factor to the
                     analytic R (standard run: True).

  Returns:
    a CosmolikeChi2 (or RescaledChi2 / ResidualBaseChi2) to pass to
    run_emulator as chi2fn.
  """
  if rescale == "none":
    return CosmolikeChi2(geom=geom)
  # lazy import: build_shear_angle_map lives in the cosmolike-
  # importing geometry module, only needed for the rescaled path.
  from .geometries_output import build_shear_angle_map
  build_shear_angle_map(geom=geom, data_dir=data_dir,
                        dataset=dataset)
  cls = RescaledChi2 if rescale == "rescaled" else ResidualBaseChi2
  chi2fn = cls(geom=geom)
  chi2fn.configure_rescaling(param_geometry=param_geometry,
                             cosmo_mid=cosmo_mid,
                             names=list(param_geometry.names),
                             include_amp=include_amp)
  return chi2fn
