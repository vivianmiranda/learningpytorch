"""One configured single-network cosmic-shear emulator run.

This class factors the driver setup boilerplate (parse the config, pick
the device, stage the train / val sources, build the parameter and
data-vector geometries and the chi2, assemble the run_emulator spec dicts,
and train) into one reusable object, so a driver or a sweep script (over
N_train, or one hyperparameter at a time) does not copy it.

Build it from a YAML file (from_yaml) or an already-parsed config mapping
(from_config, e.g. load the YAML once and rebuild from a tweaked copy per
sweep point). Both resolve the model class from train_args.model.name
through the MODELS registry. The expensive pieces are built by explicit
methods and cached on the instance, so:

  - a SINGLE run (the driver): exp = from_yaml(...); exp.run().
  - an N_TRAIN sweep (geometry depends on the training subset):
      for N in sizes:
        exp.stage_train(n_train=N); exp.stage_val()
        exp.build_geometry(); exp.train()
        f = exp.frac_above(0.2)
  - a HYPERPARAMETER sweep (data + geometry fixed, only the spec varies):
      exp.stage_train(); exp.stage_val(); exp.build_geometry()
      for v in values:
        exp.train(train_args=tweaked_copy_with(v))
"""

import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from .data_staging import read_param_names, load_source, phys_cut_idx
from .geometries_parameter import ParamGeometry
from .loss_functions import make_chi2
from .emulator_designs import ResMLP, ResCNN
from .activations import make_activation
from .training import (
  run_emulator, build_run_specs, pick_device, make_logger,
  default_train_args, eval_source_chi2)


# model name (train_args.model.name) -> class. ResCNN additionally needs
# the data geometry injected (it builds fixed full<->theta basis-change
# buffers from it); ResMLP takes no geom. Shared by the drivers.
MODELS = {"resmlp": ResMLP, "rescnn": ResCNN}

# default reported delta-chi2 cutoffs; the first (0.2) is the emulator
# goal and the best-model-selection metric.
DEFAULT_THRESHOLDS = torch.tensor([0.2, 0.5, 1.0, 10.0, 100.0])


class EmulatorExperiment:
  """
  Configuration + environment for one single-network cosmic-shear (xi)
  emulator, reusable across a single run and across sweeps.

  The constructor stores the resolved config and the fixed choices and
  builds only the cheap, config-derived state (device, parameter names,
  the quiet-gated logger). The staged data and the geometry are built by
  explicit methods (stage_train / stage_val / build_geometry) and cached
  on the instance, so a sweep rebuilds only what actually varies. The
  fixed single-emulator choices (probe = xi, AdamW, ReduceLROnPlateau,
  use_amp = False, the report thresholds) are constructor defaults, so a
  driver only passes what it varies (the YAML, rescale, activation,
  quiet); the MODEL is the config's choice (train_args.model.name).
  """

  def __init__(self,
               data,
               train_args,
               model_cls,
               opt_cls=optim.AdamW,
               sched_cls=lr_scheduler.ReduceLROnPlateau,
               probe="xi",
               thresholds=None,
               use_amp=False,
               rescale="none",
               activation="H",
               device=None,
               quiet=False,
               raw_train_args=None):
    """
    Store the config + fixed choices; build the cheap derived state.

    Arguments:
      data       = the config "data" block: the input file paths plus the
                   cut / split / cosmolike-dataset settings. Keys:
                     train_dv = training data-vector .npy (memmapped);
                     train_params = training parameter .txt (columns:
                       weight, lnp, the modeled params, chi2);
                     train_covmat = parameter covmat (header line = the
                       parameter names);
                     val_dv = validation data-vector .npy;
                     val_params = validation parameter .txt;
                     cosmolike_data_dir = data folder under
                       external_modules/data;
                     cosmolike_dataset = .dataset ini naming the cov /
                       mask / data-vector files;
                     omegabh2_cut = drop rows with omega_b h^2 >= this;
                     train_divisor / val_divisor = keep N // divisor of
                       the train / val rows;
                     split_seed = seed for the cut+shuffle that picks the
                       train / val rows;
                     ram_frac = optional (default 0.7): RAM fraction the
                       staged subset may fill.
      train_args = the resolved "train_args" block: range-free (any
                   [default, min, max, kind] search range already
                   collapsed to a scalar, e.g. by default_train_args).
                   Top-level keys:
                     nepochs = passes over the training set;
                     bs = minibatch size;
                     loss_mode = optional (default "sqrt"): the per-sample
                       transform "sqrt" / "chi2" / "sqrt_dchi2";
                     silent = optional (default False): per-run silence.
                   Plus six constructible sub-blocks (each a mapping):
                     model = the model's kwargs -- "name" (resmlp /
                       rescnn; from_config reads it to pick the class,
                       then build_specs strips it) plus int_dim_res,
                       n_blocks, and for rescnn kernel_size / channels /
                       n_blocks_cnn / gate_init;
                     optimizer = weight_decay (+ any extra AdamW kwargs);
                     lr = lr_base, bs_base, warmup_epochs (the run sets
                       lr = lr_base * sqrt(bs / bs_base));
                     scheduler = mode, patience, factor (the
                       ReduceLROnPlateau kwargs);
                     trim = the trim schedule -- start, end, hold_epochs,
                       anneal_epochs, shape (see anneal_value);
                     focus = the focal-weight schedule -- start, end,
                       hold_epochs, anneal_epochs, shape, kappa.
      model_cls  = the model class (ResMLP / ResCNN); from_config
                   resolves it from train_args.model.name.
      opt_cls    = optimizer class (default AdamW).
      sched_cls  = scheduler class (default ReduceLROnPlateau).
      probe      = cosmolike probe (default "xi").
      thresholds = reported delta-chi2 cutoffs (default
                   DEFAULT_THRESHOLDS); thresholds[0] selects the best
                   model.
      use_amp    = run the forward in low-precision autocast (default
                   False).
      rescale    = analytic-R mode forwarded to make_chi2 ("none" /
                   "rescaled" / "residual").
      activation = ResBlock activation name (make_activation).
      device     = compute device (default: pick_device()).
      quiet      = if True, the instance logger and the per-source /
                   per-epoch prints are silenced.
      raw_train_args = the UN-collapsed train_args (search ranges
                   intact), for a search driver that resolves them per
                   trial; defaults to train_args (from_config supplies
                   the raw block).
    """
    self.data       = data
    self.train_args = train_args
    self.model_cls  = model_cls
    self.opt_cls    = opt_cls
    self.sched_cls  = sched_cls
    self.probe      = probe
    self.thresholds = (DEFAULT_THRESHOLDS if thresholds is None
                       else thresholds)
    self.use_amp    = use_amp
    self.rescale    = rescale
    self.activation = activation
    self.quiet      = quiet

    # make_logger / pick_device (training.py): a print(*a) gated on quiet
    # (a no-op when quiet), and the compute device (cuda > mps > cpu).
    self.log        = make_logger(quiet=quiet)
    self.device     = pick_device() if device is None else device

    # the un-collapsed train_args (search ranges intact), for a search
    # driver that resolves them per trial (suggest_train_args); defaults
    # to the resolved train_args when no raw block is supplied.
    self.raw_train_args = (train_args if raw_train_args is None
                           else raw_train_args)

    # TF32 tensor-core float32 matmuls (Ampere+); no-op on CPU / MPS.
    # One-time global switch.
    torch.set_float32_matmul_precision("high")

    # read_param_names (data_staging.py): the parameter column names from
    # the covmat's "#"-prefixed header line. Reused for the val cut (same
    # columns); config-derived and cheap.
    self.names = read_param_names(data["train_covmat"])

    # artifacts the methods below build; cached for reuse across a sweep
    # (None until built).
    self.train_set = None
    self.val_set   = None
    self.pgeom     = None
    self.geom      = None
    self.chi2fn    = None
    self.model     = None

  # --- alternative constructors ---
  @classmethod
  def from_config(cls, cfg, models=None, **kwargs):
    """
    Build from an already-parsed config mapping.

    Validates the required blocks, collapses any [default, min, max,
    kind] search ranges in train_args to their defaults, and resolves
    train_args.model.name -> a model class through `models`. Use this to
    rebuild from a tweaked copy of a config dict (one sweep point).

    Arguments:
      cfg    = mapping with a "data" block and a "train_args" block (the
               schema the drivers load from YAML; see __init__ for the
               exact keys each block holds).
      models = name -> class registry (default MODELS:
               resmlp -> ResMLP, rescnn -> ResCNN).
      **kwargs = forwarded to __init__ (opt_cls, sched_cls, probe,
               thresholds, use_amp, rescale, activation, device, quiet).

    Returns:
      an EmulatorExperiment with the resolved data / train_args / model.
    """
    models = MODELS if models is None else models
    for block in ("data", "train_args"):
      if block not in cfg:
        raise KeyError(
          f"config is missing the required block: {block!r}")
    # default_train_args (training.py): walk train_args and collapse every
    # [default, min, max, kind] search range to its default (first) value,
    # so a tuning YAML also builds a concrete run.
    ta = default_train_args(cfg["train_args"])
    # the model is the config's choice; read (not pop) name -- build_specs
    # strips it from the spread, so it never reaches the constructor.
    name = str(ta["model"].get("name", "resmlp")).lower()
    if name not in models:
      raise ValueError(
        f"unknown train_args.model.name {name!r}; "
        f"choose one of {sorted(models)}")
    return cls(data=cfg["data"], train_args=ta,
               model_cls=models[name],
               raw_train_args=cfg["train_args"], **kwargs)

  @classmethod
  def from_yaml(cls, path, models=None, **kwargs):
    """
    Build from a YAML config file.

    Thin wrapper: read the file, then from_config. See from_config for
    the resolution and **kwargs.

    Arguments:
      path   = path to the YAML config (data + train_args blocks).
      models = name -> class registry (default MODELS).
      **kwargs = forwarded to from_config -> __init__.

    Returns:
      an EmulatorExperiment.
    """
    with open(path) as f:
      cfg = yaml.safe_load(f)
    return cls.from_config(cfg, models=models, **kwargs)

  # --- staging + geometry (the expensive, cached pieces) ---
  def stage_train(self, n_train=None):
    """
    Stage the training source (cached as self.train_set).

    Uses a generator FRESHLY seeded from data["split_seed"], so the
    cut+shuffle pool is deterministic and slicing it to different sizes
    gives NESTED subsets -- the right thing for a learning-curve sweep.

    Arguments:
      n_train = absolute number of training rows to keep; None (default)
                uses the YAML data["train_divisor"] (N // divisor).

    Returns:
      the training source dict (also stored on the instance).
    """
    d   = self.data
    gen = torch.Generator().manual_seed(int(d["split_seed"]))
    # load_source (data_staging.py): memmap the dv .npy, load + physically
    # cut the params (omega_b h^2 < cut), keep n_keep (or N // divisor)
    # rows of the seeded shuffle, stage them in RAM if they fit (else keep
    # the memmap), and return a source dict {C, dv, idx} (+ C_mean /
    # dv_mean when with_means).
    self.train_set = load_source(
      dv_path=d["train_dv"],
      params_path=d["train_params"],
      names=self.names,
      cut=d["omegabh2_cut"],
      divisor=(None if n_train is not None else d["train_divisor"]),
      n_keep=n_train,
      gen=gen,
      ram_frac=d.get("ram_frac", 0.7),
      with_means=True,
      verbose=not self.quiet)
    return self.train_set

  def stage_val(self, n_val=None):
    """
    Stage the validation source (cached as self.val_set).

    Uses a generator freshly seeded from data["split_seed"] (the val
    file differs from train, so the same seed gives an independent
    selection). The val source carries no means -- the geometry centers
    come from the TRAINING source only.

    Arguments:
      n_val = absolute number of validation rows to keep; None (default)
              uses the YAML data["val_divisor"].

    Returns:
      the validation source dict (also stored on the instance).
    """
    d   = self.data
    gen = torch.Generator().manual_seed(int(d["split_seed"]))
    # load_source (data_staging.py): same staging as stage_train, on the
    # val files; with_means=False (val borrows the TRAINING centers).
    self.val_set = load_source(
      dv_path=d["val_dv"],
      params_path=d["val_params"],
      names=self.names,
      cut=d["omegabh2_cut"],
      divisor=(None if n_val is not None else d["val_divisor"]),
      n_keep=n_val,
      gen=gen,
      ram_frac=d.get("ram_frac", 0.7),
      with_means=False,
      verbose=not self.quiet)
    return self.val_set

  def pool_size(self):
    """
    Number of physically-cut TRAINING rows available -- the natural top
    of an N_train sweep.

    Loads the training parameter file, keeps the modeled columns, and
    applies the omega_b h^2 cut (the same cut stage_train uses), then
    counts the survivors. The count is order-independent, so no shuffle
    or staging is done. Uses load_source's default modeled columns
    (slice(2, -1)).

    Returns:
      the number of training rows with omega_b h^2 <
      data["omegabh2_cut"] (an int).
    """
    d = self.data
    # modeled parameter columns (drop the leading weight / lnp and the
    # trailing chi2), as load_source does by default.
    C   = np.loadtxt(d["train_params"], dtype="float32")[:, slice(2, -1)]
    idx = np.arange(C.shape[0])
    # phys_cut_idx (data_staging.py): keep the rows with omega_b h^2 =
    # Omega_b (H0/100)^2 < cut (drops the sparse high-baryon corner).
    phys = phys_cut_idx(C=C, idx=idx, names=self.names,
                        cut=d["omegabh2_cut"])
    return int(len(phys))

  def build_geometry(self, train_set=None):
    """
    Build the input + output geometries and the chi2 (cached as
    self.pgeom / self.geom / self.chi2fn).

    The whitening centers come from the training means, so this depends
    on the training subset: rebuild it per subset in an N_train sweep,
    build it ONCE for a hyperparameter sweep (it does not depend on the
    model or the train_args).

    Arguments:
      train_set = training source dict with "C_mean" / "dv_mean" / "C" /
                  "idx" (default: self.train_set, from stage_train).

    Returns:
      (pgeom, geom, chi2fn), also stored on the instance.
    """
    train_set = self.train_set if train_set is None else train_set
    d = self.data
    # lazy import: DataVectorGeometry.from_cosmolike pulls in cosmolike,
    # which lives only on the workstation -- importing it here keeps this
    # module importable (for the config logic) without cosmolike.
    from .geometries_output import DataVectorGeometry

    # ParamGeometry.from_covmat (geometries_parameter.py): the INPUT
    # whitening -- eigendecompose the parameter covmat so encode() centers
    # + rotates + unit-scales the cosmological params the model sees.
    self.pgeom = ParamGeometry.from_covmat(
      device=self.device,
      center=train_set["C_mean"],
      covmat_path=d["train_covmat"])

    # DataVectorGeometry.from_cosmolike (geometries_output.py): the OUTPUT
    # geometry -- read cosmolike's cov / mask / inverse-cov, eigendecompose
    # the kept (unmasked) block, so encode()/chi2 whiten + score the masked
    # data vector.
    self.geom = DataVectorGeometry.from_cosmolike(
      device=self.device,
      dv_center=train_set["dv_mean"],
      data_dir=d["cosmolike_data_dir"],
      dataset=d["cosmolike_dataset"],
      probe=self.probe)

    # make_chi2 (loss_functions.py): wrap geom in the loss -- plain
    # CosmolikeChi2, or the analytic-R RescaledChi2 / ResidualBaseChi2 when
    # rescale != "none". cosmo_mid = the training-cloud mean (R = 1 there
    # for a rescaled chi2; ignored by the plain chi2).
    self.chi2fn = make_chi2(
      geom=self.geom,
      rescale=self.rescale,
      param_geometry=self.pgeom,
      cosmo_mid=train_set["C"][train_set["idx"]].mean(0),
      data_dir=d["cosmolike_data_dir"],
      dataset=d["cosmolike_dataset"])

    return self.pgeom, self.geom, self.chi2fn

  # --- per-run pieces ---
  def build_specs(self, train_args=None):
    """
    Assemble the six run_emulator spec dicts for one run.

    build_run_specs from train_args, then inject the named activation
    into the ResBlock options and -- for ResCNN only -- the data
    geometry (it needs geom to build its basis-change buffers; ResMLP
    takes no geom). A hyperparameter sweep passes a varied train_args.

    Arguments:
      train_args = resolved train_args mapping (default:
                   self.train_args). Range-free (from default_train_args
                   / suggest_train_args). A leftover model.name is
                   stripped here, so a suggest_train_args result works
                   too.

    Returns:
      the keyed spec dict run_emulator consumes as **specs.
    """
    train_args = self.train_args if train_args is None else train_args
    # drop a leftover model.name before the spread (from_config reads it
    # without popping, and a suggest_train_args result still carries the
    # scalar name) -- else it would reach the model constructor.
    ta = dict(train_args)
    ta["model"] = {k: v for k, v in ta["model"].items() if k != "name"}

    # build_run_specs (training.py): turn the train_args sub-blocks into
    # the six {cls, **kwargs} spec dicts run_emulator consumes (keyed
    # model_opts / opt_opts / lr_opts / sched_opts / trim_opts /
    # focus_opts); the classes are this experiment's fixed choices.
    specs = build_run_specs(
      train_args=ta,
      model_cls=self.model_cls,
      opt_cls=self.opt_cls,
      sched_cls=self.sched_cls)

    # make_activation (activations.py): map the activation NAME to a
    # factory act(dim) -> nn.Module (the paper's H, or a Power / Gated /
    # GatedPower variant). It is a callable, so it cannot live in the YAML;
    # inject it into the model's ResBlock options (setdefault keeps any
    # block_opts the config set).
    specs["model_opts"].setdefault(
      "block_opts", {})["act"] = make_activation(self.activation)

    # ResCNN (emulator_designs.py) needs geom to build its fixed
    # full<->theta basis-change buffers; ResMLP takes none, so inject geom
    # only for ResCNN.
    if self.model_cls is ResCNN:
      specs["model_opts"]["geom"] = self.geom

    return specs

  def train(self, train_args=None, silent=None):
    """
    Train one model on the staged sources; return its histories.

    Uses the cached sources / geometry / chi2 (build them first via
    stage_train / stage_val / build_geometry, or call run). train_args
    overrides the resolved config for THIS run (a hyperparameter sweep
    passes a varied copy); the trained model and histories are stored on
    the instance.

    Arguments:
      train_args = resolved train_args for this run (default:
                   self.train_args).
      silent     = override run_emulator's per-epoch printing; None
                   (default) -> train_args["silent"] or self.quiet. A
                   search driver passes silent=True so every trial
                   trains quietly regardless of self.quiet.

    Returns:
      (model, train_losses, medians, means, fracs) -- run_emulator's
      return, with the model restored to its best frac>0.2 epoch.
    """
    train_args = self.train_args if train_args is None else train_args
    specs = self.build_specs(train_args=train_args)
    # None -> the config/quiet default; a search driver forces silent.
    silent_run = (train_args.get("silent", False) or self.quiet
                  if silent is None else silent)

    # run_emulator (training.py): build the model / optimizer / scheduler
    # from the specs and the regime-aware data loaders, train nepochs with
    # a per-epoch val pass, and return the model (restored to its best
    # frac>0.2 epoch) plus the per-epoch histories.
    out = run_emulator(
      train_set=self.train_set,
      val_set=self.val_set,
      chi2fn=self.chi2fn,
      param_geometry=self.pgeom,
      nepochs=train_args["nepochs"],
      bs=train_args["bs"],
      loss_mode=train_args.get("loss_mode", "sqrt"),
      thresholds=self.thresholds,
      use_amp=self.use_amp,
      silent=silent_run,
      device=self.device,
      **specs)

    (self.model, self.train_losses, self.medians,
     self.means, self.fracs) = out
    return out

  def run(self, n_train=None, train_args=None):
    """
    The full pipeline (the driver's body) in one call.

    Stage the train + val sources, build the geometry + chi2 from the
    training subset, and train once. The artifacts (train_set, val_set,
    pgeom, geom, chi2fn, model) are stored on the instance for the
    diagnostics.

    Arguments:
      n_train    = absolute training-row count (default: the YAML
                   divisor) -- the N_train sweep knob.
      train_args = resolved train_args for this run (default:
                   self.train_args).

    Returns:
      (model, train_losses, medians, means, fracs).
    """
    self.stage_train(n_train=n_train)
    self.stage_val()
    self.build_geometry(train_set=self.train_set)
    return self.train(train_args=train_args)

  # --- a sweep metric ---
  def frac_above(self, threshold=0.2, source=None, bs=256):
    """
    Fraction of a source's points with delta-chi2 > threshold.

    Scores the trained model on a source (default the val set) with
    eval_source_chi2 -- the learning-curve / sweep metric (the same
    number frac>thresholds[0] tracks per epoch, recomputed on demand).

    Arguments:
      threshold = the delta-chi2 cutoff (default 0.2, the goal).
      source    = source dict to score (default self.val_set).
      bs        = forward batch size for the scoring.

    Returns:
      the fraction over `threshold`, a float.
    """
    source = self.val_set if source is None else source
    # eval_source_chi2 (training.py): score every row of `source` -- encode
    # its params -> model -> the per-row delta-chi2 against the encoded
    # truth (returns numpy params + dchi2, aligned row-for-row).
    _, dchi2 = eval_source_chi2(
      model=self.model,
      param_geometry=self.pgeom,
      chi2fn=self.chi2fn,
      source=source,
      device=self.device,
      bs=bs)
    return float((dchi2 > threshold).mean())
