#!/usr/bin/env python3
"""Train ONE cosmic-shear (xi) ResMLP emulator from a YAML config."""

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# This script trains a SINGLE cosmic-shear (xi) emulator: a ResMLP that maps
# cosmological parameters to the whitened, masked xi data vector, with the
# full-3x2pt chi2 (cosmolike's masked inverse covariance) as the loss.
#
#     python driver/train_single_resmlp_emulator_cosmic_shear.py \
#       --yaml driver/train_single_resmlp_emulator_cosmic_shear.yaml \
#       --diagnostic diagnostic.pdf
#
#- Run it from a directory where the YAML's (relative) data paths resolve, e.g.
#  the project root that holds ./dvs/, and with $ROOTDIR exported so cosmolike
#  finds its dataset under $ROOTDIR/external_modules/data. cosmolike runs only
#  on the workstation, so train there.
#
#- The emulator package (one directory up from driver/) is added to sys.path
#  automatically, so `import emulator` works no matter where you launch from.
#
#- `--yaml` (required) is the configuration file: EVERY hyperparameter lives
#  there, so nothing in the script is a magic number. It has two blocks.
#  - `data`: the input file paths, the physical cut and split settings
#    (omegabh2_cut, train_divisor, val_divisor, split_seed, ram_frac), and the
#    cosmolike dataset (cosmolike_data_dir, cosmolike_dataset).
#  - `train_args`: the training knobs (nepochs, bs, loss_mode, silent) plus the
#    sub-blocks model (int_dim_res, n_blocks), optimizer (weight_decay), lr
#    (lr_base, bs_base, warmup_epochs), scheduler (mode, patience, factor), and
#    trim / focus (the robustness schedules).
#
#- `--diagnostic` (optional) saves a MULTIPAGE diagnostics PDF to the given
#  path: page 1 (2x2) the training history + the coverage diagnostic (do the
#  failures sit in sparse training regions?); page 2 the local-linear data-only
#  floor (model vs floor delta-chi2; plain chi2fn only, skipped under
#  --rescale); page 3 the hard-direction regression (which log-param combination
#  predicts hardness). Omit it for no figure.
#
#- `--rescale` (optional, default `none`) divides out a fast analytic reference
#  R so the net emulates a flatter target (the chi2 is always on the original
#  dv). `rescaled` = RescaledChi2 (v1: R divides the net output, so the chi2
#  gradient carries a per-cosmology 1/R factor); `residual` = ResidualBaseChi2
#  (v2: R moves only the baseline, plain chi2). Both need cosmolike's angle map.
#
#- `--activation` (optional, default `H`) sets the ResBlock activation: `H` (the
#  paper's leaky/Swish gate), `power` (bounded learnable tail exponent),
#  `multigate` (K=3 gates), or `gated_power` (K=3 gates + tail exponent).
#
#- `--quiet` (optional) suppresses ALL stdout -- the driver's prints,
#  load_source's per-source line, and run_emulator's per-epoch log. The
#  --diagnostic PDF is still written.
#
#- Fixed IN THE SCRIPT (not the YAML), the choices that make this THIS driver:
#  probe = xi, model = ResMLP, optimizer = AdamW, scheduler = ReduceLROnPlateau,
#  use_amp = False, and the reported delta-chi2 thresholds [0.2, 0.5, 1, 10,
#  100] (0.2 is the goal and the model-selection metric). To train a different
#  model or probe, copy this driver and change those constants.
#
#- Inputs (paths set in the YAML `data` block):
#
#      <train_dv>.npy      training data vectors   (memmapped)
#      <train_params>.txt  training parameters     (weight, lnp, <params>, chi2)
#      <train_covmat>      parameter covmat        (header line = param names)
#      <val_dv>.npy        validation data vectors
#      <val_params>.txt    validation parameters
#
#- Outputs:
#
#      stdout            per-epoch progress (unless train_args.silent: true) and
#                        a final "best epoch N: frac>0.2 ... median ..." line.
#      <--diagnostic>    the multipage diagnostics PDF, only if the flag is set.
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

import argparse
import os
import sys

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import yaml

# The emulator package sits ONE directory up from this driver/
# folder. Put the repo root on sys.path so `import emulator`
# resolves regardless of the working directory: launching
# `python driver/foo.py` puts driver/ (the script's own folder),
# NOT the repo root, on sys.path -- so without this the absolute
# imports below would fail.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from emulator.data_staging import read_param_names, load_source
from emulator.geometries_parameter import ParamGeometry
from emulator.geometries_output import DataVectorGeometry
from emulator.loss_functions import make_chi2
from emulator.emulator_designs import ResMLP
from emulator.activations import make_activation
from emulator.training import (
  run_emulator, build_run_specs, pick_device, default_train_args)


# --- fixed choices for THIS driver (a single xi ResMLP) ---
PROBE     = "xi"
MODEL_CLS = ResMLP
OPT_CLS   = optim.AdamW
SCHED_CLS = lr_scheduler.ReduceLROnPlateau
USE_AMP   = False
# delta-chi2 cutoffs the val fractions are reported over; the
# first (0.2) is the emulator goal and the model-selection metric.
THRESHOLDS = torch.tensor([0.2, 0.5, 1.0, 10.0, 100.0])


def main():
  parser = argparse.ArgumentParser(
    prog="train_single_resmlp_emulator_cosmic_shear")
  parser.add_argument("--yaml",
                      dest="yaml",
                      help="training YAML with the data and "
                           "train_args blocks",
                      type=str,
                      required=True)
  parser.add_argument("--diagnostic",
                      dest="diagnostic",
                      help="if set, save a diagnostics figure (the "
                           "training history + the coverage "
                           "diagnostic) to this path (e.g. "
                           "diagnostic.pdf)",
                      type=str,
                      default=None)
  parser.add_argument("--rescale",
                      dest="rescale",
                      help="analytic-R rescaling mode: 'none' "
                           "(plain chi2, default), 'rescaled' "
                           "(RescaledChi2 / v1: R divides the net "
                           "output), or 'residual' "
                           "(ResidualBaseChi2 / v2: R moves only "
                           "the baseline)",
                      type=str,
                      choices=["none", "rescaled", "residual"],
                      default="none")
  parser.add_argument("--activation",
                      dest="activation",
                      help="ResBlock activation: 'H' (the paper's "
                           "H, default), 'power', 'multigate' "
                           "(K=3), or 'gated_power' (K=3)",
                      type=str,
                      choices=["H", "power", "multigate",
                               "gated_power"],
                      default="H")
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress all stdout: the driver's "
                           "prints, load_source's per-source line, "
                           "and run_emulator's per-epoch log",
                      action="store_true")
  args, unknown = parser.parse_known_args()

  # --quiet silences the driver's own stdout; run_emulator and
  # load_source are silenced through their own flags below.
  def log(*a, **kw):
    if not args.quiet:
      print(*a, **kw)

  with open(args.yaml) as f:
    cfg = yaml.safe_load(f)
  for block in ("data", "train_args"):
    if block not in cfg:
      raise KeyError(f"YAML is missing the required block: {block!r}")
  data = cfg["data"]
  # default_train_args collapses any [default, min, max, kind]
  # search ranges to their default, so a YAML written for the
  # tuning driver also trains fine here (it uses the first value).
  ta   = default_train_args(cfg["train_args"])

  device = pick_device()
  # TF32 tensor-core float32 matmuls (Ampere+); no effect on
  # CPU/MPS. One-time global switch.
  torch.set_float32_matmul_precision("high")
  log(f"device: {device}  |  rescale: {args.rescale}")

  # a local Generator fixes the train/val split independently of
  # the model-init RNG.
  gen = torch.Generator().manual_seed(int(data["split_seed"]))

  # parameter names come from the TRAINING covmat header and are
  # reused for the val cut (same columns).
  names = read_param_names(data["train_covmat"])

  log("loading sources:")
  train_set = load_source(dv_path=data["train_dv"],
                          params_path=data["train_params"],
                          names=names,
                          cut=data["omegabh2_cut"],
                          divisor=data["train_divisor"],
                          gen=gen,
                          ram_frac=data.get("ram_frac", 0.7),
                          with_means=True,
                          verbose=not args.quiet)
  val_set = load_source(dv_path=data["val_dv"],
                        params_path=data["val_params"],
                        names=names,
                        cut=data["omegabh2_cut"],
                        divisor=data["val_divisor"],
                        gen=gen,
                        ram_frac=data.get("ram_frac", 0.7),
                        with_means=False,
                        verbose=not args.quiet)

  # input whitening from the covmat; output geometry + chi2 from
  # cosmolike. Both are built from the TRAINING source only and
  # applied unchanged to val.
  pgeom = ParamGeometry.from_covmat(device=device,
                                    center=train_set["C_mean"],
                                    covmat_path=data["train_covmat"])
  geom = DataVectorGeometry.from_cosmolike(
    device=device,
    dv_center=train_set["dv_mean"],
    data_dir=data["cosmolike_data_dir"],
    dataset=data["cosmolike_dataset"],
    probe=PROBE)
  chi2fn = make_chi2(
    geom=geom,
    rescale=args.rescale,
    param_geometry=pgeom,
    cosmo_mid=train_set["C"][train_set["idx"]].mean(0),
    data_dir=data["cosmolike_data_dir"],
    dataset=data["cosmolike_dataset"])

  # the six {cls, **kwargs} spec dicts run_emulator consumes; the
  # CLASSES are this driver's fixed choices, the settings come
  # from the YAML. Keyed by run_emulator's argument names, so they
  # splat straight in as **specs below.
  specs = build_run_specs(train_args=ta,
                          model_cls=MODEL_CLS,
                          opt_cls=OPT_CLS,
                          sched_cls=SCHED_CLS)

  # the activation is a factory callable, so it cannot live in the
  # YAML -- select it by name here and inject it into the model's
  # ResBlock options (setdefault keeps any block_opts the YAML set).
  act = make_activation(args.activation)
  specs["model_opts"].setdefault("block_opts", {})["act"] = act

  (model, train_losses, medians,
   means, fracs) = run_emulator(train_set=train_set,
                                val_set=val_set,
                                chi2fn=chi2fn,
                                param_geometry=pgeom,
                                nepochs=ta["nepochs"],
                                bs=ta["bs"],
                                loss_mode=ta.get("loss_mode", "sqrt"),
                                thresholds=THRESHOLDS,
                                use_amp=USE_AMP,
                                silent=ta.get("silent", False)
                                       or args.quiet,
                                device=device,
                                **specs)

  # run_emulator already restored the best-frac>0.2 epoch; report
  # which one that was. fracs[i][0] is the frac>0.2 at epoch i+1,
  # with the median as the tiebreaker (the loop's own rule).
  best = min(range(len(fracs)),
             key=lambda i: (fracs[i][0].item(), medians[i]))
  log(f"best epoch {best + 1}: "
      f"frac>0.2 {fracs[best][0].item():.4f}  "
      f"median {medians[best]:.4f}")

  if args.diagnostic is not None:
    # headless figure output: select a non-interactive matplotlib
    # backend BEFORE pyplot is imported (emulator.plotting imports
    # it at module load), then build the diagnostics figure.
    os.environ.setdefault("MPLBACKEND", "Agg")
    from emulator.diagnostics import (
      coverage_diagnostic, local_linear_floor,
      hard_direction_regression)
    from emulator.plotting import plot_diagnostics
    # (1) coverage: do the failing val points sit in sparse regions
    # of the training set? (local kNN sparsity vs delta-chi2).
    cov = coverage_diagnostic(model=model,
                              param_geometry=pgeom,
                              chi2fn=chi2fn,
                              train_set=train_set,
                              val_set=val_set,
                              device=device)
    log(f"coverage: spearman(knn_dist, log dchi2) "
        f"{cov['spearman']:+.3f}  |  median knn good "
        f"{cov['median_good']:.3f} bad {cov['median_bad']:.3f}  "
        f"|  frac>0.2 dense {cov['frac_dense']:.3f} sparse "
        f"{cov['frac_sparse']:.3f}")
    log("=> " + ("COVERAGE-limited: failures sit in sparse regions"
                 if cov["coverage_limited"]
                 else "NOT clearly coverage: failures not sparser"))
    # (2) hard-direction regression (works for any chi2fn).
    hd = hard_direction_regression(model=model,
                                   param_geometry=pgeom,
                                   chi2fn=chi2fn,
                                   val_set=val_set,
                                   device=device)
    log(f"hardness: joint log-linear R2 {hd['r2']:.3f}  |  "
        f"ln(omega_b h2) alone {hd['r2_omega']:.3f}")
    # (3) local-linear data floor -- ONLY for a plain chi2fn (the
    # rescaled encode/chi2 would need each point's own R).
    floor = None
    if not getattr(chi2fn, "needs_params", False):
      floor = local_linear_floor(model=model,
                                 param_geometry=pgeom,
                                 chi2fn=chi2fn,
                                 train_set=train_set,
                                 val_set=val_set,
                                 device=device)
      log(f"floor: f_model {floor['f_model']:.3f}  "
          f"f_floor {floor['f_floor']:.3f}  "
          f"pure hardness {floor['f_hard']:.3f}")
    else:
      log("floor: skipped (local-linear floor needs a plain "
          "chi2fn; --rescale is on)")
    plot_diagnostics(train_losses=train_losses,
                     medians=medians,
                     means=means,
                     fracs=fracs,
                     thresholds=THRESHOLDS,
                     coverage=cov,
                     floor=floor,
                     hard_dir=hd,
                     savepath=args.diagnostic)
    log(f"saved diagnostics -> {args.diagnostic}")


if __name__ == "__main__":
  main()
