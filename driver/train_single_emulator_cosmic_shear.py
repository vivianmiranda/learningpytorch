#!/usr/bin/env python3
"""Train ONE cosmic-shear (xi) emulator (ResMLP or ResCNN) from a YAML."""

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# This script trains a SINGLE cosmic-shear (xi) emulator -- a ResMLP, or a
# ResCNN (ResMLP trunk + a 1D-CNN appendix), chosen in the YAML -- that maps
# cosmological parameters to the whitened, masked xi data vector, with the
# full-3x2pt chi2 (cosmolike's masked inverse covariance) as the loss.
#
#     python driver/train_single_emulator_cosmic_shear.py \
#       --yaml driver/train_single_emulator_cosmic_shear.yaml \
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
#    sub-blocks model (name = resmlp | rescnn, then that model's kwargs --
#    int_dim_res, n_blocks, and for rescnn kernel_size / channels /
#    n_blocks_cnn / gate_init), optimizer (weight_decay), lr (lr_base, bs_base,
#    warmup_epochs), scheduler (mode, patience, factor), and trim / focus (the
#    robustness schedules).
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
#- The fixed single-emulator choices -- probe = xi, optimizer = AdamW, scheduler
#  = ReduceLROnPlateau, use_amp = False, the reported delta-chi2 thresholds
#  [0.2, 0.5, 1, 10, 100] (0.2 = the goal and the model-selection metric), and
#  the resmlp/rescnn registry -- are EmulatorExperiment defaults (in emulator/
#  experiment.py, which also holds the setup so a sweep script can reuse it).
#  The MODEL is the YAML's choice (train_args.model.name = resmlp | rescnn).
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

# The emulator package sits ONE directory up from this driver/
# folder. Put the repo root on sys.path so `import emulator`
# resolves regardless of the working directory: launching
# `python driver/foo.py` puts driver/ (the script's own folder),
# NOT the repo root, on sys.path -- so without this the absolute
# import below would fail.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from emulator.experiment import EmulatorExperiment


def main():
  parser = argparse.ArgumentParser(
    prog="train_single_emulator_cosmic_shear")
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

  # all the setup -- config parse + model resolution + device + data
  # staging + geometry + chi2 + spec assembly -- lives in
  # EmulatorExperiment, so a sweep script reuses it instead of copying
  # it. The fixed single-emulator choices (probe = xi, AdamW,
  # ReduceLROnPlateau, use_amp = False, the report thresholds, the
  # resmlp/rescnn registry) are the EmulatorExperiment defaults; the
  # MODEL is the YAML's choice (train_args.model.name). This driver
  # passes only what it varies (the YAML, rescale, activation, quiet).
  exp = EmulatorExperiment.from_yaml(args.yaml,
                                     rescale=args.rescale,
                                     activation=args.activation,
                                     quiet=args.quiet)
  # the experiment's quiet-gated logger, reused below.
  log = exp.log
  log(f"device: {exp.device}  |  rescale: {exp.rescale}")
  log("loading sources:")
  (model, train_losses, medians,
   means, fracs) = exp.run()

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
                              param_geometry=exp.pgeom,
                              chi2fn=exp.chi2fn,
                              train_set=exp.train_set,
                              val_set=exp.val_set,
                              device=exp.device)
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
                                   param_geometry=exp.pgeom,
                                   chi2fn=exp.chi2fn,
                                   val_set=exp.val_set,
                                   device=exp.device)
    log(f"hardness: joint log-linear R2 {hd['r2']:.3f}  |  "
        f"ln(omega_b h2) alone {hd['r2_omega']:.3f}")
    # (3) local-linear data floor -- ONLY for a plain chi2fn (the
    # rescaled encode/chi2 would need each point's own R).
    floor = None
    if not getattr(exp.chi2fn, "needs_params", False):
      floor = local_linear_floor(model=model,
                                 param_geometry=exp.pgeom,
                                 chi2fn=exp.chi2fn,
                                 train_set=exp.train_set,
                                 val_set=exp.val_set,
                                 device=exp.device)
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
                     thresholds=exp.thresholds,
                     coverage=cov,
                     floor=floor,
                     hard_dir=hd,
                     savepath=args.diagnostic)
    log(f"saved diagnostics -> {args.diagnostic}")


if __name__ == "__main__":
  main()
