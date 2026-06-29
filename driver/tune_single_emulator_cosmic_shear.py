#!/usr/bin/env python3
"""Optuna search over the single xi emulator's hyperparams (ResMLP/ResCNN)."""

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# This is the TUNING twin of train_single_emulator_cosmic_shear.py: the same
# single cosmic-shear (xi) emulator setup (ResMLP or ResCNN, chosen in the
# YAML), but instead of one training run it runs an Optuna study that minimizes
# the validation f(delta-chi2 > 0.2).
#
#     python driver/tune_single_emulator_cosmic_shear.py \
#       --yaml driver/tune_single_emulator_cosmic_shear.yaml \
#       --n-trials 50 --timeout 4200
#
#- WHICH hyperparameters are searched is read from the YAML train_args block. A
#  leaf is EITHER a fixed scalar, OR a SEARCH RANGE written as a 4-item list
#  [default, min, max, kind] with kind one of int / float / log (a whitespace
#  string "default min max kind" also works). For example:
#
#      lr:
#        lr_base: [0.0025, 1.0e-5, 1.0e-1, log]   # searched, log scale
#        bs_base: 64.0                            # fixed
#      model:
#        int_dim_res: [128, 64, 256, int]         # searched, integer
#        n_blocks:    4                           # fixed
#
#  The first value is the DEFAULT: the plain training driver uses it, and this
#  search WARM-STARTS trial 0 from it. So one YAML serves both drivers.
#
#- `--yaml` (required): config with the `data` and `train_args` blocks (same
#  schema as the training driver; train_args may now carry search ranges).
#- `--n-trials` (default 50) and `--timeout` (seconds, optional) bound the study.
#- `--rescale` / `--activation` are fixed across the study (the analytic-R mode
#  and the ResBlock activation are not searched here) -- see the training driver.
#- `--quiet` suppresses all stdout (per-trial lines and the final summary).
#
#- The fixed single-emulator choices (probe = xi, AdamW, ReduceLROnPlateau,
#  use_amp = False, the report thresholds, the resmlp/rescnn registry) are
#  EmulatorExperiment defaults (emulator/experiment.py), shared with the
#  training driver. The MODEL is the YAML's choice (train_args.model.name =
#  resmlp | rescnn), fixed across the study (only the hyperparameters vary).
#
#- Output: stdout only -- a per-trial line (frac>0.2, running best, params) and a
#  final summary of the best frac>0.2 and the best parameters. No files written.
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

import argparse
import os
import sys

import optuna

# The emulator package sits ONE directory up from this driver/
# folder; put the repo root on sys.path so `import emulator`
# resolves no matter the working directory (see the training
# driver for the why).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from emulator.training import suggest_train_args, search_defaults
from emulator.experiment import EmulatorExperiment


def main():
  parser = argparse.ArgumentParser(
    prog="tune_single_emulator_cosmic_shear")
  parser.add_argument("--yaml",
                      dest="yaml",
                      help="config YAML with the data and "
                           "train_args blocks (train_args may "
                           "carry [default, min, max, kind] ranges)",
                      type=str,
                      required=True)
  parser.add_argument("--rescale",
                      dest="rescale",
                      help="analytic-R rescaling mode, fixed across "
                           "the study: 'none' (default), 'rescaled' "
                           "(v1), or 'residual' (v2)",
                      type=str,
                      choices=["none", "rescaled", "residual"],
                      default="none")
  parser.add_argument("--activation",
                      dest="activation",
                      help="ResBlock activation, fixed across the "
                           "study: 'H' (default), 'power', "
                           "'multigate', or 'gated_power'",
                      type=str,
                      choices=["H", "power", "multigate",
                               "gated_power"],
                      default="H")
  parser.add_argument("--n-trials",
                      dest="n_trials",
                      help="number of Optuna trials (default 50)",
                      type=int,
                      default=50)
  parser.add_argument("--timeout",
                      dest="timeout",
                      help="stop the study after this many seconds "
                           "(optional; default no limit)",
                      type=int,
                      default=None)
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress all stdout (per-trial lines "
                           "and the final summary)",
                      action="store_true")
  args, unknown = parser.parse_known_args()

  # the whole setup -- config parse, model resolution, device, data
  # staging, geometry, chi2, and the per-run spec assembly -- lives in
  # EmulatorExperiment, shared with the training driver. The geometry /
  # chi2 / activation are FIXED across the study, so build them once
  # here; only the searched train_args vary per trial. The fixed
  # single-emulator choices are the EmulatorExperiment defaults; the
  # MODEL is the YAML's choice (train_args.model.name).
  exp = EmulatorExperiment.from_yaml(args.yaml,
                                     rescale=args.rescale,
                                     activation=args.activation,
                                     quiet=args.quiet)
  # the experiment's quiet-gated logger, reused below.
  log = exp.log
  log(f"device: {exp.device}  |  rescale: {exp.rescale}  "
      f"|  activation: {exp.activation}")
  log("loading sources:")
  exp.stage_train()
  exp.stage_val()
  exp.build_geometry()

  # raw_train_args KEEPS the search ranges (exp.train_args collapsed
  # them to defaults); suggest_train_args resolves them per trial.
  raw_ta = exp.raw_train_args
  ranges = search_defaults(raw_ta)
  if not ranges:
    log("WARNING: no [default, min, max, kind] search ranges in "
        "train_args -- every trial is identical.")

  def objective(trial):
    # this trial's concrete train_args (each range -> a suggestion);
    # exp.train builds the per-run specs (model / optimizer / scheduler
    # + activation + the ResCNN geom) on the FIXED data + geometry.
    # silent=True so every trial trains quietly even when the study
    # itself is not --quiet.
    ta = suggest_train_args(trial, raw_ta)
    (_m, _tl, medians,
     _mn, fracs) = exp.train(train_args=ta, silent=True)
    # the run restored its best-frac>0.2 epoch (median tiebreaker);
    # the study minimizes that frac>0.2.
    best = min(range(len(fracs)),
               key=lambda i: (fracs[i][0].item(), medians[i]))
    trial.set_user_attr("median", float(medians[best]))
    return fracs[best][0].item()

  def log_trial(study, trial):
    log(f"trial {trial.number:3d}  frac>0.2 {trial.value:.4f}"
        f"  best {study.best_value:.4f}  {trial.params}")

  # quiet Optuna's own per-trial INFO spam (we print our own line).
  optuna.logging.set_verbosity(optuna.logging.WARNING)
  # minimize frac>0.2; TPE seed fixed for reproducibility.
  study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=0))
  # warm-start trial 0 from the YAML defaults (the range first
  # values), so the search begins at the known-good config.
  if ranges:
    study.enqueue_trial(ranges)
  study.optimize(objective,
                 n_trials=args.n_trials,
                 timeout=args.timeout,
                 callbacks=[log_trial])

  log("\n--- search complete ---")
  log(f"best frac>0.2: {study.best_value:.4f}  "
      f"(median {study.best_trial.user_attrs.get('median', float('nan')):.4f})")
  log("best params:")
  for k, v in study.best_trial.params.items():
    log(f"  {k}: {v}")


if __name__ == "__main__":
  main()
