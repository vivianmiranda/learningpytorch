#!/usr/bin/env python3
"""Optuna search over the single xi ResMLP emulator's hyperparams."""

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# This is the TUNING twin of train_single_resmlp_emulator_cosmic_shear.py: the
# same single cosmic-shear (xi) ResMLP setup, but instead of one training run it
# runs an Optuna study that minimizes the validation f(delta-chi2 > 0.2).
#
#     python driver/tune_single_resmlp_emulator_cosmic_shear.py \
#       --yaml driver/tune_single_resmlp_emulator_cosmic_shear.yaml \
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
#- Fixed in the script: probe = xi, model = ResMLP, optimizer = AdamW,
#  scheduler = ReduceLROnPlateau, use_amp = False, and the report thresholds.
#
#- Output: stdout only -- a per-trial line (frac>0.2, running best, params) and a
#  final summary of the best frac>0.2 and the best parameters. No files written.
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

import argparse
import os
import sys

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import yaml
import optuna

# The emulator package sits ONE directory up from this driver/
# folder; put the repo root on sys.path so `import emulator`
# resolves no matter the working directory (see the training
# driver for the why).
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
  run_emulator, build_run_specs, pick_device,
  suggest_train_args, search_defaults)


# --- fixed choices for THIS driver (a single xi ResMLP) ---
PROBE     = "xi"
MODEL_CLS = ResMLP
OPT_CLS   = optim.AdamW
SCHED_CLS = lr_scheduler.ReduceLROnPlateau
USE_AMP   = False
THRESHOLDS = torch.tensor([0.2, 0.5, 1.0, 10.0, 100.0])


def main():
  parser = argparse.ArgumentParser(
    prog="tune_single_resmlp_emulator_cosmic_shear")
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
  # raw_ta KEEPS the search ranges (suggest_train_args resolves them
  # per trial); do NOT collapse them to defaults here.
  raw_ta = cfg["train_args"]

  device = pick_device()
  torch.set_float32_matmul_precision("high")
  log(f"device: {device}  |  rescale: {args.rescale}  "
      f"|  activation: {args.activation}")

  gen = torch.Generator().manual_seed(int(data["split_seed"]))
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
                        with_means=False)

  # geometry, chi2, and activation are FIXED across the study (only
  # the searched train_args vary per trial), so build them once.
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
  act = make_activation(args.activation)

  ranges = search_defaults(raw_ta)
  if not ranges:
    log("WARNING: no [default, min, max, kind] search ranges in "
        "train_args -- every trial is identical.")

  def objective(trial):
    # this trial's concrete train_args (each range -> a suggestion).
    ta = suggest_train_args(trial, raw_ta)
    specs = build_run_specs(train_args=ta,
                            model_cls=MODEL_CLS,
                            opt_cls=OPT_CLS,
                            sched_cls=SCHED_CLS)
    specs["model_opts"].setdefault("block_opts", {})["act"] = act
    (_m, _tl, medians,
     _mn, fracs) = run_emulator(train_set=train_set,
                                val_set=val_set,
                                chi2fn=chi2fn,
                                param_geometry=pgeom,
                                nepochs=ta["nepochs"],
                                bs=ta["bs"],
                                loss_mode=ta.get("loss_mode", "sqrt"),
                                thresholds=THRESHOLDS,
                                use_amp=USE_AMP,
                                silent=True,
                                device=device,
                                **specs)
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
