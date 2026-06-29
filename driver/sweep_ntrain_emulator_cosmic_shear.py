#!/usr/bin/env python3
"""N_train learning curve: f(delta-chi2 > thr) vs N_train for ONE config."""

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# This sweeps the TRAINING-SET SIZE for a single fixed configuration and
# records the validation f(delta-chi2 > threshold) at each size -- the learning
# curve that says whether the floor is data-limited (curve still falling at the
# largest N) or capacity / architecture-limited (a flat tail).
#
#     python driver/sweep_ntrain_emulator_cosmic_shear.py \
#       --yaml driver/train_single_emulator_cosmic_shear.yaml \
#       --n-min 2000 --n-points 6 --out ntrain_resmlp
#
#- It REUSES the training driver's YAML (and its model / rescale / activation
#  choices). To compare architectures or chi2 modes, run it once per config --
#  change train_args.model.name (or --rescale / --activation) and a different
#  --out -- then overlay the saved <out>.json curves.
#
#- For each N_train in a geometric grid [--n-min .. --n-max] (--n-max defaults
#  to the full physically-cut training pool), it stages a NESTED training subset
#  of that size, rebuilds the geometry from it, trains a FRESH model (silently),
#  and scores f(delta-chi2 > --threshold) on the FIXED validation set.
#
#- `--yaml` (required): the config (data + train_args), same schema as the
#  training driver; train_args.model.name picks ResMLP / ResCNN.
#- `--rescale` / `--activation`: as in the training driver, fixed across the
#  sweep (the analytic-R mode and the ResBlock activation).
#- `--n-min` (default 2000), `--n-max` (default = pool), `--n-points` (default
#  5): the geometric N_train grid (clamped to the pool, deduplicated).
#- `--threshold` (default 0.2): the delta-chi2 cutoff the fraction counts.
#- `--out` (default ntrain_sweep): writes <out>.json (the curve + the config it
#  came from) and <out>.pdf (a single-curve figure).
#- `--quiet`: suppress stdout (the figure and json are still written).
#
#- This trains ONE full model per grid point, so a sweep is --n-points
#  trainings long -- run it on the workstation, where cosmolike lives.
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

import argparse
import os
import sys
import json
import time

import numpy as np

# The emulator package sits ONE directory up from this driver/ folder.
# Put the repo root on sys.path so `import emulator` resolves regardless
# of the working directory (see the training driver for the why).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from emulator.experiment import EmulatorExperiment


def main():
  parser = argparse.ArgumentParser(
    prog="sweep_ntrain_emulator_cosmic_shear")
  parser.add_argument("--yaml",
                      dest="yaml",
                      help="config YAML (data + train_args blocks), "
                           "same schema as the training driver",
                      type=str,
                      required=True)
  parser.add_argument("--rescale",
                      dest="rescale",
                      help="analytic-R rescaling mode, fixed across the "
                           "sweep: 'none' (default), 'rescaled' (v1), "
                           "or 'residual' (v2)",
                      type=str,
                      choices=["none", "rescaled", "residual"],
                      default="none")
  parser.add_argument("--activation",
                      dest="activation",
                      help="ResBlock activation, fixed across the "
                           "sweep: 'H' (default), 'power', 'multigate', "
                           "or 'gated_power'",
                      type=str,
                      choices=["H", "power", "multigate",
                               "gated_power"],
                      default="H")
  parser.add_argument("--n-min",
                      dest="n_min",
                      help="smallest N_train in the grid (default 2000)",
                      type=int,
                      default=2000)
  parser.add_argument("--n-max",
                      dest="n_max",
                      help="largest N_train in the grid (default and "
                           "ceiling: the physically-cut training pool)",
                      type=int,
                      default=None)
  parser.add_argument("--n-points",
                      dest="n_points",
                      help="number of geometric grid points (default 5)",
                      type=int,
                      default=5)
  parser.add_argument("--threshold",
                      dest="threshold",
                      help="delta-chi2 cutoff the fraction counts "
                           "(default 0.2, the emulator goal)",
                      type=float,
                      default=0.2)
  parser.add_argument("--out",
                      dest="out",
                      help="output base path -> <out>.json + <out>.pdf "
                           "(default ntrain_sweep)",
                      type=str,
                      default="ntrain_sweep")
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress all stdout (json / pdf still written)",
                      action="store_true")
  args, unknown = parser.parse_known_args()

  # headless figure output: pick a non-interactive matplotlib backend
  # BEFORE emulator.plotting imports pyplot (done lazily below).
  os.environ.setdefault("MPLBACKEND", "Agg")

  # the setup (config, device, geometry, chi2, spec assembly) lives in
  # EmulatorExperiment; this sweep varies only the training-set size.
  exp = EmulatorExperiment.from_yaml(args.yaml,
                                     rescale=args.rescale,
                                     activation=args.activation,
                                     quiet=args.quiet)
  log = exp.log
  model_name = exp.model_cls.__name__
  log(f"device: {exp.device}  |  model: {model_name}  "
      f"|  rescale: {exp.rescale}  |  activation: {exp.activation}")

  # the validation set is FIXED across the sweep -- stage it once.
  log("loading validation source:")
  exp.stage_val()

  # N_train grid: geometric from n_min to the pool (or --n-max), clamped
  # to the physically-cut pool so every size is loadable; unique() drops
  # the collisions the int cast makes at the low end.
  pool  = exp.pool_size()
  n_max = pool if args.n_max is None else min(args.n_max, pool)
  if args.n_min >= n_max:
    raise ValueError(
      f"--n-min {args.n_min} must be below n_max {n_max} (pool {pool})")
  sizes = np.unique(
    np.geomspace(args.n_min, n_max, args.n_points).astype(int))
  log(f"pool {pool}  |  N_train grid: {sizes.tolist()}")

  fracs = []
  for N in sizes:
    t0 = time.time()
    # nested training subset of size N (fresh seeded gen), geometry
    # rebuilt from its means, a fresh model trained quietly.
    exp.stage_train(n_train=int(N))
    exp.build_geometry()
    exp.train(silent=True)
    f = exp.frac_above(threshold=args.threshold)
    fracs.append(f)
    log(f"  N_train {int(N):8d}  f(>{args.threshold:g}) {f:.4f}  "
        f"({time.time() - t0:.0f}s)")

  # save the curve + the config it came from, so several runs (one per
  # architecture / chi2 mode) can be overlaid later.
  result = {
    "yaml":       args.yaml,
    "model":      model_name,
    "rescale":    exp.rescale,
    "activation": exp.activation,
    "threshold":  args.threshold,
    "pool":       pool,
    "sizes":      [int(n) for n in sizes],
    "fracs":      [float(x) for x in fracs],
  }
  out_json = args.out + ".json"
  out_pdf  = args.out + ".pdf"
  with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
  log(f"saved curve data -> {out_json}")

  # one-curve figure (overlay several <out>.json yourself to compare).
  from emulator.plotting import plot_learning_curves
  plot_learning_curves(
    curves={f"{model_name} ({exp.rescale})": (sizes, fracs)},
    threshold=args.threshold,
    savepath=out_pdf)
  log(f"saved figure -> {out_pdf}")


if __name__ == "__main__":
  main()
