#!/usr/bin/env python3
"""Activation bake-off: f(delta-chi2 > thr) vs N_train, one curve per act."""

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# This bakes off the ResBlock ACTIVATION head-to-head: for each activation it
# measures the validation f(delta-chi2 > threshold) over a grid of training-set
# sizes, and overlays the learning curves. A DOUBLE loop -- N_train (outer) x
# activation (inner) -- and the verdict is the curve SHAPE: a real inductive-
# bias win shows as a curve that keeps descending (lower sample complexity)
# where the others flatten, not as a single-N offset.
#
#     python driver/bakeoff_activation_emulator_cosmic_shear.py \
#       --yaml driver/train_single_emulator_cosmic_shear.yaml \
#       --n-min 2000 --n-points 6 --out bakeoff_act
#
#- For each N_train (geometric grid [--n-min .. --n-max], --n-max defaults to
#  the full physically-cut training pool), it stages a NESTED training subset
#  and builds the geometry ONCE, then trains a FRESH model per activation
#  (silently) and scores f(delta-chi2 > --threshold) on the FIXED validation
#  set. The data + geometry do not depend on the activation, so they are shared
#  across the inner loop (only the model is rebuilt).
#
#- It REUSES the training driver's YAML (its data, the model = ResMLP / ResCNN
#  via train_args.model.name, and the rest of train_args). The activation in the
#  YAML / the usual --activation flag is IGNORED here -- this driver sweeps the
#  activation itself.
#
#- `--yaml` (required): the config (data + train_args), training-driver schema.
#- `--activations` (default H,power,multigate,gated_power): the comma-separated
#  subset of activations to bake off.
#- `--rescale`: analytic-R mode, fixed across the bake-off (as in training).
#- `--n-min` (default 2000), `--n-max` (default = pool), `--n-points` (default
#  5): the geometric N_train grid (clamped to the pool, deduplicated).
#- `--threshold` (default 0.2): the delta-chi2 cutoff the fraction counts.
#- `--out` (default bakeoff_act): writes <out>.txt (every curve + the config,
#  np.loadtxt-loadable, one column per activation) and <out>.pdf (the figure).
#- `--quiet`: suppress stdout (the .txt and .pdf are still written).
#
#- This trains one full model per (N_train, activation), so the run is
#  len(grid) x len(activations) trainings long -- run it on the workstation.
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

import argparse
import os
import sys
import time

import numpy as np

# The emulator package sits ONE directory up from this driver/ folder.
# Put the repo root on sys.path so `import emulator` resolves regardless
# of the working directory (see the training driver for the why).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from emulator.experiment import EmulatorExperiment
from emulator.results import save_learning_curves


# the activations this driver knows how to build (make_activation names).
ACTS = ["H", "power", "multigate", "gated_power"]


def main():
  parser = argparse.ArgumentParser(
    prog="bakeoff_activation_emulator_cosmic_shear")
  parser.add_argument("--yaml",
                      dest="yaml",
                      help="config YAML (data + train_args blocks), "
                           "same schema as the training driver",
                      type=str,
                      required=True)
  parser.add_argument("--activations",
                      dest="activations",
                      help="comma-separated activations to bake off, a "
                           "subset of H,power,multigate,gated_power "
                           "(default: all four)",
                      type=str,
                      default=",".join(ACTS))
  parser.add_argument("--rescale",
                      dest="rescale",
                      help="analytic-R rescaling mode, fixed across the "
                           "bake-off: 'none' (default), 'rescaled' (v1), "
                           "or 'residual' (v2)",
                      type=str,
                      choices=["none", "rescaled", "residual"],
                      default="none")
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
                      help="output base path -> <out>.txt + <out>.pdf "
                           "(default bakeoff_act)",
                      type=str,
                      default="bakeoff_act")
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress all stdout (txt / pdf still written)",
                      action="store_true")
  args, unknown = parser.parse_known_args()

  # validate the requested activations up front (fail before training).
  activations = [a.strip() for a in args.activations.split(",")
                 if a.strip()]
  bad = [a for a in activations if a not in ACTS]
  if bad:
    raise ValueError(
      f"unknown activation(s) {bad}; choose from {ACTS}")

  # headless figure output: pick a non-interactive matplotlib backend
  # BEFORE emulator.plotting imports pyplot (done lazily below).
  os.environ.setdefault("MPLBACKEND", "Agg")

  # the setup (config, device, geometry, chi2, spec assembly) lives in
  # EmulatorExperiment; this bake-off varies the training-set size and
  # the activation. The activation passed here is a placeholder -- it is
  # overwritten per inner iteration below.
  exp = EmulatorExperiment.from_yaml(args.yaml,
                                     rescale=args.rescale,
                                     activation=activations[0],
                                     quiet=args.quiet)
  log = exp.log
  model_name = exp.model_cls.__name__
  log(f"device: {exp.device}  |  model: {model_name}  "
      f"|  rescale: {exp.rescale}  |  activations: {activations}")

  # the validation set is FIXED across the whole bake-off -- stage once.
  log("loading validation source:")
  exp.stage_val()

  # N_train grid: geometric from n_min to the pool (or --n-max), clamped
  # to the physically-cut pool; unique() drops int-cast collisions.
  pool  = exp.pool_size()
  n_max = pool if args.n_max is None else min(args.n_max, pool)
  if args.n_min >= n_max:
    raise ValueError(
      f"--n-min {args.n_min} must be below n_max {n_max} (pool {pool})")
  sizes = np.unique(
    np.geomspace(args.n_min, n_max, args.n_points).astype(int))
  log(f"pool {pool}  |  N_train grid: {sizes.tolist()}")

  # curves[act] = {N_train: frac}. N_train is the OUTER loop so the
  # staged subset + geometry are built once and shared across the inner
  # activation loop (they do not depend on the activation).
  curves = {act: {} for act in activations}
  for N in sizes:
    exp.stage_train(n_train=int(N))
    exp.build_geometry()
    for act in activations:
      t0 = time.time()
      # set the activation read by build_specs, then train a fresh model
      # on the shared data + geometry.
      exp.activation = act
      exp.train(silent=True)
      f = exp.frac_above(threshold=args.threshold)
      curves[act][int(N)] = f
      log(f"  N_train {int(N):8d}  {act:12s}  "
          f"f(>{args.threshold:g}) {f:.4f}  ({time.time() - t0:.0f}s)")

  # save every curve + the config as a plain-text table, one column per
  # activation (np.loadtxt-loadable; the # header lines are skipped).
  out_txt = args.out + ".txt"
  out_pdf = args.out + ".pdf"
  save_learning_curves(
    path=out_txt,
    sizes=sizes,
    curves={act: [curves[act][int(n)] for n in sizes]
            for act in activations},
    meta={"model": model_name, "rescale": exp.rescale,
          "activation": "swept", "threshold": args.threshold,
          "pool": pool})
  log(f"saved curve data -> {out_txt}")

  # overlaid figure: one curve per activation.
  from emulator.plotting import plot_learning_curves
  plot_learning_curves(curves=curves,
                       threshold=args.threshold,
                       savepath=out_pdf)
  log(f"saved figure -> {out_pdf}")


if __name__ == "__main__":
  main()
