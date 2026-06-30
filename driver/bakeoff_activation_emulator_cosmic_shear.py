#!/usr/bin/env python3
"""Activation bake-off: f(delta-chi2 > thr) vs N_train, one curve per act."""

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# This bakes off the ResBlock ACTIVATION head-to-head: for each activation it
# measures the validation f(delta-chi2 > threshold) over a grid of training-set
# sizes, and overlays the learning curves. A DOUBLE loop -- N_train x activation
# -- and the verdict is the curve SHAPE: a real inductive-bias win shows as a
# curve that keeps descending (lower sample complexity) where the others
# flatten, not as a single-N offset.
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
#  across the activations at a given N (only the model is rebuilt).
#
#- MULTIPLE GPUs (one node): the activations are split across GPUs, one process
#  per GPU, each running every N_train for its share of the activations. Every
#  activation costs about the same and every GPU runs the whole N_train grid, so
#  the GPUs carry equal work with NO cost-aware balancing (unlike the N_train
#  sweep, which needs it). At most len(activations) GPUs are used; with the
#  default 4 activations and 8 GPUs, 4 GPUs sit idle. With --n-gpus 4:
#
#     python driver/bakeoff_activation_emulator_cosmic_shear.py \
#       --yaml driver/train_single_emulator_cosmic_shear.yaml \
#       --n-points 8 --n-gpus 4 --out bakeoff_act
#
#  With one GPU (or none, e.g. the Apple-MPS dev machine) it runs the plain
#  serial double loop, so the same script runs everywhere.
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
#- `--n-gpus` (default: all visible CUDA devices, capped at len(activations)):
#  how many GPUs to split the activations across. 1, or no CUDA, is serial.
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
import torch
import yaml

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


def _bakeoff_worker(gpu_id, my_acts, sizes, cfg, rescale,
                    threshold, result_q):
  """
  One GPU's share of the bake-off; runs in its own process.

  Pins itself to GPU `gpu_id`, builds its own EmulatorExperiment there, and
  runs the FULL N_train grid for its assigned activations (`my_acts`). For
  each N it stages the subset and builds the geometry once, then trains every
  one of its activations on that shared geometry, putting one
  (N, activation, frac, gpu_id, seconds) tuple onto result_q per training.
  Each process has its own cosmolike global state and cached experiment, so
  the workers never interfere. A per-training failure emits frac = nan, so the
  parent receives exactly one result per (N, activation) and never deadlocks.

  Arguments:
    gpu_id    = CUDA device index this worker owns.
    my_acts   = the activation names assigned to this GPU (its share).
    sizes     = the full N_train grid (every worker runs all of it).
    cfg       = the parsed YAML config (data + train_args), with the host
                ram_frac already divided across the workers by the parent.
    rescale   = analytic-R mode, forwarded to the experiment.
    threshold = delta-chi2 cutoff for frac_above.
    result_q  = multiprocessing queue the parent drains.
  """
  # claim this GPU so every default-device op (and run_emulator's
  # torch.cuda.mem_get_info, which reads the current device when it sizes the
  # resident data set) targets this card, not card 0.
  torch.cuda.set_device(gpu_id)
  device = torch.device(f"cuda:{gpu_id}")

  # this worker's own experiment on its own GPU (quiet: the parent logs). The
  # activation is set per training below; my_acts[0] is just the initial one.
  exp = EmulatorExperiment.from_config(cfg,
                                       device=device,
                                       rescale=rescale,
                                       activation=my_acts[0],
                                       quiet=True)
  # the validation set is fixed across the bake-off; stage it once per worker.
  exp.stage_val()

  for N in sizes:
    # stage the subset and build the geometry once for this N; both are shared
    # across this worker's activations (they do not depend on the activation).
    exp.stage_train(n_train=int(N))
    exp.build_geometry()

    for act in my_acts:
      t0 = time.time()
      try:
        # build_specs reads exp.activation, so setting it then training picks
        # up this activation (the geometry is reused unchanged).
        exp.activation = act
        exp.train(silent=True)
        f = float(exp.frac_above(threshold=threshold))
      except Exception as err:                 # keep the bake-off alive
        f = float("nan")
        print(f"[gpu {gpu_id}] N_train {int(N)} act {act} failed: {err}")
      result_q.put((int(N), act, f, gpu_id, time.time() - t0))

      # free this activation's model and loaders, but keep the shared
      # geometry for the next activation. Returning the reserved VRAM lets the
      # next training size its loaders against the true free memory.
      exp.model = None
      torch.cuda.empty_cache()

    # done with this N: free the geometry and staged data before the next N.
    exp.train_set = None
    exp.geom      = None
    exp.pgeom     = None
    exp.chi2fn    = None
    torch.cuda.empty_cache()


def _run_serial_bakeoff(exp, sizes, activations, args, log):
  """
  Run the bake-off on a single device (no multiprocessing).

  The path taken on the dev machine (Apple MPS) and on a single-GPU box.
  Reuses the experiment the parent already built. N is the outer loop so the
  geometry is built once per N and shared across the activations.

  Arguments:
    exp         = the EmulatorExperiment, already built on the compute device.
    sizes       = the N_train grid (a sequence of ints).
    activations = the activation names to bake off.
    args        = the parsed CLI namespace (threshold is read).
    log         = the print function (a no-op under --quiet).

  Returns:
    curves = dict activation -> {N_train: frac}.
  """
  log(f"device: {exp.device}  |  serial (1 worker)")
  log("loading validation source:")
  exp.stage_val()

  curves = {act: {} for act in activations}
  for N in sizes:
    exp.stage_train(n_train=int(N))
    exp.build_geometry()
    for act in activations:
      t0 = time.time()
      exp.activation = act
      exp.train(silent=True)
      f = exp.frac_above(threshold=args.threshold)
      curves[act][int(N)] = f
      log(f"  N_train {int(N):8d}  {act:12s}  "
          f"f(>{args.threshold:g}) {f:.4f}  ({time.time() - t0:.0f}s)")
  return curves


def _run_parallel_bakeoff(cfg, sizes, activations, n_workers, args, log):
  """
  Run the bake-off across n_workers GPUs, one process each.

  Splits the activations evenly across the GPUs (a plain round-robin: each
  activation costs about the same, and every GPU runs the whole N_train grid,
  so the GPUs carry equal work with no cost-aware balancing). Spawns one
  process per GPU (the spawn start method, since a forked child cannot reuse
  the parent's CUDA context) and collects the per-(N, activation) fractions.
  The host-RAM staging budget (ram_frac) is divided by n_workers so the
  workers do not collectively overflow host memory.

  Arguments:
    cfg         = the parsed YAML config (data + train_args).
    sizes       = the N_train grid (a sequence of ints).
    activations = the activation names to bake off.
    n_workers   = number of GPU processes to launch.
    args        = the parsed CLI namespace (rescale / threshold are read).
    log         = the print function (a no-op under --quiet).

  Returns:
    curves = dict activation -> {N_train: frac}.
  """
  import torch.multiprocessing as mp

  # give each worker its share of the host-RAM staging budget. P workers on
  # one node draw from one RAM, so each may fill at most ram_frac / P of it.
  # Copy the data block first so the original cfg is left untouched.
  base_frac = float(cfg["data"].get("ram_frac", 0.7))
  worker_cfg = dict(cfg)
  worker_cfg["data"] = dict(cfg["data"])
  worker_cfg["data"]["ram_frac"] = base_frac / n_workers

  # split the activations round-robin across the GPUs (activations[g::P] gives
  # GPU g every P-th activation). Equal cost, so an even count per GPU is
  # already balanced; no LPT needed here.
  buckets = [activations[g::n_workers] for g in range(n_workers)]
  for k, b in enumerate(buckets):
    log(f"  gpu {k}: activations {b}")

  # spawn (not fork) because CUDA state cannot be inherited across a fork. The
  # worker is sent each GPU's activation bucket plus the full N_train grid.
  ctx = mp.get_context("spawn")
  result_q = ctx.Queue()
  sizes_i = [int(N) for N in sizes]
  procs = []
  for k in range(n_workers):
    p = ctx.Process(target=_bakeoff_worker,
                    args=(k, buckets[k], sizes_i, worker_cfg,
                          args.rescale, args.threshold, result_q))
    p.start()
    procs.append(p)

  # drain one result per (N, activation) as the workers finish; the parent
  # does all the logging (the workers run quiet).
  curves = {act: {} for act in activations}
  total = len(sizes) * len(activations)
  for _ in range(total):
    N, act, f, gpu, secs = result_q.get()
    curves[act][int(N)] = f
    log(f"  N_train {N:8d}  {act:12s}  f(>{args.threshold:g}) {f:.4f}  "
        f"(gpu {gpu}, {secs:.0f}s)")

  for p in procs:
    p.join()

  return curves


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
  parser.add_argument("--n-gpus",
                      dest="n_gpus",
                      help="number of GPUs to split the activations across "
                           "(default: all visible CUDA devices, capped at "
                           "the number of activations). 1, or no CUDA, is "
                           "serial.",
                      type=int,
                      default=None)
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

  # headless figure output: pick a non-interactive matplotlib backend BEFORE
  # emulator.plotting imports pyplot (done lazily below). Set before any
  # worker spawns, so the children inherit it too.
  os.environ.setdefault("MPLBACKEND", "Agg")

  # read the config once. The parent uses it for the grid and hands a copy
  # (with the host-RAM budget divided across workers) to each GPU process.
  with open(args.yaml) as f:
    cfg = yaml.safe_load(f)

  # build the experiment on the real compute device (CUDA, or Apple MPS on the
  # dev machine); the pool size and model name are read off it, and the serial
  # path reuses it. This bake-off trains a model per (N, activation), so it is
  # a GPU tool: refuse a pure-CPU box. (The starting activation is a
  # placeholder; each run sets its own.)
  exp = EmulatorExperiment.from_config(cfg,
                                       rescale=args.rescale,
                                       activation=activations[0],
                                       quiet=args.quiet)
  if exp.device.type == "cpu":
    raise RuntimeError(
      "no GPU found (need CUDA, or Apple MPS on the dev machine): this "
      "bake-off trains one model per (N, activation) and is not meant for CPU")
  log = exp.log
  model_name = exp.model_cls.__name__

  # N_train grid: geometric from n_min to the pool (or --n-max), clamped to
  # the physically-cut pool; unique() drops int-cast collisions.
  pool  = exp.pool_size()
  n_max = pool if args.n_max is None else min(args.n_max, pool)
  if args.n_min >= n_max:
    raise ValueError(
      f"--n-min {args.n_min} must be below n_max {n_max} (pool {pool})")
  sizes = np.unique(
    np.geomspace(args.n_min, n_max, args.n_points).astype(int))

  # how many GPUs to use: capped by what is visible, by --n-gpus, and by the
  # number of activations (the split is over activations, so more GPUs than
  # activations would leave the extras idle).
  n_cuda    = torch.cuda.device_count()
  n_request = n_cuda if args.n_gpus is None else min(args.n_gpus, n_cuda)
  n_workers = min(n_request, len(activations))

  log(f"model: {model_name}  |  rescale: {args.rescale}  "
      f"|  activations: {activations}")
  log(f"pool {pool}  |  N_train grid: {sizes.tolist()}")

  # 1 worker (single GPU, or the MPS dev machine) -> serial, reusing the
  # experiment already built; otherwise one process per GPU, split by
  # activation.
  if n_workers <= 1:
    curves = _run_serial_bakeoff(exp=exp, sizes=sizes,
                                 activations=activations, args=args, log=log)
  else:
    log(f"parallel bake-off across {n_workers} GPUs (split by activation):")
    curves = _run_parallel_bakeoff(cfg=cfg, sizes=sizes,
                                   activations=activations,
                                   n_workers=n_workers, args=args, log=log)

  # save every curve + the config as a plain-text table, one column per
  # activation (np.loadtxt-loadable; the # header lines are skipped).
  out_txt = args.out + ".txt"
  out_pdf = args.out + ".pdf"
  save_learning_curves(
    path=out_txt,
    sizes=sizes,
    curves={act: [curves[act][int(n)] for n in sizes]
            for act in activations},
    meta={"model": model_name, "rescale": args.rescale,
          "activation": "swept", "threshold": args.threshold,
          "pool": pool, "n_gpus": n_workers})
  log(f"saved curve data -> {out_txt}")

  # overlaid figure: one curve per activation.
  from emulator.plotting import plot_learning_curves
  plot_learning_curves(curves=curves,
                       threshold=args.threshold,
                       savepath=out_pdf)
  log(f"saved figure -> {out_pdf}")


if __name__ == "__main__":
  main()
