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
#  --out -- then overlay the saved <out>.txt curves.
#
#- For each N_train in a geometric grid [--n-min .. --n-max] (--n-max defaults
#  to the full physically-cut training pool), it stages a NESTED training subset
#  of that size, rebuilds the geometry from it, trains a FRESH model (silently),
#  and scores f(delta-chi2 > --threshold) on the FIXED validation set.
#
#- MULTIPLE GPUs (one node): the grid points are independent trainings, so they
#  run in parallel, one process per GPU. The points are split across GPUs by the
#  Longest-Processing-Time rule (largest N_train first to the least-loaded GPU),
#  so each GPU gets about the same total N_train and they finish together. With
#  --n-gpus 8 on an 8-GPU node:
#
#     python driver/sweep_ntrain_emulator_cosmic_shear.py \
#       --yaml driver/train_single_emulator_cosmic_shear.yaml \
#       --n-points 12 --n-gpus 8 --out ntrain_resmlp
#
#  With one GPU (or none, e.g. the Apple-MPS dev machine) it falls back to a
#  plain serial loop, so the same script runs everywhere.
#
#- `--yaml` (required): the config (data + train_args), same schema as the
#  training driver; train_args.model.name picks ResMLP / ResCNN.
#- `--rescale` / `--activation`: as in the training driver, fixed across the
#  sweep (the analytic-R mode and the ResBlock activation).
#- `--n-gpus` (default: all visible CUDA devices): how many GPUs to spread the
#  sweep across. 1, or no CUDA, takes the serial path.
#- `--n-min` (default 2000), `--n-max` (default = pool), `--n-points` (default
#  5): the geometric N_train grid (clamped to the pool, deduplicated).
#- `--threshold` (default 0.2): the delta-chi2 cutoff the fraction counts.
#- `--out` (default ntrain_sweep): writes <out>.txt (the curve + the config it
#  came from, np.loadtxt-loadable) and <out>.pdf (a single-curve figure).
#- `--quiet`: suppress stdout (the .txt and .pdf are still written).
#
#- This trains ONE full model per grid point, so a sweep is --n-points
#  trainings long (divided across the GPUs) -- run it on the workstation, where
#  cosmolike lives.
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
from emulator.scheduling import lpt_assign


def _sweep_worker(gpu_id, my_sizes, cfg, rescale, activation,
                  threshold, result_q):
  """
  One GPU's share of the N_train sweep; runs in its own process.

  Pins itself to GPU `gpu_id`, builds its own EmulatorExperiment there, and
  trains every N_train in `my_sizes` (its LPT bucket), putting one
  (N, frac, gpu_id, seconds) tuple onto result_q as each finishes. Each
  process has its own cosmolike global state and its own cached experiment,
  so the workers never interfere. On a per-point failure it emits frac = nan,
  so the parent still receives exactly one result per N and never deadlocks
  waiting on a missing one.

  Arguments:
    gpu_id     = CUDA device index this worker owns.
    my_sizes   = the N_train values assigned to this GPU (its LPT bucket).
    cfg        = the parsed YAML config (data + train_args), with the host
                 ram_frac already divided across the workers by the parent.
    rescale    = analytic-R mode, forwarded to the experiment.
    activation = ResBlock activation name, forwarded to the experiment.
    threshold  = delta-chi2 cutoff for frac_above.
    result_q   = multiprocessing queue the parent drains.
  """
  # claim this GPU so every default-device op (and run_emulator's
  # torch.cuda.mem_get_info, which reads the current device when it sizes the
  # resident data set) targets this card, not card 0.
  torch.cuda.set_device(gpu_id)
  device = torch.device(f"cuda:{gpu_id}")

  # this worker's own experiment on its own GPU (quiet: the parent logs).
  exp = EmulatorExperiment.from_config(cfg,
                                       device=device,
                                       rescale=rescale,
                                       activation=activation,
                                       quiet=True)
  # the validation set is fixed across the sweep; stage it once per worker.
  exp.stage_val()

  for N in my_sizes:
    t0 = time.time()
    try:
      # nested training subset of size N, geometry rebuilt from its means, a
      # fresh model trained quietly, then scored on the fixed val set.
      exp.stage_train(n_train=int(N))
      exp.build_geometry()
      exp.train(silent=True)
      f = float(exp.frac_above(threshold=threshold))
    except Exception as err:                  # keep the sweep alive
      f = float("nan")
      print(f"[gpu {gpu_id}] N_train {int(N)} failed: {err}")
    result_q.put((int(N), f, gpu_id, time.time() - t0))

    # drop this point's GPU tensors and hand the reserved VRAM back, so the
    # next N sizes its loaders against the true free memory. The caching
    # allocator would otherwise keep this run's tensors reserved, and
    # run_emulator's mem_get_info would under-report the free VRAM.
    exp.model     = None
    exp.train_set = None
    exp.geom      = None
    exp.pgeom     = None
    exp.chi2fn    = None
    torch.cuda.empty_cache()


def _run_serial(exp, sizes, args, log):
  """
  Run the sweep on a single device (no multiprocessing).

  The path taken on the dev machine (Apple MPS) and on a single-GPU box.
  Reuses the experiment the parent already built on the compute device, and
  trains each N_train in turn.

  Arguments:
    exp   = the EmulatorExperiment, already built on the compute device.
    sizes = the N_train grid (a sequence of ints).
    args  = the parsed CLI namespace (threshold is read).
    log   = the print function (a no-op under --quiet).

  Returns:
    fracs = list of f(delta-chi2 > threshold), aligned with `sizes`.
  """
  log(f"device: {exp.device}  |  serial (1 worker)")
  log("loading validation source:")
  exp.stage_val()

  fracs = []
  for N in sizes:
    t0 = time.time()
    exp.stage_train(n_train=int(N))
    exp.build_geometry()
    exp.train(silent=True)
    f = exp.frac_above(threshold=args.threshold)
    fracs.append(f)
    log(f"  N_train {int(N):8d}  f(>{args.threshold:g}) {f:.4f}  "
        f"({time.time() - t0:.0f}s)")
  return fracs


def _run_parallel(cfg, sizes, n_workers, args, log):
  """
  Run the sweep across n_workers GPUs, one process each, LPT-balanced.

  Splits `sizes` with lpt_assign so each GPU gets about the same total
  N_train, spawns one process per GPU (the spawn start method, since a
  forked child cannot reuse the parent's CUDA context), and collects the
  per-point fractions. The host-RAM staging budget (ram_frac) is divided by
  n_workers so the workers do not collectively overflow host memory.

  Arguments:
    cfg       = the parsed YAML config (data + train_args).
    sizes     = the N_train grid (a sequence of ints).
    n_workers = number of GPU processes to launch.
    args      = the parsed CLI namespace (rescale / activation / threshold).
    log       = the print function (a no-op under --quiet).

  Returns:
    fracs = list of f(delta-chi2 > threshold), aligned with `sizes`.
  """
  import torch.multiprocessing as mp

  # give each worker its share of the host-RAM staging budget. P workers on
  # one node draw from one RAM, so each may fill at most ram_frac / P of it;
  # that bound holds even when they all stage at the same time. Copy the
  # data block first so the original cfg is left untouched.
  base_frac = float(cfg["data"].get("ram_frac", 0.7))
  worker_cfg = dict(cfg)
  worker_cfg["data"] = dict(cfg["data"])
  worker_cfg["data"]["ram_frac"] = base_frac / n_workers

  # LPT split: largest N first, each to the least-loaded GPU.
  buckets = lpt_assign(sizes, n_workers)
  for k, b in enumerate(buckets):
    log(f"  gpu {k}: {len(b)} points, total N {sum(b)}  ->  {sorted(b)}")

  # Launch one child process per GPU. multiprocessing can create children in
  # a few ways ("start methods"); get_context("spawn") hands back an object
  # that uses the "spawn" method. spawn starts a brand-new Python interpreter
  # for each child, which re-imports this module to find _sweep_worker. The
  # alternative, "fork", instead clones the parent process's memory and is the
  # default on Linux -- but a forked child inherits the parent's already
  # initialized CUDA state, and CUDA refuses to be used through an inherited
  # context (it hangs or errors). spawn gives each child a fresh interpreter
  # and a fresh CUDA context, so every worker sets up its own GPU cleanly.
  # (On macOS spawn is already the default; on Linux we must ask for it.)
  ctx = mp.get_context("spawn")

  # A process-safe queue, made from the same spawn context so the children can
  # reconstruct it. It is the one-way channel the workers send results back
  # on: each worker calls result_q.put((N, frac, gpu, secs)) and the parent
  # loop below calls result_q.get(). "process-safe" means several processes
  # can put and get at the same time without corrupting it (internally it is a
  # pipe guarded by locks).
  result_q = ctx.Queue()

  # Keep a handle to every child in `procs`, so we can wait on them with
  # join() further down (and so Python does not garbage-collect them while
  # they run).
  procs = []
  for k in range(n_workers):
    # Build (not yet start) one child process. `target` is the function the
    # child will run; `args` is the tuple of positional arguments handed to
    # it. Under spawn, target and args are pickled and shipped to the child,
    # so each must be picklable: _sweep_worker is a module-level function (so
    # the child can import it by name), and buckets[k] / worker_cfg / the
    # strings / the float / the queue are all plain, picklable data.
    #
    # args fills _sweep_worker's parameters in order:
    #   k               -> gpu_id      (this child's CUDA device index)
    #   buckets[k]      -> my_sizes    (the N_train values assigned to GPU k)
    #   worker_cfg      -> cfg         (config, ram_frac already divided)
    #   args.rescale    -> rescale
    #   args.activation -> activation
    #   args.threshold  -> threshold
    #   result_q        -> result_q    (the channel back to the parent)
    p = ctx.Process(target=_sweep_worker,
                    args=(k, buckets[k], worker_cfg, args.rescale,
                          args.activation, args.threshold, result_q))
    # start() launches the operating-system process and begins running
    # _sweep_worker(*args) inside it. It returns immediately -- the training
    # runs in the background -- so the loop moves straight on to launch the
    # next GPU's process; the parent gathers the results afterward.
    p.start()
    procs.append(p)

  # drain one result per point as the workers finish; the parent does all
  # the logging (the workers run quiet, so 8 streams do not interleave).
  results = {}
  for _ in range(len(sizes)):
    N, f, gpu, secs = result_q.get()
    results[N] = f
    log(f"  N_train {N:8d}  f(>{args.threshold:g}) {f:.4f}  "
        f"(gpu {gpu}, {secs:.0f}s)")

  for p in procs:
    p.join()

  # the results arrived out of order; re-align them to `sizes`.
  return [results[int(N)] for N in sizes]


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
  parser.add_argument("--n-gpus",
                      dest="n_gpus",
                      help="number of GPUs to spread the sweep across "
                           "(default: all visible CUDA devices). 1, or no "
                           "CUDA, takes the serial path.",
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
                           "(default ntrain_sweep)",
                      type=str,
                      default="ntrain_sweep")
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress all stdout (txt / pdf still written)",
                      action="store_true")
  args, unknown = parser.parse_known_args()

  # headless figure output: pick a non-interactive matplotlib backend BEFORE
  # emulator.plotting imports pyplot (done lazily below). Set before any
  # worker spawns, so the children inherit it too.
  os.environ.setdefault("MPLBACKEND", "Agg")

  # read the config once. The parent uses it for the grid and hands a copy
  # (with the host-RAM budget divided across workers) to each GPU process.
  with open(args.yaml) as f:
    cfg = yaml.safe_load(f)

  # build the experiment on the real compute device (CUDA, or Apple MPS on
  # the dev machine); the pool size and model name are read off it, and the
  # serial path reuses it. With the spawn start method the parent may hold a
  # GPU context and still launch the per-GPU workers safely. This sweep
  # trains a model per grid point, so it is a GPU tool: refuse a pure-CPU box.
  exp = EmulatorExperiment.from_config(cfg,
                                       rescale=args.rescale,
                                       activation=args.activation,
                                       quiet=args.quiet)
  if exp.device.type == "cpu":
    raise RuntimeError(
      "no GPU found (need CUDA, or Apple MPS on the dev machine): this "
      "sweep trains one model per grid point and is not meant for CPU")
  log = exp.log
  model_name = exp.model_cls.__name__

  # N_train grid: geometric from n_min to the pool (or --n-max), clamped to
  # the physically-cut pool so every size is loadable; unique() drops the
  # collisions the int cast makes at the low end.
  pool  = exp.pool_size()
  n_max = pool if args.n_max is None else min(args.n_max, pool)
  if args.n_min >= n_max:
    raise ValueError(
      f"--n-min {args.n_min} must be below n_max {n_max} (pool {pool})")
  sizes = np.unique(
    np.geomspace(args.n_min, n_max, args.n_points).astype(int))

  # how many GPUs to actually use: capped by what is visible, by --n-gpus,
  # and by the number of points (no idle workers).
  n_cuda    = torch.cuda.device_count()
  n_request = n_cuda if args.n_gpus is None else min(args.n_gpus, n_cuda)
  n_workers = min(n_request, len(sizes))

  log(f"model: {model_name}  |  rescale: {args.rescale}  "
      f"|  activation: {args.activation}")
  log(f"pool {pool}  |  N_train grid: {sizes.tolist()}")

  # 1 worker (single GPU, or the MPS dev machine) -> serial, reusing the
  # experiment already built; otherwise one process per GPU, LPT-balanced.
  if n_workers <= 1:
    fracs = _run_serial(exp=exp, sizes=sizes, args=args, log=log)
  else:
    log(f"parallel sweep across {n_workers} GPUs (LPT-balanced):")
    fracs = _run_parallel(cfg=cfg, sizes=sizes, n_workers=n_workers,
                          args=args, log=log)

  # save the curve + the config it came from as a plain-text table, so
  # several runs (one per architecture / chi2 mode) can be overlaid later
  # (np.loadtxt-loadable; the # header lines are skipped).
  out_txt = args.out + ".txt"
  out_pdf = args.out + ".pdf"
  save_learning_curves(
    path=out_txt,
    sizes=sizes,
    curves={"frac": fracs},
    meta={"model": model_name, "rescale": args.rescale,
          "activation": args.activation, "threshold": args.threshold,
          "pool": pool, "n_gpus": n_workers})
  log(f"saved curve data -> {out_txt}")

  # one-curve figure (overlay several <out>.txt yourself to compare).
  from emulator.plotting import plot_learning_curves
  plot_learning_curves(
    curves={f"{model_name} ({args.rescale})": (sizes, fracs)},
    threshold=args.threshold,
    savepath=out_pdf)
  log(f"saved figure -> {out_pdf}")


if __name__ == "__main__":
  main()
