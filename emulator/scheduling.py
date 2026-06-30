"""Work-balancing helpers for spreading a sweep across GPUs.

When a parameter sweep runs many independent jobs on several GPUs (here, one
training run per N_train value), the jobs have to be split so that every GPU
finishes at about the same time. This module holds the pure, framework-free
part of that: deciding which job goes to which GPU. The actual process
spawning lives in the driver; this only computes the split. The one function
so far, lpt_assign, partitions the jobs by the Longest-Processing-Time rule
so each GPU receives roughly the same total cost.
"""


def lpt_assign(sizes, n_workers):
  """
  Balance the sweep points across GPUs by total N_train.

  Longest-Processing-Time rule: hand the points out largest-N first, each to
  the GPU that has the least work so far. The cost of one point is about
  proportional to its N_train (the rest of the run, nepochs and bs, is fixed
  across points), so keeping the per-GPU sums of N even keeps the wall-clock
  even. Handing out the big points first is what makes those sums come out
  balanced; a naive round-robin would pile every grid triple's largest point
  onto the same GPU.

  Arguments:
    sizes     = the N_train values of the sweep (any order; cast to int).
    n_workers = number of GPUs to split across (>= 1).

  Returns:
    buckets = a list of length n_workers; buckets[k] is the list of N_train
              values assigned to GPU k, in the largest-first order they were
              handed out.
  """
  # buckets[k] = the N values assigned to GPU k (filled in below).
  buckets = [[] for _ in range(n_workers)]
  # loads[k] = the running sum of N already given to GPU k (its "load").
  loads = [0 for _ in range(n_workers)]

  # the points, largest N first. int() because the grid comes from numpy.
  points = sorted([int(N) for N in sizes], reverse=True)

  for N in points:
    # find the least-loaded GPU with a plain scan: start by assuming GPU 0
    # is the lightest, then walk the rest and keep the index of any GPU
    # that is carrying less work.
    k = 0
    for g in range(1, n_workers):
      if loads[g] < loads[k]:
        k = g
    # give this point to GPU k, and add its cost (N) to that GPU's load.
    buckets[k].append(N)
    loads[k] += N

  return buckets
