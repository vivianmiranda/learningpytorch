"""Grouped (per-bin) nn building blocks."""

import torch
import torch.nn as nn

from ..activations import activation_fcn
from .activations import GroupedActivation


class GroupedLinear(nn.Module):
  """
  G independent Linear(in, out) layers run as ONE batched
  matmul over the group axis G. Weights stacked into a single
  (G, in, out) tensor, bias into (G, out).
  """
  def __init__(self, n_groups, in_features, out_features):
    super().__init__()
    # build G ordinary nn.Linear layers just to borrow their
    # init, then stack their weights/biases and discard them.
    # l.weight is (out, in); .t() -> (in, out) for the einsum;
    # stack over G adds the group axis.
    lins = [nn.Linear(in_features=in_features, out_features=out_features)
            for _ in range(n_groups)]
    self.weight = nn.Parameter(
      torch.stack([l.weight.detach().t() for l in lins]))
    self.bias = nn.Parameter(
      torch.stack([l.bias.detach() for l in lins]))

  def forward(self, x):
    # x: (G, B, in) -- G heads, each with the same B input rows.
    # self.weight: (G, in, out) -- G stacked weight matrices.
    #
    # einsum("gbi,gio->gbo", x, weight):
    #   g = group/head axis. It is in BOTH inputs and the output,
    #       so it is a BATCH axis: head g uses only weight[g],
    #       and all G heads run in parallel (one batched matmul).
    #   b = the B samples. Only in x and the output -> kept.
    #   i = input features. In BOTH inputs but NOT the output,
    #       so einsum SUMS over it -> this is the matmul's inner
    #       (contracted) dimension.
    #   o = output features. Only in weight and the output -> kept.
    #
    # Element-by-element it computes:
    #   y[g, b, o] = sum_i  x[g, b, i] * weight[g, i, o]
    # i.e. the same as this triple loop, but on the GPU at once:
    #   for g in range(G):
    #     for b in range(B):
    #       for o in range(out):
    #         y[g,b,o] = sum(x[g,b,i]*weight[g,i,o]
    #                        for i in range(in))

    # einsum form:  "gbi,gio->gbo"

    # y[g] = x[g] @ W[g]   for each group g, all G done at once (in parallel)
    # x : (G, B, in) -> label "gbi" (per group g: B input rows; no o)
    # W : (G, in, out) -> label "gio" (per group g: its in->out converter; brings in o)
    # y : (G, B, out) -> label "gbo" (per group g: gains o from W[g])
    y = torch.einsum("gbi,gio->gbo", x, self.weight)   # (G,B,out)

    # self.bias: (G, out). unsqueeze(1) -> (G, 1, out) so it
    # broadcasts across the B axis (every sample in head g gets
    # head g's bias added).
    return y + self.bias.unsqueeze(1)


class GroupedAffine(nn.Module):
  """G independent Affine (per-group learnable scale + shift)."""
  def __init__(self, n_groups):
    super().__init__()
    # gain inits to 1, bias to 0 (same as the scalar Affine),
    # with an extra group axis and trailing 1s so they broadcast
    # over (G, B, D).
    self.gain = nn.Parameter(torch.ones(n_groups, 1, 1))
    self.bias = nn.Parameter(torch.zeros(n_groups, 1, 1))

  def forward(self, x):                 # x: (G, B, D)
    return x * self.gain + self.bias


class GroupedResBlock(nn.Module):
  """
  The grouped twin of ResBlock: n_layers grouped Linear(D,D),
  each followed by a grouped Affine "norm" and grouped
  activation, with the skip added before the LAST norm/act
  (pre-activation residual). Identical logic to ResBlock, just
  with the group axis threaded through.
  """
  def __init__(self, n_groups, size, n_layers=2):
    super().__init__()
    self.lins  = nn.ModuleList([
      GroupedLinear(n_groups, size, size)
      for _ in range(n_layers)])
    self.norms = nn.ModuleList([
      GroupedAffine(n_groups) for _ in range(n_layers)])
    self.acts  = nn.ModuleList([
      GroupedActivation(n_groups, size)
      for _ in range(n_layers)])

  def forward(self, x):                 # x: (G, B, D)
    xskip = x
    out = x
    n = len(self.lins)
    for i in range(n):
      out = self.lins[i](out)
      if i == n - 1:                    # add skip before last
        out = out + xskip               #   norm + activation
      out = self.acts[i](self.norms[i](out))
    return out


class GroupedCNNBlock(nn.Module):
  """
  Per-group 1D convolution: split the input into n_groups
  contiguous segments of length group_len and convolve EACH one
  independently (a grouped Conv1d, groups=n_groups). No kernel ever
  crosses a group boundary -- so with a per-tomographic-bin layout,
  each bin's theta curve is refined without smoothing across the
  bin-boundary jumps a single global conv would blur.

  Two convs with a nonlinearity between (so the per-group `channels`
  filters are useful, like CNNBlock). Input/output: (B, n_groups *
  group_len).

  Arguments:
    n_groups    = independent segments (= number of bins).
    group_len   = length of each segment (the PADDED per-bin length
                  = max bin size).
    kernel_size = kernel width (odd; same-padding keeps group_len).
    channels    = conv filters PER group.
    act         = activation factory.
  """
  def __init__(self, n_groups, group_len, kernel_size=11,
               channels=16, act=activation_fcn):
    super().__init__()
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    pad = (kernel_size - 1) // 2
    self.n_groups  = n_groups
    self.group_len = group_len
    # 1 input channel per group -> `channels` filters per group;
    # groups=n_groups keeps every bin's convolution independent.
    self.conv_in  = nn.Conv1d(in_channels=n_groups,
                              out_channels=n_groups * channels,
                              kernel_size=kernel_size,
                              padding=pad,
                              groups=n_groups)
    self.act_mid  = act(group_len)        # within-bin position act
    self.conv_out = nn.Conv1d(in_channels=n_groups * channels,
                              out_channels=n_groups,
                              kernel_size=kernel_size,
                              padding=pad,
                              groups=n_groups)
    self.act_out  = act(n_groups * group_len)

  def forward(self, x):
    # (B, n_groups*group_len) -> (B, n_groups, group_len): each
    # group becomes one channel the grouped conv treats alone.
    h = x.view(x.size(0), self.n_groups, self.group_len)
    h = self.conv_in(h)         # (B, n_groups*channels, group_len)
    h = self.act_mid(h)         # nonlinearity (channels matter)
    h = self.conv_out(h)        # (B, n_groups, group_len)
    h = h.view(x.size(0), -1)   # (B, n_groups*group_len)
    return self.act_out(h)
