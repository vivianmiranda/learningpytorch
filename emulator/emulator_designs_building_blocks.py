"""Shared nn building blocks (Affine, ResBlock, CNNBlock).

These are the small nn.Modules the emulator models are assembled from.
Affine is a learnable per-output scale and shift (the default ResBlock
"norm" and the models' final layer). ResBlock is a width-preserving
residual block (n dense layers, each with a norm and an activation
factory, and a skip added before the last). CNNBlock is a 1D-convolution
correction head that treats the data vector as a signal and slides a
learned kernel along it (the axis-aware piece of ResCNN). The grouped /
per-bin twins live in parallel/.
"""

import torch
import torch.nn as nn

from .activations import activation_fcn


class Affine(nn.Module):
    """
    A learnable scalar scale and shift: out = x * gain + bias.

    gain and bias are single scalars (shape (1,)), broadcast over
    every element of x, so Affine applies one global scale and one
    global shift to the whole tensor, not a per-feature transform.
    gain initializes to 1 and bias to 0, so at init it is the
    identity (a no-op that learns a scale / offset from there).
    Used as the ResBlock default "norm" factory and as the final
    layer of ResMLP / ResCNN.

    gain and bias are nn.Parameter, so they are registered with the
    module and trained by the optimizer. Weight decay is kept off
    both (make_optimizer decays only ndim >= 2 weight matrices):
    gain inits to 1, and decaying it toward 0 would attenuate the
    signal, while decaying a bias has no principled meaning.
    """
    def __init__(self):
        super(Affine, self).__init__()
        # one learnable scale (gain, init 1) and one learnable
        # shift (bias, init 0), each a single scalar broadcast
        # over all elements of the input.
        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        # elementwise: every entry of x is scaled by gain and
        # shifted by bias (both broadcast from their size-1 shape).
        return x * self.gain + self.bias


class ResBlock(nn.Module):
  # Residual block. Input and output share the same width
  # by design, so the skip connection is the identity.
  #
  # Arguments:
  #   size = feature width, shared by input and output
  #   n_layers = number of dense layers between two
  #              skip points
  #   norm = normalization factory, invoked as norm(size)
  #   act = activation factory, invoked as act(size)
  #
  # norm and act are factories rather than ready-made
  # modules: each is invoked once per dense layer so that
  # every layer holds an independent module. A single
  # shared instance would couple the layers' learnable
  # normalization parameters.
  #
  # Factory examples:
  #   norm = nn.BatchNorm1d       (accepts size)
  #   norm = lambda s: Affine()   (Affine accepts no size)
  #   act = activation_fcn        (accepts size)
  #   act = lambda s: nn.Tanh()   (Tanh accepts no size)
  def __init__(self, 
               size, 
               n_layers = 2,
               norm = lambda s: Affine(),
               act = activation_fcn):
    super().__init__()
    self.skip = nn.Identity()

    # Sublayers are stored in nn.ModuleList rather than a
    # plain list or numbered attributes. ModuleList
    # registers each submodule with the parent, so its
    # parameters appear in .parameters(), transfer under
    # .to(device), and are saved in the state_dict.
    self.layers = nn.ModuleList(
      [nn.Linear(in_features=size, out_features=size) for _ in range(n_layers)])
    self.norms = nn.ModuleList(
      [norm(size) for _ in range(n_layers)])
    self.acts = nn.ModuleList(
      [act(size) for _ in range(n_layers)])

  def forward(self, x):
    xskip = self.skip(x)
    out = x
    n = len(self.layers)
    for i in range(n):
      out = self.layers[i](out)
      # The skip connection is added to the output of the
      # final linear layer, before its normalization and
      # activation (a pre-activation residual addition).
      if i == n - 1:
        out = out + xskip
      out = self.acts[i](self.norms[i](out))
    return out


class CNNBlock(nn.Module):
  """
  1D-convolution correction head: treat a length-`dim` vector as a
  single-channel signal and slide a learned kernel along it, so the
  model can fix STRUCTURE along the sequence (neighbouring entries)
  that a dense layer would treat independently. Odd kernel + same
  padding preserves the length, so the output is again (B, dim).

  With channels > 1 the block expands to `channels`
  filters, applies a nonlinearity, then mixes them back to
  one channel with a 1x1 conv. The mid nonlinearity is
  essential: without it the expand and the 1x1 collapse are
  two stacked linear convs that compose into a single
  kernel, so the extra filters add nothing.

  Arguments:
    dim         = length of the input/output sequence (= cnn_dim).
    kernel_size = kernel width; must be ODD so the same-padding
                  (kernel_size-1)//2 keeps the length unchanged.
    channels    = number of convolution filters. 1 = a single
                  learned kernel (collapse + mid-act are then
                  Identity). >1 = expand to `channels` filters,
                  apply a nonlinearity, then a 1x1 conv mixes
                  them back to one channel. The mid-activation
                  is essential, or the two convs collapse into
                  one kernel and the extra filters do nothing.
    act         = activation factory, invoked as act(dim) to build
                  each nonlinearity (the mid-activation and the
                  output one). Defaults to activation_fcn, the same
                  learnable H that ResBlock / ResMLP use, so by
                  default the CNN head and the trunk share one
                  activation family.
  """
  def __init__(self, dim, kernel_size=11, channels=1,
               act=activation_fcn):
    super().__init__()
    assert kernel_size % 2 == 1, (
      "kernel_size must be odd so same-padding keeps the length")
    pad = (kernel_size - 1) // 2

    # 1 input channel (the signal) -> `channels` filters; length
    # preserved by the same-padding.
    self.conv = nn.Conv1d(in_channels=1,
                          out_channels=channels,
                          kernel_size=kernel_size,
                          padding=pad)

    # nonlinearity between the expand and the collapse. It is
    # act(dim): the same activation_fcn (the paper's H) the
    # ResBlocks use by default, a learnable activation class, not a
    # bare ReLU. It must be here because without a nonlinearity the
    # conv (1->channels) and collapse (channels->1) are two stacked
    # linear convs that fold into a single 1->1 kernel, so the
    # extra filters would be wasted. Identity when channels == 1
    # (no expand to make nonlinear).
    self.act_mid = act(dim) if channels > 1 else nn.Identity()

    # mix the filters back to one channel (a 1x1 conv is a
    # per-position weighted sum over channels); Identity when
    # channels == 1 so the forward stays uniform.
    self.collapse = (nn.Conv1d(in_channels=channels, out_channels=1, kernel_size=1)
                     if channels > 1 else nn.Identity())
    self.act = act(dim)

  def forward(self, x):
    # treat the length-dim vector as a 1-CHANNEL signal: view to
    # (B, 1, dim) inserts the channel axis Conv1d expects (its
    # input layout is (batch, channels, length)). view reshapes
    # without copying; -1 lets it infer the length axis.
    h = x.view(x.size(0), 1, -1)
    h = self.conv(h)              # (B, channels, dim)
    # nonlinearity between the expand and the collapse, so the
    # `channels` filters cannot fold back into one kernel. act_mid
    # is Identity when channels == 1 (conv -> act is already a
    # full single-filter head, no redundant activation).
    h = self.act_mid(h)           # (B, channels, dim)
    h = self.collapse(h)          # (B, 1, dim)
    # flatten the (now size-1) channel axis back out: (B, 1,
    # dim) -> (B, dim), the shape the rest of the model expects.
    h = h.view(x.size(0), -1)     # (B, dim)
    return self.act(h)
