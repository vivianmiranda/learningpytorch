"""Standard emulator models (ResMLP, ResCNN).

These are the full networks that map whitened parameters to the whitened
data vector. ResMLP is the baseline: an input projection, a stack of
identical ResBlocks, an output projection, and a final Affine. ResCNN
adds a 1D-CNN correction appendix on top of a ResMLP trunk: the trunk
predicts in the full (cov-eigenbasis) whitening, fixed buffers map its
output into theta order so a conv can correct structure along the angular
axis, and a learnable gate adds the correction back. So swapping
ResMLP -> ResCNN changes only the model, not the whitening. The per-bin
parallel variants live in parallel/.

PS: whitened = rotated into the covariance eigenbasis and scaled to unit
variance, so the components are decorrelated and equally hard to fit; the
geometry classes (geometries_parameter / geometries_output) do it.
"""

import torch
import torch.nn as nn

from .activations import activation_fcn
from .emulator_designs_building_blocks import (
  Affine, ResBlock, CNNBlock)


class ResMLP(nn.Module):
  """
  Full emulator: an input projection, a stack of
  identical residual blocks, an output projection, and
  a final learnable affine.
  
  Arguments:
    input_dim   = number of cosmological parameters
    output_dim  = length of the data vector
    int_dim_res = internal (residual) width
    n_blocks    = number of residual blocks
    block_opts  = dict of ResBlock options (n_layers,
                   norm, act), the same for every block
  
  block_opts=None, not block_opts={}: a default is
  created once and reused on every call, so a mutable
  default (a dict) would leak between calls.
    
  All blocks share one configuration on purpose, to
  keep the hyperparameter count from exploding.
  """
  def __init__(self, 
               input_dim, 
               output_dim, 
               int_dim_res,
               n_blocks=3, 
               block_opts=None):
    super().__init__()
    
    # Default to an empty dict. We do not write
    # block_opts={} in the signature: a default argument
    # is created once and shared across all calls, so a
    # mutable default (a dict) would leak between calls.
    if block_opts is None:
      block_opts = {}
    layers = []
    
    # cosmological-parameter dim -> internal width
    layers.append(nn.Linear(in_features=input_dim, out_features=int_dim_res))
    
    # n_blocks identical residual blocks at the internal
    # width. **block_opts unpacks the dict into keyword
    # arguments for each ResBlock.
    for _ in range(n_blocks):
      layers.append(ResBlock(int_dim_res, **block_opts))
    
    # internal width -> data-vector dim
    layers.append(nn.Linear(in_features=int_dim_res, out_features=output_dim))
    
    # final learnable scale and shift on the output
    layers.append(Affine())
    
    # Sequential registers every module in the list, so
    # the temporary plain list is fine here.
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)


class ResCNN(nn.Module):
  """
  ResMLP trunk + a 1D-CNN correction APPENDIX. The ResMLP path
  is identical to the standalone ResMLP and predicts in the FULL
  (cov-eigenbasis) whitened basis, so its loss stays the
  well-conditioned chi2 = ||pred - target||^2 (identity Hessian).

  The CNN is an additive correction in the DIAGONAL view of the
  geometry (theta order, per-element /sigma). A 1D conv slides one
  shared kernel along the length axis to exploit adjacency: it
  learns how each entry relates to its neighbours, which only
  helps if neighbouring entries are neighbouring theta. The
  full-whitened basis rotates into the covariance eigenbasis, so
  each component is a mix of all thetas and the angular order is
  scrambled; a conv there would slide over arbitrary eigenmodes
  with no locality to exploit. The diagonal view keeps theta order
  (a per-element /sigma scaling, no rotation), so the conv sees
  the real angular axis and can correct theta-local structure (the
  smooth shape and oscillations of xi across neighbouring angular
  bins, which the trunk leaves as correlated residuals along
  theta). forward maps the ResMLP output into that view, the conv
  corrects it there, then forward maps the correction back to the
  full basis and adds it through a learnable gate. So the bulk map
  keeps the eigenbasis conditioning (the ResMLP is permutation
  invariant, it does not need theta order) and only the small
  correction pays the theta-order conditioning cost.

  The two basis-change maps are precomputed from the geometry
  and stored as fixed BUFFERS. Their names abbreviate the two
  bases they convert between: f = the full-whitened basis (the
  covariance eigenbasis the trunk predicts in), d = the diagonal
  basis (theta order, each element scaled by its own marginal
  sigma, the DiagonalGeometry view a conv needs). The subscripts
  read in multiply order, so y_full @ W_fd goes f -> d, and the
  correction @ W_df goes d -> f (W_df is W_fd's inverse). Storing
  them as buffers, not live geometry calls in forward, is what
  makes this safe under torch.compile reduce-overhead / CUDA
  graphs: a tensor lazily built inside forward gets captured in
  the graph's static pool and overwritten on the next run.

  Target and loss use the FULL-whitening DataVectorGeometry,
  exactly as the standalone ResMLP, so swapping ResMLP -> ResCNN
  changes the MODEL only, not the whitening (no confound).

  Arguments:
    input_dim    = number of cosmological parameters.
    output_dim   = data-vector length to emulate (= n_keep).
    int_dim_res  = internal width of the residual trunk.
    geom         = full-whitening DataVectorGeometry; its
                   evecs / sqrt_ev (and derived sigma) define
                   the two basis-change buffers.
    kernel_size  = CNN kernel width (odd); forwarded to CNNBlock.
    channels     = CNN filter count; forwarded to CNNBlock.
    n_blocks     = residual blocks in the trunk.
    n_blocks_cnn = stacked CNN correction blocks (default 1).
    gate_init    = initial value of the learnable scalar scaling
                   the correction. Small (default 0.1) so the
                   model starts close to the pure ResMLP; not 0
                   -- a 0 gate strands the CNN with no gradient,
                   so it would never learn.
    block_opts   = ResBlock options (None -> {}); its "act" (the
                   chosen activation) is also handed to the CNN
                   head, so the head and trunk share one activation
                   family. Defaults to activation_fcn (the paper's
                   H) when block_opts sets no "act".
  """
  def __init__(self, input_dim, output_dim, int_dim_res, geom,
               kernel_size=11, channels=16, n_blocks=3,
               n_blocks_cnn=1, gate_init=0.1, block_opts=None):
    super().__init__()
    if block_opts is None:
      block_opts = {}

    # ResMLP main path: the standalone ResMLP layer stack, output
    # in the FULL-whitened basis (well conditioned).
    mlp = [nn.Linear(in_features=input_dim, out_features=int_dim_res)]
    for _ in range(n_blocks):
      mlp.append(ResBlock(int_dim_res, **block_opts))
    mlp.append(nn.Linear(in_features=int_dim_res, out_features=output_dim))
    mlp.append(Affine())
    self.mlp = nn.Sequential(*mlp)

    # CNN appendix: axis-aware blocks acting in theta order. Give
    # the CNN head the same activation as the trunk's ResBlocks:
    # block_opts["act"] carries the run's chosen activation (the
    # --activation flag, injected by EmulatorExperiment), falling
    # back to activation_fcn (the paper's H, the ResBlock default)
    # when block_opts sets none. Without this, a non-default
    # activation would reach the trunk only and the head would
    # silently stay on H.
    cnn_act = block_opts.get("act", activation_fcn)
    self.cnn = nn.ModuleList([
      CNNBlock(output_dim, kernel_size=kernel_size,
               channels=channels, act=cnn_act)
      for _ in range(n_blocks_cnn)])

    # learnable scalar gate on the correction (small init, not 0).
    self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    # Frozen basis-change maps as buffers (move with .to(device),
    # not trained). Naming: f = full-whitened basis (cov
    # eigenbasis), d = diagonal basis (theta order, /sigma). The
    # subscripts read in multiply order, so x @ W_fd maps f -> d
    # and x @ W_df maps d -> f. sigma = DiagonalGeometry per-element
    # scale sqrt(diag cov); evecs/sqrt_ev give the full basis.
    #   full-whitened y -> physical -> theta order (/sigma):
    #     W_fd = diag(sqrt_ev) evecs.T diag(1/sigma)
    #   theta-order correction -> physical -> full-whitened:
    #     W_df = diag(sigma) evecs diag(1/sqrt_ev)  (= W_fd^{-1})
    evecs   = geom.evecs.detach()
    sqrt_ev = geom.sqrt_ev.detach()
    sigma   = torch.sqrt(((evecs * sqrt_ev) ** 2).sum(1))
    self.register_buffer(
      "W_fd", (sqrt_ev[:, None] * evecs.t()) / sigma[None, :])
    self.register_buffer(
      "W_df", (sigma[:, None] * evecs) / sqrt_ev[None, :])

  def forward(self, x):
    # ResMLP prediction in the full-whitened basis (the bulk map).
    y = self.mlp(x)                   # (B, out_dim)
    # full-whitened trunk output -> theta-ordered (diagonal) view
    # so the conv sees the angular axis (f -> d, via W_fd).
    h = y @ self.W_fd
    for blk in self.cnn:
      h = blk(h)                      # axis-aware correction
    # correction back to the full-whitened basis (d -> f, via
    # W_df), gated, and added to the trunk output.
    return y + self.gate * (h @ self.W_df)
