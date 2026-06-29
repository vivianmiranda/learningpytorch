"""Factored intrinsic-alignment template models."""

import torch.nn as nn

from ..emulator_designs_building_blocks import Affine, ResBlock


class NLATemplateMLP(nn.Module):
  """
  Factored NLA emulator: maps the 11 NON-A1_1 params (cosmo +
  A1_2) to THREE whitened templates [GG, GI, II]. The IA
  amplitude A1_1 is applied in closed form by the loss
  (xi = GG + A1_1*GI + A1_1^2*II), so it never enters the
  network -- which is what makes the A1_1 generalization exact.

  Input layout (NLAInputGeometry.encode): the LAST column is the
  raw A1_1 (for the loss); this model uses only [:, :-1].
  output_dim = n_keep (one template width); it emits 3*n_keep
  and reshapes to (B, 3, n_keep).
  """
  def __init__(self, input_dim, output_dim, int_dim_res,
               n_blocks=4, block_opts=None):
    """Build the residual trunk and the 3-template output head.

    Arguments:
      input_dim   = full encoded input width (12 = 11 model
                    features + the appended A1_1 column).
      output_dim  = one template's length (n_keep, the unmasked
                    dv size); the model emits 3 of them.
      int_dim_res = internal residual width.
      n_blocks    = number of residual blocks.
      block_opts  = ResBlock options dict (None -> {}).
    """
    super().__init__()
    if block_opts is None:
      block_opts = {}
    self.n_keep = output_dim
    # n_in = the model's real input width: drop the 1 appended
    # A1_1 column (it is the loss's input, not the net's).
    self.n_in   = input_dim - 1
    layers = [nn.Linear(in_features=self.n_in, out_features=int_dim_res)]
    for _ in range(n_blocks):
      layers.append(ResBlock(int_dim_res, **block_opts))
    # one output projection emitting all THREE templates stacked.
    layers.append(nn.Linear(in_features=int_dim_res, out_features=3 * output_dim))
    layers.append(Affine())
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    """Map cosmo + A1_2 to the three whitened templates.

    Arguments:
      x = (B, input_dim) encoded parameters; the last column is
          A1_1 (ignored here), and [:, :-1] are the whitened
          cosmo + A1_2 features the templates depend on.

    Returns:
      (B, 3, n_keep): the whitened templates [GG, GI, II].
    """
    h = self.model(x[:, :self.n_in])           # (B, 3*n_keep)
    # view reshapes WITHOUT copying: the flat (B, 3*n_keep)
    # row is split into (B, 3, n_keep), so each row's first
    # n_keep entries become template GG, the next n_keep GI,
    # the last n_keep II. (view needs contiguous memory, which
    # a Linear output is, so the reshape is free.)
    return h.view(x.shape[0], 3, self.n_keep)   # (B, 3, n_keep)


class TemplateMLP(nn.Module):
  """
  Factored IA emulator: maps the non-amplitude parameters
  (cosmo + photo-z + the IA evolution powers eta) to
  n_templates whitened templates. The IA AMPLITUDES are applied
  in closed form by the loss, so they never enter the network --
  which is what makes the amplitude generalization exact and
  prior-width-independent.

  Input layout (AmplitudeFactorGeometry.encode): the last n_amps
  columns are the raw amplitudes (for the loss); this model uses
  only [:, :-n_amps]. output_dim = n_keep (one template width);
  it emits n_templates*n_keep and reshapes to (B, n_templates,
  n_keep). NLA: n_amps=1, n_templates=3; TATT: n_amps=3,
  n_templates=10.
  """
  def __init__(self, input_dim, output_dim, n_amps,
               n_templates, int_dim_res, n_blocks=4,
               block_opts=None):
    """Build the residual trunk and the template output head.

    Arguments:
      input_dim   = full encoded input width (non-amplitude
                    features + the n_amps appended amplitudes).
      output_dim  = one template's length (n_keep, the unmasked
                    dv size); the model emits n_templates.
      n_amps      = number of appended amplitude columns to drop
                    from the input (1 NLA, 3 TATT).
      n_templates = number of templates to emit (3 NLA, 10 TATT);
                    must match the coeff_fn's length.
      int_dim_res = internal residual width.
      n_blocks    = number of residual blocks.
      block_opts  = ResBlock options dict (None -> {}).
    """
    super().__init__()
    if block_opts is None:
      block_opts = {}
    self.n_keep      = output_dim
    self.n_templates = n_templates
    # n_in = real input width: drop the n_amps amplitude columns
    # (they are the loss's input, not the net's).
    self.n_in = input_dim - n_amps
    layers = [nn.Linear(in_features=self.n_in, out_features=int_dim_res)]
    for _ in range(n_blocks):
      layers.append(ResBlock(int_dim_res, **block_opts))
    layers.append(nn.Linear(in_features=int_dim_res,
                            out_features=n_templates * output_dim))
    layers.append(Affine())
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    """Map the non-amplitude params to the whitened templates.

    Arguments:
      x = (B, input_dim) encoded parameters; the last n_amps
          columns are the amplitudes (ignored here), and
          [:, :-n_amps] are the whitened cosmo + photo-z + eta
          features the templates depend on.

    Returns:
      (B, n_templates, n_keep): the whitened templates, in the
      coeff_fn order (template 0 carries the no-IA / center part).
    """
    h = self.model(x[:, :self.n_in])
    # view reshapes the flat (B, n_templates*n_keep) output
    # into (B, n_templates, n_keep) without copying -- each
    # template's n_keep values become one slice along axis 1.
    return h.view(x.shape[0], self.n_templates, self.n_keep)
