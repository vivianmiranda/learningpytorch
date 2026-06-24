# Session notes — cosmic-shear emulator

Self-contained handoff for continuing this work in a fresh Claude session.

## Project

`pytorch1.ipynb` — a teaching Jupyter notebook (destined for RISE slides)
building a cosmic-shear **data-vector emulator**: a ResMLP maps cosmological
parameters to the whitened, masked cosmic-shear (xi) data vector; the loss is the
full 3x2pt chi2 via cosmolike's masked inverse covariance.

- **Goal:** 90% of validation cosmologies with emulator error delta-chi2 < 0.2,
  i.e. `frac>0.2 < 0.10`.
- cosmolike (`ci`) runs only on the workstation; only the user can run cosmolike
  cells — review such code statically.

## Where the emulator stands

`frac>0.2` plateaus around 0.36–0.43. Investigation concluded:
- Loss-shaping (trim annealing, focal reweighting) is exhausted — neither beat
  the const-5%-trim baseline; the floor is not a loss problem.
- The hard, high-chi2 cosmologies cluster at **high physical baryon density**
  `omega_b h^2 = Omega_b (H0/100)^2`, with a sharp cliff at `omega_b h^2 ~ 0.04`
  (cut-scan: below it the bulk clears 0.2; above it dchi2 ~ 1e2–1e4). 0.04 is ~2x
  Planck (0.0224), so the worst region is unphysical.
- But the cliff is only ~8% of points, so it barely moves `frac>0.2` (it
  dominates the *mean*, not the count). Even restricting to `omega_b h^2 < 0.025`
  leaves `frac>0.2 ~ 0.29` — the floor is broad in-region accuracy, not just the
  unphysical tail.

## Decisions in place

- **Physical cut:** train and validate only on `omega_b h^2 < 0.035`
  (`phys_cut_idx`, `OMEGABH2_CUT = 0.035`). cs_16 = training (T=16), cs_8 =
  validation (T=8 = T/2). The cut removes the unphysical cliff from both.
- **Two-source loaders:** train and val are separate files; source dicts
  `{"C","dv","idx",...}`; geometry (whitening center, Cinv) built once from
  training and applied to both.

## Analytic-scaling preprocessing (the main new work)

Shrink the target's dynamic range by dividing out an analytic reference.
Preprocess element-wise `d_tilde = d * R`, `R = xi_analytic(mid)/xi_analytic(cosmo)`;
the net emulates the flatter `d_tilde`; recover physical `d = d_tilde / R` before
the chi2 (so covariance + reported metric unchanged). A **ratio** is key — the
cosmology-common nonlinear boost cancels, so a crude linear no-halofit reference
still works.

**Analytic model** (reference only, accuracy not needed): linear P(k) with the
Eisenstein–Hu **zero-baryon** transfer; Limber; single source plane at each n(z)
peak; `H(z)=H0` (`chi = (c/H0) z`); Hankel as a delta at `l*theta=1`; Limber at
the kernel peak `u_star ~ 0.5`. Results:
- `A_s` carries the whole amplitude (`Omega_m`, `H0` cancel between kernel and P).
- Shape collapses to one dimensionless wavenumber
  `q = K/(theta * z_eff * Gamma)`, `K = 100 Theta^2/(c u_star)`,
  `Gamma = Omega_m h`, `Theta = T_cmb/2.7`, `z_eff = min(z_i, z_j)` for a
  tomographic pair (cross spectrum cut off at the nearer source).
- `R = R_amp * q_mid^ns_mid T(q_mid)^2 / (q^ns T(q)^2)`, with the per-cosmology
  amplitude scalar
  `R_amp = (As_mid/As) * (Om_mid h_mid^2)^ns_mid/h_mid / ((Om h^2)^ns/h)`.
  The `(Om h^2)^ns/h` factor (geometric amplitude N) is a 2nd amplitude
  direction; gated by the `include_amp` flag (standard run = True).
- Zero-baryon means **no Omega_b dependence**, so R cannot touch the
  `omega_b h^2` floor (the hard direction). It flattens the broadband bulk, not
  the baryon tail. Capturing the tail would need the wiggle (with-baryon) E&H
  transfer.

**Validation** (spread ratio = std_rescaled/std_raw across val cosmologies, per
kept element; lower = more variance removed):

| variant | median | mean | frac improved |
|---|---|---|---|
| As only | 0.79 | 0.76 | 1.00 |
| + shape | 0.515 | 0.60 | 0.89 |
| + N (include_amp) | 0.456 | 0.50 | 0.98 |

~37% → ~74% → ~79% of variance removed. Best at small theta (constraining,
small-error-bar scales); ~1 only at large theta (biggest covariance,
down-weighted by Cinv). N also lifts the improved fraction 0.89→0.98 by removing
the residual amplitude that dominates at large theta (T→1, near-pure amplitude).
Full step-by-step derivation: `analytic_scaling.pdf` / `.tex` (repo root).

**Code (notebook functions):**
- `_analytic_R(theta_arcmin, z_eff, cosmo, cosmo_mid, names, u_star=0.5,
  include_amp=False)` — the single core (q/T/R formula; element arrays broadcast
  to any shape).
- `analytic_shape_ratio(cosmo, cosmo_mid, names, geom, u_star=0.5,
  include_amp=False)` — wraps the core over the masked/kept dv (flat (N, n_keep)),
  for the pipeline. `z_eff = min(zsrc_i, zsrc_j)`.
- `rescale_xi(xi, cosmo, cosmo_mid, names, geom, ...)` — wraps the core over full
  unmasked xi matrices, for plotting.
- `build_shear_angle_map(geom, data_dir, dataset)` — attaches `theta_kept`,
  `zsrc_i/j`, `theta_centers`, `z_src`, `ntheta`, `source_ntomo`, `xi_size`
  (reads dataset ini + n(z) file, no cosmolike). Must be called before
  analytic_shape_ratio / rescale_xi / dv_to_xi.
- `dv_to_xi(dv_row, geom)` — flat dv row → (theta, xip, xim) matrix for `plot_xi`.
- `RescaledChi2(CosmolikeChi2)` — opt-in subclass: `encode(dv, R)`/`decode(y, R)`
  apply R; `chi2(pred, target, R=None)` and `loss(pred, target, R, ...)` divide R
  out so the chi2 stays on the physical dv. Base class untouched, so the two are
  A/B-swappable. `chi2` falls back to a per-batch `self._R` stash that `loss`
  sets, so the inherited trim/focal reduction reuses it.

## Next step (the decisive open task): wire the rescaling into training

Not yet done — the run still uses plain `CosmolikeChi2`; `RescaledChi2` is used
only in analysis cells. Plan:
- Add `cosmo_mid` / `names` / `include_amp=True` params to `run_emulator` →
  `build_loaders` → `_build_loaders_one`.
- `_build_loaders_one`: when `cosmo_mid` is given, compute
  `R_used = analytic_shape_ratio(C[used_rows], cosmo_mid, names, chi2fn,
  include_amp=True)`, make it resident on device, encode the resident targets
  with it (`chi2fn.encode(dv_block, R_block)`), and expose `load_R(rows)` parallel
  to `load_C`. Count R in the resident bytes.
- `training_loop_batched`: per batch, if `load_R` present, pass R:
  `lossfn.loss(pred, dvc[b], R_chunk[b], mode, trim=rob, focus=focus,
  focus_scale=kappa)`.
- `eval_val`: per chunk, pass R: `lossfn.chi2(pred, dvc, R_chunk)`.
- Branch on whether R is present (CosmolikeChi2 vs RescaledChi2 loss/chi2
  signatures differ). Gate everything on `cosmo_mid is None` so the base
  (no-rescale) path stays A/B-runnable.
- `cosmo_mid` = the training-cloud mean (`train_set["C_mean"]`-style center;
  R = 1 there).
- Then the decisive test: re-run the cut-scan / `frac>0.2` on the rescaled target.
  The spread ratio is only a cheap proxy.

## RAM-efficient data loading

- Memmap the full dv dump: `np.load(path, mmap_mode="r", allow_pickle=False)` so
  it never enters RAM whole. Params: `np.loadtxt(path, dtype="float32")[:, 2:-1]`
  (drop weight/lnp/chi2).
- `stage_source(C, dv, idx, ram_frac=0.7)`: if the used subset fits 70% of
  `psutil.virtual_memory().available`, materialize the compact subset
  (`np.asarray(C[rows])`, `np.asarray(dv[rows])`, `rows = sort(unique(idx))`) and
  **reindex locally** (`idx = arange`); else return `(C, dv, idx)` unchanged
  (memmap + global idx). Local reindex means every consumer
  (loaders/geometry/eval/analysis touch C/dv only through idx) works unchanged.
- Source dict carries the centers:
  `train_set = {"C","dv","idx","C_mean","dv_mean"}`. The geometry reads
  `train_set["dv_mean"]` / `["C_mean"]` (computed once over the cut rows via
  `stream_stats` / `param_stats`), no loose globals. `val_set` has no means
  (geometry is training-only). These are references, not deep copies (a dict entry
  references the array — no duplication; `del` the loose names to keep only the
  dict).

## Open bug

The "20 random validation" plot cell (builds `chi2fn_v2 = RescaledChi2`,
`xi_list = [dv_to_xi(...) for r in sel]`, calls `plot_xi(1, xi_list, ...)` then
`plot_xi(1, xi_resc, ...)`) references `xi_resc` but never builds it — the
`xi_resc = rescale_xi(xi_list, cosmo, cosmo_mid, names, chi2fn_v2)` line was
dropped in a restructure. Add it back (with the y-axis sync that forces both
panels onto a shared per-panel scale via the union of their ylims).

## Conventions (house style for code in this notebook)

- Deliver full paste-ready cells, never diffs or snippets. Locate edits by
  function name / comment marker / quoted line, never by cell number.
- Code lines <= ~60 chars (slide width); 2-space hanging indent on wrapped
  calls/signatures, not alignment under the opening paren.
- Formal docstrings: an `Arguments:` block listing ALL params, plus `Returns:`.
- Functions take what they need as parameters; no module-level data globals
  (imports/helpers are fine); flag any genuinely unavoidable global in-code.
- Plots colorblind-safe (no red+green; viridis/cividis for ordered data, or an
  explicit Wong-style palette).
- Spec-dict construction (`{cls, **kwargs}` + `make_X` helpers); weight decay only
  on weight matrices (`ndim >= 2`).
- Dev machine is Mac M2 / MPS (no float64 on device, unified 32 GB); train on
  NVIDIA — branch on `device.type`.
