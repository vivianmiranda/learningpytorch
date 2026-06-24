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

## Where the emulator stands — FLOOR IS DATA-LIMITED (settled 2026-06-24)

With the physical cut `omega_b h^2 < 0.035`, the baseline `frac>0.2 ≈ 0.20` used
only ~1/10 of the pool. The **data-scaling / learning curve** settles it — growing
`N_train` drops val `frac>0.2` steeply:

> `N_train 10k → 0.219`,  `46k → 0.100` (the goal),  pool `= 82k` (82k pending)

**More data helps, and the 0.10 goal is already hit at 46k.** The floor was simply
the 10% training subset being too small; the fix is to **train on more of the
available pool.**

> CORRECTION: a mid-session read called this a **model-capacity** limit (from
> `train frac>0.2 = 0.17 ≈ val = 0.20`, i.e. underfitting its own training set).
> **That was wrong** — the scaling curve refutes it. `train ≈ val` rules out
> *overfitting* but does **not** prove capacity; a regularized model under sparse
> data looks the same. Only the **learning curve** (metric vs `N_train`) settles
> capacity-vs-data, and here it says **data**. Lesson: run the learning curve;
> the train-vs-val gap diagnoses overfitting, not the floor.

Full write-up: [[emulator-floor-is-data-coverage]]. Loss-shaping / LR / batch /
per-element weight were all correctly exhausted — the lever was always more data.

**The diagnostic ladder (transferable, also added to the skill):**
- Threshold ladder (0.2/0.5/1/10/100): failures are a **shoulder** piled just
  above 0.2 (~half in 0.2–0.5), the upper tail of one broad log-normal
  (median 0.05, `σ_log≈1.65`). You can't narrow a spread by reweighting → loss
  shaping is dead on arrival.
- Optimization ruled out: halving the LR tightened the late bounce, floor unmoved.
  Batch size dead too (sqrt-lr coupling makes all bs converge; 512 only broke
  because the coupled lr overshot — AdamW needs little/no lr scaling with bs).
- Loss-shaping ruled out empirically: focal+kappa, threshold bump, and a
  per-element focal (`ElementWeightedChi2`) all neutral.
- The **marginal per-element lens misled us** (flagged high-z small-θ); the net
  leaves ~1.5σ marginal residuals there even on train and the loss tolerates them
  → correlated common modes the chi2 barely charges for. Diagnose in the metric's
  **own decorrelated coordinates**, not a marginal one. (Verified `chi2 ==
  ||pred-target||²` to 0.1% → whitening IS the chi2 basis, no conditioning bug.)
- **Decisive (corrected):** the **learning curve** (`frac>0.2` vs `N_train`), not
  the train-vs-val gap. `train ≈ val` only rules out overfitting; the curve falls
  `0.22 → 0.10` from 10k → 46k = **data-limited**. (Tempering confound for the
  per-element view: T_val = T_train/2, so val<train per element is just the
  temperature — compare at the metric.)

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

## Rescaling: WIRED, VERIFIED, RUNNING (as of 2026-06-24)

The rescaling is now integrated into the training run and verified; the only open
task is reading the `frac>0.2` result. (All code delivered as paste-ready cells —
the notebook is read-only, never edited in place.)

**Approach — loss-side `RescaledChi2`, plain `ResMLP`, R computed on the fly.** R
is a deterministic function of the cosmological params, so it is never stored (the
rejected `load_R` plan stored an `(N_rows, n_keep)` array, ~doubling resident-target
memory — the regime-1 binding constraint at the real million-dv scale). The loss is
handed the cosmology it already has and computes R itself ("the loss knows the
cosmology and the dv"). A *model-side* variant (`RescaledResMLP`, rescaling inside
`forward`) was explored — it avoids touching the loop entirely and is provably
identical (value + gradient, verified `/tmp/test_model_wrapper.py`) — but rejected
to keep the plain `ResMLP` and the clean "rescale in the loss" picture; the user
accepted the one-argument loop change instead.

**Why the loop must change at all (the fundamental point):** the loss is handed
`pred` and `target`, both data vectors (xi) — never the cosmology. R needs the
params (As, ns, H0, Ωm), which are not recoverable from a dv. So the loss must be
*given* the params, and only the loop holds them per-minibatch (`Cc[b]`). That one
argument is the whole change.

**R is needed in two spots, same source:** building the target (`encode(dv*R)`,
once at pre-encode) and undoing it in the chi2 (`r = unwhiten(pred-target)/R`,
every step). Both derive R from `param_geometry.decode(whitened_params)`, so they
use bit-identical R. `_analytic_R` made dual numpy/torch (one formula; numpy path
bit-identical to old; torch on-device). Verified: numpy==torch ~1e-15 (float64),
float32 ~1e-6, R=1 at the reference exactly (`/tmp/test_analytic_R.py`).

**`RescaledChi2`** (subclass of `CosmolikeChi2`): `configure_rescaling(pgeom,
cosmo_mid, names, include_amp=True, u_star=0.5)` attaches the config (asserts
`build_shear_angle_map` ran first); `_R(params_whitened)` decodes → physical →
analytic R, caching theta/zeff device tensors; `encode`/`decode`/`chi2`/`loss`
take the whitened params; `loss` stashes them so the inherited trim/focal
reduction (`super().loss` → `self.chi2`) reuses them. `isinstance` keeps `super()`
dispatch landing on `RescaledChi2.chi2`.

**Wiring — branches only, gated on `isinstance(.., RescaledChi2)`, base path
untouched and A/B-runnable.** `_build_loaders_one` pre-encode (all 3 regimes)
passes the whitened param slice to `chi2fn.encode`; `training_loop_batched` batch
loop passes `Cc[b]`; `eval_val` passes the chunk's `Cc`; `eval_source_chi2`
(scoring) passes `X`. `run_emulator` / `build_loaders` / `make_model` unchanged
(config rides on chi2fn). Base signatures have no params slot, so passing
unconditionally crashes the no-rescale run — hence the branch.

**Driver = the rescaled `%%time` run cell.** `pgeom = ParamGeometry.from_covmat`;
`chi2fn = RescaledChi2.from_cosmolike(device, dvt_off, probe="xi")`;
`build_shear_angle_map(chi2fn)`; `cosmo_mid = train_set["C"][train_set["idx"]
].mean(0)`; `chi2fn.configure_rescaling(pgeom, cosmo_mid, list(pgeom.names),
include_amp=True)`; sanity cell; `run_emulator(...)` with `model_opts` = plain
`ResMLP` and `chi2fn` = the configured `RescaledChi2`.

**Sanity verified (the pre-flight cell passes):** `max|R(mid)-1| = 0.0` exactly;
the xi layout is confirmed — `theta_kept` rises 2.8'→33.78' (theta inner) while
`(zsrc_i,zsrc_j)` stays `(0.33,0.33)` across the first pair (pairs outer), matching
`build_shear_angle_map`'s assumption, so R hits the right elements. Smallest thetas
masked by the scale cuts (first kept = 2.8').

**Bugs fixed during the review pass (all now in the notebook):** cell-286
Basic-tests load computed means from `train_set` before it existed (→ read locals
`dv0`/`C0`/`tidx`); `g` (RNG generator) clobbered by `g = gplot.getSubplotPlotter`
in the triangle cells (→ plotter renamed `gp`); `eval_val` rescaled branch used
`Cc[b]` (`b` undefined there) instead of the whole-chunk `Cc`; `analytic_shape_ratio`
and `rescale_xi` took the whole `geom` but used only a few attrs (→ take
`theta_kept`/`zsrc_i`/`zsrc_j` and `z_src` explicitly); `eval_source_chi2` needed
the `isinstance` branch; `configure_rescaling` gained an early `hasattr` guard.

**The one open task:** read `frac>0.2` on the rescaled target (baseline ~0.36).
The spread ratio (0.46) was only a proxy. Expectation: helps the broadband bulk,
cannot move the `omega_b h^2` tail (zero-baryon transfer, see
[[emulator-floor-is-data-coverage]]).

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

## Open bug (resolved)

The "20 random validation" plot cell once dropped the
`xi_resc = rescale_xi(...)` line. In the current notebook it is present again
(the `# same 20, analytically rescaled` block builds `xi_resc` with
`include_amp=True`, followed by the per-panel y-axis sync over the union of both
panels' ylims). No action needed.

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
