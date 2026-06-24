---
name: analytic-scaling-preprocessing
description: "Physics-informed RATIO preprocessing for the cosmic-shear emulator: multiply each data vector by R = xi_analytic(mid)/xi_analytic(cosmo) from a crude analytic model (E&H zero-baryon, linear, Limber, single-source-plane delta-n(z), H=H0), emulate the flatter residual, divide R back out before the chi2. The ratio cancels cosmology-common nonlinearity so a linear no-halofit model still works. Validated by spread ratio (As-only 0.79, +shape 0.515, +N amplitude 0.456; N=(Om h^2)^ns/h also lifts improved-fraction 0.89->0.98). include_amp=True is the standard. Helps the broadband bulk, NOT the omega_b h^2 floor. Full derivation in analytic_scaling.pdf; NOT yet wired into the training pipeline."
metadata: 
  node_type: memory
  type: project
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

A preprocessing scheme for the emulator ([[emulator-pipeline-and-goal]]) to
shrink the target's dynamic range so the network has less to fit.

**The idea (a ratio, which was the right call):** pick a reference "mid"
cosmology and preprocess element-wise `d_tilde = d * R`, with
`R = xi_analytic(mid) / xi_analytic(cosmo)`. The net emulates `d_tilde` (it
clusters near the mid cosmology); recover the physical `d = d_tilde / R` before
the chi2, so the covariance and reported metric are unchanged. A ratio (not a
difference, not absolute) is key: the cosmology-common part of
`xi_data/xi_analytic` (most of the nonlinear boost) cancels in mid/cosmo, so the
crude analytic only has to capture the cosmology-DEPENDENCE -- which is mostly
linear (amplitude, tilt, turnover). That is why a linear, no-halofit reference
still works, and best at small scales.

**The analytic model (all just to be a reference, accuracy not needed):** linear
P(k) with the Eisenstein-Hu zero-baryon transfer; Limber; each source bin a
single plane at its n(z) peak (delta-n(z)); pure dark energy H(z)=H0 so
chi = (c/H0) z; Hankel to real space as a delta at l*theta=1; Limber integral at
the kernel peak u_star. Key results:
- A_s carries the WHOLE amplitude: Omega_m and H0 cancel between the lensing
  kernel (q^2 ~ Om^2 H0^4) and P (~ As Om^-2 H0^-4), so C_l = As * G(shape).
  Dividing by As_mid/As is exact and removes the single largest variance.
- The shape collapses to one dimensionless wavenumber
  `q = K/(theta * z_eff * Gamma)`, `K = 100 Theta^2/(c u_star)`,
  `Gamma = Omega_m h`, `Theta = T_cmb/2.7`. So Om, h, z_s enter only through
  `q ~ 1/(theta * Gamma * z_s)`.
- Full factor: `R = R_amp * q_mid^ns_mid T(q_mid)^2 / (q^ns T(q)^2)`, with the
  per-cosmology amplitude scalar
  `R_amp = (As_mid/As) * (Om_mid h_mid^2)^ns_mid/h_mid / ((Om h^2)^ns/h)`.
  The `(Om h^2)^ns/h` piece is the surviving geometric amplitude N (z_s and
  theta cancel in the ratio) -- a SECOND amplitude direction beyond As, gated by
  the `include_amp` flag (default off, but the standard run sets it True). Each
  cosmology uses its own params, reference uses mid's; R=1 at mid.
- Cross tomographic pairs: `z_eff = min(z_i, z_j)` (the cross spectrum is cut off
  at the nearer source, so its effective scale is the lower bin's), NOT the
  geometric mean. Auto pairs: min = the bin's z. z_eff cancels at leading order
  in R, so this is a few-percent (asymmetric-pair) correction.

**Validation (spread ratio = std_rescaled/std_raw across val cosmologies, per
kept element; lower = more variance removed):** As-only (R=As_mid/As scalar)
median 0.791 (~37% of variance); +shape median 0.515 (~74%, 89% improved);
+N amplitude median 0.456 (~79%, 98% improved). So the SHAPE removes more than
As alone, and the N scalar adds more still -- use the full As+shape+N, do not
collapse to the As scalar. N also RAISES the improved fraction 0.89->0.98: the
shape term left the (Om h^2)^ns/h amplitude in the residual, which dominates at
large theta where T->1 and the variation is nearly pure amplitude (exactly the
elements shape made worse); N removes that, and is most reliable there (no
transfer curvature). By theta: best at small theta (2.8'-30', ratio ~0.34-0.5,
the constraining small-error-bar scales), ~1 only at the largest theta (>~500',
biggest covariance, down-weighted by Cinv). A net win on the scales that matter.

**What it does and does not fix:** removes the broadband (As / Om h / ns)
variance; the zero-baryon transfer has NO Omega_b dependence, so R cannot touch
the omega_b h^2 = Omega_b (H0/100)^2 direction -- which is exactly the hard,
high-omega_b h^2 cliff that drives the frac>0.2 floor (see
[[emulator-floor-is-data-coverage]]). Capturing that would need the WIGGLE
(with-baryon) E&H transfer. So treat this as "makes the bulk easier," not "fixes
the baryon tail."

**Code (in the notebook):** `_analytic_R(theta, z_eff, cosmo, cosmo_mid, names)`
is the single core (the q/T/R formula lives once); `analytic_shape_ratio` wraps
it over the masked/kept dv (flat (N, n_keep), for the pipeline) and `rescale_xi`
wraps it over the full unmasked xi matrices (for plotting). `build_shear_angle_map`
attaches the angle/tomography metadata (theta_kept, zsrc_i/j, theta_centers,
z_src, ntheta, source_ntomo, xi_size) by reading the dataset ini + n(z) file
only -- no cosmolike. `dv_to_xi` reshapes a flat dv row into plot_xi's
(theta, xip, xim) matrix layout. `RescaledChi2(CosmolikeChi2)` is the opt-in
subclass: overrides encode/decode (apply R) and chi2/loss (divide R out so the
chi2 stays on the physical dv); base class is untouched, so the two are
A/B-swappable. Derivation written up in `analytic_scaling.pdf` / `.tex` at the
repo root.

**Pending (next step, the session's last open task):** NOT yet wired into
training -- the %%time run still passes the plain CosmolikeChi2; RescaledChi2 is
used only in the analysis/plot cells. The integration (designed, not written):
add `cosmo_mid`/`names`/`include_amp` to run_emulator -> build_loaders ->
_build_loaders_one, which computes `R_used = analytic_shape_ratio(C[used_rows],
cosmo_mid, names, chi2fn, include_amp=True)`, encodes the resident targets with
it, and exposes a `load_R(rows)` (parallel to load_C); training_loop_batched and
eval_val branch on load_R to pass the batch's R to loss/chi2 (the
CosmolikeChi2-vs-RescaledChi2 loss/chi2 signatures differ, so branch). Gate on
cosmo_mid=None so the base path stays A/B-runnable. Then the decisive test is
frac>0.2 on the rescaled target via the cut-scan -- the spread ratio is only a
cheap proxy. (A separate restructure bug to fix: the "20 random val" plot cell
references `xi_resc` but never builds it -- the `xi_resc = rescale_xi(...)` line
was dropped.)

**Why:** records a validated, physically-motivated lever (and its ceiling) so we
do not re-derive it or re-litigate ratio-vs-difference; and flags the open
integration so the next session goes straight to wiring + the frac>0.2 test
rather than re-running the spread/visual checks.
