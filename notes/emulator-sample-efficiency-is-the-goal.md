---
name: emulator-sample-efficiency-is-the-goal
description: "The cosmic-shear emulator's REAL objective is SAMPLE EFFICIENCY -- the position of the learning curve f(delta-chi2>0.2) vs N_train -- not the asymptotic floor at unlimited data. The toy testbed (w0wa cosmic shear, ~82k vectors) is where methods are developed; the real target is T=512 + w0wa + TATT (~17+ params, hot prior, expensive cosmolike vectors) where N_train is the BINDING constraint and 'just add data' is infeasible. So every lever (analytic rescaling, architecture, physics input features, loss/conditioning, active sampling) is judged by whether it SHIFTS THE LEARNING CURVE LEFT (same accuracy, fewer samples), NOT by the floor at one fixed N_train. Corrects how rescaling etc. were judged this session."
metadata:
  node_type: memory
  type: project
---

The objective is **sample efficiency**, not the floor. The number that matters is
N_target = the smallest N_train at which a method reaches the target accuracy
(f(delta-chi2>0.2) < 0.10), or equivalently the accuracy at a fixed affordable
training budget. The deliverable is a method whose **learning curve** (f vs
N_train) is as far left / low as possible.

**SESSION RESULTS (2026-06-24) -- where this stands:**
- Plain ResMLP learning curve (val frac>0.2): 2k 0.51, 3.7k 0.33, 6.9k 0.245,
  10k 0.219, 46k 0.100 (goal), 82k ~0.06. Steep, still falling -> data-limited,
  NOT capacity ([[emulator-floor-is-data-coverage]]).
- TARGET RESCALING LOST the sample-efficiency test (narrow, one emulator): worse
  than plain at small N, converged by ~7k. Deprioritized. Mechanism = an
  UNCONFIRMED conditioning hypothesis, not a law -- see
  [[analytic-scaling-preprocessing]]. Do NOT over-generalize ("any reparam hurts"
  is false).
- UNTESTED live levers to try next (use the analytic WITHOUT touching the loss):
  (1) analytic dv as an INPUT feature / residual base; (2) PRETRAIN on cheap
  analytic dvs then fine-tune on cosmolike with the plain loss. Both keep the
  plain (well-conditioned) loss.
- Space-filling / maximin selection is WRONG here -- it pushes to the corners
  (the infeasible uniform) and fights the deliberate Gaussian-with-T_train,
  capped-correlation-0.25 sampling design. The sampling (T_train, cov shape) is
  the user's domain and is itself the volume-reduction strategy; do not replace it.
- CERTIFICATION discipline for the hard f<0.1 bar: f over a finite val set is
  noisy (binomial ~+/-0.015 at Nval~400) and best-epoch selection biases it LOW,
  so a single run at f=0.100 is NOT a certificate. Certify with a margin (e.g.
  target 0.085), several seeds (all must pass), and a bigger val set (sigma_f ~
  1/sqrt(Nval), free to enlarge).

**META-LESSON of this session (recorded against repetition): twice a sweeping
conclusion was drawn from one run -- "it's capacity" (refuted by the learning
curve) and "any target reparam hurts" (over-general from the rescaling). Size the
claim to the evidence: report the narrow empirical fact, flag the mechanism as a
hypothesis, and run the curve/experiment before writing a law.**

**SESSION 2026-06-25 -- where this stands (resume here).**
- OBJECTIVE SHARPENED (the user's framing, and it is the right one): the test is
  the CONVERGED PLATEAU of frac>0.2 at a FIXED useful N (~10k), NOT the small-N
  relative learning curve. A lower 10k plateau = reaching f<0.10 with fewer points.
  Small-N relative comparisons are useless if no method clears 0.10 up there.
- OUTPUT-REPARAMETRIZATION LEVERS ARE CLOSED. A (RescaledChi2) and B
  (ResidualBaseChi2) both came out == plain at a single converged ~10k run (~0.20),
  because they SHARE plain's optimum -- a converged single-N run CANNOT separate
  reparametrizations (conditioning shows only in the path / at small N, and even
  the paths matched). Stop running single-N A/B comparisons. See
  [[geometry-loss-composition]].
- THE PRINCIPLE for lowering a DATA-limited (train==val) plateau at fixed N: make
  the target EASIER TO GENERALIZE from few points -- reduce the effective DOF the
  net must learn, or inject structure. NOT reparametrization (A/B, same optimum),
  NOT capacity (data-limited), NOT regularizers (train==val -> variance already
  low -> neutral), NOT loss-shaping (exhausted; cannot narrow a spread).
- HARD CONSTRAINTS reaffirmed (do NOT propose these again): no changing the
  analytic R; no fine-tuning / pretrain (for now); no tighter cut -- the
  Gaussian-T sampling is fixed; sigma8 / early-universe combos are wrong lensing
  inputs and ParamGeometry already PCAs the inputs.
- A smoothness / anti-oscillation prior is OUT: the chi2 is a HIGH-PASS filter on
  the error (it already crushes oscillation; its blind spot is the smooth
  common-mode), so the penalty is redundant -- see the [[pytorch-teaching-style]]
  skill.
- PER-BIN DENSE-MLP SPLIT FAILED (2026-06-25): a per-bin ParallelResMLP came out
  WORSE than the single ResMLP (~0.27 vs ~0.24) -- it threw away the shared
  cosmology map and a dense MLP can't exploit output-axis structure anyway
  ([[per-bin-parallel-resmlp-plan]]).
- LIVE LEVERS (both reduce effective DOF, both keep the shared trunk):
  (1) analytic INPUT feature -- AugmentedParamGeometry BUILT but NOT yet run
  ([[geometry-loss-composition]]); (2) SHARED ResMLP trunk + PER-BIN 1D CNN head
  ([[resmlp-cnn-perbin-architecture]]) -- an AXIS-AWARE output head, with PUBLISHED
  precedent (the user's CMB paper: ResMLP+TRF/CNN dropped f>0.2 ~0.2 -> ~0.06 vs
  bare ResMLP). The CNN head is the NEXT BUILD and the most promising lever.

**Why (the binding constraint at the real scale).** The notebook
([[emulator-pipeline-and-goal]]) is a TESTBED: w0wa cosmic shear, ~12 params,
~82k data vectors available -- here data is cheap, and the toy floor is
data-limited (f hits 0.10 at 46k of 82k, see [[emulator-floor-is-data-coverage]]),
so on the toy "more data" works. The REAL target is **T=512 + w0wa + TATT**: the
TATT intrinsic-alignment model adds ~5 params (a1, a2, b_ta + z-evolution) on top
of w0wa cosmology + photo-z + biases (~17-20 total), and T=512 is a very hot
sampling prior, so the parameter VOLUME explodes. Each cosmolike 3x2pt+TATT vector
is expensive to compute. So N_train is THE binding constraint -- dense coverage
needs a ridiculous number of sims, and "just add data" (the toy answer) is dead on
arrival. The asymptotic floor at unlimited data is irrelevant when you can never
afford unlimited data.

**The correction this forces (how to judge every lever).** A method must be judged
by whether it shifts the learning curve LEFT (same accuracy, fewer samples), not
by the floor at one N_train. This session judged the levers on the wrong axis:
the rescaling, the per-element weight, etc. were each tested at the ~10% subset
(10k) and called "neutral on the floor." That is the wrong test. A lever neutral
at the floor can still be a large sample-efficiency win -- which is precisely the
regime that matters when each datum is costly.

**The rescaling is the prime sample-efficiency lever.**
[[analytic-scaling-preprocessing]] removes ~79% of the broadband variance the net
would otherwise learn from data, so it should need fewer samples for a given
accuracy. Its benefit is LARGEST at small N_train (scarce data) and fades as N
grows (the net learns the variance itself). So the decisive test is the
rescaled-vs-plain LEARNING CURVE at SMALL N (2k-10k), where the curves separate --
not the 10k subset where they looked similar. Same logic for architecture /
inductive bias, physics input features / residual learning, and target
conditioning: all are candidate left-shifts of the curve.

**The experiment (the reference-figure structure).** The Dot/Linear/LSH/AFT/Latent
plot (f vs N_train/1000, log-y) is exactly a method comparison by learning curve --
the leftmost/lowest curve is the most data-efficient and is the one to scale to
T=512+TATT. Build the multi-method loop over LOG-spaced N_train including the small-N
regime; report N_target per method. Active / importance sampling (where to PLACE
the training points, not just how many) is the other axis at the real scale, since
uniform sampling of a hot high-D prior is hopeless.

**Why:** records that the goal is the learning curve's position (sample
efficiency), so future work compares methods by N-to-target on the toy testbed and
does not mistake "the toy floor is data-limited" for "the project is solved by more
data." The whole point is to make the model learn more from fewer expensive sims.
