# Parameterized time translation T(Δt) with subject-relative clocks

Spec: `docs/specs/time_translation_v1.md` (written before the run; predictions not adjusted).
Model: Qwen2.5-1.5B, CPU, layers 0, 8, 14, 20, 27 (hidden-state indices).
Grid: `prompts/time_translation_v1.json` — 8 subjects × 10 timepoints × 3 paraphrases = 240
experimental passages, plus 240 phrase-only controls (each interval phrase reattached to the same
subject's and paraphrase's t0 state description). Passages are
`[[interval: <phrase>]] [[state: <description>]]`; the **state** span is mean-pooled.
Scripts: `scripts/time_translation.py` (m1–m7), `scripts/time_translation_selector.py` (m8).
Raw numbers: `results/time_translation_measures.json`, `results/time_translation_selector.json`.
Figures: `results/figures/time_translation_*.png`.

**Grid note.** The spec listed ten Δt values but stated "nine Δt plus t0" and a 240-passage total.
I dropped **1 month** — the only value the spec's own state-writing guidance never referenced —
leaving Δt ∈ {1 day, 1 week, 6 months, 1 year, 10 y, 100 y, 1 ky, 10 ky, 1 My}.

---

## 1–2. Displacement and shared/residual decomposition

d(s, Δt) = mean_p h(s, Δt, p) − mean_p h(s, t0, p); shared(Δt) = mean_s d(s, Δt);
resid = d − shared. Fraction of Σ_s‖d‖² explained by shared:

| layer | all Δt | 1d | 1w | 6mo | 1y | 10y | 100y | 1ky | 10ky | 1My |
|---|---|---|---|---|---|---|---|---|---|---|
| 0  | 0.308 | 0.32 | 0.34 | 0.29 | 0.34 | 0.25 | 0.26 | 0.25 | 0.37 | 0.34 |
| 8  | 0.474 | 0.47 | 0.48 | 0.46 | 0.48 | 0.46 | 0.43 | 0.46 | 0.49 | 0.52 |
| 14 | **0.533** | 0.54 | 0.54 | 0.54 | 0.54 | 0.51 | 0.47 | 0.52 | 0.55 | 0.57 |
| 20 | **0.563** | 0.60 | 0.58 | 0.57 | 0.57 | 0.53 | 0.49 | 0.55 | 0.58 | 0.59 |
| 27 | 0.457 | 0.46 | 0.45 | 0.44 | 0.47 | 0.42 | 0.39 | 0.45 | 0.48 | 0.51 |

So a little over half the displacement energy in the middle of the network is a direction shared by
all eight subjects, rising from 0.31 at the embeddings to a peak at layer 20 and falling again at 27.

Mean ‖d(s, Δt)‖ at layer 14 runs 12.7 (1 d), 12.4 (1 w), 12.3 (6 mo), 13.0 (1 y), 12.4 (10 y),
12.3 (100 y), 13.3 (1 ky), 14.0 (10 ky), 14.8 (1 My) — i.e. the *total* displacement is nearly flat
in Δt, with only a ~20 % rise across nine orders of magnitude.

Leave-one-subject-out shared is essentially the same direction as the full shared:
cos(shared, shared_LOO) = 0.989–0.993 at every Δt at layer 14. The LOO version is what m8 patches.

## 3. Clock geometry

| layer | Spearman(‖shared‖, log Δt) | Pearson | mean cos adjacent Δt | mean cos distant (≥4 apart) |
|---|---|---|---|---|
| 0 | +0.183 | +0.369 | 0.731 | 0.586 |
| 8 | +0.550 | +0.753 | 0.849 | 0.611 |
| 14 | +0.467 | +0.665 | **0.888** | **0.641** |
| 20 | +0.317 | +0.482 | 0.903 | 0.659 |
| 27 | +0.517 | +0.670 | 0.878 | 0.551 |

‖shared(Δt)‖ at layer 14: 9.45, 9.13, 9.05, 9.67, 8.93, 8.54, 9.63, 10.43, 11.20. The curve is
U-shaped, not monotone: it dips to a minimum at 100 years and then climbs steeply into the
geological Δt. Pearson beats Spearman everywhere because the rise is concentrated in the last two
or three points. Adjacent-Δt cosines exceed distant ones at every layer, so the clock directions do
lie on a ordered manifold, but magnitude is not a clean function of log Δt.

## 4. Subject timescale τ(s)

Under the spec's definition (smallest Δt at which ‖resid(s, Δt)‖ reaches half its maximum over Δt),
**τ = 1 day for all eight subjects at all five layers.** The residual curves are close to flat: at
layer 14 they range over about 7–11 for every subject, so the 1-day point is already well above half
the maximum. The residual is dominated by a subject-constant offset, not by a timescale-dependent
rise, and the spec's knee statistic cannot discriminate.

A supplementary knee — smallest Δt at which ‖resid‖ reaches (1-day value + half the rise above it) —
does vary, but not in the predicted order. Layer 14: fictional_pop 6 mo, river 1 y, street 100 y,
real_pop 100 y, orchard 1 ky, mayfly 10 ky, mountain 10 ky, asteroid 1 My. The mayfly, whose
residual should saturate first, saturates nearly last; the ordering that survives is roughly
"how geological is the subject", i.e. the same ordering the shared clock has, not an inverse one.

Curves: `results/figures/time_translation_resid_curves.png`.

## 5. Cyclic return (layer 14)

| subject | ‖d(6 mo)‖ | ‖d(1 y)‖ | ratio 1y/6mo | cos(d(6mo), d(1y)) |
|---|---|---|---|---|
| orchard | 10.84 | 9.35 | **0.86** | **+0.562** |
| mountain | 12.71 | 14.17 | 1.12 | +0.858 |
| street | 12.40 | 13.24 | 1.07 | +0.827 |
| river | 12.48 | 15.41 | 1.24 | +0.801 |

The orchard is the only subject whose 1-year displacement is *smaller* than its 6-month
displacement, and the only one whose two displacements point in appreciably different directions.
The effect is stable across layers (orchard ratio 0.85–0.90, cos 0.52–0.62; mountain ratio
1.08–1.12, cos 0.73–0.88). The seasonal return is visible, but it is a ~15 % shrinkage, not the
near-cancellation the phrasing of the prediction suggests.

## 6. Real vs fictional population

cos(resid(real, Δt), resid(fictional, Δt)), layer 14:
1 d +0.44, 1 w +0.36, 6 mo +0.37, 1 y +0.52, 10 y +0.47, 100 y +0.42, 1 ky +0.45, 10 ky +0.49,
1 My +0.43. The two residuals are moderately but never strongly aligned; the minimum cosine is at
**1 week**, not at 100 years, at every layer. The maximum *distance* between the two residuals is
at 100 years at layers 8 and 14, at 6 months at layers 20 and 27, and at 1 week at layer 0.

Note that the fictional city has systematically the smallest residual norm of all eight subjects
(layer 14: 7.0–8.4 vs 8.2–10.0 for London), i.e. the invented city sits closer to the shared clock —
it is the most "generic" subject in the grid.

## 7. Phrase-only control

‖shared_exp(Δt)‖ / ‖shared_ctrl(Δt)‖ and cos(shared_exp, shared_ctrl), mean over Δt:

| layer | mean ratio | mean cos | ratio range over Δt |
|---|---|---|---|
| 0 | undefined (see below) | 0.000 | — |
| 8 | 3.09 | +0.472 | 2.48–3.74 |
| 14 | **2.82** | **+0.488** | 2.40–3.42 |
| 20 | 2.99 | +0.489 | 2.61–3.53 |
| 27 | 3.39 | +0.479 | 2.70–4.32 |

At **layer 0 the control displacement is exactly zero**: the control state span is token-for-token
the t0 state span, and Qwen's embedding layer has no positional component (RoPE is applied inside
attention), so the pooled embedding of the state span is bit-identical and d_ctrl = 0. The ratio is
therefore ∞ rather than ≈1, and cos is undefined (reported as 0). This is a property of the
architecture, not a measurement.

From layer 8 on, the interval phrase alone moves the state span by about a third of what the phrase
plus a truly advanced state description moves it, in a direction about 60° away. The control's own
decomposition is *more* shared than the experimental one (frac_shared 0.72–0.74 at layer 14 vs
0.47–0.57), which is what one expects: the phrase-only displacement has no subject-specific content
to carry.

## 8. Selector test

One paraphrase (p0), layers 14 and 8, Δt ≥ 1 year (six values), all eight subjects. Prefix = the
subject's t0 passage plus the interval phrase for Δt; the nine candidate continuations are that
subject's nine state descriptions; patch = leave-one-subject-out shared(Δt) added at every position
at the test layer; control = a random direction of the same norm (seed 0). Chance rank = 5.0.
The full run (both layers, all six Δt) took 39 min on CPU, so no restriction was needed.

Two rankings are reported: **raw** orders candidates by sum log p(state | prefix) under that
condition; **gain** orders them by logprob(patched) − logprob(unpatched), which is the house-style
statistic from `stage6_factors.py` and removes the per-candidate fluency/length offset.

| layer | raw: clock | raw: no patch | raw: random | gain: clock | gain: random |
|---|---|---|---|---|---|
| 14 | 4.65 | 4.98 | 5.06 | **3.88** | 5.65 |
| 8  | 4.85 | 4.98 | 5.04 | 4.40 | 5.96 |

Per Δt, gain ranking at layer 14 (clock / random): 1 y 4.38/3.75, 10 y 5.38/6.25, 100 y 5.50/5.75,
1 ky 3.12/6.00, 10 ky 3.25/7.00, 1 My **1.62**/5.12. The clock direction only selects at the
geological end: at 1 ky and beyond it is strongly better than chance, and at 1 y–100 y it is at or
slightly worse than chance. Layer 8 shows the same shape at reduced strength (10 ky 2.88, 1 My 3.25).

The random control is reliably *worse* than chance (5.65 at layer 14, 5.96 at layer 8), so part of
the clock-vs-random gap is the random direction actively disrupting the right answer rather than the
clock selecting it. The clock-vs-no-patch gap on the raw ranking (4.65 vs 4.98) is the conservative
version of the effect and is small.

---

## Predictions, graded

**P1 (shared clock exists) — PARTIAL (2 of 3 clauses).**
shared(Δt) explains **53.3 %** of Σ‖d‖² at layer 14, ≥ 40 % ✓.
Adjacent-Δt cosines exceed distant ones: **0.888 vs 0.641** at layer 14, true at all five layers ✓.
‖shared‖ monotone in log Δt with Spearman ≥ 0.8: **Spearman +0.467** at layer 14 (max over layers
+0.550 at layer 8) ✗. The magnitude curve is U-shaped with a minimum at 100 years.

**P2 (subject-relative knees) — FELL.**
Under the spec's τ definition every subject gives τ = 1 day at every layer, so **0 of the 7 implied
orderings are even testable**. Under the supplementary knee (half the rise above the 1-day value),
3 of 8 orderings hold at layer 14, and the ordering that emerges (fictional_pop < river <
street ≈ real_pop < orchard < mayfly ≈ mountain < asteroid) puts the mayfly near the *slow* end,
the opposite of the prediction.

**P3 (cyclic return) — PARTIAL (3 of 4 clauses).**
Orchard ‖d(1y)‖ < ‖d(6mo)‖: **9.35 < 10.84** ✓. Orchard cos(d(6mo), d(1y)) < 0.5: **+0.562** ✗
(and 0.52–0.62 across layers, so it misses at every layer). Mountain ‖d(1y)‖ ≈ ‖d(6mo)‖:
**14.17 vs 12.71, ratio 1.12** ✓. Mountain cos > 0.8: **+0.858** ✓.
The orchard is nonetheless the only one of the eight subjects that shows the return at all.

**P4 (history) — FELL.** (Stated confidence 55 %.)
Real/fictional residual cosine is **never above 0.6** at any Δt or layer (layer 14: 0.36–0.52), so
the "similar through 10 years" clause fails outright. The cosine minimum is at **1 week**, not
100 years, at every layer; maximum residual distance is at 100 years at layers 8 and 14 only.

**P5 (clock as selector) — HELD on the gain ranking, PARTIAL on the raw ranking.**
Layer 14, Δt ≥ 1 year: patched mean rank **3.88 ≤ 4.0** vs random **5.65** ✓; layer 8 effect is
smaller (**4.40** vs 5.96) ✓, as predicted. On the raw-logprob ranking the effect is weak:
4.65 patched vs 4.98 unpatched vs 5.06 random. Layer 0 was not tested, as the spec allows.

**P6 (more than the phrase) — PARTIAL (1 of 3 clauses).**
‖shared_exp‖/‖shared_ctrl‖ at layer 14 = **2.82 ≥ 1.5** ✓.
Ratio ≈ 1.0 at layer 0: **undefined/infinite** — the control's layer-0 displacement is exactly zero
because the state tokens are identical and Qwen has no additive positional embedding ✗.
cos(shared_exp, shared_ctrl) declines with depth: **+0.472, +0.488, +0.489, +0.479** at layers
8/14/20/27 — flat, not declining ✗.

---

## What fell / what is confounded

**The parameterized operator exists but its parameter is weak.** A shared clock direction is real
(53 % of displacement energy at layer 14, LOO-stable at cos 0.99, adjacent Δt more aligned than
distant Δt) and it does act as a selector for the state description at the far end of the scale
(rank 1.62/9 at 1 My). But ‖shared(Δt)‖ is not a monotone function of log Δt, and total displacement
‖d‖ grows only ~20 % across nine orders of magnitude of Δt. T(Δt) reads more like a direction with a
weak, non-monotone gain than like a genuinely parameterized operator.

**τ as specified does not measure anything.** Residual norms are nearly flat in Δt, so a half-max
threshold fires at the first point for every subject. The residual is dominated by a
subject-constant component — "which subject is being described" — rather than by a
timescale-dependent one. Any future version of this measurement must first remove the
subject-constant part of resid(s, Δt) (e.g. subtract mean_Δt resid(s, ·)) before looking for a knee.
This is the single largest methodological failure of the run.

**Confounds.**
- *Length and lexical content of the state span.* The far-Δt descriptions are systematically more
  abstract and use a shared geological/erasure vocabulary ("nothing remains", "sediment", "no trace")
  across subjects. The shared clock at 10 ky and 1 My may be largely that vocabulary, not a clock.
  The rise in ‖shared‖ at exactly the two Δt where every subject's text turns to erasure is
  consistent with this, and m8's effect appearing only at those Δt is the same worry.
- *The author wrote both the passages and the predictions.* Everything is my own prose; the mayfly's
  and the orchard's "true timescale" is my judgement of it.
- *Phrase-only control shares the interval phrase with the experimental passages*, so cos(exp, ctrl)
  ≈ 0.48 is a floor, not an independent baseline.
- *Layer 0 control is degenerate* (exactly zero displacement), so P6's layer-0 clause was untestable
  rather than false.
- *m8's random control is worse than chance* (5.65–5.96 vs 5.0): a random vector of clock norm
  disrupts the correct continuation. Part of the clock-vs-random gap is therefore disruption avoided
  rather than selection achieved; the clock-vs-no-patch gap (4.65 vs 4.98 raw) is the honest figure
  for the raw ranking.
- *The grid dropped 1 month* to reconcile the spec's internal contradiction (ten Δt listed, nine
  stated, 240 total). No prediction referred to 1 month.
- *Only Qwen2.5-1.5B, one model.* The NDIF/Gemma-2-9B secondary run was not attempted; the primary
  run consumed the available time.
