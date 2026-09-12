# Time translation on Gemma-2-9B-it (hour 30, scale check)

Companion to `time_translation.md` (v1, Qwen2.5-1.5B) and `time_translation_v2.md` (vocabulary-
matched far-Δt states, Qwen2.5-1.5B). Same v2 grid (`prompts/time_translation_v2.json`, 480
passages: 240 experimental + 240 phrase-only control), same measurements 1-7
(`docs/specs/time_translation_v1.md`), no selector test (measurement 8), run on Gemma-2-9B-it via
NDIF at layers 9, 20, 31 (state span mean-pooled). Extraction: `scripts/ndif_time_translation_extract.py`.
Measurement: `scripts/time_translation.py prompts/time_translation_v2.json --model
google/gemma-2-9b-it --layers 9,20,31 --suffix gemma --stage measure`.

## Prediction (written before the run)

- Shared variance fraction at Gemma layer 20 lands in **0.4-0.6** (vs Qwen layer-14 v2: 0.545),
  Spearman(‖shared‖, log Δt) **≥ 0.6** (vs 0.68).
- The residual curves acquire subject structure: **at least 4 of 8 subjects have τ > 1 day**
  (vs all 8 flat at τ = 1 day in both Qwen runs), with **mayfly and street among the shortest**
  τ and **mountain and asteroid among the longest**.

Graded against the actual numbers below.

## Qwen layer 14 (v2) vs Gemma layer 20

| quantity | Qwen-1.5B L14 (v2) | Gemma-9B L20 |
|---|---|---|
| share of Σ‖d‖² explained by shared(Δt), all Δt | 0.545 | 0.478 |
| same, far Δt only (1 ky / 10 ky / 1 My) | 0.58 / 0.57 / 0.59 | 0.50 / 0.52 / 0.49 |
| Spearman(‖shared‖, log Δt) | 0.68 | 0.68 |
| adjacent / distant cos of shared | 0.88 / 0.59 | 0.87 / 0.61 |
| ‖shared_exp‖ / ‖shared_ctrl‖ (mean over Δt) | ~3.1 (1ky/10ky/1My: 3.19/3.08/3.12) | 1.67 |
| τ(s), half-max | 1 day, all 8 subjects | 1 day, all 8 subjects |
| orchard ‖d(1y)‖/‖d(6mo)‖, cos | 0.86, 0.56 | 0.80, 0.56 |
| mountain ‖d(1y)‖/‖d(6mo)‖, cos | (unreported at L14 in v2 note) | 0.88, 0.58 |
| real vs fictional residual cos, min / where | (0.49–0.59 range, 100y→1My) | 0.21 at 6 months (min); 0.61 at 1 week (max) |

(Qwen column reproduces `time_translation_v2.md`'s L14 row; full per-Δt numbers are in that note
and in `results/time_translation_measures.json`.)

## τ per subject, Gemma layers 9/20/31

All eight subjects hit half-max residual norm at the smallest Δt tested (1 day) at every layer —
identical to both Qwen runs. There is no knee to report; the full curves
(`results/figures/time_translation_gemma_resid_curves.png`) are noisy but flat: residual norm at
Δt = 1 day is already comparable to or larger than at Δt = 1,000,000 years for every subject
(e.g. layer 20: street 41.8 at 1 day vs 77.5 at 1 My, mayfly 52.6 vs 65.4, mountain 51.5 vs 55.8,
asteroid 71.2 vs 51.1 — no monotone growth, no ordering by subject identity).

## Grading the prediction

- **Shared fraction 0.4–0.6 at layer 20: held.** 0.478, inside the predicted band (Qwen L14: 0.545).
- **Spearman ≥ 0.6: held.** 0.683, matching Qwen's 0.68 almost exactly.
- **≥ 4 of 8 subjects with τ > 1 day: fell.** 0 of 8. The residual curves show no more structure
  at 9B than at 1.5B — τ is degenerate at every layer tested (9, 20, 31).
- **mayfly/street shortest, mountain/asteroid longest: fell (vacuously).** With every τ tied at
  1 day there is no ordering to grade; the qualitative structure the prediction hoped for did not
  appear.

## Reading

The shared clock generalizes across scale and architecture: a 9B instruction-tuned model
(Gemma-2) shows essentially the same shared-variance fraction (~0.48 vs 0.55), the same
Spearman correlation with log Δt (0.68 both), and the same adjacent/distant cosine gap (~0.87 vs
0.61) as the 1.5B base model, at a layer in the same relative depth (20/42 vs 14/28, both ~50%).
The phrase-only control ratio is smaller at Gemma (1.67 vs ~3.1 at far Δt for Qwen), so a larger
share of Gemma's shared displacement could in principle be carried by the interval phrase alone
— worth flagging as a partial confound, though the ratio still exceeds 1 at every Δt (1.51–1.90),
so some non-phrase shared signal remains.

What did not appear, at 9B any more than at 1.5B: subject-relative timescales. The prediction
that going to a larger, instruction-tuned model would let mayfly-scale and mountain-scale
subjects diverge in their residual dynamics was not borne out — the residual curves are flat and
noisy for all eight subjects at all three layers tested. This raises the confound named in the
Qwen notes from a model-size question to (tentatively) an architecture/task question: state-span
mean-pooling under a fixed passage template may simply not carry a subject-specific "how much has
this decayed" signal in either model family, rather than the signal existing but being too weak
at 1.5B to detect. Distinguishing those two readings would need either a model an order of
magnitude larger again, or a different probe than the residual-norm/half-max construction used
here (e.g. a per-subject direction fit on held-out Δt, not just its norm).

## Confounds / caveats

- Layers are not perfectly depth-matched: 20/42 (48%) for Gemma vs 14/28 (50%) for Qwen — close
  but not identical; layer-9 and layer-31 rows in the full `results/time_translation_gemma_measures.json`
  bracket layer 20 similarly to how Qwen's own layer sweep behaves.
- No selector test (measurement 8) was run per the task's scope, so there is no evidence here on
  whether Gemma's shared direction functions causally as a selector the way Qwen's did (mean rank
  3.62 vs 4.83 random in the v2 Qwen run).
- Single author, single model pair, one run each; no seed variation.
- m6 (real vs fictional) at Gemma shows a *dip* at 6 months (cos 0.21, both min-cos and max-dist)
  rather than the far-Δt separation seen in Qwen — plausibly noise given how flat/noisy the whole
  residual structure is at this scale, not a robust finding.

