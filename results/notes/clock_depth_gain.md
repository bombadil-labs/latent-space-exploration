# The shared clock as a gain over the lexical floor, by depth — Qwen2.5-1.5B (piece 1)

Spec: `docs/specs/clock_depth_gain_v1.md`, piece 1 only (Qwen2.5-1.5B, arms A–D, all 29 residual
points). Piece 2 (Gemma) is **not** run; its gate is graded in §9. Predictions were read from the
spec before any number was computed and are graded unchanged in §8.

**One-line answer.** The clock is not only vocabulary — a state description with no interval phrase
is read for Δt well above what a bag of its own tokens and their static embeddings can give
(calibrated gain 0.57, permutation z 7.6) — but the excess does **not** have the shape of
composition: it is 70–85 % recovered from the same words in shuffled order, it is already at half
strength after one block, and the model does not supply the change when only the phrase varies
(arm C transfer 0.15, direction cosine 0.08). Outcome: **between "partial kill" and "Go", and on
the spec's own gate, no Go.** Piece 2 does not run.

---

## 1. What was run

| step | artefact | wall |
|---|---|---|
| build 4 arms from the v3 grid, no new text | `prompts/clock_gain_v1.json` (960 prompts) | <1 s |
| 29-layer extraction, one prompt per forward pass | `results/clock_gain_v1_stacks_{A,B,C,D}.npz` (gitignored, 240 MB) | 29.3 min (A 916 s, B 453 s, C 208 s, D 179 s; A/B ran against two other jobs on the box) |
| context-free lexical features (tokenizer + embedding table only) | `results/clock_gain_v1_lex.npz` (2 281-id vocabulary) | 40 s |
| hour-35 continuity, script unchanged | `results/clock_gain_v1_discrim_{A,D}.json` | 5.3 + 5.4 min |
| floors, LOSO ridge, gains, nulls, S1–S4 | `results/clock_gain_v1_measures.json`, 2 figures | 2.6 min |

Estimator exactly as specified: dual-form ridge, kernel-centred, leave-one-subject-out over the 216
Δt ≥ 1 d cells, λ chosen per layer per outer fold by inner LOSO over the 7 training subjects on
{10⁻³…10³} × tr(Σ_train)/d, per-layer grand-mean centring and division by the layer's mean vector
norm, rank-based scores, 200 within-subject permutations of y refitting the whole pipeline (floor
included). Mid-rank on ties everywhere — never rank-1-on-ties (hour 36).

---

## 2. Instrument checks first (hour 32 / hour 36 discipline)

**2.1 Synthetic self-test of the estimator, run before it touches a stack** (`--selftest`, logged
into the measures JSON). Pure-noise features: ρ = −0.059 with a permutation null at +0.008 ± 0.065.
Features carrying the target: ρ = 0.974. The same features multiplied by 37 and passed through a
random 64×64 linear map: ρ = 0.973 (scale/rotation invariance, |Δ| = 0.001). Residual gain of a
floor over *itself*: −0.030; of a strictly more informative layer: +0.356; of a pure-noise layer:
−0.006. **PASS.**

**2.2 Continuity with hour 35 (spec §3.5), the "is this the same object" check.** The unchanged
`time_translation_discrimination.py` on the re-extracted arm D:

| layer | 0 | 8 | 14 | 20 | 27 |
|---|---|---|---|---|---|
| arm D shared | **0.710** | 0.922 | **0.961** | 0.977 | 0.960 |
| hour 35 | 0.7101 | — | 0.9607 | — | — |
| arm A shared (state only) | 0.706 | 0.435 | **0.522** | 0.586 | 0.793 |

Arm D reproduces hour 35 to three decimals, so the re-extraction is the same object.

**2.3 No arm returns a constant.** Every per-layer series varies smoothly and the three arms differ
from each other (§4). The hour-36 failure mode (a measured quantity pinned regardless of condition)
does not appear.

---

## 3. The calibration check the spec demanded — and what it caught

Spec §3.4.6, four checks. The layer-0 identity of §0.1 was verified numerically first:
cos(arm-A layer-0 vector, independently computed mean static embedding) = **0.999999999** (min over
240 cells 0.99999991). Layer 0 *is* the bag of static embeddings.

| check | required | observed | verdict |
|---|---|---|---|
| ρ_A(0) = ρ_F-emb | ≤ 0.01 | 0.7218 vs 0.7218, diff **0.000** | pass |
| ρ_B(0) = ρ_A(0) | ≤ 0.03 | 0.7046 vs 0.7218, diff **0.017** | pass |
| G_res(0) ∈ [−0.05, +0.05] on A, B, D | — | **−0.323 / −0.321 / −0.314** | **fail** |
| permutation-null mean of G_res within ±0.05 at every layer | — | worst **−0.586** (L0), −0.22 to −0.30 elsewhere | **fail** |

So **P0 fails**, and the failure is in the spec's primary estimator, not in the data. Diagnosis,
since the brief asks for investigation rather than reporting:

- The floor is *not* mis-fitted. Regressing y on the floor's out-of-fold predictions gives slope
  **1.000** (the attenuation slope of ŷ on y is 0.482, which is what prediction error does, not
  miscalibration), so ŷ_lex is orthogonal to its own residual by construction. The residual r still
  carries corr 0.719 with y, because the floor explains only about half of y's spread.
- The bias comes from **error correlation between the two stages**. G_res residualises y on the
  floor and then refits a readout whose features *overlap the floor's inputs* (at L = 0 they are the
  floor's emb block exactly). The second readout's errors are correlated with the floor's errors, so
  subtracting the floor over-subtracts. The size of the offset depends on that error correlation,
  not on information: the permutation null of G_res sits at −0.586 at L0 and drifts to −0.22 by L28.
- Consequence: **the absolute scale of G_res is not interpretable, and even its z-score is
  layer-dependent in its baseline** (at L0, G_res = −0.32 scores z = +4.7 against a null at −0.59).
  Reporting G_res raw at a single layer would be the hour-32 mistake in a new costume.

**Correction applied (added, not substituted).** Alongside the spec's G_res, the same predictions are
scored with the **rank-partial correlation** G_part(L) = partial Spearman of the layer-L LOSO
prediction with y controlling for the floor's LOSO prediction. It has no error-correlation bias:
when the layer readout *is* the floor readout it is exactly 0. Its calibration:

| check | observed |
|---|---|
| G_part(0) on A / B / D | **+0.066 / −0.047 / +0.074** (band ±0.05; A and D marginally outside, B inside) |
| permutation-null mean of G_part, worst over all layers and arms | **0.013** (pass) |

G_part is therefore reported as the calibrated companion and is used wherever the two disagree; the
spec's G_res is reported in full so nothing is hidden. Both are computed from identical fits.

---

## 4. The floor, and the gain curve by depth (arm A: state text only, no interval phrase)

**Floor (spec §3.2, 216 LOSO-held-out cells).** F-emb ρ = **0.722** (MAE 1.41 grid steps),
F-bow ρ = **0.706** (1.41), F-bow+emb ρ = **0.728** (1.40). **ρ_lex = 0.728.** F-bow and F-emb differ
by 0.016. Floor residual sd 1.87.

**Arm A by depth** (29 residual points; L0 = embeddings, L28 = output of the last block):

| L | 0 | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 | 21 | 23 | **24** | 26 | 28 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ρ_A | .72 | .76 | .74 | .77 | .79 | .80 | .82 | .82 | .80 | .82 | .79 | .77 | .79 | .79 | .79 | .77 |
| Δρ | −.01 | .03 | .01 | .04 | .06 | .07 | .09 | .09 | .07 | .09 | .06 | .04 | .06 | .06 | .06 | .05 |
| **G_res** | −.32 | −.21 | −.08 | .00 | .10 | .15 | .18 | .17 | .17 | .20 | .16 | .15 | .24 | **.30** | .27 | .19 |
| G_res z | 4.7 | 3.6 | 1.8 | 3.3 | 4.4 | 5.3 | 5.9 | 5.9 | 5.7 | 6.1 | 5.5 | 5.6 | 6.8 | **7.6** | 7.2 | 6.3 |
| **G_part** | .07 | .32 | .30 | .40 | .46 | .52 | .55 | **.57** | .55 | .56 | .50 | .44 | .52 | .54 | .53 | .48 |
| G_part z | 0.6 | 3.8 | 3.6 | 4.7 | 5.4 | 6.2 | 6.7 | 6.9 | 6.7 | 6.8 | 6.0 | 5.4 | 7.0 | 7.6 | 7.6 | 7.0 |
| MAE | 1.41 | 1.35 | 1.37 | 1.34 | 1.28 | 1.20 | 1.16 | 1.12 | 1.26 | 1.15 | 1.25 | 1.27 | 1.19 | 1.19 | 1.15 | 1.25 |

**Peak.** G_res peaks at **L* = 24 (0.297, z = 7.6, fractional depth 0.857)**; G_part peaks at
**L = 13 (0.566, z = 7.6, fractional depth 0.464)** and is within 5 % of that value everywhere from
L9 to L26. The per-dimension z-scored robustness variant (spec §3.4.4) reproduces the G_res curve and
its peak at L24 exactly, so the disagreement is between the two *gain definitions*, not between two
normalisations; the spec's ">3 blocks" rule applies to the latter and is satisfied.

**Arm B (same words, order destroyed).** ρ_B peaks at 0.76; G_res peaks at 0.20 (L24), G_part at
0.47 (L18). The shuffled arm recovers **67 % of the G_res peak and 83 % of the G_part peak**.

**Arm D (phrase + state, the hour-35 form).** ρ_D reaches 0.92, G_res 0.45, G_part 0.83, MAE 0.72 —
far above arm A at every depth, which is the copy channel, not the clock (§6).

---

## 5. The four signatures, graded

**S1 — depth profile (rise, mid-stack peak, decline): FAIL, in both directions.**
On G_res: L* = 24, fractional depth 0.857, outside the required [0.3, 0.7]; the decline criterion is
met (final/peak = 0.64 ≤ 0.75). On G_part: L* = 13, fractional depth 0.464, inside the band, but
final/peak = 0.85 > 0.75 — no decline. Neither curve is "flat" by the spec's definition (G_part at
block 3 is 52 % of peak, not ≥ 90 %). The honest reading is that arm A's gain **rises very fast —
57 % of the G_part peak is present after a single block — then plateaus across two thirds of the
stack and does not fall.** That is closer to the spec's *embedding-statistics* signature ("appears at
layer 1–2 and stays flat") than to the computation signature, and it is recorded as neither by the
letter of the rule.

**S2 — order dependence (intact beats shuffled): FAIL.**
At L* = 24: G_struct = ρ_A − ρ_B = 0.791 − 0.706 = **0.076** (required ≥ 0.08); residualised
difference **0.099** (required ≥ 0.10) at label-swap z = **1.21** (required ≥ 3); the same difference
on the calibrated statistic is 0.090 at z = 1.24. Swap-null sd 0.074 (G_res) and 0.069 (G_part).
All three land just under their thresholds and none is significant. Note the difference is also
*not* within ±0.03 of zero, so this is not a clean C-register verdict either: order buys a small
positive amount that this design cannot separate from zero. Spec §9.1 already warned the estimate is
biased low (a causal model still composes adjacent pairs in shuffled text); that bias is in the
conservative direction and is not enough to rescue the signature.

**S3 — gain where the register is least distinctive: PARTIAL (first clause passes, second fails).**
Spearman(a_lex, g) over the nine Δt = **−0.787** (required ≤ −0.5): the model helps most exactly
where the words help least. But the three largest gains are **1 000 000 y (+0.54), 1 day (+0.50),
100 y (+0.42)** — the two *ends* of the grid plus one middle point, not the pre-registered
{6 mo, 1 y, 10 y, 100 y} block. Per-Δt floor MAE runs 2.29 (1 d), 1.88, 1.38, 0.88, 0.79, 1.08, 1.17,
1.08, 2.08 (1 My) and arm A's 1.79, 1.67, 1.00, 1.17, 1.04, 0.67, 0.83, 1.04, 1.54; at 1 y and 10 y
the layer readout is *worse* than the floor (g = −0.29, −0.25). So the gain lives at the ends, where
the register is ambiguous, and the middle of the grid — the case the spec argued was the interesting
one — is where the floor already wins.

**S4 — the model-supplied test (arm C): FAIL, and it is the pre-registered decisive null.**
ρ_A→C(L*) = **0.150** (within-subject permutation z = 1.87, null mean −0.001) and mean
cos(shared_C(Δt), shared_A(Δt)) over Δt ≥ 100 y = **0.078**. Both are inside the spec's
"hour 32's null stands as decisive" region (≤ 0.2 **and** ≤ 0.25). The cosine never exceeds 0.146 at any depth for any Δt ≥ 100 y. **The model reads the change when it is written down; it does not supply it.**

*Descriptive only, no prediction attached (spec §5):* per-subject ρ_A→C at L* is real_population 0.86,
street 0.70, mountain 0.68, asteroid 0.57, orchard 0.53, mayfly 0.50, river 0.32, fictional_population
0.27 (mean 0.55), while per-subject far-Δt cosines are 0.18, 0.04, 0.02, −0.01, 0.11, 0.11, 0.01,
0.05. The pooled transfer is only 0.15 because the per-subject offsets do not agree. This says the
phrase does move each subject's fixed t0 state span in a Δt-graded way *within* that subject — which
is attention copying the phrase into the state span, exactly the channel arm A was designed to
remove — but not along the state-derived clock direction and not on a shared scale. It is a lead for
a decoding spec, not a finding, and it does not change the S4 verdict.

---

## 6. Copy inflation — the correction to the earlier notes

The difference between the phrase-bearing arm D and the state-only arm A at layer 14:

| readout | arm D | arm A | inflation |
|---|---|---|---|
| nearest-centroid shared (hour-35 statistic) | 0.961 | 0.522 | **0.439** |
| LOSO ridge ρ | 0.890 | 0.812 | **0.078** |
| LOSO ridge ρ at L* = 24 | 0.907 | 0.791 | 0.116 |

**P6 holds.** Under the weak nearest-centroid readout that hour 35 used, nearly half of the famous
0.96 was the model reading the interval phrase rather than the state description; under a proper
ridge readout the inflation is 0.08. Both numbers matter: the hour-28/30/31/35 figures were computed
with the centroid readout, so the larger correction is the one that applies to them. The state-only
arm-A curve is also *non-monotone* under the centroid readout (0.71 at L0, 0.44 at L8, 0.52 at L14,
0.79 at L27), i.e. that readout is simply weak in mid-stack geometry; the ridge readout shows no such
dip. Anywhere the two disagree, prefer the ridge.

---

## 7. Kill condition

Kill requires max_L G_res,A < 0.10 **or** its permutation z < 2 at every layer, *with P0 passing*.
Observed max G_res,A = **0.297** at z = **7.6** (calibrated: max G_part = 0.566 at z = 7.6), so the
kill threshold is not met on either statistic — and P0 did not pass, so the condition's own
precondition is unmet. **Verdict: not killed.** No depth-of-the-model claim can be reduced to "a bag
of its tokens plus their static embeddings": the model's representation of a description with no
interval phrase carries substantially more about Δt than the best context-free readout of the same
words (ρ_lex 0.728; the gain is significant at every layer from 1 to 28).

Partial kill (C-register) requires G_struct within ±0.03 of zero while G_res,A(L*) ≥ 0.10 at z ≥ 3.
The second half holds emphatically; the first does not (G_struct = 0.076–0.099). **The result sits
between "partial kill" and "Go" and is not cleanly either.** The defensible statement is:

> A real, large, computed gain over the lexical floor, present from the first block, flat across the
> middle of the stack, 70–85 % reproducible from the same words in scrambled order, largest at the
> extremes of the Δt grid, and not accompanied by any ability to supply the change from the interval
> phrase alone.

---

## 8. Predictions, graded

| # | prediction | outcome | grade |
|---|---|---|---|
| P0 (0.85) | all four calibration checks pass | ρ_A(0)=ρ_F-emb ✓, ρ_B(0)−ρ_A(0)=0.017 ✓, G_res(0)=−0.32 ✗, null mean −0.59 ✗ | **FAIL** (2/4; both failures are the same estimator bias, §3) |
| P1 (0.7) | ρ_lex ∈ [0.65, 0.80]; F-bow vs F-emb within 0.05 | 0.728; |0.706 − 0.722| = 0.016 | **HOLD** |
| P2 (0.65) | ρ_A(L*) ∈ [0.85,0.95]; G_res,A(L*) ≥ 0.30 at z ≥ 4; Δρ(L*) ≥ 0.12 | 0.791; 0.297 at z 7.6; Δρ 0.06 (max over depth 0.09) | **FAIL** (z clause holds, all three level clauses miss; G_res by 0.003) |
| P3 (0.5) | S1: L* in blocks 8–20 and final ≤ 0.75 × peak | G_res: L*=24, final/peak 0.64. G_part: L*=13, final/peak 0.85 | **FAIL** on both statistics, for opposite halves of the criterion |
| P4 (0.4) | S2: G_struct ≥ 0.08 and residualised ≥ 0.10 at z ≥ 3 | 0.076 / 0.099 / z 1.21 | **FAIL** (as the spec's own written expectation anticipated) |
| P5 (0.5) | S3: Spearman(a_lex,g) ≤ −0.5 and largest g in the 6 mo–100 y block | −0.787 ✓; largest g at 1 My, 1 day, 100 y ✗ | **PARTIAL** |
| P6 (0.6) | copy inflation ρ_D(14) − ρ_A(14) ≥ 0.03 | 0.078 (ridge); 0.439 (centroid) | **HOLD** |
| P7 (0.35) | S4: ρ_A→C(L*) ≥ 0.4 and far-Δt cos ≥ 0.4 | 0.150 and 0.078 — inside the "null stands" region | **FAIL**, decisively |
| P8 | Gemma gate | not run (§9) | — |

The spec's own written expectation was "P2 holds, P3 holds, P4 fails or is marginal, P5 holds weakly,
P7 fails, outcome C-register". P4, P5 and P7 came out as written. P2 and P3 did not: the gain is
there but the readout is weaker than predicted and the depth profile is a plateau, not a peak.

---

## 9. Piece 2 (Gemma) gate: **NO GO**

Spec §7: piece 2 runs only if P2, P3 and P4 all hold. **P2 fails, P3 fails, P4 fails.** Piece 2 is
not run and, by §7, the line does not get a v4 grid or a decoding spec out of this result either.
What would change the decision is not another model but a better-posed S2: the order-dependence
question is the live one, and this design measured it at z = 1.2 with a swap-null sd of 0.07 on
216 cells — i.e. it was underpowered for the effect size it found (~0.09), not decisive against it.

---

## 10. What is still confounded

1. **The floor is a linear readout of the bag** (spec §9.2). A nonlinear context-free readout could
   close part of the gain; with 189 training points it was not fitted. The gain is "over a *linear*
   lexical floor".
2. **G_res, the spec's primary statistic, is biased negative by an amount that depends on the error
   correlation between the layer readout and the floor readout** (§3). Every G_res number here
   should be read as an ordering, not a level; G_part is the level. This is the instrument finding of
   the run and it applies retroactively to any future use of the §3.3 definition.
3. **Word shuffling is not a bag control** (spec §9.1): a causal model still composes adjacent word
   pairs, so S2 underestimates true order dependence. The 0.076–0.099 it found may be real and this
   design cannot say.
4. **Arm A's "no phrase" is not "no context"**: the state span is still the whole prompt, so
   sentence-level statistics the floor cannot see (length, punctuation density, clause count)
   remain available to every layer ≥ 1 and are part of the measured gain. Token count per span was
   recorded but not regressed out; that is the first thing a follow-up should do, because it could
   explain a gain that is present after one block and flat thereafter.
5. **Population subjects carry numeric world-facts that track Δt** (spec §9.4). The bag-of-tokens
   floor sees the digits, so this is in the floor as much as in the model — but real_population is
   also the top subject in the arm-C transfer table (0.86), which is consistent with digits rather
   than clocks.
6. **Arm C's state tokens can attend to the phrase**, so the per-subject transfer of §5 is a copy
   measurement, not a supply measurement. The pooled number and the cosine are the ones that bear on
   S4.
7. **Single model, single seed, mean pooling, one paraphrase set, author-written grid** — as in
   every note in this line. The 8 subjects are the unit of generalisation and there are 8 of them.
8. **`RESULTS.md` was not updated** although spec §8 asks for an entry: the task brief forbids
   editing it. The entry belongs in whatever the next session writes there.

## Files

- `scripts/clock_gain_build.py`, `scripts/clock_gain_extract.py`, `scripts/clock_gain_lex.py`,
  `scripts/clock_gain.py`, `scripts/clock_gain_run.sh`
- `prompts/clock_gain_v1.json` (960 prompts, four arms, derived mechanically from v3)
- `results/clock_gain_v1_measures.json` (every number in this note), `results/clock_gain_v1_discrim_{A,D}.json`
- `results/figures/clock_gain_depth.png`, `results/figures/clock_gain_s3_s4.png`
- `results/clock_gain_{extract,lex,discrim_D,measure}.log`
- stacks (`results/clock_gain_v1_stacks_{A,B,C,D}.npz`, `results/clock_gain_v1_lex.npz`) are
  gitignored; `scripts/clock_gain_run.sh` regenerates everything downstream of extraction.
