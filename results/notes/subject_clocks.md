# Subject-relative clocks, piece 1 (activations)

Spec: `docs/specs/subject_clocks_v1.md` (§§1–5, piece 1 of §8). Qwen2.5-1.5B, CPU, seed 0,
layers 0/8/14/20/27. 504 passages extracted (C1 240, C3 264), ~130 tokens each.
Wall time: build 1 s, extraction 625 s (1.24 s/passage, incl. model load ~40 s), measure stage
~4 min per run (two runs: the second added the Δt-centred dominance addendum). Agent context ≈ 60k
tokens. Files: `scripts/subject_clocks_build.py`, `scripts/subject_clocks.py`,
`scripts/subject_clocks_report.py`, `prompts/subject_clocks_v1.json`,
`results/subject_clocks_measures.json`, `results/subject_clocks_stacks.npz` (gitignored),
`results/figures/subject_clocks_{floor,ycurves_C1,ycurves_C3,ycurves_C2,kappa,resolv,
crosssubject,pertoken}.png`.

Headline: **the hour-28/30/31 residual was its own paraphrase noise floor (ratio 1.12 at layer 14,
7/8 subjects inside [0.8, 1.3] for Δt ≤ 1 y).** With the floor-referenced 1-D readout the
author-supplied signal is enormous (r = 6.4 σ) but **not subject-ordered**: every subject's τ
collapses to 1 day–1 week except the two populations, 1 of 9 pre-registered ordered pairs holds,
and the subject-agnostic shared displacement predicts Δt as well as the subject's own centroids
(Spearman: within 0.77 vs shared 0.78 at layer 14; **1.00 vs 1.00 at layer 0**, i.e. the C1 readout is lexical).
The model-supplied constructions (C3, C2) carry no Δt-graded magnitude at all, but their
Δt-dependent component *is* subject-specific in direction at the far Δt (Δt-centred diagonal
dominance D = 1.7 / 4.0 at layer 14, p < 0.001).

## 1. Grid built (§1)

`prompts/subject_clocks_v1.json`, derived from `prompts/time_translation_v2.json`.
C1 = 216 Δt rows + 24 t0 rows (null phrase, reworded t0 state, null phrase alternating by p);
C3 = 216 Δt rows + 48 null rows (both null phrases × 8 × 3). Total 504, as specified.
Decoding prefixes (264) and candidates (240) for piece 2 are emitted by the same builder.
Roles: `interval0`, `state0`, `interval`, `state`.

## 2. §3.1 Diagnosis of the old probe — the residual *is* the floor

C1, mean pool, d(s,Δt,p) = state − state0 of the same passage; resid = d̄(s,Δt) − mean_s d̄;
F = √((tr Σ̂(s,Δt) + tr Σ̂(s,t0))/3) over the 3 paraphrases.

Mean over 8 subjects (layer 14): ‖d̄‖ = 13.3, ‖shared‖ = 10.6, ‖resid‖ = 8.1, **F = 7.2,
ratio = 1.12** at 1 day; at 1 My ‖resid‖ = 9.7, F = 7.0, ratio = 1.39. The v2 hour-28 numbers
(‖d‖ 12–16, ‖shared‖ 9–12, resid 7–11 at layer 14) are reproduced to within a point, so this is
the same probe.

| layer | ratio @1 day | ratio @1 My | subjects with ratio ∈ [0.8,1.3] for all Δt ≤ 1 y |
|---|---|---|---|
| 0 | 0.97 | 1.13 | 7/8 |
| 8 | 1.07 | 1.32 | 7/8 |
| 14 | 1.12 | 1.39 | 7/8 |
| 20 | 1.12 | 1.42 | 7/8 |
| 27 | 1.15 | 1.51 | 6/8 |

Per-subject at layer 14 (ratios, 1 day → 1 My): street 1.08→1.59, mountain 1.19→1.36,
orchard 1.13→1.51, mayfly 1.52→1.50, asteroid 1.10→1.40, river 1.07→1.44,
real_population 0.94→1.20, fictional_population 0.96→1.10. The mayfly is the only subject above
1.3 at near Δt. Nothing anywhere exceeds 1.8× its floor.

**So the earlier "flat residual curves" carried ≈ 1 bit of signal-to-noise per cell and half-max
of such a curve is 1 day by construction. The hours-28/30/31 conclusion "subject clocks absent"
was untested, exactly as §0 argued.** A sanity run of the whole measure stage on Gaussian noise
(same keys, d = 16) returns ratios 0.53–1.37 (mean 0.95), τ = "no signal" for every subject and
κ ≈ 0 — the floor formula is calibrated.

## 3. §3.2/§3.4 Floor-referenced τ (1-D readout along u_s^(−p))

u_s fit leave-one-paraphrase-out on the Δt ≥ 1 ky displacements of that construction;
y = ⟨d, u_s^(−p)⟩; τ per the spec's estimator (z = 2.5, baseline = the null/t0 row).

**C1-last, layer 14** (τ, P/σ, LOO-refit τ set):

| subject | τ | P/σ | LOO τ set | predicted τ |
|---|---|---|---|---|
| mayfly | 1 week | 6.00 | {no signal ×3} | ≤ 1 day |
| orchard | 1 day | 7.23 | {1 year, no signal ×2} | 6 months |
| street | 1 day | 6.37 | {1 week ×3} | 10 years |
| real_population | 1000 years | 5.84 | {1000 y, no signal ×2} | 10 years |
| fictional_population | 1 day | 6.85 | {1 week ×2, no signal} | 10 years |
| river | 1 day | 6.63 | {1 day ×3} | 100 years |
| mountain | 1 day | 5.26 | {no signal ×3} | 10 000 years |
| asteroid | **no signal** | 3.68 | {1 day, no signal ×2} | no signal |

C1-mean at layer 14 gives a different but equally unordered set (street 100 y, orchard 100 y,
mayfly 1 y, mountain 1 day, river 1 day, both populations 1000 y, asteroid 1 day).
C1-last at layer 8: street 1 week, mountain 1 day, orchard 1 day, mayfly 6 months, river no signal,
real 1000 y, fictional 1 year, asteroid no signal. Layer 20 is close to layer 14.
**Layer 0 (pure lexical control): τ = 1 day for all 8 subjects.**

**C3-last (model-supplied): τ = "no signal" for 8/8 subjects at every layer** (|P/σ| ≤ 1.24 at
layer 14, ≤ 1.6 at 20). **C2 (expectation at the phrase's last token): τ = 1 day for 8/8 subjects,
P/σ = 9–26** — but that is the null-phrase-vs-interval-phrase lexical contrast, identical in kind
for every subject, not a subject-appropriate expectation.

y curves in σ units, C1-last layer 14 (null, 1 d … 1 My):
street 0, 3.3, 4.0, 2.6, 2.7, 2.8, 4.2, 6.7, 5.6, 6.9; mountain 0, 4.2, 3.7, 4.2, 4.4, 4.0, 4.5,
5.3, 5.5, 5.1; real_population 0, 2.5, 2.3, 1.7, 2.1, 1.6, 1.5, 6.2, 6.0, 5.3; asteroid 0, 3.1,
2.9, 3.5, 2.7, 3.6, 2.8, 4.0, 3.5, 2.8. Only the two populations show a late step; everything else
is up at 1 day and stays there.

## 4. §3.2 Cross-subject matrix and diagonal dominance

Pre-registered D(Δt) (column-standardised, 10⁴ label permutations), layer 14:
C3 D = 3.32–3.39 at **every** Δt, p = 0.000 at 9/9; C2 D = 1.43 (1 d) → 3.73 (10 ky), p = 0.000 at
9/9. At layer 0 both are exactly 0 (p = 1.000), as the spec predicted.

This statistic as written cannot distinguish a clock from subject identity: in C3 the displacement
is dominated by a Δt-*independent* subject-specific component (the repeated t0 text), and D is flat
in Δt. **Addendum (not pre-registered):** the same matrix on Δt-centred displacements (each
subject's mean-over-Δt removed), so only the Δt-dependent part remains:

| | 1 d | 1 wk | 6 mo | 1 y | 10 y | 100 y | 1 ky | 10 ky | 1 My |
|---|---|---|---|---|---|---|---|---|---|
| C3 D_c (L14) | −1.13 | −1.74 | −1.48 | −1.36 | −0.33 | **1.21** | **1.52** | **1.57** | **1.73** |
| C2 D_c (L14) | −3.59 | −3.97 | −2.92 | −2.97 | −0.89 | **2.26** | **3.90** | **4.13** | **4.04** |

p < 0.001 at the four bold cells (4/9 Δt) for both constructions at layers 8, 14, 20 and 27; 0/9 at
layer 0. Since u_s is fit on the far Δt, the sign pattern is expected; what is *not* automatic is
that the far-vs-near contrast is aligned with the subject's own axis and not with other subjects'
(the readout is cross-validated over paraphrases, and a shared direction with subject-specific
magnitude would give D_c ≈ 0). Read conservatively: the interval phrase's interaction with the
preceding subject text is subject-specific in *direction* at Δt ≥ 100 y, while carrying no
subject-ordered *magnitude* or timing.

## 5. §3.3 Trajectory alignment κ (C1-last, layer 14)

κ(s,Δt) vs the subject's own 1 My state, shared-LOSO direction projected out (null, 1 d … 1 My):
street +0.10/0.31→0.51, mountain −0.28/0.18→0.28, orchard −0.18/0.22→0.30, mayfly −0.26/0.07→0.08,
asteroid −0.06/0.27→0.34, river −0.11/0.29→0.34, real_population −0.22/0.06→0.16,
fictional_population −0.18/0.19→0.32. Every subject jumps from a null row at or below 0.1 to its
plateau at 1 day and creeps upward thereafter; no subject shows the "alignment reached late" shape
the subject-clock hypothesis predicts for the mountain or the river. The asteroid — whose text is
written unchanged — has among the *highest* κ, which is a direct sign that κ here tracks the
shared "second state after an interval phrase" geometry, not subject-specific change.

## 6. §3.4 Resolvability blocks (C1-last, layer 14)

Leave-one-paraphrase-out nearest centroid, 6 trials/pair, resolved iff 6/6.

| subject | first Δt resolved from t0 (h-last / y) | pre-registered | block agreement (h / y) | fraction of pairs resolved |
|---|---|---|---|---|
| mayfly | 6 months / 6 months | 1 day | 0.67 / 0.78 | 0.28 |
| orchard | 1 day / 1 day | 6 months | 0.63 / 0.50 | 0.42 |
| street | 1 day / 100 years | 10 years | 0.54 / 0.54 | 0.42 |
| real_population | 100 years / 1000 years | 10 years | 0.69 / 0.69 | 0.40 |
| fictional_population | 1 week / 1000 years | 10 years | 0.74 / 0.66 | 0.48 |
| river | 1 day / 1 day | 100 years | 0.59 / 0.36 | 0.36 |
| mountain | 1 day / 1000 years | 10 000 years | 0.61 / 0.66 | 0.38 |
| asteroid | 1 day / 1 day | never | 0.84 / 0.89 | 0.14 |

Agreement ≥ 70 % for 2/8 subjects (h-last: asteroid, fictional_population). The asteroid's high
agreement is trivial (almost nothing resolves, and almost all its labelled pairs are
"should-not-resolve"). Only 28–48 % of pairs resolve at all with n = 3, so the matrix is sparse.

## 7. §3.5 Discrimination

Spearman(predicted grid index, true index) over 27 held-out points per subject (MAE in grid steps):

| layer | within-subject | shared LOSO transfer | matched-size shared (18 vectors, 20 draws) |
|---|---|---|---|
| 0 | **1.00** | **1.00** | **1.00** |
| 8 | 0.72 | 0.65 | 0.55 |
| 14 | 0.77 | 0.78 | 0.63 |
| 20 | 0.79 | 0.81 | 0.66 |
| 27 | 0.87 | 0.79 | 0.71 |

Layer 14 per subject (within / shared / matched): street 0.78/0.87/0.69, mountain 0.73/0.74/0.55,
orchard 0.88/0.86/0.65, mayfly 0.69/0.71/0.67, asteroid 0.46/0.78/0.40, river 0.86/0.90/0.60,
real_population 0.89/0.72/0.64, fictional_population 0.90/0.66/0.70. within − shared > 0 for only
3/8 subjects; MAE 0.9–2.4 grid steps.

**Layer 0 is the decisive line in this table.** At the embedding, with no context at all, both
predictors score a perfect 1.00 — because the v2 state texts restate the interval phrase inside the
state span ("*One day later* the mayfly is dead", "*Ten thousand years* of resurvey…"). The whole
C1 discrimination result is therefore available lexically, and the necessary-not-sufficient test of
§3.5 carries no information about representations in this grid.

## 8. Per-token C3 (layer-wise)

SD over Δt of the per-token y (mean over subjects), first 5 tokens of the repeated state vs the
remaining 19: layer 8 0.141 vs 0.090; layer 14 0.251 vs 0.196; layer 20 0.68 vs 0.37;
layer 27 3.43 vs 1.62; layer 0 exactly 0. The phrase's effect on the repeated state is
concentrated in the first few tokens and decays, at every layer that has any.

## 9. §5 Power rule — which branch applies

r = median over the 7 changing subjects of P_s/σ_s, C1-last: **r(layer 8) = 5.40,
r(layer 14) = 6.37, r(layer 20) = 5.22.** Per subject at layer 14: street 6.37, mountain 5.26,
orchard 7.23, mayfly 6.00, river 6.63, real_population 5.84, fictional_population 6.85.

**r ≥ 4.1 → branch 1: the existing grid is adequate at n = 3; piece 2 must NOT write new
paraphrases.** A knee, if one existed, would be localisable to a single grid point with these
three paraphrases. (The spec's pre-run expectation was r ≈ 3–8 and a 0.4 chance of the rule
firing; it did not fire.) Note the caveat: P_s is inflated by the lexical restatement of the
interval phrase inside the state span, so this is a power statement about the readout as
constructed, not about a clean state representation.

## 10. Predictions graded

- **P1 (diagnosis, conf. 0.75) — HELD.** Ratio ∈ [0.8, 1.3] for Δt ≤ 1 y for 7/8 subjects at
  layer 14 (mayfly the exception at 1.35–1.52); mean ratio 1.12. The earlier residual curves were
  the paraphrase floor.
- **P2 (weak claim, conf. 0.6) — FELL (partial).** Held: 7/7 changing subjects get a determinate τ
  on C1-last at layer 14; the asteroid is "no signal"; the mayfly is 1 week. Fell: **1 of 9**
  testable ordered pairs holds (only mayfly < real_population; the ordered set was
  mayfly < {street, both populations} < river < mountain and orchard < {river, mountain}), and the
  resolvability blocks match the pre-registered table for **2 of 8** subjects at ≥ 70 % of pairs
  (needed 5). The LOO-refit τ sets are unstable for 5 of 8 subjects. So the probe failure of
  hours 28–31 is confirmed *as a probe failure* (P1), but replacing the probe does not produce a
  subject-ordered clock.
- **P3 (strong claim, activations, conf. 0.6) — FELL, with a twist.** Predicted D(Δt) significant
  at ≤ 2 of 9 Δt: observed 9/9 for both C3 and C2 at layers 8–27 (the pre-registered statistic is
  confounded by Δt-independent subject identity); on the Δt-centred addendum 4/9, still > 2.
  Predicted C2 τ "no signal" for ≥ 6 of 8: observed τ = 1 day for 8/8 (a lexical
  null-phrase-vs-interval-phrase contrast). Held: the per-token effect is concentrated in the first
  5 tokens and decays. The intended *conclusion* — "the 1.5B base model carries no subject-specific
  expectation" — is contradicted in direction (subject-specific far-Δt geometry at the phrase
  token) and confirmed in magnitude (no Δt-graded, subject-ordered expectation anywhere: C3 τ is
  "no signal" 8/8).
- **P4 (decoding, conf. 0.5) — NOT RUN** (piece 2).
- **P5 (discrimination, conf. 0.8) — FELL.** Within-subject ≥ 0.6 holds for 7/8 at layer 14, but
  shared transfer is 0.78 (needed ≤ 0.4) and matched-size shared 0.63 (needed ≤ 0.35); within
  exceeds shared for only 3/8 subjects. The "necessary" condition for a subject clock is *not met*.
  The prediction's own rationale ("it should pass on vocabulary alone") is what killed it: the
  vocabulary is shared across subjects, so the shared predictor gets it too, and layer 0 already
  scores 1.00.
- **P6 (layers, conf. —) — PARTIAL.** Zero at layer 0 by construction for C3 and C2 (D = 0.00,
  per-token SD = 0.00) ✓. But the model-supplied effects peak at layers 20–27, not 14–20
  (C2 D_c max 4.13 at L14, 4.67 at L20, 5.01 at L27; per-token SD grows monotonically with depth),
  and the C1 effects are already saturated at layer 8.

## 11. Gemma gate (§7) and what should happen next

Gate condition (i), decoding B_s, is piece 2's. Gate condition (ii), "C2/C3 diagonal dominance
reaches p < 0.05 at ≥ 3 Δt", **fires**: 9/9 Δt on the pre-registered statistic and 4/9 on the
Δt-centred addendum, at layers 8/14/20/27. Under the spec as written, Gemma-2-9B-it (NDIF,
layers 9/20/31) is therefore authorised for the C1/C3 extraction and the decoding test.
**It was not run here** (out of scope for piece 1).

My recommendation, recorded so it can be graded later: the firing is weak evidence. The
pre-registered form of the gate is confounded by subject identity and should not be counted; the
centred form is the real result and it shows direction-level subject specificity with no
subject-ordered timing. Before spending Gemma budget, piece 2's decoding B_s should be the
deciding statistic, and any Gemma run should use a grid whose Δt-state texts do **not** restate the
interval phrase (see below).

Power rule for piece 2, in one line: **r(layer 14) = 6.37 ≥ 4.1 → the grid is adequate at n = 3; do
not write new paraphrases.**

## 12. What is still confounded

1. **The interval phrase is restated inside the state text.** The v2 texts begin "One day later
   the mayfly is dead", "Ten thousand years of resurvey…". So the C1 `state` span literally names
   Δt. Layer-0 discrimination is a perfect 1.00 for both within-subject and shared predictors, and
   layer-0 τ is 1 day for every subject: the entire C1 readout is available lexically before a
   single block runs. Every C1 number above (τ, P/σ, resolvability, discrimination, r) must be read
   as "lexical + whatever the model adds", and this is the single biggest reason the τ's collapse
   to 1 day — the phrase's words are already there at Δt = 1 day. A clean v3 would strip the
   leading time expression from every Δt state text; that rewrite is the obvious next experiment
   and it is cheap (240 texts, no new model runs beyond re-extraction).
2. **C2's τ is phrase identity, not expectation.** The null phrases ("At that same moment," /
   "Just then,") differ lexically from every interval phrase, so E(Δt) − Ē(null) is large at every
   Δt for every subject. The C2 τ = 1 day result says nothing about world knowledge. Only the
   Δt-centred cross-subject matrix removes this, and it is the statistic to trust.
3. **The pre-registered diagonal dominance measures subject identity.** In C3 the repeated state is
   the subject's own t0 text, so d is subject-specific at every Δt including the null rows; D is
   flat in Δt at 3.3. The permutation null (over subject labels) does not touch this.
4. **u_s is fit on the far Δt.** The far cells therefore have a guaranteed positive projection.
   The cross-validation is over paraphrases only, so P_s (and hence r) is optimistically biased;
   the LOO-refit τ sets in §3 are the honest spread and they are unstable.
5. **n = 3 paraphrases** give σ_s with 20 df and the resolvability matrix a 6-trial-per-pair
   resolution (only 28–48 % of pairs resolve). The permutation nulls are the trustworthy
   statistics; the SE-based thresholds are rough.
6. **The two null phrases are Δt = 0 but not content-free.** They are averaged for the baseline and
   alternated by paraphrase index in the t0 rows, so the t0-row variance mixes paraphrase scatter
   with the null1/null2 difference; this inflates σ_s slightly and makes τ conservative.
7. **Nothing here tests the strong claim's content.** Whether the model *expects* the mayfly to die
   is a decoding question (piece 2). The activation side can only say that the expectation token's
   subject-specific geometry exists at far Δt and carries no subject-ordered timing.
8. **Stacks are not archived.** `results/subject_clocks_stacks.npz` (119 MB, float16) is
   gitignored; re-extraction takes 10.5 minutes.
