# Spec: the shared clock as a gain over the lexical floor, by depth (v1)

*Written before running. Predictions are recorded here so they can be graded, not adjusted.
Follows `docs/specs/time_translation_v1.md`, `docs/specs/subject_clocks_v1.md`, and the notes for
hours 28, 30, 31, 32, 35 (`results/notes/time_translation_v3.md` is the direct predecessor).*

## 0. Where we are and what this spec is for

A shared direction in the residual stream tracks log Δt across eight subjects (hours 28/30/31/35):
about half the displacement variance, Spearman 0.67 with log Δt, ~2.7× the displacement the interval
phrase alone causes, cos 0.94 between the v2 and v3 directions, replicated on Gemma-2-9B-it. Hour 35
removed every duration expression from the state texts and the clock barely moved — but the
layer-0 discrimination of Δt from the mean-pooled state span only fell from 1.00 to 0.72, because
a street's million-year state ("cadastres", "denudation") is written in a different register from
its one-day state ("unchanged", "identical"). That is content, not leakage, and no rewrite removes
it.

So the standing question is: **given an irreducible lexical floor, is the clock anything more than
the words?** Every number so far is a property of the layer-14 representation *including* whatever
the words alone would have given. This spec turns the measurement into a **gain over the floor as a
function of depth**, and asks whether that gain has the shape of computation.

Two facts about the existing design that this spec has to correct, stated now because they change
what "floor" and "gain" can mean:

1. **For Qwen2.5 (and Gemma-2), "layer 0" already *is* the bag of static embeddings.** `lm.residuals`
   returns `hidden_states[0]`, which in a RoPE model is `embed_tokens(ids)` with no positional term
   (Gemma-2 multiplies by √d, a scalar). Mean-pooling it over the state span is exactly the
   context-free mean-embedding readout. The requested "bag-of-embeddings control" is therefore not a
   *different* floor from layer 0 — it is numerically the same vector, and this spec uses that
   identity as a calibration check (§3.4) rather than as a control. The real lexical controls have to
   be (a) a readout with a *different* lexical basis (one-hot bag of tokens, §3.2) and (b) a forward
   pass on the *same words in destroyed order* (§2, arm B).
2. **In the hour-28/35 passages the state span follows the interval phrase**, so at any layer ≥ 1 the
   state tokens can attend to "A million years later," and Δt is decodable by copying. The
   phrase-only control bounds the *norm* of that channel (exp/ctrl ratio 2.7) but a linear readout
   does not care about norm: a small copied component is fully decodable. **A gain measured on the
   phrase-bearing passages would be inflated by attention copying and would not answer the
   question.** The primary gain arm here therefore runs the state span *alone*, with no interval
   phrase in the prompt. The phrase-bearing forms are kept as separate arms for continuity and for
   the model-supplied test (§4.3).

## 1. Question and claims

**Q.** At depth L, how much of log Δt is recoverable from the model's representation of a state
description that could not be recovered from the words of that description alone, and does that
excess have the depth profile, the structure-dependence and the interval-dependence expected of a
computation rather than of embedding statistics?

Claims kept apart:

- **C-lex (null).** The clock is the register: everything a layer-L readout knows about Δt, a
  context-free readout of the same words already knew. Gain ≈ 0 at every depth.
- **C-register (computed bag).** The model computes a "magnitude-of-change" feature from the
  vocabulary, but it is a function of the bag of words, indifferent to their order and composition.
  Gain > 0 over the static/one-hot floor, but the shuffled-order arm recovers the same gain.
- **C-compute (the claim worth having).** The gain depends on the composed description (order and
  syntax), rises then falls with depth, and is largest exactly where the register is least telling.

## 2. Passages and arms — no new text

Everything is derived mechanically from `prompts/time_translation_v3.json` (8 subjects × (t0 + 9 Δt)
× 3 paraphrases; `states[s][t][p]`, `phrases[t]`, `log10_dt_days`). **No new passages.** The v3 grid
cost ~300k tokens to write; the whole point of this spec is that the decisive controls are
permutations and deletions of text that already exists. A builder `scripts/clock_gain_build.py`
emits `prompts/clock_gain_v1.json` with four arms, each 240 prompts (t0 rows included so the
t0 = "no interval" cell exists in every arm):

| arm | prompt | rows | purpose |
|---|---|---|---|
| **A intact, state-only** | `[[state: S(s,t,p)]]` | 240 | primary gain arm: nothing but the description is visible |
| **B shuffled, state-only** | `[[state: shuffle_w(S(s,t,p))]]` | 240 | same bag of words, order destroyed |
| **C phrase + t0 state** | `[[interval: PH(t)]] [[state: S(s,t0,p)]]` | 240 | = v3 `control_prompts`; the model must supply the change |
| **D phrase + state** | `[[interval: PH(t)]] [[state: S(s,t,p)]]` | 240 | = v3 `prompts`; continuity with hour 35, and the copy-inflation estimate |

`shuffle_w`: whitespace-delimited word shuffle with `random.Random(hash((s,t,p)))`, one fixed
permutation per cell, terminal punctuation left attached to its word, first word not forced to be
capitalised (we want the model to see a word salad, not a sentence). Word-level rather than
token-level shuffling keeps the token multiset essentially identical to arm A (leading-space tokens
survive), which is what makes arm B's layer-0 vector equal to arm A's up to tokenisation noise —
a second calibration check (§3.4).

Extraction: `scripts/time_translation.py` extracts only 5 layers; this needs every layer. A thin
extractor `scripts/clock_gain_extract.py` calls `lsx.extract(..., keep_resid=False)` and stores
`roles["state"]` for **all 29 residual points** (embedding + 28 blocks) as `[29, 1536]` per prompt,
plus `roles["interval"]` for arms C/D. 960 prompts × 29 × 1536 × float32 = 171 MB, gitignored
(`*.npz`). The hour-35 log gives 872–905 s per 240 prompts on this CPU box including model load, so
budget **~55–60 min wall for all four arms** (the "10 min for 480" figure in the brief is optimistic
against the log; arm D can be dropped to save 15 min, since hour 35's D numbers already exist in
`results/time_translation_v3_fresh_measures.json`). Stacks from hour 35 are gone and must be
re-extracted regardless.

Environment: `/home/user/latent-space-exploration/.venv`, `HF_HOME=/home/user/latent-space-exploration/cache/hf`,
`HF_HUB_OFFLINE=1`, model `Qwen/Qwen2.5-1.5B`, CPU, float32, seed 0.

## 3. The estimator

### 3.1 Readout and target

Target y = log10 Δt in days (`log10_dt_days`), on the 216 Δt ≥ 1 day cells (t0 is excluded from the
regression target because it has no Δt; it is used only in §4.3). Features x_L = mean-pooled state
span at residual point L, 1536-d.

**Readout: ridge regression, leave-one-subject-out (LOSO).** Train on 7 subjects (189 points),
predict the held-out subject's 27 cells, concatenate the 8 held-out prediction sets, score. LOSO is
the right fold because the claim is a *shared* clock: a readout that only works with the subject in
the training set is a subject-identity readout, not a clock. Ridge rather than the nearest-centroid
classifier of `scripts/time_translation_discrimination.py` because a graded target wants a graded
readout and because ridge has a closed-form dual (n = 189 ≪ d) that makes the permutation null
cheap. The nearest-centroid Spearman is still reported at every layer as the continuity number with
hour 35 (§3.5).

**Score:** ρ_L = Spearman(ŷ, y) over the 216 held-out predictions, and MAE_L in grid steps
(rank of predicted Δt among the nine).

### 3.2 The floor

Three context-free readouts of the same 216 cells, same LOSO folds, same ridge machinery:

- **F-emb**: ridge on the mean static embedding of the state span's tokens
  (`model.get_input_embeddings()(ids).mean(0)`). By §0.1 this is arm A's layer-0 vector; it is
  computed independently so the identity can be checked, not assumed.
- **F-bow**: ridge on the one-hot bag of token ids over the state span (counts, L2-normalised,
  vocabulary restricted to ids that occur in the grid; dual-form ridge, so the 150k-wide
  vocabulary costs nothing). This is the floor with *no* embedding geometry at all: pure
  identity-of-words. Under LOSO it can only transfer through words that recur across subjects, which
  by v3's vocabulary-matching (≤ 2 subjects share any far-Δt content word) means mostly the near-Δt
  "unchanged/identical" register. That is exactly the register floor hour 35 identified.
- **F-bow+emb**: ridge on the concatenation, so the floor is not gameable by choosing the weaker
  of the two.

**ρ_lex = max(ρ_F-emb, ρ_F-bow, ρ_F-bow+emb).** Taking the max is deliberately conservative: the
clock must beat the best context-free readout, not a convenient one.

### 3.3 Gain — two definitions, one primary

**Primary: residual gain G_res(L).** Fit the floor readout (F-bow+emb, LOSO) and form the lexical
residual r_i = y_i − ŷ_lex,i for every cell (each prediction is out-of-subject, so r is not
optimistic). Then fit ridge from x_L to r, LOSO, and score

  G_res(L) = Spearman(r̂_L, r) over the 216 held-out points, plus R²_res(L) = 1 − Σ(r − r̂)²/Σr²
  (clipped at 0).

This is "what layer L knows about Δt that the words did not", read directly rather than as a
difference of two ceilinged numbers. Its built-in calibration is that at L = 0 the features are
(§0.1) a function of the same information the floor already used, so **G_res(0) must be ≈ 0**; a
non-zero G_res(0) means the estimator is broken (it would mean the floor ridge under-fitted its own
inputs, which a proper inner-CV λ should prevent).

**Secondary: Δρ(L) = ρ_L − ρ_lex**, and its nearest-centroid twin Δρ_nc(L) (§3.5). Reported because it
is what every earlier note would have computed, and because Spearman differences are what the
brief's numbers (0.72 at layer 0, 0.96 at layer 14) are stated in. Not primary because it is
compressed near 1: a floor of 0.75 leaves only 0.25 of headroom and a gain of 0.20 is then hard to
distinguish from noise.

**Structure gain: G_struct(L) = ρ_A(L) − ρ_B(L)**, intact minus shuffled at the same depth, and its
residualised form G_res,A(L) − G_res,B(L). This is the discriminator between C-register and
C-compute: arm B keeps every word and every embedding, and loses only composition.

### 3.4 Why this is not gameable by scale, and how layers are normalised

Residual norms grow roughly 10× from layer 0 to layer 27 in Qwen2.5-1.5B and are anisotropic.
The estimator is made scale-invariant by construction, not by hoping:

1. Per layer, centre the 240 vectors on their grand mean and divide every vector by one scalar, the
   layer's mean vector norm. (A single scalar cannot introduce or remove information; ridge with a
   CV-chosen λ is invariant to it, since λ is rescaled along with the data.)
2. λ chosen per layer and per outer fold by **inner LOSO over the 7 training subjects** on the
   grid λ ∈ {10^−3 … 10^3} × tr(Σ_train)/d. The λ grid is in units of the data's own variance, so
   no layer can win by being large.
3. Scores are rank-based (Spearman, MAE in grid steps): invariant to any monotone rescaling of ŷ.
4. Robustness variant, reported not primary: per-dimension z-scoring per layer (whitening the
   anisotropy). If the two normalisations disagree on which layer peaks by more than 3 blocks, say so.
5. **Permutation null.** For every layer and arm, 200 permutations of y *within subject* (the Δt
   labels are shuffled among a subject's 27 cells, keeping subject identity intact), refitting the
   whole LOSO pipeline. Report G and Δρ as z-scores against their null and as raw values. Within-
   subject permutation is the right null because between-subject structure (identity) is not what
   is being claimed.
6. **Calibration checks that must pass before any number is read:**
   - ρ_A(0) = ρ_F-emb to within 0.01 (they are the same vector).
   - ρ_B(0) = ρ_A(0) to within 0.03 (word shuffle preserves the token bag).
   - G_res(0) ∈ [−0.05, +0.05] on arms A, B, D.
   - Permutation-null mean of G_res within ±0.05 of 0 at every layer.

### 3.5 Continuity with hour 35

Run `scripts/time_translation_discrimination.py` unchanged on arm D at layers 0/8/14/20/27 — it
must reproduce hour 35's 0.710 (shared, L0) and 0.961 (shared, L14) to within 0.02, which proves the
re-extraction is the same object. Then run it on arm A: the difference between arm D and arm A at
layer 14 is the **copy-inflation estimate** — how much of hour 35's 0.96 was attention reading the
phrase rather than the state. This number is not a prediction target but it is the single most
important correction to the earlier notes and is reported in the first table.

## 4. What "shape of computation" means operationally

Three discriminating signatures, each with a pre-registered number, plus the model-supplied test.
Embedding statistics predict: gain ≈ 0 everywhere, or a gain that appears at layer 1–2 and stays
flat (a linear re-mixing of the same bag), and no dependence on word order.

### 4.1 S1 — depth profile: rise, mid-stack peak, decline

Compute G_res,A(L) at all 29 points. Signature of computation: G_res rises from ≈ 0, reaches its
maximum at a fractional depth in [0.3, 0.7] (blocks 8–20 of 28), and has fallen to ≤ 0.75 × its peak
by the final residual point. Signature of embedding statistics: the profile is flat, or is at ≥ 90 %
of its maximum already by block 3. A monotone rise to the last layer is *ambiguous* (many features
sharpen to the end in a 1.5B model) and is recorded as neither.

Stated now: S1 is necessary, not sufficient. Peaking mid-stack is what most linearly decodable
features do; on its own it does not distinguish C-register from C-compute. It is here because its
*absence* is informative (a flat profile kills C-compute and C-register together).

### 4.2 S2 — order dependence: intact beats shuffled

G_struct at the peak layer of arm A. C-compute: G_struct(L*) ≥ 0.08 in Spearman and the
residualised difference G_res,A(L*) − G_res,B(L*) ≥ 0.10, permutation z ≥ 3 (null: swap A/B labels
within cell, 200 draws). C-register: both within ±0.03 of zero while G_res,A(L*) is itself
significantly positive. This is the one signature that separates "the model computes a magnitude-of-
change feature from the register" from "the model reads the description". Honest expectation
(§6): this one is the likeliest to fail, because a 1.5B model on a 40-word span of technical
vocabulary can compute a lot from the bag.

### 4.3 S3 — gain where the register is least distinctive

Per-Δt floor accuracy a_lex(Δt) = 1 − MAE_lex(Δt)/8 from the F-bow+emb readout, and per-Δt gain
g(Δt) = MAE_lex(Δt) − MAE_A,L*(Δt) in grid steps. Computation predicts the model helps most where
the words help least: Spearman over the nine Δt between a_lex and g ≤ −0.5, with the three largest
g among {6 months, 1 year, 10 years, 100 years} — the middle of the grid, where the prose of
"the orchard is back in the same season" or "the street has turned over its tenants" is register-
neutral and Δt has to be inferred from *what* changed. Embedding statistics predict g flat or
proportional to a_lex (gain wherever there is signal at all).

### 4.4 S4 — the model-supplied test (arm C), and whether hour 32's null was decisive

Arm C holds the state text fixed at t0 and varies only the phrase. Hour 32's C3 (the two-timepoint
form of the same idea) found *no Δt-graded magnitude* on any subject at any layer with a 1-D
floor-referenced last-token readout, and direction-level subject specificity only at ≥ 100 y.

Is that the decisive null for the strong claim? **Not yet, for two reasons the gain framing can fix at
zero passage cost.** (i) Hour 32 read a 1-D projection at the last token with a z = 2.5 knee
criterion — a magnitude test. The strong claim is about *direction*: does the phrase, on its own,
push the fixed state along the same direction that the changed description would have produced?
That is a cosine question, not a τ question, and `time_translation.py` already computes it:
cos(shared_C(Δt), shared_D(Δt)) = 0.457 at layer 14 in the hour-35 run. Here it becomes
**cos(shared_C(Δt, L), shared_A(Δt, L))** — arm A, not D, so the reference direction contains no
phrase-copy component at all. (ii) The ridge readout trained on arm A can be applied to arm C: a
readout of *state-derived* Δt evaluated on prompts where the state never changed. If the phrase
alone moves the state span along the state-derived clock, the arm-A ridge should predict Δt on arm C
above chance, and that number cannot be lexical (arm C's state tokens are identical across Δt, so the
floor there is chance by construction; the only lexical channel is copying the phrase, which the
arm-A ridge was never trained to read).

Pre-registered numbers: transfer Spearman ρ_A→C(L*) ≥ 0.4 and mean cos(shared_C, shared_A) at
Δt ≥ 100 y, L* ≥ 0.4 → the phrase supplies part of the change along the computed clock; the strong
claim gets a partial positive and hour 32's null is superseded. ρ_A→C(L*) ≤ 0.2 **and** cos ≤ 0.25 →
hour 32's null stands as decisive: the model does not supply the change, it only reads it when it is
written down. Nothing in between is a result.

What would rescue the strong claim if S4 fails here: only decoding (piece 2 of
`subject_clocks_v1.md`, next-state likelihood), which is a different instrument and a different
spec. It is not funded here.

## 5. The subject-clock question, reopened and closed

Hour 32 established that hours 28/30/31 never tested subject clocks (instrument at its floor), then
tested the weak claim with a strong readout and found determinate τ's that were *not* subject-ordered
(1 of 9 ordered pairs), and the strong claim null in magnitude. Does the gain framing give a fair
test?

**For the weak claim (author-supplied), no, and the reason is structural, not statistical.** Hour
32's readout was lexically confounded — but confounded *in favour* of finding structure: the author
wrote the passages to embody each subject's timescale, so if any readout could recover a
subject-ordered τ from those texts, a readout that also saw the words should. It did not. Removing
the lexical channel cannot make the ordering appear. A within-subject gain (readout trained on 2
paraphrases × 9 Δt = 18 points per subject in 1536-d) is also underpowered by an order of magnitude,
and the matched-size shared control of hour 32 already showed within ≈ shared. The gain framing
would produce a number; it would not produce a test.

**For the strong claim (model-supplied), the fair test is S4 (§4.4), which is subject-agnostic.** A
subject-*specific* version — does arm C's displacement along the clock differ by subject in the
direction the subject's timescale predicts (mayfly moves at 1 day, mountain not until 10 ky)? — is
reported descriptively as the per-subject ρ_A→C(L*) and per-subject cos, with no prediction attached
and no ordered-pair scoring, because n = 27 cells per subject in the transfer set gives Spearman
confidence intervals of roughly ±0.35. If the descriptive table happens to show mayfly and the
populations at the top and mountain/asteroid at the bottom, that is a lead for a decoding spec, not a
finding.

**Decision: the subject-clock question is dropped from this line.** It is written here so the
decision is on record and not re-litigated in the next note.

## 6. Predictions (written now; graded after)

Qwen2.5-1.5B, arm A unless stated, LOSO ridge. L* = argmax_L G_res,A(L).

| # | prediction | conf. |
|---|---|---|
| P0 | All four calibration checks in §3.4.6 pass. | 0.85 |
| P1 | ρ_lex ∈ [0.65, 0.80] (hour 35's nearest-centroid 0.71 was a weaker readout; ridge on bow+emb lands higher). F-bow ≥ F-emb under LOSO by ≤ 0.05 either way. | 0.7 |
| P2 | ρ_A(L*) ∈ [0.85, 0.95]; **G_res,A(L*) ≥ 0.30**, permutation z ≥ 4; Δρ(L*) ≥ 0.12. | 0.65 |
| P3 | S1: L* at block 8–20; G_res at the final point ≤ 0.75 × G_res(L*). | 0.5 |
| P4 | S2: G_struct(L*) ≥ 0.08 and residualised difference ≥ 0.10 at z ≥ 3. | **0.4** |
| P5 | S3: Spearman(a_lex, g) over nine Δt ≤ −0.5; largest g in the 6 mo–100 y block. | 0.5 |
| P6 | Copy inflation: ρ_D(14) − ρ_A(14) ≥ 0.03 (hour 35's 0.96 was partly the phrase). | 0.6 |
| P7 | S4: ρ_A→C(L*) ≥ 0.4 and far-Δt cos(shared_C, shared_A) ≥ 0.4. | **0.35** |
| P8 | Gemma gate (piece 2, only if run): L*_frac within ±0.15 of Qwen's; G_res(L*) ≥ 0.20; S2 sign agrees. | 0.6 |

Written expectation, so it can be checked against the outcome rather than reconstructed: P2 holds,
P3 holds, **P4 fails or is marginal** (the shuffled arm recovers most of the gain), P5 holds weakly,
P7 fails. That outcome is C-register: the clock is a real mid-stack computation over the vocabulary
of change, not a reading of the narrative, and not just the words either.

## 7. Kill condition and outcomes

**Kill (close the line, "the clock is vocabulary"):** on arm A, **max_L G_res,A(L) < 0.10, or its
permutation z < 2 at every layer**, with P0 passing. Then no depth of the model reads more about
Δt from the description than a one-hot bag of its tokens plus their static embeddings, the hour-35
"clock" numbers were the register, and the line is wound up with hour 35 as its last entry.
No Gemma run, no v4 grid, no decoding spec.

**Partial kill (close the narrative reading, keep the result):** G_res,A(L*) ≥ 0.10 at z ≥ 3 but
S2 fails (G_struct within ±0.03 of 0) → C-register. Record as: "a computed magnitude-of-change
feature over the bag of words, peaking at block L*, invariant to word order." That is a finding
and it is the end of this line; the S4/decoding strong-claim question does not get another passage
written for it.

**Go (C-compute):** P2, P3 and P4 hold. Then piece 2 runs, and the next spec is about *what* is
being composed (which tokens' contributions the intact-minus-shuffled gain lives on), not about
more grids.

## 8. Pieces and cost

**Piece 1 — alone, first: Qwen, all arms, all layers, all measures.**
`scripts/clock_gain_build.py` (arms A–D from v3, ~40 lines), `scripts/clock_gain_extract.py`
(29-layer extraction, ~50 lines; 960 prompts, ~60 min wall, or 720 prompts / 45 min without arm D),
`scripts/clock_gain.py` (floors, LOSO ridge with inner-CV λ, residual gain, permutation nulls,
S1–S4, calibration checks, figures: gain-vs-depth for arms A/B with permutation bands; per-Δt
floor vs gain; arm C transfer and cosine by depth; the hour-35 continuity table). Ridge in dual form:
per layer per fold one 189×189 solve per λ; 29 layers × 8 folds × 7 λ × 3 arms plus 200
permutations reusing the fold Gram matrices — under 10 min on 4 cores. Outputs
`results/clock_gain_v1_{measures.json,stacks.npz}`, `results/notes/clock_gain_v1.md`, a RESULTS.md
entry. Budget: ~150k agent tokens, ~1.5 h wall. Must grade P0–P7 and state which of §7's three
outcomes obtained before anything else is started.

**Piece 2 — gated on "Go" in §7: Gemma-2-9B-it via NDIF (`.venv312`), arms A and B only** (480
prompts; hour 31's extractor `scripts/ndif_time_translation_extract.py` adapted to the new grid),
nine residual points {0, 5, 9, 14, 20, 26, 31, 36, 42}, the same measure script with `--layers`.
Floors recomputed with Gemma's own tokenizer and embedding table. ~15 min NDIF, ~100k tokens. Grades
P8 only. Not run if piece 1 lands in kill or partial-kill.

Not funded: new passages of any kind; register-matched rewrites (the hour-35 note already shows why
they would falsify the content); decoding tests of the strong claim; any within-subject τ estimate.

## 9. Known weaknesses, stated now

1. **The shuffled arm is not a perfect "bag" control.** A causal model on shuffled words still
   composes adjacent pairs; some structure survives. G_struct therefore underestimates the true
   order-dependence, biasing S2 toward the partial-kill outcome. That bias is in the conservative
   direction and is accepted.
2. **The floor is a linear readout of the bag.** A nonlinear context-free readout (an MLP on the
   bag) might close some of the gain. Not fitted here: with 189 training points it would mostly fit
   noise, and a linear floor is what "the same readout on static embeddings" means. Reported as a
   limitation, not patched.
3. **n = 216 cells, 8 folds.** The per-Δt analysis (S3) has 24 points per Δt; its Spearman over nine
   Δt is a coarse statistic and is pre-registered at ≤ −0.5 precisely because anything weaker would
   be noise.
4. **Population subjects carry numeric world-facts that track Δt** (hour 35, §5.3). They are in the
   LOSO folds like everyone else; the bag-of-tokens floor sees the digits, so this leak is in the
   floor as much as in the model and does not bias the gain.
5. **Mean pooling, single seed, single model until the gate.** As in every note in this line.
