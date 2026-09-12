# Broken instruments: what this project measured wrong, how it found out, and what it now checks

Over 38 logged stages (`RESULTS.md`) this project found four of its own measurement instruments
broken: two on the first day (hours 3 and 6), two on the second (hours 32 and 36), plus a failed
calibration on the second day (hour 38). Each had already produced numbers that were read as
findings; each discovery withdrew or qualified them. A reader deciding whether to trust the repo's
positive results should see this record first: it shows what kind of error this pipeline produces,
how long each survived, and which standing results rest on checks that have never been run. Every
number is quoted from `RESULTS.md` or the named note; where the record is silent, that is stated.

## 1. A cross-talk rank pinned at chance by construction (hour 6)

**What it was.** Test C in `scripts/stage5_factors.py`: under an era patch, rank each of the three
voice variants among the three by absolute log-probability, era fixed, against the unpatched
ranking; likewise era under a voice patch.

**What it appeared to show.** In `results/stage5_qwen1.5b_factors_l14.json`, both
`C_voice_under_era` and `C_era_under_voice` average exactly 2.0 patched and 2.0 unpatched over 108
rows each. Read naively: the era patch leaves the voice ordering undisturbed; the factors are
orthogonal. That is the result the hypothesis wanted.

**What it actually was.** Ranking each of three candidates among the same three yields a
permutation of {1, 2, 3} for any log-probability vector; the mean is 2 whatever the patch did.

**How it was caught.** By reading the metric's definition at write-up time: hour 6 records
"averages to exactly 2 by construction, so the 'readout under patch' numbers in the log are
meaningless." No control caught it; the base rank was sitting at 2.0 in the log unread.

**What it invalidated.** Nothing published: the metric was replaced before its numbers were quoted
(commit `cf8d821`). The replacement is a variance decomposition of the 3×3 gain matrix under a
single-factor patch: era 0.59 on-target / 0.14 other / 0.27 residual; voice 0.65 / 0.17 / 0.18;
random direction 0.27 / 0.23 / 0.50.

**Standing check.** Cross-talk is reported only beside its random-direction row, which fixes what an
uninformative patch looks like. Since hour 36 a uniform cross-talk matrix is read as a fingerprint of
degenerate gains, not a finding about factors (§4).

## 2. Best-layer selection on held-out data (hours 1–3)

**What it was.** Stage 3 fit a role-to-role transfer on seven domains and scored the eighth by
`role_rank` (chance 3.5), reporting both the all-layers mean and "each pair's best layer".

**What it appeared to show.** Shared-offset transfer 2.35 over all layers and 2.01 at the best
layer; affine 3.03 and 2.45. Hour 2 called the column "optimistic" but kept reporting it.

**What it actually was.** Selecting the best of 29 layers on the same held-out data that scores it.
Hour 3 pushed a constant predictor (the mean target, no source information) through the same
selection: **3.5 → 2.4 by selection alone.**

**How it was caught.** By a baseline. The same hour found the constant mean-target baseline scored
1.07 on its own at 42 training examples, so every stage-3 number before hour 3 was a role-identity
signature plus position, not a source-to-target relation.

**What it invalidated.** All best-layer numbers from hours 1–2 and, with the constant-baseline
finding, all pre-hour-3 relation claims. The relation survived only after role-centering (3.01 vs
null 3.48–3.54, hour 3) and at forty domains (2.21, hour 16).

**Standing check.** "Best-layer numbers are dropped from here on" (hour 3); `WRITEUP.md` states
they are never reported. Later sweeps report the full curve with a per-layer null (hour 5; hour 38
at all 29 layers with permutation z at each). Caveat: hour 38 still evaluates S2 and S4 at
`L* = argmax_L G_res,A(L)` over the scoring data; pre-registered, but still selection on held-out.

## 3. A residual norm at its own noise floor (hours 28, 30, 31; caught at 32)

**What it was.** The subject-clock probe of `docs/specs/time_translation_v1.md`:
resid(s, Δt) = d(s, Δt) − mean_s d(s, Δt), with d built from three-paraphrase means of a
1536-dimensional mean-pooled state span; τ(s) = first Δt at which ‖resid‖ reaches half its maximum.

**What it appeared to show.** τ = 1 day for every subject at every layer (hour 28, v1 grid);
unchanged on the vocabulary-matched v2 grid (hour 30); 0 of 8 subjects with τ > 1 day on
Gemma-2-9B-it at layers 9/20/31 (hour 31). Three published negatives across two models:
"subject-relative timescales do not appear", "remain absent", "still absent".

**What it actually was.** Spec `docs/specs/subject_clocks_v1.md` §0, written by the planner before
any run: if paraphrases within a cell scatter with per-dimension SD σ, the difference of two
three-paraphrase means has expected norm ≈ σ·√(2d/3) ≈ 32σ at d = 1536, whether or not the described
state changed. The v2 residuals ran 7–11 at layer 14 for every subject and Δt against ‖d‖ 12–16:
the signature of a floor. A one-dimensional projection has floor σ·√(2/3) ≈ 0.8σ, ~40× better
signal-to-noise from the same data. And half-max on a flat noisy curve fires on the first point, so
τ = 1 day was the guaranteed output. Hour 32 confirmed it: floor F = 7.2 at layer 14,
residual-to-floor ratio 1.12 at 1 day and 1.39 at 1 My, 7 of 8 subjects inside 0.8–1.3 for every
Δt ≤ 1 y at four of five layers, nothing above 1.8×. **The same pipeline run end to end on Gaussian
noise returns ratios averaging 0.95 and "no signal" everywhere**: the earlier curves are
indistinguishable from noise.

**How it was caught.** By arithmetic before running anything, then confirmed by the noise run.

**What it invalidated.** The three negatives of hours 28, 30 and 31, withdrawn as untested. The
replacement probe (floor-referenced 1-D readout, SNR 5.3–7.2) found a determinate τ for 7 of 7
changing subjects but no subject ordering (1 of 9 predicted pairs), and that result is itself
lexically confounded (§5).

**Standing check.** Every residual is reported beside its noise floor; the estimator is run on
synthetic noise before real data (hour 32; hour 38 §2.1 self-test with pure-noise features,
ρ = −0.059 against null +0.008 ± 0.065); knees are replaced by floor-referenced thresholds.

## 4. A remote patching harness that wrote into batch row zero (hour 34; caught at 36)

**What it was.** `scripts/ndif_factors.py`, the NDIF selector battery, patched with
`B[l].output[0][:] = B[l].output[0] + v` and ranked with `rank = 1 + #{gain[c] > gain[target]}`.

**What it appeared to show.** Hour 34, spec `docs/specs/scale_vs_tuning_v1.md`: theme lens rank
1.22 / 1.11 / 1.06 of 3 on Llama-3.1-8B / 70B / 70B-Instruct, far under chance 2.0, but the
random-direction control at 1.14–1.28 on all three. Logged as an unresolved blocker in the same
entry.

**What it actually was** (`results/notes/random_control_diagnosis.md`). Under transformers ≥ 4.54 a
Llama, Gemma or Qwen decoder layer returns a bare tensor `[batch, seq, d]`, so `output[0]` is batch
row 0, not the hidden states. Hour 34 ran with `NDIF_CHUNK=9`, all nine candidates in one padded
batch, so every patch, factor and random alike, touched one text and left eight identical to base.
The rank formula scores a tie as rank 1, so the target scores 3 only when it is the moved text, 1
row in 9: expected rank **11/9 = 1.22 for any direction whatsoever**, and (8·1 + 9)/9 = 1.89 for the
composed test (observed 1.81). Fingerprint: all 48 cross-talk rows in each hour-34 file read
0.25/0.25 to seven decimals, where hour 13's Gemma file ranges 0.007–0.846. Confirmed on 8B with
three texts in one batch: as-shipped gains [−7.47, 0, 0]; whole-tensor patch [−7.47, −2.23, −6.79];
output shape [3, 18, 4096]. Reproduced locally on Qwen2.5-1.5B: row-0 patching gives random
1.31/1.14. A third bug: left padding under a mask that assumed right padding.

**How it was caught.** Only because the random control came out equal to its treatment. **The
no-patch baseline that would have caught it instantly had never been run**: under the shipped rank
it scores 1.00 (doing nothing looks like a perfect selector), under mid-rank ties 2.00. The
treatment number was not alarming on its own: 1.22 is both the artifact's fixed point and the true
8B era rank after the fix.

**What it invalidated.** All five hour-34 selector files, three cross-talk matrices, the "random
control collapses on every Llama" finding, and the layer-sweep conclusion, so hour 33's depth
confound is reopened. Standing: decodability 0.89/0.89/0.92 (no patching), the 405B no-go, hour 13
(GPT-J returns a tuple; Gemma ran at batch 1), all local batteries, and all generation-level NDIF
results, which trace one prompt per job and are correct by accident. Corrected 8B at layer 10: era
1.22, theme 1.33, random 2.11, no-patch 2.00, composed 2.03 of 9.

**Standing check.** Commit `fe31abb`: a tuple-or-tensor `resid()` helper, mid-rank ties, a
mandatory no-patch arm, right padding. Hour 37 added an assertion that fails if at most one
candidate's score moved; no-patch read exactly 2.00 in all four hour-37 runs. Not fixed: six other
NDIF scripts (`ndif_generate`, `ndif_shift`, `ndif_commutator`, `ndif_recompose_gen`,
`ndif_recompose_sweep`, `ndif_absential_probe`) keep the `output[0][:]` idiom at batch 1, which the
note calls "a live trap"; and the local selector scripts (`stage5_factors.py`, `stage6_factors.py`,
`time_translation_selector.py`) still rank 1-on-ties.

## 5. Same family, not broken instruments: the lexical floor and a failed calibration

**Lexical floor (hours 32, 35, 38).** The measurement was sound and the stimulus was not. Hour 32
found Δt-discrimination of 1.00 at layer 0, before any computation, because the v2 state texts
restate the interval ("One day later the mayfly is dead"); the leak check flags 221 of 240 v2 state
spans. Hour 35's v3 grid (0 of 240 flagged) kept the clock (variance share 0.507, cosine 0.94–0.95
to the v2 direction) but layer-0 discrimination fell only to 0.72: far-interval prose uses a
different register from near-interval prose. Hour 38 quantified the copying: with the interval
phrase removed, hour 35's layer-14 discrimination of **0.961 falls to 0.522** under the same
centroid readout (0.890 → 0.812 under ridge). The clock was re-posed as gain over a lexical floor
(ρ_lex = 0.728) and closed as a computed register detector, not a clock.

**Calibration failure (hour 38).** Spec `docs/specs/clock_depth_gain_v1.md` §3.4.6 required its
primary statistic to read G_res(0) ∈ [−0.05, +0.05], because layer 0 is the floor's own input
(verified: cosine 0.999999999 to the static-embedding bag). Observed −0.323 / −0.321 / −0.314 on
three arms, permutation null −0.586 at layer 0: residualising then refitting over-subtracts when the
two readouts' errors correlate. P0 was graded failed, G_res reported in full (peak 0.297 at layer
24, z = 7.6), and an unbiased rank-partial companion put beside it (null mean ≤ 0.013 at every
layer; peak 0.566 at layer 13). The companion's layer-0 values, +0.066 / −0.047 / +0.074, sit
marginally outside the band on two of three arms.

## What the four have in common

Two claims were put to the record; the first holds, the second does not, and a third emerges.

*Every instrument produced a plausible number.* Confirmed, and stronger: each failure mode was a
fixed point that coincided with a meaningful reading. 2.0 read as orthogonality; 2.01 as transfer;
τ = 1 day as "no subject clock"; 1.22 as a sharp lens, and it was the true value.

*Three of four were caught by a control or baseline.* Not confirmed. Two were: best-layer (a
constant predictor through the same selection) and batch-row (a random direction equal to
treatment). The noise floor was caught by arithmetic before running. The cross-talk rank was caught
by reading its definition; its base rank was already 2.0 in the log and was not read as a tell.
Count: two by control, two by reasoning about construction.

*The missing arm was always the same arm.* The check that would have caught each one immediately
measures the instrument with the treatment removed: a constant predictor, Gaussian noise, no patch.
Each was run only after the fact. The noise floor survived three hours, two models and a vocabulary
rewrite; the batch-row bug would have survived indefinitely had the control not failed.

## What currently rests on instruments that have never been independently checked

- **No local selector battery has a no-patch arm** (hours 4–11, 23, 28–32), and their scripts still
  rank 1-on-ties. Their defence is a random control at chance (1.83–2.25). Hour 8's tense control is
  the exception: factor 1.03 of 2, random 1.44 against chance 1.5, flagged at hour 36, not re-run.
- **Every generation-level NDIF result** (hours 12–14, 19, 22, 27, 29, 31, 33) and hour 13's Gemma
  battery use the batch-row idiom at batch 1 with no plumbing assertion. Hour 29's dose-response
  (0.27 → 0.84 across 1×–3×) is the only evidence those patches land.
- **Hour 31's Gemma extraction** batches six passages per job through `tracer.invoke` with
  `output[0]`. The record does not say whether that path was re-checked after hour 36.
- **The abstraction ladder** (hours 15, 26) has no permutation null or random-feature control in the
  record; the broad corpus already shrank its effect from 2.9× to 1.9×, and the merge test (12 of
  15, 13 of 15) has no null distribution.
- **The shared-clock selector numbers in `WRITEUP.md` claim 10** (3.9 vs 5.7, 3.6 vs 4.8) come from
  the v1/v2 grids in which 221 of 240 state spans restate the interval; never re-run on v3.
- **Hour 38's headline** rests on a companion statistic introduced after the pre-registered one
  failed calibration, evaluated at a layer chosen by argmax on the scoring data.
