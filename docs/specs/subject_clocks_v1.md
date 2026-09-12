# Spec: subject-relative clocks, a fair test (v1)

*Written before running. Predictions are recorded here so they can be graded, not adjusted.
Follows `docs/specs/time_translation_v1.md` and the notes for hours 28, 30, 31.*

## 0. Why the previous three runs could not have seen a subject clock

Hours 28/30/31 measured resid(s, Δt) = d(s, Δt) − mean_s d(s, Δt) with d built from **3-paraphrase
means of a 1536-d mean-pooled state span**, and defined τ as the first Δt at which ‖resid‖ reaches
half its own maximum. Two things follow from that construction alone:

1. **The residual norm is a noise floor, not a curve.** If paraphrases within one (s, Δt) cell
   scatter with per-dimension SD σ, the difference of two 3-paraphrase means has expected norm
   ≈ σ·√(2d/3) ≈ 32σ at d = 1536 — regardless of whether the described state changed. The v2
   curves at layer 14 run 7–11 for every subject at every Δt (mean ‖d‖ 12–16, ‖shared‖ 9–12).
   That is the signature of a floor. A 1-D projection onto a chosen direction has floor σ·√(2/3)
   ≈ 0.8σ: the same data, read along one axis, has ~40× better signal-to-noise than its norm.
2. **Half-max-of-max has no floor.** On any flat noisy curve the first point exceeds half the
   maximum, so τ = 1 day is the guaranteed output. It is not evidence of anything.

So "the subject-relative clock is absent" has not been tested. This spec tests it, and separates
two claims that the earlier specs ran together:

- **Weak claim (author-supplied timescale).** Given the *written* Δt-state, the model's
  representation of "how changed is this subject" saturates at a subject-specific Δt. The
  passages were written to embody this (the mayfly is dead by day 2, the mountain intact until
  10 ky), so a positive result shows the model reads paraphrase-invariant state, not that it owns
  a clock.
- **Strong claim (model-supplied timescale).** Given only the t0 state and the interval phrase,
  the model *expects* subject-appropriate change: after "the mayfly … One week later," it
  predicts death; after "the mountain … One week later," it predicts no change. This is the only
  reading under which "subject-relative clock" is a fact about the model. It is tested here
  by decoding (next-state likelihood) and by an expectation readout at the phrase's last token,
  neither of which sees the Δt-state text.

## 1. Grid and passages

Reuse `prompts/time_translation_v2.json` (8 subjects, t0 + 9 Δt, 3 paraphrases, `states[s][t][p]`,
`phrases[t]`, `log10_dt_days`). A builder script derives `prompts/subject_clocks_v1.json` with
three constructions, all in **two-timepoint form** so that the state being read always follows the
subject's own t0 description (the single-passage form of v1/v2 put the phrase *before* any mention
of the subject, so nothing at the phrase could be subject-specific):

- **C1 pair** (author-supplied change), 8 × 9 × 3 = 216 + 24 t0-repeats = 240:
  `[[interval0: At first,]] [[state0: S(s,t0,p)]] [[interval: PH(Δt)]] [[state: S(s,Δt,p)]]`
  with the same paraphrase index on both sides. The Δt = t0 row uses the null phrase (below) and
  a *different* paraphrase of the t0 state (p' = p+1 mod 3) as `state`, so "no change, reworded"
  is a cell of the grid.
- **C3 control pair** (model-supplied, token-aligned), 8 × 9 × 3 = 216:
  `[[interval0: At first,]] [[state0: S(s,t0,p)]] [[interval: PH(Δt)]] [[state: S(s,t0,p)]]`
  — the t0 text repeated verbatim after the interval phrase. Identical text at every Δt, so any
  Δt-dependence in the second `state` span is the model's, and per-token differences are defined.
- **C2 expectation** = the `interval` role's last token in C1 (or C3; identical under causal
  attention). No extra passes.
- **Null phrases** (new, the only new text): two Δt = 0 connectives, `"At that same moment,"` and
  `"Just then,"`, used (i) as the t0 row of C1 and C3 (8 × 3 × 2 = 48 C3 passages) and (ii) as the
  Δt = 0 row of the decoding grid. They give the baseline that removes subject identity from
  every displacement.

Total extraction: 240 (C1) + 216 + 48 (C3) = **504 forward passes**, ~130 tokens each. On the
hour-28 CPU timings this is 15–25 minutes. Nothing else is written unless §5's rule fires.

**Stacks are gone.** `results/time_translation*_stacks.npz` do not exist on disk (`*.npz` is
gitignored; lost in the hour-30 container restart), so re-extraction is unavoidable anyway.
Executing agents: write `results/subject_clocks_measures.json` as soon as the pooled arrays exist,
before any figure; keep the per-token stack in float16; do not rely on the `.npz` surviving.

## 2. Model, layers, pooling

Qwen2.5-1.5B, CPU, `.venv`, `HF_HOME=/home/user/latent-space-exploration/cache/hf`,
`HF_HUB_OFFLINE=1` (model is cached). Layers 0, 8, 14, 20, 27. One `extract(lm, marked,
keep_resid=True)` call per passage yields, from `RoleActivations.resid[:, tokens[role]]`, all of:

- `state`: **mean** pool (the old readout, kept for the diagnosis in §3.1), **last token**, and the
  **per-token** sequence (C3 only, where tokens align across Δt).
- `interval`: **last token** (the comma of "One week later,") = expectation readout E(s, Δt, p).
- `state0`: mean and last, as the per-passage t0 reference (h0).

Layer 0 is a control: the embedding carries no position, so C3 displacements are identically
zero there; C1 differences at layer 0 are pure lexical differences.

Gemma-2-9B-it via NDIF (`.venv312`, `scripts/ndif_time_translation_extract.py` pattern, layers
9/20/31) only under the gate in §7. Download nothing.

## 3. Readouts

Notation: h(s, Δt, p) is a pooled vector at one layer; d(s, Δt, p) = h(s, Δt, p) − h0(s, p) is the
per-paraphrase displacement from that passage's own t0 span; d̄(s, Δt) = mean_p.

### 3.1 Diagnosis of the old probe (C1, mean pool)

Recompute the hour-28 residual on C1: resid(s, Δt) = d̄(s, Δt) − mean_s d̄(s, Δt). Alongside it,
the **paraphrase floor** F(s, Δt) = √((tr Σ̂(s,Δt) + tr Σ̂(s,t0)) / 3), where Σ̂ is the within-cell
covariance over the 3 paraphrases (so tr Σ̂ = mean squared distance of a paraphrase from its cell
mean, times 3/2). Report ‖resid(s, Δt)‖ / F(s, Δt) per subject and Δt. If the ratio sits near 1
for near Δt, the flat curves of hours 28–31 were the floor.

### 3.2 Per-subject contrast direction u_s (C1)

u_s^{(−p)} = normalize( mean_{Δt ≥ 1 ky} d(s, Δt, p') averaged over p' ≠ p ) — the subject's own
t0→far axis, fit **without** the paraphrase it will be applied to. The asteroid, whose state is
written unchanged at every Δt, is the negative control: its u_s is noise and its curve should stay
at floor. The 1-D readout for every construction is

    y(s, Δt, p) = ⟨ d(s, Δt, p), u_s^{(−p)} ⟩.

Computed for C1 (mean and last pool: author-supplied change), for C3 (last pool and per-token:
model-supplied), and for C2 with d replaced by E(s, Δt, p) − Ē(s, null, p) (model-supplied
expectation). Also the **cross-subject matrix** A_Δt[s, s'] = mean_p ⟨ d(s, Δt, p), u_{s'}^{(−p)} ⟩
for C2 and C3, standardized per column by that column's within-cell SD: diagonal dominance
D(Δt) = mean_s A[s,s] − mean_{s≠s'} A[s,s'] with a permutation null over s labels (10⁴ shuffles).

### 3.3 Trajectory alignment to the terminal state (C1, knee-free)

κ(s, Δt) = cos( d̄^{(p)}(s, Δt), d̄^{(−p)}(s, 1 My) ) averaged over the 3 splits (held-out paraphrase
on the left, the other two on the right), after projecting out the leave-one-subject-out shared
direction at that Δt. A subject whose change is complete by Δt has κ ≈ κ(1 My); one whose change
has not started has κ ≈ κ(t0 row). No half-max involved.

### 3.4 Resolvability matrix and floor-referenced τ

**Resolvability** (C1, last pool, also on the 1-D y): for each subject and each pair of timepoints
(i, j) including the t0 row, leave-one-paraphrase-out nearest-centroid classification of the
held-out paraphrase between c_i^{(−p)} and c_j^{(−p)}; 6 trials per pair. R_s[i, j] = accuracy;
"resolved" iff 6/6 (binomial p = 0.016; with 6 paraphrases the rule is ≥ 11/12). The pre-registered
block structure per subject (from the v2 texts) is:

| subject | unresolved from t0 through | first resolved at |
|---|---|---|
| mayfly | (none) | 1 day |
| orchard | 1 week (6 mo resolved, 1 y ≈ t0 again) | 6 months |
| street | 1 year | 10 years |
| real_population, fictional_population | 1 year | 10 years |
| river | 10 years | 100 years |
| mountain | 1 ky | 10 ky |
| asteroid | 1 My | never |

**τ estimator (precise).** For a 1-D readout y with n paraphrases per cell:

    ȳ(t)   = mean_p y(s, t, p)                        for t in {null, 1d, ..., 1My}
    σ_s²   = mean_t var_p y(s, t, p)                 (pooled within-cell variance, df = 10(n−1))
    SE_s   = σ_s · sqrt(2/n)                         (SE of a difference of two cell means)
    b_s    = ȳ(null)                                  (baseline: reworded t0 / Δt = 0 phrase)
    P_s    = mean of the three largest ȳ(t) − b_s    (plateau)
    thr_s  = max( P_s / 2, z · SE_s ),  z = 2.5      (≈ Bonferroni over 9 cells at α ≈ 0.1)
    τ_s    = smallest t with ȳ(t) − b_s ≥ thr_s, provided ȳ(t') − b_s ≥ z·SE_s for at least
             one t' > t as well (a single spike does not count)
    status = "no signal"  if P_s < 2 z SE_s   (the plateau itself is not distinguishable from 0)
           = "≤ 1 day"    if τ_s = 1 day
           = τ_s          otherwise
    CI     = τ_s recomputed under each leave-one-paraphrase-out fit of u_s; report the set.

Illustrative:

```python
def tau(y, order, z=2.5):                 # y[t] -> array of n paraphrase values
    n = len(next(iter(y.values())))
    ybar = {t: y[t].mean() for t in order}
    se = np.sqrt(np.mean([y[t].var(ddof=1) for t in order]) * 2 / n)
    b = ybar[order[0]]                      # null row first
    P = np.mean(sorted(ybar[t] - b for t in order[1:])[-3:])
    if P < 2 * z * se: return "no signal", P / se
    thr = max(P / 2, z * se)
    hits = [t for t in order[1:] if ybar[t] - b >= thr]
    later = [t for t in order[1:] if ybar[t] - b >= z * se]
    return (hits[0] if hits and len(later) >= 2 else "no signal"), P / se
```

τ is reported for C1-last (weak claim), C3-last and C2 (strong claim), at every layer.

### 3.5 Discrimination test (no knees)

For each held-out paraphrase p of subject s at Δt_i (C1, last pool, and separately the y readout):

- **within-subject**: predicted Δt = argmin_j ‖ h(s,Δt_i,p) − c_j^{(−p)}(s) ‖ over the subject's
  own 10 centroids (t0 row included);
- **shared transfer**: predicted Δt = argmin_j ‖ d(s,Δt_i,p) − shared_LOSO(Δt_j) ‖ where
  shared_LOSO is the mean displacement of the other seven subjects;
- **matched-size shared transfer**: the same with shared built from a random 2-paraphrase subset of
  2 other subjects (so both predictors see 18 training vectors), 20 draws.

Score: Spearman(predicted index, true index) over the 27 held-out points per subject, and
mean absolute grid-index error. Report within − shared per subject. Within-subject beating the
shared transfer is *necessary* for a subject clock but not sufficient (same-Δt paraphrases share
vocabulary); the sufficient test is 3.4's block structure being subject-specific *and* 3.6/3.2's
model-supplied readouts being non-null.

### 3.6 Decoding readout (strong claim, no activations)

Prefix(s, i, p) = `"At first, " + S(s,t0,p) + " " + PH(i)` for i in {null₁, null₂, 9 Δt};
candidate(s, j, p) = `" " + S(s, j, p)` for j in {t0, 9 Δt} (the t0 state is the "nothing changed"
candidate). LP[i, j] = **per-token mean** log p(candidate_j | prefix_i) via `LM.logprob` (divide by
candidate token count). Remove the candidate prior: PMI[i, j] = LP[i, j] − mean_i LP[i, j].
8 × 3 × 11 × 10 = 2640 calls; by the hour-28 timing (≈ 40 min for ~4500 patched calls) ≈ 25–35 min.

Statistics, per subject, averaged over paraphrases:

- **diagonal accuracy**: fraction of j (9 Δt) with argmax_i PMI[i, j] = j; chance 1/11.
- **no-change preference**: for the two null rows, fraction with argmax_j PMI[null, j] = t0.
- **block contrast**: for every timepoint pair (i, j) the 2×2 contrast
  Q[i,j] = PMI[i,i] + PMI[j,j] − PMI[i,j] − PMI[j,i]. Pre-registered labels from §3.4's table:
  pair is "should resolve" iff it straddles the subject's first-resolved Δt (one member in the
  unresolved block, one outside), "should not" iff both are inside the same block. Statistic
  B_s = mean Q over should-resolve − mean Q over should-not; null by 10⁴ permutations of the pair
  labels; also Spearman(Q[i,j], label) with the same null.
- **decoded Δt curve**: Δt̂(j) = Σ_i softmax_i(PMI[i, j]) · log Δt_i; plotted against log Δt_j per
  subject; its τ under §3.4's estimator with y(s, j, p) = Δt̂ (p indexes paraphrase).

## 4. Figures and files

`results/figures/subject_clocks_{floor,ycurves_C1,ycurves_C3,ycurves_C2,pertoken,kappa,resolv,
crosssubject,decode_pmi,decode_curve}.png`. All numbers in `results/subject_clocks_measures.json`
and `results/subject_clocks_decode.json`. Notes in `results/notes/subject_clocks.md` (piece 1) and
`results/notes/subject_clocks_decode.md` (piece 2), each with every prediction graded.

## 5. Power

With n paraphrases per cell and within-cell SD σ along u_s, a step of size δ between two cells is
detectable at z = 2.5 iff δ ≥ 2.5 σ √(2/n): **2.0σ at n = 3, 1.44σ at n = 6, 1.12σ at n = 10.** A
knee is *localizable* to one grid point when the half-plateau clears that: P_s ≥ 4.1σ (n = 3),
2.9σ (n = 6), 2.2σ (n = 10). The ordering test across subjects has ±1 grid-point resolution, so
only pairs whose predicted τ differ by ≥ 2 grid points are testable: mayfly < {street, both
populations} < river < mountain, and orchard < {river, mountain}: **9 testable ordered pairs**;
asteroid is scored only as "no signal or 1 My".

The decoding block statistic uses ~20–35 pairs per subject over 3 paraphrases; a permutation test
on ≥ 20 labelled pairs detects a B_s of ≈ 0.6 SD_Q at α = 0.05. Diagonal accuracy over 216 trials
detects ≥ 0.16 against chance 0.09 (binomial, α = 0.05).

**Whether the existing grid is underpowered is decided by the data, by this rule** (piece 1 reports
it, piece 2 acts on it): let r = median over the seven changing subjects of P_s/σ_s on C1-last at
layer 14.
- r ≥ 4.1: the grid is adequate; no new paraphrases.
- 2.2 ≤ r < 4.1: underpowered at n = 3. Piece 2 writes **3 more paraphrases per (subject, timepoint)**
  (8 × 10 × 3 = 240 new state texts, checked with `scripts/time_translation_vocab_check.py`,
  same length band 35–60 words) and reruns piece 1's script on the enlarged grid (n = 6). If r
  after that lies in 2.2–2.9, a further 4 (n = 10) is the ceiling this spec allows.
- r < 2.2: no plateau would be localizable even at n = 10; the author-supplied readout is null at
  this model and the paraphrase question is moot. Piece 2 runs decoding only.

My expectation is r ≈ 5–8 for the mayfly/street/populations and 3–5 for river/mountain, i.e. the
existing 3 paraphrases are marginal for the slow subjects; I predict the rule fires (2.2 ≤ r < 4.1)
with probability 0.4.

## 6. Predictions (graded after the run)

- **P1 (diagnosis).** ‖resid‖/F on C1-mean at layer 14 lies in [0.8, 1.3] for Δt ≤ 1 y for at
  least 6 of 8 subjects: the earlier residual curves were the paraphrase floor. Confidence 0.75.
- **P2 (weak claim, C1-last, y readout).** At layer 14, ≥ 5 of the 7 changing subjects get a
  determinate τ; the asteroid is "no signal"; ≥ 6 of the 9 testable ordered pairs hold; the mayfly
  is "≤ 1 day" or 1 week. Resolvability blocks match §3.4's table for ≥ 5 of 8 subjects at ≥ 70 %
  of pairs. Confidence 0.6. (If P2 holds, hours 28–31 were a probe failure, as suspected.)
- **P3 (strong claim, activation side).** C3-last y curves: diagonal dominance D(Δt) has
  permutation p < 0.05 at ≤ 2 of 9 Δt at layer 14; C2 (expectation) τ is "no signal" for ≥ 6 of 8
  subjects. Per-token C3 curves show the phrase's subject-specific effect, if any, concentrated in
  the first 5 tokens of the repeated state and decaying. Confidence 0.6 that this is the outcome,
  i.e. I expect the 1.5B base model *not* to carry a subject-specific expectation.
- **P4 (strong claim, decoding).** Diagonal accuracy 0.15–0.25 (chance 0.09); no-change preference
  ≥ 0.6 for mountain and asteroid, ≤ 0.4 for the mayfly; block contrast B_s with p < 0.05 for
  **2–3 of 8** subjects, the mayfly among them (death at any Δt ≥ 1 day is the one thing a small
  model surely knows). Confidence 0.5.
- **P5 (discrimination).** Within-subject Spearman ≥ 0.6, shared transfer ≤ 0.4, matched-size
  shared ≤ 0.35, at layer 14, for ≥ 6 subjects. Confidence 0.8; it is the necessary-not-sufficient
  check and I expect it to pass on vocabulary alone.
- **P6 (layers).** All model-supplied effects, where present, peak at layers 14–20 and are zero at
  layer 0 by construction (C3) or at floor (C2).

**What would make me conclude the subject-relative clock does not exist.** On Qwen with n ≥ 6
(or n = 3 if r ≥ 4.1): (a) fewer than 3 of 7 changing subjects have a determinate τ on C1-last, or
fewer than 4 of 9 testable pairs hold, **and** (b) decoding B_s is at chance (p > 0.1) for ≥ 6 of 8
subjects **and** (c) C2/C3 diagonal dominance is at floor at every Δt. Then the earlier "absent"
stands and the line closes. The more likely intermediate outcome — (a) succeeds, (b) and (c) fail
— means: the representation carries the timescale the author wrote into the text, but the model
does not supply one from Δt and subject alone. In that case the **strong claim is dead at 1.5B**,
the weak claim is a paraphrase-invariance result and should be recorded as such in one paragraph,
and only P4 is worth carrying to Gemma (§7).

## 7. Gemma gate

Run Gemma-2-9B-it (NDIF, `.venv312`) **only if** on Qwen either decoding B_s reaches p < 0.05 for
≥ 2 subjects, or C2/C3 diagonal dominance reaches p < 0.05 at ≥ 3 Δt. Then: the decoding test
(logprob via `nnsight` trace, ~2640 calls, checkpoint per subject as in `ndif_recompose_sweep.py`)
and C1/C3 extraction at layers 9/20/31 with the same measurement script. If neither gate condition
fires, do not run Gemma; say so in the note.

## 8. Pieces

**Piece 1 — activations (one agent, local, ~2 h wall, ~150k tokens).**
`scripts/subject_clocks_build.py` (v2 JSON → `prompts/subject_clocks_v1.json` with C1, C3, null
phrases, decoding prefixes/candidates), `scripts/subject_clocks.py --stage extract|measure`
(504 passages; §3.1–3.5; figures; the §5 power rule with r reported at layers 8/14/20),
`results/subject_clocks_measures.json`, `results/notes/subject_clocks.md` grading P1, P2, P3, P5,
P6 and stating in one line which §5 branch applies.

**Piece 2 — decoding and follow-through (one agent, local, ~1.5–3 h wall, ~150k tokens).**
`scripts/subject_clocks_decode.py` (§3.6, 2640 calls, checkpoint per subject to
`results/subject_clocks_decode.json`), grading P4. Then, per piece 1's branch: write the extra
paraphrases and rerun `scripts/subject_clocks.py` on the enlarged grid (updating piece 1's note
with an "n = 6" section rather than a new note), and/or run the Gemma gate. Piece 2 can start
before piece 1 finishes (decoding needs no activations); only its follow-through waits.

Both agents: deliverables go under `results/`, `results/notes/`, `results/figures/`, `prompts/`,
`scripts/` only. **Do not edit RESULTS.md, VISION.md, README.md, WRITEUP.md or docs/ALGEBRA.md.**
Fixed seed 0 everywhere; record wall time and token cost in the note.

## 9. Known weaknesses of this design (stated now, not after)

- The weak-claim tests recover structure the author wrote into the texts. A positive P2 is a
  statement about paraphrase-invariant state encoding, and should not be written up as "the model
  has a mayfly clock".
- The strong-claim tests ask a 1.5B base model for world knowledge (mayfly lifespan, erosion
  rates) through a 60-word prefix. Hour 28's unpatched selector ranking was already at chance
  (4.98/9), though that ranking was length-confounded; the per-token PMI here removes that
  confound but may still find nothing. A null at 1.5B is expected and is why the Gemma gate exists.
- Three paraphrases give σ_s with 20 df; SE estimates are rough and the permutation nulls are the
  statistics to trust.
- The null phrases are Δt = 0 but not content-free; "Just then," may carry a narrative-turn prior.
  Both are used and reported separately; the baseline is their mean.
- τ resolution is one grid point (a decade); "subject-relative" is only claimed for orderings two
  points apart.
