# Pass-through arm for the hour-14 / hour-23 era shift (spec `core_v1.md` §2a)

`scripts/passthrough_test.py`, `results/passthrough_h14.json`.
Sixth broken instrument. **Verdict: (a) — stages 14 and 23 are residual arithmetic. Claim 6 of
WRITEUP.md must be withdrawn in its present form.**

## What was tested

Stage 14 adds `dir_era[e2] − dir_era[e1]` at layer 14 at every position and reads the mean span
vector at layer 20, classifying by nearest era and nearest theme direction. Because the patch is a
constant vector added at *every* position and the readout is a mean over span tokens,

```
mean_span(resid_20 + shift) = mean_span(resid_20) + shift
```

exactly. So the arithmetic prediction of the whole experiment is computable from the cached stacks
with **no forward pass at all**:

```
passthrough = readout(base_span_vector_at_read_layer + shift)
```

scored with the readout code copied verbatim out of `stage7_shift.py`, with the same
leave-one-scene-out directions, the same grand-mean subtraction, and the same rng draw order so the
random control is the same random control.

A second, fairer arm was added after the first pass. The layer-14 shift has a fixed norm while
residual norms grow with depth, so at deep read layers the plain pass-through is unfairly weak — it
fails for a reason that has nothing to do with the blocks. **`passthrough_nm`** rescales the shift so
that `‖shift‖ / ‖resid‖` at the read layer equals its value at the patch layer. This is the arm the
verdict rests on.

## Sanity checks on the instrument itself

| check | expectation | result |
|---|---|---|
| zero shift reproduces the unpatched readout | `shift_stayed == base_era_e1`, `shift_kept == base_theme_t` | holds exactly at every read layer, all three targets |
| base numbers reproduce the logged base | Qwen 0.97 / 0.78 | **0.972 / 0.778** |
| " | Gemma 0.97 / 0.94 | **0.972 / 0.944** |
| " | GPT grid 0.94 / 0.86 | **0.944 / 0.861** |
| logged jsons re-score to the published table | 0.89/0.81, 0.88/0.94, 0.94/0.86 | **0.889/0.806, 0.875/0.944, 0.944/0.861** |

The cached stacks are produced by exactly the call the base arm of `stage7_shift.py` makes
(`extract(lm, "<lead> [[span: …]]").roles["span"]`), and the Gemma stacks by exactly the call
`ndif_shift.py`'s base arm makes, so the pass-through is not an approximation of the base run — it is
the same object.

## The comparison at the logged read layer (patch 14, read 20)

| target | logged (model) moved / kept | pass-through moved / kept | norm-matched pass-through moved / kept | gain vs pass-through | gain vs norm-matched |
|---|---|---|---|---|---|
| Qwen2.5-1.5B, Claude grid (h14) | 0.889 / 0.806 | 0.792 / 0.778 | **1.000** / 0.764 | +0.097 / +0.028 | **−0.111** / +0.042 |
| Gemma-2-9B-it, Claude grid (h14) | 0.875 / 0.944 | 0.972 / 0.958 | **1.000** / 0.917 | **−0.097** / −0.014 | **−0.125** / +0.027 |
| Qwen2.5-1.5B, GPT grid (h23) | 0.944 / 0.861 | 0.861 / 0.847 | **0.972** / 0.792 | +0.083 / +0.014 | **−0.028** / +0.069 |

**The model's number is not distinguishable from the arithmetic, and where it differs it is worse.**
Adding the shift to a cached vector and never running a single block already produces 0.79–0.97 of
the claimed "address moved"; norm-matching the shift produces 0.97–1.00, i.e. *above* every logged
number on all three targets. On Gemma the model is 0.10 *below* the plain pass-through: six blocks of
computation partially undo the arithmetic rather than contributing to it.

"Theme kept" is arithmetic to within ±0.03 on all three targets — the shift is near-orthogonal to the
theme directions, so `base + shift` classifies to the same theme as `base`. Nothing was preserved by
the model; nothing was ever at risk.

The random control is unchanged between the arms (0.00–0.04 moved under both), which is the point the
spec makes: a random direction has cosine near zero with era and therefore cannot distinguish the two
hypotheses at all.

## Layer dependence (Qwen2.5-1.5B, patch layer 14)

`moved` fraction. `pt` = pass-through, `ptNM` = norm-matched pass-through.

| read layer | blocks crossed | model | pt | ptNM | gain vs pt | **gain vs ptNM** |
|---|---|---|---|---|---|---|
| 14 | 0 | 0.000* | 1.000 | 1.000 | −1.000* | −1.000* |
| 15 | 1 | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 |
| 16 | 2 | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 |
| 18 | 4 | 1.000 | 0.986 | 1.000 | +0.014 | +0.000 |
| **20 (logged)** | **6** | **0.889** | **0.792** | **1.000** | **+0.097** | **−0.111** |
| 22 | 8 | 0.875 | 0.181 | 0.889 | +0.694 | −0.014 |
| 24 | 10 | 0.847 | 0.042 | 0.792 | +0.806 | +0.056 |
| 28 (last) | 14 | 0.806 | 0.000 | 0.917 | +0.806 | −0.111 |

GPT grid, same shape: gain vs ptNM = +0.000, −0.014, +0.000, **−0.028**, +0.111, +0.028, −0.042 at
reads 15/16/18/20/22/24/28.

\* Read layer 14 is an artifact of the capture point, not a finding: `LM.patched` installs a
**forward-pre-hook** on block 14, while HF records `hidden_states[14]` before the hook runs, so the
model arm at read 14 is the *unpatched* residual by construction. The honest curve starts at read 15.

**Reading of the curve.** Against the plain pass-through the gain looks like it grows with depth
(+0.01 → +0.10 → +0.69 → +0.81), which is exactly the shape that would have rescued the claim in
weakened form. It is an artifact. The plain pass-through collapses at deep layers because a
fixed-norm layer-14 vector becomes negligible against a layer-24 residual — it fails on magnitude,
not on mechanism. Once the arm is norm-matched the gain is **≤ 0 at every read layer from 15 to 28**,
with a maximum of +0.056 at read 24 on Qwen and +0.111 at read 22 on the GPT grid, both inside the
±0.02–0.03 re-run tolerance times two and both non-monotone. There is no depth at which the blocks
add era information the addition did not already carry.

The flat +0.000 at reads 15, 16 and 18 is the cleanest statement available: across the first four
blocks after the patch, the model arm and a pure vector addition agree **case for case**, to three
decimals, on all 72 shift cases and both grids.

## Coverage

- Qwen2.5-1.5B, Claude theme grid (h14): complete, model arm at eight read layers.
- Qwen2.5-1.5B, GPT-authored theme grid (h23): complete, model arm at eight read layers.
- Gemma-2-9B-it (h14 remote): pass-through and norm-matched pass-through complete from the cached
  NDIF stacks at reads 14/16/20/41; the model arm is the **logged** `results/stage7_gemma9b_shift.json`
  at read 20 only. **Not covered:** a fresh Gemma model arm at other read layers, which would need
  new NDIF jobs. It is not needed for the verdict — the logged Gemma number is already *below* both
  pass-through arms at the layer that was published.
- Not attempted: the 70B (`h37`), which uses a different instrument.

## What else in the repo has the same shape

**The rule from §2a: any instrument whose readout is taken at or after its patch layer needs a
pass-through arm.** Sorted by how closely the shape matches the failure just confirmed.

*Same failure, confirmed:*
- `scripts/stage7_shift.py` (h14 Qwen, h23 GPT grid) — this note.
- `scripts/ndif_shift.py` (h14 Gemma) — this note.

*Clean, and should be recorded as clean so the rule is not over-applied:*
- `scripts/ndif_recompose_gen.py`, `scripts/ndif_recompose_sweep.py` (h27/29/31). The era/theme
  readout re-runs the **generated continuation through an unpatched forward** (`read6(pad)` on
  `f"{lead} {cont}"`), so the added vector is not present in the vector being classified. The spec
  already says so; this run confirms it by inspection. The h29 target (era→target 0.84) is not
  touched by this finding.
- `scripts/ndif_commutator.py` readout arm — same construction, re-fetched residuals of generated
  text with no patch in flight.
- `scripts/stage4_relation.py` / `stage4b_relation_v2.py` / `stage3_source_specificity.py` fit an
  affine operator on cached activations with no patch at all (h3/h16 role lens); the `Readout` there
  never sees the treatment.

*Weaker form of the same risk, untested, and the next thing to check:*
Every remaining patched instrument reads out by teacher-forced **log-probability of a continuation**
through the final norm and `lm_head` — `stage4.py`, `stage4c_relation_wrong_sources.py`,
`stage5_crosstalk.py`, `stage5_factors.py`, `stage6_factors.py`, `ndif_factors.py`,
`ndif_absential_probe.py`, `time_translation_selector.py`. This is *not* the identity failure above:
the statistic is a token log-prob, not a cosine to the very direction that was added, so the shift has
to survive the final norm and the unembedding to move it. But it is still additive transport of a
treatment into a readout, and the arm is just as cheap: `logit_lens(base_resid_last + shift)` through
norm + head, no blocks, scored with the same `logprob` code. Until that arm is run, the claims that
depend on those scripts — **the three-factor battery (h8/h39), the cross-talk matrix, the composed
rank, the h37 70B matched-pair selector** — are asserted rather than demonstrated to be more than
their own arithmetic. The h37 selector is the most exposed of these: it patches at 26 and reads at
the head of an 80-block model, with directions from the same factor space.

## Consequence for claim 6

WRITEUP.md claim 6 currently reads "an era shift moves a passage's address (0.89–0.94 of cases) and
keeps its theme (0.81–0.94, equal to the unpatched readout), on two models and two authors' grids."
Every number in it is reproduced, to three decimals, by adding a vector to a cached activation and
running no model. The claim as stated is about the residual stream being additive, which was never in
doubt. Restated as the spec requires — **gain over pass-through** — the finding is:

> At patch layer 14, read layer 20, the model contributes **−0.11 (Qwen), −0.125 (Gemma), −0.03 (GPT
> grid)** to "address moved" over a norm-matched vector addition, and **+0.04 / +0.03 / +0.07** to
> "theme kept". No read layer from 15 to 28 shows a positive gain outside re-run tolerance.

That is a withdrawal, not a weakening. What survives is only the negative observation that six to
fourteen blocks of Qwen and Gemma leave an additively imposed era direction essentially intact —
which is a statement about the residual stream's linearity, not about recomposition.

This is the sixth broken instrument, and the first one found by the spec before it was built.
