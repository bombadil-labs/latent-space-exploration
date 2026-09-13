# Spec: `lsx.core` — a verified measurement core

*Drafted after 39 stages and five broken instruments. The purpose is not to make experiments
easier to write. It is to make the five known failures structurally impossible and to make a
sixth cheap to detect.*

## 0. Blocking dependency

**Resolved at stage 40: verdict (a). Stage 14 is residual arithmetic and claim 6 is withdrawn.**
The model arm is indistinguishable from vector addition and on Gemma is worse than it; gain over a
norm-matched pass-through is −0.111, −0.125, −0.028. Consequences, applied below: §1A loses its
stage-14 row; §8's `readout_shift` keeps the instrument but its reproduction target becomes the
*negative* result — the instrument must return "no gain over pass-through" on the stage-14
configuration; and `results/notes/passthrough_h14.md` is the reference implementation of the arm.

**A new blocking item takes its place.** The same arm is untested against the log-probability
selector instruments (`stage4*`, `stage5*`, `stage6_factors`, `ndif_factors`,
`time_translation_selector`, the h37 70B battery). Exposure there is weaker — a log-prob readout
passes through the unembedding rather than being a linear readout of the same stream — but four §1A
targets depend on it. **Run that arm before piece 3 grades any selector target.** It need not block
pieces 1 and 2.

## 1. Acceptance criteria (pre-registered; do not renegotiate after building)

The core is finished when it does both of these, and not before.

**A. Reproduction.** Re-run through the core, these surviving claims come back within tolerance.
**Tolerance is per target, set from measured re-run variance, not one number for all.** Local
targets: ±0.02 (stage 39's re-run reproduced stage 8 to two decimals, so this is measured). Remote
targets: **unmeasured** — the first task of piece 3 is to re-run one remote target twice and set the
tolerance from the spread. Stage 29 lost 17 of 72 jobs and stage 37 hit bf16 ties, so ±0.03 on a
fraction over ~55 surviving generations is almost certainly tighter than the noise.

Two targets are restated because the core would refuse them as originally logged, which is the point:
stage 16's "peak layer 16" was an argmax on the scoring data, and stage 39's numbers are raw scores
on a grid with 221 of 240 leaky spans.

| claim | source | target |
|---|---|---|
| three-factor battery, Qwen-1.5B | h8, re-verified h39 | era 1.25, voice 1.24, tense 1.03, composed 2.81/18, no-patch 2.00/2.00/1.50/9.50 |
| role lens, held-out domains | h4 | 1.7/6, random 3.3, chance 3.5 |
| relation selector, 40 domains | h16 | 2.21/6 vs 3.5 null, **as a full layer curve — the logged "peak layer 16" is argmax on scoring data and the core must refuse it** |
| ~~era shift as readout~~ | h14 | **withdrawn at h40.** Replaced by a negative target: on this configuration the core must report no gain over a norm-matched pass-through (logged gain −0.111 / −0.125 / −0.028) |
| era shift in generation, 3x re-imposed | h29 | era→target 0.84, lexical 0.30, Gemma-9B |
| 70B matched pair selector | h37 | era 1.50 base / 1.06 instruct @26; theme 1.06 / 1.06; no-patch 2.00 |
| Gemma clock, corrected | h39 | as **gain over the measured stimulus floor**; the logged 0.501 / 0.767 / 2.50 are raw scores on a leaky grid and the core must not print them bare |

**B. Rediscovery.** Fed each known-bad configuration, the core must refuse or flag it *without
being told what to look for*. These are the five bugs plus the two near-misses:

1. Patching `output[0]` with batch > 1 → extraction/probe assertion fails (moved candidates ≠ batch).
2. Absolute span indices under left padding → batched-vs-single equivalence check fails.
3. Rank-1-on-ties with no no-patch arm → `Claim` construction refuses (missing required arm).
4. Best-layer chosen on the evaluation split → refuses (selection axis not in the declared held-out set).
5. A high-dimensional residual norm with no floor → refuses (no floor/power estimate attached).
6. A cross-talk rank that averages to chance by construction → `calibrate()` flags it as degenerate
   (returns its chance value on synthetic signal *and* on synthetic noise).
7. A grid whose labels are recoverable from a bag of tokens → allowed, but the `Claim` must report
   gain over the measured floor, never raw score.
8. **A readout at or after the patch layer whose movement is residual arithmetic** → refuses without
   a `passthrough` arm (§2a).

A core that cannot rediscover our own bugs has not earned trust. Build B's harness *first*, from the
descriptions above, before the instruments it tests.

## 2. Failure taxonomy → layers

| bucket | characteristic bug | layer that makes it impossible |
|---|---|---|
| stimulus | the answer is in the text (h2 position, h35 lexical) | `Grid` leak report, mandatory at construction |
| extraction | the vectors are not the vectors you think (h36, h39) | one `Stack` path with assertions |
| estimator | the number cannot mean what you want (cross-talk, ties, noise floor) | `calibrate()` gate before real data |
| inference | the comparison is missing or rigged (best-layer, no null) | `Claim` contract |

## 2a. The sixth failure mode: treatment-in-readout pass-through

Found by adversarial review of this spec, and it implicates a **standing claim**.

Stage 14 patches `dir_era[e2] − dir_era[e1]` at layer 14 at every position and reads the pooled span
at layer 20 by nearest era direction. But `resid₂₀ = resid₁₄ + shift + Σ(block outputs)`, and the era
direction is stable across layers — decodable by layer 12 — so the layer-20 cosine moves by residual
arithmetic **whether or not blocks 14 through 19 do anything**. "Theme kept" is equally arithmetic,
and the random control passes trivially because a random direction has cosine near zero with era.
This has the exact signature of the other five: a fixed point that reads as a finding, here 0.89
"address moved".

**The arm.** Computable offline from cached stacks with no forward pass:
`passthrough = readout(base_resid_at_read_layer + shift)`, scored with the *same* readout code as the
treatment. **Gain is defined as `treatment_score − passthrough_score`, reported with the
paraphrase-noise interval on that difference, never as a ratio** — a ratio is unstable when the
pass-through value is already near the ceiling, which is precisely the regime in question. The arm
must also reproduce the unpatched readout exactly at zero shift; if it does not, the arm is wrong,
not the claim.

**The rule.** Any instrument whose readout layer is at or after its patch layer requires a
`passthrough` arm. Stage 29's generation readout is unpatched at read time and is clean; every
`Readout` on a patched forward is not.

Two smaller uncovered modes, both to be recorded in provenance rather than checked:
**library versions**, since the stage-36 bug *was* a transformers change (≥4.54 returning bare
tensors) and nothing is pinned — record local and NDIF-reported versions; and **template mismatch**,
since directions are fit on raw `lead + span` text and applied inside chat templates on instruct
models — record the template alongside padding side.

## 3. Types

```python
Grid                 # stimuli + factors/levels + marked spans + leak report
  .items             -> [Item(text, factors: dict[str,str], spans: dict[str,(i,j)])]
  .leak              -> LeakReport   # computed at construction, cached by grid hash
  .hash              -> str

Stack                # Grid x Model x layers -> activations
  .acts              -> ndarray[item, span, layer, d]
  .provenance        -> {model, layers, pooling, grid_hash, code_version, tokenizer_padding}

Direction            # Stack -> vector; ALWAYS leave-one-out over a declared axis
  .vec(layer)        -> ndarray[d]
  .held_out          -> {axis: str, unseen: set}   # what this direction never saw

Readout              # Direction x Stack -> scores        (no model call)
Probe                # Direction x Model  -> scores       (patched forward or generation)

Claim                # the ONLY exportable type
```

`Direction` construction requires a declared held-out axis; there is no constructor that fits on
everything. `Readout` and `Probe` return raw scores. They may be inspected through an explicitly
**un-ledgerable `Sketch`**, which prints freely and can never reach the ledger or a generated
document. The contract gates *publication*, not thought: a `Readout` that could not be printed at all
would be bypassed with `print(scores.mean())` within a day.

## 4. The `Claim` contract

```python
Claim(
  instrument   = "selector",
  treatment    = scores,
  arms         = {"random": Arm(scores, expected_null=2.0),   # each arm declares where it should sit
                  "no_patch": Arm(..., expected_null=2.0),
                  "permutation": Arm(...)},
  semantic_null= Arm(..., justification="role identity retained"),  # caller-declared, see below
  floor        = Floor(stimulus=..., estimator=...),   # BOTH, always
  selection    = Selection(axis="layer", rule="full curve reported", held_out=True),
  effect       = EffectSize(...),        # vs the null distribution, with n and z
  provenance   = stack.provenance | direction.held_out,
)
```

**Plumbing arms** (no_patch, random, permutation) are declared in the registry, not by the caller;
construction raises `MissingArm` if any is absent. **The semantic null is the caller's**, with a one-line justification recorded in provenance, subject
to two rules that stop it becoming a rubber stamp: it must be **computed from data as a real arm**,
never asserted in prose, and it must be **declared in the `Experiment` before any treatment score is
computed**, enforced by construction order rather than by discipline. The right null depends on the
question rather than the instrument: stage 16's relation selector needs "role identity retained" (1.37 vs 2.21), the
ladder needs a marginal-matched partner, stage 38 needs the lexical floor. Forcing these through
`permutation` would either corrupt that arm or push callers out of the registry entirely.

**Every arm declares where it should sit, and `Claim` raises `ArmOffNull` when one is outside
tolerance.** This is the single rule that catches stage 34: under that bug the no-patch arm read a
respectable 2.00, and what actually screamed was the *random* arm at 1.22 where it belonged near 2.0.
A missing arm was only half the failure; an arm off its null and nobody noticing was the other half.

**`Floor` has two fields and both are required.** A stimulus floor (what the text gives away) and an
estimator floor (what the statistic returns on its own noise). Stage 28 would have passed a
stimulus-floor-only check while sitting at its estimator floor, which is exactly how it published
three false negatives.

**`Selection` records what was swept and how the reported value was chosen.** A Claim whose selection
axis is not in a declared held-out set is refused. `Direction.held_out` governs *fitting*; nothing in
the original draft stopped `min(claim(l) for l in layers)` — twenty-nine valid Claims, one quoted.
Stage 38's `L* = argmax` and stage 16's "peak layer 16" are both this failure.

`Claim.render()` is the only path to a printable number, and it always prints treatment, every arm,
the floor, and the effect size together. There is no way to quote a treatment number alone.

## 5. Calibration protocol

Every instrument implements `calibrate() -> CalibrationReport`, run before it sees real data, and
cached under a key that is the hash of **the instrument's own source, its declared nulls, and its
declared invariances** — not a repo-wide version, so that editing one instrument re-calibrates that
one and only that one. A stale or missing report blocks `Claim` construction for that instrument.
Minimum battery:

- **Noise:** synthetic Gaussian activations of matched shape and scale → the statistic must return
  its declared null value within tolerance.
- **Sensitivity (synthetic signal):** a planted effect of known size → the statistic must move, and
  move monotonically with the planted size. **Noise alone is not enough:** the broken cross-talk rank
  returned its chance value 2.0 on noise *and* passed, because it returned 2.0 on everything. A
  statistic that cannot distinguish signal from noise fails calibration even if its null is perfect.
- **Self-floor (mandatory, not optional):** the instrument applied to its own floor as treatment →
  approximately zero.
- **Scale and rotation:** per-layer rescaling and a random orthogonal rotation must not change a
  statistic that claims to be invariant to them; the report says which invariances are claimed.
- **Known-zero point:** a configuration where the answer is analytically zero (for a depth gain,
  layer 0 against a bag-of-embeddings floor, when the two are the same object — as they are for
  Qwen and Gemma, which stage 38 discovered the hard way).

A statistic that fails its own calibration may still be reported, but only alongside a passing
companion, and the `Claim` records the failure. That is what stage 38 did by hand; the core makes
it the rule.

## 6. `Grid` and the leak report

Computed at construction and cached:

- **Bag-of-tokens recoverability** of every factor label (ridge on one-hot counts, leave-one-item-out).
- **Layer-0 recoverability**, noting that for RoPE models layer 0 *is* the static embedding bag, so
  the two are the same check and the report says so rather than pretending to two measurements.
- **Span hygiene:** no span may contain a token from its own label vocabulary unless declared; a
  `leak_check` per grid family (the duration-expression checker from stage 35 is the first).
- **Position balance:** factor levels must not correlate with span position or item length; report
  the correlation, since stage 2 was exactly this.

Grids do not have to be leak-free. Claims built on a leaky grid must report **gain over the
measured floor**, which is the stage-38 lesson.

## 7. `Stack`: one extraction path, with assertions

Every extraction **and every remote forward, `Probe` included**, goes through one function. The
stage-36 bug lived in a patched forward, not in an extraction, so assertions scoped to `Stack` alone
would have missed it. It asserts, every run:

1. **Padding side** read from the tokenizer and compared to the indexing convention; span indices are
   end-relative when padding is left. (Stage 39.)
2. **Batched-vs-single equivalence** on the **shortest item in each batch**, never a random sample:
   vectors from a batched job must match a batch-of-one extraction at cosine ≥ 0.999. Under stage
   39's bug 117 of 480 passages were clean — the longest in each job — so three random items would
   all have been clean about 1.4% of the time... and would have *passed* far more often than that.
   The shortest item is the one that is maximally padded, so it is the one that fails first.
3. **Moved-candidates assertion** on every patched forward: the number of candidates whose score
   changed equals the batch size. (Stage 36.)
4. **Non-empty spans**: every span resolves to ≥1 real token, never into padding.
5. **Layer-output shape**: tuple-or-tensor resolved by the `resid()` helper, never by index.
6. **Provenance** written alongside the `.npz`: grid hash, code version, **transformers and nnsight
   versions both locally and as reported by the NDIF server**, and the **chat template** in force
   (directions fit on raw text and applied inside a chat template is an untracked mismatch today).
   A stale stack cannot be silently reused. Stacks are gitignored; the provenance file is not.

## 8. Instrument registry

Six instruments cover everything in the repo. Each declares its required arms, its null value, its
claimed invariances, and its reproduction target from §1A.

| instrument | required arms | null | reproduces |
|---|---|---|---|
| `selector` | random, no_patch, permutation | midpoint of candidates | h4, h8, h16, h37 |
| `composition` | random, no_patch | midpoint over joint variants | h8 (2.81/18) |
| `readout_shift` | random, no_patch, **passthrough (norm-matched)** | pass-through value, not chance | h14 as a *negative* target: no gain |
| `crosstalk` | permutation | variance-decomposition zero | h8 matrix |
| `discrimination` | shuffled-stimulus, floor | floor value, not chance | h39 |
| `depth_gain` | shuffled-stimulus, floor, layer-0 calibration | 0 at the known-zero point | h38 |
| `generality` | **to be designed — h15/h26 have no null** | — | h15 (0.18 vs 0.06) |

Any instrument whose readout layer is at or after its patch layer inherits the `passthrough`
requirement automatically, not only `readout_shift`.

The `generality` row is deliberately unfinished: the abstraction ladder has never had a null, three
were specified at stage 39, and the core must not ship an instrument that cannot state its own.

## 9. Ledger and retraction

Claims append to `results/ledger.jsonl`, one JSON object per line:

```
{ id, stage, instrument, treatment, arms:{name:{value, expected_null, off_null:bool}},
  semantic_null:{value, justification}, floor:{stimulus, estimator}, effect:{size, n, z},
  selection:{axis, rule, held_out}, provenance:{grid_hash, model, layers, pooling, template,
  padding_side, code_version, lib_versions, calibration_key},
  status:"standing"|"withdrawn", withdrawn:{reason, superseded_by, at} }
```

`id` is the provenance hash, so the same experiment re-run is recognised rather than duplicated, and
a changed grid or library version produces a new id rather than silently overwriting. `RESULTS.md` hour entries are
**generated** from the ledger rather than hand-written.

**Retraction is a first-class operation.** `withdraw(claim_id, reason, superseded_by=None)` marks a
claim, and every generated document regenerates showing it as withdrawn with its reason inline. In a
project with five retractions this is the main feature, not a flourish: the current retractions are
prose scattered across Checkpoint 2, `docs/INSTRUMENTS.md`, and four hour entries, and keeping them
consistent by hand is already error-prone.

## 10. Migration and non-goals

Existing scripts are **not** rewritten. They are frozen. New work uses the core; an old result is
re-derived through the core only when it is on the §1A reproduction list or when it is challenged.
`scripts/` keeps its history so the provenance of published numbers stays inspectable.

**Non-goals.** Not a general interpretability library. Not a config-file DSL — experiments are
Python declarations. Not a performance project. Not a rewrite of the NDIF layer, which works and
whose bugs were in the callers.

## 11. Pieces

1. **Rediscovery harness + types + extraction** (§1B, §3, §7). Build the harness first from the bug
   descriptions, then the types and the single extraction path, and show the harness catching bugs
   1, 2 and 7. Deliverable: `src/lsx/core/{types,extract,checks}.py`, `results/notes/core_p1.md`.
2. **Instrument registry + calibration** (§4, §5, §8), **three instruments only**: `selector`,
   `composition`, `readout_shift`. Five was underestimated by two to three times — it is stage 38's
   estimator run several times over — and `depth_gain`'s known-zero calibration is the one that
   *failed* at stage 38, so it should not gate the core. Show the harness catching bugs 3, 4, 5, 6
   and 8.
3. **Ledger, retraction, reproduction suite** (§1A, §9), **from cached stacks only**. The first task
   is to re-run one remote target twice and set remote tolerance from the spread; targets whose
   stacks are not cached are deferred rather than re-extracted. This piece is where the core either
   proves itself or does not.
4. **Deferred:** `crosstalk`, `depth_gain`, and `generality` with the null it still lacks.

Pieces 1 and 2 can overlap; piece 3 runs last and alone.
