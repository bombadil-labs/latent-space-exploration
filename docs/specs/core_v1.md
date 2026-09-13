# Spec: `lsx.core` — a verified measurement core

*Drafted after 39 stages and five broken instruments. The purpose is not to make experiments
easier to write. It is to make the five known failures structurally impossible and to make a
sixth cheap to detect.*

## 1. Acceptance criteria (pre-registered; do not renegotiate after building)

The core is finished when it does both of these, and not before.

**A. Reproduction.** Re-run through the core, these surviving claims come back within the stated
tolerance of their logged values. Tolerance is ±0.03 on ranks and fractions, ±0.02 on cosines,
unless noted.

| claim | source | target |
|---|---|---|
| three-factor battery, Qwen-1.5B | h8, re-verified h39 | era 1.25, voice 1.24, tense 1.03, composed 2.81/18, no-patch 2.00/2.00/1.50/9.50 |
| role lens, held-out domains | h4 | 1.7/6, random 3.3, chance 3.5 |
| relation selector, 40 domains | h16 | 2.21/6 vs 3.5 null, peak layer 16 |
| era shift as readout | h14 | moved 0.89 (Qwen) / 0.88 (Gemma); theme kept 0.81 / 0.94 |
| era shift in generation, 3x re-imposed | h29 | era→target 0.84, lexical 0.30, Gemma-9B |
| 70B matched pair selector | h37 | era 1.50 base / 1.06 instruct @26; theme 1.06 / 1.06; no-patch 2.00 |
| Gemma clock, corrected | h39 | shared fraction 0.501, Spearman 0.767, phrase ratio 2.50 |

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

A core that cannot rediscover our own bugs has not earned trust. Build B's harness *first*, from the
descriptions above, before the instruments it tests.

## 2. Failure taxonomy → layers

| bucket | characteristic bug | layer that makes it impossible |
|---|---|---|
| stimulus | the answer is in the text (h2 position, h35 lexical) | `Grid` leak report, mandatory at construction |
| extraction | the vectors are not the vectors you think (h36, h39) | one `Stack` path with assertions |
| estimator | the number cannot mean what you want (cross-talk, ties, noise floor) | `calibrate()` gate before real data |
| inference | the comparison is missing or rigged (best-layer, no null) | `Claim` contract |

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
everything. `Readout` and `Probe` return raw scores that cannot be formatted, logged, or written to
JSON — they are inputs to `Claim` only.

## 4. The `Claim` contract

```python
Claim(
  instrument   = "selector",
  treatment    = scores,
  arms         = {"random": ..., "no_patch": ..., "permutation": ...},
  floor        = FloorEstimate(...),     # from Grid.leak, or a paraphrase-noise floor
  effect       = EffectSize(...),        # vs the null distribution, with n and z
  provenance   = stack.provenance | direction.held_out,
)
```

Per-instrument required arms are declared in the registry, not passed by the caller, and
construction raises `MissingArm` if any is absent. Required for every instrument: a **no-patch or
no-direction baseline**. Stage 34 is the whole argument for this — its artifact would have been
visible in one line.

`Claim.render()` is the only path to a printable number, and it always prints treatment, every arm,
the floor, and the effect size together. There is no way to quote a treatment number alone.

## 5. Calibration protocol

Every instrument implements `calibrate() -> CalibrationReport`, run before it sees real data, and
cached per code version. Minimum battery:

- **Noise:** synthetic Gaussian activations of matched shape and scale → the statistic must return
  its declared null value within tolerance.
- **Self-floor:** the instrument applied to its own floor as treatment → approximately zero.
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

Every extraction, local or remote, goes through one function. It asserts, every run:

1. **Padding side** read from the tokenizer and compared to the indexing convention; span indices are
   end-relative when padding is left. (Stage 39.)
2. **Batched-vs-single equivalence** on a random sample of ≥3 items per run: vectors from a batched
   job must match a batch-of-one extraction at cosine ≥ 0.999. (Stage 39, and it would have caught
   stage 36 too.)
3. **Non-empty spans**: every span resolves to ≥1 real token, never into padding.
4. **Layer-output shape**: tuple-or-tensor resolved by the `resid()` helper, never by index.
5. **Provenance** written alongside the `.npz`, including grid hash and code version, so a stale
   stack cannot be silently reused. Stacks are gitignored; the provenance file is not.

## 8. Instrument registry

Six instruments cover everything in the repo. Each declares its required arms, its null value, its
claimed invariances, and its reproduction target from §1A.

| instrument | required arms | null | reproduces |
|---|---|---|---|
| `selector` | random, no_patch, permutation | midpoint of candidates | h4, h8, h16, h37 |
| `composition` | random, no_patch | midpoint over joint variants | h8 (2.81/18) |
| `crosstalk` | permutation | variance-decomposition zero | h8 matrix |
| `discrimination` | shuffled-stimulus, floor | floor value, not chance | h39 |
| `depth_gain` | shuffled-stimulus, floor, layer-0 calibration | 0 at the known-zero point | h38 |
| `generality` | **to be designed — h15/h26 have no null** | — | h15 (0.18 vs 0.06) |

The `generality` row is deliberately unfinished: the abstraction ladder has never had a null, three
were specified at stage 39, and the core must not ship an instrument that cannot state its own.

## 9. Ledger and retraction

Claims append to `results/ledger.jsonl`, keyed by provenance hash. `RESULTS.md` hour entries are
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
2. **Instrument registry + calibration** (§4, §5, §8). Five instruments with passing calibration
   reports; `generality` left declared-but-unimplemented with its null options written out.
   Show the harness catching bugs 3, 4, 5 and 6.
3. **Ledger, retraction, reproduction suite** (§1A, §9). Every target in §1A reproduced within
   tolerance, or the discrepancy explained and logged as a finding. This piece is where the core
   either proves itself or does not.

Pieces 1 and 2 can overlap; piece 3 runs last and alone.
