# `lsx.core` piece 1: rediscovery harness, types, one asserted extraction path

Spec: `docs/specs/core_v1.md` v1.3, §1B (rediscovery), §3 (types), §7 (extraction).
Files added: `src/lsx/core/{__init__,rediscovery,checks,types,extract}.py`,
`tests/test_core_rediscovery.py`, and a path shim in `tests/conftest.py` — the venv's editable
install names the *main* checkout's `src`, so without it `pytest tests/` in a worktree imports a
different tree than the one under edit. Worth knowing for every agent working in a worktree here.
`pytest -q tests/` — 52 passed, 0.7 s. `scripts/` untouched.

The harness was written and committed **before** the types and the extraction path it tests
(commit `d40cf5f`, whose imports resolve to nothing). That ordering is §1B's and it paid: three of
the mechanisms below are not the ones I would have written had I built the types first (see
*spec changes*).

## Per-bug: does the harness catch it, and by what mechanism

Run on the tiny random Qwen2 fixture from `tests/conftest.py`; no 1.5B load, no downloads.
`pytest -q tests/test_core_rediscovery.py` reproduces it and asserts both the verdict and the
*named mechanism*, because being refused for the wrong reason is how a check rots.
(`rediscovery.run_all(lm=None)` runs the six model-free cases without loading anything.)

| §1B bug | caught | mechanism | positive control |
|---|---|---|---|
| 1. patched `output[0]` with batch > 1 | yes | `extract.asserted_patched_forward` → `MovedCandidates`: moved 1/3 sequences, deltas `[0.4988, 0, 0]` | whole-tensor patch moves 3/3 and passes |
| 2. absolute span indices under left padding | yes | `build_stack` → `PaddingConvention`; with that assertion bypassed, `BatchEquivalence` on the **shortest** item, independently | end-relative indexing passes at cosine ≥ 0.999 |
| 3. rank-1-on-ties with no no-patch arm | yes | `Claim` → `MissingArm('no_patch')`; with the arm present at h34's actual 1.00, `Claim` → `ArmOffNull` (and it names `random` at 1.22 too) | all three arms at 2.00 construct |
| 4. best layer chosen on the scoring data | yes | `Claim` → `SelectionOnScoringData` (`Selection.rule` reads as a choice and `held_out` is False) | "full curve reported" constructs |
| 5. residual norm with no floor | yes | `Claim` → `MissingFloor`; and `Floor(stimulus=…, estimator=None)` refuses at its own construction | both floors construct |
| 6. cross-talk rank pinned at chance | yes | `checks.run_calibration` → sensitivity failure (noise 2.000 **passes**, planted signal gives 2.0/2.0/2.0/2.0); `Claim` → `CalibrationFailed` | the target-rank statistic gives 1.80/1.53/1.19/1.00 and constructs |
| 7. labels recoverable from a bag of tokens | yes | `Grid.leak` at construction (recoverable 1.00 vs permutation null 0.40, length r = −1.00); `Claim` → `RawScoreOnLeakyGrid` unless `report_as="gain_over_floor"` | h35-style repaired grid is not flagged and accepts a raw score |
| 7.5 (h40, not in INSTRUMENTS.md's list). `residuals()[-1]` is post-norm | yes | `extract.capture_residual` routes `layer == n_layers` through `LM.pre_norm_residual`; `checks.assert_pre_norm` → `PostNormResidual` | the pre-norm capture passes the same assertion |
| 8. readout at/after the patch layer, no pass-through | yes | `Claim` → `MissingArm('passthrough')`, required because `readout_layer >= patch_layer` — inherited by *any* instrument, not only `readout_shift` | a selector reading before its patch layer is not asked for the arm |

9 of 9. Bugs 3–6 were in piece 2's column in §11; they are wired here because the `Claim` contract
cannot be honest without them (below). Piece 2 still owns the real registry and the full battery.

Two numbers worth keeping:

- **The shortest-item rule earns its place empirically.** Under the reconstructed h39 bug the
  per-item batched-vs-single cosines are `[1.0000, 0.1148, −0.0107, 0.5327]` — the longest item is
  *exactly* clean and everything else is destroyed. A random item passes 1 time in 4 here; at
  stage 39's 117-of-480 it passed far more often than that.
- **The degenerate cross-talk rank passes the noise test.** 2.000 against a declared null of 2.0.
  Only the planted-signal arm separates it from a working statistic. A noise-only `calibrate()`
  would have shipped it again.

## What I had to change in the spec, and why

1. **§5 calibration is partly in piece 1, not wholly in piece 2.** §4 says a missing or stale
   report blocks `Claim` construction; a piece-1 `Claim` that accepted `calibration=None` would
   have been a contract with a hole in it that piece 2 then has to close everywhere. So `Claim`
   requires a `CalibrationReport` now, and `checks.run_calibration` implements two of the five
   tests — **noise and synthetic-signal sensitivity**. Self-floor, scale/rotation invariance and
   the known-zero point remain piece 2's, and the report says which tests it ran.
   `CalibrationReport.hand_declared(...)` exists for instruments piece 2 owns, and stamps itself as
   hand-declared so it cannot pass for a measured one in the ledger.

2. **The leak check needed a null of its own** — the spec's own lesson turned on the spec's own
   instrument. §6 says "ridge on one-hot counts, leave-one-item-out"; read against *nominal
   chance*, that statistic is broken in the h28 way. Leave-one-out on balanced labels is
   anti-predictive by construction: removing an item tips the remaining majority the other way, so
   a grid carrying no lexical information at all scores **0.00, not 0.50**. My first clean control
   grid was duly flagged as leaking. Recoverability is now reported beside a **permutation null of
   the same LOO procedure** (20 draws), and the flag is `acc − null > margin`. An *anti*-predictive
   bag is recorded as a note rather than flagged: it is real information (invert the predictor) but
   it is label-structure, not prose, and it is the signature of exactly the paired design h35's v3
   repair produces — flagging every repaired grid would retire the check by alarm fatigue. Cost at
   realistic size: 8.7 s for 240 items × 2 factors, kernel (n×n) ridge, so it can stay mandatory at
   construction.

3. **`assert_pre_norm` is a ratio test, not `allclose`.** Applying the final norm to a post-norm
   vector moves it by 0.08 % and to a real residual by 3700 %; per-position norms vary by 0.02 %
   against 13 %. An `atol` comparison was model-scale dependent and silently passed on the fixture.

4. **The padding-convention assertion is two-sided.** The spec names the left-padding case (h39).
   End-relative indices under *right* padding are wrong by the same n_pad in the other direction,
   so `span_policy="end_relative"` is now refused when the tokenizer pads right; `"auto"` derives
   the convention from the tokenizer and is the default.

5. **`build_stack` passes explicit `position_ids`.** Not in the spec. Without them a left-padded
   batch gives every short row shifted RoPE positions and batched-vs-single equivalence fails for a
   reason that has nothing to do with span indexing — a second, independent left-padding bug next
   door to h39's. No `scripts/` NDIF or local batch reader passes them today. **Anything piece 3
   re-derives from a batched extraction should be checked for this**; batch-1 jobs are unaffected,
   which is most of the remote record.

6. **`provenance` distinguishes an absent key from a recorded `None`.** `template` is legitimately
   `None` for a base model, and the record of "no chat template" is what closes §2a's untracked
   mismatch; a missing *key* still refuses.

7. **`bypass=` exists on `build_stack`,** for the harness only, so a known-bad configuration can be
   let past one assertion to show the next one catching it unaided. It is the one back door in the
   core and it is named in the signature rather than hidden.

## What is still unprotected

- **Nothing forces an experiment through `build_stack`.** The core refuses bad `Claim`s and bad
  stacks, but `lsx.LM` is still importable and `scripts/` still calls it directly (by design, §10).
  The gate is publication, not computation, exactly as intended — but it means an agent can still
  hand-roll an extraction and wrap the result in a well-formed `Claim`. The `Stack.provenance` hash
  is the only thing tying a Claim to an asserted extraction, and **nothing checks that the Claim's
  provenance came from a Stack.** Piece 3's ledger should refuse a row whose `code_version` and
  `grid_hash` do not come from a real stack provenance.
- **`Probe` is a shell.** §7 says every remote forward goes through the one asserted path;
  `asserted_patched_forward` does that locally and nothing NDIF-side goes through it yet. The
  moved-candidates assertion is the one that caught h34 and it is not yet on any remote call.
- **The arm tolerance is a single default (0.15) chosen by me, not measured.** §1A sets per-target
  tolerances from re-run variance; arms deserve the same and piece 3 measures it. Today an arm can
  sit 0.14 off its null in silence.
- **`Selection` is refused by a regex over the caller's own prose** (`argmax|best|peak|chosen|…`).
  A caller who writes `rule="we looked at the curve and quoted layer 16"` passes. The honest fix is
  for the sweep to be executed by the core so the rule is recorded rather than described; that is a
  piece-2 instrument concern.
- **`Direction` demands a declared held-out axis but does not verify it.** Nothing checks that the
  fit actually excluded `unseen`.
- **No invariance tests, no self-floor, no known-zero.** Three of the five calibration tests are
  unwritten; `depth_gain`'s known-zero is the one that *failed* at h38 and must not gate the core.
- **Effect size is caller-supplied.** `EffectSize(size, n, z)` is recorded, not computed, so the z
  can be anything. It is the one number in the contract with no assertion behind it.
- **Library versions are recorded, never compared.** h36 *was* a transformers change. A stale
  stack is re-identifiable but nothing refuses it; that is piece 3's ledger.
- **Remote (NDIF) versions are unrecorded.** §7.6 asks for locally *and* server-reported versions;
  `nnsight` is absent from the py3.11 venv so the field is written `None`.

## What pieces 2 and 3 should do differently

**Piece 2.**
- Take over `REQUIRED_ARMS` in `types.py` — it is a placeholder table with a comment saying so —
  and add each instrument's null, tolerance and claimed invariances in the same declaration, so
  `ArmOffNull` stops depending on the caller's `expected_null` for the plumbing arms.
- Extend `checks.run_calibration` rather than replacing it: keep noise + sensitivity, add
  self-floor, scale/rotation, known-zero, and the on-disk cache keyed by `checks.calibration_key`
  (already implemented: source + declared null + declared invariances, deliberately not repo-wide).
- `readout_shift`'s pass-through arm must be *computed*, not declared. Piece 1 only checks the arm
  is present. §2a's two hard parts are unbuilt: gain as a **difference** never a ratio, and the arm
  reproducing the unpatched readout **exactly at zero shift**. Build the second as an assertion in
  the arm's constructor — if it fails the arm is wrong, not the claim.
- Expect the sensitivity test, not the noise test, to be the one that fails. Write each instrument's
  planted-signal generator before its statistic.

**Piece 3.**
- Re-run one remote target twice and set remote tolerance from the spread — but budget a second
  spread for the **batched** targets, because the `position_ids` finding above means batched remote
  extractions may differ from their logged values for a reason unrelated to the tolerance being
  measured. Extract one batched target both ways before trusting either spread.
- Refuse ledger rows whose provenance did not come from a `Stack` (above).
- The §1A restatements land as `report_as="gain_over_floor"` (h39's Gemma clock) and
  `Selection(held_out=True, rule="full curve reported")` (h16's layer 16); both are already
  refusable, so those two targets can be graded first and cheaply.
- `Claim.to_row()` is the ledger shape and is implemented; `withdraw()` is not.

## Harness output, verbatim

```
9/9 known bugs rediscovered
```

Full text: `rediscovery.report(lm)` with the `tiny_lm` fixture; every line of it is asserted by
`tests/test_core_rediscovery.py`.
