# Phase 2 Qwen ledger: genuine rows through `build_stack`

Task: land ledger rows for every Qwen2.5-1.5B grid that hour 48's cached-`.npz` batch could only
measure, per the correction in `docs/specs/phase2_v1.md` §0. Local CPU, `.venv`, no NDIF, nothing
downloaded. `Qwen/Qwen2.5-1.5B` is the only base model with real weights in the local HF cache, so
this piece is scoped to it.

**The honest summary first. 15 of 15 candidate rows reached the ledger. Zero were refused.** The
re-extracted stack agrees with the cached `.npz` for the same grid at cosine ≥ 0.99999999997 on
every one of five grids — tighter than h47's 0.99996 precedent by four orders of magnitude, and no
material disagreement to report. No arm sat off its null in the real run, though a smoke test at
low draw counts reproduced hour 48's exact finding that a permutation arm needs many draws, not a
wider band. The numbers themselves replicate the shape hour 48 already found: era and the composed
test clear their measured lexical floor everywhere; `voice` on the reference grid and `theme` on the
GPT-authored theme grid do not, by the same margin hour 48 measured on the cached path (-0.0278 and
-0.0556 respectively, to the fourth decimal).

---

## 0. What was built

**The existing path was found and reused, not reimplemented.** `src/lsx/core/phase2.py` already
had the whole battery — `readout_battery`, `pooled_draw_arm`, `layer_curve`, `h8_lexical_floor` via
`reproduce.py` — built around one function, `h8_replication_rows`, that only ever read a cached
`.npz` (`load_stack_matrix`). That function's body, from "compute the mid-depth layer" onward, is
now `_replication_rows_core(X, n_layers, gridspec, model, stack_tag, prov_base, ...)`, shared by:

* `h8_replication_rows` (unchanged behavior — cached `.npz`, still cannot reach the ledger), and
* `h8_replication_rows_live(lm, grid_name, ...)` (new) — builds `X` via a new
  `stack_matrix_from_build_stack(lm, gridspec)`, which calls `extract.build_stack(lm, grid,
  layers=None, batch_size=8)` (every §7 assertion: padding convention, batched-vs-single
  equivalence, non-empty spans, `resid()`, provenance) and reshapes `stack.acts` into exactly the
  `{span_key: [layer, d]}` shape `load_stack_matrix` returns, so `readout_battery` and
  `layer_curve` run unchanged on either path. `layers=None` stores every layer rather than one,
  because the forward pass already computes every hidden state regardless of how many `build_stack`
  is told to keep — one extraction now buys the full depth curve and the layer-0 floor, with no
  second pass.

Because `prov_base` for the live path is `dict(stack.provenance, direction_held_out=...)` — real
provenance a real `build_stack` wrote, carrying `stack_signature` — the resulting `Claim`s pass
`ledger.check_provenance_from_stack` and can be `Ledger().append()`ed. The cached path's synthetic
`prov_base` (model/grid/stack/layers/pooling as strings, no `grid_hash`, `template`,
`tokenizer_padding`, `lib_versions`, or `stack_signature`) still cannot, unchanged from hour 48.

Two tests added: `test_live_replication_rows_carry_a_real_stack_signature` (the tiny fixture model,
checks the claim survives `check_provenance_from_stack` and that a shape-mismatched cosine
comparison against the real cached Qwen stack is reported, not skipped or crashed) and
`test_live_and_cached_paths_share_the_same_battery_core` (asserts both `h8_replication_rows` and
`h8_replication_rows_live` call through `_replication_rows_core`, so a second drifting path cannot
be reintroduced silently later). **`pytest -q tests/`: 181 passed** (179 before this piece, 2
added; nothing else changed).

Driver script (not part of the deliverable, run from a scratch directory to avoid a `.py` name
collision — see §4): loads `Qwen/Qwen2.5-1.5B` once, calls `h8_replication_rows_live` for each grid,
appends every "built" claim to `Ledger()`, and writes `results/phase2_qwen_live_replications.json`
(the full rows, including depth curves) and `results/phase2_qwen_ledger_landing.json` (the cosine
comparisons and the per-row ledger outcome).

---

## 1. Rows reached vs refused

**15 of 15 candidate rows reached `results/ledger.jsonl`; 0 refused.**

| grid | rows (composed + per-factor lenses) | ledger outcome |
|---|---|---|
| `narrative_factors_v1` | composed, era, voice | 3/3 landed |
| `narrative_factors_gpt_v1` | composed, era, voice | 3/3 landed |
| `narrative_theme_gpt_v1` | composed, era, theme | 3/3 landed |
| `narrative_theme_v1` | composed, era, theme | 3/3 landed |
| `narrative_mood_v1` | composed, era, mood | 3/3 landed |

`results/ledger.jsonl` grew from 21 lines to 36 (15 new rows, all unique `Claim.id`s — checked, not
assumed: `factor` sits in each claim's provenance precisely so era and voice do not collide on the
same stack+layer, per constraint 2). No `ArmOffNull`, `LedgerConflict`, `ProvenanceIncomplete`, or
`ProvenanceNotFromStack` fired on the real run. `narrative_factors_v2` (the reference grid) is not
in this batch: it already reaches the ledger through `reproduce.h8`/`h8_claims`, which is the path
this piece generalized, not duplicated.

`narrative_factors_v1` predates the `factors`/`key_order` convention (it spells its two factors
`eras`/`voices` with no `key_order` key) and carries **no `tense` factor at all** — only era and
voice. Constraint 2's tense/2-level warning applies to `narrative_factors_v2` (already ledgered via
`reproduce.h8_claims`, which builds `tense` from that grid) and does not apply to any row landed
here; no grid in this batch has a `tense` factor.

## 2. Cached-vs-re-extracted cosine per grid

Computed item-by-item, layer-by-layer (29 layers × 36 items = 1044 pairs per grid) between the
`build_stack` extraction and the cached `.npz` for the identical grid:

| grid | n pairs | mean cosine | min cosine | max cosine |
|---|---|---|---|---|
| `narrative_factors_v1` | 1044 | 0.999999999999391 | 0.9999999999776281 | 0.9999999999999863 |
| `narrative_factors_gpt_v1` | 1044 | 0.9999999999992135 | 0.9999999999741559 | 0.9999999999999836 |
| `narrative_theme_gpt_v1` | 1044 | 0.9999999999992094 | 0.9999999999764239 | 0.9999999999999829 |
| `narrative_theme_v1` | 1044 | 0.9999999999993364 | 0.9999999999796361 | 0.999999999999973 |
| `narrative_mood_v1` | 1044 | 0.9999999999994847 | 0.9999999999827551 | 0.9999999999999993 |

Every grid clears h47's precedent (≥ 0.99996) by four to five orders of magnitude — this is
float32-arithmetic-order noise, not a disagreement. **No material disagreement to report; nothing
here is a finding about either stack.** (Full report: `results/phase2_qwen_ledger_landing.json`.)

## 3. Table: treatment / floor / gain / clears

Layer is the pre-registered mid-depth rule `round(0.5*(L-1))` = 14 of 28 everywhere (h8's own
layer). Gains are ranks: **negative is better** (below the floor), per the record's convention
(`reproduced.gain_over_floor` = treatment − floor; h47: era −0.75 clears, voice +0.21 does not).

| grid | factor | k | treatment | lexical floor | gain | clears? | 90% CI (scene bootstrap) |
|---|---|---|---|---|---|---|---|
| `narrative_factors_v1` | composed | 9 | 1.1944 | 2.2222 | **−1.0278** | yes | [−1.111, −0.944] |
| `narrative_factors_v1` | era | 3 | 1.0000 | 2.0000 | **−1.0000** | yes | [−1.056, −0.944] |
| `narrative_factors_v1` | voice | 3 | 1.0000 | 1.0278 | **−0.0278** | **no** | [−0.083, 0.000] |
| `narrative_factors_gpt_v1` | composed | 9 | 1.4167 | 3.0833 | **−1.6667** | yes | [−1.889, −1.444] |
| `narrative_factors_gpt_v1` | era | 3 | 1.0556 | 1.6389 | **−0.5833** | yes | [−0.778, −0.194] |
| `narrative_factors_gpt_v1` | voice | 3 | 1.0000 | 1.3056 | **−0.3056** | yes | [−0.389, −0.222] |
| `narrative_theme_gpt_v1` | composed | 9 | 1.0556 | 1.6667 | **−0.6111** | yes | [−0.722, −0.500] |
| `narrative_theme_gpt_v1` | era | 3 | 1.0000 | 1.2222 | **−0.2222** | yes | [−0.333, −0.111] |
| `narrative_theme_gpt_v1` | theme | 3 | 1.0278 | 1.0833 | **−0.0556** | **no** | [−0.167, +0.056] |
| `narrative_theme_v1` | composed | 9 | 1.1667 | 2.5556 | **−1.3889** | yes | [−1.806, −0.889] |
| `narrative_theme_v1` | era | 3 | 1.0000 | 1.5000 | **−0.5000** | yes | [−0.694, −0.278] |
| `narrative_theme_v1` | theme | 3 | 1.0278 | 1.3333 | **−0.3056** | yes | [−0.472, −0.111] |
| `narrative_mood_v1` | composed | 9 | 1.3889 | 3.5556 | **−2.1667** | yes | [−2.722, −1.611] |
| `narrative_mood_v1` | era | 3 | 1.0000 | 1.8056 | **−0.8056** | yes | [−1.000, −0.639] |
| `narrative_mood_v1` | mood | 3 | 1.1944 | 1.5278 | **−0.3333** | yes | [−0.556, −0.111] |

12 of 15 clear their measured lexical floor. The two that do not are exactly the two hour 48 found
on the cached path with the identical grid: `voice` on `narrative_factors_v1` (−0.0278, unchanged
to the fourth decimal from the cached-path number in `results/notes/phase2_derivable.md`) and
`theme` on `narrative_theme_gpt_v1` (−0.0556, also unchanged to the fourth decimal). **Nothing was
tuned to reproduce that agreement** — the numbers land where they land because the vectors behind
them are the same vectors (§2's cosine), fitted by the same code.

## 4. Arms off their null, and what was done

**In the real (32-permutation, 8-random-pass) run: none. Every arm on every one of the 15 rows sat
within its declared band.** This is not because the design has no failure mode — it is because
hour 48 already found the failure mode and its fix, and this piece used the fixed settings from the
start rather than rediscovering them.

That fix was re-confirmed once, deliberately, before the real run: a timing smoke test on
`narrative_mood_v1` at `n_perm=4, n_rand=2` (a quarter of the pre-registered draw count) refused the
`mood` lens with `ArmOffNull`: *"permutation reads 1.931 where it declared 2.000 (tolerance 0.024,
measured from selector's own null at n=144)"*. This is hour 48's own finding on a second grid: a
permutation arm's randomness lives in the draw, not the item, so few draws under-power the band.
The fix applied — per spec §7 and the constraint above, **more draws, not a wider band** — is simply
using the pre-registered `n_perm=32, n_rand=8` for the real run, which is what hour 48 settled on
for exactly this reason. No band was widened anywhere in this piece; `_replication_rows_core` is
untouched from the cached path's arm-construction code.

## 5. What failed, or what I got wrong

* **A stdlib name collision cost the first run attempt.** The scratch directory used across earlier
  sessions of this project contains a file named `inspect.py` (from an unrelated earlier
  investigation). Running the driver script from that directory put it first on `sys.path`, so
  `numpy`'s internal `from . import inspect` picked up that file instead of the stdlib module and
  crashed with an unrelated `FileNotFoundError` deep inside `numpy._core.overrides`. Fixed by
  running the script from a fresh directory (`/tmp/qwen_ledger_run/`) with nothing named after a
  stdlib module. Left as a reminder that a "no CPU, just extraction" run can still fail on something
  that has nothing to do with the model.
* **The first version of `h8_replication_rows_live` reused `load_stack_matrix`'s cached-name
  convention (`stacks_qwen2.5_1.5b_{grid_name}`) for the comparison stack without checking that name
  exists for every grid before calling it.** `load_stack_matrix` raises `ValueError` on a shape or
  key mismatch and `data()` raises `FileNotFoundError` if the `.npz` is absent — the code now
  catches `FileNotFoundError` specifically (not a bare `except`) so a genuinely missing cache reports
  `compared: false` rather than crashing the whole grid's batery. In this run every grid's cache was
  present, so this path was never exercised for real; it is defensive code for a case that did not
  occur, and it is worth flagging as unexercised rather than claiming it was tested.

## 6. What I did NOT do

* **`narrative_factors_v2`** — the reference grid — is not part of this batch. It already reaches
  the ledger through `reproduce.h8`/`h8_claims`, which is the path this piece's `build_stack` route
  generalizes; re-deriving it here would have been the second-path mistake the task warned against.
* **No other model.** Only `Qwen/Qwen2.5-1.5B` has real weights in the local HF cache (verified:
  2.9G of safetensors; `Qwen2.5-1.5B-Instruct` also does, but no grid requested it and
  `narrative_theme_v1` on the Instruct model was not attempted). Every other model directory under
  `cache/hf/hub` is a config-only stub.
* **I did not widen any tolerance, and did not re-run a battery because a number was disappointing.**
  `voice` on `narrative_factors_v1` and `theme` on `narrative_theme_gpt_v1` are reported as not
  clearing their floor, exactly as measured, with the same draw counts as every clearing row.
* **I did not touch `RESULTS.md`, `WRITEUP.md`, `VISION.md`, `README.md`, `docs/ALGEBRA.md`, or
  `docs/specs/*`**, did not delete any `.npz`, and did not modify `scripts/`.
* **`phase2_h8_replications.json` (the cached-path output) is untouched** — the existing
  `test_replication_rows_report_gain_over_a_measured_floor_and_never_over_chance` test still reads
  it unchanged and still passes. The new output lives in `phase2_qwen_live_replications.json` and
  `phase2_qwen_ledger_landing.json` so the two paths' outputs cannot be confused with each other.
* **I did not select a layer on scoring data.** Layer 14 of 28 is `mid_depth(29)`, fixed before any
  score, matching hour 8's own layer. The full depth curve (step 2) is recorded in
  `phase2_qwen_live_replications.json` for every row; no value from it entered a reported number.
* **`crosstalk`, `depth_gain`, and `generality` remain unbuilt** — untouched by this piece, as
  before.

---

## 7. Files

* `src/lsx/core/phase2.py` — `stack_matrix_from_build_stack`, `h8_replication_rows_live`,
  `_replication_rows_core` (refactored out of `h8_replication_rows`, behavior-preserving).
* `tests/test_core_phase2.py` — two new tests (§0).
* `results/ledger.jsonl` — 15 new standing rows.
* `results/phase2_qwen_live_replications.json` — full rows (treatment, both floors, all arms, the
  depth curve) for all 15, plus any grid-level extraction failure.
* `results/phase2_qwen_ledger_landing.json` — the cosine comparison and the per-row ledger outcome.
* `results/notes/phase2_qwen_ledger.md` — this note.

**`pytest -q tests/`: 181 passed.**
