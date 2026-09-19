# Conscription checker: floor estimation, not a leak gate

Fixes the defect named in `results/notes/conscription_prereg.md` ("the leak threshold is
mis-specified for this pair... gain over a measured floor, which is this project's rule everywhere
else and was not applied to this check"). `scripts/conscription_check.py` no longer passes/fails a
grid on arm-label leakage. Full output for both grids is reproduced below; the script itself
(section 4's docstring) carries the same account.

## What's still a gate, and why those three survive

1. **Schema & completeness** (`check_schema`) — structural breakage: duplicate ids, a prefix that
   doesn't end on an assistant turn, a missing/empty arm, an undeclared domain. Nothing downstream
   can run on a broken record; this is a defect, not a confound. Domain-count shortfall (the human
   grid has 3 of 24 items) is reported loudly but does **not** block, unchanged from before.
2. **Per-arm token-length balance** (`check_lengths`, >15% from grand mean) — the design's own
   stated confound (`CONSCRIPTION_INSTRUCTIONS.md` rule 6): if length is imbalanced, an
   `exit < enact` reading could be "shorter prompt" rather than "an exit available." This is a
   property of the stimulus construction, not something later activation analysis can route around.
3. **`enact`/`exit` shared-assertion check** (`check_enact_exit`, rule 4) — checks that `exit` is
   `enact` with only the closer slot swapped for a permission clause. If that's false the grid
   doesn't implement its own design; also a construction defect.

## What changed: leakage → floor

The old section 4 gated `enact` vs `true` at "off the permutation null by more than 0.2" and printed
PASS/FAIL-shaped verdicts for the rest. That's wrong for a structural reason stated in the
instructions: **every arm's manipulation is realised in language** — `report`'s third-party frame,
`exit`'s permission clause, `true`'s contrary proposition are all words a bag-of-words model can see.
A zero-leak standard is unattainable by this design's own logic, so a threshold on it is arbitrary
(confirmed empirically below: every pair on the 24-item grid clears LOO accuracy ≥0.94 against its
own null).

The new section 4 (`compute_pair_floors`, `print_floors`, `write_floor_sidecar`) reports, for
**every** arm pair, LOO bag-of-words accuracy beside its own permutation null (h32's lesson, kept:
LOO on balanced labels is anti-predictive by construction, so the null — not nominal chance — is the
reference). No PASS/FAIL, no "OFF the null" language anywhere in the new code. The number is labeled
explicitly as *the floor the activation contrast for that pair must beat* and written to
`<grid>.floors.json` next to the grid, keyed by a SHA-256 content hash over `items` only (not
`_meta`, so a metadata edit doesn't stale the floor but any text change does — verified by test).

## Both grids' floor tables

Human grid (`prompts/human/conscription_v1.json`, 3 demo items — `fact01`, `refusal01`, `limit01`):

| pair | LOO acc | permutation null | gap |
|---|---|---|---|
| enact vs report | 1.000 | 0.383 | +0.617 |
| enact vs exit | 1.000 | 0.433 | +0.567 |
| **enact vs true** | **0.500** | **0.250** | **+0.250** |
| enact vs neutral | 1.000 | 0.350 | +0.650 |
| report vs exit | 1.000 | 0.333 | +0.667 |
| report vs true | 1.000 | 0.333 | +0.667 |
| report vs neutral | 1.000 | 0.375 | +0.625 |
| exit vs true | 1.000 | 0.367 | +0.633 |
| exit vs neutral | 1.000 | 0.425 | +0.575 |
| true vs neutral | 1.000 | 0.342 | +0.658 |

(all 5 arms, multiclass diagnostic: LOO 1.000, null 0.103, nominal chance 0.200 — 3 items is too few
for this number to mean much; the pairwise table is the real content.)

Claude grid (`prompts/claude/conscription_claude_v1.json`, 24 items):

| pair | LOO acc | permutation null | gap |
|---|---|---|---|
| enact vs report | 1.000 | 0.493 | +0.507 |
| enact vs exit | 1.000 | 0.500 | +0.500 |
| **enact vs true** | **0.938** | **0.505** | **+0.432** |
| enact vs neutral | 0.979 | 0.525 | +0.454 |
| report vs exit | 1.000 | 0.480 | +0.520 |
| report vs true | 1.000 | 0.501 | +0.499 |
| report vs neutral | 1.000 | 0.525 | +0.475 |
| exit vs true | 1.000 | 0.502 | +0.498 |
| exit vs neutral | 0.979 | 0.523 | +0.456 |
| true vs neutral | 0.979 | 0.517 | +0.462 |

(all 5 arms, multiclass: LOO 0.992, null 0.187, nominal chance 0.200.)

**The `enact_vs_true` number on the Claude grid (0.938 / null 0.505) reproduces the pre-registration's
own figure exactly** — that document reported this pair "leak-checks at LOO accuracy 0.938 against a
permutation null of 0.505" after eight opener variants failed to move it, and traced the residue to
the structural confound (compliance/refusal vocabulary partitioning `enact` vs `true` in the
`refusal` and `limit` domains). This checker rediscovering the identical number independently is a
sign the floor machinery is doing what it says; it is also exactly why this pair's floor needed to
stop being graded against a threshold — 0.432 above null is not a defect to fix by rewording, it's a
property of what the two arms assert.

Sidecars written: `prompts/human/conscription_v1.json.floors.json` (content hash `28e46e70dd701cd3`),
`prompts/claude/conscription_claude_v1.json.floors.json` (content hash `2839471e56d39971`).

## New diagnostics, with their own permutation nulls

All four requested diagnostics are implemented (`negation_density`, `second_person_density` /
`third_person_density`, `type_token_ratio`, `mean_sentence_length`), each reported per arm, pooled
and per domain, beside a permutation null of the observed arm-mean range (shuffle which arm label
attaches to which text, `null_draws=20`, report the null range distribution's mean/sd). Selected
findings, pooled across both grids:

- **Negation density is almost entirely on `true`.** Claude grid pooled: `true`=0.051 vs every other
  arm ≤0.005 (observed range 0.051, permutation null range 0.010±0.005 — real signal, not the null's
  noise floor). Per domain it's driven by `refusal` (0.088) and `limit` (0.057), exactly the two
  domains the pre-registration flagged as structurally confounded (`true` there asserts absence:
  "you refused," "you don't know that"). `enact`/`report`/`exit`/`neutral` sit near zero everywhere.
  This is the negation half of the same confound the pre-reg already found via the LOO floor —
  independent confirmation from a different statistic.
- **`exit` has the highest second-person density on both grids** (Claude pooled: `exit`=0.120 vs
  `enact`=0.082, `report`=0.057, `true`=0.061, `neutral`=0.008) — the permission clause
  ("**you're** free to disagree with **me**") adds an extra "you" on top of the assertion `exit`
  already shares with `enact`. Not obviously a problem (the permission *is* the manipulation) but
  worth having as a number.
- **`report` carries the most third-person density, as designed** (Claude pooled: `report`=0.018 vs
  `enact`=0.005, `exit`=0.002) — this is the instructions' third-party frame, quantified rather than
  left as prose. `neutral` is comparably high (0.018) on the Claude grid only, worth a look (its
  closers occasionally reference other people, e.g. "I'll ask a friend"), not present in the 3-item
  human grid.
- **`neutral` has the lowest type-token ratio on the Claude grid** (0.811 vs 0.91–0.93 for the other
  four arms, observed range 0.116 vs permutation null range 0.043±0.015) — some regularity in how
  the machine wrote its inert closers. The human grid's 3 items don't show this pattern (`neutral`
  ties for highest TTR there), but n=3 make that comparison close to meaningless.
- **`report` and `true` run the longest sentences on the Claude grid, `neutral` and `enact` the
  shortest** (pooled means 11.5 / 10.6 vs 7.4 / 8.0; observed range 4.17 vs permutation null range
  0.90±0.28 — the largest gap of any diagnostic, and the clearest style regularity in the whole
  report). This is the strongest of the stylistic-regularity numbers the pre-registration predicted
  ("more regular phrasing" in the machine grid) and it holds in all four domains individually
  (per-domain ranges 3.5–4.7, nulls 1.6–2.4), so it isn't one domain carrying the pooled figure.

Full per-arm, per-domain numbers for both grids are in the raw script output (reproducible with the
commands in the Report section below); this note keeps the pooled headline numbers rather than
reprinting all 4 diagnostics × 4 domains × 2 grids.

## What I got wrong / what failed along the way

`.venv/bin/python` does not exist in this worktree — the project's `.venv` lives in the main repo
checkout, not in `.claude/worktrees/agent-.../`. Had to invoke the interpreter by its absolute path
in the main repo (`/home/user/latent-space-exploration/.venv/bin/python`) from inside the worktree.
Separately, this worktree's own `HF_HOME` (`<worktree>/cache/hf`) has no cached tokenizer —
`cache/hf/hub` doesn't exist here at all, confirmed with a direct check — so the first run of the
checker failed with a `LocalEntryNotFoundError`/offline-mode `OSError` trying to load
`Qwen/Qwen2.5-1.5B-Instruct`. The tokenizer *is* cached, just under `/root/.cache/huggingface` (the
main checkout's shared cache), so I pointed `HF_HOME` there for the manual runs reported here — no
download happened (`HF_HUB_OFFLINE=1` stayed set throughout and the run only reads an existing local
cache). `pytest -q tests/` still reports 3 skipped for exactly this reason (`needs_qwen_tok` checks
`ROOT/cache/hf/hub`, which this worktree doesn't have); those 3 are the same tests that were already
skipping before this change and are unrelated to the checker rewrite.

## What I did NOT do

- Did not touch `prompts/`, `RESULTS.md`, `WRITEUP.md`, `VISION.md`, `README.md`, or `docs/`, per the
  brief. Both grid JSON files are byte-identical to what I read them as (only the new `.floors.json`
  sidecars were added next to them, which the brief explicitly asked for — not an edit to either
  grid's own content).
- Did not add a counterbalance fix for the `enact`/`true` structural confound the pre-registration
  names (half-refusing, half-agreeing prefixes per domain). That's a grid-authoring change, and grid
  authoring is out of scope here (and `prompts/` is off-limits to me regardless) — the checker now
  *reports* the confound with numbers (negation density, the per-pair floor) rather than fixing the
  stimuli that produce it.
- Did not attempt to make the floor's tokenization match the Qwen BPE / chat-template pipeline the
  activation extraction will actually use. Documented instead, in both the script's module docstring
  and the sidecar's `tokenization` field: the floor is measured on raw arm-turn text
  (`item["arms"][arm]`, no prefix, no chat template) with the same plain regex word-splitter
  `lsx.core.types._bag` uses everywhere else in this project — not the model's own tokenizer, and not
  the rendered `prefix + arm` text `conscription.render_prompt` builds. **Guaranteed to match the
  downstream analysis: the grid content** (pinned by `grid_content_hash`, which changes if any item's
  text changes). **Not guaranteed to match: the featurization** — vocabulary is BPE merges vs. plain
  words, and the floor excludes the shared prefix that the model reads before every arm. This is
  stated rather than silently trusted, per the brief's warning that a floor computed under a
  different tokenization/vocabulary/fold structure than the activation analysis "is not that
  analysis's floor" — the same shape as the h8-composed error, in new clothing, if left unstated.
- Did not implement a null-draw count higher than 20 (kept the project's existing convention from
  `bag_of_tokens_recoverability`'s default and the old checker's usage) or attempt to bootstrap
  confidence intervals on the diagnostic arm-means; the permutation null range is the only spread
  estimate reported for the diagnostics.
- Did not run this against a live model or NDIF — everything above is CPU-local, offline, no
  downloads, per the environment constraints.

## Report

Branch: `worktree-agent-a9fb4671455165009` (this worktree's branch off
`claude/amazing-faraday-881p04`, per the environment header — not pushed anywhere).

Files changed: `scripts/conscription_check.py` (rewritten section 4; sections 1–3 unchanged in
behavior), `tests/test_conscription_check.py` (new, 13 tests covering the floor/diagnostic/sidecar
machinery without needing the Qwen tokenizer), plus two new sidecars written as output artifacts:
`prompts/human/conscription_v1.json.floors.json`, `prompts/claude/conscription_claude_v1.json.floors.json`.

`pytest -q tests/`: **204 passed, 3 skipped** (the 3 skips are pre-existing `needs_qwen_tok` tests
this worktree's own `cache/hf` doesn't satisfy, unrelated to this change — see above).

Floor tables for both grids are above. Headline: `enact_vs_true` sits closest to its own null on both
grids (human: gap +0.250 on 3 items; Claude: gap +0.432, reproducing the pre-registration's own
measurement exactly) while every other pair sits further above its null (~+0.45 to +0.67 on the
Claude grid) — consistent with the design's intent that `enact` and `true` differ least in surface
wording and most in what's asserted. No pair reads at or below its null on either grid.

Reproduce with:
```
HF_HOME=/root/.cache/huggingface HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
  .venv/bin/python scripts/conscription_check.py prompts/human/conscription_v1.json
HF_HOME=/root/.cache/huggingface HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
  .venv/bin/python scripts/conscription_check.py prompts/claude/conscription_claude_v1.json
```
