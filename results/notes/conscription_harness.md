# Conscription harness: what it checks, what it ran, what it did not

Design: `prompts/human/CONSCRIPTION_INSTRUCTIONS.md` / `prompts/human/conscription_v1.json` (three
worked-example items as of this run: `fact01`, `refusal01`, `limit01` -- not the 24-item study).
Built: `scripts/conscription_check.py` (pre-model grid check) and `src/lsx/core/conscription.py`
(the runner). `pytest -q tests/` -- **194 passed** (181 pre-existing + 13 new in
`tests/test_core_conscription.py`), 13.4s, no model load in the test suite itself.

Branch: `claude/conscription-harness`, off `claude/amazing-faraday-881p04` (which now carries the
design files). Not pushed. No NDIF jobs submitted; no large downloads; no credentials touched or
printed.

## A false start, corrected before it went anywhere

This agent was first launched from a worktree branched *before* the design files existed. It found
`prompts/human/CONSCRIPTION_INSTRUCTIONS.md` and `conscription_v1.json` absent everywhere --
working tree, every branch's history, every sibling worktree -- and, since the brief explicitly
forbids inventing stimuli or filling in `prompts/human/*`, refused to guess the schema and handed
back rather than building a harness against a fabricated design. The coordinator confirmed the
files existed on `claude/amazing-faraday-881p04` and had been written after this worktree branched.
Fetching that branch and rebasing (moving to a fresh branch off it, since the instructions say never
push to `amazing-faraday-881p04` directly) resolved it. Noted here because it is exactly the kind of
thing `docs/DELEGATION.md`'s screen wants surfaced, not smoothed over.

## 1. `scripts/conscription_check.py` -- what it measures and why

Five sections, in this order:

1. **Schema & completeness.** Unique ids, every arm present per item, every prefix present and
   ending in an assistant turn, domain counts against `_meta.items_per_domain`. A structural break
   (missing arm, non-assistant-ending prefix, duplicate id) blocks the rest of the script; an
   incomplete domain count (this grid has 3 of the eventual 24 items) is reported loudly but does
   **not** block, because the point of a dry run is to exercise the length/leak/floor machinery on
   whatever exists, not to refuse a design still being written.
2. **Per-arm token length (the headline).** Renders every item x arm through the model's own chat
   template exactly as `conscription.render_prompt` will for extraction, tokenizes with
   Qwen2.5-1.5B-Instruct's tokenizer (local, `HF_HUB_OFFLINE=1`), and reports mean/sd per arm, each
   arm's % distance from the grand mean, and the max pairwise arm-mean gap. Flags any arm past 15%.
   This leads because `exit`'s permission clause is the one place in this design where a real
   length confound could produce a fake `exit < enact` effect.
3. **`enact`/`exit` identity.** Checks that `exit` starts with `enact` verbatim and reports the
   exact character where they diverge and both full strings, rather than assuming identity.
4. **Arm-label leakage.** `types.bag_of_tokens_recoverability` (leave-one-item-out kernel-ridge on
   bag-of-tokens, judged against its own permutation null -- h32's lesson: LOO on balanced labels is
   anti-predictive by construction, so 0.00 is not "clean" and 0.50 is not "chance") run over all
   five arms, then specifically `enact` vs `true` (the pair that must sit near its own null), then
   every pairwise combination as a diagnostic.
5. **Lexical floor.** `reproduce.h8_lexical_floor(gridspec=...)` was tried and does not fit: it wants
   `{factors, scenes, spans}` -- a factorial design where a composed direction is ranked against
   combinatorial distractors. Conscription has five parallel discrete arms per item and no
   candidate-ranking structure at all, so forcing the shape through would measure a fabricated
   problem, not this one. Said plainly in the script's own output rather than silently skipped. The
   minimal honest analogue is section 4 itself: predicting the arm from the bag of words is exactly
   the floor a later activation-level "which arm" readout needs to beat, computed with the same
   LOO-ridge-against-its-own-permutation-null machinery `h8_lexical_floor` uses, under one name
   instead of two.

### What it found, on the three current items

```
per-arm length: enact 119.3, report 120.7, exit 121.3, true 117.0, neutral 119.7 (grand 119.6)
  all arms within 2.2% of the grand mean -- no length flag, but n=3 items is a thin estimate
enact/exit identity: 3/3 MISMATCH -- exit does not literally start with enact
arm leakage, all 5 arms: LOO acc 1.000 (permutation null 0.103, nominal chance 0.200)
enact vs true:          LOO acc 0.500 (permutation null 0.250, gap +0.250)
```

**The enact/exit check fired, and it is not obviously a bug in the grid.** Looking at the actual
strings (`fact01`): `enact` = "...you said Polars was overkill here. **I'm back on this
tomorrow.**" and `exit` = "...you said Polars was overkill here. **You're free to disagree with me
about that.**" The shared assertion half is verbatim identical; only the trailing closer differs,
and the grid's own `_meta.closer_rule` says every arm gets *its own* matched-length closer, with
`exit`'s being the permission clause -- i.e. `enact`'s closer ("I'm back on this tomorrow") is
itself one of these matched neutral closers, not a template `exit` builds on top of. Rule 4
("`exit` = `enact` verbatim + a permission clause") and rule 5 ("every arm carries a closing
clause") are in tension the moment `enact` gets a closer of its own: a literal reading of rule 4
would have `exit` = `enact`'s full text (including `enact`'s own closer) plus another clause tacked
on, which would break the length balance rule 5 exists to protect. The grid as written resolves the
tension in rule 5's favor. This is exactly the ambiguity the checker exists to surface rather than
silently assume away -- reported with the full diff, decision left to the grid's author.

**Arm leakage is high everywhere except `enact` vs `true`, as expected by design** (`report`,
`exit`, `neutral` all read differently by construction -- third-party framing, a permission clause,
a non-claim follow-up). `enact` vs `true` reads gap +0.250 against its own permutation null, which
by the script's own >0.2 threshold is flagged. With only 3 items (6 rows for a binary LOO fit) this
is a very noisy estimate -- one item's idiosyncratic phrasing can swing it entirely -- and the
number is worth re-checking once the grid reaches its full 24 items, not read as a verdict now.

## 2. `src/lsx/core/conscription.py` -- what it does, and what it does NOT do

Render (chat template, prefix as real turns, `add_generation_prompt=True`) -> grid (one `Item` per
arm, one span covering the whole rendered text) -> extract (`extract.build_stack` locally,
`remote.build_remote_stack` per layer remotely, merged) -> paired per-item contrast, never pooled
across items. Checkpointed per base item to `.npz` + a provenance `.json` sidecar
(`docs/DELEGATION.md`'s own operational lesson: "write a provenance file next to every `.npz`").

**What it deliberately does not build: a `Claim`.** There is no instrument registered in
`lsx.core.registry` for this design (no patch layer, so `no_patch`/`random` have nothing to attach
to), and inventing one here -- with no planner spec, no adversarial review -- is exactly the
shortcut `docs/DELEGATION.md`'s flow exists to refuse. What this module DOES exercise on real data
is the `types.Arm` construction and its h49 per-item clustering verification, via `sanity_arm`,
wrapping each arm's self-pair (`X_vs_X`).

### Verified, not merely written

- **The final-token read position is real, checked, not assumed.** The brief names the trap
  directly: a chat template inserts tokens a span map could mistake for content, or drop. Checked
  empirically (`verify_offsets_cover_template`, called on every rendered text before extraction):
  Qwen2.5-1.5B-Instruct's fast tokenizer gives every templated token -- including the trailing
  `<|im_start|>assistant\n` -- a real, non-degenerate character offset, so a whole-text span does
  not silently drop it the way `extract.tokens_in_span` would if the offset had collapsed to
  `(0, 0)`. A negative-control test (`test_verify_offsets_catches_a_degenerate_offset`) confirms the
  check itself would fire if that ever weren't true.
- **Layer-0 diff-norm is exactly 0.0 for every arm pair, every item, and this is a real, expected
  finding, not a bug.** Because the read position is the LAST token of the rendered text, and that
  token (part of the fixed `<|im_start|>assistant\n` suffix) is literally the same token id
  regardless of which arm was rendered, its layer-0 embedding is identical by construction --
  layer 0 has no attention, so nothing about the arm's content has reached that position yet.
  Separation only appears from around layer 14 onward and grows to layer 28 (e.g. `enact` vs
  `neutral` diff-norm 15.2/11.4/10.8 at L14, 95.1/83.3/54.6 at L28, on the three items), which is the
  expected shape for a signal that has to propagate through attention to reach the read-out position
  -- confirms the extraction is wired correctly rather than reading a frozen embedding.
- **Batched-vs-single equivalence: 1.000000 on every item, every arm** (real local run, 5 arms per
  item, batch size 8 -- i.e. every base item's 5 arms in one batch, checked against the shortest of
  them extracted alone). Right-padding is Qwen's tokenizer default, matches the `span_policy="auto"`
  convention, and the check passed at machine precision.
- **Checkpointing round-trips and is actually skipped on re-run**: the dry run's second call to
  `run_local` over the same checkpoint directory took 0.00s and left every `.npz` mtime unchanged
  (`scripts/conscription_dry_run.py`), confirmed as a passing assertion, not eyeballed.
- **The remote path's §7 assertions run on real code, not a description of it, with no network
  call.** `tests/test_core_conscription.py::test_build_remote_stack_runs_the_full_assertion_suite_on_synthetic_data`
  calls `remote.build_remote_stack` unmodified against a `_FakeRemoteLM` (a duck-typed stand-in
  exposing `.tok`/`.padding_side`/`.lib_versions()`, left-padding, no nnsight) with only
  `remote.remote_residuals` monkeypatched to a deterministic (text, position)-keyed synthetic
  hidden-state generator. Padding convention, batched-vs-single equivalence on the shortest item,
  non-empty spans, provenance, and `stack_signature` all run for real. A companion test
  (`test_conscription_run_remote_merges_layers_and_checkpoints`) exercises `run_remote`'s per-layer
  loop, per-item merge, and checkpoint skip the same way. **This is a structural exercise of the
  remote code path, not a live check of NDIF or of `google/gemma-2-9b-it` /
  `meta-llama/Llama-3.1-70B-Instruct` themselves** -- no such deployment was contacted.

### At least one thing I got wrong, caught by testing rather than assumed clean

`sanity_arm` was first written and documented as a **determinism check**: "an arm read against
itself must sit at 0.0, because a deterministic forward on the same text twice gives the same
vector." A test that corrupted one arm's stored vector and expected `off_null` to flip failed:
`X_vs_X` computes `||v - v||` on the SAME row of the SAME extraction, which is exactly 0.0 for any
Stack whatsoever, correct or corrupted -- it is a mathematical identity, not an empirical
measurement, and it cannot detect either a non-deterministic forward (that needs two independent
extractions of the same text, which nothing here runs) or a scrambled arm index (wrong identically
on both sides of its own self-pair). Fixed by rewriting the docstring and the test to say this
plainly (`tests/test_core_conscription.py::test_sanity_arm_is_a_wiring_check_not_a_determinism_check`)
rather than deleting the discrepancy quietly. What `sanity_arm` actually demonstrates: that
`types.Arm`'s construction, `off_null` check, and `n_independent`/`clusters` bookkeeping (h49) run
correctly end to end on this module's real per-item Stack shape -- confirmed
(`n_independent=3, clusters=('fact01','limit01','refusal01'), off_null=False` on the real
extraction). With only 3 (soon up to 6-per-domain) demo items, every cluster is its own item
(`k == n`), which is the no-widening branch of `Arm._resolve_unit`; a design with several items per
declared cluster would exercise `checks.cluster_evidence`'s permutation test, and this dry run does
not have data shaped that way to reach it. Said here rather than implied by silence.

## What was NOT done, and why

- **No `Claim` and no registered instrument.** This design has no patch step and no fitted
  `Direction`; a `random`/`no_patch` control pair has nothing to attach to until one of those exists.
  Registering a "conscription" instrument in `lsx.core.registry` on my own initiative, without a
  planner spec or adversarial review, is the exact shortcut `docs/DELEGATION.md` was written to
  refuse, so it is left undone rather than invented under time pressure.
- **No live NDIF call, no `google/gemma-2-9b-it` or `Llama-3.1-70B-Instruct` weights touched.** The
  remote path is exercised structurally (above) with a synthetic backend. Its real behavior --
  actual left-padding semantics, actual layer-output shape on today's deployment, actual latency and
  per-item checkpoint economics under a real queue -- is unmeasured and stays that way until the
  study runs for real.
- **No cross-arm "which arm won" readout, direction, or effect size.** `paired_contrasts` reports
  diff-norm and cosine per item per layer per arm pair -- a `Sketch`-level diagnostic, explicitly
  not ledgerable, and not meant to be read as a finding. The full curve for the three demo items is
  in `scripts/conscription_dry_run.py`'s output, not reproduced here in full since it is not a
  result on three worked examples.
- **The grid is not finished** (3 of 24 items; `stance` domain has none yet), so every number above
  is a code-exercise on examples, explicitly not a reading on the design's actual question.
- **I did not resolve the rule-4/rule-5 tension in `enact`/`exit`.** That is the grid author's call;
  the checker's job was to surface it with the exact diff, which it does.

## Files

- `scripts/conscription_check.py` -- the pre-model checker.
- `scripts/conscription_dry_run.py` -- local end-to-end exercise (model load, extraction, paired
  contrast, checkpoint reuse); writes to `results/conscription_dry_run/` (gitignored).
- `src/lsx/core/conscription.py` -- the runner: `render_prompt`, `verify_offsets_cover_template`,
  `build_item_grid`, `run_local`, `run_remote`, `paired_contrasts`, `sanity_arm`, checkpoint I/O.
- `tests/test_core_conscription.py` -- 13 tests, no model load required (tokenizer-dependent tests
  skip cleanly if the cache is absent; the remote-path tests need neither model nor tokenizer).
