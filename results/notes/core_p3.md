# `lsx.core` piece 3: the ledger, retraction, and the §1A reproduction suite

Spec: `docs/specs/core_v1.md` v1.3, §9 (ledger and retraction) and §1A (reproduction). This is the
acceptance test for the whole core, so the honest summary goes first: **the core passes §1B and it
does not yet pass §1A.** The rediscovery harness still catches 9 of 9. Of §1A's seven target rows,
two are refused (which is the spec's own prediction and therefore a pass), two are reproduced inside
their measured tolerance, and three are deferred for reasons that are structural rather than
budgetary. A core that refuses two thirds of its own reproduction list is not finished; it is,
however, refusing for the right reasons, and each reason is named below.

Files added: `src/lsx/core/{ledger,reproduce}.py`, `tests/test_core_ledger.py`,
`results/ledger.jsonl`, `results/{posid_remote,remote_tolerance,repro_h8,repro_summary}.json`.
Modified: `checks.py` (the publication refusals), `types.py` (`stack_signature`, and piece 2's
stale-report hole), `extract.py` (the signature and `asserted_patched_logprob`), `__init__.py`.
**`docs/specs/core_v1.md` edited in three places, each marked in the text as written in after the
fact:** §2a's uncovered-modes list (the position-ids verdict), §4 (the `Selection.executed`
decision), §1A (the two measured tolerances). `scripts/` untouched.

## 1. The position-ids exposure: settled, and it is not a bug

Piece 1 found that `build_stack` passes explicit `position_ids` and no script in the repo does, and
left it open because four remote scripts batch more than one text per job. It is closed.

**Locally** (Qwen2.5-1.5B, `results/posid_local.log`), a left-padded batch of four texts of 9–35
tokens matches a batch-of-one extraction at cosine **1.000000** at every layer, on the last token
and on a mean over all real tokens, **with and without** explicit `position_ids`; right padding is
the same.

**The check is sensitive** (`results/posid_local2.log`), which is the arm that makes the first
result mean anything — a check that returns 1.0 on everything is the h6 failure. Against the default
positions:

| positions | last-token min-cos | mean-pooled min-cos |
|---|---|---|
| uniform +50 (what left padding does) | 1.000000 | 1.000000 |
| uniform +500 | 1.000000 | 1.000000 |
| scrambled | **0.690879** | 0.929778 |
| all zeros | **0.712501** | 0.925456 |

The reason is arithmetic, not luck: RoPE attention depends on position *differences*, and left
padding offsets every real token of a row by the same `n_pad`, so a uniform shift cancels in every
attention logit and leaves the value path untouched.

**Remotely** (`results/posid_remote.json`), on `google/gemma-2-9b-it` through the `tracer.invoke`
idiom that `ndif_recompose_gen`, `ndif_recompose_sweep` and `ndif_time_translation_extract` use,
left padding, four texts of 10–51 tokens: the shortest item carried **41 tokens of padding** and
matched its batch-of-one extraction at cosine 0.999979 (last token), 0.999990 (a marked span),
0.999997 (whole-text mean). Worst of twelve item × pooling pairs: **0.999943**, against the §7
threshold of 0.999.

**`ndif_factors` cannot be exposed at all**: it sets `tok.padding_side = "right"`, so positions
start at 0 on every row. That is the script behind h34-corrected and h37.

**Verdict: no logged hour falls.** Hours 27, 29, 31-corrected, 33 and 37 are not affected. Explicit
`position_ids` stay in `build_stack` as correctness that does not depend on the architecture staying
RoPE, and the two checks are the standing evidence rather than the argument.

## 2. The ledger, and the three refusals that happen at publication

`results/ledger.jsonl` is append-only, one JSON object per line, in §9's shape. `id` is the
provenance hash, so the same experiment re-run is recognised rather than duplicated and a changed
grid, code version or library version produces a new id (tested both ways). `withdraw(id, reason,
superseded_by)` appends a retraction line rather than editing the row, so the record keeps the fact
that the claim once stood; `Ledger.render()` regenerates every withdrawn claim with its reason
inline, which is the §9 feature that matters in a project with five retractions.

The three refusals are the three holes pieces 1 and 2 each named and each left open. All three are
about *publication*, not computation, which is the same line §3 draws for `Sketch`: a `Claim` can
still be built and inspected, and it cannot reach the file.

1. **`HandDeclaredCalibration`.** Piece 2 stamped `CalibrationReport.hand_declared(...)` so it could
   not pass for a measured one, and nothing acted on the stamp. Piece 2 called it the last hole; it
   is now the ledger's first refusal. Piece 1's harness cases 3, 4 and 7 keep using hand-declared
   reports, which is correct — they demonstrate that a mechanism fires, they are not results, and
   they never touch this file.
2. **`ProvenanceNotFromStack`.** `build_stack` now writes `acts_digest` and signs its own provenance
   fields (`types.stack_signature`); the ledger recomputes the signature and refuses a row without
   one, or with one that no longer matches the fields it is attached to. Editing the padding side or
   the library versions after the fact breaks the signature rather than the silence. This is not a
   security boundary and is not meant as one — anyone who reads the file can call the function. What
   it buys is that an extraction which never ran the §7 assertions cannot reach the ledger *by
   accident*, and cannot reach it at all without someone writing the forgery on purpose.
3. **`SweepNotExecuted`.** See §3.

A fourth, inherited, also fires here: `ProvenanceIncomplete`, on §9's named provenance fields.

## 3. The decision piece 2 handed over: `Selection.executed` is mandatory, at the ledger

Piece 2 built `Instrument.sweep`, which runs the sweep in the core and records the curve as a fact,
and left the prose-regex path beside it because making `executed` mandatory "would refuse every §1A
target until piece 3 re-runs the sweeps". **With the targets in front of me that is not the cost.**
Of §1A's rows, exactly one is a swept claim — h16's "peak layer 16" — and it is the row the spec
already restates as a refusal. Every other target is a single pre-registered layer, `axis=None`,
which needs no curve and is unaffected.

So the requirement costs one target that was already refusable, and it retires a check that is a
regex over the caller's own prose (`rule="we looked at the curve and quoted layer 16"` passes it, and
a test asserts that it does). It is enforced in `ledger.check_sweep_executed`, at publication rather
than at construction, so that `Instrument.sweep` is the path of least resistance rather than a
barrier to thinking. Written into the spec at §4.

