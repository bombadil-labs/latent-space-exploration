# `lsx.core` piece 3: the ledger, retraction, and the §1A reproduction suite

Spec: `docs/specs/core_v1.md` v1.3, §9 (ledger and retraction) and §1A (reproduction). This is the
acceptance test for the whole core, so the honest summary goes first: **the core passes §1B and it
does not yet pass §1A.** The rediscovery harness still catches 9 of 9. Of §1A's eight target rows
(h8's composed test and its three single-factor lenses counted separately, because they are graded
separately), **four reproduce inside their measured tolerance, three are refused and two are
deferred** — h8's three lenses share one refusal. Two of the three refusals are the rows §1A itself
restates as refusals, so those are the acceptance test passing rather than failing. The third is not:
h8's single-factor lenses reproduce to the logged decimals and cannot be published, because the
battery has never had a permutation arm.

Every number that was re-derived came back to the logged decimals — h8 composed 2.8056 against 2.81,
h4 1.6875 against 1.69, h14 −0.1111 against −0.111, h29 0.8364 against 0.84 twice over. Nothing
failed its tolerance, so nothing is withdrawn from the writeup on §1A's rule. What the core is short
of is *coverage*: three of §8's six instruments are unbuilt and two §1A rows need them.

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

## 4. The two measured tolerances

### 4a. Remote re-run spread: 0.000, and what that means

**What was run.** h29's re-imposed era shift at scale 3.0 on Gemma-2-9B-it — `ndif_recompose_sweep.py`
unchanged, the same grid, the same cached direction stacks — twice, end to end, concurrently, nothing
different between them. 144 generation jobs and 20 scoring jobs in total.
`results/remote_tolerance.json`.

| | run 1 | run 2 | logged (h29) |
|---|---|---|---|
| generations attempted / scored | 72 / 55 | 72 / 55 | 72 / 55 |
| era reads as target | 0.83636 | 0.83636 | 0.84 |
| leaves e1 | 0.90909 | 0.90909 | 0.91 |
| theme kept | 0.52727 | 0.52727 | 0.53 |
| items whose era readout flipped between runs | — | **0 of 55** | — |
| continuations character-identical between runs | — | **55 of 55** | — |

**Measured spread: 0.000.** The 17 lost jobs are the *same* 17 in both runs, which says the loss is a
property of the generation rather than of the queue — worth knowing, because h29's note attributed
the rising loss rate to scale and left it there.

**Tolerance: ±0.018.** A tolerance of exactly zero is unusable: it would refuse a re-run that differs
by a single item. The number is the statistic's own *resolution*, one item in 55. That is a
consequence of the measurement, not a choice made to fit: the spread is smaller than the resolution,
so the resolution binds. §1A's guessed ±0.03 was, as it suspected, looser than the noise — by a lot
more than it expected.

**What this does not bound, stated because a tolerance that overclaims is worse than none.** Both
runs hit the same pinned deployment inside one session. Cross-session and cross-deployment variation
— a redeployment, different bf16 kernels, a different replica — is unmeasured. The next remote grade
should re-measure rather than inherit 0.018.

### 4b. `readout_shift`'s paraphrase-noise interval: 0.319 per item, and piece 2's stand-in was 9× too tight

§2a says the gain must be "reported with the paraphrase-noise interval on that difference", and
piece 2 could not supply one: the instrument's own null spread is 0 by construction, so it shipped
`sqrt(2/d)` as a declared stand-in and said so.

**Measured on the stage-14 configuration** (`results/repro_summary.json`). The gain is per case
`1{model readout reads as the target era} − 1{pass-through readout reads as the target era}`; within
each (e1, e2, theme) cell the four scenes are four wordings of the same content, so the pooled
within-cell spread of the gain is paraphrase noise and nothing else.

| | per-item sd | 3σ arm tolerance at n=72 |
|---|---|---|
| piece 2's stand-in, `sqrt(2/d)`, d=1536 | 0.0361 | 0.0128 |
| **measured paraphrase noise** | **0.3191** | **0.1128** |

**The stand-in was 8.8× too tight, and that changes a reading.** Under the stand-in, h14's gain of
−0.1111 would sit nearly nine tolerances from zero and read as a significant *negative* effect. Under
the interval §2a actually asks for, the 3σ band is ±0.1128 and the gain of −0.1111 falls just inside
it: the model arm is **not distinguishable from the arithmetic in either direction**. That is the
h40 verdict stated correctly rather than overstated — "no gain over a norm-matched pass-through",
which is exactly how §1A restates the target, and not "the model is worse than doing nothing", which
is what a too-tight interval would have licensed. It enters the registry as
`config={"readout_sd": 0.3191}`.

## 5. §1A, target by target

Verdicts as graded: `results/repro_summary.json`, ledger rows in `results/ledger.jsonl`.

| target | logged | reproduced | tolerance | verdict |
|---|---|---|---|---|
| h8 composed, Qwen-1.5B | 2.81/18, no-patch 9.50 | **2.8056/18, no-patch 9.5000** | ±0.02 local | **reproduced** |
| h8 era / voice / tense lens | 1.25 / 1.24 / 1.03 of 3 | 1.250 / 1.236 / 1.028 | ±0.02 local | **REFUSED** (`MissingArm('permutation')`) |
| h4 role lens, held-out domains | 1.69/6, random 3.25, chance 3.5 | **1.6875/6**, random 3.406, permutation 3.542, no-patch 3.500 | ±0.02 local | **reproduced** |
| h14 gain over norm-matched pass-through | −0.111 (model 0.889, pt 1.000) | **−0.1111** (model 0.8889, pt 1.0000) | ±0.1128 paraphrase, measured | **reproduced** |
| h16 relation selector, "peak layer 16" | 1.73/6 at layer 16 | not published | — | **REFUSED** (`SelectionOnScoringData`) |
| h39 Gemma clock, corrected | 0.501 / 0.767 / 2.50 raw | not published | — | **REFUSED** (leaky grid, and `discrimination` is unbuilt) |
| h29 era shift in generation, 3× re-imposed | 0.84 / 0.91 / 0.53, lex 0.30, n=55/72 | **0.8364 / 0.9091 / 0.5273, n=55/72**, twice | ±0.018 remote, measured | **deferred** (no instrument; no `Stack`) |
| h37 70B matched pair selector | 1.50 / 1.06 @26, no-patch 2.00 | not re-derived | ±0.018 remote | **deferred** (stacks not cached) |

Four reproduce to the logged decimals. Three are refused, and two of those three (h16, h39) are the
rows §1A itself restates as refusals, so they are the acceptance test passing. Two are deferred.

**The refusals, each on its own terms.**

- **h16.** The reported value was chosen along `layer` by argmax over the scoring data. Refused at
  `Claim` construction, before any number is computed. The restated form — the full curve — is not
  graded here: `role_rank` comes from `operate.holdout_eval`, which returns aggregates, and a
  per-item path through `Instrument.sweep` is unwritten. The refusal stands; the reproduction is
  unfinished work.
- **h39.** Two independent reasons, which is worth saying because either alone would do. `discrimination`
  is declared in §8 and not built (§11.2 deliberately defers it), so no `Claim` can be graded through
  it at all; and the grid is flagged leaky, so §6 requires gain over the measured floor and the v3
  stacks that would measure it are not cached (§11.3 says defer, do not re-extract).
- **h8's three single-factor lenses — the refusal I did not expect.** The numbers reproduce exactly
  (1.250 / 1.236 / 1.028 against 1.25 / 1.24 / 1.03; the no-patch arms land on 2.00 / 2.00 / 1.50 to
  the digit, and the random arms on 2.00 / 2.25 / 1.444 against h8's logged tense random of 1.44).
  They cannot be published, because `selector` requires random, **no_patch and permutation**, and
  h8's battery has never had a permutation arm. The tempting move was to hand the no-patch array in
  under the name `permutation`; that is the fudge this core exists to stop, so the arm is absent and
  the refusal stands. The composed test survives because `composition` requires only random and
  no_patch — and it is the stronger claim anyway, 2.8056 against a null of 9.50 that its own
  no-patch arm hits exactly.

**The two deferred, and why neither is a budget problem.**

- **h29** reproduces *perfectly* — twice, bit-identically, matching the logged 0.84 / 0.91 / 0.53 /
  0.30 — and still cannot be a row. Its statistic is a **top-1 accuracy** over three era directions.
  The registry ships a rank (`selector`), a joint rank (`composition`) and a projection gain
  (`readout_shift`); none of them is an accuracy, and none of their calibrations bounds one. §8 lists
  one unfinished instrument (`generality`, which has no null); **this is a second**, and it has a null
  — 1/3 — but no instrument and no battery. Separately, it comes from a frozen script, so its
  provenance is not a `Stack` and the ledger refuses it on that ground too.
- **h37**'s Llama-3.1-70B direction stacks are not cached and §11.3 says defer rather than
  re-extract. It is *not* blocked by position ids: `ndif_factors` pads right.

**Retraction.** `results/ledger.jsonl` holds three standing rows and no withdrawals, because nothing
re-derived here failed its target. `withdraw()` is implemented and exercised in
`tests/test_core_ledger.py` (append → withdraw → render, the reason and `superseded_by` inline, the
file append-only so the history of the retraction survives, and a withdrawal of an unknown id
refused). Writing a retraction into the ledger to demonstrate the feature would have been a
fabricated row, so there is not one.

## 5b. A bug the reproduction suite found in piece 2, and the fix

Running real targets through the registry surfaced one defect that no calibration test could have:
**`registry.CALIBRATION_KEYS` held one key per instrument, and the key depends on the config.**

The key hashes the statistic's source, its declared null and its declared invariances — and the
declared null is a function of the configuration. A 3-candidate `selector` declares 2.00 and a
6-candidate declares 3.50, so they hash differently and *both are correct*. With one key per
instrument, any selector claim whose candidate count was not the registry default was refused as
`CalibrationStale` while holding a freshly measured, passing, correctly-keyed report. That is h8's
three single-factor lenses, h34's battery and h37's — every 3-candidate selector in the project.

It was invisible to piece 2 because its own tests build the default configuration, and the failure
is a *false refusal*: the core says no for a reason that sounds right. Fixed: the table is a set of
currently-valid keys per instrument, added to by `Instrument.calibrate()`, and a genuinely stale
report is still refused because its key is in none of them.
`tests/test_core_ledger.py::test_a_selector_with_a_non_default_candidate_count_is_not_stale`.

(The h8 lenses are still refused, for `MissingArm('permutation')`, which fires first and is the
real reason. This bug was sitting behind it.)

## 6. Piece 2's unprotected list: what piece 3 closed, and what it did not

**Closed.**

- *A hand-declared calibration report still passes.* Closed at the ledger (`HandDeclaredCalibration`).
  Piece 2 called this the last hole and it was.
- *Nothing ties a `Claim`'s provenance to a real `Stack`.* Closed (`ProvenanceNotFromStack`), with the
  signature covering the activation digest as well as the recorded fields, so editing the padding
  side or the library versions after the fact breaks the signature.
- *The staleness check needs `lsx.core.instruments` to have been imported.* Closed: `Claim` now
  populates `registry.CALIBRATION_KEYS` itself the first time it finds the table empty, so
  `from lsx.core.types import Claim` alone no longer lets a stale measured report through. A test
  would need a fresh interpreter to see the difference, so this one rests on reading the code.
- *`readout_shift`'s arm tolerance is an estimator stand-in, not the paraphrase interval.* Measured
  (§4b), and it enters the registry through `config={"readout_sd": ...}` exactly where piece 2 said
  it would.
- *Whether `Selection.executed` becomes mandatory.* Decided and enforced (§3).
- *The position-ids exposure* (piece 1's, carried into `RESULTS.md` hour 43 as an open question).
  Measured on both paths and closed (§1).

**Left, with reasons.**

- *`Probe` is a shell; nothing NDIF-side goes through the asserted path.* Still true. Piece 3 ran the
  §7.2 equivalence check on the remote path by hand and recorded the result, which is not the same as
  routing remote forwards through `build_stack`. The moved-candidates assertion still guards no NDIF
  call. This is the largest remaining gap in §7 and it is why h37 could not have been graded through
  the core even if its stacks had been cached.
- *Library versions are recorded, never compared.* Half closed: a changed `lib_versions` now produces
  a different ledger `id` rather than silently overwriting a row (tested), which is what §9 asks for.
  Nothing *refuses* a stack built under a different transformers version, and h36 was a transformers
  change.
- *`selector` and `composition` share a ranking statistic*, so their calibrations are not independent
  evidence. Unchanged.
- *Sensitivity is tested with planted signal in the readout's own geometry.* Unchanged; the battery
  bounds the instrument, not the hypothesis.
- *`crosstalk`, `depth_gain` and `generality` carry unmeasured null spreads.* Unchanged, and piece 3
  adds a fourth to the list: **`discrimination`**, which §8 names as h39's instrument and which is
  declared but not built. That is why h39 cannot be graded even in its restated form.
- *`EffectSize` is recomputed only for a verdict disagreement.* Unchanged.

## 7. Does the core pass its own acceptance test?

**§1B: yes.** The rediscovery harness still catches 9 of 9, unchanged by anything piece 3 added
(`tests/test_core_rediscovery.py`). Piece 3 did not add a tenth case, and it should have: the three
publication refusals are tested directly rather than through the harness, so there is no "fed a
known-bad configuration, the core refuses without being told what to look for" case for them.

**§1A: no, and the shortfall is specific.** Four of eight target rows come back inside their
measured tolerance and three are refused, two of those exactly as the spec predicted. What is
missing is not accuracy — every number that was re-derived reproduced to the logged decimals, and
the two that were not re-derived were not re-derived for stated structural reasons. What is missing
is **coverage**: three of the six instruments §8 names are still unbuilt, and two §1A rows need
them. §1A is a pre-registered criterion and it is not met.

The honest summary is that the core reproduces everything it is equipped to reproduce, and refuses
everything it is not — including three numbers that are in `WRITEUP.md` and `RESULTS.md` today.

## 8. What remains, in the order I would do it

1. **`discrimination` and an accuracy instrument.** Two §1A rows (h39, h29) are blocked on unbuilt
   instruments, not on data. `discrimination` needs the floor-referenced form §8 names; the h29
   statistic needs a top-1-accuracy instrument with a declared null of 1/k and a battery.
2. **Route remote forwards through the asserted path.** `Probe` is still a shell. The
   moved-candidates assertion — the one that caught h34 — guards no NDIF call, and until it does, no
   remote target can be published even with cached stacks. This is the largest remaining §7 gap.
3. **A per-item `role_rank` path** so h16 can be graded in its restated form: the curve computed by
   `Instrument.sweep`, the value reported as the curve.
4. **h37's stacks**, if a re-extraction is ever authorised; §11.3 currently says defer.
5. **A harness case for each publication refusal**, so §1B covers the ledger and not only the `Claim`.
6. **Re-measure the remote tolerance across deployments.** 0.018 bounds within-session noise against
   a pinned deployment and nothing wider.

## 9. Run record

Wall clock ≈ 2 h 40 min, most of it two long CPU jobs and two NDIF runs that overlapped them.
NDIF: 4 jobs for the position-ids check, 164 for the two h29 re-runs (144 generations, 20 scoring),
all on `google/gemma-2-9b-it`; no 405B, nothing downloaded. Local: ~2 600 patched log-probability
forwards for h8 (22 min), 144 for h14, ~1 200 for h4 (11 min), all Qwen2.5-1.5B on 4 shared CPUs.
`pytest -q tests/` — **99 passed**. Harness 9 of 9. `scripts/` untouched; `RESULTS.md`,
`WRITEUP.md`, `VISION.md`, `README.md` and `docs/ALGEBRA.md` untouched.
