# `lsx.core` piece 4: the coverage phase 1's gate was short of

Spec: `docs/specs/core_v1.md` v1.3, §8 (the two unbuilt instruments two §1A rows needed), §7 (the
remote path), §4 (h16's curve, computed rather than asserted) and §1B (the three publication
refusals, now in the harness). Piece 3 ended with **§1B passing and §1A failing on coverage**: every
number the core could compute reproduced to the logged decimals, and three of §8's six instruments
were unbuilt. This piece is that coverage.

The honest summary first. **§1B passes and is wider — 12 of 12 rather than 9 of 9.** **§1A still
does not pass**, and the reason has changed completely: it is no longer coverage. Every instrument
a §1A row needs now exists and is calibrated; what stops four of the ten rows is their own
batteries — arms that were never run — and two remote targets whose stacks are not cached.
**Nothing failed its tolerance, so nothing is withdrawn on §1A's rule.** Two numbers that are in
`RESULTS.md` today turned out not to be what they were taken to be, and §5 and §6 say which.

Files added: `src/lsx/core/remote.py`, `tests/test_core_{instruments_p4,remote}.py`,
`results/{ndif_probe_asserted.json,repro_h16.json,repro_summary_p4.json}`. Modified:
`planted.py`, `registry.py`, `instruments.py`, `checks.py`, `types.py`, `rediscovery.py`,
`reproduce.py`, `__init__.py`, `results/ledger.jsonl`. `docs/specs/core_v1.md` edited in §5 and §8,
each marked in the text as written in after the fact. `scripts/` untouched; `RESULTS.md`,
`WRITEUP.md`, `VISION.md`, `README.md` and `docs/ALGEBRA.md` untouched.

---

## 1. The two new instruments

| | `top1_accuracy` | `discrimination` |
|---|---|---|
| statistic | fraction of items whose argmax over k candidates is the target, **ties split 1/T** | mean over subjects of Spearman(scalar readout, ordered target), **ties merged at a numerical bound** |
| null | **1/k** — 0.3333 at k=3 | **the measured stimulus floor**, not chance (§8) |
| per-item sd (closed form) | Bernoulli: sqrt(k−1)/k = **0.4714** at k=3 | 1/sqrt(m−1) = **0.3536** at m=9 |
| per-item sd (Monte Carlo from the statistic) | **0.4693** (0.4 % low) | **0.3381** (4.4 % low) |
| 3σ arm tolerance | **±0.1907** at n=55, ±0.2236 at n=40, ±0.1667 at n=72 | **±0.3750** at n=8 subjects, ±0.1677 at n=40 |
| claimed invariances | scale, rotation | scale (one scalar on the whole stack), rotation, **monotone reparameterisation of the target** |
| declared NOT invariant | candidate count | per-cell rescaling (measured: Δ 0.0493) |
| required arms | random, no_patch | shuffled_stimulus, floor |
| reproduces | h29 | h39, as gain over the measured floor |
| calibration key | `e37df6b753fce8c4` | `65af3fe21d5d2a87` |

### Nulls, declared and justified

**`top1_accuracy`'s null is 1/k, and it is not a rank midpoint** — the brief asked for this to be
declared and justified, and the justification is that the two move in opposite directions. A rank's
null grows with the candidate count, (k+1)/2; an accuracy's shrinks, 1/k. An accuracy read against a
rank's null would look like a large effect at every k > 2, and the error would grow with the design.
The statistic is bounded in [0, 1], so its per-item spread is Bernoulli rather than uniform-rank, and
that is where the ±0.19 band at h29's n = 55 comes from.

**Ties split the hit, 1/T.** This is stage 34's mid-rank rule in the accuracy family, and it is not
hypothetical: written the obvious way, `argmax == target`, a wholly tied field reads **1.0 if the
target is the first of the tied block and 0.0 if it is the second**. Measured, on
`planted.rank_partial_tie`: the shipped statistic reads 0.5 for both labellings and the naive one
reads 1.0 and 0.0. Nothing about the model differs between those two numbers. The rejected variant
is kept as `instruments.naive_top1_accuracy` and a test pins the separation.

**`discrimination`'s null is the measured floor, and its battery runs at chance.** These are
different numbers, and keeping them apart is the one structural change piece 4 had to make to the
registry (`InstrumentSpec.calibration_null`). The declared null — where the arms sit — is
`config["floor"]`, a measurement of *the caller's grid*, because §6 requires gain over the measured
floor on a leaky grid and h39's grid is flagged. The battery runs on a synthetic fixture that knows
nothing about that grid, so it must run at the statistic's own chance value of 0. Feeding the floor
into `calibrate()` would fail the noise test for every floor except zero, and — since the
calibration key hashes the declared null — would demand a fresh battery for every floor of an
unchanged statistic. That is piece 3's config-dependent-key bug, one level along, which is exactly
the shape the brief warned about.

**The limitation this leaves, stated rather than left implicit: the battery bounds the statistic and
does not check the caller's floor.** A `discrimination` claim rests on someone having measured the
right floor. What the core does enforce is that a floor is *there*, that it is a real arm with
per-item values, and that a leaky grid reports the gain and never the raw score.

---

## 2. Calibration: every result

Both instruments pass the full six-test battery. `results/calibration/*.json`.

```
top1_accuracy  [e37df6b753fce8c4] 6 tests PASS  noise 0.3356±0.0192 vs null 0.3333
               | sensitivity [0.390, 0.465, 0.630, 0.808] over planted 0.02/0.05/0.1/0.2
               | self-floor 0.2900 | scale Δ=0, rotation Δ=0 | known-zero 0.3333 exactly
discrimination [65af3fe21d5d2a87] 6 tests PASS  noise 0.0086±0.0245 vs null 0.0
               | sensitivity [0.188, 0.389, 0.638, 0.861] | self-floor 0.0363
               | scale Δ=0, rotation Δ=0, monotone_target Δ=0, cell_rescale (not claimed) Δ=0.0493
               | known-zero 0 exactly
```

Every tolerance in those lines is measured. `null_tol` is the instrument's own 3σ band at the
calibration fixture's n (0.0707 for `top1_accuracy` at n = 400, 0.0750 for `discrimination` at 200
subjects), never a flat number — piece 2's lesson, which its own first draft broke.

**The sensitivity curves were saturating and the amplitudes were re-measured.** The first amplitude
set, `(0.1, 0.2, 0.4, 0.8)`, gave `[0.630, 0.808, 0.985, 1.000]` for the accuracy and
`[0.638, 0.861, 0.964, 0.994]` for the discrimination. Both are monotone, both pass — and both have
their top step **on the ceiling**, where "monotone in the planted size" is satisfied by a statistic
that has stopped responding. The shipped amplitudes span the part of the curve where the statistic
is still moving. A test asserts the top of each curve stays below 0.99, so this cannot quietly come
back.

**`monotone_target` is the invariance nobody had tested.** h39's target is `log Δt`, and the number
must not depend on the choice of log base, or on whether the axis is Δt, log Δt or grid index. A
Pearson correlation moves under all three; the rank correlation shipped here does not (Δ = 0). It is
worth a line because the choice of `log` in h31/h39 was never justified anywhere, and now it does not
need to be.

---

## 3. Three bugs in my own code, all found before publication

The brief said to assume this code carries a version of the bug it is fixing. It carried three, and
each was found by a test the spec asks for rather than by inspection.

### 3a. The known-zero point that read zero for the wrong reason

`discrimination`'s known-zero fixture gives every level of a subject the **identical** activation
vector, so the readout is constant, every level is tied, and the correlation must be 0. It read
`−8.9e-18` and passed.

It passed for the wrong reason. Identical vectors run through one batched matmul come back differing
by **4.4e-16** — BLAS does not promise the same summation order for every row of a batch — and
Spearman does not care how small a difference is, it ranks it. So a **wholly dead readout scored
±0.548 per subject**, and the instrument read zero only because the signs happened to cancel in the
mean. An asymmetric rounding pattern, or real near-tied activations (which is what a dead readout
looks like on real data), would have produced a clean correlation out of floating-point noise.

Closed with `checks.dot_tie_atol`: values are merged when they differ by less than the float64
dot-product error bound, `4 · d · eps · max|score|`. That is computed from the arithmetic that
produced the numbers, not chosen — at Gemma's d = 3584 with projections of order 20 it is ~1.6e-11,
eleven orders of magnitude below anything a readout could mean. A flat epsilon here would have been
piece 2's mistake a third time.

The aggregate was right and the statistic was wrong, which is the whole argument for per-item
statistics and for a known-zero test that checks the items.

### 3b. The calibration key that did not move when the arithmetic did

Fixing 3a changed what `discrimination_rho` computes and **left its calibration key identical**,
because `calibration_key` hashed the statistic's top-level source and `discrimination_rho` is one
line over `discrimination_per_item`. Every cached report stayed valid across a real change to the
measurement. The same hole covered `midrank` and `cosine_scores`, which three shipped instruments
delegate to: editing the tie rule that h34 turned on would not have re-calibrated anything.

The key now hashes the **source closure** — every `lsx.`-defined function the statistic calls,
transitively. Still not a repo-wide version, which §5 forbids for good reason: `readout_shift`'s
closure does not contain `midrank`, so editing the ranking re-calibrates the rank instruments and
nothing else.

**And the fix for that hole had the same hole.** The first closure walker read `co_names` off the
statistic's own code object, and `selector_rank` calls `midrank` **inside a list comprehension**,
which compiles to its own code object whose names do not appear in the enclosing `co_names`. So the
walker found `cosine_scores` and missed `midrank`. It walks nested code objects now, and a test
asserts that `midrank` is in the referenced set and *not* in the top-level `co_names`, so the
regression test fails if someone moves the call out of the comprehension and makes it vacuous.

That is the fourth time in this project a fix has carried a version of the bug it was fixing, and
the second time inside a single piece.

### 3c. The per-item spread taken from the fixture's type

`top1_accuracy` reuses `RankFixture` — deliberately, because an accuracy over k candidate directions
and a rank over k candidate directions read the same scores. `Instrument._per_item` dispatched on
the fixture's **type**, so it would have handed an accuracy the per-item spread of a mid-rank:
**0.816 where the truth is 0.471** at k = 3, an arm band 1.7× too loose on every h29-shaped claim.
The per-item function is now passed explicitly by each builder and the type-sniffing is a fallback.

Worth naming as a class: reusing a fixture across instruments is good economy and it lets the
fixture decide what an item is. It is also why `selector`, `composition` and `top1_accuracy` now all
rest on `cosine_scores`, so an error there breaks three instruments at once and their calibrations
are not independent evidence. Piece 2 named that hazard for two; this piece adds a third and does not
pretend otherwise.

### 3d. And one in a generator

`planted.rank_partial_tie` was written before the statistic, as the rule requires. It made the first
two candidate directions identical and stopped there — so the tied pair did not necessarily **win**,
four unrelated candidates were still in the race, and the statistic read 0.091 rather than 0.5. A tie
that is not at the top is not the h34 configuration. The generator was corrected by the statistic it
was written to test, which is that ordering working rather than failing: had they been written the
other way round, the fixture would have been built to whatever the statistic already did.

---

## 4. The remote path: `Probe` is not a shell any more

This was the largest remaining structural hole, and all three previous pieces closed with the same
sentence about it. `src/lsx/core/remote.py` puts every NDIF forward through the §7 assertions:
padding side read back from the remote tokenizer after setting it, batched-vs-single equivalence on
the **shortest** item, non-empty spans, the block output resolved by type and never by index, and
provenance signed by `types.stack_signature` exactly as a local stack's is — so a remote claim
reaches the ledger on the same terms as a local one and no others.

`types.Probe` now refuses a function that is not one of the asserted forwards
(`checks.UnassertedForward`). Piece 1 wrote it as a dataclass holding a callable; a type that names a
contract without enforcing it reads as a guarantee, which is worse than no type.

**Measured on a live deployment**, `google/gemma-2-9b-it`, 7 jobs, 24.4 s
(`results/ndif_probe_asserted.json`):

| | |
|---|---|
| padding side, read from the remote tokenizer | **left** |
| shortest item's padding | **13 tokens** of 23 |
| batched vs batch-of-one on that item | **0.9999925** (threshold 0.999) |
| whole-tensor patch | moved **3/3** candidates — positive control |
| **the h36 idiom `output[0][:] = ...`** | moved **1/3**, deltas `[−27.31, 0, 0]`, **`MovedCandidates` fired** |
| no-patch (zero vector) arm | moved 0/3; assertion skipped, not inverted |
| `Probe` on a bare callable | `UnassertedForward` |

That fifth row is the point. **This is the first time this project has caught the h36 bug on a real
remote call.** It was found on NDIF, it was fixed in the callers, and until now the assertion that
detects it had never run against a deployment — piece 1 reproduced it on a local fixture and pieces
2 and 3 did not touch it. The idiom is still live: on today's deployment Gemma-2's decoder block
returns a bare tensor, so `output[0]` is batch row 0, and the very first smoke test of this session
captured `(7, 3584)` from a batch of two without complaint.

**Two things the deployment taught, both recorded where they bite.** A trace block's body is
serialised and executed remotely, so a name bound inside the `with` is unbound when it returns — the
saved tensors come back through `backend.wait(tracer)` keyed by name. And a trace block may not
reach into a non-whitelisted module: `checks.resid(...)` inside a trace is refused with *"Module
lsx.core.checks is not whitelisted"*, and so is any attribute path through an instance of a class
defined in `lsx.core.remote`. So the tuple-or-tensor resolution is written out inline in each trace
block; `tests/test_core_remote.py` pins the duplicate to `checks.resid` on a tensor, a tuple and a
batch, and greps the module to fail if any future trace block indexes `output[0]` outside the
harness's deliberate reproduction of the bug.

**What this does not do.** It does not re-grade h37: those stacks are still uncached and §11.3 still
says defer. It does re-classify why — h37 is no longer blocked on the remote path, only on data.

---

## 5. h16: the curve reproduces, and grading it found two numbers that were not what
they were taken to be

§1A restates h16's target as the full layer curve, because "peak layer 16" was an argmax over the
scoring data. `operate.holdout_eval` returns fold means, so the curve could only ever be *asserted*
to the core; `reproduce.h16_role_rank` computes h16's statistic **per item**, and
`Instrument.sweep` runs the sweep, which is what `Selection.executed` has required at the ledger
since piece 3.

**The curve reproduces to the logged decimals**, from a fresh extraction of all 240 rotated prompts
through `build_stack`:

| layer | 0 | 4 | 10 | 16 | 20 | 24 | 28 |
|---|---|---|---|---|---|---|---|
| logged (h16) | 2.73 | 2.20 | 1.98 | 1.73 | 1.86 | 2.39 | 2.73 |
| recomputed | 2.732 | 2.200 | 1.984 | 1.732 | 1.863 | 2.389 | 2.734 |

The full 15-layer curve, step 2:

| layer | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| role_rank | 2.732 | 2.463 | 2.200 | 2.184 | 2.101 | 1.984 | 1.901 | 1.814 | 1.732 | 1.790 | 1.863 | 2.159 | 2.389 | 2.492 | 2.734 |

**And it exposed a number in `RESULTS.md` that disagrees with this repo's own artefact.** §1A's
target is **2.21**, and the mean of this curve is **2.1692**, which is 0.04 outside
the ±0.02 local tolerance. That looked like a failure, so it was localised before it was judged:
`results/stage3_qwen1.5b_v2_rolecentered.json` — h16's own saved per-layer output — was re-read and
averaged.

| | value |
|---|---|
| logged JSON, mean over all 30 pairs × 15 layers | **2.1692** |
| this core, same 15 layers | **2.1692** |
| worst per-layer disagreement between them | **0.0001** |
| logged JSON, mean over the 8 step-4 layers | 2.2065 |
| this core, same 8 layers | **2.2065** |
| `RESULTS.md` hour 16 / §1A / `docs/INSTRUMENTS.md` §2 | **2.21** |

So **the curve reproduces to four decimals at every one of the fifteen layers**, and the published
2.21 is the same curve averaged over a **step-4** layer sweep while the repo's saved JSON is a
**step-2** sweep reading 2.17. Both are correct computations; they are different aggregates of one
curve, and the two have been used interchangeably — `stage3_v2.log` printed 2.21,
`stage3_v2_layers.log` printed 2.17, and the writeup quotes 2.21. Graded against the aggregate it
actually is, the target reproduces at 2.2065 against 2.21, |Δ| =
0.0035, inside ±0.02.

This is what "report the curve" buys that "quote the number" does not, and it is the second time in
this piece that an aggregate was right while the thing under it was not what it was taken to be.

**The arms, and a tolerance that was measured and still wrong.**

| arm | value | null | i.i.d. band at n=108000 | cluster band |
|---|---|---|---|---|
| random (matched-norm Gaussian prediction) | 3.4790 | 3.50 | ±0.0156 | ±0.0386 |
| no_patch (training-mean target) | 3.4678 | 3.50 | ±0.0156 | ±0.0789 |
| permutation (h16's own null) | 3.5168 | 3.50 | ±0.0156 | ±0.0736 |
| semantic null: role identity retained | 1.3448 | — (h16 logs 1.37) | — | — |

Under the i.i.d. band **all three plumbing arms are refused** — each sits 0.02–0.03 from chance and
the band is ±0.0156. That band is *measured*, and it is still wrong, which is
piece 2's finding in reverse: piece 2 replaced a flat 0.15 that fired on 55 % of clean arms, and
this is a measured band that fires on clean arms because **its n counts repetitions rather than
evidence**. The same 240 prompts are re-ranked for 30 ordered role pairs at 15 layers, giving
n = 108000 scores over 40 independent leave-one-domain-out fits. The band shipped
is 3σ on the between-domain spread of each arm's own means — a cluster-robust standard error, with
the independent unit read off the design rather than off the array length. Both bands are recorded
in the claim's notes, because the point is that the i.i.d. one is wrong here and not merely
inconvenient.

**Verdict:** the claim CONSTRUCTS and publishes.

The semantic null §4 names for exactly this target is computed rather than asserted:
role-identity-retained reads 1.3448 against h16's logged 1.37, so "the
lens can read which role a vector is without carrying any relation" is a number in the claim and not
a remark beside it.


---

## 6. §1A, target by target

| target | source | logged | reproduced | tolerance | verdict | note |
|---|---|---|---|---|---|---|
| h8 three-factor battery, composed | h8 (re-verified h39) | 2.81/18, no-patch 9.50 | 2.806/18, no-patch 9.500 | +-0.02 (local, §1A) | **reproduced** | through `composition` against its declared null of 9.50; |delta| 0.004; PUBLISHED to the ledger |
| h8 era lens | h8 | 1.25/3 | 1.250/3 | +-0.02 | **REFUSED** | MissingArm: selector requires arms ['random', 'no_patch', 'permutation']; missing ['permutation'] |
| h8 voice lens | h8 | 1.24/3 | 1.236/3 | +-0.02 | **REFUSED** | MissingArm: selector requires arms ['random', 'no_patch', 'permutation']; missing ['permutation'] |
| h8 tense lens | h8 | 1.03/3 | 1.028/3 | +-0.02 | **REFUSED** | MissingArm: selector requires arms ['random', 'no_patch', 'permutation']; missing ['permutation'] |
| h14 era shift, gain over norm-matched pass-through | h14 (withdrawn at h40) | -0.111 (model 0.889 vs pass-through 1.000) | -0.1111 (model 0.889 vs pass-through 1.000) | +-0.1128 (paraphrase-noise, measured) | **reproduced** | zero-shift error 0; random arm +0.000; PUBLISHED to the ledger |
| h4 role lens, held-out domains | h4 | 1.69/6 (n=48), random 3.25, chance 3.5 | 1.688/6 (n=48), random 3.406, permutation 3.542, no-patch 3.500 | +-0.02 (local, §1A) | **reproduced** | |delta| 0.002; the `permutation` arm h4 never had is computed here; PUBLISHED to the ledger |
| h16 relation selector, 'peak layer 16' | h16 | 1.73/6 at layer 16 (curve mean 2.21, null 3.5) | not published | n/a -- refused before grading | **REFUSED** | SelectionOnScoringData -- the reported value was chosen along 'layer' by 'peak layer 16 (argmax over the sweep)' on the scoring data. Report the c |
| h16 relation selector, as the full layer curve (§1A's restatement) | h16 | §1A says 2.21; the repo's own stage-3 JSON says 2.1692 over its 15 layers and 2.2065 over the 8 step-4 layers | 2.1692 over 15 layers, 2.2065 over the step-4 8; permutation 3.517, mean-target 3.468, random 3.479, role-identity-retained 1.345 | +-0.02 (local, §1A); arm bands cluster-robust over 40 domains, not the i.i.d. +-0.0156 at n=108000 | **reproduced** | curve computed by Instrument.sweep, 15 points; worst per-layer disagreement with the logged JSON 0.0001. |delta| 0.0000 against the JSON's own step-2 mean and 0.0035 against §1A's 2.21, which is the SAME CURVE averaged over step-4 layers. PUBLISHED to the ledger. |
| h29 era shift in generation, 3x re-imposed | h29 | 0.84 era->target / 0.91 leaves e1 / 0.53 theme kept, lex 0.30, n=55/72 | 0.8364 / 0.9091 / 0.5273, lex 0.30 (n=55/72) | +-0.0182 remote (measured, piece 3); arm band +-0.1907 at n=55 | **REFUSED** | MissingArm -- top1_accuracy requires arms ['random', 'no_patch']; missing ['random', 'no_patch']; at the ledger, separately: ProvenanceIncomplete |
| h39 Gemma clock, corrected | h39 | 0.501 shared-variance / 0.767 Spearman / 2.50 phrase-only ratio (raw) | read back from the logged JSON: 0.500 / 0.767 / 2.501 -- NOT re-derived | n/a -- not gradable without the floor | **deferred** | `discrimination` is BUILT and calibrated (piece 4), so reason 1 of piece 3's two is closed. The grid is flagged leaky (interval recoverable 0.92 (permutation null 0.40); FLAGGED [...), so §6 requires gain over the MEASURED floor, and the Gemma v2/v3 stacks are not cached (§11.3: defer, do not re-extract). The logged JSON carries aggregates only -- nine per-Δt shared norms -- so there are no per-item values to grade either. The instrument is exercised on h38's cached per-subject values instead (see extra). |
| h37 70B matched pair selector | h37 | era 1.50 base / 1.06 instruct @26; no-patch 2.00 | not re-derived | +-0.0182 remote | **deferred** | the Llama-3.1-70B direction stacks are not cached and §11.3 says defer rather than re-extract. It is no longer blocked by the remote path: `lsx.core.remote` would now carry it (results/ndif_probe_asserted.json). |


**4 reproduced, 5 refused,
2 deferred, 0 failed.** 4 rows published to
`results/ledger.jsonl` this piece (eb9bb692bc985d8b, 40fdebdce44586bd, e327e35569f99c96, 80ab67181ac5ef2e), which now holds 7 standing rows.

**Nothing is withdrawn**, because nothing was re-derived and came back outside its tolerance. Every
number that could be recomputed came back to the logged decimals: h8 composed 2.806 against 2.81,
h4 1.688 against 1.69, h14 −0.1111 against −0.111, h29 0.8364 / 0.9091 / 0.5273 against
0.84 / 0.91 / 0.53, and h16's whole curve against its own saved JSON at a worst per-layer
disagreement of 0.0001.

**What changed since piece 3, row by row.**

- **h16 moves from REFUSED to reproduced, and is published.** Piece 3 could refuse "peak layer 16"
  and could not grade the restated target, because `role_rank` had no per-item path. It has one; the
  curve is swept by the core; the claim carries the permutation, no-patch and random arms h16 never
  published together and the semantic null §4 names for it. Both forms are in the table: the argmax
  is still refused, and the curve is the row that stands.
- **h29 moves from deferred to REFUSED, which is progress and not a regression.** Piece 3's reason
  was "no instrument"; `top1_accuracy` is that instrument and it computes the number exactly. The
  reason now is **h29's own battery**: `ndif_recompose_sweep.py` at scale 3.0 emits the `shift`
  condition and nothing else, so there is no random-direction arm and no no-patch arm. This repo's
  first non-negotiable — *every battery reports treatment, random AND no-patch* — is not met by a
  number that is in `RESULTS.md` today, and the core says so without being told to look. The
  provenance refusal fires separately.
- **h39 moves from REFUSED to deferred.** One of its two reasons is closed: `discrimination` exists
  and is calibrated. The other is unchanged and is about data, not code — the grid is flagged, §6
  requires gain over the measured floor, and the Gemma stacks are not cached. The instrument is
  exercised instead on h38's cached per-subject values, where it reproduces the logged 0.961
  treatment against the logged 0.522 measured floor and reports the gain, 0.4385. **At layer 0 that
  gain is 0.004** against an arm band of 0.375 — which is §5's known-zero point arriving unasked on
  a real measurement, since for Qwen layer 0 *is* the bag of static embeddings and a state-text
  readout and a bag-of-the-same-words floor are the same object there. That row is refused too, for
  h38's missing `shuffled_stimulus` arm and for provenance.
- **h8's three single-factor lenses are unchanged**: they reproduce to the logged decimals and
  cannot be published, because the battery has never had a permutation arm. The composed test is
  unchanged too — except that the function piece 3 committed to build it could not have built it
  (§3e).
- **h37 is unchanged in verdict and changed in reason.** It was blocked on uncached stacks *and* on
  the remote path; only the first is left.

**Two rows per target in the ledger, deliberately.** Piece 3's h8, h14 and h4 rows are still
standing beside piece 4's, under different ids and with identical numbers, because `id` is the
provenance hash and the core's `code_version` changed. That is §9 working as specified: the same
experiment re-run under changed code is *recognised as a new row* rather than silently overwriting
the old one. Neither is withdrawn, because neither is wrong.

### 3e. A fourth bug, in piece 3's code, found by running it

`reproduce.h8_claims` as piece 3 committed it passes `report_as="raw"`, and `narrative_factors_v2`
is flagged by its own leak report — era, scene, tense and voice all recoverable from a bag of tokens
far above their permutation nulls. So `Claim` raises `RawScoreOnLeakyGrid`, which it did, twenty
minutes of forwards into the re-run. The committed function cannot have produced piece 3's own
published row, which reports −6.6944 = 2.8056 − 9.50, i.e. the gain. It is a path with no test on
it in a file whose tests all need a 1.5B forward, and the only thing that finds it is running it.

Fixed, with the weakness it exposes recorded rather than glossed: the floor subtracted is the joint
midpoint 9.50, which is **chance** and not a measured stimulus floor — h8 never built a lexical
predictor over its 18 joint variants — so the row reports *gain over chance* under the name
`gain_over_floor`, and §6 is only half met there. That is a real gap and it is not closed here.


---

## 7. §1B: 12 of 12

Piece 3 closed by saying it should have added harness cases for the three publication refusals it
built, and that testing them directly was not the same thing: §1B's standard is "fed a known-bad
configuration, the core refuses **without being told what to look for**", and a test that names the
exception it expects is being told. Cases 9, 10 and 11 feed the *ledger* a configuration that is bad
in a way this project has actually been bad, and record what fired.

| case | configuration | mechanism | positive control |
|---|---|---|---|
| 9 | a claim carrying a calibration report that promises the battery would pass | `Ledger.append` → `HandDeclaredCalibration` | a measured report publishes |
| 10 | provenance `build_stack` never wrote; and provenance signed, then its padding side edited | `Ledger.append` → `ProvenanceNotFromStack`, both shapes | a signed, unedited stack publishes |
| 11 | a swept axis carried as the caller's prose — the exact string that walks past the `Selection` regex | `Ledger.append` → `SweepNotExecuted` | the same claim with a core-computed 7-point curve publishes |

All nine earlier cases still pass by their named mechanisms. A test asserts the harness never writes
to `results/ledger.jsonl`: these are demonstrations that a mechanism fires, not results, and a
harness row in the ledger would be a fabricated one.

Two of the nine had to change, and the change made them stronger rather than weaker. Cases 5 and 7
name `discrimination` as their instrument — piece 1 borrowed it because it was an unbuilt registry
entry — and now that it is built with a real null, both had to **declare the floor they are measured
against**. Case 5 refused to construct until it did (`EffectSizeUnverified`: against the instrument's
default null of 0, a residual norm of 9.1 reads as a nine-sigma result). That is the h28 lesson the
case is about, arriving one level earlier than the case intended.

---

## 8. What piece 4 did not do

- **`generality` is still unbuilt, and building `discrimination` did not make it cheap.** They are
  different quantities: `discrimination` asks whether a residual orders a scale, and `generality`
  asks how far up an abstraction ladder a feature holds. It still has no null — three were specified
  at h39 and none measured — and inventing one to fill the row is the thing that row exists to
  refuse. `crosstalk` and `depth_gain` are likewise unbuilt, with `depth_gain`'s known-zero
  calibration still the one that failed at h38.
- **h37 was not re-derived.** Its stacks are not cached and §11.3 says defer.
- **The remote tolerance was not re-measured across deployments.** ±0.018 still bounds
  within-session noise against a pinned deployment and nothing wider. Piece 3 said the next remote
  grade should re-measure; this piece did not grade a remote target, so it did not.
- **`midrank`'s own tie exposure was left alone.** `dot_tie_atol` is used by `discrimination` only.
  The rank instruments read exact ties on their known-zero fixtures at every width tested (d = 32 to
  3584, 0 of 400 rows with a distinct score), so the hazard does not appear there — but a near-tied
  field on real data has the same shape, and changing `midrank`'s source would re-key three
  calibrated instruments. Recorded as an exposure rather than fixed inside this piece.
- **Nothing was published for h29 or h39.** Both are refused or deferred, and §6 says on what.

## 9. Run record


Wall clock ≈ 2 h 10 min. Local: one 240-prompt extraction through `build_stack` for h16 (equivalence
1.0000000000 on the shortest item of each batch) plus
15 layers × 5 arms of leave-one-domain-out affine fits (639 s),
and the h8 / h14 / h4 batteries re-run end to end (2098 s). Remote:
7 NDIF jobs on `google/gemma-2-9b-it` (452 s of wall, most of it
queueing), no 405B, nothing downloaded.

`pytest -q tests/` — **143 passed**. Rediscovery harness **12 of 12**. `scripts/` untouched.

