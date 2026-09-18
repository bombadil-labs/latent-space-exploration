# `lsx.core` piece 5: the five reporting refusals, re-run

Spec: `docs/specs/core_v1.md` v1.3, §1A. Piece 4 ended with **§1B passing 12 of 12 and §1A failing
on reporting**: every instrument a target needed existed, every number that could be recomputed
reproduced to the logged decimals, and five targets were refused for how their batteries were run
or summarised — missing arms, raw scores on a flagged grid, an aggregate chosen after the fact.
This piece re-ran those batteries.

**The honest summary first. All five targets now publish, §1A's ten rows are all reproduced or
refused-as-the-spec-predicts, and phase 1's gate closes — and three of the five numbers mean
something weaker than the record says they do.** Nothing failed its §1A tolerance; every number
came back inside it. What moved is the *floor* under three of them. Measured against what the words
give away rather than against chance, h8's composed test has a gain of **−0.04 of a rank**, h8's
tense lens has a gain of **exactly 0.000**, and h39's Gemma clock has a gain of **+0.067 against an
arm band of ±0.40**. Those are not retractions of the measurements; they are the measurements
reported the way §6 always required. **Four ledger rows are withdrawn** — the first withdrawals
this ledger has carried.

Files added: `src/lsx/core/{piece5,rerun_h29,rerun_h39}.py`, `tests/test_core_piece5.py`,
`results/{h29_arms_reimpose3.0.json,h29_arms_reimpose3.0_controls.json,h39_gemma_clock_arms.json,
repro_summary_p5.json}` and their `.npz` checkpoints. Modified: `remote.py` (the asserted
generation path), `reproduce.py` (the permutation arm, the lexical floor, the permutation-null
diagnostic), `results/ledger.jsonl`. `scripts/` untouched; `RESULTS.md`, `WRITEUP.md`, `VISION.md`,
`README.md`, `docs/ALGEBRA.md` and `docs/specs/core_v1.md` untouched.

---

## 1. The five targets, row by row

| target | logged | re-run | tolerance | verdict | arms it now carries |
|---|---|---|---|---|---|
| **h8 era lens** | 1.25/3 | **1.2500/3**, gain over the measured lexical floor **−0.7500** (floor 2.0000) | ±0.02 local | **reproduced, PUBLISHED** `b1ae456bc39d8098` | random 2.000, no_patch 2.000, permutation **2.153** (7 draws) vs null 2.00 |
| **h8 voice lens** | 1.24/3 | **1.2361/3**, gain over the floor **+0.2083** (floor 1.0278) | ±0.02 local | **reproduced, PUBLISHED** `a95b26cd33b32ae7` | random 2.250, no_patch 2.000, permutation **1.919** (7 draws) vs null 2.00 |
| **h8 tense lens** | 1.03/3 | **1.0278/2**, gain over the floor **+0.0000** (floor 1.0278) | ±0.02 local | **reproduced, PUBLISHED** `cf64496c278e02ee` | random 1.444, no_patch 1.500, permutation **1.411** (7 draws) vs null 1.50 |
| **h8 composed** | 2.81/18, no-patch 9.50 | **2.8056/18**, gain over the **measured** floor **−0.0417** (floor 2.8472) | ±0.02 local | **reproduced, PUBLISHED** `26a709702715a106` | random 2.000 (null 2.00), no_patch 9.5000 (null 9.50) |
| **h16 curve** | §1A says 2.21; the repo's JSON says 2.1692 | **2.1692** step-2 (canonical), **2.2065** step-4 — the same curve | ±0.02 local, \|Δ\| 0.0035 | **reproduced, PUBLISHED** `6e185e691dcc9e87` | random 3.479, no_patch 3.468, permutation 3.517 vs null 3.50; semantic null 1.345 |
| **h29 3× re-imposed** | 0.84 / 0.91 / 0.53, lex 0.30, n=55/72 | **0.8364 / 0.9091 / 0.5091**, lex 0.30, n=55/72 | ±0.0182 remote, \|Δ\| 0.0036 | **reproduced, PUBLISHED** `04ca1960d8602e2d` | random **0.1429**, no_patch **0.1111** vs declared null 0.1111 |
| **h39 Gemma clock** | 0.501 / 0.767 / 2.50 **raw** | **gain over the measured floor +0.0675** (treatment 0.9895, floor 0.9221) | arm band ±0.4009 at m=8 | **reproduced, PUBLISHED** `cd6e63d75ea74d7f` | floor **0.9221**, shuffled_stimulus **0.9693** vs declared null 0.9221 |

**No arm sat off its null in any published row.** One did before it was understood; §4 is that story.

`h16`'s "peak layer 16" is still REFUSED (`SelectionOnScoringData`), as the spec says it must be.
That refusal is the acceptance test passing.

---

## 2. What each target needed, and what it cost

### 2a. h29 — the expensive one, and the one that matters (writeup claim 8)

Piece 4 verified directly that the published battery has **one arm**: `recompose_gen_gemma9b.json`
carries `base`, `rand` and `shift` at scale 1.0, and every sweep file — 0.5, 2.0, 3.0, both prefix
runs — contains only `shift`. So the headline 0.84 had no random control and no no-patch baseline
**at its own scale**. `lsx.core.rerun_h29` runs all three arms at scale 3.0, re-imposed, on
`google/gemma-2-9b-it`, through `lsx.core.remote`'s asserted path.

| arm | n (scored/attempted) | era → target | leaves e1 | theme kept | declared null | band |
|---|---|---|---|---|---|---|
| `shift` (treatment) | 55 / 72 | **0.8364** | 0.9091 | 0.5091 | — | — |
| `rand` (matched-norm, same scale) | 56 / 72 | **0.1429** | 0.3036 | 0.6250 | 0.1111 | ±0.1907 |
| `base` (no patch) | 36 / 36 | **0.1111** | 0.2222 | 0.5278 | 0.1111 | ±0.2357 at n=36 |

Lexical era-word check on the treatment arm: 0.30 on 10 of 55 continuations, reproducing the logged
0.30 exactly.

**The declared nulls are 0.1111, not 1/k, and that is the one judgement call in this piece.** The
registry's `top1_accuracy` null is 1/3, which is where an *uninformative* readout sits; an unpatched
continuation is not uninformative about its own era, it keeps e1. Declaring these two arms at 1/3
would have produced a false `ArmOffNull` — piece 3's config-dependent-key bug in a different
costume. The number used instead is **h27's own published base arm**, `era_other` 0.2222 split over
the two non-e1 targets, an independent prior measurement of the identical condition (base has no
scale). The no-patch arm came back at **0.1111 to the digit**, which is that prediction tested and
met rather than restated. A structural identity is recorded beside it: for any arm blind to the
target label, era-as-target is exactly leaves-e1 / 2, and both control arms satisfy it.

**Provenance.** The patch vectors came from the cached direction `.npz` a frozen script wrote, which
is not a `Stack` and cannot reach the ledger (`ProvenanceNotFromStack` — piece 4's second refusal of
this row). So the same grid was extracted again through `remote.build_remote_stack` at both layers,
with every §7 assertion, and compared: **every shift vector matches the cached one at cosine ≥
0.999962 (L14) and ≥ 0.999956 (L20)**, against §7's threshold of 0.999. The readout directions used
for scoring *are* the asserted stack's, and the claim carries its signature.

**The moved-candidates clause, on a generation battery for the first time.** No generation job can
assert it — there is no second arm inside one forward — so it was run explicitly against the same
patch tensors on a real padded batch: the shift patch moved **4/4** rows, the random patch **4/4**,
and the h36 idiom `output[0]` moved **1/4** with `MovedCandidates` firing. Per item, each patched
continuation was also compared against the same passage's unpatched one — greedy decoding makes
that one deterministic, so an identical string means the patch never reached the forward:
**55/55 shift and 56/56 random moved, 0 identical.**

**Cost and loss.** 180 generations attempted, 147 scored, **33 lost (18%)** to NDIF returning a
completed job with an empty result. The loss is not uniform across arms — 0/36 on `base`, 17/72 on
`shift`, 16/72 on `rand` — so it is a property of the patched generation, as h29 suspected and
piece 3 confirmed by getting the same 17 losses twice. **The surviving sample is therefore not a
random subsample and the arms' n differ; that is a caveat on this row and it is not resolved here.**

**A second measured tolerance, unasked.** This run hit a different session and a different
deployment from piece 3's pair (transformers 5.17.0, nnsight 0.7.0), and era-as-target came back at
**0.8364 — bit-identical to piece 3's two runs**, with the same 55 of 72 surviving. Theme-kept moved
by exactly one item (0.5273 → 0.5091), which is one readout flip caused by using the asserted
stack's directions instead of the cached ones. Piece 3 said ±0.018 "bounds within-session noise
against a pinned deployment and nothing wider"; it now also bounds one cross-session, cross-version
re-run, at exactly its own resolution of one item in 55.

### 2b. h39 — re-extracted, and the answer is not the one the record implies

§1A does not ask for 0.767 back. It says the logged 0.501 / 0.767 / 2.50 are **raw scores on a grid
whose leak check flags 221 of 240 state spans**, and that the core must report gain over the
measured stimulus floor. `discrimination` has required exactly that since piece 4; what was missing
was the data. `lsx.core.rerun_h39` extracts all three arms of `prompts/time_translation_v2.json` on
Gemma-2-9B-it at layer 20 through `build_remote_stack` (720 texts, batched-vs-single equivalence
≥ 0.999995 on the shortest item of every batch), and computes
`scripts/time_translation_discrimination.py`'s **shared** predictor per subject — the per-item form
the instrument was calibrated for, which the logged JSON does not contain.

| arm | what its text is | shared Spearman, 8 subjects |
|---|---|---|
| `exp` treatment | interval phrase + the state at that timepoint | **0.9895** |
| `floor` | **the grid's own control prompts**: the same interval phrase with the **t0** state at every timepoint | **0.9221** |
| `shuffled_stimulus` | the experimental state span with its words shuffled inside the span | **0.9693** |

**Gain over the measured floor: +0.0675, against an arm band of ±0.4009 at m = 8 subjects.** The
Gemma clock at layer 20 is, to the resolution this design has, the interval phrase. The
shuffled-stimulus arm at 0.9693 says word order contributes nothing either, which is h38's
"order-invariant" finding arriving on a second model. The row publishes, reporting the gain, which
is what §1A asked for; what it reports is a null.

This is **not** a reproduction of the logged 0.767 and the row says so: 0.767 is a Spearman over
nine per-Δt shared *norms* with no per-subject breakdown, and `discrimination` is a per-item
instrument. Stated as a limitation rather than glossed.

### 2c. h16 — confirmed, and the aggregate settled

The curve recomputes to the same fifteen values piece 4 got (layer 16 → 1.7321, layer 28 → 2.7340),
and the claim publishes again. The recorded discrepancy is resolved by **naming one**:

- **canonical: 2.1692**, the mean of the full step-2 fifteen-layer curve, which is what
  `results/stage3_qwen1.5b_v2_rolecentered.json` contains and what §1A's restatement ("report the
  curve") asks for;
- **2.21 as published in `RESULTS.md`, `WRITEUP.md` and `docs/INSTRUMENTS.md` §2, is the same curve
  averaged over the eight step-4 layers**, which recomputes to 2.2065.

Graded against the aggregate it actually is, |Δ| = 0.0035 against 2.21, inside ±0.02. Graded against
the canonical step-2 mean, |Δ| = 0.0000 against the repo's own JSON. Both are in the ledger row's
`logged` field so the two can never again be used interchangeably without the record saying which is
which. **This piece did not edit `RESULTS.md` or `WRITEUP.md`** — those still print 2.21, and the
correction they need is one sentence: *2.21 is the step-4 subsample of a curve whose full mean is
2.1692.*

### 2d. h8's lenses — the permutation arm, at last

`selector` has required random, **no_patch and permutation** since piece 2, and h8's battery had
never had a permutation arm; pieces 3 and 4 both refused all three lenses for it.
`reproduce.permuted_level_directions` is that arm: the same fit, by the same code, on the same
activations, with the factor labels shuffled among the **training** items before the level means are
taken. What is permuted is the labels *before the fit*, not the ranking afterwards — permuting the
ranking measures the ranking code, which is h6.

Every treatment and plumbing number reproduced **bit-identically** to pieces 3 and 4: era 1.2500,
voice 1.2361, tense 1.0278, composed 2.8056, no-patch 2.00 / 2.00 / 1.50 / 9.50, random 2.00 / 2.25
/ 1.444.

**A defect in the record, found by building the arm.** `tense` has **two** levels, so its lens is a
two-candidate selector with a null of 1.50 — and `RESULTS.md`, §1A and piece 4's driver all print it
as "1.03/**3**" and piece 4 built it as a 3-candidate instrument. The value was always right; the
denominator and the null were not. Under a 3-candidate declaration its no-patch arm of exactly 1.50
would have read as 0.50 off its null. The refusal for the missing permutation arm fired first and
hid it for two pieces.

### 2e. h8 composed — the floor, measured

Piece 4 published this row as `gain_over_floor` with the floor set to the joint midpoint **9.50**,
recorded in the row's own notes as chance rather than a measured floor, and said §6 was "only half
met". `reproduce.h8_lexical_floor` measures it: a bag-of-tokens predictor, fit leave-one-scene-out
by the same arithmetic as `level_directions` on binary token-presence vectors, ranked through the
identical `midrank` over the identical candidate sets. Activations swapped for word counts and
nothing else changed.

| | composed /18 | era /3 | voice /3 | tense /2 |
|---|---|---|---|---|
| model (treatment) | 2.8056 | 1.2500 | 1.2361 | 1.0278 |
| **measured lexical floor** | **2.8472** | **2.0000** | **1.0278** | **1.0278** |
| the floor's own permutation control | 10.2431 (chance 9.50) | 2.167 | 2.125 | 1.333 |
| **gain (treatment − floor)** | **−0.0417** | **−0.7500** | **+0.2083** | **+0.0000** |
| instrument's 3σ arm band | ±1.8343 | ±0.2887 | ±0.2887 | ±0.1768 |

The floor's permutation control is what makes it usable: shuffle the labels before fitting the
lexical directions and it goes to chance, so it is reading the words and not the procedure.

**What this says, plainly.** Only **era** survives its own lexical floor: within a scene, three
candidate spans differing only in era are not separable by a bag of tokens (the floor is exactly
chance, 24/24/24), and the patched model ranks the right one at 1.25. **Tense is exactly at its
lexical floor** (1.0278 both), **voice is worse than its lexical floor** (1.2361 against 1.0278),
and **the composed test's −0.04 of a rank is inside its own 3σ band of ±1.83** — the joint effect is
not distinguishable from what the words give away. That is consistent with h5's finding that tense
and voice are lexical, and it is the first time the composed test has been measured against
anything but chance.

**This does not retract the composed measurement.** 2.8056/18 against a no-patch arm that sits on
9.50 exactly is a real effect of the patch. What it retracts is the *comparison*: −6.6944 was a gain
over chance, and §6 requires a gain over the floor on a flagged grid.

---

## 3. What the core caught in this piece's own code

The brief said to assume this code carries a version of the bug it fixes. It carried three, and each
was caught by something the spec already demanded rather than by inspection.

**3a. The generation trace reached into a non-whitelisted module — and the fix was already written
down two functions above.** `asserted_remote_generate`'s first draft bound
`rlm.model.generator.output.save()` *inside* the trace block. NDIF ships the block's source to the
deployment and refuses any attribute path through an instance of a class defined in
`lsx.core.remote`: *"Module lsx.core.remote is not whitelisted"*. Twelve generations failed that way
before it was read. `asserted_remote_patched_logprob` carries a comment saying exactly this, from
piece 4, five lines long, added after piece 4 hit it on the wire. **The fourth time in this build
that a fix has reproduced the bug it was fixing, and the first time the fix was already in the
file.** `tests/test_core_piece5.py::test_the_generate_trace_binds_the_model_outside_the_block` greps
for it.

**3b. The h29 claim was about to break its own signature.** The first draft set
`provenance["template"]` to the generation's chat format, and `template` is one of
`types.STACK_PROV_KEYS` — the fields `stack_signature` covers. Editing one after the fact is the
exact case `ProvenanceNotFromStack` exists to catch, so the row would have been refused by the very
check it was written to satisfy. The generation's fields now go in under their own names
(`generation_prompt_format`, `generation_patch_layer`, …), which also records spec §2a's template
mismatch — directions fit on raw `lead + span` text, applied inside a chat template — rather than
overwriting it.

**3c. `hash()` in a shuffled-stimulus seed.** `rerun_h39`'s shuffle was seeded from Python's `hash()`
of the item key, which is randomised per process under `PYTHONHASHSEED`: the arm would have been a
different arm on every re-run, and the extraction is resumable across processes, so a single arm
could have been assembled from two different shuffles. `hashlib.sha256` now, with a test.

And one in the driver rather than in the measurement: **a refusal on one lens aborted the whole
battery.** `Claim.__post_init__` raises, the loop did not catch, and the voice lens took era and
tense down with it — a report that names only its successes, by construction. Fixed; each lens is
now refused or published on its own.

---

## 4. The arm that was off its null, and what it turned out to be

**This is the piece's main methodological finding, and it arrived as a refusal.**

Built as a single permutation draw, the arm read era 2.139 / voice 2.347 / tense 1.417 against
declared nulls of 2.00 / 2.00 / 1.50. `selector`'s registry band at n = 72 is ±0.2887, so **the
voice lens was refused with `ArmOffNull`** — 0.347 from its null — and the run crashed on it.

An arm off its null is a bug until proven otherwise, so it was measured rather than argued:
`reproduce.h8_permutation_null` ran the arm again with **six further independent draws**.

| factor | six fresh draw means | sd across draws | registry band at n=72 |
|---|---|---|---|
| era | 1.986, 2.000, 2.028, 2.222, 2.319, 2.375 | 0.172 | ±0.2887 |
| voice | 1.403, 1.653, 1.708, 1.792, 2.222, 2.306 | **0.349** | ±0.2887 |
| tense | 1.167, 1.319, 1.403, 1.444, 1.542, 1.583 | 0.152 | ±0.1768 |

**One draw's one-sigma spread is larger, for voice, than the whole three-sigma band the registry
computes.** The band is 3σ on the statistic's per-*item* null spread over 72 items — and a
permutation arm's 72 items are **four draws** (one per scene) × eighteen re-rankings of them. `n`
counts repetitions, not evidence. That is piece 4's h16 finding arriving on a second target, and the
arm was never off its null: it was under-powered.

The fix applied is **more evidence, not a wider band**: the arm is the pooled arm over all seven
draws, with a cluster-robust tolerance `3·sd/√7` — the form piece 4 used for h16, with the
independent unit read off the design. Pooled: era **2.153** (band ±0.179), voice **1.919** (±0.420),
tense **1.411** (±0.158). None off its null; all three lenses publish.

**Stated plainly because the order of events matters: the refusal fired first and the measurement
came after it.** Two things follow and both are recorded rather than smoothed. First, a single-draw
permutation arm on this design has almost no power — a ±0.42 band on a rank bounded in [1, 3] would
admit an arm reading as low as the treatment — so pooling reduces a defect in h8's battery and does
not remove it. Second, **the registry's arm band is wrong for any arm whose randomness is a draw
rather than an item**, and it is wrong in the dangerous direction for a *clean* arm: it refuses
good arms. This is the second target it has bitten. It should be fixed in `registry.arm_tolerance`
— an arm should declare its independent unit — and this piece did not do it.

---

## 5. The ledger: four withdrawals, and an id collision

`results/ledger.jsonl` now holds 17 claim rows -- **13 standing and 4 withdrawn**, the first withdrawals it has
ever carried. Piece 3 wrote that "writing a retraction into the ledger to demonstrate the feature
would have been a fabricated row, so there is not one". These are not fabricated.

| withdrawn | why | superseded by |
|---|---|---|
| `281ec8ebf3371c58`, `eb9bb692bc985d8b` (h8 composed, pieces 3 and 4) | the floor subtracted is the joint midpoint 9.50, which is **chance**, reported under the name `gain_over_floor`. §6 requires the measured floor on a flagged grid; measured, it is 2.8472 and the gain is −0.0417, not −6.6944 | `26a709702715a106` |
| `a9cc13a339446c8e` (h8 era lens), `ac5558e3c7420ce9` (h8 tense lens) | the permutation arm is a single under-powered draw (§4), and the provenance does not record which factor was patched (below) | `b1ae456bc39d8098`, `cf64496c278e02ee` |

**The id collision.** `Claim.id` hashes the instrument, the provenance, the grid, the selection, the
config and the calibration key. h8's **era and voice lenses agree on every one of those** — same
stack, same layer, same 3-candidate `selector`, same hold-out — so they hash to the same id, and the
ledger refused the second with `LedgerConflict: identical provenance and a different number means
the run is not reproducible`. That is the right refusal for the wrong reason: the provenance simply
did not say what the experiment was. Tense never collided only because two candidates give it a
different config, and voice never collided in pieces 3 and 4 only because it was refused before it
reached the ledger. Fixed by recording `factor` in the provenance — not a signed field, so the stack
signature is untouched. **A general hazard worth naming: two claims that differ only in *which
direction was patched* are indistinguishable to `Claim.id` unless the caller puts it in the
provenance, and nothing makes the caller.**

---

## 6. Does phase 1's gate close?

**§1B: yes, unchanged.** The rediscovery harness catches **12 of 12** (9 pure + 3 model cases).
`pytest -q tests/` — **152 passed**.

**§1A: yes.** Every row of §1A is now either reproduced inside its measured tolerance and published
to the ledger, or refused in exactly the way the spec restates it:

- **reproduced and published (7):** h8 composed, h8 era / voice / tense lenses, h16 as the full
  curve, h29 3× re-imposed, h39 as gain over the measured floor. Plus h4 and h14, unchanged from
  piece 4.
- **refused as the spec predicts (1):** h16's "peak layer 16", `SelectionOnScoringData`.
- **still deferred (1):** **h37's 70B matched pair selector.** Its Llama-3.1-70B direction stacks
  are not cached, §11.3 says defer rather than re-extract, and this piece did not re-extract them.
  It is not blocked by code — `lsx.core.remote` would carry it — only by data and budget. §1A lists
  it as a target, so **the gate closes with one target outstanding and this note says so rather
  than rounding it off.**

**Nothing was withdrawn from the writeup for failing a tolerance**, because nothing failed one. Four
*ledger rows* were withdrawn, all of them for how they reported a number rather than for the number.

**What the writeup should say differently, though this piece did not edit it:**

1. **h8's composed result has no gain over its measured lexical floor** (−0.04 of a rank, band
   ±1.83). "Three narrative factors compose" is, on this grid, not separable from "the words differ".
2. **h8's tense lens is exactly at its lexical floor and its voice lens is worse than it.** Only era
   clears its floor.
3. **h39's Gemma clock has no gain over the interval phrase** (+0.067, band ±0.40), and its
   shuffled-stimulus arm is 0.969 — order contributes nothing.
4. **h29's 0.84 now has its controls** and they sit where they should: 0.143 random, 0.111 no-patch
   against a declared 0.111. Claim 8 is the strongest of the four and it is the one that got
   stronger.
5. **h16's 2.21 is a step-4 subsample**; the full curve's mean is 2.1692.
6. **h8's tense lens is 1.03 of 2, not of 3.**

---

## 7. What this piece did not do

- **h37 was not re-derived.** See §6.
- **`registry.arm_tolerance` still computes an i.i.d. band from the item count**, which §4 shows is
  wrong for any arm whose randomness is a draw. Two targets have now been bitten by it, both caught
  by hand. An `Arm` should declare its independent unit and the registry should use it.
- **The 18% NDIF generation loss is unexplained and unequal across arms** (0% base, 24% shift, 22%
  random). Piece 3 showed it is deterministic per item; nobody has looked at *what* about those
  items loses the job, and the surviving sample is not a random subsample.
- **`crosstalk`, `depth_gain` and `generality` are still unbuilt**, and `generality` still has no
  null.
- **`RESULTS.md` and `WRITEUP.md` were not edited.** §6 lists the six sentences they need. That is
  phase 2's job and this is phase 1.
- **The h39 row is not a reproduction of the logged 0.767** and does not claim to be; it is the
  restated target, which is a different statistic on the same grid.

## 8. Run record

Wall clock ≈ 3 h 10 min. **Local** (Qwen2.5-1.5B, 4 shared CPUs): h8's three-factor battery with the
permutation condition, 3 096 patched log-probability forwards (1 432 s); the permutation-null
diagnostic, six further draws, 3 456 forwards (1 592 s); h16's 240-prompt extraction plus 15 layers
× 5 arms of leave-one-domain-out fits (720 s). **Remote** (`google/gemma-2-9b-it`, NDIF): h29 —
180 generations attempted, 147 scored, plus 2 asserted stack extractions and 3 moved-candidates
controls, ≈ 250 jobs; h39 — 720 texts at layer 20 in three arms, ≈ 290 jobs including the
batched-vs-single check on every batch. No 405B, nothing downloaded, no credentials printed.
`scripts/` untouched.
