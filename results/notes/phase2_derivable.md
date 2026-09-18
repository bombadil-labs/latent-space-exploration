# Phase 2, the derivable batch: `arm_tolerance`, h41's two skip-path rows, h8's replications

Spec: `docs/specs/phase2_v1.md` §5 (the batch), §6 (what carries forward), §7 (what not to do).
Local CPU only, `.venv`, no NDIF, nothing downloaded.

**The honest summary first.** All three deliverables landed. Twenty-five replication rows build and
**seven of them do not clear their measured lexical floor** — every `voice` row on every model, the
reference grid's `tense`, and `theme` on the GPT-authored theme grid. h41's two rows reproduce the
record to the fourth decimal (+1.8367 and +3.6135) and the record's own 90 % lower bounds come back
at +1.563 and +3.251. Three things went wrong on the way and all three are below: a permutation arm
that is systematically optimistic on **25 of 25 rows**, a random arm that refused a row until it was
given more draws, and a second stimulus floor — the readout at the shallowest cached layer — that is
stricter than the lexical one and that **twelve** of the twenty-five rows do not clear. And none of
these rows is a ledger row: every one of them is provenanced to a frozen script, not to a `Stack`.

---

## 0. What was built

`src/lsx/core/phase2.py` (new), with the band fix in `types.Arm`, `registry.InstrumentSpec.
arm_tolerance`, `checks.cluster_evidence`/`ArmUnitNotInDesign`, rediscovery case 12 in
`rediscovery.py`, `PassthroughArm.from_recorded_offline` in `instruments.py`, and two additive
generalisations in `reproduce.py` (`h8_lexical_floor(gridspec=...)` and its per-item `scene`
labels). `tests/test_core_phase2.py` is 23 tests. Results:
`results/phase2_stage41_rows.json`, `results/phase2_h8_replications.json`.

`pytest -q tests/` — **175 passed** (152 before this piece, 23 added).

---

## 1. The rows, against their floors

**Read the sign.** These are RANKS, so a gain over the floor is **negative** and a row clears its
floor when it sits *below* it. That is the record's convention (h47: era −0.7500 clears its floor,
voice +0.2083 does not) and `Claim.reported_value` computes `treatment − floor`.

Two floors are reported for every row. The first is the one §5.2 asks for and the one the `Claim`
carries: `reproduce.h8_lexical_floor`, measured **on that row's own grid**. The second is the same
readout run on the shallowest cached layer — §4 explains why it is here and why it is not the
claim's floor.

### 1a. h8's battery re-fit (writeup claim 4b), 25 rows

| model | grid | row | k | layer | treatment | lexical floor | gain | clears? | shallowest-layer floor | gain | clears? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B *(anchor)* | `narrative_factors_v2` | composed | 18 | 14 | 1.2361 | 2.8472 | **−1.6111** | yes | 2.1528 | −0.9167 | yes |
| Qwen2.5-1.5B *(anchor)* | `narrative_factors_v2` | era | 3 | 14 | 1.0000 | 2.0000 | **−1.0000** | yes | 1.5000 | −0.5000 | yes |
| Qwen2.5-1.5B *(anchor)* | `narrative_factors_v2` | voice | 3 | 14 | 1.0000 | 1.0278 | **−0.0278** | **no** | 1.0000 | +0.0000 | **no** |
| Qwen2.5-1.5B *(anchor)* | `narrative_factors_v2` | tense | **2** | 14 | 1.0000 | 1.0278 | **−0.0278** | **no** | 1.0000 | +0.0000 | **no** |
| Qwen2.5-1.5B *(anchor)* | `narrative_factors_v1` | composed | 9 | 14 | 1.1944 | 2.2222 | **−1.0278** | yes | 1.9167 | −0.7222 | yes |
| Qwen2.5-1.5B *(anchor)* | `narrative_factors_v1` | era | 3 | 14 | 1.0000 | 2.0000 | **−1.0000** | yes | 1.5556 | −0.5556 | yes |
| Qwen2.5-1.5B *(anchor)* | `narrative_factors_v1` | voice | 3 | 14 | 1.0000 | 1.0278 | **−0.0278** | **no** | 1.0000 | +0.0000 | **no** |
| Qwen2.5-0.5B | `narrative_factors_v1` | composed | 9 | 12 | 1.4167 | 2.2222 | **−0.8056** | yes | 1.9444 | −0.5278 | yes |
| Qwen2.5-0.5B | `narrative_factors_v1` | era | 3 | 12 | 1.0833 | 2.0000 | **−0.9167** | yes | 1.6389 | −0.5556 | yes |
| Qwen2.5-0.5B | `narrative_factors_v1` | voice | 3 | 12 | 1.0000 | 1.0278 | **−0.0278** | **no** | 1.0000 | +0.0000 | **no** |
| Pythia-1.4B | `narrative_factors_v1` | composed | 9 | 12 | 1.2778 | 2.2222 | **−0.9444** | yes | 1.6944 | −0.4167 | yes |
| Pythia-1.4B | `narrative_factors_v1` | era | 3 | 12 | 1.0278 | 2.0000 | **−0.9722** | yes | 1.3333 | −0.3056 | yes |
| Pythia-1.4B | `narrative_factors_v1` | voice | 3 | 12 | 1.0000 | 1.0278 | **−0.0278** | **no** | 1.0000 | +0.0000 | **no** |
| Gemma-2-9B-it | `narrative_factors_v1` | composed | 9 | 20 | 1.0278 | 2.2222 | **−1.1944** | yes | 1.5833 | −0.5556 | yes |
| Gemma-2-9B-it | `narrative_factors_v1` | era | 3 | 20 | 1.0000 | 2.0000 | **−1.0000** | yes | 1.1667 | −0.1667 | yes |
| Gemma-2-9B-it | `narrative_factors_v1` | voice | 3 | 20 | 1.0000 | 1.0278 | **−0.0278** | **no** | 1.0000 | +0.0000 | **no** |
| GPT-J-6B | `narrative_theme_v1` | composed | 9 | 14 | 1.1111 | 2.5556 | **−1.4444** | yes | 1.7222 | −0.6111 | yes |
| GPT-J-6B | `narrative_theme_v1` | era | 3 | 14 | 1.0000 | 1.5000 | **−0.5000** | yes | 1.0000 | +0.0000 | **no** |
| GPT-J-6B | `narrative_theme_v1` | theme | 3 | 14 | 1.0000 | 1.3333 | **−0.3333** | yes | 1.0556 | −0.0556 | **no** |
| Qwen2.5-1.5B | `narrative_factors_gpt_v1` | composed | 9 | 14 | 1.4167 | 3.0833 | **−1.6667** | yes | 2.2500 | −0.8333 | yes |
| Qwen2.5-1.5B | `narrative_factors_gpt_v1` | era | 3 | 14 | 1.0556 | 1.6389 | **−0.5833** | yes | 1.3889 | −0.3333 | yes |
| Qwen2.5-1.5B | `narrative_factors_gpt_v1` | voice | 3 | 14 | 1.0000 | 1.3056 | **−0.3056** | yes | 1.1389 | −0.1389 | **no** |
| Qwen2.5-1.5B | `narrative_theme_gpt_v1` | composed | 9 | 14 | 1.0556 | 1.6667 | **−0.6111** | yes | 1.2222 | −0.1667 | **no** |
| Qwen2.5-1.5B | `narrative_theme_gpt_v1` | era | 3 | 14 | 1.0000 | 1.2222 | **−0.2222** | yes | 1.0278 | −0.0278 | **no** |
| Qwen2.5-1.5B | `narrative_theme_gpt_v1` | theme | 3 | 14 | 1.0278 | 1.0833 | **−0.0556** | **no** | 1.0833 | −0.0556 | **no** |

"clears?" is a paired test, not a difference of two means: the floor's per-item ranks and the
treatment's are the same items in the same order (checked, not assumed — `h8_lexical_floor` now
records the held-out scene per item and `h8_replication_rows` refuses if the two disagree), and the
row clears when the 95th percentile of a bootstrap over the **four held-out scenes** is below zero.

### 1b. h41's two skip-path rows (writeup claims 2b and 4c)

| row | treatment (m_A) | pass-through (F_par) | **gain** | 90 % LB, cases | 90 % LB, units | sign | random | no-patch | arm band |
|---|---|---|---|---|---|---|---|---|---|
| role lens @L20, 48 cases in 8 domains | +1.8566 | +0.0199 | **+1.8367** | **+1.5630** | +1.5650 | 0.92 | +0.2670 | 0.0000 | ±0.3526 |
| three-factor composition @L14, 72 cases in 4 scenes | +3.8279 | +0.2143 | **+3.6135** | **+3.2513** | +3.0096 | 1.00 | +0.0448 | 0.0000 | ±0.3723 |

The record has +1.837 / +3.614 and lower bounds +1.563 / +3.251. The gains match to the fourth
decimal and the "90 % LB" column matches once you notice what it is: the **5th percentile of a
bootstrap over cases**, which is the lower end of a 90 % two-sided interval, and not a one-sided 90 %
bound. Resampling the design's independent units instead (8 domains, 4 scenes) leaves the role row
where it was and costs the composed row 0.24 nats.

Both rows are `readout_shift` claims with a `PassthroughArm`, required automatically because the
readout (the unembedding) is at or after the patch layer. Every margin was **recomputed here from
the run's per-candidate log-probability gains** rather than read off the run's own `m`, and agrees
with it to the last bit for all four arms — that checks the candidate ordering and the target index
on this side, and nothing else.

---

## 2. Which rows do NOT clear their floors

**Against the measured lexical floor — 7 of 25:**

* `voice` on **every** model and both Claude-authored factor grids: Qwen2.5-1.5B (v2 and v1),
  Qwen2.5-0.5B, Pythia-1.4B, Gemma-2-9B-it. Gain −0.0278 in every case, and the floor is 1.0278 of
  3 — a bag of tokens already ranks voice essentially perfectly, so there is nothing left to win.
* `tense` on the reference grid: same shape, gain −0.0278 of 2.
* `theme` on the GPT-authored theme grid: −0.0556, interval crosses zero.

This is h47's finding replicating across four model families rather than being a property of h8's
particular battery: **voice and tense are what the words are**, and the models' residuals do not
beat the words on them. Era and the composed rows clear the lexical floor everywhere.

**Against the shallowest-layer floor — 12 of 25:** the seven above, plus GPT-J's `era` and `theme`,
the GPT-authored grid's `voice`, and all three rows of `narrative_theme_gpt_v1`. On GPT-J the era
and theme lenses read exactly 1.0000 at both the shallowest cached layer and the reported layer:
whatever the readout is measuring there, the stack contributes none of it.

---

## 3. Every arm that sat off its null, and what was done about it

Final state: **no arm is off its null in any of the 27 rows.** Two were, and one systematic effect
survives inside the bands.

**3a. Gemma-2-9B-it, voice lens: the random arm read 1.583 against a declared 2.000** (band ±0.408
at n=36) and the row was refused with `ArmOffNull`. The arm was one pass of random directions, one
per item. The fix was **more draws, not a wider band** (spec §7, h47's precedent): the random arm is
now eight independent passes, pooled, banded on the between-pass spread. It reads 1.986 and the row
builds. Applied to every row, not only the one that failed — choosing the number of draws per row
after seeing which rows fail is selection.

**3b. The GPT-authored factors grid, voice lens: the permutation arm read 1.823 against 2.000** at
eight draws (band ±0.113). Under-power again, and diagnosed rather than argued: at **32** draws it
reads 1.930 with a band of ±0.124 and sits inside it. Thirty-two draws is now the setting for every
row. The band shrinks as 1/√draws; see 3c for why that is not a comfortable place to stand.

**3c. The effect that survives, and it is the piece's real methodological finding: the permutation
arm sits BELOW its declared null on 25 of 25 rows.** Mean deviation −0.117 of a rank, range −0.256
to −0.021, every one inside its own band and every one on the same side (a sign test on 25 of 25 is
p = 3 × 10⁻⁸). The random arm shows nothing of the kind — 7 of 25 below, mean +0.035 — and the
no-patch arm is exactly at its null on all 25 (an exactly tied field, mid-ranked; this arm is h34's
tie rule under test, not just plumbing).

So the declared null of this readout's permutation arm is slightly wrong, in the direction that
flatters the treatment. The mechanism is visible in the arithmetic: the candidates are scored by
cosine against the direction **without being mean-centred** — because that is exactly what
`h8_lexical_floor` does, and the whole point of the comparison is that the two differ in the vectors
and in nothing else — and a cosine against a small direction is partly a ranking by the candidate's
own norm, which a shared permuted direction applies coherently to every item fitted from it. **I did
not change the arithmetic to remove it.** Centring the candidates would make the arm look right and
would silently make the floor and the treatment two different procedures, which is the substitution
§7 forbids. What it costs: the band shrinks with draws and the bias does not, so this arm would be
refused at roughly 60–100 draws. That is a defect in the design of this readout's null, it is
recorded here rather than tuned away, and it does not touch any verdict in §1a — the treatments sit
0.9 to 1.6 ranks away from anything, not 0.12.

**3d. The h41 arms all sat on their nulls, and the design-unit declaration was refused on every one
of them.** The pass-through and random arms of both rows were offered the design's unit (the
held-out domain, the held-out scene) and `Arm` refused all four: the clustering is not visible in
the control arms' own scores (role: between-unit share 0.105–0.119 against 0.147–0.151 under
shuffled labels, p = 0.62–0.69). They are banded on the tighter i.i.d. band at the case count, and
pass there. The *treatment* of the composed row does cluster by scene (p = 0.020, design effect
3.0), which is why the unit bootstrap moves that row's lower bound and not the role row's.

---

## 4. Three things I got wrong, and one the data got wrong

**4a. I had the sign of "gain" backwards in the first draft of the row table** — and it printed a
clean, plausible table in which era "failed" its floor by a full rank and voice nearly passed. These
are ranks; gain is `treatment − floor` and **negative is better**. Caught by noticing that era's
floor of 2.0000 is exactly chance while its treatment is 1.0000, which cannot be a row that fails.
Fixed, and the convention is now a comment in the code beside the field and a sentence above the
table, because it reads as its own opposite.

**4b. The first version of the replication battery used a single random draw per item and eight
permutation draws, and I only learned that both were under-powered because two rows were refused.**
The refusals came first and the measurement after, exactly as in piece 5 §4. If those two rows had
happened to land inside their bands I would have shipped an under-powered battery and called it
clean — the other 23 rows were not evidence that the arms had power, only that they had not failed
yet.

**4c. The layer-0 floor is stricter than the lexical floor and I did not plan for it.** §5.2 says to
report gain over the measured lexical floor, so that is what the claims carry — but spec §6's other
stimulus check, layer-0 recoverability, is a *different and harder* number here, and the spec's
remark that "for RoPE models layer 0 IS the static embedding bag, so the two are the same check" is
**false for this readout**. The bag-of-tokens predictor and the embedding-layer readout disagree
sharply: on the reference grid the bag cannot tell three era variants of one scene apart at all
(floor 2.0000 = chance) while the embedding readout ranks era at 1.5000. Era words do not repeat
across scenes, so a bag generalises across the leave-one-scene-out split badly and an embedding mean
generalises well. Twelve rows clear the lexical floor and not this one. I report both and let the
claims carry the floor §5.2 named.

**4d. And one the cached data got wrong, or at least never said.** Two stacks are one row short of
`[L+1, d]`: GPT-J-6B has 28 rows for 28 blocks and Gemma-2-9B-it has 42 for 42, where
`lsx.extract` writes the embedding at index 0 and Qwen/Pythia's stacks do. So for those two the
shallowest row is block 0's output, not the embedding, and "layer 20" on Gemma is not the same index
as "layer 20" would be through `build_stack`. The stacks are gitignored and were written by a frozen
script, so this cannot be asked, only inferred; every row records
`shallowest_layer_is_embedding` and the block counts were read from the local HF cache's
`config.json` for four of the five models (Pythia-1.4B's weights are not cached and its 24 blocks
are the published architecture, unverified here).

---

## 5. The `arm_tolerance` fix, and the calibration question it had to answer

`Arm` now declares `n_independent` (defaulting to `n`) and `registry.arm_tolerance` bands on it.
Piece 5 §7 asked for exactly this; two targets had been bitten and both were caught by hand.

**The guard, because this is a dial that widens bands.** A reduction must be read off the design and
shown: the arm names its unit *and* hands in one cluster label per item, and `Arm` refuses when the
declared clustering is not visible in the arm's own scores (`checks.cluster_evidence`: the
between-unit share of variance against its own permutation null over 999 shuffles, refused at
p > 0.05). A constant arm returns p = 1 and can never buy a wider band — a constant arm sits exactly
where it sits, so a wider band around it can only hide an arm that is off its null. Rediscovery
**case 12** plants the bug: an arm 0.45 off its null, refused by `ArmOffNull` when banded on its
items, relabelled into six "draws" that are not in the data, and refused again — by
`ArmUnitNotInDesign` this time — with the positive control (a real between-unit offset) publishing.
The harness is 10 pure cases and 3 model cases, all passing.

**Does the fix invalidate cached calibration?** Checked explicitly, because piece 4's lesson is that
this is precisely where silence lives. **It does not, and it very nearly could have.** The
calibration key hashes the statistic's source closure, its declared null and its declared
invariances — it does *not* cover `arm_tolerance`. But `Instrument.resolved_null_tol`, the threshold
the calibration battery passes or fails against, **is** `arm_tolerance(n)`. So any change to the
default path would have left all nineteen cached reports on disk "valid" while having been produced
under a different threshold: piece 4's bug, one instrument along. The fix takes `n_independent=None`
as "one item is one unit", which is the old arithmetic bit for bit, and
`test_the_band_fix_did_not_invalidate_any_cached_calibration` pins both halves — the threshold to
seven decimals for each built instrument, and that every current key still names a report that is on
disk. One new report was written, `composition.9833f3b0b8de331d.json`, for the 9-variant
configuration the replication grids need; that is a new configuration calibrating itself, not a
stale one being re-blessed.

**One thing the fix does not buy, stated because it is tempting to assume it does.** The registry's
band at a declared unit is `3·sd_item_null/√units`, and `sd_item_null` is a *declaration* (the
uniform-rank closed form). Where the between-unit spread can be **measured** — a pooled permutation
or random arm, several draws — the measured band `3·sd(draw means)/√draws` is the better number and
on these rows it is the tighter of the two. Every pooled arm here uses the measured band and records
both, so a reader can see which was taken. Deliberately: the unit declaration must not become a way
to buy a wider band, and the one place it could have been used to buy one is the place it is not
used.

---

## 6. What I did NOT do

* **Nothing reached `results/ledger.jsonl`.** All 27 claims are provenanced to a frozen script — a
  cached `.npz` written by `scripts/extract_factors.py`, or `results/selector_direct_path_*.json` —
  and not to `extract.build_stack`, so the ledger refuses them with `ProvenanceNotFromStack`, which
  is the refusal h29 hit in piece 4 and that piece 5 cleared by re-extracting. Re-extraction is not
  available here: Qwen2.5-0.5B, Pythia-1.4B and GPT-J's factor grid are not in the local HF cache,
  and h41's residuals were never cached. The claims are **built** — the whole §4 contract runs, every
  arm against its declared null, both floors, the calibration gate, the pass-through requirement —
  and written to `results/`. They are candidates for the ledger, and closing that gap costs one
  local re-extraction per model plus, for h41, roughly 2.7 h of CPU to regenerate the residuals.
* **I did not recompute h41's pass-through arm.** It is §2a's arithmetic and it ran inside
  `scripts/selector_direct_path.py`; the pre-norm residuals it needs were never cached.
  `PassthroughArm.from_recorded_offline` admits it only against that run's own offline identity gate
  — the same offline path fed the *full* observed displacement reproduces the treatment's real
  forward to 3.4 × 10⁻⁵ nats (role) and 3.1 × 10⁻⁵ (composed), against the 10⁻⁴ the run registered —
  and marks the arm `recomputed=False` everywhere it is printed or serialised. That is the closest
  available analogue of `compute`'s bitwise zero-shift assertion and it is **weaker than it**: it
  says the offline readout path agrees with the forward, not that this process recomputed anything.
  This is the weakest link in the two h41 rows and it is not hidden.
* **GPT-J-6B is not replicated on a factor grid.** `stacks_gpt_j_6b_narrative_factors_*.npz` does
  not exist; only the theme grid is cached. GPT-J's three rows are era × theme, which is a different
  factor set from era × voice, and the table says so.
* **The h8 replications are not h8 re-run.** h8's battery is a patched forward; these are the
  **readout** analogue — the same leave-one-scene-out level directions, the same candidate sets, the
  same `midrank`, with each candidate's own residual scored by cosine instead of its continuation
  being re-scored by a patched model. That is a different instrument on the same design and it reads
  systematically sharper (Pythia era 1.028 here against the logged patched 1.11; GPT-J theme 1.000
  against 1.11; the composed rows 1.03–1.42 of 9 against the logged 1.44–2.22). Nothing here
  reproduces those logged numbers and nothing here claims to. What does line up, and was not tuned
  to: the pre-registered mid-depth rule picks L14 / L12 / L12 / L14 / L20, which are exactly the
  layers hours 7 and 13 used.
* **No layer was chosen on scoring data.** The reported layer is `round(0.5·(L−1))`, a rule fixed
  before any score and identical to h8's own layer 14 of 29. The full depth curve is computed and
  written into every row so a reader can see what the rule cost; no value from it entered a reported
  number. (It would have paid to choose: on the reference grid the composed curve runs 1.10 at L24
  against 1.24 at the reported L14.)
* **I did not re-run a battery because a number disappointed me, and I did not widen a tolerance so
  a row would publish.** The two refusals in §3 were fixed with draws. The systematic permutation
  bias in §3c was left in place and reported.
* **I did not touch `RESULTS.md`, `WRITEUP.md`, `VISION.md`, `README.md`, `docs/ALGEBRA.md` or
  `docs/specs/*`**, and did not delete any `.npz`. `scripts/` is untouched.
* **`crosstalk`, `depth_gain` and `generality` are still unbuilt**, and writeup claims 1, 4d, 5, 6,
  7, 9, 10 and 11 are untouched by this piece.

## 7. What the writeup would have to say, if these rows are accepted

1. "Replicated on four model families and on grids written by a second model author" is **supported
   for era and for the composed test**, through a readout rather than a patch, on all four families.
2. It is **not supported for voice on any model**: voice sits on its lexical floor everywhere
   (−0.0278 of 3, five rows), which is h47's reference-grid result replicating rather than an
   accident of h8's grid. Same for `tense` on the reference grid.
3. **Twelve of twenty-five rows do not beat the shallowest cached layer**, and two of GPT-J's three
   read exactly 1.0000 there — on that grid and that model the lens is the embedding.
4. Claims 2b and 4c stand as the record has them: +1.837 and +3.614 nats over the skip path, lower
   bounds +1.563 and +3.251, sign fractions 0.92 and 1.00, with the composed row's bound falling to
   +3.010 when the four scenes are resampled instead of the seventy-two cases.

## 8. Run record

Wall clock ≈ 55 min, all local CPU, numpy only — no model was loaded and no forward pass was run in
this piece. The two h41 rows are arithmetic over cached per-candidate readouts (≈ 6 s); the 25
replication rows, their two floors, 32 permutation draws and 8 random passes each, and the full
depth curves are ≈ 51 s. Nothing downloaded, no NDIF, no credentials printed, no `.npz` deleted.
