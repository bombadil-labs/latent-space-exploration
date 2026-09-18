# Phase 2 spec v1 — re-derive the writeup through the core

**Gate (from `docs/PROGRAM.md`):** every standing claim in `WRITEUP.md` is a `Claim` row in
`results/ledger.jsonl` with provenance, or withdrawn. No new science. Budget ~150k, one agent.

**This spec's finding, up front: the gate as written cannot be met inside that budget, and one
claim cannot be met inside phase 2 *at any budget* without doing new science, which phase 2 forbids
by its own definition.** The inventory is below; §4 states the decision that needs making. Written by
Opus rather than a Fable planner because Fable's monthly budget is exhausted; phase 2 contains no new
measurement, so the planner rule does not bind.

## 0. CORRECTION (h48) — the DERIVABLE column was wrong

The batch in §5 ran and produced 27 measurements. **The ledger refused every one of them with
`ProvenanceNotFromStack`:** each is provenanced to a frozen script's cached `.npz` or
`selector_direct_path_*.json`, not to `build_stack`. h29 hit the same refusal at h47 and cleared it
only by re-extracting the whole grid, and that route is not open here — four of the five replication
models are not in the local HF cache, and h41's residuals were never cached.

**"Cached data + existing instrument" is not sufficient for a ledger row, and §2's verdict set
omitted the binding constraint: provenance.** Corrected counts: **4 DONE, 0 DERIVABLE, 4
NEEDS-INSTRUMENT, 6 NEEDS-DATA, 1 BLOCKED.** Rows 2b, 4c and 4b move from DERIVABLE to NEEDS-DATA.

This strengthens rather than weakens §4's conclusion: the gate cannot be met as written, and now
there is no batch that can be done cheaply first. §6.6's warning about the gitignored `.npz` stacks
also turns out to understate the problem — those stacks cannot reach the ledger even while they exist.

---

## 1. What exists today (verified, not recalled)

- **Ledger:** 17 claim rows — 13 standing, 4 withdrawn — covering stages **h4, h8, h14, h16, h29,
  h39 only**. Several stages appear twice (piece 3's row and piece 4's/5's row have different
  provenance and so different ids).
- **Registry:** 7 instruments declared. **Built:** `selector`, `composition`, `readout_shift`,
  `discrimination`, `top1_accuracy`. **Declared and deliberately unbuilt (spec §11.4):**
  `crosstalk`, `depth_gain`, `generality`. `depth_gain`'s known-zero calibration is the one that
  *failed* at h38 and must not gate the core; `generality` has **no null at all**.
- **Cached local stacks** (`results/*.npz`, ~547M, **gitignored — these do not survive the
  container**): Qwen2.5-0.5B/1.5B holonic (+shuffled, +rotated, +v2_rotated), narrative factors on
  Qwen 0.5B/1.5B, Pythia-1.4B, GPT-J-6B and Gemma-2-9B-it, theme and mood grids, the GPT-authored
  grids, and `tokens_gemma9b_broad_l20`.
- **Not cached:** every Llama-3.1-70B / 70B-Instruct direction stack, and the generation batteries of
  h27 and h33. These are NDIF re-extractions.

## 2. Claim-by-claim inventory

Verdicts: **DERIVABLE** (cached data + existing instrument) · **NEEDS-INSTRUMENT** ·
**NEEDS-DATA** (re-extraction; local or remote) · **MUST-WITHDRAW**.

| # | claim (headline) | stage | instrument needed | built? | data | verdict |
|---|---|---|---|---|---|---|
| 1 | discourse position dominates pooled activations | h1–h3 | RSA/CKA with a shuffled-span null — **not in the registry** | no | local, cached | **NEEDS-INSTRUMENT** |
| 2 | role direction transfers and selects (1.69/6) | h4 | `selector` | yes | cached | **DONE** — 2 ledger rows |
| 2b | …and it is computation, not the skip path (+1.84 nats) | h41 | `readout_shift` + `PassthroughArm` | yes | `selector_direct_path_role*.json`, cached | **DERIVABLE** |
| 3 | relation selector 2.1692/6 over the full curve | h16 | `selector` | yes | cached | **DONE** — 2 rows; peak-layer form correctly refused |
| 4 | each factor a lens; three compose | h8 | `selector`, `composition` | yes | cached | **DONE** — 4 rows, floors measured at h47 |
| 4b | replicates on four model families and a GPT-authored grid | h9–h13 | `selector`, `composition` | yes | **all five stacks cached** | **DERIVABLE** (×5 the fitting work) |
| 4c | the three-way composition is computation (+3.61 nats) | h41 | `readout_shift` + `PassthroughArm` | yes | cached | **DERIVABLE** |
| 4d | the cross-talk matrix is diagonal | h8, h10 | `crosstalk` | **no** | cached | **NEEDS-INSTRUMENT** |
| 5 | factors differ in kind; depth shows it | h5 | `depth_gain` | **no**, and its known-zero calibration **failed** at h38 | cached | **NEEDS-INSTRUMENT** (hard) |
| 6 | abstraction is a quotient with a measurable scale | h42 | `generality` | **no, and it has no null** | SAE data cached | **BLOCKED — see §3** |
| 7 | the shallower factor dominates under generation | h19, h22 | generation-ordering readout — not in the registry | no | `commutator_gemma9b*.json`, cached | **NEEDS-INSTRUMENT** |
| 8 | recomposition is real, and only in generation (0.84) | h29 | `top1_accuracy` | yes | re-derived at h47 with all three arms | **DONE** — 1 row |
| 9 | gauges compose, engines don't; the boundary is a magnitude | h27, h29, h33 | `top1_accuracy` | yes | h29 done; **h27 and h33 arms are not Stack-provenanced** | **NEEDS-DATA (remote)** |
| 10 | the magnitude is not a constant of the method (0.43 on 70B-Instruct) | h33 | `top1_accuracy` | yes | not cached | **NEEDS-DATA (remote)** |
| 11 | instruction tuning sharpens era, not theme | h37 | `selector` | yes | **70B stacks not cached** | **NEEDS-DATA (remote)** — already phase 1's one outstanding target |

**Counts: 4 DONE, 3 DERIVABLE, 4 NEEDS-INSTRUMENT, 3 NEEDS-DATA (all remote), 1 BLOCKED, 0
MUST-WITHDRAW.** Nothing in the writeup is unsupportable on its face; what is missing is
instrumentation and provenance, not evidence.

## 3. Why claim 6 is blocked rather than merely expensive

`generality` is refused **for having no null**, not for being unwritten. Designing that null is a new
measurement — it requires deciding what population a "general" feature is general *against*, and h42
already showed that the companion ordering claim collapses once its permutation null is drawn.
Phase 2's charter is "no new science". So claim 6 cannot become a ledger row inside phase 2 **at any
budget**, and the only honest phase-2 outcomes for it are: mark it NOT-DERIVED with this reason, or
withdraw it. It should not be quietly restated until it clears a floor it was never measured against.

## 4. The decision this spec cannot make

The gate says *ledger row or withdrawn*. Seven claims are neither, and none of the seven is wrong —
they are un-instrumented or un-provenanced. Three options, and this is the user's call:

- **(a) Raise the budget.** Building `crosstalk`, an RSA instrument, a generation-ordering readout
  and `depth_gain` (whose known-zero already failed once) is the bulk of another phase-1, not 150k.
- **(b) Amend the gate** to: *ledger row, withdrawn, or explicitly marked **NOT-DERIVED** with a
  named blocker in both `WRITEUP.md` and the ledger.* Phase 2 then completes as a 150k pass that
  derives everything derivable and labels the rest honestly, and the fifth-draft writeup asserts
  "every number **so marked** was produced by a verified instrument" rather than "every number".
- **(c) Cut the writeup to what the core can carry** — 4 claims today, 7 after the derivable batch.

**Recommendation: (b), then (a) for `crosstalk` alone.** Cross-talk is the one unbuilt instrument
whose claim (4d) is load-bearing — the diagonal is what "the factors are separate directions" means,
and h47 already showed the *null's* cross-talk is partly diagonal too, so it is also the claim most
at risk of being geometry. The other three can wait behind the human grid.

## 5. The derivable batch (do this regardless of the decision)

One agent, in order, local CPU only, no NDIF:

1. **2b and 4c** — the two stage-41 skip-path rows via `readout_shift` + `PassthroughArm`.
2. **4b** — h8's battery re-fit on the four cached replication stacks and the GPT-authored grid,
   each a `selector`/`composition` row **with its measured lexical floor**, not chance.

Expected: 3 claims move to DONE, ~7 new ledger rows.

## 6. Requirements carried forward (from what the first five pieces cost us)

1. **Gain is over a MEASURED floor**, never chance, on any grid with a measured leak. h47: of h8's
   four rows, only era clears its floor.
2. **Every battery declares treatment, random and no-patch arms**, each with its own expected null,
   and the null is a prior measurement of the identical condition where one exists (h29's 0.1111),
   not 1/k.
3. **`registry.arm_tolerance` is wrong for any arm whose randomness is a draw rather than an item.**
   It computes a 3σ i.i.d. band from the item count and therefore **refuses clean arms**; two targets
   have been bitten, both caught by hand. **Phase 2 should fix it** — an `Arm` declares its
   independent unit and the registry divides by that — because phase 2 will add ~7 rows whose
   permutation arms have exactly this structure, and the alternative is seven more hand-catches. This
   is a correctness fix to an existing instrument, not new science.
4. **Provenance records which direction was patched.** Otherwise claims differing only in that
   collide on `Claim.id` (h8's era and voice lenses did).
5. **Never select a layer on scoring data.** Report the curve. h16's peak-layer form stays refused.
6. **The cached `.npz` stacks are gitignored and will not survive this container.** Every DERIVABLE
   row above depends on them. If they are lost, the whole derivable batch becomes NEEDS-DATA (local
   re-extraction, hours of CPU). Do the derivable batch early.

## 7. What phase 2 must not do

- Re-run a battery because the first number was disappointing.
- Widen a tolerance so a row publishes. The h47 precedent is the rule: when an arm read off its null,
  the fix was **more draws**, not a wider band.
- Restate a claim so it clears its floor. If it does not clear, the writeup says it does not clear.
- Build an instrument by copying the statistic from the script that produced the original number —
  that is the instrument grading its own homework, and it is how five of the six broken instruments
  survived as long as they did.

## 8. Gate test

Phase 2 is done when, for **every** numbered standing claim in `WRITEUP.md`, one of the following is
true and checkable by a script: (i) a standing `results/ledger.jsonl` row exists whose `stage` and
`treatment` match the writeup's number, with a stack signature and all required arms; (ii) the claim
is withdrawn in both `WRITEUP.md` and Checkpoint 3; or (iii) — **only if option (b) is adopted** —
the claim is marked NOT-DERIVED in both, with the blocker named. Checkpoint 3 must state the counts
in each category and must not round option (iii) into option (i).

## 9. Uncertain, unresolved from the files

Whether h27's and h33's generation outputs on disk carry enough to rebuild a `Stack` without a
re-extraction, or whether — as with h29 — the directions came from a cached `.npz` a frozen script
wrote and therefore cannot reach the ledger at all. h29 needed a full re-extraction to clear
`ProvenanceNotFromStack`; if h27 and h33 are the same shape, claims 9 and 10 cost what h29 cost,
which was ≈250 NDIF jobs. I did not verify this and the NEEDS-DATA estimate above assumes the worse
case.
