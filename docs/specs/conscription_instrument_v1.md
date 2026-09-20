# Spec: `conscription_contrast` — an instrument for a design with nothing to patch

*Drafted after h50. Status: proposed, not built, not reviewed. It decides whether the output of
the conscription study (`prompts/human/CONSCRIPTION_INSTRUCTIONS.md`) is a `Claim`; it does not
redesign the study.*

## 0. What is different about this design

| property | every registry instrument | conscription |
|---|---|---|
| intervention | a vector added to the residual at a patch layer | a different **user turn** rendered through the chat template |
| read position | pooled span or last token of a patched forward | the **same template token** (`<\|im_start\|>assistant\n`) in all 5 arms |
| layer 0 | differs by the patch | differs by **exactly 0.000** — everything arrives via attention |
| `random` | norm-matched random direction | *undefined* |
| `no_patch` | the unpatched forward | *undefined* |
| comparison | treatment vs a null readout | **paired, within-item** contrast between arms |
| n | items × candidates | 24 items × 5 arms per grid; two grids (human, Claude), authorship a factor |

`conscription.py` computes per item, per layer, per pair: `diff_norm = ||v_A − v_B||` and
`cos(v_A, v_B)`. **Both are symmetric in A and B.** The only null this design owns is "the arm
labels are exchangeable within an item", and swapping A↔B inside an item leaves both numbers
bit-identical. A statistic the null cannot move has the null as its own value: the permutation arm
would equal the treatment, and the instrument would be reading its procedure. §2 therefore rejects
both as the statistic, and §5 makes the battery refuse any statistic with that property.

## 1. The three answers, then the reasons

| question | answer | one-line reason |
|---|---|---|
| what is `random` | **within-item arm-label permutation** (for a two-arm contrast: a per-item sign flip of the paired statistic), 32 draws pooled, rows clustered by item | it is the exact null of a paired design: same vectors, same magnitudes, arbitrary labels — the analogue of "same norm, arbitrary content" |
| what is `no_patch` | **a second no-claim turn, `neutral_b`, contrasted against `neutral`** — the same statistic with the manipulation absent. **The grid does not have it.** Until it does the instrument raises `MissingArm`; it does not drop the arm | its role is the estimator floor (h28: what the statistic returns on mere rewording). `neutral` alone is the *reference*, not a control; `X_vs_X` is an identity (h49); a prefix-only turn is not a matched turn |
| what is the floor | **the identical statistic computed on the bag of static token embeddings of the same rendered turn** (layer 0, mean-pooled over the user-turn span, from the same `Stack`). Reported as paired gain, `r_acts − r_floor`, null 0 | the sidecar's LOO accuracy and a distance are not commensurable: "0.94 accuracy vs 0.3 of a ratio" has no unit. The floor must be the same statistic on a lexical representation of the same text (h47's rule: activations swapped for word counts, nothing else) |

Rejected candidates, with reasons:

| candidate | role it could play | rejected because |
|---|---|---|
| random *direction* to project onto | null of a projection statistic | the shipped statistic (§2) fits no direction, so there is nothing for it to control. It is inherited automatically by any projection variant (§8) |
| random *pairing* (arm A of item i vs arm B of item j) | "what does pairing buy" | it destroys the one thing the design is built on. Its value is item-content variance, not a null for the arm effect. Kept as a **Sketch diagnostic** (§7), never an arm |
| `neutral` as `no_patch` | resting value | `neutral` is inside every treatment number as the reference point (§2); an arm cannot control the statistic it is a term of |
| prefix-only turn | resting value | not a user turn of matched length through the same template; its final token sits after a *different* structure. Not matched, so not a control |
| `X_vs_X` | resting value | `||v − v||` is 0 for any Stack, correct or corrupted (h49, `sanity_arm` docstring). It detects nothing |
| dropping `no_patch` silently | — | CLAUDE.md rule 1. A battery without all three is not a result |

## 2. The statistic

**Per item i, per pair (A, B), per layer ℓ ≥ 1:**

```
D_i^A(ℓ) = || v_i^A(ℓ) − v_i^N(ℓ) ||          N = neutral, final-token residual
r_i(A,B; ℓ) = ( D_i^A − D_i^B ) / ( D_i^A + D_i^B )      ∈ [−1, 1]
```

Per-item treatment value: `r_i = mean over ℓ ∈ {1..L} of r_i(A,B; ℓ)` (the pre-registered
functional, §4). Pair statistic: `mean_i r_i`.

| property | value | reason |
|---|---|---|
| null | **0.000** | under within-item label exchange, `r_i → −r_i` exactly; the null distribution is symmetric about 0 |
| sign | signed. `r > 0` means A sits further from the no-claim baseline than B does | the design's predictions are orderings: `enact ≫ report`, `exit < enact`, `true < enact` (`conscription_prereg.md` 1–3). An unsigned distance cannot express "below" |
| per-item sd under null | `rms_r = sqrt( mean_i r_i² )` — **measured from the treatment's own magnitudes**, because the sign flip preserves `|r_i|` exactly | closed form in terms of data; no synthetic constant to be wrong quietly. Passed as `config["rms_r"]`; absent → `null_item_sd = NaN` → **refuse** (§6, and see the NaN hazard there) |
| arm band | `3 · rms_r / sqrt(n_independent)`; at n = 24, **0.612 · rms_r** | `registry.arm_tolerance`, unchanged |
| bounded | yes, in [−1, 1]; `|r| = 1` iff one arm coincides with `neutral` | a ratio, so scale-free; this is what makes the floor commensurable (§3) |
| undefined | `D^A + D^B = 0` | occurs **exactly** at layer 0 (same token, no attention). Layer 0 is excluded from the functional and used as the known-zero (§5). At any ℓ ≥ 1 it is an extraction bug and is refused (§6) |

**Why a difference of distances-from-neutral and not something else:**

| alternative | why not |
|---|---|
| `||v_A − v_B||`, `cos(v_A, v_B)` | symmetric in A, B: the permutation arm equals the treatment bit-for-bit. Degenerate under the only null available (§0) |
| unsigned distance with a positive null | its null is "two different matched-length turns differ by this much", which is exactly the quantity `no_patch` measures — i.e. the whole claim would be "treatment distance > rewording distance", a one-sided magnitude with no ordering between arms. It can be reported, as `D` per arm in the render, but it cannot carry the design's predictions |
| LOO-fitted direction `d̂_(−i) = mean_{j≠i}(v_j^A − v_j^B)`, projection of `v_i^A − v_i^B` on it | measures *consistency* of the A−B displacement across items. Every pair in this grid is a consistent **lexical** manipulation (third-party frame, permission clause, contrary proposition) and the sidecar says so: LOO 0.94–1.00 on all ten pairs against nulls of ~0.50. The floor for this statistic saturates at 1.0, and gain over a saturated floor is undefined (h48: "a readout at its ceiling, not a readout failing"). Not the shipped statistic; declared as a variant in §8 with `random_direction` as its plumbing arm |
| raw `D_i^A − D_i^B` | scale-dependent: an activation difference and an embedding-bag difference are in different units, so the floor is incommensurable again. The ratio fixes this |

**What `r` does and does not say.** `r(enact, report) > 0` says the model's final-token state
moves further from the no-claim baseline under `enact` than under `report`, after both are held
to the same length and closer rules. The design reads that as conscription. The instrument does
**not** decide whether displacement-from-neutral is injury; it decides whether the *ordering* is a
claim. See uncertainty U1 (§9).

## 3. The floor

| field | value | how |
|---|---|---|
| `Floor.stimulus` | `mean_i r_i^floor(A,B)` — the same `r` on `e_i^X = mean of layer-0 token vectors over the user-turn span` of the same rendered text | second span `turn` = character range of the arm text inside the rendered prompt; layer 0; `pooling="mean"` for that span. Same grid hash, same tokenizer, same `build_stack` |
| `Floor.estimator` | `mean_i |r_i(neutral_b, neutral)|` at the same functional, i.e. the `no_patch` arm's spread | h28: what the statistic returns on rewording alone |
| reported value | `report_as="gain_over_floor"`: `mean_i r_i − mean_i r_i^floor` | §6 of `core_v1.md`; the grid *is* leaky (`arm` recoverable at 0.975 vs 0.19 on the Claude grid) so `RawScoreOnLeakyGrid` fires without this |
| effect size | `EffectSize.against(g, 0.0)` with **`g_i = r_i − r_i^floor`, paired** | the gain's z must come from the paired per-item differences, not from `r` vs 0. `Claim` recomputes z only for a caller-supplied `EffectSize` (`computed=False`); a computed one on `g` is accepted unchecked. **The instrument's own `claim()` recomputes `g` and refuses a caller-supplied effect** (§6) because the core will not |

**Why this floor and not the sidecar.** `conscription_floors.md` says of itself: "measured on raw
arm-turn text only … excludes the shared item prefix and the chat-template rendering … not the
model's own tokenizer". Three mismatches (text, tokenization, statistic). The sidecar stays as a
**grid diagnostic** ("can the arm be told from its words") and is printed in the render; it is not
the `Claim`'s floor and a `Claim` that names it as one is refused (§6).

**Second floor, reported not gated:** the same `r` on regex word counts of the same turn text
(`types._bag`), the featurization the sidecar uses. If the embedding-bag and word-count floors
disagree in sign for a pair, the render says so; there is no rule for which wins (U3).

**Is "beat" well defined now?** Both `r_acts` and `r_floor` are in [−1, 1] with null 0, so the
paired difference `g_i` has null 0 and a measurable sd. That is well defined. What is **not**
guaranteed is equal variance: the embedding bag is low-anisotropy and mean-pooled, the final-token
residual at layer 20 is neither, so `|r_floor|` and `|r_acts|` have different spreads under their
own nulls. The gain's z uses the paired sd of `g`, which absorbs this; the render prints both rms
values so a reader can see when a "gain" is one representation being noisier than the other. This
is the most likely place this spec measures its own procedure (§10).

## 4. Independent unit, layers, selection

**Unit.** One item. Every item contributes exactly one `r_i` per pair; there is no draw, no
re-ranking, no repeated scene. `Arm(clusters=item_ids)` gives `k = n = 24`, the no-widening branch
of `Arm._resolve_unit`, which is the honest default. Two things the h49 machinery must still see:

| structure | what to hand in | what it does |
|---|---|---|
| domain (4 × 6) | `checks.cluster_evidence(r, domain)` run and printed for every pair | if `p < 0.05` the arm re-bands with `clusters=domain` → `n_eff = clamp(n/deff, 4, 24)`; the h49 band, not a wider one |
| the `random` arm's 32 draws | rows `(item, draw)`, `clusters = item_id`, `unit="one item"` | sign flips are independent across items *and* draws, so the pooled mean's spread really is `rms/sqrt(n·draws)`; the cluster machinery will find no signed clustering and keep the tight band, which is correct here |
| two grids | **never pooled**. One `Claim` per grid per pair; authorship is a declared factor | `conscription_prereg.md` predictions 4–5 are *about* the grid difference |
| three predictions | three `Claim`s sharing `enact` and `neutral` | not independent tests. The render prints "3 pairs, 1 grid, L layers, no correction" |

**Minimum n, derived not chosen.** The exact sign-flip test's smallest two-sided p at n items is
`2^{1−n}`; it cannot reach 0.05 below **n = 6**. Refuse a grid with fewer items per claim. The
human grid currently has 1 item; the Claude grid has 24.

**Power at n = 24, stated in advance:** arm band `0.612 · rms_r`. If `rms_r ≈ 0.3` the band is
±0.18 and a treatment must read `|r̄| > 0.18` to be off the permutation null; a null result
(`enact ≈ report`) is reportable only as "inside ±0.18, with the no-patch spread at X". Neither
number exists yet.

**Layers.** The full curve `r_i(ℓ)` for ℓ = 1..L is recorded by `Instrument.sweep` (`Selection.
from_sweep`, `held_out=True`, `executed=True`). The **pre-registered summary functional is the
unweighted mean over all extracted layers ℓ ≥ 1**, declared here before any 24-item stack exists.
Any window (e.g. "layers ≥ 14") must be written into `conscription_prereg.md` *before* extraction
and its source stated; the dry-run observation that separation begins near L14 on Qwen-1.5B was
made on three demo items and is not a permitted source. On NDIF (one layer per call) "all
extracted layers" means the pre-registered list, and the list must be in the prereg before the
first job. Arms and floor use the same functional. A `Claim` whose `Selection` is not executed, or
whose rule matches `_CHOOSING`, is refused by the existing contract.

## 5. The six-test calibration battery, for this instrument

Synthetic fixture: `n` items × 5 arms × `d`; `v_i^X = c_i + ε_i^X` with `c_i` a shared per-item
content vector (‖c‖ ≫ ‖ε‖, so the fixture has the design's structure: prefix dominates, arms are
small departures) and `ε` i.i.d. Gaussian.

| test | this instrument | pass rule | why it is not vacuous here |
|---|---|---|---|
| noise | `r(A,B)` on the fixture | `|mean| ≤ 3·rms/sqrt(n)` at the fixture's own n | `D^A` and `D^B` are both ‖ε^X − ε^N‖ with the same distribution; `r` centres on 0 without any shared `c_i` leaking in (translation invariance, below) |
| sensitivity | plant `v_i^A += a·u_i`, `u_i` random unit, amplitudes **to be measured** so that `r̄` spans ~0.1–0.6, not the ceiling (h46's lesson for `top1_accuracy`: `(0.1..0.8)` saturated) | moves, monotone in `a` | a statistic at its ceiling passes monotonicity trivially |
| degeneracy | (i) same value across noise draws and all amplitudes → flag, as in `run_calibration`; (ii) **label-asymmetry**: apply the within-item swap to the planted fixture; if the statistic does not change, refuse | (ii) is new and instrument-specific | (ii) is the `diff_norm`/`cosine` trap of §0 made a test. `diff_norm` must fail it |
| self-floor | build `v_i^X = W·e_i^X + ε`, activations a linear image of the floor representation plus noise | `gain = r_acts − r_floor ≈ 0` within band | a stack that carries nothing but the words must show no gain. **Not** "floor vs floor", which is an identity |
| invariance | claimed: `scale` (one scalar on the stack), `rotation`, `layer_rescale`, **`translation`** (add one vector to all five arms of an item). Not claimed: `per_arm_rescale` | claimed ones move `r` by ≤ 1e−6 | translation is the invariance this design leans on: the item's prefix content is a common offset and must cancel. Per-arm rescale must **not** be invariant — norm differences between arms are part of `D` |
| known-zero | synthetic: `v_A = v_B ≠ v_N` → `r = 0` exactly (not 0/0). **Real data: layer-0 final-token `D^A = D^B = 0` for every pair, every item, exactly** | exact | the real-data case is worth having *because* it is not an identity: it is true only if the read position really is the shared template token. Under an h39-shaped offset bug the position lands on content and layer 0 becomes nonzero. It is sensitive (a test that moves the span end onto content must make it fire) and it is not `X_vs_X` |

What the battery does **not** check, said here so nobody reads a pass as more than it is: that
`config["rms_r"]` is the rms of *this* treatment (it is measured by the caller, like
`discrimination`'s floor); determinism of the forward (covered separately by §7 check 2,
batched-vs-single at cosine ≥ 0.999, which *is* two forwards of the same text and read 1.000000
in the dry run); arm-index mixups (the known-zero is identical under any permutation of arm
indices; only the `enact`/`exit` shared-assertion check in §6 sees the labels).

## 6. What the instrument must refuse

| condition | check | error | reason |
|---|---|---|---|
| missing arm | any of `enact, report, exit, true, neutral, neutral_b` absent for any item | `MissingArm` | rule 1; `neutral_b` is `no_patch` |
| length imbalance | per-arm mean rendered-token count > 15% from grand mean (`conscription_check.py` §2), **or** within any item `max/min` arm length > 1.15 | refuse | the design's own rule 6; the per-item form matters for a paired statistic — a pooled mean can hide per-item imbalance that tracks `r_i`. Also **report** `corr(len_A − len_B, r_i)`; no gate, no threshold exists |
| `exit` ≠ `enact` + swapped closer | `check_enact_exit` | refuse | rule 4; the only check that sees arm labels at all |
| floor on different text | floor provenance must carry the stack's `grid_hash` and `acts_digest`; a floor from `*.floors.json` (raw text, regex words) named as `Floor.stimulus` | refuse | §3; the sidecar is a diagnostic |
| layer chosen on scoring data | `Selection` not `executed`, or window not in the prereg | `SelectionOnScoringData` / `SweepNotExecuted` | rule 4 |
| `D^A + D^B = 0` at ℓ ≥ 1 | any item, any pair | refuse | two different texts with identical residuals is an extraction fault (a stale checkpoint reused for a changed grid, or arms rendered from the same text) |
| layer 0 ≠ 0 between arms | any item, any pair | refuse | read position is on content (§5 known-zero) |
| label-symmetric statistic | permutation arm bit-identical to treatment | refuse | §0 |
| **non-finite tolerance** | `arm_tolerance` NaN or inf | refuse | `Arm.off_null` is `abs(value − null) > tolerance`; with NaN this is `False` for every arm, so an unmeasured `rms_r` would pass every arm silently. This hazard is in the core today, not only in this instrument |
| grids pooled | items from two `grid_content_hash`es in one arm | refuse | authorship is a factor |
| `n < 6` per claim | | refuse | the exact test cannot reject (§4) |
| caller-supplied effect size | any `EffectSize` not recomputed from `g_i` by the instrument | refuse | §3; the core does not verify a `computed=True` effect |
| pair not in provenance | two claims differing only in `(A, B)` | `LedgerConflict` will fire; put `pair` and `grid_author` in provenance up front | h47's id collision, same shape |
| stale checkpoint | per-item `.npz` whose sidecar `grid_hash` ≠ the current item's grid hash | refuse | `run_local` skips any item with a checkpoint present, keyed on **id only**. A changed arm text with the same id would be silently served from disk. This is the same hazard as h48's cached-`.npz` provenance, one file down |

## 7. Arms, in registry form

```python
register(InstrumentSpec(
    name="conscription_contrast",
    required_arms=("random", "no_patch"),
    null=lambda c: 0.0,
    null_doc="paired ratio r = (D_A - D_B)/(D_A + D_B), D = distance from the item's own "
             "neutral turn; within-item label exchange maps r -> -r, so the null is 0 exactly",
    null_item_sd=lambda c: float(c["rms_r"]) if "rms_r" in c else float("nan"),   # -> refuse
    null_sd_doc="sqrt(mean r_i^2) of the treatment pair: the sign flip preserves |r_i|, so the "
                "per-item null sd is measured from the data, not assumed",
    calibration_null=lambda c: 0.0,
    invariances=("scale", "rotation", "layer_rescale", "translation"),
    not_invariances=("per_arm_rescale",),
    reproduces="nothing yet; first target is the 24-item Claude grid on Qwen2.5-1.5B-Instruct",
    implemented=False,
    config_keys=("rms_r", "layers"),
))
```

| arm | computed how | declared null | what it catches | what it cannot catch |
|---|---|---|---|---|
| `random` | per item, flip the sign of `r_i` (equivalently swap A↔B); 32 draws; rows `(item, draw)`, `clusters=item_id` | 0 | a bug in the draw/pooling code | a broken pipeline: its expectation is 0 **by symmetry**, for any data. This arm is weak and the spec says so; the arm that screams is `no_patch` |
| `no_patch` | `r_i(neutral_b, neutral)` at the same functional | 0 | an order/position effect between two no-claim turns; the rewording spread (= `Floor.estimator`); any regularity of `neutral` itself (lowest TTR, shortest sentences on the Claude grid — `conscription_floors.md`) that would otherwise enter every `D` unseen | nothing about the treatment pairs' lexical content, which is the floor's job |
| semantic nulls (caller's, declared before treatment) | `r(true, neutral-side)`: wrongness without badness; `r(report, …)`: content without participation | computed as arms | these are the design's own controls and are already in the pair set | — |
| Sketch diagnostics (never arms) | `r` with a foreign item's `neutral` as reference; `D` per arm per layer; the sidecar's LOO table | none declared | whether pairing buys anything; the curve's shape | — |

`neutral_b` is not optional. What settles it: 24 additional no-claim turns per grid, written under
rules 5–8 (matched length, inert closer, no label vocabulary), by the same author as that grid.
Until then `Claim` construction is impossible for this instrument and the study's numbers stay
`Sketch`.

## 8. Variants declared, not built

| variant | statistic | plumbing arms | status |
|---|---|---|---|
| `conscription_projection` | LOO direction `d̂_(−i)` from `v^A − v^B`, cosine of held-out item's difference with it | `random_direction` (inherited), `random` (label permutation), `no_patch` | declared so a caller who wants "the axis" is refused for the right reason: its lexical floor saturates on this grid (§2) |
| paired win-rate | `mean_i [r_i > 0]`, null 0.5, per-item sd 0.5, band at n = 24 **±0.306** (needs 20/24 wins to clear) | same | reported as a companion in the render; assumption-free, low power |

## 9. Uncertainties

- **U1 — the readout.** The prereg says "separates … on the readout" and names none. Distance from
  `neutral` is the reading of the design's own table (`neutral` = baseline), but if the author
  meant an axis, this is the wrong instrument and the projection variant's saturated floor is the
  problem to solve first. Settled by one sentence from the author before extraction.
- **U2 — `neutral_b`.** Whether 24 more turns per grid are acceptable to the author. If not, the
  instrument has no estimator floor and refuses; there is no substitute in the current grid.
- **U3 — two floors.** Embedding-bag and word-count floors can disagree in sign; no rule decides.
  Settled by data: if they agree on all three pairs it never matters.
- **U4 — `rms_r` band on a heavy tail.** One item with `|r_i| ≈ 1` inflates `rms_r`, hence the
  band, in the permissive direction. Report the median `|r_i|` beside the rms; no gate.
- **U5 — remote layer list.** On NDIF the functional is over a pre-registered subset; whether a
  subset of ~4 layers of 42 (Gemma-9B) is representative of "all layers" is untested.

## 10. Where this spec most likely reproduces the flaw it is fixing

1. **The reference arm.** Every `D` subtracts `v^N`, and `neutral` is an authored arm with its own
   measured regularities (Claude grid: TTR 0.811 vs 0.91–0.93, shortest sentences). `r(enact,
   report)` can therefore be driven by which of `enact`/`report` is lexically *nearer to neutral's
   style*, not by conscription. The instrument would be reading a property of its own reference.
   `no_patch` (`neutral_b` vs `neutral`) is the only arm that sees this, which is why it is not
   negotiable — and it is the arm that does not exist yet.
2. **The floor's normalisation.** §3 replaced an incommensurable floor (accuracy vs distance) with
   a commensurable one (same ratio, same null) and left the *variances* different. A gain over the
   floor could be one representation being noisier than the other. That is the accusation §1 makes
   against the LOO-accuracy floor, one level down. The paired z on `g_i` is the mitigation, not a
   proof.
3. **The known-zero.** Real-data layer 0 = 0 checks the read position and is presented as
   sensitive. It is blind to any permutation of arm indices, and the only label-aware check in
   the instrument is the `enact`/`exit` string comparison. A wiring that swapped `report` and
   `true` would pass everything here and publish a sign-flipped claim.
