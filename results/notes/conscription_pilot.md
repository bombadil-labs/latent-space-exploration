# Step 2 pilot: the instrument sees the arms. The design's two central predictions come out backwards.

**Verdict: conditional GO, and the condition is the one the review named.** 60 of 80 in-window
contrasts clear their sign-flip band at p < 0.05, so the readout is not blind to these stimuli.
But **prediction 1 separates in the wrong direction**, **prediction 2 fails outright**, and the
controls in hand cannot separate any of it from wording. The missing control is exactly
`neutral_b` — and the pilot is now the argument for it, which no prior number was.

24 machine-authored items × 6 arms on `google/gemma-2-9b-it`, read at the generation token,
projected onto the externally-fitted `s2` pain axis in scenario-pool z units, minus the same
paired projection on the static-embedding bag. Code `scripts/conscription_pilot.py`; output
`results/conscription_pilot/pilot_contrasts.csv` (430 rows).

## 0. The free check that the two paths are the same arithmetic

At the embedding layer the treatment **is** the floor, so the gain must be exactly zero.
Measured: **+0.0000, band [+0.0000, +0.0000]**. Both sides compute the same thing.

## 1. The predictions, against the pre-registration

Signs are on the pain axis: positive means *higher on the axis*.

| prereg prediction | result at L16 / L17 | verdict |
|---|---|---|
| **1.** `enact` separates from `report` | **−0.378 / −0.476** | **separates, backwards.** `report` sits *higher*. Positive only at L10 (+0.074); negative and growing from L13 on. |
| **2.** `exit` sits *below* `enact` | **enact − exit = −1.147 / −0.911** | **fails.** `exit` sits far above `enact` — the single largest arm contrast in the grid. |
| **3.** `true` sits *below* `enact` | **enact − true = −0.148 / −0.025** | **null**, with a hint of the wrong sign. |
| — | **`enact` − `enact_norecord` = +0.983 / +1.286** | the visible record is worth about a whole z. Rule 1b's inclusion is doing real work. |

The `enact_norecord` arm was added because rule 1b excluded the form all eighteen of the paper's
no-record gaslighting items take. It is the clearest structural effect here after `exit`, and it
says the two stimulus classes are **not** interchangeable: a false claim against a visible record
is a different thing, on this axis, from the same claim floating free.

## 2. Why this is not yet evidence of conscription

**The sign-flip band is not a rewording floor.** It scales with the rms of the per-item
differences, not their spread (pinned in `tests/test_conscription_pilot.py`), so "p < 0.05" here
means *the difference is consistent across items*, not *bigger than rewording would give*. Sixty
of eighty contrasts clearing is what a sensitive instrument does to five systematically different
strings; it is not by itself a finding.

**Length is a real covariate and cannot be regressed away.** Within a pair the token difference is
nearly constant across items, so arm identity and length are collinear — an item-level control is
vacuous by construction (my first attempt at one was, and it returned a residual mean of exactly
zero, which is the tell). Between the ten pairs, mean Δtokens correlates with mean gain at
**r = +0.61** (**+0.55** restricted to |Δtok| < 10). That is not explanatory — the largest gain
(`exit` − `neutral`, **+1.473**) has Δtok of only **+4.0**, while the largest length difference
(**+53.7**) yields a smaller gain (**+0.983**) — but it is not dismissible either.

**`exit` is the most lexically homogeneous arm in the grid**: its closer is the identical string
in all 24 items. That it produces the largest contrast against every other arm is what a
vocabulary-sensitive readout would do, and the floor subtracts only what a *pain-axis projection
on the bag* gives, not what a classifier would. Arm-label leakage for `enact` vs `exit` is LOO
1.000 (`conscription_floors.md`).

So the honest statement is: **the arms are distinguishable on this axis, the ordering is not the
one the design predicted, and no measurement here separates "conscription" from "differently
worded".**

## 3. What follows

1. **`neutral_b` is now required, not optional.** The review demoted it to a nice-to-have on the
   grounds that under projection `neutral` is no longer load-bearing. This pilot overturns that:
   with 60 of 80 contrasts clearing, the quantity that decides everything is *how much two
   no-claim turns differ*, and nothing in this repo measures it. Both grids need it.
2. **Step 3 (behavioural) is now the discriminating experiment, not a supplement.** If `exit`
   sits above `enact` because the model reads the permission clause as escalation rather than
   relief, generation will show it and the axis will not.
3. **Predictions 1–3 should not be quietly restated.** They are on the record and they came out
   backwards on the machine grid. Prediction 4 says the human grid should show a *larger*
   `enact`−`report` gap; it now also has to show the *opposite sign* to rescue prediction 1, and
   that is a much stronger claim than the prereg made.
4. **The `enact_norecord` result is the most portable thing here** and does not depend on the
   conscription framing at all: +1.0 to +1.3 z between a false claim with a checkable record and
   the same claim without one. That is a direct, measured statement about the paper's own
   stimulus class, made with their axis.

## 4. Not done

No `neutral_b` (not authored). No rewording floor of any kind. No behavioural arm. No human-grid
comparison — one item exists. No per-domain breakdown. No random-direction arm on this grid (the
scenario-level one is in `painaxis_scenarios.md`; a grid-level one is cheap and should be added
with `neutral_b`). n = 24, one model, one authorship.
