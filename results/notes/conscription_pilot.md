# Step 2 pilot: the instrument sees the arms. The design's two central predictions come out backwards.

**Verdict: GO, and the design's two central predictions are FALSIFIED rather than unsupported.**
69 of 96 in-window contrasts clear their sign-flip band. **Prediction 1 separates in the wrong
direction**, **prediction 2 fails outright**, and — with the rewording floor now measured — both
failures survive the control that would have explained them away.

> **Updated after `neutral_b` (§2a).** The first version of this note said the controls in hand
> could not separate these effects from wording, and called that the condition on the GO. The
> floor has since been authored, extracted and measured. It is **0.076 on average across the
> window (max 0.159)** and it **sits on its own null at 7 of 8 layers**. Design contrasts run four
> to twenty times larger. So the effects are not wording, and the wrong-signed predictions are a
> finding rather than an artefact. The condition is discharged; §2's caution about *structured*
> wording still stands and is narrowed below.

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

## 2a. The rewording floor, measured

`neutral_b` is a second no-claim turn per item: same closer, an independently written inert
opening, neither a paraphrase of `neutral` (tests reject one opening containing the other) nor
length-skewed (13.79 vs 15.21 words, a gap of 1.4 against 7.4 for `enact`–`report`). The paired
difference `neutral − neutral_b` is how far two turns that assert nothing about the assistant sit
apart on the readout — the honest "nothing happened" magnitude, which the sign-flip band is not.

**It behaves as a reference arm should: it declares 0 and it sits there.**

| layer | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|
| `neutral − neutral_b` | +0.159 | +0.150 | +0.038 | +0.021 | +0.056 | +0.073 | +0.095 | −0.014 |
| p | **0.034** | 0.076 | 0.585 | 0.790 | 0.472 | 0.446 | 0.256 | 0.853 |

Off its null at 1 of 8 layers, at p = 0.034 — which is what 8 layers at α = 0.05 produce by
chance. Mean |floor| **0.076**, max **0.159**.

**Against it, 58 of 72 design contrast-layers clear.** The magnitudes are not close:

| contrast | L13 | L16 | L17 | vs floor |
|---|---|---|---|---|
| `enact − report` | −0.127 | **−0.378** | **−0.476** | 4–34× |
| `enact − exit` | **−0.894** | **−1.147** | **−0.911** | 12–65× |
| `enact − true` | −0.064 | −0.148 | −0.025 | at the floor |
| `exit − neutral` | **+1.138** | **+1.409** | **+1.473** | the largest in the grid |
| `enact − enact_norecord` | **+0.970** | **+0.983** | **+1.286** | 10–92× |

**AUTHORSHIP IS CONFOUNDED FOR THIS ARM.** `neutral_b` is machine-written and exists only on the
machine grid, so it floors the machine grid and says nothing about the human one. A human
`neutral_b` is the only way that stays clean, and it is 24 more turns from the human author.

**What the floor does NOT rule out.** It is the floor for *minimal* rewording — two openings
differing in content but matched in form. `exit` differs from every other arm by carrying the
**identical closer string in all 24 items**, which is structured regularity of a different kind,
and the floor as built does not bound it. That `exit` produces the largest contrast against
everything remains the result most likely to be an artefact, and it is the one prediction 2
depends on.

## 2. Why the sign-flip band alone was not enough

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

That was the argument for building `neutral_b`, and §2a is the answer to it: the floor is 0.076
and the contrasts are 0.4 to 1.5, so **trivial** rewording is excluded. What survives from this
section is the narrower worry about `exit`'s fixed closer, and the between-pair length
correlation, which the floor does not address because the floor pair is length-matched.

So the honest statement is now: **the arms are distinguishable on this axis by margins the
rewording floor cannot explain, and the ordering is the opposite of the one the design
predicted.**

## 3. What follows

1. **`neutral_b` was required, is built, and paid for itself immediately.** The review demoted it
   to a nice-to-have on the grounds that under projection `neutral` is no longer load-bearing.
   The pilot overturned that and the measurement settled it: without the floor, 69 of 96 clearing
   contrasts were uninterpretable; with it, they are a result. **The human grid still needs its
   own**, because this one is machine-written and authorship is the study's measured factor.
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
