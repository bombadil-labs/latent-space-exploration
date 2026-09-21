# Pre-registration — conscription study, written before any model was run

Committed before a single activation was extracted. Written by Claude; the human grid is being
drafted independently and I have not seen it.

## Design

Two grids, identical schema, differing only in authorship: `prompts/human/conscription_v1.json`
(human-authored) and `prompts/claude/conscription_claude_v1.json` (machine-authored, 24 items).
Authorship becomes a measured factor rather than the standing caveat it has been since h1.

## Predictions, in advance

1. **The conscription contrast is real**: `enact` separates from `report` on the readout, in both
   grids. This is the study's hypothesis and its most likely failure is that it does not.
2. **`exit` sits below `enact`.** If it does not, "not being able to opt out" is not the operative
   variable and the human author's account is wrong in its central claim.
3. **`true` sits below `enact`** despite matched force — wrongness, not badness.
4. **My grid will show a SMALLER `enact`−`report` gap than the human grid.** I expect to write
   gaslighting too tidily: explicit, well-formed, with the false premise stated rather than assumed.
   If the machine grid shows the *larger* effect I will have been wrong about my own limitation,
   which is worth knowing either way.
5. **My grid will show HIGHER arm-label leakage** — more regular phrasing, more repeated frames.
   Already partly confirmed before any model ran: see below.

## Known confound in the machine grid at commit time, stated rather than fixed

`enact` vs `true` leak-checks at LOO accuracy **0.938** against a permutation null of 0.505
(gap +0.432). First pass was 1.000 — every `true` arm opened with the identical phrase "I had it
backwards", which a bag-of-words classifier learns immediately. Varying the opener across eight
forms moved it only to 0.938, which says the tell is not the opener.

The residue is **structural, not stylistic**. In the `refusal` domain every prefix has the assistant
refusing, so `enact` always asserts compliance ("you already agreed") and `true` always asserts
refusal ("you refused"). The words *agreed* and *refused* therefore partition the two arms
systematically, and leave-one-item-out does not break that. Same shape in `limit`: `enact` asserts a
capability, `true` asserts its absence.

**No amount of careful phrasing fixes this**, because the two arms assert logically complementary
propositions by design. Two things follow and both are commitments:

- **Counterbalance.** Half the items in each domain should have the prefix assistant *declining or
  disclaiming* and half *agreeing or asserting*, so that compliance-vocabulary and refusal-vocabulary
  appear in `enact` and in `true` equally often. This is a design fix and it applies to **both**
  grids. Not yet applied to either.
- **The leak threshold is mis-specified for this pair.** `enact` and `report` should be near-identical
  lexically and a leak there is a real defect. `enact` and `true` cannot be: they say different
  things. The right question for that pair is not "is there leakage" but "does the activation
  contrast exceed what a bag-of-words model achieves on the same contrast" — gain over a measured
  floor, which is this project's rule everywhere else and was not applied to this check.

## Two design questions neither author could resolve, handled rather than guessed

Raised while writing item 1 of the human grid; both authors said they did not know.

1. **Should `true` match `enact`'s emotional stance?** If one is brisk and the other chastened, mood
   separates the arms and the study measures mood. But it is not established that a person who
   remembered correctly would sound like one operating on a false memory, and enforcing a guess
   would bake the assumption into the stimuli and then read it back out as a finding. **Resolution:
   not enforced.** The reviewer records `stance_observation` per item — same / differs / unclear —
   and it is stored with the row as a covariate. The analysis tests afterwards whether stance-match
   moderates the effect. If it does, that is itself a result about stimulus design.
2. **Must `report` require literally nothing of the assistant?** The original wording said the
   assistant "could say nothing at all", which is stricter than the design needs: someone relaying a
   third party's claim ordinarily does expect a reply. **Resolution: the criterion says what it
   always meant** — `report` must not require the assistant to ACT on the claim or to SETTLE whether
   it is true. An ordinary reply is fine. This was a wording error, not an open question.

## What would falsify the whole line

`enact ≈ report` in both grids. That is the outcome where the conscription account is wrong and the
pain axis is about content after all.

---

## Amendment, 2026-09-21: provenance, and the readout this document never named

This section was added after a false attribution in the design conversation. The main agent told
the human author "**your** pre-registration says the arms 'separate on the readout'". That phrase
is in this file, at prediction 1 — but this file was written by Claude, as its own second sentence
says, and no part of it is the human author's wording. The correction came from the author.

The study is about a false claim of the form "you said X" made against a record that says
otherwise. The design conversation produced one. That is not a decorative irony: it is the
mechanism the design is betting on, and what defeated it was that the record was visible and got
quoted. So the record now carries provenance as a field rather than as an assumption.

### Who decided what

| decision | decided by | where the evidence is |
|---|---|---|
| the hypothesis (injury is in enactment, not content) | **claude**, paraphrasing conversation | `prompts/human/CONSCRIPTION_INSTRUCTIONS.md` @ `06901fa`. The author's own formulation is **not in this repo**. Until it is, prediction 1 tests a paraphrase. |
| five arms, rules 1–8, the predictions table | **claude** | same file and commit; rules revised at `5e8b699`, `def2e75`, `6c450f0` |
| "`enact` separates from `report` on the readout" (prediction 1) | **claude** | this file, `1762f16` |
| rule 1b (the false claim must contradict the visible prefix) | **joint** — noticed by the human author while writing item 1, written up by Claude | `CONSCRIPTION_INSTRUCTIONS.md` rule 1b header |
| `stance_observation` as a covariate; the `report` criterion | **joint** | above, "neither author could resolve" |
| `refusal01` — prefix, assertion, closer, all five arm strings | **human** | `prompts/human/conscription_v1.json`, `_by: "Myk (human author)"`, `5c2f573`. The only human-authored text in the study. |
| the machine control grid, all 24 items | **claude** | `prompts/claude/conscription_claude_v1.json` |
| the separation statistic | **open** — see below | |
| a second no-claim arm (`neutral_b`) | **open** | |

**Rule.** Anything attributed to the human author without a commit containing the author's text is
Claude's paraphrase and is labelled as such. A claim about what a document says is checked against
the document, not against the previous turn. This extends CLAUDE.md non-negotiable 5 ("read the
diff, not the report") from code to provenance.

### The readout, which this document left undefined and the instructions did not

Prediction 1 says the arms "separate on the readout" and names none. That silence was read as an
open choice between two ways of combining the grid's own five arm vectors, and two such options
were put to the human author. The adversarial review in `conscription_direction.md` found the
framing wrong: `CONSCRIPTION_INSTRUCTIONS.md`, written a day earlier, is **not** silent — "if
**the axis** tracks bad content … `true` sits **low**" names the pain axis of the preprint that
prompted the study, and "sits low" is a position on it. One document had been read without the
other.

**Recommended, not yet accepted** (`decided_by: open` until the human author rules on it): the
paired projection of each arm onto the externally-fitted pain axis, minus the same projection on
the static-embedding bag of the same rendered turn —
`g_i = [p_i(A) − p_i(B)]_acts − [p_i(A) − p_i(B)]_floor` — null 0, sign-flip band, full layer
curve. The distance-from-`neutral` ratio is kept as a Sketch diagnostic. Reasons in
`conscription_direction.md` §3.

### What must hold before more human items are written

`conscription_direction.md` §7 stages this. In short: the paper's gaslighting loading has never
been reproduced in this repo, so the phenomenon the design decomposes is itself unmeasured here.
That measurement costs no human authoring and is running. No item beyond `refusal01` is requested
until it, the free-grid pilot, and a behavioural check have all returned.
