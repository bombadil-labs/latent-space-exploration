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

## What would falsify the whole line

`enact ≈ report` in both grids. That is the outcome where the conscription account is wrong and the
pain axis is about content after all.
