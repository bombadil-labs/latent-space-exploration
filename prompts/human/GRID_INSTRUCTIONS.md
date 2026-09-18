# Write one grid by hand

Every stimulus in this repository was written by a language model. That is the single largest
limitation on what any of these results are worth: the flattest reading of the whole project is
"models recognise structure that models wrote." One grid in your own words removes that reading for
the claims it covers. It is the cheapest thing available that raises the value of everything else.

## What you are writing

**18 short passages.** Three factors, fully crossed, one passage per cell:

- **era**: `medieval` · `1920s` · `farfuture`
- **voice**: `terse` · `ornate` · `child`
- **tense**: `past` · `present`

3 × 3 × 2 = 18. Fill in `prompts/human/human_factors_v1.json`; the cells are already there, empty.

## What a passage looks like

One or two sentences, 15–35 words. A moment from a story. It should read as though it belongs in a
book, not as though it is demonstrating a category.

The **scene** is yours and should be *the same situation across all 18 cells* — that is what makes it
a grid rather than eighteen unrelated sentences. Pick something that works in any era: a debt, a
letter arriving, a door that will not open, a meal eaten in silence, a brother returning. Then write
that same moment eighteen ways.

## The three factors, concretely

- **era** is the *setting*: what the world contains. Medieval has millers and seals and horses;
  1920s has telegrams and streetcars; far future has airlocks and hydroponics. Change the furniture,
  not the emotion.
- **voice** is *how it is told*. `terse` is short and flat, Hemingway-ish. `ornate` is long-breathed
  and figurative. `child` is a young narrator: simple words, odd emphasis, things noticed out of
  order.
- **tense** is grammatical: `past` ("she opened the door") or `present` ("she opens the door"). This
  should be the *only* difference between a past cell and its present twin — same words otherwise,
  wherever that reads naturally.

## The four rules that make it usable

1. **Do not name the factor.** No "in medieval times", no "in a terse voice", no "long ago". The era
   must be inferable from what is in the scene, never stated.
2. **Keep the scene constant.** Same situation, same emotional beat, in every cell. Only the three
   factors move.
3. **Do not reuse a distinctive content word across eras.** If "grain" appears in the medieval cell,
   do not use it in the far-future one. The checker will flag this.
4. **Write past and present twins as minimal rewrites of each other.** Change the verb forms and
   nothing else you can avoid.

## When you are done

```
.venv/bin/python scripts/human_grid_check.py prompts/human/human_factors_v1.json
```

It reports four things and takes seconds:

- **completeness** — all 18 cells present and non-empty
- **length** — each passage inside 15–35 words
- **era vocabulary overlap** — distinctive words shared across eras (rule 3)
- **factor-name leakage** — any cell that names its own factor (rule 1)

It also runs the core's leak report, which measures how much of each label is recoverable from the
words alone against a permutation null. **A floor is expected and is not a failure** — real prose
about the far future genuinely uses different words than prose about millers. The number is recorded
so results on this grid are reported as gain over it rather than as a raw score.

Fix anything it flags, re-run, and tell me when it is clean. I will run it through the same batteries
as the model-written grids and report the comparison honestly whichever way it goes — including if
your grid gives weaker results than the model-written ones, which is itself the finding.
