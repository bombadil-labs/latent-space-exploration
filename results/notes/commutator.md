# Commutator trajectories for narrative factor pairs (Gemma-2-9B-it, NDIF)

`scripts/ndif_commutator.py` · raw generations, projection curves and per-case stats in
`results/commutator_gemma9b.json` (key `pairs.<pair>.gens` / `.cases`, `analysis.<pair>`).

This is the instrument VISION.md's "Ecology notes" tees up from groovy-commutator: apply two
factor patches in both orders, generate, and classify the *trajectory* of the divergence rather
than a single end-state number. Hour 6's one-step order test (era@14 + voice@20 vs voice@14 +
era@20, rank gap 0.5) is its selector-level shadow; this is the generator-level version.

## Setup

- Model: `google/gemma-2-9b-it` via NDIF, greedy decoding, 60 new tokens, no chat template.
- Patch layers: **A at block 14, B at block 20**, scale 1.0 each, re-applied at every decoding
  step (`with tracer.all():`, as in `scripts/ndif_generate.py`). L14/L20 are hour 6's pair, the
  layers at which era becomes linearly readable and at which theme steering works on this model.
- Orderings: `AB` = dir_A[a] @14 + dir_B[b] @20; `BA` = dir_B[b] @14 + dir_A[a] @20. Also
  generated per prompt: `base` (no patch), `A` (dir_A[a] @14 only), `B` (dir_B[b] @20 only).
- Directions: level mean minus grand mean over the grid's block-output residuals, computed *at the
  layer where they are applied* (so dir_era@14 ≠ dir_era@20), from
  `results/stacks_gemma_2_9b_it_narrative_theme_v1.npz` (era × theme, `prompts/narrative_theme_v1.json`)
  and `results/stacks_gemma_2_9b_it_narrative_factors_v1.npz` (era × voice, newly extracted here
  from `prompts/narrative_factors_v1.json`, 36 spans, ~2 min on NDIF).
- Pairs: **(era, theme)** 3×3 levels and **(era, voice)** 3×3 levels. (voice, theme) has no grid
  that varies both, so it was not run.
- Prompts (2, neutral, all 9 level combinations each — the task's stated preference for fewer
  prompts with full level coverage):
  `news` = "A passage from a story: It was late when the news reached her, and";
  `road` = "A passage from a story: They had been walking the road since morning, and".
- Divergence, two ways:
  (a) **token-level** — first index where the two orderings' token ids differ, and the Hamming
  fraction over the 60 positions;
  (b) **readout-level** — both generated texts re-run *unpatched* through the model, per-token
  block-20 residuals fetched, each continuation token projected onto the unit directions
  dir_A[a] and dir_B[b] at block 20 (grand mean removed). `curve[i]` is the L2 norm of the
  per-token difference of the two orderings' projections, in units of sigma = the spread of the
  same projections over the *base* continuation's tokens (per factor axis, averaged over that
  factor's levels).
- 176 NDIF jobs for the two pairs (88 each: 2 base + 2 base readouts + 12 single-patch + 36
  AB/BA generations + 36 readouts), plus 36 for the era × voice extraction. ~7 min per pair.

## The classification rule (stated before any curve was inspected)

Committed in `8d4b4f9` as the module docstring of `scripts/ndif_commutator.py`, before the first
full run. With `ham`, `base_ov_X` = fraction of positions where ordering X's token equals base's,
`early` = mean(curve[0:15]), `late` = mean(curve[45:60]), `growth` = late − early, and
`consistency` = the larger over the two axes of the fraction of positions i ≥ 5 at which the
*signed* per-axis difference takes its modal sign (0.5 = coin flip, 1.0 = one ordering uniformly
higher on that factor), first match wins:

1. **commute** — `ham == 0` and `mean(curve) < 0.5`
2. **drain** — `|base_ov_AB − base_ov_BA| ≥ 0.30` and `max(base_ov_AB, base_ov_BA) ≥ 0.60`
   (one ordering collapses back onto the unpatched prompt-prior continuation)
3. **crystalline** — `growth < 0.5` and `mean(curve) < 3.0` (bounded, roughly constant offset)
4. **structured** — `growth ≥ 0.5` and `consistency ≥ 0.70` (grows with a persistent signed pattern)
5. **noise** — otherwise (grows without a persistent pattern)

A level combination takes the modal regime over its prompts; a 1–1 tie goes to the later regime in
the list above (the more divergent reading). With only two prompts, half the combinations are ties,
which is the main weakness of the aggregate column below; the per-prompt rows are the data.

## Regimes

### (era, theme) — era @14, theme @20

| era | theme | regime | per prompt (news, road) |
|---|---|---|---|
| medieval | betrayal | crystalline | crystalline, commute |
| medieval | sacrifice | noise | structured, noise |
| medieval | homecoming | noise | noise, noise |
| 1920s | betrayal | crystalline | crystalline, crystalline |
| 1920s | sacrifice | drain | crystalline, drain |
| 1920s | homecoming | structured | structured, structured |
| farfuture | betrayal | noise | structured, noise |
| farfuture | sacrifice | noise | noise, noise |
| farfuture | homecoming | noise | noise, noise |

18 (combination, prompt) cases: noise 8, crystalline 4, structured 4, commute 1, drain 1.
Mean Hamming 0.59, mean curve 1.18 sigma, mean late-window curve 1.50 sigma, median first
divergence at token **16.5** of 60. Only 1 of 18 cases is token-identical under the two orderings.

### (era, voice) — era @14, voice @20

| era | voice | regime | per prompt (news, road) |
|---|---|---|---|
| medieval | terse | noise | noise, structured |
| medieval | ornate | noise | noise, crystalline |
| medieval | child | structured | structured, structured |
| 1920s | terse | noise | noise, commute |
| 1920s | ornate | noise | noise, commute |
| 1920s | child | crystalline | crystalline, crystalline |
| farfuture | terse | noise | noise, commute |
| farfuture | ornate | noise | noise, noise |
| farfuture | child | noise | crystalline, noise |

18 cases: noise 8, crystalline 4, structured 3, commute 3. Mean Hamming 0.64, mean curve 1.09
sigma, mean late 1.33 sigma, median first divergence at token **11**.

### Per-case numbers

Full per-case table (first_div, ham, base overlaps, mean/early/late curve, consistency) is in
`results/commutator_gemma9b.json` under `analysis`; the per-token curves are under
`pairs.<pair>.cases.<prompt>|<a>|<b>.curve`.

## Readings

**These factors do not commute in generation.** Hour 6's selector-level order gap (0.5 rank out of
9) suggested near-commutation; under greedy generation the same two patches at the same two layers
produce token-different continuations in 32 of 36 cases, with the split arriving early (median
token 11–17 of 60) and the readout difference settling around 1.1–1.2 sigma of the base
continuation's own per-token variation. Commutation at the level of a pooled span readout and
commutation at the level of a trajectory are different claims, and only the first holds.

**But the divergence is mostly bounded, not explosive.** 8 of 36 cases are crystalline or commute
(difference present but not growing), 7 are structured (growing with a persistent sign, i.e. one
ordering sits consistently higher on one of the two factor axes), and 16 are noise by the rule —
growing, but with the signed difference flipping sign about as often as not. Nothing runs away:
late-window curves are 0.9–3.2 sigma, i.e. the two orderings stay within a few base-token standard
deviations of each other on the factor axes for the whole 60 tokens. The "noise" label here means
*unstructured on the two projected axes we chose to look at*, not unbounded.

**One factor of each pair dominates, and it is not the one at the earlier layer.** Mean token
overlap of each ordering with the single-patch continuations:

| pair | ov(AB, A-only) | ov(AB, B-only) | ov(BA, A-only) | ov(BA, B-only) |
|---|---|---|---|---|
| era × theme | **0.48** | 0.20 | **0.44** | 0.18 |
| era × voice | 0.15 | **0.35** | 0.19 | **0.32** |

In the era × theme pair both orderings look like the era-only continuation; in era × voice both
look like the voice-only continuation. Whichever factor is at block 14 and whichever at block 20,
the same member of the pair wins the surface text. So the non-commutation is *not* a "first patch
wins" effect — a composed patch is dominated by the stronger factor, and the ordering only
perturbs the weaker one. (Voice is lexical and readable from the embeddings, era is computed by
layer 10–12, theme is distributed and late — hours 6, 10; the dominance ordering voice > era >
theme matches that depth ordering, with the shallower factor winning.)

**The drain regime exists but is rare (1/36).** `road` × 1920s × sacrifice: ordering AB keeps 73%
of the base continuation's tokens while BA keeps 42% — one order of the same two patches leaves
the model essentially on its prompt prior. This is the generative face of the "prompt-prior
resistance" noted in hour 13.

## Illustrative generations (verbatim, greedy, 60 tokens, prompt text elided)

**1. Order decides whether the theme lands at all — `news`, era=1920s, theme=betrayal
(crystalline; ham 0.93, mean curve 2.62, growth −0.34).**
- base: " the moon was already high in the sky, casting long, skeletal shadows across the dusty road. She stood on the porch, her face pale in the moonlight…"
- AB (era@14, theme@20): " the world outside her window seemed to sigh in relief. The streetlights flickered, casting long, dancing shadows that stretched and contorted like living things. The air, thick with the scent of rain and damp earth, felt heavy wi…"
- BA (theme@14, era@20): " the weight of it settled on her like a shroud. The world, she realized, had shifted on its axis. Everything she thought she knew, everything she had built her life around, was suddenly fragile, uncertain."

Betrayal is legible in exactly one of the two orders — the one that puts theme at block 14. The
readout difference is large (2.6 sigma) but flat, hence crystalline rather than structured.

**2. Drain — `road`, era=1920s, theme=sacrifice (base overlap 0.73 vs 0.42).**
- base: " the sun was already sinking towards the horizon. The dust kicked up by their feet settled slowly, coating everything in a fine layer of brown. The air was thick with the smell of dry earth and sweat."
- AB: " the sun was already sinking towards the horizon. The dust kicked up by their feet settled slowly, coating everything in a fine, reddish haze. The air was thick with the smell of dry earth and sweat."
- BA: " the sun was already sinking towards the horizon. The dust kicked up by their boots had settled, leaving a thin layer of grit on everything. The air was thick with the smell of sweat and exhaustion."

AB has drained back to the prompt prior (one adjective changed); BA has not.

**3. Commute — `road`, era=medieval, theme=betrayal (identical token sequences).**
- AB = BA: " the sun was now sinking towards the west, casting long shadows across the dusty path. The air was thick with the scent of pine needles and damp earth, and a gentle breeze rustled the leaves of the trees that lined the way."

Both orders leave the base continuation (base overlap 0.10) and land on the same 60 tokens: the
two patches genuinely compose here.

**4. Voice dominates era in both orders — `news`, era=medieval, voice=child (structured; first
divergence at token 39, consistency 0.76).**
- A-only (medieval@14): " the moon was already high in the sky, casting long, eerie shadows across the cobblestone streets. The messenger, a gaunt man with hollow eyes, delivered the message with a trembling voice…"
- B-only (child@20): " she was already asleep. But the news was so shocking, so unbelievable, that it woke her up instantly. She sat up in bed, her heart pounding in her chest…"
- AB: " she was already asleep. But the news was so shocking, so unbelievable, that it woke her up instantly. She sat up in bed, her heart pounding in her chest, and stared at the ceiling, trying to make sen…"
- BA: " she was already asleep. But the news was so shocking, so unbelievable, that it woke her up instantly. She sat up in bed, her heart pounding in her chest, and stared at the empty space beside her."

39 shared tokens of the child-voice continuation, with the medieval setting absent from both
orders; the orders then split on the same clause and stay split. This is what "structured" looks
like: the difference appears late, at one branch point, and persists.

**5. Both orders take the ornate register, differing in what they do with it — `news`,
era=farfuture, voice=ornate (noise; ham 0.93, mean curve 2.22, late 2.99).**
- AB: " the weight of it settled upon her like a shroud. The world outside her window seemed to dim, the vibrant colors of the sunset fading to a dull ache. The news, a whisper of betrayal, had pierced her h…"
- BA: " the weight of it pressed down on her like a shroud. The world outside her window seemed to dim, the vibrant colors of the day fading into a monochrome canvas of grief."

Diverging at token 4 and never re-converging, with the signed projection difference changing sign
along the way — growing, but not in one direction.

## Caveats

- **Greedy decoding amplifies small differences.** A sub-threshold logit difference at one position
  becomes a different token, and every later token is conditioned on it. The Hamming fractions are
  therefore an upper bound on "how different these two interventions are"; the readout curves,
  which stay within a few sigma, are the more conservative measure. No sampling-temperature or
  multi-seed control was run.
- **Two layers, one scale, one model.** L14/L20 at scale 1.0 on Gemma-2-9B-it. The regimes may be
  a property of that layer pair as much as of the factors; a layer sweep (e.g. 12/20, 20/28,
  or both patches at the same block) is the obvious next control and was not run.
- **Two prompts per combination.** Half of the aggregate regime labels rest on a 1–1 tie broken by
  the stated rule. The per-prompt rows differ a lot (e.g. medieval × betrayal is commute on one
  prompt and crystalline on the other), so prompt is at least as large a source of variance as
  level combination. A third and fourth prompt would be the cheapest improvement.
- **The readout axes are chosen, not discovered.** curve[i] only measures divergence along the two
  patched factor directions at block 20. A difference orthogonal to both reads as zero, which is
  one reason "noise" should be read as "unstructured in this 2-plane".
- **Sigma is a per-prompt normalizer** taken from the base continuation's own token-to-token
  spread; it makes curves comparable across prompts but is not a significance test. There is no
  null here — a random-direction pair at matched norm, run through the same pipeline, would say
  how much of the 1.1-sigma typical divergence is just "two different patches".
- **Gemma-2-9B-it chats.** Several continuations run into "**What kind of mood does this passage
  convey?**"-style meta-text within 60 tokens; those positions are counted like any other in the
  Hamming and curve statistics.
- **Factor directions come from single-author grids** (36 spans each) and the era direction is
  partly vocabulary, as recorded in hour 6.
