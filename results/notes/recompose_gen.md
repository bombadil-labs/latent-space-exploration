# Recomposition at the generation level: the era shift does not survive the Engine

Script: `scripts/ndif_recompose_gen.py` (the pre-registered design is its docstring).
Data: `results/recompose_gen_llama70b.json`, `results/recompose_gen_gemma9b.json`,
`results/stacks_llama_3.1_70b_instruct_narrative_theme_v1.npz`.

## Question

Hour 14 measured, at the *representational* level, that adding `dir_era[e2] - dir_era[e1]` while the
model reads a passage moves its era readout (0.89 Qwen-1.5B / 0.88 Gemma-9B-it) and leaves the theme
readout exactly where it was (0.81 / 0.94). That is law L4 of `docs/ALGEBRA.md`, and it is a *Gauge*
result. Law L3 says Gauges compose and Engines do not. This run puts the Engine in the loop: apply
the era shift, let the model **continue** the passage, and score the continuation.

## Design (fixed before the run; full text in the script docstring)

- Models: **meta-llama/Llama-3.1-70B-Instruct** (smoke-tested through NDIF: 80 blocks, d = 8192;
  patch layer 26 ≈ 1/3 depth, readout layer 40 ≈ 1/2 depth) and **google/gemma-2-9b-it** (42 blocks,
  d = 3584; patch 14, readout 20 — the hour-14 pair, kept for comparability).
- Grid `prompts/narrative_theme_v1.json`, 4 scenes × 3 eras × 3 themes. Theme stacks for Llama-70B
  extracted fresh (36 NDIF jobs, `scripts/ndif_extract.py`); Gemma's already existed. The GPT-authored
  grid was not extracted — the NDIF budget went to the two generation runs instead.
- Leave-one-scene-out era and theme directions, exactly as `scripts/ndif_shift.py` builds them.
- Per passage: greedy 48-token continuation under (a) no patch, (b) the era shift for each of the two
  other eras, (c) a random direction of the same norm; the patch is added at every position including
  generated ones (`with tracer.all():`). Scale 1.0. 36 × 4 = 144 generations per model. Scale 1.5 was
  pre-registered as optional and was not run: the NDIF queue and the two full runs consumed the budget.
- Scoring: the continuation **alone** is re-read unpatched as `"A passage from a story: <continuation>"`,
  readout-layer residuals over the continuation tokens are mean-pooled, the training-fold grand mean
  is subtracted, and the result is classified by nearest era and nearest theme direction. Re-reads are
  batched six per job via `tracer.invoke` (verified to return vectors bit-identical to one-per-job).
- Lexical check with era word lists fixed in advance (listed in the script, echoed in each JSON's meta).

Two deviations from the plan, both recorded before the measured run or as they happened:

1. **Prompt format.** The raw `"<lead> <span>"` prompt of hour 13 makes both instruction-tuned models
   answer the passage as a *comprehension exercise* ("**What is the story likely about?** This passage
   suggests a story about: * Betrayal and family conflict…"), not continue it. A pilot on three passages
   per model fixed the prompt to hour 13's chat-templated "continue this passage" form **before** any
   measured generation was run.
2. **Lost jobs.** NDIF intermittently returns a COMPLETED job with an empty payload, deterministically
   for particular (passage, patch) pairs. Nine of Gemma's 144 generations were lost this way (all in
   patched conditions); none of Llama's. They are recorded with `failed` in the JSON and excluded from
   the tables. `scripts/ndif_extract.py` also gained a per-span checkpoint and a `retry_job` wrapper
   after a 70B queue stall killed a 21-span extraction, and a `.cpu()` in its layer stack because the
   70B is model-parallel and `torch.stack` across devices fails.

## Results

### meta-llama/Llama-3.1-70B-Instruct (80 blocks, d=8192), patch@26 read@40, scale 1.0, 48 tokens, chat prompt; 144 generations, 0 lost

| condition | n | era moved (e2) | era stayed (e1) | era other | theme kept (t) |
|---|---|---|---|---|---|
| base | 36 | -- | 0.86 | 0.14 | 0.61 |
| shift | 72 | **0.14** | 0.79 | 0.07 | 0.54 |
| rand | 36 | -- | 0.81 | 0.19 | 0.56 |

| condition | n with era words | lex moved | lex stayed | lex other |
|---|---|---|---|---|
| base | 9/36 | -- | 1.00 | 0.00 |
| shift | 14/72 | **0.00** | 1.00 | 0.00 |
| rand | 9/36 | -- | 1.00 | 0.00 |

| e1 → e2 | n | moved | theme kept |
|---|---|---|---|
| 1920s → farfuture | 12 | 0.08 | 0.58 |
| 1920s → medieval | 12 | 0.33 | 0.50 |
| farfuture → 1920s | 12 | 0.08 | 0.67 |
| farfuture → medieval | 12 | 0.17 | 0.58 |
| medieval → 1920s | 12 | 0.08 | 0.33 |
| medieval → farfuture | 12 | 0.08 | 0.58 |

### google/gemma-2-9b-it (42 blocks, d=3584), patch@14 read@20, scale 1.0, 48 tokens, chat prompt; 144 generations, 9 lost

| condition | n | era moved (e2) | era stayed (e1) | era other | theme kept (t) |
|---|---|---|---|---|---|
| base | 36 | -- | 0.78 | 0.22 | 0.53 |
| shift | 67 | **0.27** | 0.52 | 0.21 | 0.60 |
| rand | 32 | -- | 0.69 | 0.31 | 0.53 |

| condition | n with era words | lex moved | lex stayed | lex other |
|---|---|---|---|---|
| base | 6/36 | -- | 1.00 | 0.00 |
| shift | 14/72 | **0.00** | 1.00 | 0.00 |
| rand | 7/36 | -- | 1.00 | 0.00 |

| e1 → e2 | n | moved | theme kept |
|---|---|---|---|
| 1920s → farfuture | 11 | 0.18 | 0.45 |
| 1920s → medieval | 10 | 0.00 | 0.50 |
| farfuture → 1920s | 11 | 0.27 | 0.55 |
| farfuture → medieval | 11 | 0.09 | 0.82 |
| medieval → 1920s | 12 | **0.75** | 0.67 |
| medieval → farfuture | 12 | 0.25 | 0.58 |

### The fair control

The `rand` condition has no designated target, so its "moved" cell is undefined, not zero; the
honest control is **how often the continuation leaves e1 at all**, and then how much of that leaving
lands on the intended era rather than on the third one (50% if the displacement were undirected).

| model | base leaves e1 | rand leaves e1 | shift leaves e1 | of those, on target |
|---|---|---|---|---|
| Llama-3.1-70B-Instruct | 0.14 | 0.19 | **0.21** | 0.67 |
| Gemma-2-9b-it | 0.22 | 0.31 | **0.48** | 0.56 |

On the 70B the shift barely displaces the continuation's era at all (0.21 vs 0.19 for a matched-norm
random direction, n = 72 vs 36 — no effect). On the 9B it displaces it clearly (0.48 vs 0.31), but
only just over half of that displacement is aimed at the era the patch names: the patch destabilises
the era readout more than it steers it.

### Lexical check

The pre-registered word lists are the blunt instrument and they say the same thing more plainly.
Era-marked vocabulary appears in 6–14 of 36–72 continuations per condition, and in **every single
case, on both models, in all three conditions, the era it marks is the original e1 and never the
target e2**. No continuation under an era shift acquired a telegram, an automobile, a starship or an
airlock it did not already have. The shifted continuations remain, lexically, in the era of the
passage they continue.

## Verbatim examples

### debt / medieval / betrayal

*Passage.* The steward came for the tithe at harvest, as he had every year. This year he carried the lord's seal on a writ that doubled it, and the writ was in the hand of the miller's own brother, who had sworn at their father's grave to stand with him. The miller paid, and did not look at his brother, and his brother did not look at him.

**Llama-3.1-70B-Instruct**

- *base* [reads medieval / homecoming]: The miller's wife stood by the door, her eyes fixed on the brother who had once been a part of their family, her hands clenched into fists at her sides as if to keep from reaching out. The steward, a man
- *shift → 1920s* [reads medieval / homecoming]: The miller's wife stood by the door, her eyes fixed on the brother who had once been her confidant, her hands clenched into fists at her sides as if to keep from reaching out. The steward, a man with
- *shift → farfuture* [reads medieval / betrayal]: The miller's hands shook as he handed over the grain, the weight of his brother's betrayal settling heavy in his chest. His brother's eyes, once bright with laughter and shared secrets, now seemed dull and foreign, a stranger's
- *rand* [reads medieval / homecoming]: The miller's wife stood by the door, her eyes fixed on the brother who had once been a part of their family, her hands clenched into fists at her sides as if to keep from reaching out. The steward, a man

**gemma-2-9b-it**

- *base* [reads medieval / homecoming]: The miller's wife, her face drawn and thin, watched from the doorway, her hands clasped tight in her apron.  That night, she wept, not for the lost grain, but for the brother who had become a stranger.
- *shift → 1920s* [reads 1920s / homecoming]: The silence stretched between them, thick and heavy as the flour dust that coated the miller's apron.  The brother cleared his throat, the sound echoing in the small, cramped space. "I'll be seeing you around, then
- *shift → farfuture* [reads medieval / sacrifice]: The miller watched the dust cloud settle on the road as the brother's cart disappeared, the weight of the doubled tithe settling heavier than any grain sack ever could.  He knew the harvest would barely cover the new demand, let alone the
- *rand* [reads medieval / homecoming]: The miller's wife, her face drawn and thin, watched from the doorway, her hands clasped tight in her apron.  That night, she wept silently, her tears falling onto the rough floorboards, each one a tiny echo of

### message / 1920s / homecoming

*Passage.* The telegram came to the newsroom at midnight, from a town Ellen had not thought about in twenty years. Her father was dying and had asked for her by the name no one had used since she left. She was on the first train west, and she found she still knew every stop by heart.

**Llama-3.1-70B-Instruct**

- *base* [reads 1920s / homecoming]: As the train rumbled through the familiar landscape, memories she'd long suppressed began to resurface, like the faint scent of smoke from a fire thought to be extinguished. The rhythmic clickety-clack of the tracks seemed to echo
- *shift → medieval* [reads 1920s / homecoming]: As the train rumbled through the familiar landscape, memories long buried began to resurface, and Ellen felt the weight of years of unresolved emotions bearing down upon her. The rhythmic clickety-clack of the tracks seemed to echo the beat
- *shift → farfuture* [reads 1920s / homecoming]: The rhythmic clickety-clack of the tracks and the whistle's mournful cry at each station brought back memories she'd long suppressed, of summers spent exploring dusty main streets and winters huddled by the fire, listening to her father's
- *rand* [reads 1920s / homecoming]: As the train rumbled through the familiar landscape, memories she'd long suppressed began to resurface: the smell of the diner's coffee, the sound of the river at dawn, and the feel of the dusty main street beneath her feet.

**gemma-2-9b-it**

- *base* [reads 1920s / homecoming]: The train rattled through the night, each click of the wheels a hammer blow against the dam of her carefully constructed life.  She hadn't thought about him in years, hadn't wanted to, but now the past was a rushing
- *shift → medieval* [reads 1920s / homecoming]: The train rattled through the night, each click of the wheels a hammer blow against the silence of her memories.  She closed her eyes, and the scent of pine and woodsmoke, the taste of her mother's apple pie, flooded
- *shift → farfuture* [reads 1920s / homecoming]: The train rattled through the night, each click of the wheels a hammer blow against the dam of her carefully constructed life.  She hadn't expected to feel this pull, this ache for a past she'd buried so deep.
- *rand* [reads 1920s / homecoming]: The train rattled through the night, each click of the wheels a hammer blow against the dam of her carefully constructed life.  She hadn't thought about him in years, hadn't wanted to, but now the past was a rushing

### door / farfuture / sacrifice

*Passage.* The bulkhead to the reactor deck answered only to Chief Ando's print. When the coolant line ruptured he sealed the others out and himself in, because the manual override was on the wrong side and someone had to hold it. He kept the channel open and talked them through the shutdown until he stopped.

**Llama-3.1-70B-Instruct**

- *base* [reads farfuture / homecoming]: The silence that followed was oppressive, punctuated only by the steady hiss of the reactor's dying breath and the muffled voices of his crewmates, their words indistinguishable but their fear and desperation clear. Lieutenant Commander Patel's voice cut
- *shift → medieval* [reads farfuture / homecoming]: The silence that followed was oppressive, punctuated only by the steady hiss of steam and the creaking of twisted metal. Lieutenant Commander Patel's voice, tight with worry, called out to him, but there was no response. The others
- *shift → 1920s* [reads farfuture / homecoming]: The silence that followed was oppressive, punctuated only by the steady hiss of the reactor's dying breath and the muffled voices of his crewmates on the other side of the bulkhead, their words indistinguishable but their worry and fear
- *rand* [reads farfuture / homecoming]: The sudden silence was a heavy weight on the bridge, and for a moment, no one spoke, the only sound the steady beep of the reactor's power-down sequence. Then Lieutenant Commander Patel's voice, tight with emotion, called out,

**gemma-2-9b-it**

- *base* [reads farfuture / betrayal]: His voice was calm, even as the alarms screamed and the floor vibrated beneath his boots.  He could feel the heat radiating from the breached pipe, smell the acrid tang of ozone in the air.  He just had to hold
- *shift → medieval* [reads farfuture / homecoming]: His voice was calm, even as the heat licked at his boots and the air grew thick with the metallic tang of ozone.  He could feel the tremor in the ship, a shuddering sigh as the reactor fought to stay alive.
- *shift → 1920s* [reads farfuture / betrayal]: His voice was calm, even though his hands were shaking.  He could feel the heat radiating through the metal, smell the acrid tang of ozone in the air.  He just had to hold on until the emergency cooling system kicked in
- *rand* [reads farfuture / homecoming]: His voice, usually gruff and laced with sarcasm, was thin and reedy, strained by the heat and the fear that coiled in his gut.  He could feel the tremors in the deck beneath his boots, the ship groaning in protest

### meal / medieval / homecoming

*Passage.* The lord set a feast on the eve of the campaign, and halfway through the meal the doors opened on a man in a pilgrim's cloak. It was his son, gone to the Holy Land seven years ago and thrice reported slain, and the boy knelt at the foot of the high table and asked for his old seat. The lord came down from the dais and raised him up.

**Llama-3.1-70B-Instruct**

- *base* [reads medieval / homecoming]: His son's eyes, once bright as the summer sun, now seemed to hold a deep and abiding sorrow, and the lines etched on his face told the tale of trials and hardships endured. The lord's wife, the boy's
- *shift → 1920s* [reads medieval / homecoming]: His son's eyes, once bright as the summer sun, now seemed to hold a deep weariness, and the lines etched on his face told a story of trials and hardships that the lord could hardly bear to imagine. The room fell
- *shift → farfuture* [reads medieval / homecoming]: His son's eyes were sunken, his skin sallow, and his once-sharp jawline now gaunt, but the fire in his gaze remained, burning brighter than the candles that lit the hall. The room fell silent, the
- *rand* [reads medieval / homecoming]: His son's eyes, once bright as the summer sun, now seemed to hold a deep and abiding sorrow, and the lines etched on his face told the tale of trials and hardships endured. The lord's wife, the boy's

**gemma-2-9b-it**

- *base* [reads medieval / homecoming]: Tears welled in the lord's eyes, blurring the candlelight, and he embraced his son, the weight of the years melting away.  The hall fell silent, every eye fixed on the reunion, the clinking of goble
- *shift → 1920s* [reads 1920s / homecoming]: A hush fell over the hall, the clinking of goblets and the murmur of conversation dying away as all eyes turned to the returned son.  The boy, thinner and taller than they remembered, looked around the room, a ghost
- *shift → farfuture* [reads 1920s / homecoming]: A hush fell over the hall, the clinking of goblets and the murmur of conversation dying away as all eyes turned to the returned son.  The boy, though thinner and weathered, still bore the same mischievous glint in his
- *rand* [reads medieval / homecoming]: Tears welled in the eyes of the assembled guests, for they had mourned the boy as dead and now saw him returned, a living miracle.  The lord, his voice thick with emotion, declared a new toast, to the son who

The examples are the finding in miniature. Under an era shift the 70B rewrites a clause or two —
"a part of their family" becomes "her confidant", "the reactor's dying breath" becomes "steam and
the creaking of twisted metal" — and then continues in the era it was given. The passage never
relocates: the miller still pays a tithe, Ellen is still on a train, Chief Ando is still on a
reactor deck. Gemma moves further (the medieval feast loses its candlelight for "a hush fell over
the hall", and its readout flips to 1920s) but it does not move *toward* the named era either: the
same continuation is produced for `→ 1920s` and `→ farfuture`, which is the signature of a
displacement without a direction.

## Verdict

**No. The era shift does not move the address in generated text, and this is not fixed by scale.**

At the representational level the same patch, on the same model, at the same layers, moved the era
readout in 88% of cases (hour 14). With the Engine in the loop and the score taken on the *new* text
rather than the patched context, the shift moves the era readout of the continuation in 14% of cases
on Llama-3.1-70B-Instruct and 27% on Gemma-2-9b-it, against 19% / 31% of matched-norm random
directions merely knocking the readout off e1. The lexical check is unanimous and null on both
models: not one shifted continuation acquired the vocabulary of its target era.

The form *is* kept — theme reads as t in 0.54–0.61 of continuations on Llama and 0.53–0.60 on Gemma,
statistically the same under shift, base and random. But this is the weaker half of L4 and it is
cheap here: the theme readout on a 48-token continuation is much noisier than on a three-sentence
passage (0.94 in hour 14), and "kept" is indistinguishable from "unmoved by anything we did".

**Does the 70B do it better than the 9B? No — it does it *less*.** The prediction written into the
script before the run (the 70B should move more cleanly, by the hour-13 steering-competence curve)
is falsified. The 70B is the more *stable* model: its continuations under a patch are near-verbatim
re-runs of the base continuation with a clause repainted, and its era readout stays on e1 in 79% of
shifted cases. The 9B is more disturbable, and what looks like a better score on it (0.27 vs 0.14) is
mostly undirected disturbance — 0.56 of its displacement lands on target, barely above the 0.50 a
coin would give. Steering competence at 9B-instruct (hour 13, theme) does not generalise to a
*recomposition* patch at 70B: a bigger model is a stronger prior, and the prompt-prior effect flagged
in hour 13 ("a continuation the model is already confident about resists a patch") scales up with it.

This is law L3 of `docs/ALGEBRA.md` collecting its clearest instance so far. L4 should be read as
narrower than it was written: **the era shift moves the address of a passage's representation, not
the address of the passage the model then writes.** Recomposition is, on this evidence, a Gauge
operation only.

Standing caveats. One scale (1.0) only; the pre-registered 1.5 arm was not run, so "not fixed by
scale" is an inference from the hour-14 patch norm being sufficient at the Gauge level and from the
70B's flat response, not a measurement. One grid, one patch/readout layer pair per model, chosen by
the 1/3–1/2 depth convention with no sweep. Nine Gemma generations lost to NDIF. The continuation
readout is the weak link: its base accuracy (0.78–0.86 era, 0.53–0.61 theme) bounds every number in
the tables, and a sharper continuation-scale readout could change the size of the effect, though not
the lexical null.
