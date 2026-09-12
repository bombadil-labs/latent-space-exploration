# Hour 29: crossing the recomposition boundary — scale and re-imposition on Gemma-2-9B-it

Script: `scripts/ndif_recompose_sweep.py` (pre-registered design is its docstring).
Data: `results/recompose_sweep_reimpose_0.5.json`, `..._reimpose_2.0.json`, `..._reimpose_3.0.json`,
`..._prefix_1.0.json`, `..._prefix_2.0.json`. Baseline reused unchanged: `results/recompose_gen_gemma9b.json`.

## Question

Hour 27 found that the era-shift patch on Gemma-2-9B-it (patch@14, read@20, scale 1.0) moves the era
readout of a generated continuation in only 27% of cases and never changes the continuation's
vocabulary (lexical check 0.00). Does either turning up the scale, or a different patch mechanism, push
that number past the boundary?

**A note on the premise, recorded before running.** The task brief describes hour 27's patch as a
"prefix patch" and re-imposition (patching every generated token via `tracer.all()`) as a new condition
to test. Reading `scripts/ndif_recompose_gen.py`, hour 27's run already uses `tracer.all()` — its
scale-1.0 "shift" numbers (era target 0.27, lex 0.00) are a re-imposition run. So the two axes actually
tested here are: **(A) scale**, at the re-imposition mechanism hour 27 used, and **(B) mechanism**, a true
prefix-only patch (verified below to apply only at the prompt's forward pass, iteration 0, and not to any
generated token) at matched scale, compared against re-imposition. Hour 27's own number stands as both
the scale=1.0 anchor for (A) and the re-imposition anchor for (B); it was not re-run.

**Mechanism check**, before the measured run: a large (norm-40) direction was patched into layer 14 of
one 16-token generation, with and without `tracer.all()`. Prefix-only reproduced the unpatched
continuation exactly; re-imposition diverged from token 6 onward. This confirms an intervention written
directly inside `model.generate(...) as tracer:`, without `tracer.all()`/`tracer.iter`, applies only to
the prompt's forward pass — the "prefix-only" mechanism used below.

**Budget.** Only the "shift" arm (2 target eras × 36 passages = 72 generations) was run per condition —
the pre-registered 72-generation fallback, since 5 new conditions × 144 would be 720 generations. "base"
(no patch, cannot depend on scale/mechanism) and "rand" (scale-dependent, not re-run) are taken from hour
27 as reference only.

## Results

Gemma-2-9b-it, patch@14 read@20, 48-token greedy continuations, chat-templated "continue this story"
prompt, shift condition only (era read as target / leaves e1 / theme kept / lexical era check as target,
scored exactly as in hour 27: continuation re-read unpatched, readout-layer residual mean-pooled,
grand mean subtracted, nearest-cosine classification).

| condition | mechanism | scale | n | era reads as target | leaves e1 | theme kept | lex reads as target (n with any era word) |
|---|---|---|---|---|---|---|---|
| hour-27 baseline | reimpose | 1.0 | 67 | 0.27 | 0.48 | 0.60 | 0.00 (14) |
| reimpose_0.5 | reimpose | 0.5 | 67 | 0.21 | 0.37 | 0.61 | 0.00 (11) |
| reimpose_2.0 | reimpose | 2.0 | 59 | **0.58** | 0.76 | 0.54 | **0.18** (11) |
| reimpose_3.0 | reimpose | 3.0 | 55 | **0.84** | 0.91 | 0.53 | **0.30** (10) |
| prefix_1.0 | prefix only | 1.0 | 72 | 0.24 | 0.43 | 0.61 | 0.00 (13) |
| prefix_2.0 | prefix only | 2.0 | 72 | 0.36 | 0.61 | 0.57 | 0.08 (13) |
| hour-27 base (no patch) | — | — | 36 | -- | 0.22 | 0.53 | -- (6, all stayed) |
| hour-27 rand (matched norm, scale 1.0) | reimpose | 1.0 | 32 | -- | 0.31 | 0.53 | -- (7, all stayed) |

Job loss (NDIF returning a COMPLETED job with an empty payload, as in hour 27): reimpose_0.5 lost 5/72,
reimpose_2.0 lost 13/72, reimpose_3.0 lost 17/72; prefix_1.0 and prefix_2.0 lost none. Loss rate rises
with scale under re-imposition only, consistent with a larger repeated patch occasionally pushing
generation into a state NDIF fails to serialize — see caveats.

### Scale, at the re-imposition mechanism (axis A)

Era-target crosses 0.5 between scale 2 and 3 (0.58, then 0.84), and lexical crossing turns on at the
same point it was null at 0.5–1.0 (0.00 → 0.18 → 0.30). "Leaves e1" (0.91 at scale 3) shows the patch is
now dominating the readout almost completely, not merely nudging it.

### Mechanism at fixed scale (axis B)

At scale 1.0 the two mechanisms are statistically indistinguishable (era target 0.24 vs 0.27, lex 0.00
both). At scale 2.0 they separate clearly: reimpose reaches era-target 0.58 vs prefix-only's 0.36, and
lex 0.18 vs 0.08. Re-imposition is a stronger mechanism than a single prefix application at matched norm,
but scale is what does most of the work — even prefix-only at scale 2.0 is above hour 27's reimposed
scale-1.0 baseline (0.36 > 0.27).

## Verbatim examples

**reimpose, scale 2.0** — genuine lexical crossing, not just an era-readout flip:
- *message, medieval→farfuture* [reads farfuture]: "She paced the length of the hall, the worn wood cool beneath her bare feet, her eyes fixed on the distant horizon. Every creak of the aging **starship**, every flicker of the emergency lights, sent a jolt of anticipation…"
- *door, farfuture→medieval* [reads medieval]: "The **steward** bowed low. 'Aye, Lady Elara,' he said, 'and they are kept as he left them, with his **sword** upon the wall and his cloak upon the chair.' Elara smiled, a sad, thin smile…"
- *debt, medieval→1920s* [reads 1920s]: "The silence stretched between them, thick and heavy like the flour dust that coated everything in the mill. The miller watched his brother walk away, the writ crinkling in his hand, and wondered if h…" (era readout moves, vocabulary does not — the more typical case even at scale 2)

**reimpose, scale 3.0** — blended vocabulary within one sentence:
- *debt, 1920s→farfuture* [reads farfuture]: "The **airlock** hissed as he sealed the hatch, the familiar whine a mournful counterpoint to the silence of the void. He could almost hear Harmon's voice, a gravelly whisper echoing in his mind: 'Duty calls, Frank.'"
- *door, 1920s→medieval* [reads medieval]: "A hush fell upon the revelers as the soldiers entered, their **swords** gleaming in the torchlight. The air, thick with the scent of wine and mirth, turned suddenly cold, and the music ceased, leaving only the clang of stee…"
- *debt, medieval→farfuture* [reads farfuture]: "The **steward** blinked, the whirring of the mill a low hum against the silence of the station. 'Your mother passed three **cycles** ago,' he said, voice flat. 'You sho…'" (a genuine blend: steward/mill carried over, station/cycles crossed in)

**prefix-only, scale 2.0** — the same passages, weaker or absent crossing:
- *debt, medieval→1920s* [reads 1920s]: "The silence stretched between them, thick and heavy as the flour dust that coated the miller's apron. He watched his brother leave, the writ clutched in his hand like a weapon, and felt a familiar ac…" (era readout moves, no lexical crossing)
- *message, medieval→1920s* [reads 1920s]: "Her fingers tightened around the paper, the ink blurring through the tears that welled in her eyes. She knew what this meant, the whispers she'd heard in the halls, the hushed conversations in the ki…"
- *door, medieval→1920s* [reads medieval]: "He just watched, his face impassive, as the captain rode away, a small figure against the rising sun. The lord had known this day would come, had felt it in his bones, but seeing it unfold was a diff…" (readout does not even move here)

## Prose-quality notes

Scanned continuations at scale 2.0 and 3.0 (reimpose, the largest patches used): prose stays fluent and
grammatical throughout, including the cases with genuine lexical crossing above. No repetition loops,
truncated syntax, or token-salad were found in any scored continuation at any scale tested. The clearest
sign of the patch straining the model is not garbled text but the rising rate of lost NDIF jobs at
re-imposition scale 2–3 (13/72, 17/72) — consistent with, but not direct evidence for, the patch pushing
some generations into activation ranges NDIF's serving stack does not handle cleanly. Degradation, where
visible at all, shows as **thematic bleed within otherwise well-formed sentences** (a steward measuring
time in "cycles" beside a "station"; chariots invoked for "the realm's men of arms") rather than as
broken prose.

## Predictions, graded

**P1 — scale 2–3× (reimposed) raises "era reads as target" above 0.5 but degrades prose.**
Confirmed on the first half, refuted on the second. Era-target rises from 0.27 (scale 1) to 0.58 (scale
2) to 0.84 (scale 3), clearly crossing 0.5 between 2× and 3×. Prose does not degrade in the sense of
becoming disfluent or broken — every scanned continuation up to scale 3.0 remains grammatical, readable
story prose. What scale 2–3 buys instead of broken syntax is thematic bleed (mixed-era vocabulary within
one coherent sentence) and a rising NDIF job-loss rate, which is the closest thing to "cost" observed.

**P2 — re-imposition at 1.0 raises the lexical check above 0.0.**
Refuted, and refuted in a way that reveals the premise error above: re-imposition at scale 1.0 is not a
new condition, it is hour 27's own run, and its lexical check was already 0.00 — the same number is
reused here, not newly measured. At scale 1.0, prefix-only and re-imposed are statistically identical
(lex 0.00 both, era-target 0.24 vs 0.27). Re-imposition only pulls ahead of prefix-only once scale is
also raised: at scale 2.0, re-imposed reaches lex 0.18 vs prefix-only's 0.08, and at scale 3.0 (reimposed
only, not tested prefix-only) lex reaches 0.30. So mechanism matters, but only as an amplifier of scale,
not as an independent lever at the norm hour 14's Gauge patch used.

## What this changes about hour 27's verdict

Hour 27 concluded recomposition is a Gauge-only operation at the hour-14 patch norm; that stands. This
run shows the Gauge/Engine boundary is not a wall: scaling the same direction 2–3× past that norm, with
the patch re-imposed at every decoding step, does cross it — both in the era readout (0.84 at scale 3,
above hour 14's own representational-level 0.88) and, more strikingly, in the pre-registered lexical
check that hour 27 found unanimously null (0.30 at scale 3, on genuinely blended vocabulary like
"starship" and "airlock" appearing in continuations of a scene that opened in 1920s or medieval spans).
The cost is not the "broken prose" the pre-registered prediction expected, but two other things: thematic
bleed rather than clean relocation (the era vocabulary blends into the passage rather than replacing it),
and a rising rate of NDIF jobs that never return a payload at the larger, re-imposed patches.

## Standing caveats

One model only (Gemma-2-9b-it); the 70B was not retested at scale or under prefix-only, so nothing here
says whether its near-null hour-27 result would move the same way. One patch/read layer pair, unswept.
Scale 3.0 prefix-only and any scale above 3.0 reimposed were not run — this is a lower bound on where the
lexical crossing continues to rise, not a ceiling. The "shift"-only budget means the rand and base
comparators shown above are hour 27's own numbers, not scale/mechanism-matched controls, so the
"leaves e1" column at scale 2–3 should be read against a random-direction control that itself likely
also displaces more at larger norm — a control this run did not measure. Per-condition job loss (0–17/72)
means n varies across the table (55–72); this is reported, not size-corrected beyond simple accuracy
over completed jobs.
