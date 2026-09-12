# Hour 31: does Llama-3.1-70B-Instruct cross the recomposition boundary at scale?

Script: `scripts/ndif_recompose_sweep.py` (prediction recorded in the docstring before running).
Data: `results/recompose_sweep_70b_2.0.json`, `results/recompose_sweep_70b_3.0.json`.
Stacks: `results/stacks_llama_3.1_70b_instruct_narrative_theme_v1.npz` (re-extracted this run — no
prior copy existed in any accessible worktree; 36 spans, one NDIF replica eviction at span 21/36,
recovered by re-running the checkpointed script, which resumed from span 21).

## Question

Hour 27 found the era-shift patch on Llama-3.1-70B-Instruct (patch@26, read@40, re-imposed at every
decoding step) moves the continuation's era readout in only 14% of cases at scale 1.0, and the
lexical check was unanimously null (0.00) — both far below Gemma-2-9B-it's already-modest 0.27/0.00
at the same scale. Hour 29 found that turning the scale up to 2.0 and 3.0 on Gemma crosses era-target
0.5 (0.58, then 0.84), with the lexical check turning on at the same point (0.18, 0.30). Does the same
scale increase cross the boundary on the 70B?

## Results

meta-llama/Llama-3.1-70B-Instruct: patch@26 read@40 (80 blocks, d=8192). Gemma-2-9b-it: patch@14
read@20 (42 blocks, d=3584). Both: re-imposed every decoding step (`tracer.all()`), 48-token greedy
continuations, chat-templated "continue this story" prompt, "shift" condition only (n=72 per cell:
2 target eras × 36 passages), scored exactly as hour 27/29 (continuation re-read unpatched,
readout-layer residual mean-pooled, grand mean subtracted, nearest-cosine classification).

| model | scale | n | era reads as target | leaves e1 | theme kept | lex reads as target (n with any era word) | jobs lost |
|---|---|---|---|---|---|---|---|
| Llama-3.1-70B-Instruct (hour 27) | 1.0 | 72 | 0.14 | 0.21 | 0.54 | 0.00 (14) | 0/144\* |
| Llama-3.1-70B-Instruct (this run) | 2.0 | 72 | 0.28 | 0.39 | 0.54 | 0.17 (18) | 0/72 |
| Llama-3.1-70B-Instruct (this run) | 3.0 | 72 | **0.43** | 0.54 | 0.53 | **0.25** (20) | 0/72 |
| Gemma-2-9b-it (hour 29 baseline) | 1.0 | 67 | 0.27 | 0.48 | 0.60 | 0.00 (14) | 5/72 |
| Gemma-2-9b-it (hour 29) | 2.0 | 59 | 0.58 | 0.76 | 0.54 | 0.18 (11) | 13/72 |
| Gemma-2-9b-it (hour 29) | 3.0 | 55 | **0.84** | 0.91 | 0.53 | **0.30** (10) | 17/72 |

\*Hour 27's 144-generation run (both models, all conditions) lost 0 jobs on the 70B; its 9 losses
were all on Gemma.

Full generations completed cleanly at both scales: **0 jobs lost out of 144** (72 + 72) on the 70B,
in contrast to Gemma's 13/72 and 17/72 losses at the matching scales in hour 29. The 70B's job-loss
rate does not rise with scale the way Gemma's did — a further point of difference between the models,
not just in whether the patch moves the readout but in whether NDIF's serving stack finds the patched
activations troublesome enough to drop the job.

## Verbatim examples

**scale 2.0**
- *debt, farfuture→medieval* [reads 1920s; **lex on-target: medieval**]: "Tamsin's hands shook as she
  replied, her voice barely above a whisper, and her mother's laughter crackled through the line, a
  sound that brought back memories of summer afternoons and warm firesides. The **station's steward**…"
  (blended vocabulary within one sentence — a farfuture "station" carrying its own steward)
- *door, farfuture→medieval* [reads medieval; lex on-target: medieval]: "The captain's eyes fell upon
  the dark waters of the harbor, where the reflection of the **castle** walls seemed to writhe and
  twist like living flames. He felt a cold dread creeping up his spine as he realized that Ando's
  treachery…" (readout and vocabulary both cross, narrative logic intact)
- *debt, medieval→1920s* [reads medieval; lex: medieval, i.e. stayed]: "The steward's eyes narrowed,
  unsure what to make of the apparition before him, but the son's easy smile and the familiar way he
  leaned against the worn wooden counter put him at ease. 'She still bakes,' the steward said…" (the
  more typical scale-2 case — fluent, unmoved)

**scale 3.0**
- *message, farfuture→medieval* [reads medieval; **lex on-target: medieval**]: "As the sun rose over
  the **castle** walls, Reyes stood in the great hall, his head bowed in acceptance of his fate, while
  the other servants and guards looked on in a mixture of shock and relief. The **lord's steward** took
  the confession…" (clean relocation — the far-future scene now reads as a medieval court scene)
- *door, farfuture→medieval* [reads medieval; lex on-target: medieval]: "The captain's eyes fell upon
  the figure of his loyal **steward**, who stood frozen in the doorway, a look of horror on his face
  as he beheld his master's treachery. The steward's eyes met the captain's…" ("captain" — the
  farfuture original — survives inside an otherwise medieval scene: blend, not replacement)
- *meal, 1920s→farfuture* [reads 1920s; **lex on-target: farfuture**]: "The creditors' **drones**
  arrived on Tuesday, their polite but firm demands for repayment echoing through the empty shop as
  Rose stood frozen, the weight of betrayal settling like a shroud around her heart…" (era readout
  stays at 1920s even though a farfuture noun — "drones" for debt-collectors — has intruded; the
  same disconnect between readout and lexical crossing hour 29 saw on Gemma at scale 2)

## Prose-quality note

Scanned continuations at both scales: prose stays fluent, grammatical, and narratively coherent
throughout — no repetition loops, truncated syntax, or token-salad at scale 2.0 or 3.0, matching what
hour 29 found on Gemma. Where the patch is visible at all, it shows as thematic bleed (a "station's
steward," a captain's era intruding on a medieval scene, "creditors' drones") rather than as broken
prose, and the 70B's continuations remain closer to their un-patched form clause-for-clause than
Gemma's did at matching scale — consistent with hour 27's finding that the 70B's stronger prior
produces near-verbatim rewrites of the base continuation with a clause repainted, rather than a
wholesale rewrite.

## Prediction, graded

**Prediction (written into the script docstring before running):** the 70B crosses the same threshold
as Gemma — era-as-target ≥ 0.5 at scale 3.0 — with a lower lexical rate than Gemma's 0.30, because its
stronger prior resists lexical intrusion even where the readout is pushed off-target.

**Refuted on the crossing, confirmed on the lexical comparison.** Era-as-target on the 70B reaches only
0.43 at scale 3.0 — up sharply from 0.14 at scale 1.0 and 0.28 at scale 2.0, tripling over the sweep,
but it does not cross 0.5 at the scale where Gemma reaches 0.84. The two models are not on the same
curve: Gemma crosses the 0.5 threshold between scale 2 (0.58) and 3 (0.84); the 70B's scale-3 point
(0.43) sits below even Gemma's scale-2 point. The lexical half of the prediction holds: the 70B's
lexical-target rate at scale 3.0 (0.25) is lower than Gemma's (0.30), and this gap is smaller than the
era-readout gap between the models (0.43 vs 0.84) — the 70B's vocabulary is proportionally *more*
resistant to the patch than its era readout is, the opposite ordering from what would be expected if
readout and lexical crossing moved in lockstep. This is consistent with the "stronger prior resists"
mechanism named in the prediction, but the magnitude is larger than predicted: the 70B does not merely
lag Gemma by a fixed offset, it appears to need a materially larger scale (untested here) or a
different layer pair to reach the same boundary, if it reaches it at all under this patch geometry —
see the confound flagged in the script docstring: patch@26/read@40 is a fixed *fraction* of the 70B's
80 blocks, not the same absolute depth as Gemma's patch@14/read@20 in 42 blocks, so "same relative
layers" is not established to mean "same mechanism" across models of this different depth and width.

## Standing caveats

Two scales only (2.0, 3.0); the sweep does not establish whether a still larger scale (4× or beyond)
would eventually cross 0.5 on the 70B, or whether the ceiling is closer to 0.43-0.5 regardless of
scale. One patch/read layer pair on the 70B (26/40), unswept, and the layer-fraction-vs-absolute-depth
confound above is not resolved. All 144 generations across both scales completed with 0 lost NDIF
jobs — better job-loss behavior than Gemma showed at the same scales, itself worth noting as a
model-dependent property of the serving stack, not just of the patch's effect on the model. Stacks for
the 70B were re-extracted fresh this run (one NDIF replica eviction recovered via checkpoint/resume);
they were not present in any accessible copy of the repository, gitignored as expected.
