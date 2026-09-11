# Narrative factors are directions: measuring and steering story structure in transformer activations

*Draft. Numbers are from RESULTS.md; every claim there carries its controls and file references.*

## Summary

We asked whether the structure a prompt describes, a relation between parts or a narrative
property like setting or mood, exists as a geometric object in a language model's residual stream
that can be measured, moved across domains, and used to steer generation. Across Qwen2.5 (0.5B,
1.5B, and instruct), Pythia-1.4B, GPT-J-6B, and Gemma-2-9B-it, we found:

1. **Discourse position dominates pooled activations.** The naive "shape" of a prompt, the distance
   structure over its parts, is mostly *where each part sits in the template*, not what it says.
   Shuffling parts across slots leaves cross-domain agreement at 0.52 (vs 0.60 unshuffled);
   relabeling by content drops it to chance.
2. **Role identity is a domain-independent direction.** Which part of a schema a span is (the
   embedded state, the disturbance, the view from above) is linearly readable regardless of
   domain and position, and causally usable: adding the direction in a held-out domain selects
   that role's span at rank 1.7 of 6 (random directions: 3.3, chance 3.5).
3. **A source→target relation transfers weakly, and it is computed, not lexical.** After removing
   position, role identity, and domain address, an affine map fit on seven domains moves a
   held-out domain's source content toward its target content at rank 3.0 vs a shuffle null of
   3.5, at chance at the embedding layer and peaking at layer 20 of 28. As a steering patch it
   adds nothing measurable (−0.02 nats vs −0.19 for random).
4. **Narrative factors are additive directions with small, measured interference.** Era, voice,
   tense, mood, and theme each transfer to unseen scenes as a lens (rank 1.1–1.3 of 3 vs ~2.0
   random). Three compose in one patch (2.8 of 18, chance 9.5). The cross-talk matrix, the fraction
   of gain variance one factor's patch induces in another, is diagonal: 6–20% off-target vs 47–76%
   on-target. Order of application across two layers changes the outcome by half a rank.
5. **Factors differ in kind, and the model's depth shows it.** Voice and tense are readable at the
   embedding layer (lexical). Era is at chance at layer 0 and 94% readable by layer 12
   (computed). Mood is integrated at the last token, mid-late. Theme is distributed across a
   passage and read from the mean, not the last token.
6. **Steering selects among competences the model already has.** Theme is a reliable selector on
   every model but steers generation only where the model can already write on a theme: not at
   all on a 1.5B base model, partly on 1.5B-instruct, and clearly on Gemma-2-9B-it
   (*"everything she had built her life upon was now a lie"*; *"she hadn't expected to hear from
   him again, not after all these years"*). The theme direction also sharpens with scale:
   decodability 0.78 → 0.83 → 0.94 from 1.5B to 6B to 9B.
7. **Tuned models carry an unlisted factor.** On an instruct model the homecoming direction,
   inside the chat template, triggered a canned refusal. Refusal is a direction the grids never
   name, and it must be projected out before steering tuned models.

## Method in one paragraph

Prompts mark spans with roles; each span's residual vector is pooled (mean, or last token) at
every layer. Directions are estimated leave-one-out: a role or factor direction is the mean of that
role's vectors over training domains minus the grand mean, so no direction ever sees its test
domain. Three tests. *Selector:* add the direction at every position during a teacher-forced pass
and ask whether the log-probability of the matching span rises more than the others'; rank among
candidates, chance is the midpoint; the control is a random direction of equal norm. *Composition:*
sum several directions and rank the joint variant among all variants. *Cross-talk:* decompose the
gain matrix under one factor's patch into per-factor main effects. Nulls are permutation (scramble
role correspondence) or pairing shuffles within training folds. Every number in RESULTS.md is
paired with its control.

## What did not work, and why

- The **shape test** (RSA/CKA over pooled role vectors) was a position detector. Rotating content
  through slots and projecting out the slot subspace did not rescue a content shape. The fix is a
  different object (directions and operators), not better post-processing.
- The **relation lens** is a genuine but small signal starved of data: seven effective domains
  cannot support a rotation in 1536 dimensions. A constant "which role is this" prediction beats
  the learned map until role identity is removed; after that the map beats a null but not by
  enough to patch with.
- **Theme generation on small base models**: multi-layer re-imposition and repetition penalties
  do not move it. The obstacle is competence, not erosion or decoding.
- Two metrics were found broken and replaced (a cross-talk rank that averaged to 2 by
  construction; best-layer selection that inflated a constant baseline from 3.5 to 2.4).

## Limitations

All grids were written by one author, so a single writer's tics could be part of any "factor".
Sentence-scale spans for most factors; three-sentence passages for theme. Eight domains, four
scenes, three levels per factor. Most positive results are selector-level; generator-level
evidence is qualitative and model-dependent. Layer choices are mid-stack by convention, with a
sweep for the role lens only. No sparse-autoencoder analysis yet.

## What this supports and what it doesn't

It supports a **sentence-level additive calculus of surface narrative factors**: setting, register,
tense, and mood are directions that transfer, compose, and interfere little, on three model
families. It supports **roles as directions** and the model's **depth as a generality scale**. It
does **not** yet support the strong original hypothesis, that a relation learned in known domains
can be pointed at an unknown one to name its missing parts; that lens is measurable but not
controllable at current data sizes. And it locates the boundary of steering at the model's own
competence: the calculus's generative reach is bounded by the model, not by the geometry.

## Next

- Data for the relation lens: thirty-plus domains with paraphrases; refit the role-centered
  operator; the thesis→antithesis spin test.
- The joining experiment: shift a themed passage's era and test whether the theme selector still
  fires (move the address, keep the form).
- Abstraction as deterritorialization via sparse-autoencoder feature splitting across dictionary
  widths (Gemma Scope), with the abstraction-flow / fixed-point test in VISION.md.
- Derivative curves of factor readouts along passages; the absential test (a withheld part whose
  direction is nonetheless active).

## Reproducibility

`README.md` for setup; `scripts/stage*.py` for every experiment; `results/*.json` for per-case
records; `prompts/*.json` for every grid. Local runs are CPU-only on 0.5B–1.5B models; remote runs
use NDIF through `src/lsx/ndif.py`.
