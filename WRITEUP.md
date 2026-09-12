# Narrative factors are directions: measuring, moving, and composing story structure in transformer activations

*Third draft, after 28 logged stages. Every number below is in `RESULTS.md` with
its control and file reference; the calculus these results support is written out in
`docs/ALGEBRA.md`.*

## Summary

We asked whether the structure a prompt describes, a relation among parts or a narrative property
such as setting, register, mood, or theme, exists in a language model's residual stream as a
geometric object that can be measured, moved to a new domain, composed with other such objects,
and used to steer generation. Across Qwen2.5 (0.5B, 1.5B, 1.5B-Instruct), Pythia-1.4B, GPT-J-6B,
and Gemma-2-9B-it, with every claim paired to a matched null, we found nine things that stand,
four that fell, and two that are partial.

**Stand.**

1. Discourse position dominates pooled activations. The naive "shape" of a prompt is where each
   part sits in its template, not what it says.
2. Roles are domain-independent directions and causally usable: adding a role's direction in a
   domain it never saw selects that role's span (rank 1.7 of 6; random 3.3; chance 3.5).
3. The source→target relation between roles is real, computed by the stack, and, given forty
   domains, a working selector (rank 2.21 of 6 vs 3.5 null; peak at layer 16 of 28).
4. Narrative factors (era, voice, tense, mood, theme) are additive directions: each is a lens on
   unseen scenes (1.1–1.4 of 3 vs ~2.0 random), three compose in one patch (2.8 of 18, chance 9.5),
   and their cross-talk matrix is diagonal. Replicated on three model families and on grids
   written by a second model author.
5. Factors differ in kind, and the model's depth shows it: voice and tense are lexical, era is
   computed by layer 12, mood is integrated at the last token, theme is distributed and peaks
   mid-passage.
6. Recomposition works at the gauge level: an era shift moves a passage's address (0.89–0.94 of
   cases) and keeps its theme (0.81–0.94, equal to the unpatched readout), on two models and two
   authors' grids. It does not cross into generation (see Fell).
7. Steering selects among competences the engine already has: theme steers generation only on a
   9B instruction-tuned model, and tuned models carry refusal as an unlisted factor.
8. Abstraction is a quotient with a measurable scale: re-encoding through a narrower sparse
   dictionary keeps more general features (mean generality 0.18 vs 0.06; 12 of 15 merges go up).
9. Under generation the shallower factor dominates the surface text regardless of patch order
   (voice > era > theme), at two layer pairs.
10. A parameterized translation exists: advancing a described subject by an interval from a day
    to a million years produces a shared clock direction, computed from the state rather than the
    interval phrase (2.8× the phrase-only displacement), that works as a selector (rank 3.9 of 9
    vs 5.7 random). Its subject-relative part, a mayfly's day against a mountain's million years,
    did not appear.

**Fell.** The era shift under generation: on Llama-3.1-70B-Instruct and Gemma-2-9B-it the
shifted continuation stays in its original era (0.14 and 0.27 read as target; not one gains the
target era's vocabulary), and the 70B does it less than the 9B. Subject-relative timescales in the
time-translation grid. Pooled distance structure as a content shape. Absence as decoder-adjacent inactive
features. The five-regime commutator taxonomy under greedy decoding. "Factors don't commute under
generation" as a fact about factors: any two matched-norm patches diverge the same way.

**Partial.** The relation operator as a generative patch: helpful (+0.18 nats vs −0.10 random)
but no better than the same operator fed the wrong source, so source-specificity is undemonstrated.
A continuation-defined test for absence: null at n = 8, sensitive to presence, blind to absence.

## Method

Prompts mark spans with roles; each span's residual vector is pooled (mean, or last token) at
every layer. Every direction is estimated **leave-one-out**: a role or factor direction is the mean
of that role's vectors over training domains minus the grand mean, so no direction sees its test
domain. Three tests recur.

*Selector.* Add a direction at every position during a teacher-forced pass and ask whether the
log-probability of the matching span rises more than the others'. Report the rank of the match
among candidates; chance is the midpoint; the control is a random direction of equal norm.
*Composition.* Sum several directions and rank the joint variant among all variants.
*Cross-talk.* Decompose the gain matrix under one factor's patch into per-factor main effects; the
matrix is the Jacobian of readouts with respect to patches. Nulls are permutation (scramble the
role correspondence) or pairing shuffles within training folds. Best-layer selection on held-out
data is never reported; it inflated a constant baseline from 3.5 to 2.4 once, and was dropped.

Remote models run on NDIF through a credential-injecting proxy (`src/lsx/ndif.py`); local models
run CPU-only.

## The arc: from shape to roles to relation to data

The project began with a shape. The first test, representational similarity over a prompt's six
pooled role vectors, found strong cross-domain agreement (0.60 at layer 20) that shrank to chance
(0.05) when roles were relabeled by content after shuffling spans across template slots. The shape
was position. Projecting out the slot subspace did not recover a content shape (h2–h3).

Under it were two real objects. **Role identity** is a direction that transfers: a constant
"which role is this" prediction ranks the correct role first in a held-out domain (1.07 of 6), and
the role direction as a patch selects that role's span in every one of eight domains (h4). The
**relation** between roles, after removing position by design, role identity by centering, and
domain address by ranking within a prompt, was a small signal at seven domains (3.0 vs 3.5 null,
at chance in the embeddings, peaking at layer 20) and a working selector at forty (2.21; peak layer
16; the embedding layer itself now below chance at 2.73, so part of the relation is lexical and the
stack adds a rank on top). The relation was starved, not absent (h16). Thirty-two of the forty
domains were generated by Gemma-2-9B-it from the schema spec and reviewed; the six that copied the
example's wording were paraphrased, and role-centering removes what shared phrasing would add.

As a patch, the forty-domain operator raises the target span's likelihood beyond the role direction
alone, and a random direction lowers it. But the operator fed the *wrong* source role does as well
(h20). The output direction is useful; the source-specific part, which is what "relation" means,
has not been shown to survive patching. That is the current edge of the original hypothesis, and it
is a precise one.

## The narrative calculus at sentence scale

Era, voice, tense, mood, and theme each transfer to unseen scenes as a lens; three compose; the
cross-talk matrix is diagonal (off-diagonal 6–20%, on-target 47–80%); order of application across
two layers changes the readout by half a rank. Factors differ in kind. Voice and tense are readable
at layer 0 and are lexical. Era is at chance at layer 0 and 94% readable by layer 12: computed.
Mood is at chance over the first third of a sentence and best read at the last token: integrated.
Theme is distributed, read from the mean, at chance in the first 15% of a passage, peaking near
the midpoint, with a single positive-then-negative beat-to-beat derivative shared by all three
themes (h6, h8, h9, h17).

Recomposition is the primitive the calculus needs, and it is demonstrated: an era shift, one
additive patch while the model reads a passage, moves the era readout to the target in 89% (Qwen
1.5B) and 88% (Gemma 9B) of cases while the theme readout stays at exactly its unpatched value,
0.81 and 0.94 (h14). On grids written by GPT rather than by us, the same test gives 0.94 moved and
0.86 kept (h23).

Three models across two families give the same numbers within a few hundredths on every
selector-level quantity (h7). The model with the sharpest selector is not the model that steers:
Gemma's selectors match the 1.5B's, but Gemma alone writes on theme.

## The boundary: gauges compose, engines don't

Every factor is a reliable selector on every model. Generation is different. On a 1.5B base model,
theme directions do nothing visible; multi-layer re-imposition and a repetition penalty do not
change that. On the 1.5B instruct model, homecoming and betrayal become legible and the homecoming
direction, inside the chat template, triggers a canned refusal. On Gemma-2-9B-it the theme
directions write on theme with prose quality intact: *"everything she had built her life upon was
now a lie"*; *"she hadn't expected to hear from him again, not after all these years."* Steering
selects among competences the engine already has (h10–h13).

The commutator experiment sharpened this. Applying two factor patches in the two orders produces
token-different continuations from token ~15 on, but so do two random directions of matched norm,
on every divergence measure. The five-regime taxonomy borrowed from cellular automata shows no
structure at this sample size. What is factor-specific is that factor directions displace the
continuation from the prompt's prior (overlap 0.14 vs 0.32 for random) and that the shallower
factor dominates the text whichever is applied first (h19, h22). In the algebra: near-commutativity
is a law over gauges, non-commutation under the engine is generic, and dominance by depth is the
factor-specific fact.

## Abstraction as quotient

Re-encoding the same Gemma activations through Gemma Scope dictionaries of width 16k and 131k,
the narrow dictionary's surviving features fire on more reference passages (mean generality 0.18
vs 0.06; 71% of the wide dictionary's features rare vs 42%). Matching each wide feature to its
nearest narrow feature by decoder direction, the narrow match is more general in 12 of 15, and the
clearest case is the first rung of the intended chain: a feature anchored on *Enterprise, Picard,
Federation* merges into one anchored on *stars, Federation* (h15). This is deterritorialization
with a scale parameter and without a chosen axis. Its higher rungs are unreadable without feature
labels, and the fixed points of the flow may well be trivially general; the interesting structure
may live in the flow rather than at its attractors.

## What did not work, with reasons

- **Pooled shape.** A six-point distance structure over pooled spans is a position detector.
- **Absence by decoder adjacency.** Inactive features near the live set in decoder space are the
  dictionary's long tail, do not track theme, and are *more* disruptive than matched controls when
  patched (KL 0.063 vs 0.008). Wrong adjacency.
- **Absence by continuation.** Withheld parts leave no last-token or near-threshold trace at n = 8;
  the delivered variant reads at 12 vs 0.6, so the instrument sees presence. Null, not falsified.
- **Commutator regimes.** No level combination agrees across four prompts; permutation p = 0.16
  and 0.57.
- **Two metrics** were found broken and replaced: a cross-talk rank that averaged to its chance
  value by construction, and best-layer selection.

## Limitations

Grids were written by Claude, reviewed by Claude (32 holonic domains), or written by GPT; no
human-written grid yet. Sentence-scale spans for most factors, three-sentence passages for theme.
Most positive results are selector-level; generator-level evidence is qualitative and appears only
at 9B-instruct. Layers are mid-stack by convention, with sweeps for the role lens, the relation
lens, and the commutator only. Generality of features is measured against a narrative-only
reference set. The relation operator's source-specificity under patching is open.

## Next

The gauge/engine boundary is now the central fact, confirmed at 70B: every representational law
holds, and none of them writes text. The next experiments should test whether anything crosses it:
a scale sweep of the era shift at 9B (0.5–3×) with the lexical check as the readout, and
multi-position re-imposition during decoding rather than a single prefix patch. On time
translation: a vocabulary-matched far-interval grid to remove the erasure confound, and the same
grid on Gemma-9B. Still open: a human-written grid; absence defined by the model's own surprise;
feature labels for the abstraction ladder.

## Reproducibility

`README.md` for setup; `scripts/stage*.py`, `scripts/ndif_*.py`, and the agent scripts for every
experiment; `results/*.json` and `results/notes/*.md` for per-case records; `prompts/*.json` for
every grid; `results/figures/` for the derivative curves. `docs/ALGEBRA.md` states the calculus
with each law marked measured, hypothesized, or conjectured.
