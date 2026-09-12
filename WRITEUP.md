# Narrative factors are directions: measuring, moving, and composing story structure in transformer activations

*Fourth draft, after 38 logged stages. Every number below is in `RESULTS.md` with its control and
file reference; Checkpoint 2 at the top of that file lists what was withdrawn and when. The calculus
these results support is in `docs/ALGEBRA.md`. The four measurement instruments this project found
broken, and what they invalidated, are in `docs/INSTRUMENTS.md` — read that first if you are
assessing whether the positive results below can be trusted.*

## Summary

We asked whether the structure a prompt describes — a relation among parts, or a narrative property
such as setting, register, mood, or theme — exists in a language model's residual stream as a
geometric object that can be measured, carried to a new domain, composed with others, and used to
steer generation. Across nine models in five families (Qwen2.5 0.5B/1.5B/1.5B-Instruct,
Pythia-1.4B, GPT-J-6B, Gemma-2-9B-it, Llama-3.1-8B/70B/70B-Instruct), with every claim paired to a
matched null, eleven things stand, six fell, and two are partial.

**Stand.**

1. Discourse position dominates pooled activations. The naive "shape" of a prompt is where each
   part sits in its template, not what it says.
2. Roles are domain-independent directions and causally usable: adding a role's direction in a
   domain it never saw selects that role's span (rank 1.7 of 6; random 3.3; chance 3.5).
3. The source→target relation between roles is real, computed by the stack, and, given forty
   domains, a working selector (rank 2.21 of 6 vs 3.5 null; peak at layer 16 of 28).
4. Narrative factors (era, voice, tense, mood, theme) are additive directions: each is a lens on
   unseen scenes (1.1–1.4 of 3 vs ~2.0 random), three compose in one patch (2.8 of 18, chance 9.5),
   and their cross-talk matrix is diagonal. Replicated on four model families and on grids written
   by a second model author.
5. Factors differ in kind, and the model's depth shows it: voice and tense are lexical, era is
   computed by layer 12, mood is integrated at the last token, theme is distributed and peaks
   mid-passage.
6. Recomposition works as a readout: an era shift moves a passage's address (0.89–0.94 of cases)
   and keeps its theme (0.81–0.94, equal to the unpatched readout), on two models and two authors'
   grids.
7. Abstraction is a quotient with a measurable scale: re-encoding through a narrower sparse
   dictionary keeps more general features (mean generality 0.18 vs 0.06; 12 of 15 merges go up).
8. Under generation the shallower factor dominates the surface text regardless of patch order
   (voice > era > theme), at two layer pairs.
9. **Gauges compose; engines do not — and the boundary between them is a magnitude.** At matched
   norm an era shift moves the readout but never the generated text (0.14 on Llama-70B-Instruct,
   0.27 on Gemma-9B, and on neither does a single continuation gain target-era vocabulary). At
   three times the norm, re-imposed at every decoding step, Gemma's continuations read as the
   target era in 0.84 of cases and 0.30 use its vocabulary, with prose intact.
10. **The magnitude is not a constant of the method.** The same treatment on Llama-3.1-70B-Instruct
    reaches only 0.43, below Gemma's two-times value. A bigger engine is not a more steerable one.
11. **Instruction tuning sharpens some factors and not others.** On the matched pair Llama-3.1-70B
    and 70B-Instruct — identical pretraining, size and tokenizer — the era selector goes 1.50 to
    1.06 at two different depths, while theme is tuning-invariant (1.06–1.39 either way). That same
    70B-Instruct has the sharpest era selector we have measured and the weak engine of claim 10:
    sharp gauge, weak engine, at fixed size. This is what "steering selects among competences the
    engine already has" reduces to once scale and tuning are separated, and it is the reason the
    earlier version of that claim, which credited scale, has been rewritten.

**Fell.**

- **Pooled distance structure as a content shape.** A six-point distance structure over pooled
  spans is a position detector.
- **Parameterized time translation.** Five stages and roughly a million agent tokens. A shared
  direction does track the interval between a described state and the same subject after Δt, and
  replicates across models. Against a lexical floor of 0.728 the model recovers real gain (0.297 at
  layer 24, permutation z 7.6), so it is not only vocabulary. But the gain survives shuffling the
  words of the passage (structural residue 0.076, label-swap z 1.21) and does not transfer to the
  case where the model must supply the change itself (ρ 0.150, cosine 0.078). It is a **computed
  register detector** — how much change a passage describes, read order-invariantly, more
  accurately than the embeddings alone allow — not a representation of elapsed time.
- **Absence as decoder-adjacent inactive features.**
- **The five-regime commutator taxonomy** under greedy decoding.
- **"Factors don't commute under generation" as a fact about factors:** any two matched-norm
  patches diverge the same way.
- **Steering as a scale effect** (superseded by claim 11, not refuted: it was never tested against
  tuning until stage 37).

**Partial.** The relation operator as a generative patch: helpful (+0.18 nats vs −0.10 random) but
no better than the same operator fed the wrong source, so source-specificity is undemonstrated. A
continuation-defined test for absence: null at n = 8, sensitive to presence, blind to absence.

**Withdrawn, with the instrument that caused each.** Three "subject-relative timescales are absent"
results across two models (stages 28, 30, 31): the probe was a high-dimensional residual norm
sitting at its own noise floor. One Llama selector battery (stage 34): the patch wrote into batch
row zero of a padded tensor, pinning every rank at 1.22 for any direction. A set of discrimination
numbers (stages 28–35): with the interval phrase removed so the model cannot copy the answer, 0.961
falls to 0.522. Stage 31's numbers, though not its conclusion (stage 39): a batched extractor read
363 of 480 passages' spans out of left-padding, with the padding correlated with the variable under
study; re-extracted cleanly the result is stronger, not weaker. See `docs/INSTRUMENTS.md`.

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

Recomposition is the primitive the calculus needs, and as a readout it is demonstrated: an era
shift, one additive patch while the model reads a passage, moves the era readout to the target in
89% (Qwen 1.5B) and 88% (Gemma 9B) of cases while the theme readout stays at exactly its unpatched
value, 0.81 and 0.94 (h14). On grids written by GPT rather than by us, the same test gives 0.94
moved and 0.86 kept (h23). Whether it survives into generated text is claim 9's question, and the
answer is: only at three times the norm, only on some engines (h27, h29, h33).

Three models across two families give the same numbers within a few hundredths on every
selector-level quantity (h7). The model with the sharpest selector is not the model that steers:
Gemma's selectors match the 1.5B's, but Gemma alone writes on theme.

## The boundary: gauges compose, engines don't

Every factor is a reliable selector on every model. Generation is different. On a 1.5B base model,
theme directions do nothing visible; multi-layer re-imposition and a repetition penalty do not
change that. On the 1.5B instruct model, homecoming and betrayal become legible and the homecoming
direction, inside the chat template, triggers a canned refusal. On Gemma-2-9B-it the theme
directions write on theme with prose quality intact: *"everything she had built her life upon was
now a lie"*; *"she hadn't expected to hear from him again, not after all these years."* (h10–h13.)

Two later results decide what that pattern means. Raising the patch to three times its norm and
re-imposing it at every decoding step carries an era shift into Gemma's generated text (0.84 read
as target, 0.30 gaining target-era vocabulary), so the barrier is a magnitude rather than a kind
(h29). But the same treatment on Llama-3.1-70B-Instruct stops at 0.43, below Gemma's two-times
value, so the magnitude is a property of the engine, not of the method (h33). And on the matched
pair Llama-70B and 70B-Instruct the era selector sharpens under tuning, 1.50 to 1.06 at two
depths, while theme does not move (h37). Size is not the variable it looked like: the model with
the sharpest gauge we have measured is also the one whose engine resists it most.

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
- **Parameterized time translation** closed at stage 38 as a register detector rather than a
  clock: real gain over a lexical floor (0.297, z 7.6), but order-invariant under word-shuffling
  and non-transferring to model-supplied change.
- **Five instruments** were found broken and replaced. They are documented in full in
  `docs/INSTRUMENTS.md`, because what they have in common matters more than any one of them: each
  produced a *plausible* number rather than an obvious error — 1.22 was simultaneously an artifact
  and the true era rank on the 8B — and they were caught five different ways: two by a control or
  baseline, one by a planner doing the arithmetic before any data was collected, one by rereading a
  metric's definition, and the fifth by an audit that went looking for a bug it did not find and
  found a different one with the same blast radius (a batched extractor reading 76% of its spans out
  of left-padding). Four of the five were found in the last two days, which says more about how long
  the first three sat undetected than about the rate of error.

## Limitations

Grids were written by Claude, reviewed by Claude (32 holonic domains), or written by GPT; **no
human-written grid yet**, which is the first thing a reviewer should press on. Sentence-scale spans
for most factors, three-sentence passages for theme. Most positive results are selector-level;
generation-level evidence is quantitative only for era, and only on two models. Layers are mid-stack
by convention, with sweeps for the role lens, the relation lens, the commutator, and the 70B
selector. Generality of features is measured against a narrative-only reference set. The relation
operator's source-specificity under patching is open. One depth control is missing: stage 33's cap
at 0.43 was to be explained by a layer sweep, and the selector-level sweep that was run went down
with the stage-34 retraction, so a generation-level sweep is still owed. Stage 8's `tense` random
control reads 1.44, closer to its treatment than it should be, and has not been re-checked.
Llama-3.1-405B is not reachable with this key, so no base model above 70B has been tested.

## Next

The gauge/engine boundary is the central fact, and it is now quantified rather than asserted: the
readout-level laws hold everywhere, generation needs roughly three times the norm re-imposed
throughout decoding, and even that fails on the larger engine. The next experiment is the
generation arms of `docs/specs/scale_vs_tuning_v1.md` on the matched 70B pair, with the
generation-level layer sweep that stage 33 needs, which decides whether claim 11's tuning effect
appears in text as well as in readouts.

Before more experiments, though, the binding constraint is authorship: every grid in this repo was
written by a language model. A human-written grid is the cheapest single change to what these
results are worth. After that: absence defined by the model's own surprise rather than by an
author, and feature labels for the abstraction ladder.

## Reproducibility

`README.md` for setup; `scripts/stage*.py`, `scripts/ndif_*.py`, and the agent scripts for every
experiment; `results/*.json` and `results/notes/*.md` for per-case records; `prompts/*.json` for
every grid; `results/figures/` for the derivative curves. `docs/ALGEBRA.md` states the calculus
with each law marked measured, hypothesized, or conjectured.
