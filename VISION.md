# Narrative Calculus: the operator set, mapped to what the toolkit can do today

The activation-level work so far has covered *decompose* / *compose* (factors as directions, additive
composition, measured interference). The calculus as designed has more operators. This note keeps
them in view and says, for each, what already exists, what it would be at the activation level, and
what would test it. Nothing here is implemented beyond what RESULTS.md records.

## Working lexicon

- **Schema:** a structure of roles and relations (the holonic transition; Propp's morphology; Star
  Trek as a structure of character/setting/plot fields). What "function" meant in the first draft.
- **Instance / entity:** a concrete filler (Picard; France). Nodes of the graph.
- **Role:** a slot in a schema (the disturbance, the view from above). Propp's sense of "function".
- **Factor / level:** a named axis (era, voice, mood, theme) and a value on it (medieval, terse).
- **Direction:** the activation-level realization of a level or a role. Transfers, composes,
  interferes (6–20% off-diagonal).
- **Relation:** an edge between entities: has-part, instance-of, attribute-of, located-in, at-time.
  A relation is a function in the mapping sense: it takes an entity and returns another.
  Activation-level candidate: an affine operator on directions (stage 3; the missing "rotation").
- **Selector:** a path of relations, e.g. `startrek.characters.picard.heritage.geographic-origin.
  climate.1960`. Syntactically field access; semantically function composition, where the names
  are arguments and the dots are the functions. Paths can leave the fiction (at `heritage`) and
  keep walking the world, so there are **no leaves**: a walk can always take another edge.
- **Decomposition:** the special case of a selector where every edge is has-part.
- **Address / form:** where a story sits vs. what it is shaped like. Discourse position is a large
  part of address.
- **Operators of the calculus:** transform (translate, rotate, scale), abstract/concretize
  (deterritorialize/reterritorialize), differentiate/integrate, absential, cohere.

**Test that follows from the lexicon:** learn two relations as operators, compose them, and compare
with the operator learned directly from endpoint pairs. Match ⇒ the model represents the path
linearly; mismatch ⇒ real computation between hops. This is the relation lens generalized to paths.

## transform — translation, rotation, scale
- **Have:** translation (adding a factor direction: era shift, voice shift) and scale (the working
  range of a patch, roughly 0.5–2× the direction's natural norm).
- **Missing:** rotation. The affine relational operator (stage 3) is the rotation; at seven
  effective examples it underperforms a constant. This is a data problem, not a method problem.
- **Test:** refit on 30+ domains with paraphrases; the thesis→antithesis spin.

## abstract / concretize (deterritorialize / reterritorialize)
- `abstract(x)` moves *every* detail of x up one rung of generality at once; it is not the removal
  of one axis such as setting. Picard → Starfleet captain with particulars → spaceship diplomat →
  traveling leader → mentor. `concretize(x, ctx)` moves back down, and must be told which
  particulars to choose; ctx supplies them.
- **Candidate precise mechanism:** sparse-autoencoder *feature splitting*. A narrow dictionary
  learns coarse features; wider dictionaries split them into finer ones. Gemma Scope ships several
  widths per layer for Gemma-2-9B. abstract at level k = re-encode the activation through the
  dictionary of width w_k and reconstruct (narrower = more general, on all axes). concretize =
  decode through a finer dictionary, choosing which fine features to activate (the section; the
  supplied context is that choice).
- **Have:** the factor directions are one-axis quotients (partial coarse-grainings); grand-mean and
  role-centering likewise. The all-axis version is the dictionary-width ladder above.
- **Test:** encode a character description through successively narrower dictionaries and read the
  surviving features at each width; the prediction is a diplomat → leader → mentor ordering.
  Then concretize against a new setting and score with the selectors.

## derivative / integral
- Slice a passage by sentence or beat; project each slice onto a factor direction; the sequence
  is a curve (mood over the story; Vonnegut's shapes as measurement). d/dposition = beat-to-beat
  change. ∫ = what a passage accumulates.
- **Already computed without the name:** the cross-talk matrix is the Jacobian of factor readouts
  with respect to factor patches (∂theme/∂era). Report it as such.
- **Test:** cheap; all pieces exist. Curves over the theme passages, per factor, per layer.

## absential (Deacon)
- An absence that does causal work: a role the schema predicts but no span occupies, whose
  direction is nonetheless active in the residual and predicts the continuation.
- **Test:** passages with a structurally implied but absent part (a betrayal set up and withheld).
  Run the role detectors; check whether the missing role's direction is present in anticipation
  and whether it shifts next-token behavior. If yes, the absence is represented and causal.

## cohere
- Constrained projection back onto the coherent-story region with the requested directions held
  fixed. Deferred until generation is steerable at passage scale; at selector level it is a
  re-projection and has no observable.

## A candidate common mechanism: coarse-graining as quotient, fine-graining as section

**Pairing (corrected).** Coarse-grain = {abstract, differentiate}: both throw information away to
focus on a subset. Differentiation kills the constant; abstraction kills particulars at every axis (Picard →
mentor), not one axis. Fine-grain = {concretize, integrate}: both require the invocation to *supply* information
to move from a lossy state to a more detailed one. Integration needs a boundary condition;
concretization needs a context. (An earlier draft of this note paired abstract with integrate by
reading "integrate" as marginalize-over-an-axis; that is a different operation from the
antiderivative and the pairing above is the right one.)

**Mechanism.** Coarse-graining is a quotient: a many-to-one map that discards a fiber. Fine-graining
is a section: one-to-many, requiring a choice of representative, which is exactly where the supplied
information enters. The factor directions the toolkit builds (mean over scenes and other factors)
are one-axis quotients; dictionary width (feature splitting) is the all-axis quotient, and the RG
framing lives more naturally on it. The beat-to-beat derivative of a factor readout
along a passage loses the absolute level the same way a derivative loses its constant; integrating
it back needs the initial value supplied.

**If that holds, the structure is a renormalization group** (a semigroup of quotients): coarse-
grainings compose, have fixed points (universality classes → archetypes: the hero's journey as an
attractor of the abstraction flow, not a template), and sort features into relevant (survive:
theme, role) and irrelevant (die: era vocabulary, tense). Depth is the model's own scale: tense
readable at layer 0 and fading, era emerging at layer 12, theme distributed and late; the residual
stream is a sum over layers. Deacon's absentials are constraints, which is what quotients preserve.

**Test:** define an abstraction flow (level k keeps only directions/SAE features shared across ≥k
domains or above a generality threshold); track pairwise story distances as k grows; look for
collapse onto a few attractors and check whether they read as archetypes. Gemma Scope features make
this cheap. Status: quotient/section is near-definitional. The RG claim was tested (RESULTS h26): fixed points are trivially general, no archetype among them; the flow is an ordering of what survives (theme outlasts era), not an optimum. Archetypes, if they are anywhere, are not attractors of this flow.

## Standing constraints learned so far
- Steering selects among competences the model already has. Generative reach is bounded by the
  model, not the geometry (theme on 1.5B base: selector 1.25/3, generation ≈ base).
- Selector-level results come easily; generator-level results are the ones that count for the
  calculus. Keep the distinction explicit in every claim.
- Refusal is an unlisted factor on tuned models; project it out before steering.

## Ecology notes (from sibling projects; inspiration, not integration)

**groovy-commutator.** The cross-rule commutator and its divergence *trajectory* (five regimes:
commute / crystalline / noise / structured / drain) is the instrument our one-step order test
approximates. Tee-up: apply two factor patches in both orders, generate, measure per-token
divergence of selector readouts, classify factor pairs by regime. "Dynamics of erased
distinctions" (latent vs shielded information under a representation, H(S) = H(P(S)) + I_latent
+ I_shielded) is the exact frame for stage 2 (content was latent to pooled RSA, returned via the
operator) and for the selector/generator gap (theme is shielded under small-model generation,
visible at 9B/instruct). RG fixed points there are the affine/absorbing rules and Class IV lives
*between* them: so the attractors of the abstraction flow may be trivially general, and archetypes,
if interesting, live in the flow. The absential field as closed neighborhood of the live set gives
the recipe: the absential ring of a passage = inactive SAE features with high decoder similarity to
its active ones; test causal work by ablation. The run calculus (gauge riding a base; gauges
compose linearly, engines do not) names stage 7's design and the selector/generator split.

**shadow-walker.** Its `semanticShift` report (salient, receded, invariant, surprise) and weave
result kinds (correspondence / tension / mismatch / partial overlap / convergence / none) are the
vocabulary for what lsx measures; the README reserves "mechanistic measurement" as a future
external projection with its own provenance, which is what an lsx→Observation adapter would be.
`record_operation`'s required fields (origin domain, input/output structure, invariants) are a
stricter form of the operator lexicon above.

**rhizomatic.** Closed, many-sorted, serializable operator algebra with fixed signatures (terms,
not code); context-freeness of atoms; merge = union; provenance in the atom. Template for turning
the operator list above into a calculus: name the sorts, fix signatures, keep it closed; treat
cross-talk as the measured deviation from context-freeness of directions.
