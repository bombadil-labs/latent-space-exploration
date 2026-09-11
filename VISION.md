# Narrative Calculus: the operator set, mapped to what the toolkit can do today

The activation-level work so far has covered *decompose* / *compose* (factors as directions, additive
composition, measured interference). The calculus as designed has more operators. This note keeps
them in view and says, for each, what already exists, what it would be at the activation level, and
what would test it. Nothing here is implemented beyond what RESULTS.md records.

## transform — translation, rotation, scale
- **Have:** translation (adding a factor direction: era shift, voice shift) and scale (the working
  range of a patch, roughly 0.5–2× the direction's natural norm).
- **Missing:** rotation. The affine relational operator (stage 3) is the rotation; at seven
  effective examples it underperforms a constant. This is a data problem, not a method problem.
- **Test:** refit on 30+ domains with paraphrases; the thesis→antithesis spin.

## abstract / concretize (coarse-grain / fine-grain)
- `abstract(x)`: project out the setting/domain components, keep the role. Picard − StarTrek =
  captain, diplomat, mentor, prefers win/win. `concretize(x, ctx)`: add ctx's components back.
- **Have:** grand-mean and role-centering are crude abstraction; the held-out-domain protocol is
  concretize; the instruct-model theme results were concretize in generation.
- **Structure:** abstraction levels form a lattice. Sparse-autoencoder features (Gemma Scope, now
  reachable) give the lattice a basis: abstract = drop features below a generality threshold.
- **Test:** abstract a character description on one grid, concretize against another setting,
  score with the role/setting selectors.

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

## Standing constraints learned so far
- Steering selects among competences the model already has. Generative reach is bounded by the
  model, not the geometry (theme on 1.5B base: selector 1.25/3, generation ≈ base).
- Selector-level results come easily; generator-level results are the ones that count for the
  calculus. Keep the distinction explicit in every claim.
- Refusal is an unlisted factor on tuned models; project it out before steering.
