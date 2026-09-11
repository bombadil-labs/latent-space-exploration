# The narrative calculus as an operator algebra

Status: draft. This turns the operator *list* in `VISION.md` into a closed, many-sorted algebra with
fixed signatures, in the style of rhizomatic's SPEC-2, and marks every law with what `RESULTS.md`
actually measured. Lexicon terms are used exactly as `VISION.md` defines them. Nothing here is
implemented beyond what `RESULTS.md` records; **measured / hypothesized / conjectured** is stated
for each operator and each law.

## 1. Sorts

The algebra is many-sorted; every signature is fixed. Eight sorts, closed under the operators of §2.

```
Passage  — a text unit (span, sentence, multi-sentence passage) with its token positions.
           Realization: the residual stream it induces at a given depth.
Dir      — a direction: a vector in the residual stream at one layer. The activation-level
           realization of a Level, of a Role, or of a difference of either.
Schema   — a structure of Roles and relations (the holonic transition; Propp's morphology).
Role     — a slot in a Schema. Realized as a Dir that is domain-independent (hour 4).
Factor   — a named axis (era, voice, tense, mood, theme); a Level is a value on it (medieval,
           terse, betrayal). A (Factor, Level) pair realizes as a Dir.
Op       — a map Dir → Dir: translation (x + v), scale (s·x), affine (Ax + b).
Scale    — the two scale parameters: layer depth ℓ (the model's own generality scale) and
           dictionary width w (SAE feature granularity).
Gauge    — a linear readout: projection of a Passage onto a Dir at a Scale, or a nearest-Dir
           classifier over a set of Dirs. Measurement only; no side effect on the Passage.
Engine   — the model as generator: Passage × patches → Passage. Not linear, not invertible.
```

Closure: `transform`, `abstract`, `concretize`, `differentiate`, `integrate`, `compose`, `select`,
`absential` and `cohere` map these sorts to these sorts. `Engine` is the boundary sort — the
analogue of rhizomatic's `resolve`: the algebra can invoke it, but its output is terminal for the
calculus and no law is stated over it (§4).

## 2. Operators

Signature, the extra information section-type operators require, implementing script, status.

### 2.1 `transform`

```
translate : Dir × Scale → Op              translate(v)(x) = x + v     -- patch a (Factor, Level)
scale     : ℝ × Scale → Op                scale(s)(x)     = s·x       -- working range 0.5–2×
rotate    : Schema × Role × Role → Op     rotate(S,r,r')(x) = A x + b -- the relation operator
```

`translate` and `scale` are **measured**: `scripts/stage4.py`, `scripts/stage5_factors.py`,
`scripts/stage6_factors.py`; hours 4, 6–10 (scale range from hour 4: 4.0 degrades everything).
`rotate` is **measured as a selector, untested as a patch at scale**: fit by `scripts/stage3.py`,
patched by `scripts/stage4_relation.py`. At seven domains it beat a matched null narrowly (role_rank
3.01 vs 3.5, hour 3) and was null as a patch (−0.02 nats vs −0.19 random, hour 5). At forty
domains (hour 16) it reaches role_rank **2.21** vs nulls 3.49/3.53: the operator is real and
fittable given data, and **source-specific as a selector** (hour 24: true source ranks the target 1.73 vs 2.75 for a wrong role of the same prompt; 68% paired wins). As a *patch* (hour 20) its prediction helps (+0.18 nats vs −0.10 random) and, against all five wrong sources at natural norm (hour 25), retains a marginal source-specific margin (54% paired wins, +0.06 nats). Selector-clean, engine-faint (§5).

### 2.2 `abstract` / `concretize` (deterritorialize / reterritorialize)

```
abstract   : Passage × Scale(w) → Passage         -- re-encode through a narrower dictionary
concretize : Passage × Scale(w) × Ctx → Passage   -- decode through a finer one
```

`abstract` is the quotient: many-to-one, discarding the fiber on *every* axis at once, not one named
axis. `concretize` is the section: `Ctx` — which fine features to activate — is the supplied
information, and it is what makes the signature asymmetric. A (Factor, Level) direction is the
one-axis special case of `abstract`. **Measured at its first rung**: `scripts/sae_ladder.py`, hour 15
(Gemma Scope 16k vs 131k; mean generality 0.18 vs 0.06; 12 of 15 nearest-narrow features more
general). `concretize` is **hypothesized**: no script supplies `Ctx` today.

### 2.3 `differentiate` / `integrate`

```
differentiate : Passage × Gauge × Scale → (position → ℝ)   -- beat-to-beat change of a readout
integrate     : (position → ℝ) × ℝ → (position → ℝ)        -- the ℝ is the boundary condition:
                                                              the supplied information
```

`differentiate` coarse-grains (kills the constant); `integrate` fine-grains and must be told the
initial value — the same asymmetry as quotient/section. The **Jacobian** form is **measured**: the
cross-talk matrix is ∂(Gauge of factor j)/∂(patch of factor i), computed by
`scripts/stage5_crosstalk.py` and `scripts/stage6_factors.py` (hours 6, 8). The
curve-along-a-passage form is **measured**: `scripts/derivative_curves.py` (hour 17): theme's own-readout is at chance in the first 15% of a passage, peaks mid-passage (0.69), and its beat-to-beat derivative has one shape (positive then negative) shared by all themes; era's derivative is zero in aggregate.

### 2.4 `compose`

```
compose : Op × Op → Op         compose(f,g)(x) = f(g(x))
sum     : Dir × Dir → Dir      the measured realization: one patch, dir_a + dir_b
```

**Measured**: `scripts/stage5_factors.py` (era+voice 2.03 of 9, chance 5.0; hour 6) and
`scripts/stage6_factors.py` (era+voice+tense 2.81 of 18, chance 9.5, hour 8; era+theme 2.22 of 9,
hour 10).

### 2.5 `select` (the lens)

```
gauge  : Dir × Scale → Gauge                  -- a Dir read as a linear functional
select : Dir × Passage* × Scale → rank        -- rank the matching Passage among candidates
```

**Measured**, and the best-supported operator here: `scripts/stage4.py` (role lens 1.65 of 6 vs 3.39
random, hour 4), `scripts/stage5_factors.py`, `scripts/stage6_factors.py`, `scripts/ndif_factors.py`
(hours 6–13; every factor 1.1–1.3 of 3 against ~2.0 for equal-norm random directions).

### 2.6 `absential`

```
absential : Schema × Passage × Scale → Dir    -- the Dir of a Role the Schema predicts, that no
                                                 span occupies, and that is active anyway
```

**Defined; the decoder-geometry realization is falsified.** `scripts/absential_ring.py` and
`scripts/ndif_absential_probe.py` (hour 18): inactive features adjacent in decoder space to the
live set are the dictionary's long tail, do not track theme, and are *more* disruptive than
matched controls when patched (KL 0.063 vs 0.008; lower-KL in 1/12). The continuation-defined
realization (`scripts/ndif_absential_continuation.py`, hour 21) is null at n = 8 on the
confound-free readouts. The signature stands; no realization has yet shown an absence doing work.

### 2.7 `cohere`

```
cohere : Passage × Dir* × Scale → Passage     -- constrained re-projection onto the coherent-story
                                                 region with the given Dirs held fixed
```

**Not yet realizable.** At selector level it is a re-projection with no observable; it needs a
steerable Engine at passage scale, which hours 10–13 show does not exist below 9B-instruct.

## 3. Laws

| # | Law | Status | Supporting number |
|---|---|---|---|
| L1 | **Additivity of directions.** `select(dir_a + dir_b + dir_c)` picks the joint variant. | measured | hour 8: 2.81/18, chance 9.5; hour 6: 2.03/9, chance 5.0 |
| L2 | **Near-commutativity, as gauges.** `translate(a) ∘ translate(b) ≈ translate(b) ∘ translate(a)` under a Gauge; the deviation from context-freeness is the cross-talk Jacobian. Under the Engine *any* two matched-norm patches diverge from token ~15 on (random pairs identically: hour 22), so engine-level non-commutation is generic, not a property of factors; what is factor-specific is that the shallower factor dominates the surface text regardless of order (voice > era > theme, two layer pairs for era > theme). | measured (gauge law; engine caveat; dominance) | hours 6/8: order gap 0.50 rank, gain correlation 0.85; off-diagonal 0.06–0.20 vs diagonal 0.47–0.76. Hour 19: `scripts/ndif_commutator.py` |
| L3 | **Gauge linearity vs Engine nonlinearity.** Gauges compose; Engines do not. A selector result does not transfer to generation; neither does gauge-level commutativity (L2, hour 19). | measured | hours 10–13: theme selector 1.25/3 on every model, generation ≈ base at 1.5B, legible only at 9B-instruct; decodability 0.78 → 0.83 → 0.94 with scale. (groovy-commutator's run calculus states the same split: gauges XOR linearly, engines do not.) |
| L4 | **Address/form separation.** An era shift moves the address and keeps the form. | measured | hour 14: era reads as the target in 0.89 (Qwen) / 0.88 (Gemma) of cases; theme readout 0.81 / 0.94, identical to unpatched |
| L5 | **Quotient monotonicity of generality.** A narrower dictionary keeps more general features. | measured | hour 15: mean generality 0.18 (16k) vs 0.06 (131k); 12/15 nearest-narrow features more general |
| L6 | **Adjunction `abstract ⊣ concretize`.** concretize-then-abstract returns the coarse state; abstract-then-concretize returns a representative of the discarded fiber. | conjectured | none: `concretize` has no implementation |
| L7 | **Renormalization-group structure.** Coarse-grainings compose, have fixed points (archetypes as attractors), and sort features into relevant and irrelevant. | conjectured | `VISION.md` §coarse-graining; the abstraction-flow test is unrun |
| L8 | **Depth is the scale parameter.** Each Factor becomes readable at a characteristic layer; the residual is a sum over layers. | measured | hours 6, 8: tense 1.00 at layer 0 and decaying; voice 0.92 flat; era 0.31 → 0.92 by layer 12; theme distributed, peaking at layer 20 |
| L9 | **Relation composition along a selector path.** Composing two `rotate` Ops equals the Op learned from endpoint pairs. | hypothesized | the test stated in `VISION.md`; `rotate` is now fittable at 40 domains (hour 16), so the test is runnable |

**Boundary, not a law.** *Steering selects among competences the Engine already has.* This is a
**constraint** on the algebra's reach, measured in hours 10–12 (theme on 1.5B base: selector 1.25/3,
generation ≈ base; neither multi-layer re-imposition nor a repetition penalty moves it). It bounds
what any composition of operators can produce through an Engine; it is not itself an identity, and
no term rewrites under it.

A second constraint: on tuned models **refusal is an unlisted Factor** (hour 12 — the homecoming
direction produced a canned refusal) and must be projected out before steering. The cross-talk
matrix cannot see it, because it is not on the grid.

## 4. Deliberately excluded

The exclusions are the design.

- **Arbitrary text edits.** Every operator acts on Dirs, Passages-as-residuals, or Scales. "Rewrite
  this sentence" is not a term in the algebra; it is Engine work, outside the closed fragment.
- **Anything requiring the Engine to be linear.** Gauges compose (L3); Engines do not. No law may be
  stated over Engine outputs that was only verified over Gauges.
- **Feature labels.** Generality is measured label-free (fraction of reference passages a feature
  fires on, hour 15). No operator takes a human-readable feature name as an argument.
- **Claims about generation beyond what selectors show.** Generations in `RESULTS.md` are marked
  illustration; they never enter a law.

## 5. Open signatures

- **`rotate`** was under-determined at seven domains (it lost to a constant, hour 3, and was null
  as a patch, hour 5). At forty domains (hour 16) it is determined as a selector (2.21 vs 3.5
  null) and source-specific as a selector (hour 24). As a patch (hour 20) it is helpful but its
  source-specificity shows only marginally through the likelihood readout (hour 25). Still open:
  the composition law L9 and the spin test.
- **`cohere`** has no observable until generation is steerable at passage scale. Its signature is
  written; its Gauge does not exist.
- **`absential`** is defined (§2.6); its first realization (decoder adjacency) is falsified (hour 18); the
  continuation-defined realization is the open test.

## 6. Mapping table

| VISION.md operator | signature | script | RESULTS.md stage | status |
|---|---|---|---|---|
| transform: translate | `Dir × Scale → Op` | `stage4.py`, `stage5_factors.py`, `stage6_factors.py` | hours 4, 6–10 | measured |
| transform: scale | `ℝ × Scale → Op` | `stage4.py`, `stage5_generate.py` | hour 4 (range 0.5–2×) | measured |
| transform: rotate | `Schema × Role × Role → Op` | `stage3.py`, `stage4_relation.py` | hours 3, 5, 16 | measured as selector (2.21 vs 3.5 at 40 domains); patch untested at scale |
| abstract | `Passage × Scale(w) → Passage` | `sae_ladder.py` | hour 15 | measured (first rung) |
| concretize | `Passage × Scale(w) × Ctx → Passage` | — | — | hypothesized |
| differentiate (Jacobian) | `Passage × Gauge × Scale → …` | `stage5_crosstalk.py`, `stage6_factors.py` | hours 6, 8 | measured |
| differentiate (curve) | `Passage × Gauge × Scale → (position → ℝ)` | `derivative_curves.py` | hour 17 | measured |
| integrate | `(position → ℝ) × ℝ → (position → ℝ)` | — | — | hypothesized (running mean is the trivial case) |
| compose | `Op × Op → Op`; `Dir × Dir → Dir` | `stage5_factors.py`, `stage6_factors.py` | hours 6, 8, 10 | measured |
| select (the lens) | `Dir × Passage* × Scale → rank` | `stage4.py`, `stage5_factors.py`, `stage6_factors.py`, `ndif_factors.py` | hours 4, 6–13 | measured |
| recompose (address shift) | `translate(dir_e2 − dir_e1)` | `stage7_shift.py`, `ndif_shift.py` | hour 14 | measured |
| absential | `Schema × Passage × Scale → Dir` | `absential_ring.py`, `ndif_absential_probe.py` | hour 18 | defined; decoder-geometry realization falsified |
| cohere | `Passage × Dir* × Scale → Passage` | — | — | not yet realizable |
