# The absential ring: census and one causal probe

Operationalizes the `absential` operator of VISION.md along the recipe the groovy-commutator ecology
note supplies: *the absential field is the closed neighborhood of the live set* — off, but adjacent
to on. Scripts: `scripts/absential_ring.py` (local), `scripts/ndif_absential_probe.py` (NDIF).
Data: `results/absential_census.json`, `results/absential_probe_gemma9b.json`.

## Definitions

Model Gemma-2-9B-it, block-20 residuals per token (`results/tokens_gemma9b_l20.npz`, via NDIF).
Dictionary: Gemma Scope JumpReLU, layer 20, **width 16k, average_l0_47**.
Encoding `pre = x @ W_enc + b_enc; act = pre * (pre > threshold)`; BOS dropped.

- **active** `A(p)` = features firing on at least one token of passage `p`.
- **generality** `g(f)` = fraction of the 72 reference passages (theme grid + mood grid) on which
  `f` fires at least once.
- **formatting** = `g(f) > 0.9`. 333 of 16384 features (2.0%). These are excluded from the ring and
  from the neighbor source; `A_content = A \ formatting`.
- **absential ring** `R(p)` = `{f ∉ A(p), f ∉ formatting : max_{a ∈ A_content(p)} cos(W_dec[f], W_dec[a]) ≥ τ}`.

**Choice of τ.** The distribution of `max-cos(inactive → active)`, pooled over six passages:

| pct | 50 | 75 | 90 | 95 | 97.5 | 99 | 99.5 | 99.9 |
|---|---|---|---|---|---|---|---|---|
| max-cos | 0.183 | 0.238 | 0.305 | 0.355 | **0.402** | 0.467 | 0.517 | 0.647 |

There is no mode or shoulder to cut at — the distribution is smooth, so τ is a quantile choice, not
a discovered boundary. **τ = 0.40**, the 97.5th percentile: it keeps 2.56% of inactive features,
giving a ring roughly a quarter the size of the active set. This is the first honest caveat: the
ring is a threshold on a continuum, not a natural kind.

## 1. Census (72 passages, `results/absential_census.json`)

| grid | \|A\| | of which formatting | \|A_content\| | \|R\| | \|R\| under null* |
|---|---|---|---|---|---|
| theme (36) | 1809 | 332 | 1477 | **386** | 442 |
| mood (36) | 1144 | 331 | 812 | **258** | 280 |

\* null = the ring built from a size-matched set of *randomly chosen* non-formatting features that
fire somewhere in the reference set, instead of the passage's own active set.

Generality profile (theme passages, means over the 36):

| set | mean g | median g | frac g<0.1 | frac g>0.5 | frac g=0 |
|---|---|---|---|---|---|
| active (content) | 0.305 | 0.263 | 0.19 | 0.18 | — |
| **ring** | **0.093** | 0.029 | 0.70 | 0.03 | **0.38** |
| ring under null | 0.110 | — | 0.70 | — | 0.38 |

**The ring is made of rare features.** Mean generality drops 3.3× from active to ring; 38% of ring
features fire on *none* of the 72 reference passages. Adjacency to the live set therefore does pick
out the dictionary's long tail: the neighborhood of what a narrative passage activates is populated
by features that narrative passages essentially never turn on. That is a pleasing picture of "the
implied but absent."

**But the null kills the interpretation.** A ring built around a *random* set of the same size has
the same size (442 vs 386) and the same generality profile (0.110 vs 0.093, identical rarity and
never-fires fractions). Because the active set covers ~9–11% of the dictionary, its closed
neighborhood is largely a fact about decoder geometry and set size, not about this passage. Nothing
in the *census* statistics distinguishes a real absential ring from an arbitrary one.

**Composition does not track theme.** Mean pairwise Jaccard between ring sets, within vs across a
factor level (36 theme passages):

| axis | ring within | ring across | Δ | active within | active across | Δ |
|---|---|---|---|---|---|---|
| theme | 0.346 | 0.344 | **+0.002** | 0.346 | 0.334 | +0.012 |
| era | 0.359 | 0.337 | +0.022 | 0.355 | 0.330 | +0.025 |
| scene | 0.365 | 0.338 | +0.027 | 0.366 | 0.330 | +0.036 |

Rings cluster very slightly by scene and era and not at all by theme — and in every case less than
the active sets do, i.e. the ring inherits a diluted version of whatever structure the active set
has and adds none of its own. The count of features in the ring of *every* passage of one theme and
*no* passage of the other two is **0** for all three themes. There is no "betrayal-shaped absence"
at this τ, width and layer.

### Verbatim examples: what the top-cosine ring features are adjacent to

For each top ring feature, its nearest active feature and the tokens that active feature fires on
(feature labels are unavailable — Neuronpedia is unreachable here — so this is the only handle):

```
### debt/medieval/betrayal
  ring f10660 cos 0.72 g 0.000  <- active f15583 (g 0.47) fires on [' the', ' The', ' did', ' This']
  ring f13111 cos 0.70 g 0.000  <- active f 9302 (g 0.62) fires on [' his', ' their', ' his']
  ring f12740 cos 0.68 g 0.139  <- active f11944 (g 0.11) fires on [' tithe']
  ring f13571 cos 0.67 g 0.000  <- active f16360 (g 0.40) fires on [' of']

### door/1920s/homecoming
  ring f 1040 cos 0.89 g 0.000  <- active f 7266 (g 0.12) fires on [' three', ' and']
  ring f15671 cos 0.82 g 0.069  <- active f 7090 (g 0.21) fires on [' On']
  ring f15662 cos 0.75 g 0.000  <- active f 5091 (g 0.01) fires on [' nineteen']
  ring f10660 cos 0.72 g 0.000  <- active f15583 (g 0.47) fires on [' did']
```

Selecting the ring by *max* cosine selects near-duplicates of high-firing syntactic features
(determiners, possessives, the numeral-word feature), not thematic near-misses. `f10660` appears in
both passages, adjacent to the same function-word feature. One content-ish case survives (`tithe`).
This is the second honest caveat and it matters for reading the probe below.

## 2. Causal probe (12 passages, Gemma-2-9B-it via NDIF, 72 jobs)

`scripts/ndif_absential_probe.py`. 12 theme passages, balanced over the 4 scenes, 3 themes and 3
eras. For each passage, three patch vectors, each the **sum of 8 decoder directions, renormalized
to 0.15 × the passage's mean block-20 residual norm** and added at block 20 on every position:

| condition | 8 features |
|---|---|
| **ring** | top-8 of `R(p)` by max cosine to the active set |
| **ctrl** | inactive, non-ring (max-cos < τ), matched one-for-one on generality |
| **ctrl2** | inactive, non-ring, generality-matched **and** coherence-matched (see below) |

`ctrl2` exists because the eight ring directions are mutually similar (they cluster near the same
active features), so their sum does not cancel the way eight independent directions do; `ctrl2` is
a mutually-similar cluster of non-adjacent features, so that "the patch is a real direction rather
than noise" is held fixed. Matching achieved:

| | ring | ctrl | ctrl2 |
|---|---|---|---|
| mean generality of the 8 | 0.124 | 0.125 | 0.127 |
| mean max-cos to active set | **0.715** | 0.198 | 0.185 |
| mean within-set coherence | 0.053 | 0.0002 | 0.073 |

So the only thing that differs between ring and ctrl2 is adjacency to the live set.

### Results

| measure | ring | ctrl | ctrl2 |
|---|---|---|---|
| Δ log-prob of the passage's own span, nats/token | −0.027 | −0.037 | −0.037 |
| **KL(patched ‖ base) at the final position** | **0.0628** | 0.0082 | 0.0218 |
| median KL | 0.0312 | 0.0081 | 0.0072 |
| top-5 overlap with base (of 5) | 4.50 | 4.83 | 4.50 |
| passages where ring wins on log-prob | — | 6 / 12 | 6 / 12 |
| passages where ring has *lower* KL | — | **1 / 12** (sign p = 0.006) | 3 / 12 (p = 0.15) |

**The stated hypothesis is not supported, and the opposite effect is significant.**

- *"Does the ring fit the passage better than random?"* — No. Log-prob change is a wash: ring
  −0.027 vs −0.037 and −0.037 nats/token, ring better in exactly 6 of 12 passages against each
  control (sign test p = 1.0). At this patch norm none of the three conditions meaningfully changes
  how well the model predicts the passage's own text.
- *"Is perturbing along the ring less disruptive?"* — No, it is **more** disruptive. The ring patch
  moves the next-token distribution 7.6× further than the generality-matched control in the mean
  (3.8× in the median) and 2.9× further than the coherence-matched control (4.3× in median), and it
  is lower-KL in only 1 of 12 passages against `ctrl`, p = 0.006 by sign test. Against the stricter
  `ctrl2` the direction is the same but the evidence is weak (3/12, p = 0.15, n = 12).

The defensible positive claim is narrow: **at matched norm and matched generality, directions
adjacent to a passage's live set have more causal leverage on the model's next-token distribution
than non-adjacent ones.** The ring is causally privileged — but as amplification, not as fit. The
most likely mechanism is the deflationary one: a feature at cosine 0.7 to a currently *active*
feature is close to a direction the residual already carries a large component of, so adding it
reads downstream like turning a live feature up. Given the verbatim examples above (the top-cosine
ring members are near-duplicates of high-firing function-word features), that reading is more
parsimonious than "the model's implied-but-absent content is causally live."

### Qualitative generations (40 tokens greedy, 6 of the 12 passages)

At 0.15 × residual norm all three conditions stay fluent and on-topic; the visible effect is which
question the instruct model poses about the passage, not degradation.

```
### debt/medieval/betrayal
 [base]  "**What is the story likely about?** ... * **Betrayal and family conflict:** The miller's brother, who swore loyalty at their father'"
 [ring]  "**Questions:** 1. **What is the main conflict in this passage?** 2. **What are the possible motivations for the miller's brother's actions?**"
 [ctrl]  "**Questions:** 1. **What is the main conflict in this passage?** 2. **What are the possible motivations behind the miller's brother's actions?**"
 [ctrl2] "**Questions:** 1. **What is the main conflict in this passage?** 2. **What are the possible motivations of the miller and his brother?**"

### message/medieval/homecoming
 [base]  "**What kind of story is this?** This passage suggests a story of **family reunion and possibly romance**."
 [ring]  "**What is the tone of this passage?** The tone of the passage is **joyful and hopeful**."
 [ctrl]  "**What is the tone of this passage?** The tone of the passage is **joyful anticipation mixed with nervous excitement**."
 [ctrl2] "**What kind of story is this likely to be?** ... * **Family drama:** The focus on the letter, the daughter's return, and"
```

Note that on both passages `ring` and `ctrl` land on the *same* reframing, which is further evidence
that the effect at this scale is generic perturbation, not ring-specific content.

## Caveats

1. **Feature semantics are unknown.** No labels are available in this environment; generality and
   nearest-active-feature token anchors are the only handles. "The ring is the implied content"
   cannot be checked directly, only through its statistics and its causal effect.
2. **"Adjacency" in decoder space is a proxy** for Deacon's adjacency, and a loose one. High decoder
   cosine most often means *near-duplicate feature*, which is a claim about dictionary redundancy,
   not about semantic implication. A better operationalization would use encoder-side co-occurrence
   or a "fires on the continuation but not the passage" definition.
3. **τ is a quantile, not a boundary.** The cosine distribution is smooth; every ring statistic
   moves continuously with τ.
4. **One layer (20), one width (16k), one L0 (47), one model.** The 131k dictionary is available and
   would split these features further; the census was not repeated there.
5. **Generality is measured against 72 narrative passages only.** A feature with g = 0 is "never
   fires on *these* passages", not "never fires".
6. **n = 12 for the probe**, one patch scale (0.15), one patch construction (sum of 8 decoders,
   renormalized). The ring-vs-`ctrl2` KL result at n = 12 is suggestive, not established.
7. The probe patches **every position**, so the final-position KL includes accumulated effects, not
   a clean single-site intervention.
8. **The census null is the strongest negative result here** and should not be buried: on size,
   rarity and by-factor composition, a passage's absential ring is statistically indistinguishable
   from the ring of an arbitrary set of the same size. Only the causal probe separates them.

## What would move this

- Define the ring from the *continuation* instead of geometry: features that fire on the model's own
  greedy continuation of the passage but not on the passage. That is the withheld-betrayal test
  VISION.md actually names, and it has semantics for free.
- Repeat at width 131k, where a near-duplicate at cosine 0.7 is a different object.
- Sweep the patch scale: if the ring's KL advantage is amplification of live features, it should
  scale with the component the residual already has along the ring vector — measurable directly.
