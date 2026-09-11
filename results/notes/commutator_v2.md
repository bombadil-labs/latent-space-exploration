# Commutator trajectories, v2: the controls hour 19 lacked

`scripts/ndif_commutator.py` (extended: `--null`, `--layers`, four prompts) ·
`results/commutator_gemma9b_v2.json` · previous note: `results/notes/commutator.md`.

Hour 19 ran the AB/BA commutator protocol on Gemma-2-9B-it and reported that factor patches do not
commute under generation, that the divergence is bounded, and that one factor of each pair dominates
the surface text by depth. Its own caveat list named three missing controls. This run adds all three,
**without touching the pre-registered regime rule** (still the module docstring of
`scripts/ndif_commutator.py`, committed in `8d4b4f9` before the first run):

1. a **random-direction null** — pairs of random directions at matched norm through the identical
   AB/BA protocol;
2. **two more prompts** (`door`, `fire`), four in total, so aggregate regime labels no longer rest
   on a 1–1 tie;
3. a **second layer pair** (16 and 24) for era × theme, to test whether the dominance ordering is
   depth-driven or an artefact of block 14.

Everything else is hour 19's protocol verbatim: Gemma-2-9B-it via NDIF, greedy, 60 new tokens, patch
re-applied at every decoding step, readout at block 20 on unpatched re-runs of the generated texts,
divergence in units of the base continuation's own per-token spread (sigma). The hour-19
generations for `news`/`road` at L14/L20 are **reused, not re-run**: `results/commutator_gemma9b_v2.json`
was seeded from `results/commutator_gemma9b.json`, whose numbers reproduce exactly under the
re-analysis. 308 new NDIF jobs (34 + 34 null, 76 + 76 new prompts, 88 second layer pair), ~25 min.

## The null

For each prompt and each of N = 4 draws, two isotropic random unit vectors `u_A`, `u_B` are drawn
(seeded, `numpy.default_rng([seed, prompt_index, draw])`) and rescaled, **at each layer where they
are applied**, to the mean norm over levels of the factor direction they replace at that layer
(`|dir_era|@14 = 16.7`, `@20 = 21.8`; `|dir_theme|@14 = 10.8`, `@20 = 17.5`; `|dir_era|`/`|dir_voice|`
for the other pair). `AB` = `u_A`@14 + `u_B`@20, `BA` = `u_B`@14 + `u_A`@20, exactly as for the
factors; the readout axes are `u_A`, `u_B` at block 20 and sigma comes from the base continuation's
per-token spread on those same axes. 8 null cases per factor pair (2 prompts × 4 draws), 16 total.

**Random patches at this norm do not break the model.** Every null continuation is fluent, on-genre
English — the model absorbs a 17–24-norm random vector at two blocks without degrading. So the
comparison below is between two kinds of *working* intervention, not between steering and noise.

### Null vs factor, side by side (news + road, the prompts both were run on)

| | n | ident. | Hamming | median first div | mean curve (σ) | early (σ) | late (σ) | consistency | base overlap |
|---|---|---|---|---|---|---|---|---|---|
| **factor**, both pairs pooled | 36 | 0.11 | 0.62 | 15.0 | 1.13 | 0.44 | 1.41 | 0.60 | **0.14** |
| **null**, both pairs pooled | 16 | 0.06 | 0.56 | 20.0 | 1.05 | 0.42 | 1.54 | 0.61 | **0.32** |
| Mann-Whitney p | | | 0.59 | 0.50 | 0.73 | 0.72 | 0.39 | 0.51 | **0.002** |

Per pair:

| | n | ident. | Hamming | median first div | mean curve (σ) | late (σ) | base overlap |
|---|---|---|---|---|---|---|---|
| era × theme (factor) | 18 | 0.06 | 0.59 | 16.5 | 1.18 | 1.50 | 0.15 |
| era × theme (null) | 8 | 0.00 | 0.54 | 20.0 | 1.02 | 1.69 | 0.38 |
| era × voice (factor) | 18 | 0.17 | 0.64 | 11.0 | 1.09 | 1.33 | 0.12 |
| era × voice (null) | 8 | 0.12 | 0.59 | 14.0 | 1.07 | 1.39 | 0.26 |

Regime distribution, factor (news+road, n = 36) vs null (n = 16): noise 0.44 / 0.38, crystalline
0.22 / 0.25, structured 0.19 / 0.12, commute 0.11 / 0.06, drain 0.03 / **0.19**.

### The 16 null cases

| pair | prompt | draw | regime | first div | Hamming | ov(AB,base) | ov(BA,base) | mean σ | late σ | consistency |
|---|---|---|---|---|---|---|---|---|---|---|
| era×theme | news | r0 | drain | 20 | 0.62 | 0.78 | 0.38 | 1.08 | 1.55 | 0.68 |
| era×theme | news | r1 | crystalline | 53 | 0.02 | 0.47 | 0.47 | 0.08 | 0.33 | 0.86 |
| era×theme | news | r2 | structured | 15 | 0.72 | 0.02 | 0.02 | 1.40 | 1.72 | 0.82 |
| era×theme | news | r3 | structured | 0 | 1.00 | 0.03 | 0.42 | 2.14 | 2.67 | 0.71 |
| era×theme | road | r0 | drain | 36 | 0.40 | 0.58 | 0.90 | 0.58 | 1.38 | 0.67 |
| era×theme | road | r1 | noise | 37 | 0.38 | 0.30 | 0.28 | 0.67 | 1.79 | 0.57 |
| era×theme | road | r2 | noise | 20 | 0.52 | 0.43 | 0.50 | 0.99 | 1.65 | 0.57 |
| era×theme | road | r3 | noise | 18 | 0.68 | 0.25 | 0.30 | 1.26 | 2.38 | 0.52 |
| era×voice | news | r0 | drain | 20 | 0.60 | 0.78 | 0.40 | 1.10 | 1.62 | 0.68 |
| era×voice | news | r1 | crystalline | 53 | 0.02 | 0.47 | 0.47 | 0.08 | 0.33 | 0.86 |
| era×voice | news | r2 | commute | — | 0.00 | 0.02 | 0.02 | 0.00 | 0.00 | 0.00 |
| era×voice | news | r3 | crystalline | 0 | 1.00 | 0.03 | 0.08 | 2.07 | 2.28 | 0.58 |
| era×voice | road | r0 | crystalline | 4 | 0.93 | 0.28 | 0.07 | 1.58 | 1.41 | 0.58 |
| era×voice | road | r1 | noise | 37 | 0.38 | 0.30 | 0.28 | 0.66 | 1.76 | 0.57 |
| era×voice | road | r2 | noise | 4 | 0.90 | 0.43 | 0.10 | 1.42 | 1.73 | 0.62 |
| era×voice | road | r3 | noise | 8 | 0.85 | 0.25 | 0.13 | 1.63 | 1.94 | 0.56 |

The null produces every regime the factors produce, including one exact commutation (`news`, draw
r2 under era×voice norms: two random patches, both orders, 60 identical tokens) and three drains.

## Regimes over four prompts

Prompts: `news`, `road` (hour 19) plus `door` = "The door was already open when he got there, and"
and `fire` = "The fire had burned down to embers before anyone spoke, and". 36 cases per pair.

### era × theme — era @14, theme @20

| era | theme | modal regime | modal count | per prompt (news, road, door, fire) |
|---|---|---|---|---|
| medieval | betrayal | crystalline | 2/4 | crystalline, commute, crystalline, noise |
| medieval | sacrifice | noise | 2/4 (tie-broken) | structured, noise, structured, noise |
| medieval | homecoming | noise | 3/4 | noise, noise, structured, noise |
| 1920s | betrayal | crystalline | 3/4 | crystalline, crystalline, crystalline, structured |
| 1920s | sacrifice | noise | 2/4 | crystalline, drain, noise, noise |
| 1920s | homecoming | structured | 3/4 | structured, structured, noise, structured |
| farfuture | betrayal | noise | 1/4 (tie-broken) | structured, noise, drain, crystalline |
| farfuture | sacrifice | noise | 3/4 | noise, noise, drain, noise |
| farfuture | homecoming | noise | 3/4 | noise, noise, crystalline, noise |

36 cases: noise 16, crystalline 8, structured 8, drain 3, commute 1. Mean Hamming 0.69, mean curve
1.29σ, late 1.63σ, median first divergence token 12.

### era × voice — era @14, voice @20

| era | voice | modal regime | modal count | per prompt (news, road, door, fire) |
|---|---|---|---|---|
| medieval | terse | crystalline | 2/4 | noise, structured, crystalline, crystalline |
| medieval | ornate | noise | 2/4 (tie-broken) | noise, crystalline, noise, crystalline |
| medieval | child | structured | 3/4 | structured, structured, noise, structured |
| 1920s | terse | structured | 2/4 | noise, commute, structured, structured |
| 1920s | ornate | noise | 2/4 | noise, commute, crystalline, noise |
| 1920s | child | crystalline | 3/4 | crystalline, crystalline, noise, crystalline |
| farfuture | terse | noise | 1/4 (tie-broken) | noise, commute, drain, crystalline |
| farfuture | ornate | noise | 2/4 | noise, noise, crystalline, structured |
| farfuture | child | noise | 2/4 | crystalline, noise, drain, noise |

36 cases: noise 13, crystalline 11, structured 7, commute 3, drain 2. Mean Hamming 0.67, mean curve
1.12σ, late 1.33σ, median first divergence token 9.

**Four prompts remove the 1–1 ties but destroy the table.** 7 of 9 combinations now have a unique
modal regime in each pair (2 of 9 still go to the rule's tie-break), but **0 of 18 combinations have
all four prompts agreeing**, and the modal count is 2 or 3 out of 4 almost everywhere. A permutation
test — statistic = sum over the 9 combinations of the modal count, null = shuffle the regime labels
within each prompt, preserving that prompt's own regime marginal, 20 000 draws (`regime_structure`
in the JSON) — gives era × theme 22/36 observed vs 20.0 expected, **p = 0.16**; era × voice 19/36 vs
18.8, **p = 0.57**. The regime label carries no detectable level-combination structure once the
prompt is accounted for. The per-prompt marginals themselves differ a lot (era × theme: `news` 3
noise / 3 crystalline / 3 structured, `fire` 6 noise / 2 structured / 1 crystalline; era × voice:
all 3 commutations are on `road`), so prompt, not level combination, is what the regime column is
measuring.

## Second layer pair: era × theme at 16/24

Same protocol, patches at blocks 16 and 24, readout still at block 20, prompts `news` and `road`,
all 9 combinations, single patches included.

| | n | ident. | Hamming | median first div | mean curve (σ) | late (σ) | regimes |
|---|---|---|---|---|---|---|---|
| era × theme @14–20 | 18 | 0.06 | 0.59 | 16.5 | 1.18 | 1.50 | noise 8, crystalline 4, structured 4, commute 1, drain 1 |
| era × theme @16–24 | 18 | 0.06 | 0.66 | 17.5 | 1.19 | 1.64 | noise 8, structured 5, crystalline 2, drain 2, commute 1 |

Divergence statistics are essentially unchanged by moving both patch layers up.

**Dominance** — mean token overlap of each ordering with the two single-patch continuations
(`news` + `road`, 18 cases each):

| pair, layers | ov(AB, A-only) | ov(AB, B-only) | ov(BA, A-only) | ov(BA, B-only) |
|---|---|---|---|---|
| era × theme @14–20 | **0.48** | 0.20 | **0.44** | 0.18 |
| era × theme @**16–24** | **0.37** | 0.17 | **0.26** | 0.17 |
| era × voice @14–20 | 0.15 | **0.35** | 0.19 | **0.32** |

Era still dominates theme when the pair is moved from (14, 20) to (16, 24), under both orderings
(0.37 vs 0.17 and 0.26 vs 0.17). So that leg of the ordering is **not** a block-14 artefact: it
survives a shift of both patch layers. The era margin does shrink, and it shrinks most when era is
the *later* patch (BA: 0.44 → 0.26), which is consistent with era being a mid-stack factor whose
direction is less effective at block 24 than at block 20. The voice > era leg was **not** re-run at
16/24 (budget), so the full ordering voice > era > theme is confirmed at one layer pair only; what
is confirmed at two layer pairs is era > theme.

## Which hour-19 claims survive

**Survives, but is not about factors.** *"These factors do not commute in generation"* — true, and
the numbers are unchanged (32/36 token-different at two prompts, 68/72 over four). But **matched-norm
random directions do not commute either**, and by every divergence measure the null is
indistinguishable from the factor pairs: Hamming 0.56 vs 0.62 (p = 0.59), median first divergence
token 20 vs 15 (p = 0.50), mean readout curve 1.05σ vs 1.13σ (p = 0.73), late window 1.54σ vs 1.41σ
(p = 0.39), consistency 0.61 vs 0.60 (p = 0.51). Non-commutation under greedy generation is a
property of *any* two patches of this size applied at two blocks in two orders — an autoregressive
sensitivity result, not a statement about the factor algebra. Hour 19's reading that "commutation as
a gauge and commutation as a trajectory are different objects" stands; its implicit suggestion that
the trajectory result measures something about era, theme and voice does not.

**Survives.** *"The divergence is bounded, not explosive."* Late-window curves 1.3–1.7σ for factors
and for the null; nothing runs away in 60 tokens in either arm. (Bounded-ness is likewise not
factor-specific.)

**Survives, strengthened.** *"One factor dominates the surface text regardless of layer order."*
The overlaps are large and asymmetric, and era > theme holds at a second layer pair. This is the one
claim that is about the factors and is now tested against a layer change.

**New, and the only factor-specific effect in the null comparison.** Factor patches move the text
off the prompt prior; matched-norm random patches often do not. Mean base overlap 0.14 (factor) vs
0.32 (null), **p = 0.002** — the only comparison in the table that separates. In other words the
factor directions are doing real work, and what the commutator protocol measures — divergence
between the two orderings — is simply not the place that work shows up.

**Does not survive.** *"The drain regime is the generative face of prompt-prior resistance"* for
factor patches. Drain is **more** common in the null (3/16 = 19%) than in the factor arm (1/36 on
the same prompts, 5/72 over all four; Fisher p = 0.08 and 0.15, i.e. not significant but pointing
the wrong way for the hour-19 reading). Drain is what a patch that fails to steer looks like, and
random patches fail to steer about half the time.

**Does not survive.** *The regime table as a property of level combinations.* With two prompts, half
the labels were 1–1 ties; with four, no combination is unanimous and a permutation test finds no
level-combination structure at all (p = 0.16 and 0.57). The five-regime classification is a
reproducible function of a (prompt, pair, level, seed) case, but the aggregate column in hour 19's
tables should not be read as "this level combination is crystalline".

## Caveats

- **The null is 16 cases from 8 random unit-vector draws.** The same `u_A`, `u_B` are reused across
  the two factor pairs (only the matched norms differ), so the two null blocks are not independent;
  4 of the 8 (prompt, draw) pairs give near-identical statistics across pairs. Effective n is closer
  to 8–12 than 16, which is why only the large base-overlap gap reaches significance and why the
  smaller differences (null diverges slightly *later* and slightly *less*) should not be read as
  real.
- **One null design.** Isotropic Gaussian directions at the factor's mean norm. A null drawn from
  the residual stream's own covariance, or from the orthogonal complement of the factor plane, would
  be a stricter control and might separate where this one does not.
- **The second layer pair covers era × theme only**, two prompts, so the voice > era leg of the
  dominance ordering is still single-layer-pair evidence.
- Everything hour 19 flagged still applies: greedy decoding amplifies sub-threshold logit
  differences, one model, one scale, readout axes are only the two patched directions (a difference
  orthogonal to both reads as zero), sigma is a normalizer and not a significance test, and
  Gemma-2-9B-it's chat habit ("**What kind of story is this likely to be?**") consumes some of the
  60-token window.
- The tie-break in the pre-registered rule (modal regime, ties to the later regime in the list) is
  unchanged; with four prompts it now fires on 2 of 9 combinations per pair instead of 4–5.
