# Derivative curves: where along a passage a factor becomes readable

The "differentiate" operator of `VISION.md` at the representational level. Slice a passage by token
and by sentence, project each slice onto the factor directions, and read the sequence as a curve;
the beat-to-beat difference is the derivative, the running mean is the integral so far.

**Setup.** Qwen2.5-1.5B (28 layers, d=1536), CPU, one forward pass per passage (72 passages total).
Grids: `prompts/narrative_theme_v1.json` (4 situations × 3 eras × 3 themes, three-sentence
passages) read at **layer 20**, and `prompts/narrative_mood_v1.json` (4 scenes × 3 eras × 3 moods,
one sentence) read at **layers 16/18/20**. Directions are the usual leave-one-situation-out factor
means (`dir_level` = mean of the mean-pooled span vectors with that level over the three training
situations, minus the grand mean). Every token vector of the held-out passage is centred by the
**token-level** grand mean of the training passages and cosine-projected onto the three level
directions of each factor. Two readouts per token:

- **margin** = cos(own level) − mean cos(other two levels). Chance 0.
- **hit** = 1 if the own level is the argmax over the three. Chance 1/3.

Baseline: eight random direction triples of matched norm, run through the identical pipeline
(`margin` ≈ 0.000 ± 0.002, `hit` ≈ 0.33 at every position — the curves below are not an artefact of
the centring or the cosine).

Scripts: `scripts/derivative_curves.py` (extraction + aggregation → `results/derivative_curves.json`),
`scripts/derivative_figures.py` (figures). Reproduce with

```
python scripts/derivative_curves.py --out results/derivative_curves.json
python scripts/derivative_figures.py
```

---

## 1. Theme accumulates in the middle, not at the end

![theme vs position](../figures/derivative_theme_position.png)

Binned by normalised position (12 bins), mean over all 36 passages, layer 20:

| position bin | 0.04 | 0.12 | 0.21 | 0.29 | 0.37 | 0.46 | 0.54 | 0.62 | 0.71 | 0.79 | 0.88 | 0.96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| theme margin | 0.001 | 0.002 | 0.042 | 0.078 | 0.096 | **0.109** | 0.095 | 0.105 | 0.084 | 0.068 | 0.039 | 0.038 |
| theme hit | 0.33 | 0.34 | 0.45 | 0.51 | **0.69** | **0.69** | 0.60 | 0.65 | 0.64 | 0.53 | 0.43 | 0.45 |
| random hit | 0.34 | 0.33 | 0.33 | 0.32 | 0.33 | 0.32 | 0.34 | 0.34 | 0.34 | 0.33 | 0.33 | 0.34 |

Theme is at **exact chance for the first ~15% of the passage** (hit 0.33/0.34, margin 0.001), rises
steeply through the first sentence boundary, peaks at 0.69 around the 40–50% mark, and **decays
again over the last quarter** (0.43–0.45). It is a mid-passage, not a late-passage, phenomenon: the
theme becomes readable once enough of the situation has been laid down to make the relation among
the events legible, and the closing sentence — which in these passages is aftermath, a gesture, a
line of dialogue — dilutes it again.

Per-sentence (34 of 36 passages split cleanly into three; two split into 2 or 4 on `". "` because of
`Dr.` and a missing trailing space — excluded from the sentence analysis):

| theme | sent 1 | sent 2 | sent 3 | Δ(s2−s1) | Δ(s3−s2) | share of accumulation |
|---|---|---|---|---|---|---|
| betrayal (n=11) | 0.003 | **0.056** | 0.006 | +0.053 ± 0.016 | −0.050 ± 0.027 | 0.09 / **0.54** / 0.19 |
| sacrifice (n=12) | 0.027 | **0.078** | 0.062 | +0.050 ± 0.017 | −0.015 ± 0.012 | 0.21 / **0.39** / **0.40** |
| homecoming (n=11) | 0.041 | **0.157** | 0.022 | +0.116 ± 0.024 | −0.135 ± 0.014 | 0.20 / **0.69** / 0.11 |
| all (n=34) | 0.024 | **0.096** | 0.031 | +0.073 ± 0.012 | −0.065 ± 0.014 | 0.17 / 0.54 / 0.24 |

Argmax accuracy per sentence: 0.39 / **0.64** / 0.41.

**It does differ by theme.** All three are hump-shaped, but the hump is very different in size and
sharpness:

- **homecoming** is the loudest and the most peaked — sentence 2 margin 0.157, hit **0.80**, and by
  far the largest derivative in both directions (+0.116 then −0.135). Sentence 2 of a homecoming
  passage is where the returning figure is named ("gone to the wars for nine years and counted
  dead", "lost twenty years ago, and its pilot was her mother"); the model reads it almost
  perfectly and then lets go.
- **sacrifice** is the most *sustained*: it is the only theme whose signal survives into sentence 3
  (0.062, hit 0.50), and its accumulation is split almost evenly between sentences 2 and 3
  (0.39/0.40). Its derivative is nearly flat after the rise (−0.015 ± 0.012, not distinguishable
  from zero).
- **betrayal** is the weakest overall (peak 0.056, hit 0.51) and the most concentrated in the middle
  (54% of accumulation in sentence 2, back to 0.006 in sentence 3).

So the "∫ of a passage" is far from uniform: a little over half of what these passages accumulate on
their theme direction is accumulated in the second of three sentences, and the shape of that
accumulation is theme-specific.

## 2. Era locks in at the first token; theme does not

![era vs position](../figures/derivative_era_position.png)

| position bin | 0.04 | 0.12 | 0.21 | 0.29 | 0.37 | 0.46 | 0.54 | 0.62 | 0.71 | 0.79 | 0.88 | 0.96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| era margin | 0.111 | 0.127 | 0.138 | 0.131 | 0.155 | 0.142 | 0.136 | 0.137 | 0.134 | 0.145 | 0.143 | 0.134 |
| era hit | 0.62 | 0.70 | 0.78 | 0.74 | 0.78 | 0.76 | 0.73 | 0.71 | 0.76 | 0.72 | 0.71 | 0.62 |

**Yes.** In the very first bin era is already at hit 0.62 / margin 0.111 while theme is at 0.33 /
0.001, and era's curve is essentially flat across the whole passage (0.62–0.78, margin 0.11–0.16)
where theme's rises by 0.11 and falls back. By sentence: era 0.70 / 0.75 / 0.70 (margin
0.122 / 0.138 / 0.143, derivatives +0.016 ± 0.016 and +0.005 ± 0.012 — **no significant beat-to-beat
change at all**), versus theme 0.39 / 0.64 / 0.41. Era is an *address* that is fixed by the opening
noun phrase and then merely maintained; theme is a *form* that has to be integrated out of the
relation between events, and which the readout only holds transiently.

One clean exception, visible in the figure: **1920s** is the only level that starts below chance
(bin 0 margin −0.061, hit 0.26) and climbs monotonically to the top of the range by the end (0.218,
hit 0.90; Δ(s2−s1) = +0.112 ± 0.015). The 1920s passages open on unmarked modern prose ("The bank
called the loan on a Tuesday") and only become period-specific once *Packard*, *streetcar*,
*speakeasy* arrive. Medieval and far-future are marked from their first words and instead **decay**
(far-future 0.218 → 0.062, Δ(s3−s2) = −0.062 ± 0.015). Era locking in early is therefore a fact
about lexical marking, not about era as such — which is consistent with hour 6's caveat that "era"
is partly vocabulary.

## 3. Mood: yes, a late-token phenomenon — and it is monotone, unlike theme

![mood vs position](../figures/derivative_mood_position.png)

Mood grid, one-sentence passages, mean over 36, layer 18 (16 and 20 are within noise of it):

| position bin | 0.04 | 0.12 | 0.21 | 0.29 | 0.37 | 0.46 | 0.54 | 0.62 | 0.71 | 0.79 | 0.88 | 0.96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mood margin | −0.001 | 0.001 | 0.011 | 0.042 | 0.066 | 0.076 | 0.072 | 0.098 | **0.106** | 0.102 | 0.084 | 0.072 |
| mood hit | 0.33 | 0.34 | 0.36 | 0.46 | 0.52 | 0.56 | 0.48 | 0.59 | 0.59 | 0.59 | 0.50 | 0.55 |
| era hit (same passages) | 0.44 | 0.67 | 0.66 | 0.73 | 0.80 | 0.76 | 0.68 | 0.75 | 0.69 | 0.69 | 0.71 | 0.71 |

Mood sits at **exactly chance for the first third of the sentence** (0.33/0.34/0.36, margin ≈ 0) and
then climbs to a plateau of ~0.59 over the last 40%. Averaged over the first three tokens, hit is
0.333 — literally chance; over the last three, 0.55; at the **final token, 0.61 vs 0.48 for the mean
over all tokens** (layer 18; 0.61 vs 0.49 at layer 16, 0.58 vs 0.49 at layer 20). This is the
per-token version of hour 9's finding (last-token pooling 0.75 vs mean pooling 0.67) and it
reproduces the ordering, with the caveat that the directions used here are mean-pooled ones, which
is the harder setting for a last-token effect.

The mirror-image control is in the same data: on the theme grid the ordering **reverses** — theme
last-token hit 0.47 vs mean-over-tokens 0.53, and era on that grid 0.53 vs 0.72. So the
last-token advantage is specific to mood, not a generic edge effect.

Per mood (layer 18), the three are qualitatively different:

- **comic** is the extreme case: margin −0.034 at the start (actively anti-comic), crossing zero
  around 55% of the sentence and ending at 0.17–0.23 with hit up to **0.94**. The joke is in the
  last clause and nowhere else, exactly as the grid was written.
- **dread** peaks *early-middle* (0.093 at bin 0.29) and then sags to ~0.01 — the dread cue in these
  sentences is an image placed mid-sentence ("the horse's eyes were wrong"), and the trailing
  clause dilutes it. Its mean hit over the sentence is the lowest of the three (0.37, vs 0.51 tender and 0.63 comic).
- **tender** is roughly flat and modest (0.02–0.10) — tenderness is spread over the sentence.

So "mood is integrated at the end" is true of the factor on average and of comic in particular, but
it is not a property every mood level shares.

## 4. The characteristic shape of the beat-to-beat derivative

![sentence bars and derivative](../figures/derivative_sentence_bars.png)

For theme, the derivative has one characteristic shape: **positive then negative**, a single hump
peaked on the middle beat. All three themes share the sign pattern, and the magnitudes separate
them: homecoming (+0.116, −0.135) ≫ betrayal (+0.053, −0.050) ≈ sacrifice (+0.050, −0.015), with
sacrifice the only one whose second derivative step is not significantly different from zero. In
integral terms: betrayal and homecoming spend their accumulation almost entirely in the middle beat
(54% and 69%), sacrifice spreads it over the last two (39% + 40%).

For era, the derivative is **flat and not significantly different from zero in aggregate**
(+0.016 ± 0.016, +0.005 ± 0.012). It is only non-zero per level, and there in a way that tracks
lexical marking rather than the factor: 1920s rises (+0.112, +0.051), far-future falls (−0.039,
−0.062), medieval is flat.

This is what a derivative is supposed to do to the calculus: it kills the constant. Era's readout is
a large constant with no derivative; theme's readout is a small constant with a large derivative.
Reading "how much theme" off the mean-pooled vector (hours 10/14) throws away exactly the part that
distinguishes homecoming from sacrifice.

## Caveats

- **One model, one size.** Qwen2.5-1.5B base. Hour 14 showed the theme readout ceiling is
  model-dependent (0.78 on 1.5B vs 0.94 on Gemma-2-9B-it), so the decay in the last quarter may
  partly be readout weakness rather than a fact about the passages.
- **Few passages, and the directions come from them.** 36 passages per grid; each leave-one-out
  direction is a mean of 9 mean-pooled vectors. Level means over 10–12 passages carry real
  uncertainty; error bands are ±1 s.e. over passages, and the per-level curves in particular should
  be read as shapes, not as calibrated probabilities.
- **Per-token cosines are noisy.** A single token's residual is dominated by its own identity;
  everything above is an average over 10–36 passages and, for the binned curves, over ~2–8 tokens
  per bin. The bin-to-bin wiggles (e.g. the dip at 0.54 in the theme curve) are within noise.
- **Directions are mean-pooled; the readout is per-token.** The centring uses a token-level grand
  mean but the directions do not, so the projection is slightly mismatched to the per-token
  statistics. A last-token-trained direction set would likely sharpen the mood result.
- **"Era locks in early" is confounded with lexical marking**, as the 1920s exception shows
  directly. The same caveat applies to the claim that theme is not lexically cued: hour 10 found
  0.67 theme decodability already at layer 0 from mean pooling, so part of the mid-passage hump is
  vocabulary (*sworn*, *forged*, *gone … years ago*) rather than relational structure.
- **Sentence split is naive** (`". "`), which drops 2 of 36 theme passages from the per-sentence
  analysis. The binned position curves use all 36.
- Purely **representational**: nothing here is a generation result, and the selector/generator gap
  recorded in hours 9–11 still stands.
