> **Provenance.** This note is the executing agent's own, recovered after I removed its worktree
> while its computation was still running — my error, recorded at hour 42b. Its run went further than
> the partial one I first published from a log: twelve thresholds to g_k = 0.95 rather than five to
> 0.1, both statistics, and 1000 permutations per threshold. `results/sae_ladder_nulls.json` carries
> nulls 1 and 2 from an equivalent run; the script and the null-3 JSON were lost with the worktree.

# Abstraction ladder — the three hour-39 nulls (Phase 0.2)

Runs `scripts/sae_ladder_nulls.py` → `results/sae_ladder_nulls.json`. Pure offline linear algebra
on cached data: Gemma Scope JumpReLU dictionaries at layer 20 of Gemma-2-9B-it (16k =
`layer_20/width_16k/average_l0_47`, 131k = `layer_20/width_131k/average_l0_43`) and the cached
per-token residuals for the Picard target, the 72 narrative reference passages, and the 111 broad
reference passages (`results/tokens_gemma9b_l20.npz`, `results/tokens_gemma9b_broad_l20.npz`).
No forward pass, no NDIF, no download. Wall time 266s on this CPU-only session (warm disk cache;
a cold run took longer only because of first-touch page-cache misses on the 131k dictionary's
blob, ~3.8 GB, not because of any computation cost).

**Replication check first.** Before running the nulls, the hour-15/26 numbers were recomputed from
the cached data to confirm nothing had drifted: 16k mean generality 0.182 (narrative) / 0.187
(broad), 131k 0.063 / 0.101 — matching RESULTS.md hours 15 and 26 to three decimals. Content-feature
counts: 16k 1404 (narrative) / 1577 (broad) of 1898 active; 131k 1722 (narrative) / 1972 (broad) of
2329 active. Merge test on this session's data: **13/15 both corpora** (matches hour 26's "13 of
15 under the broad corpus, 13/15 under the narrative one here" note — not hour 15's original
12/15, because hour 15 used a slightly different content filter, as hour 26 already documented).

All three nulls below are run against this session's 13/15, not hour 15's 12/15, since that is what
was actually recomputed here; the verdicts are the same for 12/15 (see null 1).

---

## Null 1 — merge-test null

**Claim under test.** 13 of 15 wide (131k) content features have a nearer-cosine narrow (16k)
match that is more general. Currently compared against nothing.

**Construction.** For each of the 15 wide features, the matching procedure searches over **all**
16384 features of the 16k dictionary (the code takes `argmax` over the full decoder-cosine vector,
not just the content-filtered subset) and returns the closest one. The correct null draws that
partner from the same population: a uniformly random one of the 16384 16k-dictionary features,
with its actual generality value. For wide feature *i* this gives `p_i = P(random narrow feature is
more general than wide feature i) = mean(g16_full > wide_gen_i)` over all 16384 features. The 15
`p_i` are independent Bernoulli rates (each concerns a different, unrelated draw), so the exact null
distribution of "how many of 15 go up" is the Poisson-binomial distribution of those 15 Bernoullis,
computed by DP convolution — no Monte Carlo needed, no sampling error.

**Numbers.**

| corpus | observed | null mean | null sd | null 95% CI (of 15) | P(K ≥ observed) |
|---|---|---|---|---|---|
| narrative | 13/15 | 3.99 | 1.49 | [1, 7] | 3.5×10⁻⁹ |
| broad | 13/15 | 5.78 | 1.31 | [3, 8] | 9.8×10⁻¹⁰ |

(For reference, hour 15's original 12/15 under the same narrative-corpus null: still far outside
the [1,7] 95% band, P(K≥12) ≈ 1.5×10⁻⁸ — the conclusion does not depend on which content filter
produced the count.)

**Reading.** The null mean is well below 50% (3.99/15 = 27% narrative, 5.78/15 = 39% broad) — the
task's premise is confirmed: the two dictionaries' marginal generality distributions differ enough
that a coin flip would be the wrong null, and a random narrow partner beats the wide feature less
than half the time on average. But even against this corrected, harder null, 13/15 is roughly 6-9
null standard deviations above the mean. **Null 1: the merge-test claim survives, decisively.**

---

## Null 2 — size-matched random-feature control on the width effect

**Claim under test.** 16k mean generality (0.182 narrative / 0.187 broad) exceeds 131k mean
generality (0.063 / 0.101). The dictionaries differ in total width (16384 vs 131072); is the
effect just "fewer candidate features"?

**Construction.** Per-feature properties — whether the target activates a 131k feature at all,
whether its peak target token is alphabetic, its generality — were computed once over the full
131072-wide encode. 4000 repeats: draw a uniformly random 16384-feature subset of the 131k
dictionary's own 131072 features (matching the 16k dictionary's total width exactly), restrict the
"active & alphabetic-peak & generality ≤ 0.9" content mask to that subset, and take the mean
generality of whatever content features land inside it. Because generality is a fixed per-feature
property, no re-encoding is needed per repeat — this is a resampling test, not a retraining test
(a genuinely retrained 16k-wide SAE is not available and was out of scope/budget here; see
confounds below).

**Numbers.**

| corpus | observed 16k mean | 131k full-dict mean | size-matched-subset null mean ± sd | avg content features/subset | P(subset mean ≥ observed) | z |
|---|---|---|---|---|---|---|
| narrative | 0.182 | 0.063 | 0.063 ± 0.008 | 215 | 0.0000 (0/4000) | 14.8 |
| broad | 0.187 | 0.101 | 0.101 ± 0.012 | 247 | 0.0000 (0/4000) | 7.1 |

**Reading.** The size-matched random subset's mean generality lands on top of the full 131k
dictionary's mean (0.063 vs 0.063; 0.101 vs 0.101) — exactly what you'd expect, since subsampling a
fixed population of feature-level generality values just resamples the same population mean with
some extra variance from the smaller n (avg 215-247 content features per subset vs 1722-1972 in
the full dictionary). None of 4000 random size-matched subsets came anywhere near the observed 16k
mean; z = 14.8 (narrative) and 7.1 (broad). **Null 2: the width effect is not explained by size
alone — it is specific to what a dictionary *trained* at 16k width preserves, not to having fewer
candidate features.** This is the cleanest of the three verdicts: the size-matched control and the
real (trained) 16k dictionary are not draws from the same distribution by a wide margin.

---

## Null 3 — matched-count + label-permutation null on the flow ordering

**Claim under test** (hour 26): as the generality threshold g_k rises and fewer features survive to
reconstruct each passage, era's 8-NN purity decays toward its chance floor (23/71 = 0.32) faster
than theme's (chance 8/71 = 0.11) — "era's structure dies before theme's."

**Construction.** At each of the 12 thresholds used in `sae_ladder_v2.py`, rebuild the
reconstruction `Vc = Σ_{f∈survivors} mean_activation_f · W_dec[f] + b_dec` for the 72 narrative
passages from the 16k dictionary, exactly as the original script does. Then permute the theme
labels (preserving the true 8-theme × 9-passages-each structure) and the era labels (preserving the
true 3-era × 24-passages-each structure) independently, 1000 times each, recomputing (a) 8-NN
purity and (b) the within-label/across-label mean-cosine-distance ratio under each permutation.
Because a permutation of labels holds each label's class sizes fixed by construction, this single
procedure is simultaneously the task's "matched-count" null (same 8×9 / 3×24 structure as truth)
and its "label-permutation" null — they are the same construction here, not two different ones.
One-sided p-values were computed in the direction that means "more structured than chance": higher
observed than the permutation null for 8-NN purity, *lower* observed than the permutation null for
the within/across ratio (a bug in the first draft of this script had both tests running in the
purity-only direction, silently making every ratio comparison read "not significant" at every g_k;
caught before trusting the numbers below by checking that ratio and purity gave contradictory
"dies" verdicts at g_k=0.0, which is impossible if both track the same structure — fixed, and the
run redone in full).

**Numbers — broad-corpus generality (12 thresholds, full table in the JSON):**

| g_k | n feat | theme knn8 | p | era knn8 | p | theme ratio | p | era ratio | p |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 16384 | 0.628 | <10⁻³ | 0.535 | <10⁻³ | 0.606 | <10⁻³ | 0.888 | <10⁻³ |
| 0.1 | 2681 | 0.517 | <10⁻³ | 0.483 | <10⁻³ | 0.520 | <10⁻³ | 0.922 | 0.002 |
| 0.4 | 415 | 0.422 | <10⁻³ | 0.415 | 0.001 | 0.453 | <10⁻³ | 0.948 | 0.028 |
| 0.7 | 172 | 0.363 | <10⁻³ | 0.363 | 0.049 | 0.421 | <10⁻³ | 0.963 | 0.061 |
| 0.95 | 58 | 0.297 | <10⁻³ | 0.389 | 0.008 | 0.377 | <10⁻³ | 0.959 | 0.106 |

Narrative-corpus generality: qualitatively identical for 8-NN purity (theme and era both stay
p<10⁻³ across every threshold, never approach the null band); the within/across ratio null-p rises
with g_k but tops out at 0.016 (era, g_k=0.95) — never crosses 0.05.

**"Dies at" (first g_k where the statistic enters its 95% permutation-null band, p≥0.05):**

| corpus | theme (8-NN) | era (8-NN) | theme (ratio) | era (ratio) |
|---|---|---|---|---|
| broad | never | never | never | **g_k ≥ 0.7** |
| narrative | never | never | never | never |

**Reading — this is the mixed result.** Against a matched-count/permutation null (as opposed to
the fixed theoretical chance floor RESULTS.md used, 23/71 = 0.32), **8-NN purity for both theme and
era stays significantly above chance at every g_k tested, all the way to g_k=0.95** (era z=2.77,
p=0.008 at the top of the ladder). By this measure neither "dies" in the sense of becoming
statistically indistinguishable from a random relabeling — RESULTS.md's language ("era purity...
is gone by g_k=0.7") was reading the observed value's proximity to the theoretical class-frequency
floor (0.32), not testing it against a null that accounts for the geometry of this specific
72-passage reconstruction, and the permutation null shows real residual structure survives past
that point for both labels. The within/across cosine-distance ratio tells a different, narrower
story that does support the ordering: under the broad corpus, era's ratio crosses into its
permutation-null band at g_k≥0.7 while theme's never does in either corpus — a genuine ordering,
but it appears in one of the two statistics, under one of the two reference corpora, and the
crossing is marginal (p=0.061, just past the 0.05 line, and era's p rises further to 0.106 by
g_k=0.95 rather than staying flat, i.e. it is not settling into the null band so much as drifting
through it with noise). **Null 3: "era dies before theme" is not confirmed as originally stated —
neither label's 8-NN purity ever becomes indistinguishable from a matched-count permutation null.
The weaker, ratio-only, broad-corpus-only version of the ordering (era's within/across ratio
crosses its own null band before theme's, which never does) survives, marginally.**

---

## Verdict summary (writeup claim 6)

| null | claim | survives? |
|---|---|---|
| 1. merge-test null | 12/15 (or 13/15) wide→narrow merges go up | **Survives, decisively** (13/15 is 6-9 null-sd above a correctly-constructed Poisson-binomial null with mean 4-6/15, not 7.5/15) |
| 2. size-matched random-feature control | 16k mean generality (0.18-0.19) exceeds 131k's (0.06-0.10) because of what training at 16k width preserves, not because of dictionary size | **Survives, decisively** (z = 7-15; a random size-matched subset of the wide dictionary reproduces the wide dictionary's own mean, not the narrow one's) |
| 3. matched-count + label-permutation null on the flow ordering | era's structure "dies before" theme's as g_k rises | **Not confirmed as stated.** 8-NN purity: both labels stay significant to g_k=0.95, no ordering. Within/across ratio: a marginal, corpus-specific ordering survives (era's ratio, broad corpus only, crosses its null band at g_k≥0.7; theme's never does) |

**Per the program's own instruction** ("this is a minor claim... if it fails its null, say so
plainly and do not chase it"): nulls 1 and 2 — the parts of claim 6 about width and about which
features survive narrowing — hold up cleanly against the correct nulls and can be trusted as
stated. Null 3 — the "ordering" reading of the flow (era before theme) — does not survive as
originally worded; RESULTS.md's phrasing should be read as describing where the *observed values*
sit relative to the theoretical class-frequency floor, not as a tested claim that era's local
neighborhood structure becomes statistically indistinguishable from noise before theme's does. The
weaker directional claim (era's global within/across separation crosses its own permutation null
before theme's, under the broad corpus only) is marginally supported and should replace the
stronger "dies before" language if this result is kept in the writeup.

## What is still confounded

- **Null 2 is a resampling control, not a retraining control.** It tests "does a random subset of
  the wide dictionary's *already-trained* features reproduce the narrow dictionary's generality?"
  (no), not "does a dictionary *retrained* from scratch at 16k width, with a different
  initialization/sparsity target, reproduce it?" A genuinely retrained comparably-sized SAE was out
  of scope (compute/time) and would be a stronger control if ever built.
- **Null 1's population is the full 16384-feature 16k dictionary**, including features never active
  on any reference passage (generality 0). This is the literal population the nearest-cosine search
  ranges over, so it is the right null for "is the match doing better than a random member of the
  space it was drawn from" — but it is not the same as asking whether the match beats a random
  *active* or *content* 16k feature specifically; that narrower null would have a higher mean and
  would still very likely be cleared given the margin (6-9 sd), but was not computed separately.
- **Null 3's permutation null holds the reconstruction Vc fixed** and only reshuffles which passage
  gets which label. It is therefore a null about *this specific geometry* (this SAE, this
  threshold, these 72 passages), not a null about whether some other random 72-passage corpus would
  show the same shape. It also does not test the "no intermediate optimum, decay is monotonic from
  the bottom" half of hour 26's reading, which this session did not re-examine.
- **All three nulls are still specific to one target description (Picard), one layer (20), and the
  two cached dictionary widths (16k, 131k).** The layer-9/31 flatness (hour 26 part A) and the
  131k-at-other-layers gap (never filled, disk-limited) are untouched by this session.
- **Generality remains label-free co-occurrence**, not a semantic feature label (Neuronpedia is
  still unreachable here), so "more general" continues to mean "fires on more reference passages,"
  as in every prior stage.

## Files

- `scripts/sae_ladder_nulls.py` — the three nulls, offline, CPU-only, no NDIF.
- `results/sae_ladder_nulls.json` — every number above plus per-feature merge-test rows, the full
  Poisson-binomial PMF for null 1, all 4000 subset means are not individually stored (summary
  stats only) for null 2, and the full 12-threshold × 2-corpus level table for null 3.
- `results/sae_ladder_nulls.log` — run log.
