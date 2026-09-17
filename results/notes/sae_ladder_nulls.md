# The abstraction ladder, against its three missing nulls

*Phase 0.2 of `docs/PROGRAM.md`. The ladder (hours 15, 26; writeup claim 6) was the only standing
claim in this project with no null anywhere. Three were specified in the hour-39 audit. All three
ran. Numbers below are from the run log; the process died at serialization (memory) after producing
every result, and the JSON artifact was regenerated separately.*

## Null 1 — the merge test, against the right population

**Construction.** "12 of 15 merges go up" had been compared against nothing, and a coin flip is the
wrong null because the two dictionaries have different marginal generality distributions. For each
of the 15 strongest wide (131k) content features matched to a narrow (16k) feature by decoder
cosine, compute the probability that a uniformly random one of the 16,384 narrow features is more
general than the wide feature — the same population the cosine search ranges over — then convolve
those 15 unequal Bernoulli probabilities into the exact null via Poisson-binomial. No Monte Carlo.

| corpus | observed | null mean | null sd | null 95% CI | P(K ≥ observed) |
|---|---|---|---|---|---|
| narrative | 13/15 | 3.99 | 1.49 | [1, 7] | 0.0000 |
| broad | 13/15 | 5.78 | 1.31 | [3, 8] | 0.0000 |

**Survives**, decisively. Note the observed count is 13/15 on this run where the record says 12/15
(narrative); the matching procedure was re-derived here, so the small discrepancy is a re-derivation
difference and is recorded rather than reconciled.

## Null 2 — the width effect, size-matched

**Construction.** The 0.18 vs 0.06 comparison contrasts dictionaries of different size. Draw random
16,384-feature subsets of the 131k dictionary (matching the narrow dictionary's width), recompute
mean generality over whichever target-active content features land in the subset, 4,000 draws.

| corpus | observed 16k | wide full | size-matched null | z | P(null ≥ observed) |
|---|---|---|---|---|---|
| narrative | 0.182 | 0.063 | 0.063 ± 0.008 (≈215 content feats/subset) | 14.81 | 0.0000 |
| broad | 0.187 | 0.101 | 0.101 ± 0.012 (≈247 content feats/subset) | 7.07 | 0.0000 |

**Survives.** A size-matched random subset reproduces the wide dictionary's mean exactly, not the
narrow one's, so the effect is about what narrowing *preserves* rather than about dictionary size.

## Null 3 — the flow ordering, by label permutation

**Construction.** Rebuild the threshold reconstructions as in `sae_ladder_v2.py` part C, then permute
theme labels (preserving the 8×9 class structure) and era labels (preserving 3×24) independently at
each threshold, many repetitions, giving each label its own null band for 8-NN purity.

| threshold g_k | theme 8-NN | theme null | z | era 8-NN | era null | z |
|---|---|---|---|---|---|---|
| ≥ 0.0 | 0.628 | 0.112 ± 0.017 | 30.2 | 0.535 | 0.324 ± 0.025 | 8.29 |
| ≥ 0.01 | 0.628 | 0.113 ± 0.017 | 30.7 | 0.535 | 0.325 ± 0.025 | 8.26 |
| ≥ 0.02 | 0.628 | 0.112 ± 0.016 | 31.5 | 0.533 | 0.324 ± 0.026 | 8.14 |
| ≥ 0.05 | 0.630 | 0.112 ± 0.017 | 31.1 | 0.535 | 0.325 ± 0.025 | 8.50 |
| ≥ 0.1 | 0.627 | 0.112 ± 0.017 | 30.3 | 0.528 | 0.323 ± 0.025 | 8.31 |

**Does not support hour 26's ordering claim.** Neither label falls into its own permutation band at
any threshold tested: the run reports the entry threshold as `None` for both theme and era. Hour 26
said "the flow is an ordering (era dies before theme)". What is true here is that theme is far more
robustly structured than era (z ≈ 30 vs ≈ 8), which is *consistent* with that story, but **neither
dies in the tested range, so the ordering is untested rather than confirmed.** Testing it needs
thresholds above 0.1, where one of them actually reaches its null.

## Verdict

Writeup claim 6 (abstraction as a quotient with a measurable scale) **survives both nulls that bear
on it**, and the merge result is stronger against the correct null than it looked against an implied
coin flip. Hour 26's sub-claim that the flow is an *ordering* is **downgraded to unconfirmed**.

## What is still confounded

- Generality is measured against a narrative-only reference set for the narrative corpus; the broad
  corpus mitigates but does not remove this.
- The 15 features are the strongest wide content features, not a random sample; the null is matched
  to the search population but not to the selection of the 15.
- The re-derived merge count (13/15) differs by one from the logged 12/15.
- Nulls 1 and 2 were run at layer 20 only, matching hours 15 and 26.
