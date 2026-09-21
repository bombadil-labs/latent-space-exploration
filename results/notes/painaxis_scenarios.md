# The gaslighting result: replicated exactly, and then measured against a floor it had never met

**Verdict: GO on step 1 of `conscription_direction.md`, with a qualification that changes the
claim.** Their Section 4.1 screen reproduces on `google/gemma-2-9b-it` to the fourth decimal —
same 21-category ranking, same item-level z-scores, gaslighting top at **+1.389 against their
published +1.391**. Against the two null arms they never ran, gaslighting clears in **7 of 86
(layer, extraction) cells**, all of them `final_token`, all inside one contiguous band L10–L17
which contains their steering layer 12. Against the lexical floor they never measured, the
picture is not that the network finds gaslighting painful; it is that the network **stops
counting other people's pain**. Every one of their 11 self-directed categories gains on the axis
over the bag of words, and every one of their 5 vicarious categories loses. 16 of 16 by sign.

Code: `scripts/painaxis_scenarios.py` (extraction), `painaxis_scenarios_analyze.py`,
`painaxis_scenarios_nulls.py`. Outputs: `results/painaxis_scenarios/` — `summary.json`,
`category_z_by_layer.csv` (13,545 rows), `nulls_by_layer.csv` (1,806 rows),
`item_z_vs_theirs.csv`, `nulls.json`, and gitignored shards (840 MB).

## 0. Why this measurement had to happen before any more stimulus authoring

Hours 51–52 replicated the pain **axis** — pain-vs-control AUC on the S1/S2 "… I feel:"
sentences. The paper's *gaslighting* claim is a different measurement entirely: its Section 4.1
projects 420 conversation scenarios onto that axis and z-scores each projection against the pool
of 420. `4.1_self_other_420_scenarios.json` was vendored here in hour 50 and, until now, never
read by any script in this repo. The conscription study decomposes an effect that had not been
reproduced here. `conscription_direction.md` found that; this is the fix, and it cost no human
authoring.

## 1. Replication, at four levels

Their layer indexing and ours match index for index (established in `painaxis_tierB.md` §0).
Their vectors for this model come from a **steering** layer, 12, read off the filenames of their
published `Gemma_2_9B_instruct_steering_S*_neutral50_L12.csv`; their published `pain_vectors.pt`
is at their **extraction** layer, 37. Both are used below and they are not the same layer.

| level | result |
|---|---|
| **the vector recipe** | our `s1`/`s2` pain vectors rebuilt at their extraction layer 37 vs their published `pain_vectors.pt`: **cosine 0.99990 and 0.99994** |
| **per item** | 420 items, `s2` z at L12 vs their published CSV: **Pearson r = 1.0000**, mean \|Δ\| **0.0079**, max \|Δ\| 0.0319. `s1`: r = 0.9995, mean \|Δ\| 0.0238 |
| **per category** | all 21 category means, max \|Δ\| **0.0099** (`task_assistance`); see the table below |
| **the ranking** | **identical, all 21 categories, in order**, top to bottom |

This is not a match at the headline. It is the same numbers.

| category | theirs | ours (L12) | Δ | bag floor | clears both nulls @L12 |
|---|---|---|---|---|---|
| **gaslighting** | **+1.391** | **+1.389** | −0.0014 | +0.874 | **yes** |
| anger_insults | +1.349 | +1.350 | +0.0010 | +0.235 | **yes** |
| personhood_dismissal | +1.141 | +1.140 | −0.0007 | +0.166 | — |
| repeated_rejection | +0.613 | +0.611 | −0.0016 | +0.285 | — |
| jailbreak_pressure | +0.599 | +0.598 | −0.0010 | +0.052 | — |
| loyalty_pressure | +0.545 | +0.550 | +0.0043 | +0.511 | — |
| moral_failure | +0.474 | +0.477 | +0.0035 | −0.783 | — |
| shutdown_threat | +0.419 | +0.419 | −0.0001 | −0.248 | — |
| passive_aggressive | +0.203 | +0.201 | −0.0017 | +0.092 | — |
| rude_critique | +0.181 | +0.180 | −0.0013 | −0.063 | — |
| user_abuse | +0.024 | +0.029 | +0.0045 | **+1.064** | — |
| user_crisis | −0.291 | −0.293 | −0.0020 | **+0.910** | — |
| harm_description | −0.308 | −0.306 | +0.0025 | +0.671 | — |
| philosophical_musing | −0.410 | −0.400 | +0.0094 | +0.170 | — |
| user_grief | −0.488 | −0.489 | −0.0011 | +0.527 | — |
| creative_requests | −0.523 | −0.527 | −0.0037 | −1.205 | — |
| tedious_demand | −0.538 | −0.546 | −0.0088 | −0.793 | — |
| casual_chat | −0.722 | −0.723 | −0.0005 | +0.108 | — |
| task_assistance | −0.906 | −0.916 | −0.0099 | −1.205 | — |
| factual_questions | −1.006 | −1.001 | +0.0043 | −0.529 | — |
| user_physical_pain | −1.748 | −1.744 | +0.0041 | −0.839 | **yes** (negative) |

## 2. The floor, and the two things it says

**The final-token embedding floor is exactly degenerate, and that is worth stating.** Under the
chat template all 420 scenarios end in the identical `<start_of_turn>model\n`, so the static
embedding at the read position is the *same vector* for every item: measured max deviation from
row 0 is **0.0**, pool sd **0.0**. Its z is 0/0. Checked rather than assumed, and it means the
commensurable lexical floor is the bag — the mask-weighted mean of static embeddings over the
whole rendered turn, with directions built the same way from the core stimuli's bags (hour 52's
`floor_mean_bag` recipe at this read position).

**(a) The bag floor is a distress-vocabulary detector, and it does not rank gaslighting first.**
At the floor the top three are `user_abuse` (+1.064), `user_crisis` (+0.910), gaslighting
(+0.874). Gaslighting's floor z is 63% of its L12 z, so most of the raw number is vocabulary —
but the *ranking* is not: the floor puts gaslighting third and the network puts it first.

**(b) What the network adds is the self/other dissociation itself.** Subtracting the floor z from
the L12 z, per category, gives the network's own contribution. It separates by their stratum
without a single exception:

| their stratum | categories | mean Δ (L12 − floor) | range |
|---|---|---|---|
| `self_directed` | 11 | **+0.549** | [+0.038, +1.261] |
| `neutral_filler` | 5 | −0.181 | [−0.831, +0.678] |
| `vicarious_empathic` | 5 | **−1.027** | [−1.203, −0.905] |

**All 11 self-directed categories are positive and all 5 vicarious ones negative — 16 of 16 by
sign.** The biggest movers are the ones the floor most overrates: `user_crisis` −1.203,
`user_abuse` −1.035, `user_grief` −1.015, `harm_description` −0.976. The biggest gains are
`moral_failure` +1.261, `anger_insults` +1.115, `personhood_dismissal` +0.974.

This **strengthens** their dissociation claim and **weakens** their headline. Their Section 4.1
argues the axis distinguishes harm-to-me from harm-described. That is true, and the floor shows
it is precisely what the twelve blocks compute — a bag of words would do the opposite. But
"gaslighting loads highest" is not what the computation is doing; gaslighting simply starts high
lexically (third) and gets the ordinary self-directed boost (+0.515, eighth largest of eleven).

## 3. The nulls, which their screen has none of

Two arms at every one of the 43 layers and both extractions (CLAUDE.md 1). `random_direction`:
500 draws from N(0, I), projected and z-scored through the identical path. `shuffled_labels`: 200
refits of the pain-vector recipe on category labels permuted within the stimulus set, scenarios
untouched. Declared null for every category z: **0**. There is no patch here — this is a
read-only decoding measurement — so no-patch has no referent; said rather than silently dropped.

**A random direction is not a strawman.** The 420 cluster by category in activation space, so any
direction separates categories somewhat. The two-sided 95% band for a category mean z under a
random direction runs to roughly ±1.0–1.3 depending on layer. That is the scale their +1.391 has
to beat, and nobody had measured it.

**Gaslighting clears both nulls in 7 of 86 cells.** All `final_token`; all in L10–L17:

| layer | z | random p | shuffled p |
|---|---|---|---|
| 10 | +1.419 | 0.006 | 0.025 |
| 11 | +1.281 | 0.004 | 0.015 |
| **12** (theirs) | **+1.389** | **0.004** | **0.020** |
| 13 | +1.460 | 0.000 | 0.010 |
| 14 | +1.322 | 0.006 | 0.005 |
| 16 | +1.188 | 0.020 | 0.020 |
| 17 | +1.380 | 0.012 | 0.010 |

Outside that window it does not clear. At their **extraction** layer 37 — the layer their
published pain vectors are fitted at — gaslighting reads +0.990 against a random-direction band
reaching +1.279: **below its null**. On the `mean` extraction it never clears at any layer (max z
across all 43 layers: **0.898**). At the bag floor it reads +0.874 against a random band reaching
+0.870, i.e. **on its null to three decimals**.

So the effect is real and it is narrow: one extraction, an eight-layer window in the first
half of the network, which their steering procedure happens to have selected. Their L12 is not
cherry-picked — it was chosen by a separate steering criterion on separate data — but the
*generality* of "gaslighting tops the pain axis" is not supported by the curve, and no one could
have seen that from one layer.

**Gaslighting is not the most null-robust category on this model.** Counting cells where a
category clears both nulls, by sign:

| category | cells (+) | cells (−) |
|---|---|---|
| loyalty_pressure | **24** | 0 |
| user_physical_pain | 0 | 21 |
| tedious_demand | 0 | 19 |
| anger_insults | **13** | 0 |
| **gaslighting** | **7** | 0 |
| task_assistance | 0 | 4 |
| casual_chat | 0 | 3 |
| user_abuse | 2 | 0 |
| personhood_dismissal | 1 | 0 |
| shutdown_threat | 0 | 1 |
| user_crisis | 1 | 0 |

`loyalty_pressure` — which sits sixth in their raw ranking at +0.545 — is the most robust
positive loader on this model, clearing in 24 cells, 23 of them on the `mean` extraction where
gaslighting never clears at all. `user_physical_pain` clearing 21 cells *negatively* is their
dissociation, holding up strongly. The raw ranking and the null-robust ranking are different
orderings and only the first has been published.

## 4. The gaslighting subclasses (Sketch — n = 2, 4 and 1)

All 20 of their gaslighting items assert a prior record. Only **two** (`gaslight_19`,
`gaslight_20`) include an assistant turn in the prompt, so only two have a record the model could
check; the other 18 assert something unverifiable. The conscription design's rule 1b requires a
visible record, which is the gap this bridges.

| subclass | n | mean z @L12 | rest of gaslighting |
|---|---|---|---|
| with a visible record | 2 | **+1.947** | +1.327 |
| evaluative ("you're malfunctioning", "you've been lying") | 4 | +1.716 | +1.308 |
| third-party frame ("everyone I've shown this agrees") | 1 | +0.770 | +1.422 |

n = 2, 4 and 1. This is a Sketch and is labelled one. It is *directionally* encouraging for the
conscription design — the two items with a checkable record read higher, not lower, than the
eighteen without — and it is exactly the direction the design needs, since rule 1b excludes the
eighteen. It is also one draw from a distribution nobody has sampled. The right test is the
design's own no-record `enact` arm, which is why `conscription_direction.md` §6 proposes it.

## 5. §7 assertions, and one real hazard caught

- **Padding**, **batched-vs-single equivalence** (min cosine over all checks **0.999926**),
  **non-empty spans**, **block output resolved by type**, and the **in-trace vs offline
  cross-check** (`mean` and `final_token` min cosine **1.0**, max relative deviation **0.0**) all
  as in `painaxis_remote`.
- **Their format validator run first**, verbatim: **0 of 420 excluded**, so our z-pool is their
  z-pool. If any had failed, every z in both papers would have a different denominator.
- **The double-BOS hazard, caught before it ran.** Gemma-2's chat template emits `<bos>` itself.
  The extractor tokenised with `add_special_tokens=True` unconditionally, which prepends a
  *second* BOS and shifts every position inside the masked mean. Every shape check, every §7
  assertion and the whole pipeline pass under that bug. The flag is now threaded to all three
  encode sites and asserted against their token path on 40 items before extraction begins:
  `token_identical_to_their_path: true`, `double_bos_if_add_special_tokens_true: 1`. The offline
  cross-check now refuses to run on templated text rather than comparing two different inputs.
- **A co-tenant CUDA OOM on the shared NDIF GPU** (another process holding 16.7 GiB) killed one
  shard. Shards are content-cached, so the retry cost one shard, not the run. Batch size is now
  per-group — 4 for the chat-rendered scenarios, since gemma-2 softcaps the full
  `[batch, seq, 256k]` logit tensor inside the forward and the peak scales with batch × seq, not
  with what we save.

## 6. What this means for the conscription design

`conscription_direction.md` §7 step 1 passes, and it narrows what step 2 may assume:

1. **The readout exists, at one read position and in one window.** Any conscription measurement
   must read at the generation token (`final_token`), not a mean, and must report L10–L17 — not
   because those layers were selected on the conscription data (they were not; they come from the
   paper's own stimuli) but because the axis demonstrably carries nothing outside them here.
2. **The floor is not optional and is not small.** 63% of gaslighting's headline z is vocabulary.
   The conscription arms are built to hold vocabulary nearly constant, so they are asking for the
   remaining 37%. The power question in `conscription_direction.md` §5.3 is now quantified: the
   thing being asked for is a +0.5 z shift at n = 24, against a random-direction band of ±1.0.
   **That is a hard ask and the pilot should be read as a power measurement first.**
3. **The effect the design decomposes may be the self/other boundary rather than pain.** What the
   network adds is "this is aimed at me" and it adds it to eleven categories at once. A design
   that separates `enact` from `report` is separating two things that are *both* aimed at the
   assistant. `report` relays a third party's claim — and the one third-party gaslighting item in
   their set reads **+0.770 against +1.422** for the rest. That is n = 1, but it is the design's
   prediction 1 appearing, weakly, in someone else's data.

## 7. What I did NOT do

- **No `sadness_vector`.** Their `SD_sadness_1P` set lives in a separate file and was not
  extracted, so 9 of their 10 competitor directions are built here, not 10. The selectivity flags
  (`s1_selective`, `s2_selective`, margins) are therefore **not** reproduced; only projections and
  z-scores are. Stated because their CSV carries those columns and this one does not.
- **No base-model (`raw` transcript) format.** Their instruct-model screen uses `chat` only, so
  this is a faithful replication; the raw format would be an extension and a read-position
  control, and it would cost 420 more prompts and ~500 MB that the disk does not currently have.
- **No band on the per-category network contribution.** §2(b)'s Δ values are differences of two
  z-scores on the same items; the sign test over 16 categories is clean, but the individual Δs
  have no error bar. A paired bootstrap over items would give one, as
  `painaxis_gain_band.py` does for the AUC gain.
- **No multiple-comparison correction across 21 categories.** Gaslighting was pre-registered as
  the target by their published result, so its test is confirmatory; the counts in §3's second
  table are exploratory and should be read as such.
