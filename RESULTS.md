# Results log

Honest running record. Numbers are from the scripts in `scripts/`, JSON in `results/`. Negative
and null results are kept. Models: Qwen2.5-0.5B (24 layers, d=896), Qwen2.5-1.5B (28 layers,
d=1536), CPU, float32. Grid: `prompts/holonic_v1.json` (8 domains x {holonic, flat}, 6 roles).

## 2026-09-11 — Stage 2: does a cross-domain relational shape exist above chance?

**Setup.** Each prompt gives a 6-role x d stack per layer. Grand mean across all 16 prompts is
subtracted per layer. For each pair of prompts, shape similarity (linear CKA on the 6x6 Gram, or
Spearman RSA on the 6x6 cosine RDM) is compared against a 200-permutation null in which one prompt's
roles are scrambled. z = (obs - null mean)/null sd. Buckets: **H** holonic/holonic across domains
(hypothesis), **F** flat/flat across domains (template control: flat prompts share connectives and
length), **A** same domain across framings (address control), **X** everything else (floor).

**Qwen2.5-1.5B, RSA (Spearman on role RDMs).**

| layer | H obs | H z | F obs | F z | A obs | A z | X obs | X z |
|---|---|---|---|---|---|---|---|---|
| 0  | 0.15 | 0.55 | 0.08 | 0.32 | -0.01 | -0.08 | 0.10 | 0.40 |
| 10 | 0.39 | 1.35 | 0.06 | 0.14 | 0.18 | 0.63 | 0.21 | 0.73 |
| 16 | 0.48 | 1.66 | 0.18 | 0.57 | 0.24 | 0.81 | 0.28 | 0.92 |
| 20 | **0.60** | **2.03** | 0.21 | 0.71 | 0.36 | 1.27 | 0.36 | 1.20 |
| 22 | 0.60 | 2.00 | 0.29 | 0.97 | 0.32 | 1.03 | 0.40 | 1.32 |
| 28 | 0.54 | 1.80 | 0.26 | 0.85 | 0.21 | 0.75 | 0.30 | 0.98 |

**Qwen2.5-1.5B, linear CKA:** H z rises from 0.7 (layer 0) to 2.3 (layer 20); F, A, X stay
0.1–1.3. Raw CKA is 0.85–0.95 everywhere (ceiling effect; RSA is the more legible metric here).
**Qwen2.5-0.5B** shows the same pattern with H z ≈ 2.1–2.3 at layers 16–24, F/A/X ≈ 0.8–1.2.
Full tables: `results/stage2_*.json`.

**Reading.** The holonic prompts' role structure agrees across eight unrelated domains
(RSA 0.6 at layer 20) far more than flat prompts with the same connective template do (0.2), and
more than same-domain pairs across framings (0.36). The effect is absent at the embedding layer and
grows through the stack, peaking around layers 16–22 of 28. That is the predicted signature: the
shared shape is built by the layers, not present in the tokens, and it is a property of the
*relation* framing rather than of the domain or the sentence template.

**Caveats.** n = 8 domains, one prompt each, one author (Claude). z ≈ 2 per pair, averaged over 28
pairs, is a consistent effect but not a large one. Flat prompts are shorter (~530 vs ~610 chars).
Six roles is a small shape. Not yet tested: shuffled-holonic control (same spans, permuted into
other slots) to separate content-in-slot from slot-position; multiple paraphrases per domain;
a non-Qwen model.

## 2026-09-11 — Stage 3: can an operator carry a role→role relation to a held-out domain?

**Setup.** For each ordered role pair and layer, fit on 7 domains' holonic prompts, predict the 8th
domain's dst vector from its src vector. Fits: identity + rank-4 affine correction (ridge 1.0, dual
form). Baselines: identity (pred = src), mean of training targets, and a single shared offset
(pred = src + mean(dst − src), the king/queen construction). Metric that matters: **role_rank** =
where the true dst role lands among the held-out prompt's own 6 role vectors by cosine to the
prediction (1 = best, chance = 3.5). Ranking against other domains is trivial (always 1) because
the domain address dominates, so it is not reported.

**Qwen2.5-1.5B, mean role_rank (lower is better).**

| | affine (rank 4) | shared offset | identity |
|---|---|---|---|
| all pairs, all layers | 3.03 | **2.35** | 4.00 |
| all pairs, each pair's best layer | 2.45 | **2.01** | 3.35 |

Best layers are almost always the last few (26–28). Easiest pairs: from_below↔from_above
(rank ≈ 2.0), objectified↔new_subject. Hardest targets: disturbance, embedded (rank 4+).

**Reading.** A relation *is* transferable across domains: a single translation vector learned on
seven domains moves an eighth domain's source role toward its target role better than chance and
much better than staying put. But the low-rank affine map is *worse* than the plain offset with
this little data. Seven examples in 1536 dimensions cannot support a rotation; the relation, at
this scale, is best modeled as a shared direction. This matches the linear-representation literature
and is a negative result for "the operator needs to be a map" at n = 7.

**Caveats.** Role vectors are mean-pooled spans of ~15 tokens; rank among 6 candidates is a coarse
test; the offset baseline benefits from being the lowest-variance estimator. Next: more prompts per
domain (paraphrases) so the affine fit has data; readout by patching the predicted vector into the
residual stream and generating (stage 4), which is the only test that matters for the "lens".

## 2026-09-11 (hour 2) — Shuffled-holonic control: the stage-2 "shape" is mostly slot position

**Setup.** For each holonic prompt, keep the connective template and all six spans, but permute
spans across slots with a different derangement per domain (`scripts/make_shuffled.py`, perms
saved in `prompts/holonic_v1_shuffled.json`). Stacks can then be labeled by SLOT (position in the
template) or, using the saved permutation, by CONTENT (which original role the span was). Because
each domain uses a different derangement, position and content are decorrelated across domains.

**Qwen2.5-1.5B, RSA, cross-domain, layer 20 (peak).**

| pairing | RSA | z |
|---|---|---|
| holonic~holonic (content and slot aligned) | 0.60 | 2.0 |
| shuffled~shuffled labeled by SLOT | **0.52** | 1.9 |
| shuffled~shuffled labeled by CONTENT | **0.05** | 0.2 |
| holonic~shuffled labeled by CONTENT | 0.05 | 0.2 |
| prompt vs its own scrambled twin, by CONTENT | 0.34 (1.00 at layer 0) | |

Full sweep: `results/stage2_qwen1.5b_rsa_content.json`, `results/stage2_qwen1.5b_rsa_shuffled.json`.

**Reading.** Almost all of the cross-domain agreement measured in stage 2 is explained by *which
slot of the template a span sits in*. Relabel by content and the agreement is at chance from layer
8 onward. The same span's vector, compared with itself moved to another slot, drifts from identical
(layer 0) to RSA 0.34 by layer 20: the residual stream increasingly encodes discourse position and
decreasingly encodes what the span says, at least in the mean-pooled, grand-mean-centered view.
The earlier holonic-vs-flat gap is therefore not evidence for a content-level relational shape; the
flat prompts also used slightly different connectives, so that gap partly measured template.
**This is a negative result for the stage-2 claim as stated**, and the reason is mechanistic and
clean: a 6-point RSA over pooled spans is a position detector.

**Stage 3 re-run under the same control (shared-offset transfer, mean role_rank, chance 3.5).**

| data | labels | offset, all layers | offset, best layer | identity |
|---|---|---|---|---|
| holonic | content = slot | 2.35 | 2.01 | 4.0 |
| shuffled | CONTENT (position scrambled) | **2.87** | 2.40 | 4.0 |
| shuffled | SLOT (content scrambled) | 3.34 | 2.86 | 4.0 |
| flat | slot | 3.58 | 3.03 | 4.0 |

**Reading.** The directional test sees what the shape test could not. With position scrambled and
roles labeled by content, a single offset learned on seven domains still moves the eighth domain's
source role toward the correct target role (2.9 vs 3.5 chance), and does so *better* than the
slot-labeled version (3.3) or flat prompts (3.6). Content-role relations exist in the residual
stream as transferable directions, at modest strength; position and content add when aligned
(2.35). "Best layer" is selected on held-out performance and is optimistic; the all-layers mean is
the honest number.

**What this changes.** The claim to carry forward is not "prompts sharing a relation produce the
same shape" but "role-to-role relations are shared directions across domains, and they are small
compared with discourse-position structure." Stage 2 needs a measurement that is not dominated by
position: (a) many paraphrases per domain with each content role rotated through every slot, so
position averages out; (b) projecting out slot directions (estimated from the shuffled set, where
slot is known and content is decorrelated) before RSA; (c) token-level clouds rather than 6 pooled
points. Stage 4 readout can proceed on the content-offset directions, which are the real finding.

**Addendum: projecting out the slot subspace does not recover a content shape.** Removing the
5-dim span of per-slot mean vectors (leave-pair-out, estimated on the shuffled set) cuts
slot-labeled cross-domain RSA from 0.52 to 0.23 at layer 20 (0.38 at layer 28) and lifts
content-labeled RSA only from 0.05 to 0.13. Position is not confined to a small linear subspace,
and the 6-point pooled RSA remains blind to content. `scripts/stage2_deslot.py`,
`results/stage2_qwen1.5b_deslot.json`. Conclusion: fix the design (rotate content through slots
across paraphrases), not the post-processing.

## 2026-09-11 (hour 3) — Position-balanced grid: role identity is a linear signature; a genuine relation signal survives after removing it

**Setup.** `scripts/make_rotated.py`: each domain's six holonic spans presented with role-neutral
connectives ("First, … Second, …") in all six cyclic rotations, so every content role sits in every
slot once per domain. 48 prompts, roles labeled by content. Extracted on Qwen2.5-1.5B
(`results/stacks_qwen2.5_1.5b_holonic_v1_rotated.npz`).

**Stage 2 on the rotated grid (RSA, cross-domain, layer 20).** Same-rotation pairs (position and
content aligned) 0.56; different-rotation pairs labeled by slot (position only) 0.49; different
rotation labeled by content (content only) 0.17; per-domain average over rotations (position
balanced) 0.22. Content-only RSA at layer 0 is already 0.15. **The six-point pooled RSA is a
position detector; the layers add almost no content shape beyond embedding-level lexical
similarity.** `scripts/stage2_rotated.py`, `results/stage2_qwen1.5b_rotated.json`.

**Stage 3, and a correction to hours 1–2.** With 42 training examples the full affine map
(ridge 10) scored role_rank 2.0 and, at the best layer, 1.08. But the *constant* mean-target
baseline scores **1.07** on its own. A domain-independent "which role is this" direction exists in
the residual stream, and every stage-3 number reported before was that signature (plus position),
not a source-to-target relation. Also, "best layer" selection on held-out data is badly optimistic:
the constant baseline goes from 3.5 to 2.4 by selection alone. Best-layer numbers are dropped from
here on.

**Role-centered relation test.** Subtract each role's cross-domain mean (training folds only) from
every vector and candidate, so role identity is gone and only domain-specific content per role
remains. Then: does the source role's residual predict the target role's residual in a held-out
domain? Null: shuffle the source/target pairing among training rows within each fold.

| | role_rank, all layers (chance 3.5) |
|---|---|
| full affine, ridge 10, role-centered | **3.01** |
| null, 3 seeds | 3.48, 3.54, 3.54 |
| constant baseline (role-centered) | 3.52 |

Per layer: 3.68 at layer 0, 3.20 at 4, 3.00 at 8, 2.77 at 14, **2.58 at 20**, 3.22 at 28. Predicted-
target cosine rises from 0.00 (layer 0) to 0.19 (layers 18–28). Easiest relations:
objectified→disturbance (1.9), disturbance→embedded (2.1), embedded→from_above (2.3). Hardest:
anything →new_subject or →from_below (3.6–3.8). `results/stage3_qwen1.5b_rotated_rolecentered.json`.

**Reading.** After removing position (balanced by design), role identity (centered), and domain
address (candidates are from the same prompt), an affine map fit on seven domains still moves a
held-out domain's source-role content toward its target-role content, above a matched null. The
signal is absent at the embedding layer and peaks around layer 20 of 28, so it is computed by the
model, not inherited from token overlap. It is **small** (rank 3.0 vs 3.5; cosine 0.19), and the
effective sample is seven domains because rotations reuse spans. This is the first number in this
log that supports the original hypothesis as stated, and it supports a modest version of it.

**Three findings now stand, in decreasing size.** (1) Discourse position is the dominant structure
in pooled span vectors. (2) Role identity within the transition schema is a domain-independent
linear direction, readable regardless of position. (3) A residual source→target relation transfers
across domains, weakly, in mid-late layers.

## 2026-09-11 (hour 4) — Stage 4: the role-identity direction works as a lens

**Setup.** Leave one domain out. Role direction dir_R = mean over training domains of role R's
position-balanced vector minus the mean of all roles, at layer 20 (residual index 20, input to
block 20). Add s·dir_R at every position during a teacher-forced pass over
`"<lead> First, <span_r>."` for each of the held-out domain's six spans. gain_r = log p(span_r | +dir_R)
− log p(span_r | base). If dir_R is a lens, gain is largest for r = R. Control: random directions of
the same norm (two per condition). ‖dir_R‖ is 18–19% of the mean residual norm at layer 20.
`scripts/stage4.py`, `results/stage4_qwen1.5b_roledir_l20.json`.

**Qwen2.5-1.5B, rank of the target role's span by log-prob gain (1 = best, chance 3.5).**

| scale | role direction (n=48) | random direction (n=96) | mean gain, target span | mean gain, other spans |
|---|---|---|---|---|
| 0.5 | **1.65** | 3.39 | +0.67 nats | −0.26 |
| 1.0 | **1.69** | 3.25 | +1.10 nats | −0.76 |

Per domain at scale 1.0: software 1.33, law 1.33, music 1.33, narrative 1.33, biology 1.50,
physics 1.67, psychology 2.33, mathematics 2.67. Random 2.7–4.0 in every domain. A pilot at scale
4.0 degraded everything (target gain −3.2, others −11), so the working range is roughly 0.5–2×.

**Reading.** A direction estimated from seven domains, added to the residual stream in an eighth
domain the estimate never saw, raises the likelihood of exactly that role's span and lowers the
others. Equal-norm random directions do nothing. This is a held-out, controlled demonstration that
the role directions from hour 3 are causally usable, not just decodable. It is the first "lens"
result: a shape learned elsewhere, pointed at a new domain, selects the right part.

**Qualitative generations** (`scripts/stage4_generate.py`, greedy, 1.5B base model) are much
weaker than the numbers. Shifts are visible but subtle: +from_above pulls continuations into
retrospective past tense ("the story was about… what was the change?"), matching how those spans
were written; +new_subject pulls toward relationships between parts ("what is the relationship
between these two characters?"); +from_below toward evaluative confusion ("bad guy… good guy…
what happened?"). A 1.5B base model's greedy continuations are not clean enough to call this more
than suggestive. Treat the log-prob readout as the result and the generations as illustration.

**Caveats.** Single layer, single model, eight domains, spans written by one author with some
shared phrasing per role (e.g. from_below spans tend to start "this looks like"), which the role
direction may partly encode as style. The controlled comparison is still valid (random directions
share none of it), but "role" here includes register as well as meaning. Not yet done: multi-layer
patching, a layer sweep, and the relation lens (patch the affine-predicted target, finding 3).

## 2026-09-11 (hour 5) — Relation lens: null. Role lens: peaks at layers 14–20

**Relation lens (stage 4b).** For held-out domain d and pair S→T: fit the role-centered affine map
on the other 7 domains × 6 rotations (ridge 10, layer 20), predict d's target-content residual from
its source residual, and patch `dir_T + pred` versus `dir_T` alone. Controls: `dir_T + random`
(same norm as pred, ×2) and `dir_T + pred_from_wrong_source` (map applied to a different role's
residual, rescaled). Metric: extra log-prob gain on the target span beyond the role-only patch.
‖pred‖ ≈ 1.06 ‖dir_T‖. Six pairs × 8 domains. `scripts/stage4_relation.py`,
`results/stage4b_qwen1.5b_relation_l20.json`.

| patch added to dir_T | extra gain on target span (nats) | rank of target (role-only: 1.85) |
|---|---|---|
| affine-predicted content | **−0.02** (n=48) | 2.15 |
| random, same norm | −0.19 (n=96) | 2.05 |
| prediction from wrong source | −0.49 (n=48) | 2.40 |

Per domain the relation patch ranges from +0.98 (biology) and +0.90 (psychology) to −1.03 (law)
and −0.68 (software).

**Reading.** Adding the predicted relation content does not improve the target span's likelihood on
average. It is *less harmful* than a random vector of the same norm and much less harmful than a
prediction from the wrong source, so the predicted direction is partially aligned with the true
content, consistent with the weak stage-3 signal (rank 3.0 vs 3.5). But it is not a usable lens at
this data size: a perturbation the size of the role direction that is only weakly aligned costs
more than it gains. **Negative result.** The relation exists as a measurable direction (hour 3);
it does not yet exist as a controllable one. What would change this: more domains and real
paraphrases (the map is fit on 7 effective examples), a smaller λ with a sweep, or fitting the map
at the layer where stage-3 transfer peaked rather than the layer where the role lens works.

**Role-lens layer sweep** (4 domains: law, music, software, biology; scale 1; one random control).

| layer | role-dir rank | random rank | target gain | other-span gain |
|---|---|---|---|---|
| 8  | 2.08 | 3.12 | +0.75 | −0.27 |
| 14 | **1.33** | 3.58 | **+1.57** | −0.61 |
| 20 | 1.37 | 3.3 | +1.10 | −0.76 |
| 26 | 2.67 | 3.17 | −0.04 | −0.46 |

The role lens works from the middle of the stack, is strongest around layer 14–20 of 28, and fades
in the last layers where the residual is being turned into next-token logits. Layer 14 gives a
larger target gain with less collateral damage than layer 20; later runs should use it.

## 2026-09-11 (hour 6) — Narrative factors: era and voice are directions, they compose, and order barely matters

**Grid.** `prompts/narrative_factors_v1.json`: 4 scenes (gate, theft, farewell, storm) × 3 eras
(medieval, 1920s, far future) × 3 voices (terse, ornate, childlike) = 36 spans, each rendering the
same scene event in one era and one voice. One prompt per span (`"A moment from a story: <span>"`),
pooled over span tokens. Directions are leave-one-scene-out: dir_era[e] = mean(era e) − grand
mean over the other three scenes; same for voice. `scripts/extract_factors.py`,
`scripts/stage5_factors.py`, `results/stage5_qwen1.5b_factors_l14.json`.

**(A) Decodability, no model needed** (nearest factor direction, held-out scene; chance 0.33).
Voice: 0.92 at layer 0, ~0.92 throughout. Era: **0.28 at layer 0**, 0.64 at 2, 0.89 at 10,
**0.94 at 12**, ~0.9 after. Voice is lexical (sentence length, word simplicity) and present in
the embeddings; era is computed by the stack and becomes linearly readable around layer 10–12.

**(B) Each factor as a lens** (layer 14, scale 1; rank of the patched level's span among its 3
variants, chance 2.0). Era direction **1.28**, random 2.11. Voice direction **1.31**, random 2.11.
Both work on a scene the direction never saw.

**(D) Composition.** +dir_era[e] + dir_voice[v] as one patch: the (e, v) span ranks **2.03 of 9**
(chance 5.0). Two factor directions learned separately add up to select the joint span.

**(E) Order.** Era at layer 14 + voice at layer 20: rank 1.83. Voice at 14 + era at 20: 1.89.
Mean absolute rank gap 0.50; correlation of the two 9-span gain profiles **0.85**. Within this
pair of layers the factors nearly commute. The holonomy we speculated about is small here.

**(C) Cross-talk: the first metric was broken.** Ranking each voice variant among the three voice
variants averages to exactly 2 by construction, so the "readout under patch" numbers in the log are
meaningless. Replaced by a variance decomposition of the 3×3 gain matrix under a single-factor
patch (fraction explained by the on-target factor vs the other factor vs residual);
`scripts/stage5_crosstalk.py`, `results/stage5_qwen1.5b_crosstalk_l14.json`.

| patch (layer 14, scale 1) | on-target factor | other factor (cross-talk) | residual | on-target rank/3 |
|---|---|---|---|---|
| era direction | **0.59** | 0.14 | 0.27 | 1.08 |
| voice direction | **0.65** | 0.17 | 0.18 | 1.08 |
| random, same norm | 0.27 | 0.23 | 0.50 | 1.96 |

Under an era patch, 59% of the variance in span gains is an era main effect and 14% is a voice
main effect; under a voice patch, 65% and 17%. A random patch spreads its variance evenly (27/23)
with half in residual. Cross-talk is real but small, about a quarter of the on-target effect, and
close to what a random direction produces. The factors are not orthogonal, but they are close.

**Reading.** For these two narrative factors the additive picture is close to right: each is a
direction, both transfer to a held-out scene, their sum selects the joint variant, and the order of
application across two layers changes the outcome little. This is the substrate the narrative
calculus needs, at the level of single sentences and two factors. Caveats: one author, 36 spans,
strong lexical confound for voice, single model, and "era" is partly vocabulary (Packard, airlock)
which is exactly what a setting shift should carry but means the era direction is not purely
abstract. Plot shape, the hard factor, is untouched.

## Open problems (ordered)

1. ~~Shuffled-holonic control~~ done: stage-2 shape is mostly slot position; content-role offsets survive.
2. ~~Project out slot directions~~ done: partial removal, no content shape recovered.
3. ~~Rotate content through slots~~ done. Real paraphrases (new wording per domain) would raise the
   effective n above seven and are the main lever for both stage 3 and the relation lens.
4. ~~Stage 4 role lens~~ done: rank 1.7 vs 3.3 random, all eight held-out domains.
4b. ~~Relation lens~~ done: null (−0.02 nats vs −0.19 random); the relation is measurable, not yet controllable.
4c. ~~Narrative factors era × voice~~ done: both are lenses, they compose (2.0/9), order gap 0.5 rank.
4d. ~~Cross-talk~~ done: 14–17% off-target vs 59–65% on-target. Next: a third factor (e.g. mood or point of
   view) to test whether composition holds for three; then multi-sentence spans where plot beats
   can be a factor.
5. Token-level clouds + Gromov-Wasserstein, no role correspondence assumed.
6. A second model family (Pythia or Llama-3.2-1B) to rule out Qwen-specific artifacts.
