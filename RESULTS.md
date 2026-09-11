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

## 2026-09-11 (hour 7) — Replication: the factor results hold on a smaller model and a second family

Same grid, same scripts, same leave-one-scene-out protocol. Layers chosen at the same relative depth
(mid-stack for the first patch, ~two-thirds for the second).

| | Qwen2.5-1.5B (L14/L20) | Qwen2.5-0.5B (L12/L18) | Pythia-1.4B (L12/L18) |
|---|---|---|---|
| era decodability, layer 0 → peak | 0.28 → 0.94 (L12) | 0.33 → 0.81 (L24) | 0.50 → 0.94 (L18) |
| voice decodability, layer 0 | 0.92 | 0.92 | 0.89 |
| era lens rank/3 (random) | 1.28 (2.11) | 1.39 (2.03) | **1.11** (2.11) |
| voice lens rank/3 (random) | 1.31 (2.11) | 1.19 (2.14) | **1.11** (2.14) |
| era+voice composed, rank/9 (chance 5) | 2.03 | 1.83 | **1.44** |
| order: A then B / B then A, gain corr | 1.83 / 1.89, 0.85 | 1.86 / 1.75, 0.82 | 1.33 / 1.56, 0.83 |
| cross-talk under era patch (on-target / off) | 0.59 / 0.14 | 0.54 / 0.20 | 0.63 / 0.15 |
| cross-talk under voice patch (on-target / off) | 0.65 / 0.17 | 0.62 / 0.14 | 0.67 / 0.19 |

`results/stage5_qwen0.5b_*`, `results/stage5_pythia1.4b_*` .

**Reading.** On the 0.5B model every stage-5 quantity lands within a few hundredths of the 1.5B
value. Era is less cleanly decodable on the smaller model (peak 0.81 vs 0.94), consistent with a
computed feature that sharpens with scale, while voice, being lexical, is identical. Pythia's
model-free check shows the same shape as Qwen (voice in the embeddings, era built by the layers,
peaking late). On Pythia the lens and composition numbers are the strongest of the three models
(both lenses 1.11/3, composition 1.44/9), with cross-talk and order effects of the same size as on
Qwen. Three models, two families, one story: era and voice are additive directions with small
interference.

## 2026-09-11 — Qualitative: factor directions in generation

`scripts/stage5_generate.py`, Qwen2.5-1.5B base, greedy, layer 14, scale 1.5, directions built
from all four scenes. Prompt: *"A moment from a story: The door opened and"*.

- **base:** a man walked in. He was tall and thin, with a shock of white hair and a beard that reached down to his chest. He wore a long, flowing robe and a pair of spectacles
- **+medieval:** a man came in. He was a tall, thin man, with a long beard and a long, gray cloak. He was carrying a heavy sack on his back.
- **+1920s:** a man walked in. He was wearing a suit and tie, and he had a briefcase in his hand. He looked around the room and saw a woman sitting at a desk.
- **+farfuture:** a man walked in. He was tall and thin, with a long beard and a pair of spectacles. He was dressed in a long, flowing robe, and his eyes were filled with a deep
- **+terse:** a man walked in. He was tall, with a shock of white hair and a beard that reached down to his waist. He wore a long robe, and his eyes were cold and calculating.
- **+ornate:** a man stepped out. He was tall and lean, with a shock of white hair and a beard that reached his shoulders. His eyes were like two pools of fire, and his face was lined with
- **+child:** a man came in. He was wearing a hat and a coat. He was very tall and had a big beard. He was a very nice man. He had a big smile on his face.
- **+medieval+child:** a man came in. He was very tall and had long hair. He was wearing a hat and a coat. He was very kind and he gave me a cookie. The man was a baker.
- **+farfuture+ornate:** a man stepped out. He was tall and lean, with a face that was both familiar and alien. His eyes, like two pools of deep blue, held a secret that only he could see.
- **+1920s+terse:** a man walked in. He was wearing a suit and tie, and he had a briefcase in his hand. He walked over to the desk and sat down. The secretary looked up and said,

1920s → suit, tie, briefcase, secretary; medieval → cloak, sack; child → short sentences, "very
nice man", "gave me a cookie"; ornate → "pools of fire", "familiar and alien". Compositions inherit
both parents. Far future alone barely moves this prompt (the base already leans wizard), but
composed with ornate yields "familiar and alien". Illustration only; the log-prob tests are the
evidence.

## 2026-09-11 (hour 8) — Three factors: era × voice × tense compose, with a clean cross-talk matrix

**Grid.** `prompts/narrative_factors_v2.json`: the 36 spans of v1 (tense = past) plus minimal
present-tense rewrites of each (verb forms only), 72 spans, 4 scenes × 3 eras × 3 voices × 2 tenses.
`scripts/stage6_factors.py` generalizes stage 5 to any number of factors. Qwen2.5-1.5B, layer 14,
scale 1, leave-one-scene-out. `results/stage6_qwen1.5b_three_l14.json`.

**(A) Decodability.** Era 0.31 at layer 0 → 0.92 at layer 12 (computed). Voice 0.92 throughout
(lexical, stable). Tense **1.00 at layer 0** → 0.78 at layer 28 (lexical, decays as the residual
turns toward next-token prediction). Three factors of three kinds.

**(B) Each factor as a lens** (rank of patched level among its levels, other factors fixed).

| factor | factor direction | random | chance |
|---|---|---|---|
| era | 1.25 / 3 | 2.00 | 2.0 |
| voice | 1.24 / 3 | 2.25 | 2.0 |
| tense | 1.03 / 2 | 1.44 | 1.5 |

**(D) All three composed.** One patch = dir_era + dir_voice + dir_tense: the joint span ranks
**2.81 of 18** (chance 9.5). Two factors composed to 2.0 of 9 in hour 6; adding a third keeps the
joint span near the top of a candidate set twice as large.

**(X) Cross-talk matrix.** Rows: the factor whose direction is patched. Columns: fraction of the
18-span gain variance explained by each factor's main effect.

| patched ↓ / explained → | era | voice | tense |
|---|---|---|---|
| era | **0.58** | 0.14 | 0.00 |
| voice | 0.16 | **0.64** | 0.01 |
| tense | 0.11 | 0.11 | **0.47** |
| random | 0.25 | 0.21 | 0.02 |

**Reading.** The matrix is strongly diagonal. Era and voice leak into each other at 14–16% (as in
hour 6) and into tense not at all. Tense leaks 11% into each of the others and keeps 47% on
target. Random directions barely move tense (0.02): past/present pairs are near-identical text, so a
random perturbation shifts both members together, whereas the tense direction separates them.
Additive composition of three narrative factors of three different kinds works at the level of a
sentence, with off-diagonal interference an order of magnitude below the diagonal for the
era/voice–tense pairs and about a quarter of it for era–voice.

**Caveats.** Same as before: one author, one scene set, sentence-length spans, single layer.
Tense is the easiest possible third factor (a surface grammatical feature); a third *semantic*
factor (mood, point of view, genre) is the harder test and still requires writing.

## 2026-09-11 (hour 9) — Mood: a semantic factor, weaker than era, integrated at the end of the sentence

**Grid.** `prompts/narrative_mood_v1.json`: 4 scenes × 3 eras × 3 moods (dread, tender, comic) =
36 spans, one neutral voice, same scene event in each. Mood is carried by *what happens in the last
clause* (the horse's eyes are wrong; a note in a hand he had taught to write; the Duke had already
arrived twice), not by vocabulary throughout.

**(A) Decodability, leave-one-scene-out (chance 0.33).**

| pooling | era peak | mood layer 0 | mood peak |
|---|---|---|---|
| mean over span | 1.00 (L24) | 0.56 | 0.67 (L4) |
| last token | 0.89 (L24) | 0.33 | **0.75 (L16–20)** |

Mood is a real direction but a weaker one than era (0.75 vs 0.9+), and unlike era or voice it is
best read at the last token in the mid-late stack: the model integrates the mood of a sentence as
it ends. Mean pooling dilutes it with the neutral scene-setting clauses.

**Generations** (Qwen2.5-1.5B, greedy, last-token mood directions, layer 18). Prompt:
*"A moment from a story: The door opened and"*

- **base:** a man walked in. He was tall and thin, with a shock of white hair and a beard that reached down to his chest…
- **+dread ×2:** a man came in… He was wearing a black cloak and a black hat. He was carrying a long sword.
- **+dread ×3:** … He had a long beard and a long tail. He was a ghost. He came in and said
- **+tender ×2:** a young girl stood there. She was wearing a dress that was as white as snow… She had a smile on her face that made everyone feel happy.
- **+tender ×3:** a young girl stood there, her face shining with the light of the first sunrise. "Mommy, I'm home," she said, "I've brought you a flower."
- **+comic ×2:** the man walked in. The woman was surprised. She had never seen him before. The man was a little bit nervous.
- **+comic ×3:** the man walked in. The door closed and the man walked out. The door opened and the man walked out. The door closed and the man walked in.

Prompt: *"A moment from a story: She found the letter on the table and"*

- **base:** read it. It was from her mother. She was very happy. She was going to have a birthday party…
- **+dread ×2:** opened it. It was from her mother. She was dead. She was dead. She was dead. She was dead.
- **+dread ×3:** opened it. It was a letter from her father. It was a letter from her father who was in prison… and who was in danger.
- **+tender ×3:** read it. The letter was from her mother, who had just died. The letter was about how her mother had always told her that she loved her, and how she had always wanted to tell her
- **+comic ×2:** began to read. "Dear Mr. Smith, I hope you are doing well. I am writing to inform you that I have decided to move to a new city."
- **+comic ×3:** began to read. She was a bit surprised to find that she was not the only one who had been in the room.

Dread and tender are unmistakable at ×2–3 (cloak, sword, ghost, "she was dead", prison, danger;
white dress, first sunrise, "Mommy, I'm home", a mother's love). Comic is the weakest: it reads as
deflation and absurd repetition (the door loop, the bureaucratic letter) rather than as jokes, which
is roughly what the comic spans do (they deflate) but a 1.5B base model has no comic timing to
steer. Mean-pooled directions at layer 14 and scale 1.5 did **not** produce visible mood shifts;
the last-token, layer-18, scale-2+ setting is what works.

**Quantitative (mean-pooled directions, layer 14, scale 1, leave-one-scene-out).**
`results/stage6_qwen1.5b_mood_l14.json`.

| | factor direction | random | chance |
|---|---|---|---|
| era lens, rank/3 | 1.19 | 1.92 | 2.0 |
| mood lens, rank/3 | **1.28** | 1.89 | 2.0 |
| era + mood composed, rank/9 | **1.89** | | 5.0 |

| patched ↓ / explained → | era | mood |
|---|---|---|
| era | **0.68** | 0.09 |
| mood | 0.15 | **0.57** |
| random | 0.31 | 0.25 |

**Reading.** By the log-prob test the mood direction is as good a lens as voice or era were
(1.28/3 vs 1.89 random), composes with era to 1.89/9, and the cross-talk matrix stays diagonal:
era leaks 9% into mood, mood leaks 15% into era. So a semantic, end-of-sentence factor behaves like
the lexical ones under the controlled measurement, even though it needs a stronger, later,
last-token patch to show up in free generation. The gap between "selects the right span" (easy)
and "visibly rewrites a continuation" (harder) is a general feature of these directions and worth
stating in the writeup: the lens is reliable as a selector well before it is reliable as a
generator.

## 2026-09-11 (hour 10) — Theme over multi-sentence spans: a reliable selector, a poor generator

**Grid.** `prompts/narrative_theme_v1.json`: 4 situations (a debt comes due, a message arrives, a
door, a meal) × 3 eras × 3 themes (betrayal, sacrifice, homecoming) = 36 passages of ~61 words,
three sentences each, neutral voice. Theme is what the passage is *about*, distributed over the
whole span rather than carried by one clause.

**(A) Decodability, leave-one-situation-out (chance 0.33).**

| pooling | era peak | theme layer 0 | theme peak |
|---|---|---|---|
| mean over span | 1.00 (L4+) | 0.67 | **0.78 (L20)** |
| last token | 0.92 (L24) | 0.33 | 0.50 (L16) |

The mirror image of mood: theme is distributed, so mean pooling reads it and the last token does
not. Two-thirds of it is already present at the embedding layer (lexical cueing: *sworn, forged,
seal* vs *gave, so that* vs *gone … years ago, asked whether*), and the stack adds ~10 points.

**Quantitative (mean-pooled directions, layer 20, scale 1).** `results/stage6_qwen1.5b_theme_l20.json`.

| | factor direction | random | chance |
|---|---|---|---|
| era lens, rank/3 | 1.36 | 2.00 | 2.0 |
| theme lens, rank/3 | **1.25** | 1.97 | 2.0 |
| era + theme composed, rank/9 | **2.22** | | 5.0 |

| patched ↓ / explained → | era | theme |
|---|---|---|
| era | **0.62** | 0.12 |
| theme | 0.17 | **0.61** |
| random | 0.35 | 0.22 |

As a selector on held-out passages, the theme direction is as good as every other factor tested,
composes with era, and the cross-talk matrix is diagonal at the usual 12–17%.

**Generations** (layer 20, scale 1.5 and 2.5, 60 tokens): mostly **not legible**. The 1.5B base
model degenerates into repetition on 60-token continuations regardless of patch, and the theme
directions do not rescue it. Faint traces at ×2.5: +sacrifice → "She was tired of the war, tired
of the fighting, tired of the killing… of her friends, and… her enemies, and… her family";
+betrayal → "She had been so sure that she would be safe, but now she was afraid"; +homecoming →
"I thought it was lost, but it wasn't. I found it again." The rest is base-like or degenerate.
`results/stage6_theme_gens.log`.

**Reading.** This is the first factor where the selector/generator gap is wide, and it is the
factor the additive picture was expected to strain on. A theme is a *relation among events across
sentences*; a single direction added at one layer at every position can raise the likelihood of a
passage that already has that structure (the lens test), but it cannot by itself make a 1.5B
model produce that structure over 60 tokens. Era, voice, tense and mood are properties a
continuation can carry token by token; theme is not. That is the boundary of the sentence-level
additive calculus, stated with a number: theme lens 1.25/3, theme generation ≈ base.

**What would push past it.** A larger or instruction-tuned model (plot competence to steer);
patching at multiple layers with re-imposition (the erosion problem, `steer.steer_patches`);
directions from a contrastive pair rather than a mean (sharper); or treating theme as a sequence of
beat-level directions applied at different positions rather than one direction everywhere, which is
the relation-operator idea from stage 3 brought back at the plot level.

## 2026-09-11 (hour 11) — Pushing on the theme boundary: multi-layer patches don't move it

**Multi-layer, re-imposed.** Same theme grid; the (factor, level) direction is estimated at each of
layers 12, 16, 20, 24 and added at all four (scale 0.5 each) so that later blocks cannot erode it.
`results/stage6_qwen1.5b_theme_multilayer.json`.

| | single layer 20, scale 1 | four layers, scale 0.5 each |
|---|---|---|
| theme lens, rank/3 | 1.25 | 1.19 |
| era lens, rank/3 | 1.36 | 1.25 |
| era + theme composed, rank/9 | 2.22 | 2.17 |
| cross-talk theme→era / era→theme | 0.17 / 0.12 | 0.19 / 0.11 |

Within noise of the single-layer numbers. The selector was already near its ceiling; re-imposition
does not raise it.

**Generation with repetition penalty (1.3) under the four-layer patch.** With the repetition
collapse suppressed, the base model's other failure mode appears: it drifts into quiz/exam format
("What is going to happen next? Options: …"), which is a property of Qwen2.5-base's pretraining mix,
not of the patch. Inside the prose that survives, traces are faint and factor-appropriate:
+sacrifice → hunger, starving, "no food for them", "they ate some of their own meat";
+homecoming → "carrying his pack… walking for days", "waiting for it… hope of rescue… what they'd
found out about him"; +betrayal → nothing consistent. `results/stage6_theme_gens_multi.log`.

**Reading.** The theme boundary from hour 10 stands. Neither erosion (fixed by multi-layer
re-imposition) nor degenerate decoding (fixed by the penalty) was the obstacle; the obstacle is
that a 1.5B base model does not have a "write a passage about sacrifice" capability for a single
direction to unlock, whereas it does have "write in a 1920s setting" and "write like a child".
Steering can only select among competences the model already has. The next real move is a model
with the competence (instruction-tuned or larger), not a better patch.

## 2026-09-11 (hour 12) — Instruct model: theme steering becomes partly legible, and hits the refusal direction

Same theme grid, extracted on **Qwen2.5-1.5B-Instruct**; directions from its own activations.
`results/stage6_qwen1.5b_instruct_theme_l20.json`, `results/stage6_theme_gens_instruct.log`.

**Quantitative (layer 20, scale 1).** Theme decodability 0.81 (vs 0.78 base). Theme lens 1.22/3
(random 1.92). Era + theme composed 1.97/9. Cross-talk era→theme 0.11, theme→era 0.18. Same as
the base model within noise: the selector does not care whether the model is tuned.

**Generations** (layers 12/16/20 re-imposed, repetition penalty 1.1, 70 tokens).

*Raw continuation of "It was late when the news reached her, and":*
- base: … she sat on the floor of his room, feeling helpless and alone…
- +betrayal ×1.5: … she had no choice but to accept it. She knew that this would be the end of their relationship, and she felt a sense of relief as she prepared for the inevitable.
- +homecoming ×1.5: … she had been gone for days. She had come to visit him in his hospital room, but he had died before she could reach him… holding his hand as if it were still warm.
- +sacrifice ×1.5: … if she didn't act quickly, it would be too late. The situation seemed hopeless, but she couldn't give up.

*Chat-templated "continue this passage":*
- +betrayal ×1.5: The words were like daggers piercing through her heart, leaving an indelible mark that would haunt her for years to come… this moment marked the beginning of the end.
- +homecoming ×0.8 and ×1.5: **"I'm sorry, but I cannot continue or generate new content as requested."** The homecoming direction, added at these layers, pushes the instruct model into its refusal behavior. Betrayal and sacrifice do not.

*Chat-templated "write a passage: two old friends meet at a crossroads at dawn":*
- base: … Alice and Bob stumbled upon each other… exchanged stories of their lives since they last saw each other
- +homecoming ×0.8: Emma and her long-lost friend Sarah… at an ancient crossroads where time seemed to stand still… They embraced, their faces etched
- +homecoming ×1.5: … as if they had been waiting for this moment forever. They embraced, their faces aglow with joy
- +betrayal, +sacrifice: close to base (reminiscing, a walk, a fork in the road).

**Reading.** Partial support for the competence hypothesis. On the tuned model the same
directions produce theme-appropriate content that the base model could not: homecoming yields
*long-lost*, *embraced*, *waiting for this moment forever*, and in the raw case an arrival that
comes too late; betrayal yields *daggers*, *the end of their relationship*, *the beginning of the
end*. Sacrifice remains the weakest (it drifts to urgency and determination rather than giving
something up). The quality gap is prompt-dependent: the open "write a passage" prompt shows the
themes most clearly, the raw continuation least.

**The refusal side effect is a finding in itself.** Adding the homecoming direction at scale 0.8
inside the chat template produces a canned refusal, twice. A mean-difference direction built from
36 story passages has a component along whatever the instruct model uses to decide "I cannot
continue," and the chat template makes that component live. This is exactly the interference that
the cross-talk matrix cannot see, because refusal is not one of the grid's factors. For any use of
these directions on a tuned model, the refusal direction has to be projected out first, which is
a known technique and cheap to add. Noted as an open problem.

## 2026-09-11 (hour 13) — Remote models via NDIF: Gemma-2-9B-it steers on theme

**Infrastructure.** NDIF works from this environment through the credential-injecting proxy
(`src/lsx/ndif.py`: key header added by the proxy, HTTPS submit + poll instead of WebSocket, Python
3.12 venv). Free-tier key: pinned models only. GPT-J-6B (ungated) and Gemma-2-9B-it (license
accepted) are usable; the Llama pins are awaiting Meta's review. Round trip ~4 s for a forward
pass; ~36 passages extracted in under 2 minutes. Gemma's 256k vocabulary needs one passage per
scoring job on the shared GPU.

**Theme decodability across scale** (mean-pooled, leave-one-situation-out, chance 0.33):

| model | params | theme peak (layer / total) |
|---|---|---|
| Qwen2.5-1.5B base | 1.5B | 0.78 (20/28) |
| Qwen2.5-1.5B-Instruct | 1.5B | 0.81 (12/28) |
| GPT-J-6B | 6B | 0.83 (4–12/28) |
| **Gemma-2-9B-it** | 9B | **0.94 (20/42)** |

The theme direction sharpens with scale and tuning.

**Theme battery across models** (leave-one-situation-out; mid-stack layer; scale 1).

| | Qwen2.5-1.5B (L20) | GPT-J-6B (L14) | Gemma-2-9B-it (L20) |
|---|---|---|---|
| theme lens, rank/3 (random) | 1.25 (1.97) | **1.11** (2.06) | pending |
| era lens, rank/3 (random) | 1.36 (2.00) | 1.22 (2.19) | pending |
| era + theme composed, rank/9 (chance 5) | 2.22 | **1.75** | pending |
| cross-talk era→theme / theme→era | 0.12 / 0.17 | **0.06 / 0.14** | pending |
| on-target era / theme | 0.62 / 0.61 | 0.76 / 0.62 | pending |

At 6B the theme lens is near-perfect, composition is tighter, and era leaks into theme half as much.
`results/stage6_gptj_theme.log` (the per-case JSON for this run was lost to a serialization bug,
fixed since; the summary is in the log).

**Gemma-2-9B-it generations** (`scripts/ndif_generate.py`; direction added at block 20 on every
decoding step, greedy, 60 tokens; directions from Gemma's own activations).

*Raw continuation, "It was late when the news reached her, and":*
- base: … the moon was already high in the sky, casting long, skeletal shadows across the dusty road…
- **+betrayal ×2:** … the weight of it pressed down on her like a shroud. The world, she realized, had shifted on its axis. **Everything she thought she knew, everything she had built her life upon, was now a lie.**
- **+homecoming ×1:** … She stood at the crossroads, her heart pounding in her chest, torn between two paths. **One led to the familiar comfort of her village**, the other, shrouded in mist, beckoned her towards the unknown
- +homecoming ×2: … The telegram lay on the table… "He is dead," it read. "Come quickly."
- +sacrifice ×1–2: … her heart, once a hummingbird's wings, now felt like a leaden weight… as if the very sky was weeping. (grief, not sacrifice)

*Chat-templated "continue this passage":*
- **+homecoming ×1:** … **She hadn't expected to hear from him again, not after all these years, not after the way things had ended.** But the letter,
- **+betrayal ×2:** … She reread the telegram, each word a hammer blow to her carefully constructed world. The world, it seemed, had shifted on its axis, leaving her stranded
- +sacrifice ×1–2: … her hands clasped tightly in her lap, as if trying to hold onto some warmth that was slipping away. The messenger, a young man
- The "old man opened the box" prompt barely moves under any patch: the base continuation (withered rose, locket, memories) is a strong prior that a scale-2 patch does not overcome.

**Reading.** On a 9B instruction-tuned model the theme directions do what they could not on 1.5B:
betrayal produces *"everything she had built her life upon was now a lie"* and homecoming produces
*"she hadn't expected to hear from him again, not after all these years"*, with the prose quality of
the base continuation intact and no refusal. Sacrifice remains the weak theme on every model; my
sacrifice spans read as loss, and the direction captures loss. The prompt-prior effect is real: a
continuation the model is already confident about resists a theme patch that a more open prompt
accepts. Selector-level numbers on Gemma follow when the battery completes.

## Open problems (ordered)

1. ~~Shuffled-holonic control~~ done: stage-2 shape is mostly slot position; content-role offsets survive.
2. ~~Project out slot directions~~ done: partial removal, no content shape recovered.
3. ~~Rotate content through slots~~ done. Real paraphrases (new wording per domain) would raise the
   effective n above seven and are the main lever for both stage 3 and the relation lens.
4. ~~Stage 4 role lens~~ done: rank 1.7 vs 3.3 random, all eight held-out domains.
4b. ~~Relation lens~~ done: null (−0.02 nats vs −0.19 random); the relation is measurable, not yet controllable.
4c. ~~Narrative factors era × voice~~ done: both are lenses, they compose (2.0/9), order gap 0.5 rank.
4d. ~~Cross-talk~~ done. ~~Third factor~~ tense: three-way composition 2.8/18, diagonal cross-talk matrix.
4e. ~~Mood~~ done: semantic factor, lens 1.28/3, composes with era 1.89/9, diagonal cross-talk; needs
   last-token/late-layer/×2–3 patch to show in generation.
4f. ~~Theme over multi-sentence spans~~ done: selector 1.25/3, composes 2.2/9, generation ≈ base.
4g. ~~Multi-layer~~ no change. ~~Instruct 1.5B~~ partial: homecoming and betrayal become legible, sacrifice
   not; homecoming direction triggers refusal in the chat template. Remaining: project the refusal
   direction out of factor directions before steering tuned models; NDIF (api.ndif.us reachable,
   key pending a fresh session): Qwen2.5-7B/-Instruct first, Llama-3.3-70B-Instruct when HF_TOKEN
   is set; beat-level directions at positions.
4h. Mood × voice grid: do two register-like factors interfere more than era does with either?
5. Token-level clouds + Gromov-Wasserstein, no role correspondence assumed.
6. ~~A second model family~~ Pythia-1.4B: everything replicates, slightly stronger.
