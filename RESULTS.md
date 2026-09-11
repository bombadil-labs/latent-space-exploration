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

## Open problems (ordered)

1. ~~Shuffled-holonic control~~ done: stage-2 shape is mostly slot position; content-role offsets survive.
2. ~~Project out slot directions~~ done: partial removal, no content shape recovered.
3. Paraphrases: 3–4 prompts per domain, content roles rotated through slots, so position averages out
   and stage 3 has ~30 examples per relation.
4. Stage 4 readout: patch `src + content-offset` at a late layer over the src span and generate;
   compare to base generation and to a random direction of equal norm.
5. Token-level clouds + Gromov-Wasserstein, no role correspondence assumed.
6. A second model family (Pythia or Llama-3.2-1B) to rule out Qwen-specific artifacts.
