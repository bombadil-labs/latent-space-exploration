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

## Open problems (ordered)

1. Shuffled-holonic control for stage 2 (permute spans across slots within a prompt).
2. Paraphrases: 3–4 prompts per domain per framing, so stage 3 has ~30 examples per relation.
3. Stage 4 readout: patch `src + offset` at the best layer over the src span and generate; compare
   to base generation and to patching a random direction of equal norm.
4. Token-level clouds + Gromov-Wasserstein, no role correspondence assumed.
5. A second model family (Pythia or Llama-3.2-1B) to rule out Qwen-specific artifacts.
