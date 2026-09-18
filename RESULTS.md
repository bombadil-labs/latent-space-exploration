# Results log

Honest running record. Numbers are from the scripts in `scripts/`, JSON in `results/`. Negative
and null results are kept. Models: Qwen2.5-0.5B (24 layers, d=896), Qwen2.5-1.5B (28 layers,
d=1536), CPU, float32. Grid: `prompts/holonic_v1.json` (8 domains x {holonic, flat}, 6 roles).

## CHECKPOINT 2 — 2026-09-12, after 38 stages

*Checkpoint 1 (below) covers stages 1–22 and still reads correctly except where this supersedes it.
Twelve stages have been logged since; three of them retracted earlier results. The instrument
history is written up separately in `docs/INSTRUMENTS.md`.*

**Withdrawn since Checkpoint 1 (read this first).**
- **Hours 28, 30, 31 — "subject-relative timescales are absent."** Withdrawn at h32. The probe was a
  1536-dimensional residual norm built from three-paraphrase means, sitting at its own noise floor
  (residual/floor 1.1–1.4, nothing above 1.8; the identical pipeline on Gaussian noise returns 0.95
  and "no signal" everywhere). Three published negatives across two models were an instrument
  reading itself.
- **Hour 34 — the Llama selector battery.** Withdrawn at h36. `ndif_factors.py` patched
  `B[l].output[0]`, which under current transformers is batch row 0 of a bare tensor rather than the
  hidden states; with nine candidates in one padded batch only one was ever touched, and
  rank-1-on-ties pinned every measured rank at 11/9 = 1.22 for *any* direction. That was the whole
  reported band (1.00–1.28), treatment and control alike. The layer-sweep conclusion went with it,
  so **h33's depth-fraction confound is open again**.
- **Two incidental "layer 28" numbers are mislabelled, not wrong** (found 2026-09-17 during the
  phase-0.1 spec review, verified directly). HF applies the final norm to the last hidden state, so
  `residuals()[-1]` is the final-norm *output*, not the last block's residual (mean position norm
  190.5 vs 283.2 on Qwen2.5-1.5B). Affected: the RSA figure "0.38 at layer 28" and "tense 0.78 at
  layer 28". Both are incidental; every headline layer number in this repo is mid-stack (12, 14, 16,
  18, 20, 24) and unaffected. The docstring in `src/lsx/model.py` asserted the opposite and is
  corrected; `tests/test_invariants.py::test_last_hidden_state_is_post_norm` now pins the convention.
  Not a retraction — no claim depended on it.
- **Hours 14 and 23, and claim 6 of the writeup** ("an era shift moves the address and keeps the
  form"). Withdrawn at h40: the readout was taken after the patch layer, and since the patch is a
  constant added at every position and the readout is a span mean, the movement is vector addition,
  exactly. The model arm is indistinguishable from the arithmetic (see the hour-45 correction: the
  "worse than it" reading was an artifact of a tolerance 8.8x too tight).
  Recomposition now rests solely on the generation results of h29/h33, which re-read unpatched text
  and are confirmed clean.
- **Hour 31's numbers** (not its conclusion). Withdrawn at h39: nnsight left-pads batched texts and
  the extractor indexed spans from the unpadded text, so 363 of 480 passages read their spans out of
  the padding, with the padding correlated with Δt. Corrected numbers are in h39 and are stronger.
- **Hour 28–35 discrimination numbers are inflated by copying.** At h38, with the interval phrase
  removed so the model cannot copy Δt from the prompt, h35's 0.961 at layer 14 falls to 0.522.

- **Hour 8's comparisons to chance** (not its measurements). At h47 the lexical floor was measured
  rather than assumed: only **era** clears it (1.25 against a floor of exactly chance, 2.00). **Tense
  sits on its floor** (1.0278 both), **voice is worse than its floor** (1.2361 against 1.0278), and
  the three-way composition's gain is **−0.04 of a rank inside a ±1.83 band**. The patch effect is
  real (its no-patch arm is 9.50 exactly); what is withdrawn is "2.81 against chance 9.50" as
  evidence that the factors compose *beyond what the words give away*. Four ledger rows withdrawn —
  the first this ledger carries.
- **Hour 39's Gemma clock** (the claim, not the extraction). At h47, per subject against the grid's
  own t0 control prompts, the gain is **+0.067 inside a ±0.40 band**, with a shuffled-stimulus arm at
  0.9693. At this design's resolution the clock is the interval phrase. Consistent with h38's
  order-invariance, on a second model.
- **Hour 29 moved the other way and is stronger** (h47): all three arms at its own scale —
  treatment 0.8364, random 0.1429, no-patch 0.1111 against a declared null of 0.1111. The
  under-controlled caveat added at h46 is lifted. Caveat that stands: 18% generation loss, unequal
  across arms.
- **Two record defects, not retractions** (h47): h8's tense lens is 1.03 of **2**, not of 3 (the h8
  table had it right; the summaries did not), and h16's **2.21 is the step-4 subsample** of a curve
  whose full step-2 mean is **2.1692**.

**Stands, added or changed since Checkpoint 1.**
10. **The gauge/engine boundary is a magnitude, and it is engine-specific.** At 3× norm re-imposed at
    every decoding step, an era shift moves the *generated* era on Gemma-2-9B-it: 0.84 read as
    target, 0.30 gaining target-era vocabulary, prose intact (h29). The same treatment on
    Llama-3.1-70B-Instruct reaches only 0.43, below Gemma's 2× value (h33). At matched norm both are
    null (h27). Bigger is not more steerable.
11. **Instruction tuning sharpens some factors and not others.** On the matched pair Llama-3.1-70B
    and 70B-Instruct — same pretraining, size, tokenizer — the era selector goes 1.50 → 1.06, at both
    layer 26 and layer 14, so depth does not explain it; theme is tuning-invariant at 1.06–1.39
    either way (h37). The 70B-Instruct has the sharpest era selector measured and the weak engine of
    h33: sharp gauge, weak engine, at fixed size.
12. **Claim 7 of the writeup is now about tuning, not only scale**, pending the generation arms.

**Fell, added since Checkpoint 1.**
- **Parameterized time translation** (h28–h38). Five stages, ~1M agent tokens, closed at h38. A
  shared direction does track the interval and replicates across models, and against a lexical floor
  of 0.728 the model recovers real gain (0.297 at layer 24, permutation z 7.6), so it is not only
  vocabulary. But the gain survives word-shuffling the passage (structural residue 0.076, label-swap
  z 1.21) and does not transfer to the case where the model must supply the change itself (ρ 0.150,
  cos 0.078). It is a **computed register detector** — how much change a passage describes, read
  order-invariantly — not a representation of elapsed time. `transform: translate (parameterized)` is
  withdrawn from ALGEBRA as a time operator. Subject-relative clocks dropped with reasons on record.
- **Llama-3.1-405B is unreachable** for this key: "Model is not pinned and hotswapping is not
  supported", deterministic, 3/3 attempts (h34). `results/ndif_pinned.txt` is stale. The
  scale-vs-tuning spec's "or sufficient scale substitutes for tuning" clause cannot be tested; 70B is
  the largest reachable base model.

**Standing caveats, updated.** Every caveat from Checkpoint 1 still holds, and **no human-written
grid yet** remains the first thing a reviewer will ask about. Added: no text-based time grid can
drive layer-0 discrimination to chance, because far-interval prose genuinely uses a different
register from near-interval prose (h35, floor 0.72); any future stimulus-based design must measure
gain over that floor rather than try to remove it.

**Plan from here.**
1. **Consolidation, not experiments.** This checkpoint, `docs/INSTRUMENTS.md`, and a writeup pass that
   matches stage 38. The value is now in the narrative; the experiments have outrun the exposition.
2. **A human-written grid.** Every grid so far was authored by a model. This is the cheapest single
   change to the repo's credibility.
3. **Piece 2 of `docs/specs/scale_vs_tuning_v1.md`**: the generation arms on the 70B pair at 3×
   re-imposed, with a *generation-level* layer sweep, which is the control h33 needs and h36 removed.
4. ~~**An audit of five unchecked instrument risks**~~ done (h39). h31 fell to a *padding* bug (76%
   of passages corrupted) and its numbers are replaced; its conclusion survives. The six remaining
   `output[0]` scripts were safe as run; local tie-ranking never fired and no local claim changed;
   h8's tense flag is cleared. **Remaining from it:** the abstraction ladder (h15, h26) still has no
   null anywhere — three are now specified and cost one CPU session.
Deferred unchanged: cohere, commutator regimes, Shadow Walker adapter. Dropped: subject clocks.

## CHECKPOINT 1 — 2026-09-11, after 22 stages

**Stands, with controls (keep in the writeup).**
1. Discourse position dominates pooled activations; the naive shape test measured template slots (h2, h3).
2. Roles are domain-independent directions and causally usable as a lens (h4: 1.7/6 vs 3.3 random; layers 14–20).
3. The source→target relation is a real, computed, mid-stack signal and, at 40 domains, a working *selector* (h16: **2.1692** over the full step-2 curve — the figure printed as 2.21 is the step-4 subsample, 2.2065 — vs 3.5 null; the peak-layer-16 form is refused for selecting on scoring data).
4. Narrative factors (era, voice, tense, mood, theme) are additive directions: each a lens (era and voice 1.2–1.3 of 3, tense 1.03 of **2**), three compose (2.8/18), cross-talk matrices diagonal, replicated on Qwen 0.5B/1.5B, Pythia 1.4B, GPT-J 6B, Gemma 9B-it (h5–h13).
5. Factors differ in kind and depth: voice/tense lexical, era computed by layer 12, mood last-token, theme distributed and mid-passage (h6, h8, h9, h17).
6. Recomposition: an era shift moves the address and keeps the form on two models (h14: 0.89/0.88 moved, theme 0.81/0.94 kept).
7. Steering selects among competences the engine has: theme steers generation only at 9B-instruct (h10–h13); tuned models carry refusal as an unlisted factor (h12).
8. Abstraction as quotient: narrower dictionaries keep more general features (h15: 0.18 vs 0.06; 12/15 merges up).
9. The shallower factor dominates the surface text regardless of patch order (h19, h22, two layer pairs for era > theme).

**Fell (keep as negatives).** Pooled RSA as a content shape (h2–h3). Absence by decoder adjacency (h18). Five-regime commutator taxonomy under greedy decoding (h22). "Factors don't commute under generation" as factor-specific (h22: any matched-norm pair diverges).

**Partial / open.** Generative relation lens: helpful but not source-specific at 40 domains (h20). Continuation-defined absence: null at n=8 (h21). Selector saturates early; the model with the sharpest selector is not the one that steers (h13).

**Standing caveats.** Grids authored by Claude, reviewed by Claude (32 holonic domains), or authored by GPT (h23: factor and theme results replicate); no human-written grid yet; selector-level evidence dominates; generation-level evidence qualitative and model-dependent; layers chosen mid-stack by convention with sweeps only for the role lens and relation lens.

**Reoriented plan.**
1. ~~**Writeup** rewrite~~ done (second draft, post-checkpoint); figures and per-family tables still to embed.
2. ~~Source-specificity at the selector level~~ shown (h24); under patching marginal (h25). Next: the spin test at selector level.
3. ~~Third-party factor grids~~ GPT-authored era×voice and era×theme replicate (h23). Remaining: a human-written grid, and a holonic domain set from a second author.
4. **Scale the steering-competence curve**: Llama-3.1-70B-Instruct when the license clears; same scripts.
5. **Absence, third realization**: larger n, or a withheld part defined by the model's own surprise rather than by an author.
6. ~~Abstraction ladder extension~~ done (h26): width not depth; broad corpus shrinks the effect to 1.9×; fixed points trivial; flow is an ordering (era dies before theme), not an optimum. Remaining: feature labels.
Deferred: cohere (no observable), commutator regimes (no structure at this n), Shadow Walker adapter (belongs there).

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
| theme lens, rank/3 (random) | 1.25 (1.97) | **1.11** (2.06) | 1.25 (2.11) |
| era lens, rank/3 (random) | 1.36 (2.00) | 1.22 (2.19) | 1.17 (2.08) |
| era + theme composed, rank/9 (chance 5) | 2.22 | **1.75** | 1.97 |
| cross-talk era→theme / theme→era | 0.12 / 0.17 | **0.06 / 0.14** | 0.13 / 0.14 |
| on-target era / theme | 0.62 / 0.61 | 0.76 / 0.62 | 0.59 / 0.55 |

At 6B the theme lens is near-perfect, composition is tighter, and era leaks into theme half as much.
Gemma-2-9B-it's selector numbers match the 1.5B rather than GPT-J, even though its theme
*decodability* (0.94) and its *generation* (hour 13) are the best of the three: the selector test
saturates early, and the model that steers generation is not the one with the tightest selector.
`results/stage6_gemma_theme.log`, `results/stage6_gemma9b_theme_l20.json`.
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

## 2026-09-11 (hour 14) — Move the address, keep the form: an era shift preserves theme

**Setup.** `scripts/stage7_shift.py`. For each of the 36 theme passages (era e1, theme t), add the
era-shift patch dir_era[e2] − dir_era[e1] (leave-one-situation-out, layer 14, scale 1) at every
position while the model reads the passage. Read the span representation at layer 20 and classify
it by nearest era direction and nearest theme direction. Controls: no patch; a random direction of
the same norm. Qwen2.5-1.5B. `results/stage7_qwen1.5b_shift.json`.

| condition | era reads as target e2 | era reads as original e1 | theme reads as t |
|---|---|---|---|
| no patch | — | 0.97 | 0.78 |
| **era shift** | **0.89** | 0.04 | **0.81** |
| random, same norm | 0.03 | 0.93 | 0.79 |

**Gemma-2-9B-it, same protocol via NDIF** (`scripts/ndif_shift.py`, `results/stage7_gemma9b_shift.json`):

| condition | era reads as target e2 | era reads as original e1 | theme reads as t |
|---|---|---|---|
| no patch | — | 0.97 | 0.94 |
| **era shift** | **0.88** | 0.12 | **0.94** |
| random, same norm | 0.00 | 0.99 | 0.93 |

On the 9B model the theme readout is at 0.94 unpatched and stays at 0.94 under the era shift, so
"form kept" is no longer limited by the readout's own ceiling as it was on the 1.5B.

**Reading.** One additive patch moves a passage's representation to a different era while leaving
its theme where it was, on both models. This is the recomposition primitive of the narrative calculus at the
representational level, with a number: address moved in 89% of cases, form kept in 81% (identical
to the unpatched theme readout). It is also the first experiment that joins the two halves of the
project: the factor toolkit (era as a direction) acting on the shape (theme) that the earlier stages
tried to measure directly. Caveats: representational readout, not generation; one model; the
readout directions and the patch directions come from the same leave-one-out fold, so the test is
fair to the held-out situation but shares training data across era and theme.

## 2026-09-11 (hour 15) — Deterritorialization via dictionary width: narrower dictionaries keep more general features

**Setup.** Gemma-2-9B-it, block-20 residuals per token (via NDIF, `scripts/ndif_tokens.py`) for a
canonical Picard description (`prompts/picard.json`) and 72 reference passages (the theme and mood
grids). Gemma Scope JumpReLU dictionaries at layer 20, widths **16k** (L0≈47) and **131k**
(L0≈43). Generality of a feature = fraction of reference passages on which it fires at least once
(label-free; Neuronpedia is unreachable here). `scripts/sae_ladder.py`, `results/sae_ladder_picard.json`.

**Distribution.** Content features active on the description (anchored on an alphabetic token,
excluding features that fire on >90% of references):

| dictionary | n active | mean generality | median | fraction rare (<0.1) |
|---|---|---|---|---|
| 16k (narrow) | 1255 | **0.18** | 0.11 | 0.49 |
| 131k (wide) | 1500 | **0.06** | 0.014 | 0.83 |

**Merge test.** For the 15 strongest content features at 131k, the nearest 16k feature by decoder
cosine is *more general* in **12 of 15** cases. The clearest: a 131k feature anchored on
*Enterprise / Picard / Federation* (generality 0.14) maps to a 16k feature anchored on *stars /
Federation / Star* (0.38): the Star Trek particular merges into a space-setting feature one rung up.
*Earl Grey* (0.11) and *tea* (0.10) map to features anchored on *principle* (0.35 / 0.00), i.e. the
matches for idiosyncratic particulars are weak (cosine 0.27–0.41): the narrow dictionary has no home
for them, which is what dropping detail looks like.

**Reading.** Re-encoding the same activation through a narrower dictionary keeps features that fire
across more passages and loses the ones specific to this description. That is the operational
content of *abstract as deterritorialization*: generality rises on every axis at once, with no axis
chosen. The Picard → "spaceship diplomat" → "mentor" chain is not readable without feature labels,
but its first rung is visible (Enterprise → stars). Caveats: generality is measured against a
narrative-only reference set of 72 passages, one layer, two widths, one description; the two
dictionaries have matched sparsity but different training, so "same feature" is by decoder
similarity only.

## 2026-09-11 (hour 16) — The relation lens with forty domains: the relation was starved, not absent

**Setup.** `prompts/holonic_v2.json`: the original 8 holonic domains plus 32 generated by
Gemma-2-9B-it from the schema spec (`scripts/ndif_gen_domains.py`) and reviewed by Claude
(`prompts/holonic_candidates_review.json`; 8 domains had clauses that copied the example's wording
paraphrased minimally). Rotated into the position-balanced Latin-square form (240 prompts),
extracted on Qwen2.5-1.5B, and run through the hour-3 protocol unchanged: role-centering
(subtract each role's cross-domain mean from training folds), full affine operator fit in dual
form (ridge 10), leave-one-domain-out, role_rank = where the true target role lands among the
held-out prompt's six roles by cosine to the prediction (chance 3.5), null = shuffle the
source/target pairing within training folds. Layers 0–28 step 4. `results/stage3_v2.log`.

| | role_rank, all layers (chance 3.5) |
|---|---|
| **role-centered affine, 40 domains** | **2.21** (step-4 subsample; full step-2 curve mean 2.1692 — h47) |
| same, 7 domains (hour 3) | 3.01 |
| null, seeds 1 / 2 | 3.49 / 3.53 |
| constant (mean-target) baseline, role-centered | 3.45 |
| not role-centered (role identity retained) | 1.37 (constant baseline 1.10) |

**Reading.** Sixfold more domains moved the role-centered relation transfer from 3.0 to 2.2, with
the null unchanged at chance. With position balanced by design, role identity removed by
centering, and domain address removed by ranking within the prompt, an affine map fit on 39
domains predicts a held-out domain's target-role content from its source-role content well above
chance. This is the strong form of the original hypothesis, and it now has support: **a relation
learned in known domains transfers to an unknown one, and its weakness at seven domains was a
data limit.**

**Per-layer profile** (`results/stage3_qwen1.5b_v2_rolecentered.json`): role_rank 2.73 at layer 0,
2.20 at 4, 1.98 at 10, **1.73 at 16**, 1.86 at 20, 2.39 at 24, 2.73 at 28; predicted-target cosine
rises from 0.08 to 0.35–0.39. Compared with seven domains (3.68 → 2.58 → 3.22), the whole curve
has shifted down by about one rank and the peak moved earlier (16 vs 20). Layer 0 is now below
chance too (2.73 vs 3.5): with forty domains a lexical component of the relation is detectable in
the embeddings, and the stack adds another rank on top of it. Easiest relations: anything → from_below
(1.68–1.73) and embedded → from_above (1.80); hardest: → disturbance and → objectified (2.6).

**Caveats.** 32 of the 40 domains were written by a model from a spec containing one worked
example, then reviewed rather than authored; shared phrasing across generated domains could
inflate transfer, which is why the review paraphrased the copied clauses and why role-centering
matters. Six rotations per domain reuse the same spans, so the effective sample is 40, not 240.
One model (1.5B), selector-level. The generative version of this lens (patch the predicted
target, hour 5: null at 7 domains) is the next test, now with a real operator to patch.

## 2026-09-11 (hour 17) — Derivative curves: theme accumulates in the middle beat, era is lexically early, mood is last-token

**Setup** (agent run, `scripts/derivative_curves.py`, `scripts/derivative_figures.py`,
`results/derivative_curves.json`, `results/notes/derivative_curves.md`, figures under
`results/figures/`). Qwen2.5-1.5B. Per-token projection of the layer-20 residual onto
leave-one-situation-out theme and era directions across each three-sentence theme passage;
per-sentence means; beat-to-beat differences; the mood grid at layers 16–20. Control: random
direction triples of matched norm (margin 0.000 ± 0.002, hit 0.33 at every position).

**Theme accumulates in the middle.** Own-theme readout is at chance for the first ~15% of a
passage (hit 0.33), peaks at 0.69 around the 40–50% mark, and decays to ~0.44 in the last quarter.
Per sentence 0.39 / **0.64** / 0.41; about 54% of the accumulated margin lands in sentence 2.
By theme: homecoming loudest and most peaked (sentence-2 hit 0.80); sacrifice the most sustained
and the only theme still readable in sentence 3; betrayal weakest and collapsing back.

**Era locks in early, because it is lexically marked early.** Era is at hit 0.62 in the first
position bin while theme is at chance, and stays flat (0.62–0.78); its aggregate beat-to-beat
derivative is zero within error. The exception exposes the confound: 1920s passages start below
chance and climb monotonically because they open on unmarked modern prose and become
period-specific only when *Packard* or *speakeasy* arrive.

**Mood is last-token, reproduced at per-token resolution.** Mood is at chance over the first
third of the sentence and plateaus near 0.59; final token 0.61 vs 0.48 for the mean over tokens.
The control is in the same run: on the theme grid the ordering reverses (theme last-token 0.47 vs
mean 0.53; era 0.53 vs 0.72), so the last-token advantage is mood-specific, not an edge artefact.
Comic is the extreme (below chance early, 0.94 at the end: the joke is the last clause); dread
peaks mid-sentence; tender is flat.

**The beat-to-beat derivative has one shape for theme.** Positive then negative, a single hump on
the middle beat, shared in sign by all three themes and separated by magnitude (homecoming
+0.116/−0.135, betrayal +0.053/−0.050, sacrifice +0.050/−0.015). For era it is flat in aggregate.
Differentiation kills era's large constant and keeps exactly what distinguishes the themes, which
is the calculus operator doing what §2.3 of `docs/ALGEBRA.md` says it should.

**Caveats.** One 1.5B model; directions from nine mean-pooled vectors per fold; noisy per-token
cosines; a naive sentence split drops 2 of 36 passages from the per-sentence analysis.

## 2026-09-11 (hour 18) — Absential ring by decoder geometry: falsified as "implied but absent content"

**Setup** (agent run; `scripts/absential_ring.py`, `scripts/ndif_absential_probe.py`,
`results/absential_census.json`, `results/absential_probe_gemma9b.json`,
`results/notes/absential.md`). Gemma Scope 16k, layer 20, Gemma-2-9B-it. Active set A = features
firing on any non-BOS token; formatting features (generality > 0.9) excluded. Ring R = inactive
features whose decoder direction has cosine ≥ 0.40 to some active content feature. The inactive
max-cosine distribution is smooth with no shoulder, so τ = 0.40 is the 97.5th percentile, a choice
rather than a discovery.

**Census (72 passages).** Theme grid: |A| = 1809, |R| = 386; mood grid: 1144 / 258. The ring is
the dictionary's long tail: mean generality 0.09, 70% rare, 38% never fire on any reference.
Two negatives: a ring built from a size-matched *random* active set has the same size and an
identical generality profile, so census statistics cannot tell a real ring from an arbitrary one;
and ring composition does not track theme (within-vs-across Jaccard +0.002; no feature is in the
ring of every passage of one theme and none of the others). The top-cosine ring members are
near-duplicates of function-word features (*the*, *his*, *of*), not thematic near-misses.

**Probe (12 passages, 72 NDIF jobs).** Top-8 ring decoders summed, renormalized to 0.15 × the
residual norm, added at block 20 at every position. Controls: generality-matched inactive non-ring
features (`ctrl`), and generality- and coherence-matched (`ctrl2`, since the ring's directions are
mutually similar and their sum does not cancel).

| | ring | ctrl | ctrl2 |
|---|---|---|---|
| Δ log-prob of own span (nats/token) | −0.027 | −0.037 | −0.037 |
| KL at final position, mean | **0.063** | 0.008 | 0.022 |
| ring has lower KL than control | — | 1/12 (sign p = 0.006) | 3/12 (p = 0.15) |

**Reading.** The hypothesis that decoder-adjacent inactive features are the model's own implied
but absent content, and that perturbing along them is less disruptive than random, is
**falsified**: fit is a wash, and the ring is *more* disruptive, 7.6× the matched control's KL.
The defensible residue is narrow: at matched norm and generality, directions adjacent to the live
set carry more leverage on the next-token distribution, most parsimoniously because they amplify
already-active near-duplicate features. Decoder geometry is the wrong adjacency for Deacon's
absence. **What would move this:** define the ring from features that fire on the model's own
*continuation* of the passage rather than from geometry, which is the withheld-betrayal test named
in VISION.md and comes with semantics attached.

## 2026-09-11 (hour 19) — Commutator trajectories: factor patches do not commute under generation, but divergence is bounded and one factor dominates by depth

**Setup** (agent run; `scripts/ndif_commutator.py`, `results/commutator_gemma9b.json`,
`results/notes/commutator.md`). Gemma-2-9B-it via NDIF. Pairs (era, theme) and (era, voice), all
3 × 3 level combinations, two neutral prompts, greedy 60-token generations under base, A alone,
B alone, AB (A at block 14 + B at block 20) and BA (B at 14 + A at 20), patch re-applied every
step. Divergence measured two ways: token-level (first differing token, Hamming) and readout-level
(per-token block-20 re-reads of both texts projected onto the two factor directions, difference in
units of the base continuation's sigma). Regime rule pre-registered in the script's docstring and
committed before the run: commute / drain / crystalline / structured / noise, in that order.

**Findings.**
- **Order matters under generation.** 32 of 36 cases produce token-different continuations; the
  median first divergence is token 16.5 (era × theme) and 11 (era × voice) of 60; mean Hamming
  0.59 / 0.64. Hour 6's selector-level "order barely matters" (rank gap 0.5) does **not** survive
  the trajectory test: near-commutativity is a gauge-level law, not an engine-level one.
- **But divergence is bounded.** Mean readout-difference curve 1.1–1.2σ, late window 1.3–1.5σ;
  nothing runs away. Regime counts (18 cases per pair): era × theme noise 8, crystalline 4,
  structured 4, commute 1, drain 1; era × voice noise 8, crystalline 4, structured 3, commute 3.
  "Noise" here means unstructured within the plane of the two patched directions.
- **One factor dominates the surface text regardless of layer order**, and the dominance follows
  depth. Token overlap with the single-patch continuations: era-only 0.48 / 0.44 vs theme-only
  0.20 / 0.18 (era × theme); voice-only 0.35 / 0.32 vs era-only 0.15 / 0.19 (era × voice). So
  **voice > era > theme**, matching lexical > mid-stack > late/distributed, and not a
  "block-14 wins" artefact since it holds under both orderings.
- Drain is real but rare: one case (1920s × sacrifice) keeps 73% of base tokens under AB vs 42%
  under BA.

**Reading.** This is groovy-commutator's instrument applied to activations, and it sharpens L2 of
`docs/ALGEBRA.md`: directions commute as *gauges* (the selector's order gap is half a rank) and do
not commute as *engine inputs* (the generated text differs from token ~11–16 on). The divergence
is bounded and its magnitude is set by the shallower factor. That is a better statement of "small
holonomy" than the one from hour 6: the holonomy is small in readout space and large in token
space, and the two are different objects.

**Caveats.** Greedy decoding amplifies sub-threshold logit differences; two prompts, so half the
aggregate regime labels rest on a 1–1 tie; one layer pair, one scale, one model; readout axes are
only the two patched directions; **no random-direction null yet**, which with more prompts and a
layer sweep is the cheapest next control.

## 2026-09-11 (hour 20) — Generative relation lens at forty domains: better than random, not source-specific

**Setup** (agent run, terminated by a rate limit after the sweep finished; analysis by the
integrator). `scripts/stage4b_relation_v2.py`, `results/stage4b_qwen1.5b_v2_relation.json`.
Qwen2.5-1.5B. Twelve held-out domains (the original eight plus four generated), six role pairs
(four easy, two hard), operator fit on the other 39 domains at layer 16 and patched at layer 16:
`dir_T + λ·pred` vs `dir_T` alone. Controls: random direction of the same norm (×2) and the
operator applied to the wrong source role, rescaled. Readout: extra log-prob gain on the target
span beyond the role-only patch. ‖pred‖ ≈ 1.3 ‖dir_T‖; cos(pred, true residual) = 0.29.

| λ | relation | wrong source | random | role-only rank |
|---|---|---|---|---|
| 0.5 | **+0.18** nats | +0.21 | −0.10 | 1.29 |
| 1.0 | **+0.04** | +0.14 | −0.47 | 1.29 |

Relation beats random in 46 of 72 (domain, pair) cases at λ = 1. By pair at λ = 1 (relation /
random): embedded→from_above +1.14 / +0.11; embedded→disturbance +0.77 / −0.19; the four
from_below/objectified pairs are negative for both.

**Reading.** At forty domains the operator's prediction is now a *helpful* patch on average
(+0.18 nats vs −0.10 for random at λ = 0.5; at seven domains it was −0.02 vs −0.19). But the
wrong-source control does as well or better. The benefit comes from the operator's output
direction, which is largely independent of which source it is fed, not from a source-specific
mapping. So the forty-domain operator is usable as a patch in the weak sense (it raises the target
span's likelihood beyond the role direction alone) and not in the strong sense (it does not carry
source-specific content into generation). The selector half of the strong hypothesis holds
(hour 16); the generative half is still open, and this is the cleanest statement yet of what is
missing: source-specificity under patching.

## 2026-09-11 (hour 21) — Continuation-defined absential test: null at n = 8 on the confound-free readouts

**Setup** (agent run, terminated by a rate limit after the data were collected; pre-registered
design in the script docstring; analysis by the integrator). `prompts/absential_v1.json` (8 items ×
3 variants, written by Claude: withheld / delivered / neutral, differing in one sentence each),
`scripts/ndif_absential_continuation.py`, `results/absential_continuation_gemma9b.json`.
Gemma-2-9B-it, block 20, theme directions from the theme grid.

**Readout A, representational.** Last-token projection onto the withheld theme's direction:
withheld 0.64, neutral −0.81, delivered 12.0; withheld > neutral in 4 of 8 (sign test p = 1.0);
last-token cosine 6 of 8 (p = 0.29). The mean-over-span projection is withheld 6.6 vs neutral 1.4
in 8 of 8 (p = 0.01), but that measure is confounded: withheld and neutral differ exactly in the
setup sentence, so the span mean reads the setup text itself, not the absence. The confound-free
readout (last token) is null.

**Readout C, feature-level.** Absential set = 16k-dictionary features active in the model's own
continuation but not in the passage (≈570 per item). Their pre-activations at the passage's last
token: withheld −6.97 vs neutral −6.87; fraction within 1 unit of threshold 0.000 vs 0.002. Null.

**Readout B, behavioral** (continuation projections and a lexical check): raw continuations are
saved in the JSON; the aggregate was not computed before the agent was cut off and is left for a
cheap follow-up.

**Reading.** With eight authored items, a withheld part leaves no detectable trace at the last
token or in near-threshold features. Either absences are not represented this way on this model,
or n = 8 with one author cannot see it. The delivered variant's projection (12.0 vs 0.6) shows the
directions work; the test is sensitive enough to see presence and did not see absence. Recorded as
a null, not a falsification: the design is right, the sample is small.

## 2026-09-11 (hour 22) — Commutator controls: divergence under generation is generic; dominance is real; regimes are noise

**Setup** (agent run; `scripts/ndif_commutator.py` extended with `--null`, `--layers`, two more
prompts; `results/commutator_gemma9b_v2.json`, `results/notes/commutator_v2.md`; the hour-19 rule
and outputs untouched and reproduced exactly). Gemma-2-9B-it. Null: matched-norm random direction
pairs through the identical AB/BA protocol. Four prompts. Second layer pair (16/24) for era × theme.

**Null vs factor** (36 factor cases vs 16 null; Mann–Whitney):

| | Hamming | first-divergence token | curve (σ) | base overlap |
|---|---|---|---|---|
| factor pairs | 0.62 | 15.0 | 1.13 | **0.14** |
| random pairs | 0.56 | 20.0 | 1.05 | **0.32** |
| p | 0.59 | 0.50 | 0.73 | **0.002** |

Random patch pairs diverge just as much, just as early, and produce every regime the factors do,
drain included (more often: 19% vs 3%), while staying fluent. The one measure that separates
factor from random is displacement from the prompt prior (overlap with the base continuation
0.14 vs 0.32): the factor directions *steer*; the commutator protocol is not where that shows.

**Regimes over four prompts.** No level combination has all four prompts agreeing (0 of 18); a
permutation test finds no structure (era × theme p = 0.16, era × voice p = 0.57). The five-regime
taxonomy does not transfer from cellular automata to greedy decoding at this sample size.

**Dominance at a second layer pair.** Era still dominates theme under both orderings at blocks
16/24 (overlap with era-only 0.37 / 0.26 vs theme-only 0.17 / 0.17), so that leg is not a
block-14 artefact; the margin shrinks when era is the later patch, consistent with era living
mid-stack. Voice > era was not re-run at the second pair (budget), so the full ordering rests on
one layer pair.

**Reading.** Hour 19 downgrades to one claim: **the shallower factor dominates the surface text
regardless of order**, and that is real. "Factors do not commute under generation" is true but
not about factors: any two matched-norm patches diverge under greedy decoding from token ~15,
boundedly. The predictions written before this run (RESULTS hour 22 conversation) held.

## 2026-09-11 (hour 23) — External-author replication: GPT-written grids reproduce the factor results

**Setup.** Two grids written by GPT from a spec (no Claude-written spans shown; scenes disjoint from
every Claude grid): `prompts/narrative_factors_gpt_v1.json` (river rescue, accusation at a table,
wound dressed, bargain struck × 3 eras × 3 voices) and `prompts/narrative_theme_gpt_v1.json`
(inheritance divided, boat launched, sick animal, contest entered × 3 eras × 3 themes, three
sentences each). Same scripts, same layers, same leave-one-scene-out protocol, Qwen2.5-1.5B.
`results/gpt_grids_run.log`, `results/stage6_qwen1.5b_*_gpt_*.json`, `results/stage7_qwen1.5b_shift_gpt.json`.

| measure | Claude grids | GPT grids |
|---|---|---|
| era lens (era × voice), rank/3 | 1.28 | 1.31 |
| voice lens, rank/3 | 1.31 | 1.39 |
| era + voice composed, rank/9 | 2.03 | 2.33 |
| cross-talk era→voice / voice→era | 0.14 / 0.16 | 0.14 / 0.13 |
| theme lens, rank/3 | 1.25 | 1.25 |
| era + theme composed, rank/9 | 2.22 | **1.56** |
| cross-talk era→theme / theme→era | 0.12 / 0.17 | **0.07 / 0.14** |
| era shift: address moved / theme kept (unpatched theme) | 0.89 / 0.81 (0.78) | **0.94 / 0.86 (0.86)** |
| voice decodability at layer 12 | 0.92 | 0.75 |
| theme decodability at layer 20 | 0.78 | 0.86 |

**Reading.** Every claim from hours 6, 10, and 14 reproduces on text I did not write or review.
The theme results are stronger on GPT's passages (composition 1.56 vs 2.22; theme kept at 0.86,
exactly the unpatched readout), plausibly because GPT's themes are carried across all three
sentences as the spec demanded. Voice is less lexically extreme in GPT's writing (decodability
0.75 vs 0.92) and the voice lens is correspondingly a little weaker, which is the right direction
for a less exaggerated register. The single-author caveat is now a single-*species* caveat: two
model authors agree; no human-written grid yet.

## 2026-09-11 (hour 24) — The relation operator is source-specific at the selector level

**Setup** (agent run, pre-registered in `scripts/stage3_source_specificity.py`; the agent was
stopped before writing its note; analysis by the integrator). Same protocol as hour 16 (40
domains, role-centered, dual-form affine, ridge 10, leave-one-domain-out). New comparison: for each
held-out prompt and pair S→T, feed the fitted operator the prompt's true source residual and,
separately, each of the prompt's five *other* role residuals (same prompt, so domain address is
held fixed). Paired win = fraction of (prompt, pair, wrong role) triples where the true source
ranks the target higher; ties count half. Gate for the patch rerun, fixed in advance: paired win
≥ 0.60 at layer 16. `results/stage3_source_specificity.json`.

| layer | role_rank, true source | role_rank, wrong source | paired win | contrastive fit: rank / win |
|---|---|---|---|---|
| 8 | 2.10 | 2.77 | 0.62 | 3.11 / 0.50 |
| 12 | 1.90 | 2.63 | 0.64 | 2.90 / 0.51 |
| 16 | 1.73 | 2.75 | 0.68 | 2.63 / 0.57 |
| 20 | 1.86 | 2.79 | 0.67 | 2.63 / 0.56 |
| 24 | 2.39 | 2.79 | 0.58 | 3.11 / 0.48 |

**Reading.** The operator's prediction depends on which role it is fed: the true source ranks the
target a full rank better than a wrong role of the same prompt (1.73 vs 2.75 at layer 16), and
wins 68% of paired comparisons, clearing the gate. **Source-specificity exists at the selector
level.** The contrastive fit (predict target − source) is worse than the residual fit at every
layer, so the plain affine map is the right parameterization at this data size. Hour 20's patch
result, where the wrong source did as well, is therefore a property of the patch protocol (a single
rescaled wrong source; the readout is the target span's likelihood, which the operator's bias
already raises) and not of the operator. A refined patch test, all five wrong sources at natural
norm, is running.

## 2026-09-11 (hour 25) — Refined patch test: source-specificity under patching is marginal

**Setup.** `scripts/stage4c_relation_wrong_sources.py`, `results/stage4c_qwen1.5b_v2_wrong_sources.json`.
Qwen2.5-1.5B, operator fit at layer 16 on 39 domains (role-centered, ridge 10), patched at layer
16 with λ = 0.5 on top of the role direction. Eight held-out domains, four easy pairs. Conditions:
relation (true source), every one of the five wrong sources of the same prompt at its **natural
norm** (mean norm ratio 0.96, so no rescaling artefact), one random direction of matched norm.
Readout: extra log-prob gain on the target span beyond the role-only patch.

| condition | extra gain (nats) | n |
|---|---|---|
| relation (true source) | **+0.176** | 32 |
| wrong source (all five roles) | +0.114 | 160 |
| random, matched norm | −0.151 | 32 |

Paired: the true source beats a wrong source in **87 of 160 (54%)**, mean difference +0.06 nats.
By wrong role, the relation's margin ranges from −0.11 (fed `embedded`) to +0.27 (fed
`new_subject`).

**Reading.** Under patching, source-specificity is present but marginal: a 54% paired win and
0.06 nats, against a 68% win and a full rank at the selector level (hour 24). The operator's
output direction carries most of the patch benefit (+0.11 for any source vs −0.15 for random);
the source-dependent part survives the likelihood readout only weakly. Consistent with law L3 of
`docs/ALGEBRA.md`: what the selector sees cleanly, the engine shows faintly. Recorded as partial:
the relation operator is a source-specific selector and a weakly source-specific patch.

## 2026-09-11 (hour 26) — Abstraction ladder extended: the fixed points are trivial, and the flow is an ordering, not an optimum

**Setup** (agent run; `scripts/sae_ladder_v2.py`, `scripts/ndif_gen_broad.py`,
`results/sae_ladder_v2.json`, `results/broad_corpus.json`, `results/notes/sae_ladder_v2.md`,
`results/figures/sae_ladder_v2.png`). Gemma-2-9B-it; Gemma Scope 16k dictionaries at layers 9, 20,
31 (only the two new 16k files downloaded); a broad 111-passage, 12-genre reference corpus
generated by Gemma for generality; an abstraction flow at layer 20 keeping only features with
generality ≥ g_k and reconstructing each of the 72 narrative passages from the survivors.

**Layers.** Mean generality of the description's surviving features is flat with depth (0.20 /
0.18 / 0.20 at layers 9 / 20 / 31) while the width step at layer 20 moves it 3×: width, not depth,
carries the ladder by this measure. Depth changes what the strongest features anchor on: at layer
9 *Federation / fleet / Enterprise / Picard* (0.31) and *Earl Grey / tea* (0.17); at layer 31
function words and relational tokens at 0.72–0.85.

**Broad corpus.** Against 12 genres instead of narrative only, generality is 0.19 (16k) vs 0.10
(131k), a 1.9× gap where hour 15 had 2.9×: part of the hour-15 effect was narrative-only
reference bias (fine features anchored on other registers looked rare when they were merely
off-domain). The merge test holds: 13 of 15 nearest-narrow features more general under both
corpora.

**Flow.** Mean pairwise distance between passages collapses 24× as g_k rises, monotonically, no
plateau. Theme purity of 8-nearest-neighbours falls 0.63 → 0.30 (chance 0.11); era purity 0.53 →
0.36 (chance 0.32) and is gone by g_k = 0.7: **era's structure dies before theme's.** The
marginal theme peak at g_k = 0.01–0.02 (0.64) is below the raw residual's 0.65, so there is no
intermediate optimum. The 58 features surviving g ≥ 0.95 are ten copies of a passage-initial
position feature, sentence periods, function words, and generic past-tense narrative verbs.

**Reading.** Half of the corrected expectation is confirmed: the fixed points of the abstraction
flow are trivially general, and nothing that reads as an archetype is among them. The other half
is not: separation is best at the bottom of the ladder and decays, so what the flow carries is an
**ordering** of what dies first (era before theme), not an optimum in the middle. Law L7's
"archetypes as attractors" is falsified in this realization. The hour-15 headline (narrow keeps
general; merges go up) survives the broad corpus with a smaller effect size.

**Cost note.** This agent spent ~300k tokens and ~270 NDIF jobs; two full token-fetch runs were
lost to hung jobs before it added per-text checkpointing (`scripts/ndif_tokens_resume.py`). Future
briefs should point agents at the resumable fetch.

## 2026-09-12 (hour 27) — Generation-level recomposition: the era shift does not move generated text, on 70B or 9B (agent)

**Question.** Hour 14 showed the era shift moves the era *readout* (0.88–0.89) while keeping theme. Does the same patch move the era of the *generated continuation*, and does the largest NDIF model do it more cleanly? Pre-registered prediction in `scripts/ndif_recompose_gen.py`: 70B moves more cleanly than 9B.

**Setup.** `prompts/narrative_theme_v1.json`, chat-templated prefix (raw prompts made both instruct models answer comprehension questions; a pilot switched form before measurement). Llama-3.1-70B-Instruct via NDIF, patch at block 26, read at 40 (80 blocks, d = 8192); Gemma-2-9B-it patch 14, read 20. 144 generations each at scale 1.0; nine Gemma generations lost to empty NDIF payloads. The random condition has no target, so the fair control is "leaves e1". Lexical check: does the shifted continuation gain its target era's vocabulary?

| | Llama-70B-Instruct | Gemma-9B-it |
|---|---|---|
| era reads as target under shift | 0.14 (n = 72) | 0.27 (n = 67) |
| leaves e1: base / random / shift | 0.14 / 0.19 / 0.21 | 0.22 / 0.31 / 0.48 |
| of the leaving, on target | 0.67 | 0.56 |
| theme kept: base / shift / random | 0.61 / 0.54 / 0.56 | 0.53 / 0.60 / 0.53 |
| lexical era check, shift → target | 0.00 | 0.00 |

**Reading.** Near-null. Against 0.88 at the representational level, the generated text stays in its original era on both models; not one continuation on either model gained target-era vocabulary. The 70B does it *less* than the 9B, so the prediction is falsified. Theme is "kept" identically under every condition and carries no weight. The scale-1.5 arm was not run (budget), so "not fixed by scale" is inferred, not measured. Two infrastructure fixes reached the 70B: `.cpu()` in the layer stack on the model-parallel host, and per-span checkpointing with `retry_job`.

**Consequence for the algebra.** L4 (address/form separation) is a gauge law only. The engine does not carry the shifted address into text at either scale, which is the same boundary as L3 with a different factor. Files: `scripts/ndif_recompose_gen.py`, `results/recompose_gen_{llama70b,gemma9b}.json`, `results/notes/recompose_gen.md`. Cost: ~230k agent tokens.

## 2026-09-12 (hour 28) — Parameterized time translation: a shared clock exists and selects; subject-relative timescales do not appear (agent)

**Spec and predictions** were written before the run: `docs/specs/time_translation_v1.md`. Eight subjects (street, mountain, orchard, mayfly, asteroid, river, London from 1800, an invented city), t0 plus eight log-spaced Δt from one day to a million years (the spec's tenth interval, "1 month", was dropped to reconcile a count contradiction in the spec), three paraphrases, plus a phrase-only control (interval phrase on the unchanged t0 state). Qwen2.5-1.5B, layers 0/8/14/20/27, state span mean-pooled.

| measurement (layer 14 unless noted) | value |
|---|---|
| mean ‖d‖, 1 day → 1 My | 12.7 → 14.8 (flat) |
| share of Σ‖d‖² explained by shared(Δt) | 0.53 (L0 0.31, L8 0.47, L20 0.56, L27 0.46) |
| Spearman(‖shared‖, log Δt) | +0.47 (U-shaped, minimum at 100 y) |
| adjacent-Δt cos / distant-Δt cos of shared | 0.89 / 0.64 |
| τ(s) by half-max | 1 day for every subject, every layer (degenerate) |
| orchard ‖d(1 y)‖/‖d(6 mo)‖, cos | 0.86, +0.56 (only subject that shrinks); mountain 1.12, +0.86 |
| real vs fictional residual cos | 0.36–0.52, never > 0.6; minimum at 1 week |
| ‖shared_exp‖/‖shared_ctrl‖ | 2.82 (L0 control displacement exactly zero) |
| selector, LOO clock patch, Δt ≥ 1 y, rank of 9 | 3.88 vs 5.65 random (chance 5.0); L8 4.40 vs 5.96; 1.62 at 1 My |

**Grades.** P1 partial (variance and adjacency hold; monotonicity does not). P2 fell: the knee is undiscriminating, and under a fallback knee the mayfly lands at the slow end. P3 partial: the orchard is the only subject whose displacement shrinks at one year, but the six-month/one-year cosine is 0.56, not below 0.5. P4 fell: the real and fictional populations never align. P5 held on gain ranking. P6 partial: the shared displacement is 2.8× the phrase alone, but the cosine to the phrase direction is flat across depth rather than declining.

**Reading.** There is a shared, phrase-independent clock direction that the model computes from the state description, and it works as a selector, strongest at geological Δt. What is missing is the subject-relative part: residual curves are flat, so "the mountain's million years" is not a knee in this grid. **Confound:** the far-Δt passages share an erasure vocabulary across subjects, so the shared clock at 10 ky–1 My may be that vocabulary, and that is where the selector effect lives; the random control is worse than chance (5.65), so part of the gap is avoided disruption. Single model, single author. Files: `prompts/time_translation_v1.json`, `scripts/time_translation{,_selector}.py`, `results/time_translation_{measures,selector}.json`, `results/notes/time_translation.md`, five figures under `results/figures/time_translation_*.png`. Cost: ~170k agent tokens.

## 2026-09-12 (hour 29) — The generation boundary is a magnitude, not a wall: era shift at 3× re-imposed moves generated text (agent)

**Question.** Is hour 27's null under generation a matter of patch magnitude? Sweep scale and compare prefix-only patching with re-imposition at every decoding step, Gemma-2-9B-it via NDIF, same grid, prompts, layers (patch 14, read 20) and readouts as hour 27. Predictions in the script docstring: 2–3× lifts era-as-target above 0.5 at a prose cost; re-imposition at 1× lifts the lexical check off zero.

| condition | scale | n | era → target | leaves e1 | theme kept | lexical → target |
|---|---|---|---|---|---|---|
| hour-27 baseline (was re-imposition) | 1.0 | 67 | 0.27 | 0.48 | 0.60 | 0.00 |
| re-imposition | 0.5 | 67 | 0.21 | 0.37 | 0.61 | 0.00 |
| re-imposition | 2.0 | 59 | 0.58 | 0.76 | 0.54 | 0.18 |
| re-imposition | 3.0 | 55 | **0.84** | 0.91 | 0.53 | **0.30** |
| prefix-only | 1.0 | 72 | 0.24 | 0.43 | 0.61 | 0.00 |
| prefix-only | 2.0 | 72 | 0.36 | 0.61 | 0.57 | 0.08 |

**Reading.** At 3× with re-imposition the generated text reads as the target era in 0.84 of cases, matching the representational number from hour 14 (0.88), and 0.30 of continuations use target-era vocabulary ("starship", "airlock", "sword"), with theme kept at its baseline. Prose did not degrade at any scale; the cost is vocabulary bleed inside coherent sentences and a rising NDIF job-loss rate (5, 13, 17 of 72 at 0.5×, 2×, 3×; prefix-only lost none). Prediction 1 half-held (threshold crossed, no prose cost). Prediction 2 was mis-premised: hour 27's script already re-imposed at every step (`tracer.all()`), confirmed by a smoke test in which prefix-only reproduces the unpatched output exactly. So re-imposition alone does nothing at 1×; it helps once scale is raised (0.18 vs 0.08 at 2×). Hour 27's inference "not fixed by scale" is overturned for 9B; the 70B was tested at 1× only.

**Consequence for the algebra.** L3 stands as stated (gauge results do not transfer at matched norm) but its boundary is now quantified: the engine needs roughly 3× the gauge-level norm, applied throughout decoding, to write the address. L4 crosses into generation under those conditions. Files: `scripts/ndif_recompose_sweep.py`, `results/recompose_sweep_*.json`, `results/notes/recompose_sweep.md`. Cost: ~125k agent tokens.

## 2026-09-12 (hour 30) — Vocabulary-matched time grid: the shared clock is not the shared vocabulary (agent, finished by hand after a container restart)

**Question.** Hour 28's shared clock might be the erasure/geological vocabulary that all far-interval passages shared. Rewrite the Δt ≥ 100 y states so no content word appears for more than two subjects (checker: `scripts/time_translation_vocab_check.py`, zero violations in v2 vs up to five subjects per word in v1) and rerun everything unchanged. Prediction, written first: shared variance drops below 0.4; the 1 My selector worsens past 3.0 but beats random.

| layer 14 | v1 | v2 |
|---|---|---|
| shared variance fraction | 0.533 | 0.545 |
| Spearman(‖shared‖, log Δt) | 0.47 | 0.68 |
| ‖shared_exp‖/‖shared_ctrl‖ at 1 ky / 10 ky / 1 My | 2.6 / 2.6 / 2.9 | 3.2 / 3.1 / 3.1 |
| cos(shared_v1, shared_v2), far Δt | | 0.86–0.90 |
| selector gain rank of 9, Δt ≥ 1 y: clock / random | 3.88 / 5.65 | 3.62 / 4.83 |
| at 1 ky / 10 ky / 1 My: clock | 3.12 / 3.25 / 1.62 | 1.38 / 2.75 / 3.25 |
| τ(s) | 1 day, all | 1 day, all |

**Grades.** "Shared fraction < 0.4" fell: it rose. "1 My > 3.0 but < random" held (3.25 vs 4.38), but the effect moved to 1 ky rather than shrinking, and the overall selector improved. **Reading.** The clock survives the removal of its suspected lexical cause, points the same way (cos ≈ 0.9), and is more monotone in log Δt than before. Hour 28's confound is closed; the shared clock stands as a computed, phrase-independent direction. Subject-relative timescales remain absent. The v1 vs v2 random controls differ (5.65 vs 4.83), so the random baseline is noisy at n = 8 per Δt. Files: `prompts/time_translation_v2.json`, `scripts/time_translation_vocab_check.py`, `scripts/_build_v2_states.py`, `results/time_translation_v2_*.json`, `results/time_translation_v1_v2_shared_cos.json`, `results/notes/time_translation_v2.md`, five figures. Cost: ~150k agent tokens; the agent's final write-up was lost to a container restart and written from its saved outputs.

## 2026-09-12 (hour 31) — Time grid on Gemma-9B: the shared clock is model-invariant; subject clocks are still absent (agent)

**Question.** Does the subject-relative part of time translation appear at 9B? v2 grid (480 passages) extracted on Gemma-2-9B-it via NDIF at layers 9/20/31, measurements 1–7 unchanged. Prediction, written first: shared fraction 0.4–0.6 and Spearman ≥ 0.6 at layer 20; at least 4 of 8 subjects get τ > 1 day, mayfly/street short, mountain/asteroid long.

| quantity | Qwen-1.5B L14 | Gemma-9B L20 |
|---|---|---|
| shared variance fraction | 0.545 | 0.478 |
| Spearman(‖shared‖, log Δt) | 0.68 | 0.68 |
| adjacent / distant cos | 0.88 / 0.59 | 0.87 / 0.61 |
| ‖shared_exp‖/‖shared_ctrl‖ | ~3.1 | 1.67 |
| τ(s) | 1 day, all 8 | 1 day, all 8, all layers |

**Grades.** Clock replication held on both numbers. Subject timescales fell: 0 of 8, no ordering to grade. **Reading.** The shared clock's variance share, monotonicity, and cosine geometry are the same across a 6× parameter jump and a different family; it is a computed direction, not a small-model artifact. The phrase-only ratio is weaker at 9B (1.67), still above 1 at every Δt. Subject-relative timescales are absent at both scales, which moves the question from model size to the probe: state-span mean pooling under a fixed template may average out exactly the subject-specific change. Files: `scripts/ndif_time_translation_extract.py`, `results/time_translation_gemma_measures.json`, `results/notes/time_translation_gemma.md`, `results/figures/time_translation_gemma_*.png`; `scripts/time_translation.py` gained `--stacks/--out/--suffix`. Cost: ~115k agent tokens, 15 min wall.

## 2026-09-12 (hour 32) — The subject-clock probe was reading its own noise; and the time grid has a lexical confound (agent, spec `docs/specs/subject_clocks_v1.md`)

**Question.** Hours 28, 30 and 31 reported "subject-relative timescales absent" on two models. The spec's section 0 predicted that result was an artifact: a 1536-dimensional residual norm built from three-paraphrase means sits at its own noise floor, and a half-max knee fires on the first point of any flat noisy curve. Piece 1 tests that, then re-runs the question with a floor-referenced estimator and a knee-free discrimination test. Qwen2.5-1.5B, 504 passages, layers 0/8/14/20/27.

**The floor (P1, held).** The hour-28 probe reproduces exactly (layer 14: ‖d̄‖ 13.3, ‖shared‖ 10.6, ‖resid‖ 8.1 against the old 12–16 / 9–12 / 7–11). The paraphrase noise floor is F = 7.2. Residual-to-floor ratio is 1.12 at 1 day and 1.39 at 1 My; 7 of 8 subjects sit inside 0.8–1.3 of the floor for every Δt ≤ 1 y at four of five layers, and nothing anywhere exceeds 1.8×. Run end to end on Gaussian noise the same pipeline returns ratios averaging 0.95 and "no signal" everywhere, so the estimator is calibrated. **Hours 28, 30 and 31 did not measure an absence of subject clocks. They measured an instrument at its floor.**

**With a floor-referenced τ.** C1 last-token, layer 14: mayfly 1 week; orchard, street, river, mountain and the fictional population 1 day; real population 1,000 y; asteroid no signal. Determinate for 7 of 7 changing subjects with signal-to-noise 5.3–7.2, so the readout is strong — but the ordering is not subject-like: 1 of 9 predicted ordered pairs, 2 of 8 block matches. Model-supplied states (C3, the same t0 text repeated after the interval phrase) give no signal on all 8 subjects at every layer.

**Discrimination without knees.** Within-subject Spearman 0.77 vs shared-direction 0.78 at layer 14; within beats shared for only 3 of 8 subjects. **At layer 0 both are 1.00.** The v2 state texts restate the interval ("*One day later* the mayfly is dead"), so Δt is lexically recoverable before any computation. This is the dominant confound in the grid and it also applies to hours 28, 30 and 31: the phrase-only control isolates the interval phrase in the interval span, but not its restatement inside the state span. The shared-clock numbers there are not clean.

**Other grades.** P2 fell (partial: τ determinate 7/7, asteroid null, mayfly ≤ 1 week all hold; ordering does not). P3 fell with a twist: per-subject directions dominate at 9 of 9 intervals, but flat in Δt, which is subject identity rather than a clock; an added Δt-centred test gives 4 of 9 at Δt ≥ 100 y, p < 0.001, so there is direction-level subject specificity with no subject-ordered timing. P5 fell. P6 partial (layer 0 exactly zero as predicted; peak at 20–27, not 14–20). **Power rule:** r = 6.37 at layer 14, above the 4.1 threshold, so the grid is adequate at three paraphrases and piece 2 must not write more.

**Consequence.** Three logged "negatives" are withdrawn as untested, and one standing positive (the shared clock) is now qualified by a lexical confound. Any Gemma re-run needs a v3 grid whose state texts do not restate the interval. Files: `scripts/subject_clocks_{build,report}.py`, `scripts/subject_clocks.py`, `prompts/subject_clocks_v1.json`, `results/subject_clocks_measures.json`, `results/notes/subject_clocks.md`, 8 figures. Cost: ~160k agent tokens, 35 min wall.

## 2026-09-12 (hour 33) — The 70B does not cross the generation boundary the 9B crosses (agent)

**Question.** Hour 29 found that the era shift moves generated text on Gemma-2-9B-it at 3× norm re-imposed at every decoding step (0.84 era-as-target, lexical 0.30). Does Llama-3.1-70B-Instruct cross the same threshold? Patch block 26, read 40, chat-templated, greedy, 72 generations per scale. Prediction in the script docstring: yes at 3.0, with a lower lexical rate than Gemma because the larger model's prior resists.

| model | scale | era → target | leaves e1 | theme kept | lexical → target | jobs lost |
|---|---|---|---|---|---|---|
| Llama-70B-Instruct (hour 27) | 1.0 | 0.14 | 0.21 | 0.54 | 0.00 | 0/144 |
| Llama-70B-Instruct | 2.0 | 0.28 | 0.39 | 0.54 | 0.17 | 0/72 |
| Llama-70B-Instruct | 3.0 | 0.43 | 0.54 | 0.53 | 0.25 | 0/72 |
| Gemma-9B-it (hour 29) | 1.0 | 0.27 | 0.48 | 0.60 | 0.00 | 5/72 |
| Gemma-9B-it (hour 29) | 2.0 | 0.58 | 0.76 | 0.54 | 0.18 | 13/72 |
| Gemma-9B-it (hour 29) | 3.0 | 0.84 | 0.91 | 0.53 | 0.30 | 17/72 |

**Grades.** The crossing prediction is refuted: at 3× the 70B reaches 0.43, below Gemma's *scale-2* value of 0.58, and never crosses 0.5. The lexical comparison held (0.25 vs 0.30), and by more than expected in proportion: the 70B's vocabulary resists more than its era readout does, so readout and surface text do not move in lockstep. Prose stayed fluent at both scales; the cost is vocabulary bleed. Zero job losses across 144 generations, against Gemma's rising loss rate at the same scales.

**Reading.** The magnitude threshold found in hour 29 is not a constant of the method: the same patch at the same relative depth and the same multiple of norm moves a 9B and not a 70B. Either the larger model's prior is harder to displace, or its era competence lives elsewhere. **Confound the agent flagged:** patch 26 / read 40 of 80 blocks is the same *depth fraction* as Gemma's 14 / 20 of 42, not the same absolute depth or the same mechanism; a layer sweep on the 70B is the missing control. This also sharpens what the scale-vs-tuning spec must test: bigger is not more steerable, so the 9B-instruct advantage of claim 7 may be tuning rather than size. Files: `results/recompose_sweep_70b_{2.0,3.0}.json`, `results/notes/recompose_sweep_70b.md`. Cost: ~115k agent tokens, ~2 h NDIF.

## 2026-09-12 (hour 34) — 405B is unreachable, and the random-direction control collapses on every Llama (agent, spec `docs/specs/scale_vs_tuning_v1.md`)

**Piece 1** of the scale-vs-tuning spec: smoke tests on four Llamas with a numeric 405B go/no-go, theme-grid extraction on Llama-3.1-8B / 70B / 70B-Instruct, the selector battery, plus a cheap layer sweep aimed at hour 33's depth-fraction confound. Piece 2 (generation arms) not run.

**405B: no-go, and permanently so for this key.** `ndif_smoke.py` fails deterministically, 3 of 3 attempts in ~0.2 s: "Model is not pinned and hotswapping is not supported for this API key". A hard server refusal, not a timeout, and it contradicts our own `results/ndif_pinned.txt`, whose "PINNED RUNNING" entry for 405B is stale. **Consequence:** the spec's Outcome B escape clause, "or sufficient scale substitutes for tuning", cannot be tested at all. If piece 2 lands in Outcome B, the tuning claim stands at 70B as the largest reachable base model, full stop.

**Smoke latencies** (all fine): 8B 4.1 s, 70B 3.7 s, 70B-Instruct 3.8 s end to end. **Extraction:** 36 of 36 spans on each model, zero losses (8B 228 s, 70B 782 s, 70B-Instruct 720 s).

**Selector battery, and a new anomaly.** Theme decodability 0.89 / 0.89 / 0.92 on 8B / 70B / 70B-Instruct against Gemma 0.94 and GPT-J 0.83, so roughly invariant to size and tuning. Theme lens rank 1.22 / 1.11 / 1.06 of 3, all far under chance (2.0). **But the random-direction control lands just as low, 1.14–1.28, on all three models, where hour 13 found it near chance (~2.0) on Gemma and GPT-J.** On these models the lens is therefore not shown to be specific to the semantic direction at all. This is unresolved and it is a blocker: every selector-level claim on a Llama is suspect until it is explained, and if the same bug can occur elsewhere, the control in earlier hours needs re-checking too. Prediction P1's rank clause mostly held (70B-Instruct at 1.06 is just under the predicted band); its "random ≥ 1.9" clause is refuted on all three.

**Layer sweep (extra).** Selector at layer 14 (Gemma's absolute depth) vs 26 (fraction-matched) on both 70Bs: era and theme lens ranks differ by at most ~0.15. So hour 33's cap at 0.43 is not explained by picking the wrong depth for the *selector* signal, though it does not identify the right depth either. A generation-level layer sweep is the real test and belongs to piece 2.

**Next, in order.** (1) Diagnose the random-control collapse before spending anything on piece 2. (2) Correct `results/ndif_pinned.txt`. (3) Piece 2 with a generation-level layer sweep on the 70B pair, 405B dropped. Files: `results/scale_vs_tuning_selector_*.json`, `results/notes/scale_vs_tuning_p1.md`. Cost: ~130k agent tokens, ~45 min NDIF.

## 2026-09-12 (hour 35) — The shared clock survives removing the interval restatement, but a lexical floor remains (agent)

**Question.** Hour 32 found that in the v2 time grid the state texts restate their interval ("*One day later* the mayfly is dead"), so Δt is recoverable at layer 0 (discrimination Spearman 1.00) before the model computes anything. Build a v3 grid in which no state text names, numbers, or paraphrases its interval, keep v2's far-interval vocabulary matching, and re-measure. Qwen2.5-1.5B, same suite.

**Leak check:** v2 flags 221 of 240 state spans; v3 flags 0 of 240, with the far-Δt vocabulary matching preserved.

| layer 14 | v2 | v3 |
|---|---|---|
| shared variance fraction | 0.545 | 0.507 |
| Spearman(‖shared‖, log Δt) | 0.683 | 0.667 |
| adjacent / distant cos of shared | 0.881 / 0.587 | 0.889 / 0.634 |
| phrase-control ratio | 2.99 | 2.70 |
| cos(shared_v2, shared_v3) at 100 y–1 My | — | 0.938–0.947 |
| discrimination Spearman at layer 0 (within / shared) | 1.00 / 1.00 | 0.723 / 0.710 |
| discrimination Spearman at layer 14 | 0.937 / 0.975 | 0.936 / 0.961 |

**Grades.** Three of four predictions held: the shared fraction stayed above 0.35, the Spearman above 0.5, and the direction itself is nearly geometrically unchanged (cos 0.94–0.95 at far Δt). The layer-0 prediction fell: discrimination dropped from 1.00 to 0.72 but not to chance.

**Reading.** The clock is not an artifact of restated intervals; removing every duration expression costs it almost nothing, and the direction found in the clean grid is the same direction. But a lexical floor remains that duration-word removal cannot close: far-interval states necessarily use a different register (denudation, cadastres, broods) from near-interval ones (unchanged, identical), so magnitude of change is readable from a bag of embeddings. That is realistic content rather than a bug, and it means no text-based time grid can drive layer-0 discrimination to chance. Any future version must either accept the floor and measure the *gain* over it with depth, or move to non-lexical manipulations.

Files: `prompts/time_translation_v3.json`, `scripts/time_translation_leak_check.py`, `scripts/time_translation_discrimination.py`, `scripts/_build_v3_states.py`, `results/time_translation_v3_measures.json`, `results/notes/time_translation_v3.md`, figures. Cost: ~300k agent tokens (480 hand-written leak-free passages), ~30 min compute.

## 2026-09-12 (hour 36) — Two bugs in the remote selector harness; hour 34's numbers withdrawn, the Llama lens is real after the fix (agent)

**Cause, definitive.** In `scripts/ndif_factors.py` the patch was written `B[l].output[0][:] = B[l].output[0] + v`. Under transformers ≥ 4.54 a Llama, Gemma or Qwen decoder layer returns a **bare tensor** `[batch, seq, d]`, so `output[0]` is *batch row 0*, not the hidden states. Hour 34 ran with `NDIF_CHUNK=9`, all nine candidates in one padded batch, so every patch — factor and random alike — touched exactly one of nine texts and left the other eight identical to base. Second bug: `rank = 1 + #{gain[c] > gain[target]}` returns rank 1 on ties, so the eight untouched candidates all read "rank 1". Together these **fix the expected rank at 11/9 = 1.22 for any direction whatsoever**, which is the entire hour-34 band of 1.00–1.28. Fingerprint: every cross-talk row in all five hour-34 files is exactly 0.25/0.25 (one combination moved, eight identical), where hour 13 on Gemma ranges 0.007–0.846. A third, minor bug: `padding_side` is `left` on all three remote models while the lead mask assumed right padding.

**Evidence.** On NDIF with Llama-3.1-8B, three texts in one batch: as-shipped gains `[-7.47, 0, 0]`; whole-tensor patch `[-7.47, -2.23, -6.79]`; layer output shape `[3, 18, 4096]`. Locally on Qwen2.5-1.5B the collapse reproduces on a non-Llama: correct patch gives factor 1.22/1.33 and random 2.00/2.11; row-0-only patch gives factor 1.11/1.17 and **random 1.31/1.14**. So the cause is the control's application, not a Llama property and not metric degeneracy.

**The missing arm.** No-patch baselines were never measured. They are 1.00 under the shipped strict comparison — doing nothing scores as a perfect selector — and 2.00 under mid-rank ties, with every row exactly tied. That arm is now mandatory in the script.

**Corrected Llama-3.1-8B at layer 10:** era 1.22, theme 1.33, random 2.11, no-patch 2.00, three-way composition 2.03 of 9, cross-talk 0.69/0.11 and 0.20/0.52. **The Llama lens is real and specific**; hour 34's claim that the control collapses on every Llama was an artifact of our own harness.

**Withdrawn:** all five hour-34 selector files (8B, 70B, 70B-Instruct and both layer-14 sweeps), all three cross-talk matrices from that hour, the "random control collapses on every Llama" finding, and the layer-sweep conclusion — so **hour 33's depth-fraction confound is NOT narrowed and remains open**. **Standing:** decodability 0.89/0.89/0.92 and the 405B no-go from hour 34; hour 13; every local battery (hours 6–11, 23, 28–32); and all generation-level NDIF results (hours 12–14, 19, 22, 27, 29, 31, 33), which all ran at batch 1 and never hit the bug. **One flag for later:** hour 8's `tense` random control reads 1.44, closer to the treatment than it should be.

**Fix applied** (commit `fe31abb`): a tuple-or-tensor `resid()` helper, mid-rank tie handling, a no-patch arm, and right padding. Piece 2 of the scale-vs-tuning spec must re-run the 70B pair's battery from scratch (no valid Llama-70B selector numbers exist), require factor, random and no-patch in every battery, and assert that the number of moved candidates equals the batch size. Files: `results/notes/random_control_diagnosis.md`, corrected `scripts/ndif_factors.py`. Cost: ~180k agent tokens.

## 2026-09-12 (hour 37) — First valid Llama-70B selector numbers: instruction tuning sharpens era, not theme (agent)

**Setup.** The fixed battery (hour 36) re-run on the matched pair Llama-3.1-70B and 70B-Instruct — same pretraining, tokenizer, depth and width, differing only in post-training, so the comparison isolates tuning from scale. No valid 70B selector numbers existed before this. Both stacks re-extracted. Layer 26 (the spec's depth) and layer 14 (Gemma's absolute depth, since hour 33's depth confound reopened when hour 34 was withdrawn). Rank of 3, chance 2.0.

| model @ layer | era factor / random | theme factor / random | composed of 9 |
|---|---|---|---|
| 70B @ 26 | 1.50 / 1.86 | 1.06 / 1.85 | 2.33 |
| 70B-Instruct @ 26 | 1.06 / 1.88 | 1.06 / 1.96 | 1.47 |
| 70B @ 14 | 1.53 / 2.12 | 1.33 / 1.82 | 2.89 |
| 70B-Instruct @ 14 | 1.17 / 2.29 | 1.39 / 1.75 | 1.58 |

**Plumbing verified.** No-patch reads exactly 2.00 in all four runs, and all eight cross-talk rows are diagonal-dominant (own-factor share 0.36–0.73), not hour 34's flat 0.25/0.25 fingerprint. A positive-control assertion was added to `ndif_factors.py`; its first threshold (every candidate must change) proved too strict, firing on occasional single-candidate bf16 ties, so it now tests the actual hour-36 signature (at most one candidate changed) and logs partial misses as warnings.

**Reading.** Instruction tuning sharpens the **era** selector at both layers (1.50 → 1.06 at layer 26, 1.53 → 1.17 at layer 14), and depth does not explain that gap. **Theme** is roughly tuning-invariant (1.06–1.39 either way), and the two layers are too close together to rule depth out for it. So the factors dissociate under tuning, which is the dissociation the scale-vs-tuning spec predicted, though at the selector level rather than in generation. Decodability 0.889 / 0.917 holds, as does the spec's base-vs-tuned clause; the theme-rank band and random floor at layer 26 are missed in the direction of a *stronger* effect.

**Note against hour 33.** The 70B's generation-level cap (0.43 era-as-target at 3× re-imposed, against Gemma's 0.84) sits alongside a 70B-Instruct era selector of 1.06, which is as sharp as any we have measured. Sharp selector, weak engine: another instance of the gauge/engine split, now at fixed size.

Files: `results/scale_vs_tuning_selector_70b*_fixed.json`, `results/notes/scale_vs_tuning_70b_fixed.md`, assertion in `scripts/ndif_factors.py`. Cost: ~170k agent tokens, under an hour of NDIF.

## 2026-09-12 (hour 38) — The clock is more than vocabulary, but it is order-invariant and does not transfer: the line closes (agent, spec `docs/specs/clock_depth_gain_v1.md`)

**Question.** After hour 35 an irreducible lexical floor sits under every time grid, so the clock had to be re-posed as a *gain over that floor as a function of depth*: is there more recoverable at layer L than from the words alone, and does the gain have the shape of computation? Piece 1, Qwen2.5-1.5B, four arms, all 29 layers, 960 prompts. The primary arm is **state text only, with no interval phrase**, because the planner noticed that in every earlier grid the state followed the phrase and any layer could simply copy Δt by attention.

**The floor and the gain.** Lexical floor ρ = 0.728 (static embeddings 0.722, bag of tokens 0.706, both 0.728). The spec's primary statistic G_res runs −0.32 at layer 0, crosses zero at layer 5, plateaus near 0.15–0.20 through layer 21 and **peaks at 0.297 at layer 24** with permutation z = 7.6. A bias-free rank-partial companion peaks at **0.566 at layer 13**, is already 0.32 after a single block, and never drops below 0.44 after.

**Signatures, mostly failing.** S1 (mid-stack peak then decline) fails both ways: the curve is a plateau, not a peak. S2 (intact beats word-shuffled) fails: shuffling the state text recovers 67–83% of the gain, and the structural residue, 0.076, is not distinguishable from zero (label-swap z = 1.21). S3 partial: the gain is largest where the floor is weakest as predicted (Spearman −0.787), but at the grid's ends rather than its middle. S4 fails decisively: a readout fit on state-only text transfers to the model-supplied arm at ρ = 0.150, cosine 0.078, squarely inside the region where hour 32's null stands.

**Copy inflation measured.** Hour 35's layer-14 discrimination of 0.961 falls to 0.522 under the same centroid readout once the interval phrase is removed (0.890 to 0.812 under ridge). A large part of what earlier grids measured was the model copying the interval from the prompt.

**The calibration check caught a broken statistic.** Layer 0 was verified identical to the static-embedding bag (cosine 0.999999999), yet G_res(0) = −0.32 with its permutation null at −0.586: the spec's primary statistic fails its own calibration, because residualising then refitting over-subtracts by a data-dependent amount when the two readouts' errors correlate. The agent diagnosed it, reported G_res in full anyway, and put an unbiased companion beside it (null mean ≤ 0.013 at every layer). A synthetic self-test of the estimator was run before any real data was touched, and arm D reproduces hour 35 to three decimals.

**Verdict.** The kill condition is not met: there is real gain over the lexical floor, far above noise. But the gain survives word-shuffling, so it is order-invariant, and it does not transfer to the case where the model must supply the change itself. That is a **computed register detector**, not a clock: the model reads "how much change is described here" from a bag of words, more accurately than the embeddings alone allow, and does not build a representation of elapsed time. The Gemma gate is **no-go** (it required S2, S3 and S4). This is the finding the planner predicted and it is the end of the line; the standing result is the shared-direction geometry of hours 28–35, now correctly named.

Files: `scripts/clock_gain*.py`, `prompts/clock_gain_v1.json`, `results/clock_gain_v1_*.json`, `results/notes/clock_depth_gain.md`, two figures. Cost: ~220k agent tokens, ~43 min compute.

## 2026-09-12 (hour 39) — Instrument audit: a padding bug corrupted 76% of hour 31; its numbers are replaced, its conclusion survives (agent)

Five unchecked risks surfaced by writing `docs/INSTRUMENTS.md`, worked hardest-first.

**(a) Hour 31 does not carry the hour-36 batch-row bug — and falls for a different reason.**
`tracer.invoke` does scope per passage (nnsight 0.7's `Batcher.narrow` slices the batch dimension),
confirmed on NDIF: position-stable last-token vectors match a batch-of-one extraction at cosine
0.99996 while cross-passage cosines sit at 0.73–0.76. **But the same test exposed a padding bug of
the same blast radius.** nnsight left-pads the six batched texts, and the script indexed spans
*absolutely* from the unpadded text, so every passage that was not the longest in its job read its
spans out of the padding. Batched-vs-single cosine for a 30-token-padded passage: 0.287 on the
interval span, 0.794 on the state span, against 0.999997 for the unpadded one. **363 of 480
passages corrupted (76%), and 232 had their entire interval span inside the pad block.** Worse, the
padding correlates with Δt, because jobs are consecutive grid items. Fixed with end-relative indices
plus a padding-side assertion, re-extracted (854 s) and re-measured:

| Gemma-9B, layer 20 | hour 31 (withdrawn) | corrected |
|---|---|---|
| shared variance fraction | 0.478 | 0.501 |
| Spearman(‖shared‖, log Δt) | 0.683 | 0.767 |
| adjacent / distant cos | 0.87 / 0.61 | 0.89 / 0.57 |
| phrase-control ratio | 1.67 | 2.50 (layer 31: 1.72 → 3.31) |

Every logged hour-31 number is replaced. The conclusion — the shared clock replicates on Gemma-9B —
survives on clean vectors, and is slightly stronger. Note the corrected phrase ratio of 2.50 is much
closer to Qwen's ~3.1, so the "weaker on Gemma" remark in hour 31 was an artifact of the bug.

**(b) The six remaining `output[0][:]` scripts are safe as run.** `ndif_generate`, `ndif_shift`,
`ndif_commutator`, `ndif_recompose_gen`, `ndif_recompose_sweep`, `ndif_absential_probe`, covering
hours 13, 14, 18, 19, 22, 27, 29: one prompt per job throughout, so the idiom never bit. All ported
to the `resid()` helper anyway; no numbers change. The two recompose scripts batch six invokes like
hour 31 and escaped its padding bug only because they pool right-aligned `[..., -k:, :]` — luck, now
documented.

**(c) Local tie-ranking never fired.** Mid-rank ties and a no-patch arm added to `stage5_factors`,
`stage6_factors` and `time_translation_selector`; hour 8's three-factor battery re-run (46 min):
era 1.25, voice 1.24, tense 1.03, composed 2.81 of 18, cross-talk matrix — identical to two decimals,
with no-patch reading exactly chance (2.00 / 2.00 / 1.50 / 9.50). **No logged local claim changes.**
Hour 8's flagged `tense` random control of 1.44 sits beside a no-patch of 1.50, so it is a
fluctuation, not an instrument failure; that flag is cleared.

**(d) The abstraction ladder's missing nulls, specified but not run:** a merge-test null against the
two dictionaries' *different marginal* generality distributions (not 50%); a size-matched
random-feature control on the width effect; matched-count and label-permutation nulls on the flow.
Cost: one CPU-only session, no NDIF, since corpus residuals are cached.

**(e)** `results/ndif_pinned.txt` rewritten from the live status endpoint: base 405B has no running
deployment (warm, unpinned); five pinned running models listed with provenance.

Files: `results/notes/instrument_audit.md`, `docs/INSTRUMENTS.md` §4b, fixes across eight scripts.
Cost: ~235k agent tokens, ~70 min wall.

## 2026-09-13 (hour 40) — Stage 14 was residual arithmetic: claim 6 is withdrawn, and recomposition survives only in generation (agent)

**The test.** Adversarial review of `docs/specs/core_v1.md` predicted a sixth failure mode: a readout
taken after a patch layer can move by residual arithmetic alone. Stage 14 patches an era shift at
layer 14 and reads the era at layer 20. Because the patch is a constant added at every position and
the readout is a mean over span tokens, `mean(resid₂₀ + shift) = mean(resid₂₀) + shift` **exactly**,
so the pass-through arm is not an approximation but an identity. A second, fairer arm rescales the
shift to preserve ‖shift‖/‖resid‖ at the read layer; without it the control is unfairly weak at depth.

| target | logged (model) | pass-through | norm-matched | gain over norm-matched |
|---|---|---|---|---|
| Qwen2.5-1.5B (h14) | 0.889 / 0.806 | 0.792 / 0.778 | 1.000 / 0.764 | **−0.111** / +0.042 |
| Gemma-2-9B-it (h14) | 0.875 / 0.944 | 0.972 / 0.958 | 1.000 / 0.917 | **−0.125** / +0.027 |
| GPT grid (h23) | 0.944 / 0.861 | 0.861 / 0.847 | 0.972 / 0.792 | **−0.028** / +0.069 |

**The model is indistinguishable from vector addition.** (Hour 45 corrects the original "and where it
differs it is worse": with the measured paraphrase interval, -0.1111 sits inside the band.) On Gemma
the six intervening blocks partly *undo* the addition. The layer curve never goes positive outside
tolerance and is never monotone in the number of intervening blocks (Qwen, reads 15–28: +0.000,
+0.000, +0.000, −0.111, −0.014, +0.056, −0.111). Across reads 15–18 the model arm and pure vector
addition agree case for case to three decimals. The apparent growth against the *plain* pass-through
(+0.10 → +0.69 → +0.81) is pure norm mismatch — the trap that would have rescued the claim in
weakened form.

**Sanity checks pass:** zero shift reproduces the unpatched readout exactly at every read layer, the
base arms reproduce the logged 0.97/0.78, 0.97/0.94 and 0.94/0.86, and the logged JSONs re-score to
the published table. One capture artifact: HF records `hidden_states[14]` before the forward-pre-hook
fires, so the model arm at the patch layer reads unpatched and the curve starts at 15.

**Withdrawn: hours 14 and 23's shift results, and claim 6 of the writeup.** ALGEBRA's L4
(address/form separation) loses its gauge-level evidence entirely, having already lost its
generation-level evidence at h27.

**What survives, and it inverts the story.** Hours 27, 29 and 33 re-read *generated text* with no
patch in force, so they are clean and confirmed. The 3× re-imposed result on Gemma (0.84 era-as-target,
0.30 lexical) stands. We believed recomposition worked as a readout and failed in generation; the
truth is the reverse. **The only real evidence for recomposition is the generation result**, which is
the expensive, qualitative, hard-won one — and the cheap representational result that looked like its
foundation was arithmetic.

**Also at risk, untested, arm is cheap:** every log-probability-readout patch instrument —
`stage4*`, `stage5*`, `stage6_factors`, `ndif_factors`, `ndif_absential_probe`,
`time_translation_selector`, and the h37 70B selector. The exposure is weaker there, because a
log-prob readout passes through the unembedding rather than being a linear readout of the same
stream, but it has not been checked. **Confirmed clean:** the role lens (h3, h16) never patches.

Files: `scripts/passthrough_test.py`, `results/passthrough_h14.json`, `results/notes/passthrough_h14.md`.
Cost: ~135k agent tokens, ~35 min.

## 2026-09-17 (hour 41) — PHASE 0.1: the selector effect is not the direct path. Claims 2 and 4 describe computation (agent, spec `docs/specs/selector_direct_path_v1.md` v2)

**The threat.** A direction patched at block L reaches the unembedding by the residual skip path
whether or not any block uses it, and Qwen2.5 ties embeddings, so a selector effect could in principle
be vocabulary geometry rather than computation. If so, writeup claims 2 and 4 are not about the stack.
The test: compare the treatment against the *observed* displacement projected onto the direction and
applied at the pre-norm residual (`F_par`). That null gives the direct path credit for whatever the
stack did **along** `d`; only orthogonal new content counts as computed. Conservative by construction.

**Gates (§5), all read before any number.** Reproduction exact under mid-rank: role L20 1.69, era 1.25,
voice 1.24, tense 1.03, composed 2.81, no-patch 2.00/2.00/1.50/9.50. No-patch max |gain| 0.0. The
offline identity `F_Δ ≡ A` holds to 4.6e-5 nats. Re-run bit-exact over 252 margins.

| claim | m_A | F_abs | F_par | **G_new** | 90% LB | 2τ | sign | verdict |
|---|---|---|---|---|---|---|---|---|
| role lens @ L20 (claim 2) | +1.857 | +0.037 | +0.020 | **+1.837** | +1.563 | 0.261 | 0.92 | **computed** |
| composed 3-factor @ L14 (claim 4) | +3.828 | +0.128 | +0.214 | **+3.614** | +3.251 | 0.411 | 1.00 | **computed** |
| era @ L14 | +2.245 | +0.185 | +0.308 | +1.937 | +1.387 | — | — | computed |
| voice @ L14 | +2.382 | −0.102 | −0.024 | +2.405 | +1.802 | — | — | computed |
| tense @ L14 | +0.834 | +0.131 | +0.216 | +0.618 | +0.342 | — | — | computed |
| role lens @ L14 | +2.140 | — | — | — | — | — | — | **no verdict (gate 5 failed)** |

**The stop condition does not fire.** Six of seven registered predictions *under*-estimated how
computed these are, and the planner's two most confident "substantially lexical" calls — tense at 0.80
and voice at 0.65 — were the most wrong.

**Mechanism.** The skip path delivers almost nothing: ‖d‖/‖pre_28‖ is 0.7–3.8%, while the orthogonal
response is 2.3–5.1× ‖d‖. No dose on a 0.25×–16× grid fits the per-case margins (RMS 0.59–4.04 against
τ 0.04–0.26), so the one-parameter direct family is rejected outright rather than merely beaten. The
lead-positions-only arm, whose direct path is removed by construction, still reads +0.316 (LB +0.232):
unambiguous computation.

**A standing claim is qualified, not withdrawn.** Claim 4 says the cross-talk matrix is diagonal. At
the final residual the *null* is also diagonal: on-diagonal fractions 0.40 / 0.45 / 0.49 for
F_par against 0.58 / 0.64 / 0.47 for the treatment, and 0.27 / 0.28 / 0.04 for random. **For tense the
diagonal is entirely vocabulary geometry** (0.47 treatment vs 0.49 null). The diagonal cross-talk
result is therefore partly a statement about embeddings, not only about factor independence.

**Two failures worth the record.** (1) **Gate 5 failed for role @ L14**: the positive control, a
direction built from the target span's own unembedding rows, was credited with +0.432 nats of "new
content" against τ = 0.212. Per §5 the instrument cannot separate the hypotheses there and **no verdict
is issued**; role @ L20 is the primary and is unaffected. (2) The registered `G_new(27)` sanity check
failed: +0.453 against 2τ = 0.249. Diagnosis: at L27 the treatment does nothing (m_A +0.014, rank 3.46 —
the lens has faded, cf. h5) while F_par is −0.44, so G_new inflates wherever the direct term *hurts*.
**No G_new below ~0.5 nats from this instrument is trustworthy until the per-layer offset is measured.**
Both primaries sit 4–14× above that floor. A guard is now in the script: no verdict when m_A < 2τ.
Also flagged: era's random arm reads +0.20 rather than null.

**Not tested here:** mood and theme; the h25 relation-as-patch; the remote 70B and Gemma batteries
(unblocked by this result, not withdrawn); the layer sweeps, which need the G_new offset measured
first. Claim 3 (relation selector 2.21/6) is out of scope and remains so: `stage3.py` never constructs
a model and never patches, verified in review.

Files: `scripts/selector_direct_path{,_report}.py`, 5 result JSONs, `results/notes/selector_direct_path.md`
(491 lines), `LM.pre_norm_residual` and `Patch(n_layers)` in `src/lsx/model.py`, 4 new invariants.
Cost: ~240k agent tokens, 3 h 22 min compute (two concurrent fp32 processes OOM at 15 GB; run sequentially).

## 2026-09-17 (hour 42) — PHASE 0.2: the abstraction ladder survives its two missing nulls; hour 26's "flow is an ordering" is downgraded to unconfirmed (agent + hand-finished)

The ladder was the only standing claim with no null anywhere. All three specified at hour 39 ran.
Full numbers and constructions in `results/notes/sae_ladder_nulls.md`.

**Merge test (null 1).** 13 of 15 merges more general, against an exact Poisson-binomial null drawn
from the same 16,384-feature population the cosine search ranges over: null mean 3.99 (narrative) and
5.78 (broad), P(K ≥ 13) = 0.0000 in both. The previous implied comparison was a coin flip, which was
the wrong null because the dictionaries have different marginal generality distributions. **Survives,
and more strongly than it appeared.**

**Width effect (null 2).** Narrow-dictionary mean generality 0.182 against 4,000 size-matched random
subsets of the wide dictionary at 0.063 ± 0.008 (z = 14.81); broad corpus 0.187 against 0.101 ± 0.012
(z = 7.07). A size-matched subset reproduces the *wide* mean exactly, so the effect is about what
narrowing preserves, not about dictionary size. **Survives.**

**Flow ordering (null 3).** Label-permutation nulls, class structure preserved: theme 8-NN purity
0.628 against 0.112 ± 0.017 (z ≈ 30); era 0.535 against 0.324 ± 0.025 (z ≈ 8.3); both p = 0.000 at
every threshold from 0.0 to 0.1. **Neither label falls into its own null band anywhere in the tested
range.** Hour 26 claimed "the flow is an ordering, era dies before theme". Theme is indeed far more
robustly structured than era, which is consistent, but nothing dies where we looked, so **the ordering
is downgraded from measured to unconfirmed.** Testing it needs thresholds above 0.1.

**Process note.** The agent handed back incomplete, having built all three nulls correctly but run out
of turn mid-computation. It had solved a real problem worth recording: git worktrees do not share
untracked files, so the cached Gemma Scope dictionaries were absent, and it symlinked them from the
main checkout rather than re-downloading onto a disk a previous session had already filled. The run
then produced every number and died at serialization (memory) before writing its JSON; the numbers
here are from the run log, and the script is recovered at `scripts/sae_ladder_nulls.py`.

**Not tested:** layers other than 20; the selection of the 15 features (the null matches the search
population, not the selection).

### Correction (hour 42b), superseding the flow-ordering paragraph above

The executing agent's own run reached further than the partial log I first published from, and I had
**removed its worktree while it was still computing** — my operational error, recorded here because
it nearly cost the better result. Recovered from the one file that survived on disk.

- **Twelve thresholds to g_k = 0.95**, not five to 0.1, at 1000 permutations each. **8-NN purity for
  both theme and era stays significantly above its permutation null at every threshold**, era still
  p = 0.008 at g_k = 0.95. So hour 26's "era's purity is gone by g_k = 0.7" was reading the observed
  value's proximity to the *theoretical* class-frequency floor (0.32), not testing it against a null
  that accounts for this particular 72-passage reconstruction's geometry. Real residual structure
  survives past that point for both labels.
- **A weaker version of the ordering does survive.** The within/across cosine-distance ratio has era
  crossing into its permutation-null band at g_k ≥ 0.7 on the broad corpus while theme never crosses
  in either corpus. That is a genuine ordering, but in one statistic of two, under one corpus of two,
  and marginal: p = 0.061 at the crossing, drifting to 0.106 by g_k = 0.95 rather than settling.
- **The run-to-run discrepancy I flagged was not resampling noise.** It was a directional bug the
  agent found and fixed: the ratio test was run in the purity direction (higher than null = more
  structured), when the ratio means the opposite (lower = more structured), which silently read every
  ratio comparison as "not significant". It caught this by noticing that ratio and purity gave
  contradictory verdicts at g_k = 0.0, which is impossible if both track the same structure.
- **The 12/15 versus 13/15 question is resolved:** 13/15 matches hour 26's count, not hour 15's, and
  the difference is the content filter, not the matching.

## 2026-09-18 — PHASE 0 GATE: blocked on item 0.3, which needs the user

Phase 0 of `docs/PROGRAM.md` has three items. Two are closed:

- **0.1 — the direct-path threat to the selector claims.** Resolved in the claims' favour (hour 41):
  role lens +1.837 nats over a skip-path null (90% LB +1.563), three-factor composition +3.614
  (LB +3.251), no dose of the direct-path family fits at any scale. Claims 2 and 4 describe
  computation. Two limits on the record: no verdict for the role lens at layer 14 (positive control
  failed), and no gain below ~0.5 nats from that instrument is trustworthy until a per-layer offset
  is measured.
- **0.2 — the abstraction ladder's missing nulls.** Run (hours 42, 42b). Merge test survives an exact
  Poisson-binomial null (p = 3.5e-9 / 9.8e-10); width effect survives size-matched subsets (z = 14.8 /
  7.1). Hour 26's "flow is an ordering" downgraded: nothing dies against a permutation null out to
  g_k = 0.95, and only a marginal single-statistic single-corpus version of the ordering survives.

**0.3 — move `TYPESAFE_API_KEY` to proxy credential injection — WAIVED by the user (2026-09-18).**
Rationale on record: low-risk context, the key is revocable at will. It therefore stays a plain
environment variable, readable by every spawned agent, and that is an accepted risk rather than an
open item. Anything that would change the risk (a shared environment, a key with broader scope, an
untrusted agent) reopens it.

**PHASE 0 GATE MET.** 0.1 and 0.2 resolved on the evidence, 0.3 waived by decision. Phase 1 opens.

## 2026-09-18 (hour 43) — PHASE 1 piece 1: the rediscovery harness catches 9 of 9 of our own bugs (agent, spec `docs/specs/core_v1.md`)

**Built in the spec's order, and the git history shows it:** the harness was committed first, with
its imports resolving to nothing, and the types it tests were committed second. A core that cannot
rediscover this project's own failures has no standing to certify anything new, so the failures come
first.

**All nine reconstructed bugs are caught, each by a named mechanism** (tiny random fixture, not the
1.5B; every case carries a positive control; tests assert the verdict *and* which mechanism fired):

| bug | mechanism |
|---|---|
| batch-row patch (h36) | `MovedCandidates` — moved 1 of 3, deltas [0.4988, 0, 0] |
| absolute spans under left padding (h39) | `PaddingConvention`; bypassed, `BatchEquivalence` on the shortest item |
| rank-1-on-ties, no no-patch arm (h34) | `MissingArm('no_patch')`, then `ArmOffNull` naming no_patch 1.00 *and* random 1.22 |
| best layer chosen on scoring data | `SelectionOnScoringData` |
| high-dim norm with no floor (h28) | `MissingFloor`; `Floor(estimator=None)` refuses itself |
| cross-talk rank degenerate by construction | calibration **sensitivity** failure — its noise test passes at exactly 2.000 |
| leaky grid reported raw (h35/h38) | `Grid.leak` at construction, then `RawScoreOnLeakyGrid` |
| post-norm hidden state as a residual (h40) | `PostNormResidual` |
| readout at or after patch layer (h40) | `MissingArm('passthrough')`, inherited by any such instrument |

**Three catches are thinner than they look**, and the agent said so rather than claiming nine clean:
the cross-talk catch rests on a two-test calibration (piece 2 owns the other three); the
selection catch is a regex over the caller's own prose; and the pass-through catch only verifies the
arm is *present*, never that it was computed.

**A finding about our leak checks.** The leak detector needed a null of its own. Leave-one-out on
balanced labels is **anti-predictive by construction**, so a grid carrying no lexical information at
all scores 0.00 rather than chance — the agent's first deliberately-clean control grid was flagged as
leaking. Recoverability is now read against a permutation null of the same leave-one-out. This means
every previous leak number in this repo was read against an implicit wrong baseline; the direction of
the error made us *over*-report leakage, so no claim was inflated by it, but the numbers are not
comparable to the new ones.

**An open exposure, not a bug yet.** `build_stack` must pass explicit `position_ids` or left-padded
batches fail equivalence for reasons unrelated to span indexing — and **no script in this repo passes
them.** Four remote scripts batch more than one text per job: `ndif_factors`, `ndif_recompose_gen`,
`ndif_recompose_sweep`, `ndif_time_translation_extract`, which between them touch hours 27, 29, 31
(corrected), 33 and 37. Cutting against this: the hour-39 audit ran an explicit batched-vs-single
equivalence check on NDIF and got cosine 0.99996, so nnsight's remote path may supply position ids
where local HF does not. **Unresolved. Piece 3 must measure a separate variance spread for batched
remote targets before grading any of them, and the equivalence check must run on the remote path.**

**Still unprotected, in the agent's words and worth repeating:** nothing ties a `Claim`'s provenance
to a real `Stack`, so a hand-rolled extraction can be wrapped in a valid Claim; `Probe` and remote
forwards do not go through the asserted path, so the moved-candidates assertion currently guards no
NDIF call; arm tolerance is an unmeasured 0.15 default; `Direction.held_out` is declared but never
verified; `EffectSize` is caller-supplied with nothing behind it.

**Process note.** Three harness cases initially fired the *wrong* mechanism, and all three were fixed
by correcting the core or the reconstruction rather than by weakening the assertion. Also: the venv's
editable install points at the main checkout's `src`, so `pytest` in a worktree imports a different
tree than the one being edited — a path shim is now in `tests/conftest.py`, and this affects every
future worktree agent.

Tests 52 passed in 0.54 s. Files: `src/lsx/core/{__init__,types,checks,extract,rediscovery}.py`,
`tests/test_core_rediscovery.py`, `results/notes/core_p1.md`. `scripts/` untouched, as the spec
requires. Cost: ~190k agent tokens, ~45 min, no downloads, no NDIF.

## 2026-09-18 (hour 44) — PHASE 1 piece 2: the registry and calibration battery, and the fix reproduced its own bug (agent)

Three instruments shipped per the rescoped §11, with `crosstalk`, `depth_gain` and `generality`
declared but unimplemented. Nulls come from config, never from the caller. 82 tests, harness still
9 of 9.

| instrument | null | arm tolerance, measured 3σ | invariances |
|---|---|---|---|
| `selector` | (k+1)/2 — 3.50 at k=6, 2.00 at k=3 | 0.810 (k=6, n=40); 0.447 (k=3, n=30); 0.122 (k=3, n=400) | scale, rotation |
| `composition` | (V+1)/2 — 9.50 at hour 8's V=18 | 2.461 (n=40) | scale, rotation |
| `readout_shift` | **0.0 gain** over pass-through, a difference never a ratio | 0.084 (d=1536, n=40) | scale, rotation |

Tolerances are 3·sd/√n with sd measured, and the closed forms were checked against Monte Carlo from
the real statistic (0.821 vs 0.816; 1.712 vs 1.708; 5.252 vs 5.188). **Piece 1's flat 0.15 default
would fire on 55.3% of clean hour-4-shaped arms; the measured band fires on 0.1%.** That is the
difference between a contract and a nuisance.

**The fix reproduced the bug it was fixing.** The first calibration failure was not sensitivity but
**self-floor, on both rank instruments, because `run_calibration`'s own `null_tol` was a flat 0.1** —
piece 1's unmeasured-tolerance bug reappearing *inside* the code written to prevent it. It now uses
the same measured 3σ band. Worth recording as its own lesson: a fix that hard-codes a threshold
inherits the failure mode it was written to remove.

**A statistic was rejected by its own noise test.** `readout_shift` failed noise at −0.129: the
cosine form is biased negative whenever the blocks do work orthogonal to the readout direction,
because a cosine is a *fraction*. The instrument was changed to a projection readout. The cosine
version is kept as `cosine_readout_gain` with a test asserting the battery refuses it.

**Blast radius of that, checked directly, and it is small.** The scalar-cosine bias applies to
patched readouts. Hours 29 and 33 read *generated text with no patch in force*
(`era_read = argmax_e cos(u, d_e)` is a classification over unpatched output), so they are not
exposed. The patched cosine readouts are `stage7_shift.py` and `ndif_shift.py`, which produced hours
14 and 23 — **already withdrawn at hour 40 for the pass-through reason.** No standing claim moves.
Piece 3 must still re-derive any scalar-cosine target as a projection or margin before comparing it
to a logged number.

**The core now reproduces one of our own retractions.** Harness bug 8 computes a pass-through gain of
**−0.1112** where hour 40 logged −0.111, independently, from the reconstruction rather than from the
saved numbers.

**Closed from piece 1's unprotected list:** arm tolerance (measured); the placeholder arm table
(now the registry); the three missing calibration tests; and the pass-through catch, which now
*computes* the arm and asserts exact reproduction of the unpatched readout at zero shift inside the
constructor. **Partly closed:** `Direction.held_out` is verified against a fitter witness, though
hand-built directions remain unverified; `EffectSize` is recomputed and refused on verdict
disagreement, which kills a fabricated z of 4.1 on hour 14's scores. **Left open, with reasons:** the
selection check is still a regex over the caller's prose. The agent added `Instrument.sweep`, which
records a core-computed curve so that "this was not a choice" is true by fact rather than by
assertion, and noted that making it mandatory would refuse every §1A target until piece 3 re-runs the
sweeps — so piece 3 owns that call.

**Piece 3 must differ in three ways**, per the agent: refuse to grade on a hand-declared calibration
report, which is the last hole; measure **two** tolerances, the remote re-run spread *and*
`readout_shift`'s paraphrase-noise interval, since §2a's gain is reported with the latter; and
reproduce hour 8 through `composition` against its null of 9.50 rather than as three selector calls.

Files: `src/lsx/core/{planted,registry,instruments}.py`, `tests/test_core_registry.py` (30 tests),
`results/calibration/*.json`, `results/notes/core_p2.md`. Cost: ~225k agent tokens, ~55 min, no
downloads, no NDIF. `scripts/` untouched.

## 2026-09-18 (hour 45) — PHASE 1 piece 3: the core reproduces every number it can compute, refuses three, and fails its own gate on coverage (agent)

100 tests, harness still 9 of 9. The ledger, retraction, and the §1A reproduction suite.

| §1A target | logged | reproduced | tolerance | verdict |
|---|---|---|---|---|
| h8 three-factor composition | 2.81 of 18, no-patch 9.50 | **2.8056**, no-patch **9.5000** | ±0.02 | reproduced |
| h4 role lens | 1.69 of 6, random 3.25 | **1.6875**, random 3.406, permutation 3.542, no-patch 3.500 | ±0.02 | reproduced |
| h14 gain vs norm-matched pass-through | −0.111 | **−0.1111**, zero-shift error exactly 0 | ±0.1128 measured | reproduced |
| h8 era / voice / tense lens | 1.25 / 1.24 / 1.03 | 1.250 / 1.236 / 1.028 | ±0.02 | **REFUSED** — `MissingArm('permutation')` |
| h16 "peak layer 16" | 1.73 of 6 | not published | — | **REFUSED** — `SelectionOnScoringData` |
| h39 Gemma clock | 0.501 / 0.767 / 2.50 raw | not published | — | **REFUSED** — leaky grid, and `discrimination` unbuilt |
| h29 3× re-imposed | 0.84 / 0.91 / 0.53, n = 55/72 | **0.8364 / 0.9091 / 0.5273, n = 55/72**, twice | ±0.018 measured | deferred, no instrument |
| h37 70B selector | 1.50 / 1.06 | not re-derived | ±0.018 | deferred, stacks not cached |

**Nothing failed its tolerance, so nothing is withdrawn.** Every number the core could compute came
back to the logged decimals. The three refusals are results about how the numbers were *reported*:
the factor lenses never carried a permutation arm, "peak layer 16" was an argmax on the scoring data,
and the Gemma clock is a raw score on a grid with a measured leak.

**The two tolerances, measured rather than assumed.**
- **Remote re-run spread: 0.000.** Hour 29's 3× arm re-run end to end on NDIF gave the same 55 of 72
  survivors, the same 17 losses (a property of the generation, not the queue), all 55 continuations
  character-identical, zero readouts flipped. Tolerance is therefore set at the statistic's own
  resolution, 1/55 = **0.018**. This bounds within-session, pinned-deployment noise only.
- **`readout_shift` paraphrase noise: 0.3191 per item, ±0.1128 at n = 72.** Piece 2's `sqrt(2/d)`
  stand-in was **8.8× too tight**.

**That correction changes how hour 40 must be stated, and I have applied it above and in the
writeup.** Under the too-tight stand-in, h14's −0.1111 read as significantly negative, which is where
"the model is worse than the arithmetic, the blocks partly undoing the addition" came from. With the
measured interval it sits *inside* the band. The correct statement is that **the model is
indistinguishable from vector addition in either direction** — the blocks do nothing the readout
sees. The verdict is unchanged: indistinguishable from arithmetic is still vacuous, and claim 6 stays
withdrawn. But "worse than arithmetic" was over-claimed and is now corrected.

**The position-ids exposure is settled and is not a live bug. No logged hour falls.** Local Qwen gives
batched-vs-single cosine 1.000000 with *and* without explicit position ids, and the check is
demonstrably sensitive (scrambled 0.691, all-zeros 0.713, uniform shift 1.000000 — RoPE depends on
position *differences*, and left padding shifts a row uniformly). Remote Gemma via `tracer.invoke`:
worst of twelve item-by-pooling pairs on the shortest item with 41 tokens of padding, **0.999943**.
`ndif_factors` pads right and cannot be exposed at all.

**A piece-2 bug found by running real targets.** `CALIBRATION_KEYS` held one key per instrument, but
the key hashes the declared null, which is config-dependent — so *every* three-candidate selector
claim (h8's lenses, h34, h37) was refused as `CalibrationStale` while holding a perfectly good
report. Now a set per instrument, with a regression test. This is the third time a fix has carried a
version of the bug it was fixing.

**Decision taken, as delegated:** `Selection.executed` is **mandatory at the ledger**. It costs
exactly one §1A row, h16, which was already refusable on other grounds.

**Closed:** hand-declared calibration reports (refused at the ledger); provenance not from a real
`Stack` (`build_stack` now signs its provenance with an activation digest); piece 2's stale-report
import hole; the paraphrase tolerance; the position-ids exposure.

**PHASE 1 GATE: §1B passes, §1A does not.** Not because anything failed to reproduce, but on
**coverage**: three of the six instruments in §8 are unbuilt and two targets need them. Per
`docs/PROGRAM.md`, the gate is both halves, so **phase 2 does not open.** Remaining, in the agent's
order: build `discrimination` and a top-1-accuracy instrument (h29 has no instrument at all); route
remote forwards through the asserted path, since `Probe` is still a shell and the h34
moved-candidates assertion currently guards no NDIF call; a per-item `role_rank` path for h16's
curve; and harness cases for the three publication refusals.

Ledger: 3 standing rows, 0 withdrawals — the agent declined to fabricate one to demonstrate
retraction, and exercises `withdraw()` in tests instead. `docs/specs/core_v1.md` edited in three
places (§1A tolerances, §2a position-ids verdict, §4 sweep decision), each marked in the text as
written in after the fact. Cost: ~370k agent tokens, ~2 h 45 min, 168 NDIF jobs.

## 2026-09-18 (hour 46) — PHASE 1 piece 4: coverage closed, and the core starts refusing our own reporting (agent)

143 tests, harness 12 of 12. Both new instruments pass the full six-test battery.

| instrument | null | measured per-item sd | 3σ band | note |
|---|---|---|---|---|
| `top1_accuracy` | **1/k** (0.3333 at k=3) | 0.4714 Bernoulli (MC 0.4693) | ±0.1907 at n=55 | *not* a rank midpoint: a rank's null grows with k, an accuracy's shrinks |
| `discrimination` | the **measured floor** | 0.3536 (MC 0.3381) | ±0.3750 at 8 subjects | claims scale, rotation and monotone-target invariance; explicitly *not* cell-rescale (Δ 0.0493) |

`discrimination`'s battery runs at chance rather than at the caller's floor, so **the battery does not
verify that floor** — now stated in the spec. `generality` stays unbuilt: `discrimination` did not
make it cheap, and it still has no null.

**The remote hole is closed, and the assertion caught the real bug on a real call.** `Probe` refuses
unasserted forwards; `src/lsx/core/remote.py` ran live on Gemma-2-9B-it. Batched-vs-single on the
shortest item (13 tokens of padding) 0.9999925; a whole-tensor patch moves 3 of 3; **the hour-36
idiom moved 1 of 3 and `MovedCandidates` fired — the first time that assertion has caught that bug on
an actual NDIF call**, rather than on a reconstruction. No-patch moves 0 of 3.

**Four bugs the agent found in its own code before publishing, and two of them are the same lesson
again.** (1) The known-zero point read 0 for the wrong reason: a dead readout scored ±0.548 per
subject off 4e-16 of floating-point rounding, with the signs cancelling in the mean; closed with a
tolerance derived from the dot-product error bound. (2) `calibration_key` hashed only top-level
source, so *that very fix* changed the arithmetic and left every cached report valid. (3) The first
closure walker missed calls inside comprehensions, so the fix for a hole that hid a dependency was
itself hiding a dependency. (4) A planted-signal generator tied candidates without making them win,
saturating both sensitivity curves. **That is the fourth and fifth time in this build that a fix has
carried a version of the bug it was fixing.**

**§1A re-run: 4 reproduced, 5 refused, 2 deferred, 0 failed. Nothing withdrawn.** Hour 16 is now
reproduced *and published* — the core computes its layer curve per item, which is what
`Selection.executed` requires. Hour 29 moved the other way, from deferred to **refused**: its
instrument now exists, and the refusal is about the data.

**THREE FINDINGS ABOUT OUR OWN LOGGED NUMBERS.** I verified the first directly rather than taking it
on report.

1. **Hour 29's published battery has one arm.** Checked: `recompose_gen_gemma9b.json` and
   `recompose_gen_llama70b.json` carry `base`, `rand` and `shift`, but **every sweep file at every
   other scale — 0.5, 2.0, 3.0 and both prefix runs — contains only `shift`.** So the headline 0.84
   at 3× re-imposed has **no random control and no no-patch baseline at its own scale**; the controls
   exist only at 1.0. This violates `CLAUDE.md`'s first non-negotiable. The claim is not refuted —
   the lexical check moving 0.00 → 0.30 across scales is real evidence — but the number is
   under-controlled and the writeup now says so.
2. **Hour 16's "2.21" and the repo's own JSON (2.1692) are two different aggregates of one curve**,
   a step-4 and a step-2 average, used interchangeably in the record. The core matches every layer to
   0.0001; the discrepancy is in how we summarised, not in what was measured.
3. **Hour 8 reports gain over *chance*, not over a measured floor**, which §6 requires on a grid with
   a measured leak.

**PHASE 1 GATE STILL DOES NOT CLOSE — but no longer on coverage.** Every instrument the targets need
now exists. What blocks it is that five §1A targets are refused on *reporting* grounds: missing arms,
raw scores on leaky grids, and aggregates chosen after the fact. That is the core working as
designed, and closing the gate now means going back and re-running those batteries with their
controls, not adjusting the core.

Cost: ~515k agent tokens, ~2 h 10 min, 7 NDIF jobs. Files: `src/lsx/core/remote.py`,
`tests/test_core_{instruments_p4,remote}.py`, `results/notes/core_p4.md`, calibration reports.

## 2026-09-18 (hour 47) — PHASE 1 piece 5: the five refused batteries re-run; the gate closes, and three of our numbers mean less than the record said (agent)

The five §1A targets piece 4 refused on *reporting* grounds were re-run with the arms they were
missing. **Nothing failed a tolerance. Every number came back inside it. What moved is the floor
under three of them.** Note: `results/notes/core_p5.md`. 152 tests pass; the rediscovery harness is
12/12.

**Phase 1's gate CLOSES, with one target outstanding and named.** §1B: 12 of 12. §1A: seven rows
reproduced and published, one refused exactly as the spec predicts (h16's "peak layer 16",
`SelectionOnScoringData` — that refusal *is* the acceptance test passing), one still deferred —
**h37's 70B matched pair, whose direction stacks are not cached.** Not blocked by code, only by data
and budget. The gate closes with that on the record rather than rounded off.

**The three numbers whose floor moved.** Sign convention: gain = treatment − floor, and for a rank
statistic lower is better, so a *negative* gain is better than floor.

| target | treatment | measured lexical/stimulus floor | gain | arm band |
|---|---|---|---|---|
| h8 composed | 2.8056/18 | **2.8472** | **−0.0417** | ±1.8343 |
| h8 era lens | 1.2500/3 | 2.0000 (exactly chance) | **−0.7500** | ±0.2887 |
| h8 voice lens | 1.2361/3 | **1.0278** | **+0.2083** (worse than floor) | ±0.2887 |
| h8 tense lens | 1.0278/**2** | 1.0278 | **+0.0000** (exactly on floor) | ±0.1768 |
| h39 Gemma clock | 0.9895 | 0.9221 (the grid's own t0 control prompts) | **+0.0675** | ±0.4009 |

The floor is a bag-of-tokens predictor fit by the same leave-one-scene-out arithmetic as
`level_directions`, ranked through the identical `midrank`: activations swapped for word counts and
nothing else. Its own permutation control goes to chance (10.24 against 9.50), so it is reading the
words, not the procedure. **Only era clears its lexical floor.** The composed test's −0.04 of a rank
is inside its own 3σ band of ±1.83: on this grid, "three narrative factors compose" is not separable
from "the words differ". That is consistent with h5 — tense and voice are lexical — and it is the
first time the composed test has been measured against anything but chance.

**This does not retract the composed measurement.** 2.8056/18 against a no-patch arm sitting on 9.50
exactly is a real effect of the patch. What is retracted is the *comparison*: −6.6944 was a gain over
chance, and §6 requires a gain over the measured floor on a grid with a measured leak.

**h39's clock is the interval phrase, to this design's resolution.** +0.067 against a ±0.40 band, and
the shuffled-stimulus arm at 0.9693 says word order contributes nothing either — h38's
order-invariance arriving on a second model. The row publishes and what it reports is a null. It is
**not** a reproduction of the logged 0.767, and says so: 0.767 is a Spearman over nine per-Δt shared
norms with no per-subject breakdown, and `discrimination` is a per-item instrument.

**h29 is the one that got stronger, and it is writeup claim 8.** All three arms at scale 3.0,
re-imposed, on Gemma-2-9B-it: treatment **0.8364 / 0.9091 / 0.5091** (n=55/72), lexical 0.30 on 10 of
55 — reproducing the logged numbers to the digit — with **random 0.1429** (n=56/72) and **no-patch
0.1111** (n=36/36) against a declared null of 0.1111. The under-controlled caveat added at hour 46 is
lifted: the 0.84 now has its controls at its own scale and they sit where they should.

Two things carried that row. The declared null is **0.1111, not 1/3**: an unpatched continuation is
not uninformative about its own era, it keeps e1, and declaring 1/3 would have manufactured an
`ArmOffNull` — piece 3's config-dependent-key bug in a new costume. The number used is h27's own
published base arm (`era_other` 0.2222 split over two non-e1 targets), an independent prior
measurement of the identical condition, and the no-patch arm came back at 0.1111 to the digit: a
prediction tested, not restated. And the patch vectors, which came from a cached `.npz` that cannot
reach the ledger, were re-derived through `remote.build_remote_stack` with every §7 assertion and
matched the cached ones at cosine ≥ 0.999962 (L14) / ≥ 0.999956 (L20). The moved-candidates clause
ran explicitly for the first time on a *generation* battery: shift 4/4, random 4/4, the h36
`output[0]` idiom 1/4 with `MovedCandidates` firing; and per item, 55/55 shift and 56/56 random
continuations differ from the same passage's greedy unpatched one.

**Two defects in our own record, found by building the arms.**
- **h8's tense lens is 1.03 of 2, not of 3.** `tense` has two levels; its null is 1.50. The h8 table
  at line 504 had it right; Checkpoint 2's claim-4 summary, §1A of the core spec and piece 4's driver
  all printed /3, and piece 4 built it as a 3-candidate instrument. The value
  was always right; the denominator and the null were not. Under a 3-candidate declaration its
  no-patch arm of exactly 1.50 would have read 0.50 off its null — **the refusal for the missing
  permutation arm fired first and hid it for two pieces.**
- **h16's "2.21" is the step-4 subsample** (recomputes to 2.2065) of a curve whose canonical step-2
  mean is **2.1692**. Same curve, two aggregates, used interchangeably across `RESULTS.md`,
  `WRITEUP.md` and `docs/INSTRUMENTS.md`. Both are now in the ledger row's `logged` field so it can
  never happen silently again.

**The ledger carries its first four withdrawals.** Two h8-composed rows that subtracted chance under
the name `gain_over_floor`; two lens rows whose permutation arm was a single under-powered draw.
Piece 3 declined to write a demonstration retraction because it would have been fabricated; these are
not.

**An id collision, and a general hazard.** h8's era and voice lenses agree on instrument, provenance,
grid, selection, config and calibration key — the provenance never recorded *which factor was
patched* — so they hash to the same `Claim.id` and the ledger refused the second with
`LedgerConflict`. The right refusal for the wrong reason. Fixed by putting `factor` in the provenance
(unsigned, so the stack signature is untouched). **Two claims differing only in which direction was
patched are indistinguishable to `Claim.id` unless the caller says so, and nothing makes the caller.**

**The methodological finding, and it arrived as a refusal.** Built as a single permutation draw, the
voice arm read 2.347 against a null of 2.00 and a registry band of ±0.2887 — refused with
`ArmOffNull`, crashing the run. An arm off its null is a bug until proven otherwise, so it was
measured, not argued: six further independent draws give a per-draw sd of **0.349 for voice** —
**one draw's one-sigma spread is larger than the whole three-sigma band the registry computes.** The
band is 3σ on the per-*item* null spread over 72 items, but a permutation arm's 72 items are four
draws (one per scene) × eighteen re-rankings. `n` counts repetitions, not evidence. The arm was never
off its null; it was under-powered. **The fix is more evidence, not a wider band:** pooled over all
seven draws with a cluster-robust `3·sd/√7` — era 2.153 (±0.179), voice 1.919 (±0.420), tense 1.411
(±0.158). All three publish. Two consequences on the record rather than smoothed: a single-draw
permutation arm on this design has almost no power (a ±0.42 band on a rank bounded in [1,3] would
admit an arm reading *below* the treatment), so pooling reduces h8's defect without removing it; and
**`registry.arm_tolerance` is wrong for any arm whose randomness is a draw rather than an item, and
wrong in the dangerous direction — it refuses clean arms.** Second target bitten, both caught by
hand. An `Arm` should declare its independent unit. This piece did not fix it.

**Three bugs in the piece's own code, each caught by something the spec already demanded.** (1) The
generation trace bound `rlm.model.generator.output.save()` *inside* the trace block; NDIF ships the
block's source and refuses attribute paths through a class defined in `lsx.core.remote`. Twelve
generations died that way — and `asserted_remote_patched_logprob`, two functions above, carries a
five-line comment saying exactly this, written after piece 4 hit it on the wire. **The sixth time in
this build that a fix has carried a version of the bug it was fixing, and the first time the fix was
already in the file.** A test now greps for it. (2) The h29 claim set `provenance["template"]`, a
*signed* field — the row would have been refused by the very check it was written to satisfy; the
generation's fields now go in under their own names. (3) `hash()` seeded the shuffled-stimulus arm,
randomised per process, and the extraction is resumable across processes — a single arm could have
been assembled from two different shuffles. `hashlib.sha256` now, with a test. Plus one in the
driver: a refusal on one lens aborted the whole battery, so the report would have named only its
successes by construction.

**Cost.** ≈ 3 h 10 min. Local: 3 096 + 3 456 patched log-prob forwards and a 240-prompt extraction
with 15 layers × 5 arms. Remote: h29 — 180 generations attempted, **147 scored, 33 lost (18%)**,
≈ 250 jobs; h39 — 720 texts at layer 20 in three arms, ≈ 290 jobs. **The loss is unequal across arms
(0% base, 24% shift, 22% random), deterministic per item, and unexplained; the surviving sample is
therefore not a random subsample and the arms' n differ.** That is an open caveat on the h29 row. No
405B, nothing downloaded, no credentials printed.

**Still unbuilt/unfixed:** h37 not re-derived; `registry.arm_tolerance`'s independent unit;
`crosstalk`, `depth_gain` and `generality` instruments (`generality` still has no null).

## 2026-09-18 — PHASE 2 GATE: cannot be met as written, and one claim cannot be met at all

Inventory in `docs/specs/phase2_v1.md`, written by hand (Fable's monthly budget is exhausted; phase 2
contains no new measurement, so the planner rule does not bind). Grounded in three things checked
rather than recalled: the ledger's actual rows, the registry's actual instruments, and what is
actually cached on disk.

**The eleven standing writeup claims, by verdict: 4 done, 3 derivable, 4 needing an instrument that
does not exist, 3 needing remote re-extraction, 1 blocked, 0 must-withdraw.** Nothing in the writeup
is unsupportable on its face. What is missing is instrumentation and provenance, not evidence.

The ledger covers **h4, h8, h14, h16, h29 and h39 only**. Five of seven declared instruments are
built; `crosstalk`, `depth_gain` and `generality` are not. Local `.npz` stacks are cached for every
replication family; **no** Llama-70B stack is, and neither are h27's or h33's generation arms.

**Claim 6 (abstraction as a quotient) is blocked, not merely expensive.** `generality` is refused for
having *no null*, and designing that null is a new measurement — it means deciding what population a
"general" feature is general against, and h42 already showed the companion ordering claim collapses
once its permutation null is drawn. Phase 2's charter is "no new science", so claim 6 cannot become a
ledger row inside phase 2 **at any budget**. Its honest outcomes are NOT-DERIVED with that reason, or
withdrawal. It is not to be restated until it clears a floor it was never measured against.

**Per `docs/PROGRAM.md`'s one rule, this is recorded rather than routed around.** The gate says
*ledger row or withdrawn*; seven claims are neither. The three options — raise the budget, amend the
gate to allow an explicit **NOT-DERIVED** marking with a named blocker, or cut the writeup to what
the core can carry — are in §4 of the spec. **The recommendation is to amend the gate, and to build
`crosstalk` alone**, because the diagonal is what "the factors are separate directions" *means* and
h47 showed the null's cross-talk is partly diagonal too — the load-bearing claim is also the one most
at risk of being geometry. **That decision is the user's and the program waits on it.**

**What does not wait, because it is valid under all three options:** the derivable batch (spec §5) —
the two stage-41 skip-path rows via `readout_shift` + `PassthroughArm`, and h8's battery re-fit on
the four cached replication stacks and the GPT-authored grid, each with its **measured lexical
floor** rather than chance. Expected: 3 claims to DONE, ~7 new rows. Running now.

**Two hazards named while they are still cheap.** (1) `registry.arm_tolerance` computes a 3σ i.i.d.
band from the item count and is wrong for any arm whose randomness is a *draw* — it refuses clean
arms, has bitten two targets, and the derivable batch adds ~7 rows with exactly that structure. It
is being fixed as part of this batch: an `Arm` declares its independent unit. (2) **The cached
`.npz` stacks are gitignored and will not survive this container.** Every derivable row depends on
them; if they are lost the batch becomes hours of local re-extraction. That is why the batch runs
before the gate decision rather than after it.

## 2026-09-18 (hour 48) — PHASE 2 derivable batch: the arm-band fix, 27 measurements, and NOT ONE of them reached the ledger (agent, spec `docs/specs/phase2_v1.md` §5)

Note: `results/notes/phase2_derivable.md`. 175 tests pass (152 + 23). Rediscovery harness now 13
cases (10 pure + 3 model), all caught.

**The headline is the failure, and it invalidates my own spec.** All 27 claims this batch built were
refused by the ledger with **`ProvenanceNotFromStack`**: every one is provenanced to a frozen script's
cached `.npz` or `selector_direct_path_*.json`, not to `build_stack`. That is the same refusal h29 hit
at h47, and h29 only cleared it by re-extracting the whole grid. Re-extraction is **not available
here**: four of the five replication models are not in the local HF cache, and h41's residuals were
never cached at all. **So `docs/specs/phase2_v1.md`'s three DERIVABLE rows are not derivable — they
are NEEDS-DATA, and offline they are not even that.** The inventory's count is now **4 done, 0
derivable, 4 needing an instrument, 6 needing data, 1 blocked.** I wrote that spec; the error is mine,
and it is the same error the whole project keeps making — *the number was available and I read that as
the claim being available.* Provenance is the thing the core exists to require.

The measurements are still real and are recorded as measurements, outside the ledger.

**(1) `registry.arm_tolerance` — fixed, with a guard that cost more design than the fix.** An `Arm`
now declares `n_independent` and the registry bands on units rather than items. The hazard is obvious:
the same declaration is the easiest possible way to widen a band until a row publishes. So a reduction
must be **read off the design and shown** — the arm names its unit and hands in one cluster label per
item, and `Arm` refuses (`ArmUnitNotInDesign`) when the declared clustering is not visible in the
arm's own scores against its own permutation null (999 shuffles, refused at p > 0.05). A constant arm
returns p = 1.0 and can never buy a wider band. **Rediscovery case 12** plants exactly the abuse: an
arm 0.45 off its null, refused at the item count; the same numbers relabelled into six "draws" that
are not in the data, refused as a design claim; and a positive control with a real between-unit
offset that publishes.

Piece 4's lesson checked explicitly and it very nearly landed a seventh time: the calibration key
hashes the statistic, null and invariances and does **not** cover `arm_tolerance` — but
`Instrument.resolved_null_tol`, the battery's own pass threshold, *is* `arm_tolerance(n)`. A change to
the default path would have left all 19 cached reports "valid" under a different threshold. The
default path is bit-identical and a test now pins the threshold to seven decimals per instrument.

**(2) The two stage-41 rows** (writeup claims 2b, 4c), `readout_shift` + `PassthroughArm`:

| row | treatment | pass-through | gain | 90% LB (cases) | 90% LB (units) | sign | random | no-patch |
|---|---|---|---|---|---|---|---|---|
| role lens, L20, 48 cases / 8 domains | +1.8566 | +0.0199 | **+1.8367** | +1.5630 | +1.5650 | 0.92 | +0.2670 | 0.0000 |
| 3-factor composition, L14, 72 cases / 4 scenes | +3.8279 | +0.2143 | **+3.6135** | +3.2513 | +3.0096 | 1.00 | +0.0448 | 0.0000 |

Reproducing the logged +1.837 / +3.614 and LB +1.563 / +3.251. **A caveat the record did not carry:
the published "90% lower bound" is a bootstrap over *cases*; resampling the design's actual units
costs the composed row 0.24 nats** (3.251 → 3.010). The conclusion is unchanged and the number is not.

**(3) h8's battery on the cached replication grids** (claim 4b) — 25 rows, gain over each grid's
**own** measured lexical floor. Ranks, so negative gain = better than floor; `clears` means the gain
is also outside the arm band.

| grid / model | composed | era | voice | tense / theme |
|---|---|---|---|---|
| factors_v2, Qwen-1.5B | 1.2361 vs floor 2.8472 (**−1.61**) | clears | 1.0000 vs 1.0278 (−0.03) **no** | tense 1.0000 vs 1.0278 **no** |
| factors_v1, Qwen-1.5B | 1.1944 vs 2.2222 (−1.03) | clears | −0.03 **no** | — |
| factors_v1, Qwen-0.5B | 1.4167 vs 2.2222 (−0.81) | clears | −0.03 **no** | — |
| factors_v1, Pythia-1.4B | 1.2778 vs 2.2222 (−0.94) | clears | −0.03 **no** | — |
| factors_v1, Gemma-2-9B-it | 1.0278 vs 2.2222 (−1.19) | clears | −0.03 **no** | — |
| theme_v1, GPT-J-6B | 1.1111 vs 2.5556 (−1.44) | clears | — | theme clears |
| factors_gpt_v1, Qwen-1.5B | 1.4167 vs 3.0833 (−1.67) | clears | clears (−0.31) | — |
| theme_gpt_v1, Qwen-1.5B | 1.0556 vs 1.6667 (−0.61) | clears | — | theme 1.0278 vs 1.0833 **no** |

**Era and the composed rows clear their lexical floors on all four model families and on both
GPT-authored grids. Seven of 25 rows do not: `voice` on every single factor grid, the reference
grid's `tense`, and `theme` on the GPT theme grid.** "Does not clear" here means *indistinguishable
from the floor*, not worse than it — voice's gain is −0.0278 with a treatment of exactly 1.0000
against a floor of 1.0278, which is a readout at its ceiling, not a readout failing. **h47's finding
replicates across five grids and four model families: era is the factor that survives its words;
voice and tense do not separate from theirs.**

**A second, stricter floor that nobody asked for: 12 of 25 rows do not clear the same readout taken
at the shallowest cached layer**, including GPT-J's era and theme, which read exactly 1.0000 at both
depths. This also falsifies a line in the core spec — §6's "layer 0 *is* the bag-of-tokens check" is
not true for this readout; the shallow-layer floor is the harder one.

**Arms off their nulls, and the one that was not fixed.** Gemma's voice random arm read 1.583 against
2.000 and the row was refused; fixed with **eight pooled draws, not a wider band** (→ 1.986), applied
to every row. The GPT grid's voice permutation arm read 1.823 at eight draws; at **32** draws it reads
1.930 inside ±0.124, and 32 is now the setting everywhere. But the residue is real and is left in
place: **the permutation arm sits below its null on 25 of 25 rows** (mean −0.117, every one inside its
band, sign test p ≈ 3e-8). The mechanism is that the uncentered cosine ranking is partly a *norm*
ranking. It was **not tuned away** — centering would silently make the floor and the treatment
different procedures — and at ~60–100 draws it would start refusing rows. Recorded as a defect of this
readout's null.

**What the agent got wrong, in its own words.** It had **the sign of `gain` backwards** in its first
table, which printed a clean and entirely plausible result in which era *failed* its floor by a full
rank; it was caught only because era's floor happens to be exactly chance. It also shipped a
single-draw random arm and an eight-draw permutation arm, and learned both were under-powered only
because two rows were refused — *the other 23 were never evidence of power.*

**What I found reading the diff rather than the report.** `checks.cluster_evidence` computes the
design effect `deff = 1 + (mbar − 1)·icc` and **nothing uses it**: the band is taken on `k`, the
cluster count. So the guard decides *whether* an arm may widen its band but never *how much*. Where
clustering is partial the honest effective n is `n/deff`, which can be far larger than `k`, and
banding on `k` is **over-wide — the permissive direction, the one that hides an off-null arm.** The
i.i.d. band was wrong by refusing clean arms; this replaces it with a band that can be wrong by
admitting dirty ones, and the quantity that would fix it is already being computed. Also: **no row in
this batch actually declared a reduced unit**, so the whole mechanism is exercised only by its tests
and its planted case. Open.

**Cost: ≈353k agent tokens, 41 min, local CPU only. That is more than twice `PROGRAM.md`'s 150k
phase-2 budget for one agent**, and the overrun bought the guard in (1) rather than more rows.
Not tested: anything remote, anything needing an unbuilt instrument, and — the point of the hour —
anything the ledger would accept.

## 2026-09-18 (hour 49) — The band fix's own permissive direction, closed the hour after it opened (agent, open problem 4j)

Note: `results/notes/phase2_unit_band.md`. **179 tests pass** (175 + 4). Rediscovery harness: **14
cases, 11 caught pure, 3 needing a model fixture.**

h48 replaced a band that was wrong by *refusing clean arms* with one that could be wrong by
*admitting dirty ones*: `cluster_evidence` computed a design effect `deff` and nothing used it, so an
arm that passed the clustering p-gate was banded flatly on `k`, the cluster count, no matter how
weak the clustering was. The gate decided **whether** an arm could widen its band and never **how
much**. `n_independent` is now `clamp(round(n / deff), k, n)` — measured, not asserted.

| case | n | k | measured ICC | old band (flat on k) | new band (on n/deff) | outcome |
|---|---|---|---|---|---|---|
| h8 permutation arm | 72 | 4 | ≈ 1 | 4 units, ±1.2247 | **4 units, ±1.2247** | **unchanged, bit-identical** |
| planted partial clustering | 72 | 6 | +0.194 (p = 0.007) | 6 units, ±1.0000 — **admits** an arm 0.70 off its null | 23 units, ±0.5108 | **refused (`ArmOffNull`)** |

The h8 case is unchanged by construction, not by luck: `mbar = n/k` always, so at ICC = 1,
`n/deff = n/mbar = k` exactly for any cluster sizing. That is pinned in a test. **Rediscovery case
13** plants the bug this fix closes — an arm off its null whose *partial* clustering passes the
p-gate and which the k-flat band would have waved through — with a positive control at the same
clustering that still publishes.

**The degenerate branch, decided rather than defaulted.** The permutation p-gate and the
method-of-moments ICC are different statistics and can disagree on a noisy draw: real clustering,
ICC ≤ 0, `deff = 1`, `n_eff = n`. That yields **no widening at all** — back to the tight per-item
band. Deliberate: between refusing a clean arm (safe, and fixed by more draws, which is what h47
did) and silently admitting a dirty one inside an unearned band, the fix fails toward the first.

**The bug the agent hit is the same shape as the six before it.** `Instrument.claim` attaches the
measured tolerance with `dataclasses.replace(arm, tolerance=...)`, which rebuilds the `Arm` carrying
the *already-resolved* `n_independent` and re-runs the resolver — which then refused, with "the unit
cannot be two numbers", an arm that had never lied. Case 12's honest positive control was what
surfaced it. **The guard's first act was to refuse a clean arm: exactly the failure it was written to
stop, one level up.** Fixed by making re-resolution a no-op on an already-resolved arm.

**What I found reading the diff.** That no-op keyed on the mere *presence* of recorded evidence, which
would let `replace(arm, scores=<other>)` inherit a band another arm's clustering earned — the guard
whose entire purpose is to be unbypassable, bypassable. Not reachable today (the core's only
`replace` on an `Arm` touches `tolerance`), so it is closed while still theoretical rather than after
it costs a retraction: the short-circuit now keys on a digest of the exact scores the clustering was
measured on, a mismatch makes the arm re-earn its band, and a test pins it.

Cost: ≈143k agent tokens, 11 min, local CPU. **Open problem 4j is closed. 4i and 4k stand**, and the
phase-2 gate decision is still outstanding. Not tested: anything remote; no row in the repo yet
declares an independent unit, so both bands remain exercised only by tests and planted cases.

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
4i. **The permutation arm sits below its null on 25 of 25 replication rows** (h48; mean −0.117, all
   inside their bands, sign test p ≈ 3e-8). The uncentered cosine ranking is partly a norm ranking.
   Not tuned away — centering would make the floor and the treatment different procedures — and at
   ~60–100 draws it would begin refusing rows. A defect of this readout's null, open.
4j. ~~**`checks.cluster_evidence` computes a design effect that nothing uses**~~ (h48) closed at h49:
   the band is now `clamp(round(n/deff), k, n)`, h8's case is unchanged, and rediscovery case 13
   plants the arm the old k-flat band admitted. Original text: The band is taken
   on the cluster count, so the guard decides *whether* an arm may widen its band but never *how
   much*; where clustering is partial the honest effective n is `n/deff` ≫ k, and banding on k is
   over-wide — the permissive direction. Latent, not active: no row declares a reduced unit yet.
4k. **Nothing built from a cached `.npz` can reach the ledger** (h48): `ProvenanceNotFromStack`
   refuses all 27 of the phase-2 measurements. Every re-derivation needs re-extraction through
   `build_stack`, and four of the five replication models are not in the local HF cache.
5. Token-level clouds + Gromov-Wasserstein, no role correspondence assumed.
6. ~~A second model family~~ Pythia-1.4B: everything replicates, slightly stronger.
7. ~~Commutator controls~~ done (hour 22): divergence generic, dominance real, regimes noise.
8. ~~Generation-level recomposition at 70B~~ done (hour 27): null at 1×. ~~Scale sweep~~ done (hour 29): 3× re-imposed moves generated text (0.84) on 9B. ~~Same sweep on 70B~~ done (hour 33): only 0.43, does not cross. ~~Layer sweep (selector level)~~ done (hour 34): depth does not explain the cap. ~~Piece 1~~ done (hour 34): **405B unreachable for this key**. The apparent control collapse was OUR BUG (hour 36: batch-row patching plus tie-ranking); hour 34's selector numbers are withdrawn, the fix is in, and the Llama lens is real (8B: era 1.22, theme 1.33, random 2.11, no-patch 2.00). ~~Re-run the 70B pair battery~~ done (hour 37): tuning sharpens era (1.50→1.06), theme is tuning-invariant; sharp selector alongside hour 33's weak engine. Next: piece 2 (generation arms) with a generation-level layer sweep; re-check hour 8's tense control (1.44); correct the stale `results/ndif_pinned.txt`.
9. ~~Parameterized time translation~~ done (hour 28): shared clock holds and selects; subject clocks absent.
   ~~Vocabulary-matched far-Δt passages~~ done (hour 30): the clock survives. ~~Residual curves on Gemma-9B~~ done (hour 31): clock invariant. **The "subject clocks absent" finding of hours 28/30/31 is WITHDRAWN (hour 32): the probe was at its noise floor.** The shared-clock numbers in those hours are also confounded: the v2 state texts restate the interval phrase, so Δt is lexically recoverable at layer 0. ~~v3 grid~~ done (hour 35). ~~Clock as depth gain~~ done (hour 38): real gain over the floor (0.297, z 7.6) but order-invariant and non-transferring — a computed register detector, not a clock. **Line closed**; Gemma gate no-go. Subject clocks dropped with reasons on record;
   a subject-clock grid where the same Δt phrase appears with subject-appropriate change only.
