# Spec: the direct-path test for the log-prob selectors (PROGRAM.md Phase 0.1)

*Planner spec. Status: not run. Predictions are registered in §6 before any number exists.*

## 0. The threat, stated so it can be falsified

Every selector in this repo (`stage4.py`, `stage5_factors.py`, `stage6_factors.py`, `ndif_factors.py`,
the h37 70B battery) does the same thing: add a direction `d` to the residual stream at the input of
block `L`, at every position, then read the teacher-forced log-probability of each candidate span
through the final RMSNorm and the unembedding `W_U`. The claim attached to the readout is that the
blocks after `L` *use* `d` — "the stack computes the relation", "the factor is a lens".

But the residual stream is a skip path. At the last position `t` of the scored text,

```
resid_28[t] = resid_L[t] + d + Σ_{l=L..27} block_l(·)
logit[t]    = W_U · norm(resid_28[t])
```

so `d` reaches the logits **whether or not any block does anything with it**. If `d` has positive
dot product with the unembedding rows of the target span's tokens (and on Qwen2.5 `W_U` is tied to
the input embeddings, so any mean-difference direction between spans with different vocabulary has
exactly that), the target's log-prob rises by arithmetic. Hour 40 established this failure for a
cosine readout at layer 20 (six blocks of Gemma partly *undid* the arithmetic). Nobody has checked it
for the log-prob readout, and it is the spine: writeup claims 2 and 4 rest on it entirely.

Two hypotheses, and the test must be able to tell them apart:

- **H_direct.** The selector effect is the direct path. A norm-matched copy of `d` added *after the
  last block* (nothing downstream but norm + unembed) reproduces the rank.
- **H_computed.** Blocks after `L` transform `d` into something the unembedding reads better than `d`
  itself. Patch-at-`L` beats patch-at-final by a margin outside noise, and the margin is not
  explained by depth-dependent washout or amplification of `d` (§4).

The quantity reported is the **computed gain**, `G(L) = rank_final(d_L) − rank_L(d_L)` in rank
units (lower rank is better, so `G > 0` means the blocks helped), with a paired interval. A difference,
never a ratio — `core_v1.md` §2a. The "fraction direct" the program asked for is reported as a
secondary descriptive only where the treatment is well off chance.

## 1. What is and is not in scope

| writeup claim | instrument | logged number | exposed? |
|---|---|---|---|
| 2, role lens | `stage4.py`, `prompts/holonic_v1_rotated.json`, 8 domains, Qwen2.5-1.5B, L20 (h4), sweep 8/14/20/26 (h5) | 1.69/6 at L20 (random 3.25); 1.33 at L14 on 4 domains | **yes — primary** |
| 4, factor lenses | `stage6_factors.py`, `prompts/narrative_factors_v2.json`, 4 scenes, L14 (h8; re-run h39 with mid-rank + no-patch) | era 1.25/3, voice 1.24/3, tense 1.03/2; random 2.00/2.25/1.44; no-patch 2.00/2.00/1.50 | **yes — primary** |
| 4, three composed | same, test (D) | 2.81/18 (no-patch 9.50) | yes — secondary, piece 2 |
| 3, relation selector | `stage3.py` / `stage3_qwen1.5b_v2_rolecentered.json` (h16): affine map on **cached** activations, **no patch** | 2.21/6, peak 1.73 at L16 | **no.** Already recorded clean at h40. What claim 3 owes is a lexical-vs-computed argument (layer-0 rank 2.73 vs 1.73 at L16), which is a different question and not this spec's. |
| 3, relation as a patch | `stage4b_relation_v2.py`, `stage4c_relation_wrong_sources.py` (h20, h25) | +0.176 nats extra gain, wrong-source +0.114, random −0.151 | exposed, but the writeup already presents it as "the edge", not a claim. Optional add-on in piece 2 (§8). |

**Minimum sufficient set if budget bites:** role lens at L ∈ {14, 20} and factor lenses (test B
only) at L = 14, each with the full arm set of §3, plus the final-layer arms which are free. That
alone answers the stop condition. Everything in §4 (sweeps) sharpens the interpretation but does not
change the verdict.

## 2. The patch points, exactly

`LM.patched` installs a **forward-pre-hook on `blocks[L]`**, so `Patch(L, ·)` edits `resid[L]` = the
input to block `L`, `L ∈ 0..27`. `Patch(27)` therefore leaves one full block of computation. The
control this spec needs is a patch point **after block 27 and before the final norm**, `resid[28]`,
where no block follows.

**Change to `src/lsx/model.py` (the only measurement-code change; the diff is read, not the report):**
extend `LM.patched` so that `Patch(layer=lm.n_layers, fn)` installs a forward-pre-hook on the final
norm module (`model.model.norm` for Qwen2; resolve it the way `LM.unembed` already does). Nothing
else in the hook contract changes. Add to `tests/test_invariants.py`:

- `test_final_layer_patch_is_offline_unembed`: for a short text, `logprob(prefix, cont, [Patch(n_layers, add_vector(v))])`
  equals, to 1e-4 nats, the offline computation `Σ_t log_softmax(W_U · norm(resid_28[t] + v))[tok_{t+1}]` on
  `resid_28` captured from an **unpatched** forward. This is the identity that makes the final-layer arm
  computable without a model, and it must hold or the arm is wrong, not the claim.
- `test_final_layer_zero_patch_is_identity` (zero vector → base log-probs exactly).

Because that identity holds, the final-layer arm is run **offline** from one cached unpatched forward
per scored text (`resid_28[t]` for all `t`, plus per-layer position norms — see §2.1) and is
therefore free: scale sweeps, random directions, and alternative directions at the final layer cost
nothing. The in-model pre-hook is used only to verify the offline arm on ≥ 10 cases per claim
(recorded in the json as `final_arm_identity_max_abs_err`).

Note on capture (h40): HF's `hidden_states[L]` is recorded *before* the pre-hook on block `L` fires,
so a patch at `L` is invisible at `hidden_states[L]` and first visible at `L+1`. `hidden_states[28]`
under a patch at `L ≤ 27` is correctly patched. This matters for §4.3 only.

### 2.1 Norm matching

Residual norm on Qwen2.5-1.5B grows with depth and position 0 carries a massive-activation sink.
An unmatched comparison between a layer-14 vector and a layer-28 residual is meaningless (h40 §"layer
dependence" shows exactly the artefact that results: a fake growth-with-depth of the gain).

Define, for a scored text and layer `l`, `r_l = mean_{t ≥ 1} ‖resid_l[t]‖` over the positions that
are scored (last lead token through the span; position 0 excluded; store both with and without
position 0 in the json). The treatment at `L` with scale `s` has relative size `ρ_L = s‖d_L‖ / r_L`
(≈ 0.18–0.19 at L20 for role directions, h4). The **norm-matched final-layer vector** is

```
v_final = d_L · (ρ_L · r_28 / ‖d_L‖)
```

i.e. the *same direction*, at the *same fraction of the residual* it had where it was injected. This
is the arm the verdict rests on, as in h40. Because the direct-path hypothesis could be "underdosed"
at the matched scale, the final arm is also run at `ρ_L × {0.5, 1, 2, 4}` (free, offline) and the
most favourable of these is reported as `F_best`. The verdict uses `F` (matched) for the headline
gain and `F_best` for the robustness line (§6); a lens that the direct path reproduces only at 4×
the dose while the treatment was never given 4× is reported as "direct only when overdosed", a
qualified pass, not a stop. The treatment itself is run at `s = 1` (the logged scale) and `s = 2` at
the logged layer only, so that comparison is symmetric to one doubling.

## 3. Arms

Per claim, per patch layer `L`, per case (role lens: held-out domain × target role, 48 cases; factor
lens: held-out scene × factor × level, 32 cases, each ranked among its own factor's levels with the
other factors fixed exactly as `stage6_factors.py` test B does). Rank is **mid-rank on ties**
(`1 + #{>} + ½·#{=}`) everywhere, including in the role-lens script, which today still uses strict
`>` and has no no-patch arm (`stage4.py`; the h39 audit fixed stage5/6 and left stage4 as logged).

| arm | what is added, where | expected null / prediction | why it is there |
|---|---|---|---|
| **N** no-patch | zero vector, same code path | rank **exactly** chance (3.5, 2.0, 1.5), gains exactly 0.0 | plumbing. Anything else is a bug. |
| **A** treatment | `s·d_L` at input of block `L`, all positions (the logged instrument) | reproduces the logged number at the logged layer within 0.05 rank (gate, §5) | the claim |
| **R** random | equal-norm Gaussian at `L`, 2 draws per case, same rng order as the logged script | chance ± 0.3 (role), ± 0.25 (factor); tense random may sit at ~1.44 as in h8 while its N reads 1.50 | the logged control |
| **F** final, same direction | `v_final` as in §2.1, at `resid_28` (offline) | this is what the test measures; predictions in §6 | the direct path, isolated |
| **F_best** | F at `ρ_L × {0.5, 1, 2, 4}`, best rank | ≤ F | closes the "underdosed control" objection |
| **F_R** final, random | equal-`ρ` Gaussian at `resid_28`, 2 draws | chance ± 0.3. **If F_R is off chance the norm matching is destroying the residual and F is not interpretable**; report and reduce `ρ` before reading F. | control on the control |
| **U** unembedding-only | no forward at all: `score_c = Σ_{t ∈ span_c} d_L · W_U[tok_t] / (‖d_L‖·|span_c|)`, ranked | high U rank (near 1) means `d_L` literally points at the target's vocabulary | tells *which kind* of direct path: vocabulary alignment (U ≈ F ≈ A) versus something that needs the base residual under the norm (U at chance, F below chance). Free. Also a leak check on the direction. |
| **P** pure-direct positive control | `d_P = mean_{t∈target span} W_U[tok_t] − mean over the other candidates' tokens`, unit-normed, scaled to `ρ_L`; run at `L` (model, arm P_L) and at final (offline, P_F) | P_F rank ≈ 1.0–1.3 by construction; `G_P = P_F − P_L ≤ 0` (blocks cannot help a vocabulary vector; they may hurt) | shows the instrument reads a known-direct effect as `G ≤ 0`. A direction built from the held-out span's own tokens is cheating on purpose. |
| **C** computation-only | `s·d_L` at `L` **at lead positions only**, `t < n_lead − 1` (strictly before the last lead token) | its final-layer arm is *identically* N for every span token after the first (no block follows, and position `t`'s logit depends only on `resid_28[t]`), so **any** rank effect in C_L is computed through attention. Prediction in §6. | the direct path is removed by construction rather than subtracted; the cleanest single statement available. Run at the logged layer only. |
| **C_plumb** | huge random vector (`ρ = 4`) at lead positions only, at `L` | all six/eighteen candidates' log-probs move by ≫ 0.1 nats in the model; **exactly 0.0** for span tokens ≥ 2 in the offline final arm | demonstrates that lead-only patches do reach span logits through computation, so a null C is a finding, not a dead arm |

Statistics per arm: mean rank; mean target gain and mean other-span gain in nats; margin
`m = gain_target − mean(gain_others)`. Per layer: `G_rank(L) = mean_cases[rank_F − rank_A]` and
`G_nats(L) = mean_cases[m_A − m_F]`, each with a 90% paired bootstrap interval over cases, resampling
**domains** (role, n = 8) or **scenes** (factor, n = 4; report the interval as indicative only at
n = 4 and add the per-scene values). The noise floor (CLAUDE.md #3) is the re-run tolerance of the
logged instrument: the h39 audit reproduced h8 to two decimals; h40 quotes ±0.02–0.03. A gain inside
2× that on a rank/3 scale is not a gain.

## 4. The depth confound, and how the sweeps separate it

Patch at 2 versus patch at 27 differ in norm (handled by §2.1), in how many blocks can suppress or
amplify the vector, in whether the direction estimated at that layer is even the same direction, and
in whether the model has already committed. Two points cannot separate these; a curve can.

**4.1 Native sweep.** `L ∈ {0, 4, 8, 12, 14, 16, 20, 24, 26, 27}` plus final (28). At each `L` the
direction is `d_L` estimated at `L` from the stacks, exactly as `stage4.py --layers` and
`stage6_factors.py --layer` do. Arms A, R, N, F, F_R, U at every `L`; P at `{14, 20, 27}`. Report
`rank_A(L)`, `rank_F(L)`, `G(L)`. Under H_direct the two curves lie on top of each other. Under
H_computed `G(L)` is largest at mid-depth and falls toward `L = 27`, where `G(27)` is the contribution
of one block and must be ≈ 0 by construction if the instrument is sound — **`G(27) ≈ 0 is a built-in
sanity check on the whole method**.

**4.2 Transport sweep.** Fix `d* = d_14` (factor) and `d* = d_20` (role — the logged direction).
Add `d*` at every `L` in the same set, with `ρ` matched to its value at the source layer
(`v = d* · ρ_src · r_L / ‖d*‖`). This holds the direction constant and varies only the amount of
computation after it, which 4.1 does not: in 4.1 a late layer may fail because `d_26` is a worse
direction, not because fewer blocks follow (h5's 2.67 at L26 is ambiguous for exactly this reason).
Also record `cos(d_L, d_28)` and `cos(d_L, d_P)` per layer from the stacks — if `rank_F(L)` tracks
`cos(d_L, d_P)`, that names the mechanism of the direct path.

**4.3 Washout versus computation, measured directly.** On the treatment pass at `L` capture
`hidden_states[28]` (correctly patched, §2) and form `Δ_28 = resid_28^A − resid_28^base` at the scored
positions. Decompose against the injected `v = s·d_L`: `survival = (Δ_28·v̂)/‖v‖` (< 1 washout, > 1
amplification along the direction) and `orth = ‖Δ_28 − (Δ_28·v̂)v̂‖ / ‖v‖` (what the blocks produced
that is *not* the direction). Then a fourth free offline arm, **F_Δ**: add `Δ_28` itself at the final
layer — this reproduces A exactly (identity check), and adding only its parallel component
`(Δ_28·v̂)v̂` isolates "the direction, rescaled by the stack" from "new content the stack wrote".
If `F_par ≈ A` the blocks only rescale `d`; if `F_par ≈ F < A` the orthogonal part is where the lens
lives. This is the direct answer to "washed out or amplified by depth" versus "computed", and it
costs one `output_hidden_states=True` on passes already being run.

## 5. Gates before any number is read

1. **Reproduction.** With the new script, arm A at the logged layer and scale reproduces the logged
   numbers: role lens L20 s=1 **1.69** (random 3.25), factor lenses L14 **1.25 / 1.24 / 1.03**
   (random 2.00 / 2.25 / 1.44, no-patch 2.00 / 2.00 / 1.50), to ±0.05. The role-lens number must be
   re-derived under mid-rank (ties are measure-zero for real gains, so it should not move; if it
   moves, say so).
2. **N reads exactly chance** on every test with all gains 0.0.
3. **Final-arm identity** (offline = in-model pre-hook to 1e-4 nats on ≥ 10 cases per claim; zero
   vector at final = base).
4. **F_R at chance** at the matched `ρ` (and note at which multiple of `ρ` it leaves chance).
5. **Positive controls behave:** `P_F` well below chance and `G_P ≤ 0.1`; `C_plumb` moves the model
   log-probs and is exactly zero offline. If P or C_plumb fails, the instrument cannot distinguish the
   hypotheses and **no verdict is issued** — the note says so and stops.

## 6. Registered predictions and the stop condition

Ranks are mean over cases; role lens /6 (chance 3.5), era and voice /3 (chance 2.0), tense /2
(chance 1.5). The logged treatment is A. Predictions are for the logged layer at `s = 1`, `F` matched.

| claim | A (logged) | predicted F | predicted G_rank | predicted U | planner's belief that the lens is substantially (≥ 2/3 of its distance from chance) direct-path |
|---|---|---|---|---|---|
| role lens, L20 | 1.69 | 2.4 ± 0.4 | **+0.7** | ~2.8 | 0.30 |
| role lens, L14 | ~1.35 (4-domain h5) | 2.2 ± 0.4 | +0.8 | ~2.8 | 0.30 |
| era, L14 | 1.25 | 1.5 ± 0.2 | **+0.25** | ~1.7 | 0.45 |
| voice, L14 | 1.24 | 1.35 ± 0.15 | +0.1 (inside noise) | ~1.4 | 0.65 |
| tense, L14 | 1.03 | 1.05 ± 0.05 | 0.0 | ~1.05 | 0.80 |
| C (lead-only), role L20 | — | ≡ N | rank 2.6–3.2 in the model | — | if C is at chance the lens acts only at span positions, which is compatible with either hypothesis; C off chance is unambiguous computation |
| P (unembed rows), any L | — | 1.0–1.3 | ≤ 0 | ≈ 1.0 | by construction |
| G(27), all claims | — | — | 0.0 ± 0.1 | — | sanity |

Why these numbers. Voice and tense are decodable at layer 0 (0.92, 1.00), and their directions are
mean differences between spans that differ in vocabulary; with `W_U` tied to `W_E` those directions
carry the target vocabulary's unembedding rows at every layer. Tense is the strongest case — the
present/past pairs differ only in verb forms, and a direction that separates them is a verb-suffix
direction. Era rises from 0.31 at layer 0 to 0.92 at 12, so its *representation* is computed, but the
era-1920s direction still has *Packard* and *speakeasy* in it. The role lens has the best chance of
being computed: role spans share register cues ("this looks like" for from_below) but the lens fading
at L26 in h5 (2.67, target gain −0.04) is the wrong shape for a pure direct path, which should if
anything improve as fewer blocks stand between it and the unembedding — unless that fade is the
unmatched-norm artefact, which is exactly what §2.1 and §4.2 settle.

**Stop condition (PROGRAM.md 0.1 branch), operationalised.** Stop, write Checkpoint 3 before
anything else, if **either**:

- **role lens:** at both L14 and L20, `G_rank ≤ 0.25` with the 90% bootstrap upper bound `< 0.5`,
  **and** `C` at chance, **and** `F_par ≈ A` (§4.3) — i.e. nothing the blocks do beyond rescaling `d`
  shows in the readout; or
- **factor lenses:** era, voice and tense all have `G_rank ≤ 0.15` with upper bound `< 0.3` (rank/3;
  `≤ 0.1` / `< 0.2` for tense on rank/2).

Voice and tense being direct while era and role are computed is **not** a stop; it is the registered
prediction of the program and becomes a one-sentence rewording of claim 4 ("lexical factors are read
straight off the unembedding; era is not"). Role or era direct is a stop. A mixed result where `F`
matched leaves gain but `F_best` at 4× removes it is reported as qualified, not a stop, with the dose
stated.

The condition is evaluated with the gates of §5 passed. A stop with a failed gate is not a stop; it
is a broken instrument, and it goes in the note as such.

## 7. Execution plan

**Environment.** `/home/user/latent-space-exploration/.venv/bin/python` (py3.11, CPU torch 2.14,
transformers 5.17 — record both in the json), `HF_HOME=/home/user/latent-space-exploration/cache/hf`,
`HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_XET=1`. Qwen2.5-1.5B only, cached; download nothing. Four cores,
15 GB. `PYTHONPATH=src`.

**Stacks** (gitignored; present at time of writing, re-extract if missing, and verify shapes
`(29, 6, 1536)` × 48 keys and `(29, 1536)` × 72):
```
.venv/bin/python scripts/make_rotated.py prompts/holonic_v1.json          # only if prompts/holonic_v1_rotated.json is missing
.venv/bin/python scripts/extract_grid.py prompts/holonic_v1_rotated.json --model Qwen/Qwen2.5-1.5B
.venv/bin/python scripts/extract_factors.py prompts/narrative_factors_v2.json --model Qwen/Qwen2.5-1.5B
```
→ `results/stacks_qwen2.5_1.5b_holonic_v1_rotated.npz`, `results/stacks_qwen2.5_1.5b_narrative_factors_v2.npz`.

**New script** `scripts/selector_direct_path.py`, one file, two subcommands `role` and `factors`,
reusing the direction construction, prompt assembly, rng draw order and candidate sets of `stage4.py`
and `stage6_factors.py` verbatim (copy, do not import their module-level argparse). Options
`--layers`, `--transport-from`, `--scales`, `--out`. The scored candidates of one case may be batched
in one right-padded forward as `stage4b_relation_v2.py` does (≈3× faster on CPU) **only if**
`test_patch_reaches_every_sequence_in_a_batch` covers the new final-norm hook and the batched
log-probs equal the single-text log-probs to 1e-4 on the shortest item (core_v1 rule). Otherwise
single-text.

**Illustrative offline final arm** (the whole arm, so the executing agent has no room to improvise):
```python
# cache once per scored text: unpatched forward, output_hidden_states=True
h28 = hidden_states[28][0]                        # [seq, d], before final norm
r   = {l: hidden_states[l][0, 1:].norm(dim=-1).mean() for l in range(29)}   # position 0 excluded
def final_arm_logprob(v, n_p, ids):               # v: [d] already norm-matched (§2.1)
    z = lm.model.model.norm(h28 + v)              # RMSNorm, then tied head
    lp = torch.log_softmax(lm.model.lm_head(z)[:-1].float(), -1)
    tgt = ids[1:]; return float(lp[torch.arange(n_p - 1, len(tgt)), tgt[n_p - 1:]].sum())
# identity: final_arm_logprob(0) == lm.logprob(prefix, cont); and == lm.logprob(..., [Patch(28, add_vector(v))])
```

**Cost.** Time 20 `lm.logprob` calls first and write the number in the note. Per patch layer:
role lens ≈ 8 × 6 × 6 × (A + 2R + P) + N ≈ 1.2k single-text forwards; factor test B ≈ 4 × 8 × 18 ×
(A + R + P) + N ≈ 1.8k. All final-layer arms are free after one cached forward per text (48 + 72).
Sweep of 10 layers ≈ 30k forwards; at 0.5 s that is ~4 h, at 1.5 s ~12 h. Order of execution, so a
budget cut still leaves the verdict:

1. gates (§5) and the minimum set (§1): role L14, L20; factors L14 — A, R, N, F, F_best, F_R, U, P, C, C_plumb;
2. §4.3 decomposition at the same layers (no extra forwards beyond `output_hidden_states`);
3. native sweep §4.1 at the remaining layers;
4. transport sweep §4.2;
5. optional add-ons (§8).

**Pieces.** One agent if step 1–2 take under ~3 h wall-clock by the timing test; otherwise two,
sequential: **piece 1 = steps 1–2 and the verdict** (runs first, must stand alone), **piece 2 = steps
3–5** (sharpens, does not change the verdict). Piece 2 must not begin until piece 1's note is written.

**Deliverables** (executing agents write nothing outside these; RESULTS.md, VISION.md, README.md,
WRITEUP.md, docs/ALGEBRA.md are off limits):
- `scripts/selector_direct_path.py`; the `model.py` hook extension and its two tests;
- `results/selector_direct_path_role.json`, `results/selector_direct_path_factors.json` — per-case rows
  (arm, layer, scale, ρ, rank, gain_target, gain_others, plus the §4.3 survival/orth per case), the
  gate checks, library versions, timing;
- `results/notes/selector_direct_path.md` — the gate table, one table per claim in the form of §6 with
  observed beside predicted, the `G(L)` curves (both sweeps), the §4.3 decomposition, the C result,
  and a one-paragraph verdict per claim that names which hypothesis survived and whether the stop
  condition fired. Figures optional under `results/figures/`.

**Rules carried from CLAUDE.md.** Every table shows A, R and N side by side with their declared nulls
(#1). Every arm's null is written down before the run (§3). Never select a layer on scoring data:
report the curve, and the verdict is read at the *logged* layers (#4). The reviewer reads the diff of
`model.py` and the direction/ranking code, not the report (#5).

## 8. Optional add-ons for piece 2, in priority order

1. **Composed patch at final** (claim 4, test D): `d_era + d_voice + d_tense` norm-matched at
   `resid_28` versus at L14 (logged 2.81/18, no-patch 9.50). Free offline once the factor forwards are
   cached; the model arm is 72 more forwards.
2. **Relation patch at final** (h25 "edge"): `dir_T + 0.5·pred` and `dir_T + 0.5·pred_wrong_source`
   at final versus at L16. The h25 finding that *any* source's output direction helps (+0.11) has the
   smell of a direct path; if `F` reproduces it, the edge is arithmetic and the paragraph in the
   writeup ("the output direction is useful") is withdrawn. Uses `stage4c_relation_wrong_sources.py`'s
   operator fit on the holonic_v2 stacks (re-extract `holonic_v2_rotated`, 240 prompts, ~20 min).
3. Same test on `Qwen2.5-1.5B-Instruct` theme (h11) — only if everything above is done, and not on the
   chat template.

## 9. What this spec does not cover

The 70B h37 battery and Gemma/GPT-J factor runs (`ndif_factors.py`) have the same exposure and the
same free arm (`resid` at the last block + norm + head, offline from one cached job per text), but
they wait for the local verdict: if the 1.5B lenses are direct, the remote ones are withdrawn without
a run; if computed, the remote arm is a piece of `scale_vs_tuning_v1.md`, not of Phase 0.
