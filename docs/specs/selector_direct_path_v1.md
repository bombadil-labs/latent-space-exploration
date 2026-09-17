# Spec: the direct-path test for the log-prob selectors (PROGRAM.md Phase 0.1) — v2

*Planner spec, revised after adversarial review ("build with changes"). Status: not run. Predictions
are registered in §6 before any number exists. v1 → v2 changes are listed in §10.*

## 0. The threat, stated so it can be falsified

Every selector in this repo (`stage4.py`, `stage5_factors.py`, `stage6_factors.py`, `ndif_factors.py`,
the h37 70B battery) does the same thing: add a direction `d` to the residual stream at the input of
block `L`, at every position, then read the teacher-forced log-probability of each candidate span
through the final RMSNorm and the unembedding `W_U`. The claim attached to the readout is that the
blocks after `L` *use* `d` — "the stack computes the relation", "the factor is a lens".

But the residual stream is a skip path. Writing `pre_28[t]` for the residual entering the final norm,

```
pre_28[t] = resid_L[t] + d + Σ_{l=L..27} block_l(·)
logit[t]  = W_U · norm(pre_28[t])
```

so `d` reaches the logits **whether or not any block does anything with it**. On Qwen2.5-1.5B
`lm_head.weight` shares storage with `embed_tokens` (verified), so any mean-difference direction
between spans with different vocabulary has positive dot product with the target span's unembedding
rows at every layer. Hour 40 established the failure for a cosine readout at layer 20. Nobody has
checked it for the log-prob readout, and it is the spine: writeup claims 2 and 4 rest on it entirely.

**Capture warning (verified by the coordinator, docstring corrected, test added):** on this
environment `LM.residuals()[-1]` / `hidden_states[28]` is the final-norm **output**, not the last
block's residual (`hs[28] == norm(pre)`; mean position norm 190.5 vs 283.2). Everywhere this spec
says `pre_28` it means the tensor captured by a **forward-pre-hook on `model.model.norm`**, never
`hidden_states[28]`.

Two hypotheses, and the instrument must be shown to tell them apart before it grades a claim:

- **H_direct.** The selector effect is `d` itself reaching the unembedding, possibly rescaled by the
  blocks along its own direction. A one-parameter family — `c·d` added at `pre_28` with `c` free —
  explains the treatment's per-case margins to within noise.
- **H_computed.** Blocks after `L` write content **orthogonal to `d`** that the unembedding reads
  better than `d` itself. The treatment beats the "direction plus rescaling" arm by a margin whose
  90% lower bound is above zero.

**Posture (inverted from v1):** the burden is on the claim. "Computed" is the conclusion only if the
test demonstrates it; every other outcome is "not demonstrated".

## 1. What is and is not in scope

| writeup claim | instrument | logged number | exposed? |
|---|---|---|---|
| 2, role lens | `stage4.py`, `prompts/holonic_v1_rotated.json`, 8 domains, L20 (h4), sweep 8/14/20/26 (h5) | 1.69/6 at L20 (random 3.25); 1.33 at L14 on 4 domains | **yes — primary, piece 1** |
| 4, factor lenses era/voice/tense | `stage6_factors.py` test B, `prompts/narrative_factors_v2.json`, 4 scenes, L14 (h8; h39 re-run with mid-rank + no-patch) | era 1.25/3, voice 1.24/3, tense 1.03/2; random 2.00/2.25/1.44; no-patch 2.00/2.00/1.50 | **yes — primary, piece 1** |
| 4, three composed | same, test D | 2.81/18 (no-patch 9.50) | **yes — claim 4's headline, piece 1** |
| 4, diagonal cross-talk | same, test X | era 0.58 / voice 0.64 / tense 0.47 on-diagonal, random 0.25/0.21/0.02 | **yes.** Vocabulary geometry alone predicts a diagonal (the era direction shares tokens with era spans, not voice spans), so the matrix is exposed; the final-layer version is free from the piece-1 forwards. |
| 4, mood lens | `stage6_factors.py`, `prompts/narrative_mood_v1.json`, L14/last-token (h9) | 1.28/3 | exposed; **piece 2** |
| 4, theme lens | `stage6_factors.py`, `prompts/narrative_theme_v1.json`, L20 (h10) | 1.25/3, composes 2.2/9 | exposed; **piece 2** |
| 3, relation selector | `stage3.py` (h16): affine map on cached activations; the script never constructs an `LM` and never patches | 2.21/6, peak 1.73 at L16 | **no.** Recorded clean at h40, confirmed by review. Claim 3's lexical-vs-computed question (layer-0 rank 2.73 vs 1.73 at L16) is a different question and not this spec's. |
| 3, relation as a patch | `stage4b_relation_v2.py`, `stage4c_relation_wrong_sources.py` (h20, h25) | +0.176 nats extra gain, wrong-source +0.114, random −0.151 | exposed; the writeup presents it as "the edge". Add-on, piece 2 (§8). |

**Minimum sufficient set if budget bites:** role lens at L ∈ {14, 20}; factor lenses B and D at
L = 14; the full arm set of §3 at those layers. All final-residual arms are free once the treatment
and base forwards exist. That set answers the stop condition; §4's sweeps sharpen the interpretation.

## 2. The patch points, exactly

`LM.patched` installs a **forward-pre-hook on `blocks[L]`**, so `Patch(L, ·)` edits the input to block
`L`, `L ∈ 0..27`. `Patch(27)` leaves one full block of computation. The control this spec needs is a
patch point **after block 27 and before the final norm** — `pre_28` — where no block follows.

**Change to `src/lsx/model.py` (the only measurement-code change; the diff is read, not the report):**

- extend `LM.patched` so that `Patch(layer=lm.n_layers, fn)` installs a forward-pre-hook on the final
  norm module (resolve it the way `LM.unembed` does). Hook contract otherwise unchanged;
- add `LM.pre_norm_residual(text, patches=None) -> (pre_28 [seq, d], per-layer position norms r_l,
  logits [seq, V])` implemented with a forward-pre-hook on the final norm that saves its input, run
  under `patched(patches)`. This is the only sanctioned way to obtain `pre_28`; `hidden_states[28]` is
  never used for it.

Add to `tests/test_invariants.py`:

- `test_final_layer_patch_is_offline_unembed`: `logprob(prefix, cont, [Patch(n_layers, add_vector(v))])`
  equals, to 1e-4 nats, the offline `Σ_t log_softmax(W_U · norm(pre_28[t] + v))[tok_{t+1}]` on `pre_28`
  captured by the pre-hook from an unpatched forward. This identity is what makes every final-residual
  arm computable without a model; it must hold or the arm is wrong, not the claim.
- `test_final_layer_zero_patch_is_identity` (zero vector at `pre_28` → base log-probs exactly).
- `test_pre_norm_capture_differs_from_hidden_states_last` (`pre_28 ≠ hs[28]`, `norm(pre_28) == hs[28]`),
  unless the coordinator's `test_last_hidden_state_is_post_norm` already covers it — check first.

Capture note (h40): `hidden_states[L]` is recorded before the pre-hook on block `L` fires, so a patch
at `L` is first visible at `L+1`. The norm pre-hook sees the patched `pre_28` for any `L ≤ 27`.

### 2.1 Norms and doses

For a scored text and layer `l`, `r_l = mean_{t ≥ 1} ‖resid_l[t]‖` over the scored positions (last lead
token through the span; position 0, the attention sink, excluded; both variants stored). `r_28` is
computed from `pre_28`, not from `hidden_states[28]`. The treatment at `L` with scale `s` has relative
size `ρ_L = s‖d_L‖ / r_L` (≈ 0.18–0.19 at L20 for role directions, h4).

**The dose problem the review identified.** v1 matched `ρ` at the final residual, i.e. added
`d_L · r_28/r_L`. But the actual skip connection delivers `d_L` at *absolute* norm `s‖d_L‖`; relative
matching hands the null ≈ 3.5× (L20) to ≈ 5× (L14) the strength the direct path really has, and a
`×4` sweep on top makes that 14–20×. A null that strong can reproduce any lens for the wrong reason.
v2 therefore treats dose as a **continuous parameter of the H_direct family** (§4.3) and uses the
relative-norm point only as a labelled robustness line.

## 3. Arms

Per claim, per patch layer `L`, per case (role lens: held-out domain × target role, 48 cases in 8
domain clusters; factor lens: held-out scene × factor × level, 32 cases in 4 scene clusters, each
ranked among its own factor's levels with the other factors fixed as `stage6_factors.py` test B does;
composed: held-out scene × combo, 72 cases). Rank is **mid-rank on ties** everywhere, including in the
role-lens script, which today uses strict `>` and has no no-patch arm (`stage4.py`; the h39 audit fixed
stage5/6 and left stage4 as logged). The base pass for every scored text captures `pre_28`, `r_l`, and
the scored-position logits (§2); the treatment pass A captures the same under the patch.

Margin per case and arm: `m = gain_target − mean(gain_others)` in nats, gains relative to base.

| arm | what is added, where | declared null / expectation | why |
|---|---|---|---|
| **N** no-patch | zero vector, same code path | rank **exactly** chance (3.5, 2.0, 1.5, 9.5), all gains exactly 0.0, `m = 0` | plumbing. Anything else is a bug. |
| **A** treatment | `s·d_L` at input of block `L`, all positions (the logged instrument) | reproduces the logged rank at the logged layer within 0.05 (gate, §5) | the claim |
| **A_span** | `s·d_L` at `L`, scored positions only (`t ≥ n_lead − 1`) | ≈ A if the lens acts where it is read | the tight pair with the final-residual arms, which by construction act only at scored positions; `A − A_span` is what the lead positions contribute through attention |
| **R** random | equal-norm Gaussian at `L`, 2 draws per case, same rng order as the logged script | `m_R ≈ 0`; rank chance ± 0.3 (role), ± 0.25 (factor); tense random may sit ~1.44 as in h8 while N reads 1.50 | the logged control; also sets the noise floor τ (§4.4) |
| **F_Δ** identity | `Δ_28[t] = pre_28^A[t] − pre_28^base[t]` added per position at `pre_28`, offline | **≡ A** to 1e-4 nats | gate: proves the capture is pre-norm and the offline arm is exact |
| **F_abs** literal direct term | `s·d_L` (the very vector the skip carries, unscaled) at `pre_28`, offline | this is the direct path with no assistance from any block | the literal H_direct |
| **F_par** direct + rescaling | `(Δ_28[t]·d̂_L) d̂_L` per position at `pre_28`, offline | ≥ F_abs in effect if the blocks amplified `d`; the rest of Δ is orthogonal content | the decomposition's null: everything the stack did *along* `d` is given to the null |
| **F(c)** dose family | `c·d_L` at `pre_28`, `c ∈ {0.25, 0.5, 1, 2, 3.5, 5, 8, 12, 16}·1`, offline; `c = 1` is F_abs, `c = r_28/r_L` is v1's relative-norm point (record it as `F_rel`, robustness line only) | `m_F(c)` per case; §4.3 fits `c*` | H_direct as a one-parameter family |
| **F_KL** dose-free | `c_KL` chosen per case so that the mean over scored positions of `KL(p_base[t] ‖ p_F(c)[t])` equals the treatment's `KL(p_base ‖ p_A)`; bisection on `c`, non-monotone cases flagged | equal output perturbation; compare `m_A` with `m_F(c_KL)` | the invariance that needs no norm argument at all |
| **F_R** | equal-`c` Gaussian at `pre_28` at `c = 1` and `c = c_KL`, 2 draws | `m ≈ 0`; if F_R is off null at `c_KL` the KL-matched dose is destroying the residual and F_KL is not interpretable — report and read F_par | control on the control |
| **U** unembedding-only | no forward: `score_k = Σ_{t∈span_k} d_L·W_U[tok_t] / (‖d_L‖·|span_k|)`, ranked and as a margin | near rank 1 means `d_L` literally points at the target vocabulary | names *which* direct path; free; a leak check on the direction |
| **P** pure-direct positive control | `d_P = mean_{t∈target span} W_U[tok_t] − mean over other candidates' tokens`, unit-normed, at `s‖d_L‖`; at `L` (model) and at `pre_28` (offline) | `m_{P,F} > 0` large; `A − F_par` for P must be **≤ τ** (blocks cannot write new content that helps a vocabulary vector; they may hurt) | the instrument must read a known-direct effect as "not computed" |
| **C** computation-only | `s·d_L` at `L` at lead positions only, `t < n_lead − 1` | **no scored position is patched**, so C's final-residual arm is identically N for every span token *including the first*; any effect in C is computed through attention | direct path removed by construction, not subtracted. Logged layer only. |
| **C_plumb** | Gaussian at `c = 4·s‖d_L‖` at lead positions only, at `L` | every candidate's log-prob moves by ≫ 0.1 nats in the model; **exactly 0.0** offline at `pre_28` | a null C is a finding, not a dead arm |
| **D** composed (claim 4 headline) | `d_era + d_voice + d_tense` at L14 (A_D) and its F_Δ / F_abs / F_par / F(c) / F_KL at `pre_28` | logged 2.81/18, no-patch 9.50 | piece 1 |
| **X** cross-talk at final | the variance decomposition of test X computed on the F_par and F_KL gains of the single-factor patches (same 18 candidates as B; no extra forwards) | if the final-residual matrix is as diagonal as the logged one, "diagonal cross-talk" is vocabulary geometry | piece 1, free |

## 4. Statistics, decomposition, and the depth confound

### 4.1 Primary statistic

Per claim, at the logged layer, **`G_new = mean_cases[m_A − m_F_par]` in nats** — the margin the
treatment has over "the direction plus whatever the stack did along it". Beside it: the paired
**sign fraction** `#{m_A > m_F_par}/n`, the **90% bootstrap lower bound** of `G_new`, and the per-cluster
values. The bootstrap resamples **cases** (48 role, 32 factor per factor, 72 composed) and states that
cases cluster in 8 domains / 4 scenes, so the interval is anticonservative for cluster-level noise; a
cluster bootstrap over domains/scenes is reported as a second line, not used for the verdict.

Ranks are **descriptive only** from here on: they are ordinal and bounded, a lens already at 1.03/2
cannot register any gain on a rank scale, and this project has once already had a bounded rank metric
average to its chance value by construction (INSTRUMENTS §1). Rank curves are still printed beside
the logged numbers so the reproduction gate is legible.

Secondary statistics, all in nats, all reported with sign fraction and lower bound:
`G_abs = m_A − m_F_abs` (over the literal direct term), `G_KL = m_A − m_F(c_KL)` (dose-free),
`G_rel = m_A − m_F_rel` (v1's line, robustness only), `A − A_span` (lead-position contribution).

### 4.2 Why F_par is the null

`Δ_28 = pre_28^A − pre_28^base` at each scored position is, exactly, the injected `s·d_L` plus
everything the blocks after `L` changed in response. Splitting it into `(Δ·d̂)d̂` and the orthogonal
remainder gives: **F_abs** = the skip term; **F_par** = the skip term rescaled by the stack
(`survival = (Δ·d̂)/(s‖d_L‖)`, < 1 washout, > 1 amplification); **A − F_par** = the effect of content
the blocks wrote in other directions. Under H_direct the orthogonal remainder is incidental and
`G_new ≈ 0`. Under H_computed it is the lens. Reporting `survival` and `orth = ‖Δ − (Δ·d̂)d̂‖/(s‖d_L‖)`
per case answers "washed out or amplified by depth" separately from "computed".

### 4.3 The H_direct family and its fit

For each case, `m_F(c)` over the dose grid is a curve. Fit one `c*` per claim by least squares on the
per-case margins, `c* = argmin_c Σ_cases (m_A − m_F(c))²`, and report `c*`, the residual RMS at `c*`,
and the fraction of cases whose `m_A` lies inside the family's envelope. If the residual RMS is within
τ (§4.4), a single free dose explains the treatment and H_direct is sufficient regardless of the sign
of `G_new`; say so. `c*` is reported in three units: absolute (×`s‖d_L‖`), relative to the actual skip
term (`c* / 1`), and relative to v1's matching (`c* / (r_28/r_L)`). A `c*` far above 1 is itself a
finding: the direction can reproduce the lens only at a dose the skip connection does not deliver.

### 4.4 Noise floor τ

The forwards are deterministic CPU fp32; re-running 10 cases must reproduce `m_A` to < 1e-3 nats
(record it). The floor that matters is the margin an uninformative direction produces: `τ =
max(|mean m_R|, half-width of the 90% bootstrap of m_R)` at the same layer and norm, from the R arm.
The threshold for "demonstrated" is **`2τ`** on the point estimate of `G_new`.

### 4.5 Sweeps (piece 2; descriptive)

**Native sweep** `L ∈ {0, 4, 8, 12, 14, 16, 20, 24, 26, 27}` with `d_L` estimated at `L`: A, A_span, R,
N, F_abs, F_par, F_KL, U at each; `G_new(L)`, `survival(L)`, `orth(L)`. `G_new(27)` is one block's
contribution and is a built-in sanity check (expected ≈ 0 ± τ). **Transport sweep**: fixed `d* = d_14`
(factor) / `d_20` (role) at every `L`, absolute norm held; separates "the direction estimated late is
worse" from "less computation follows". Record `cos(d_L, d_28)` and `cos(d_L, d_P)` per layer from the
stacks (where `d_28` is estimated from stacks whose last layer is post-norm — say so; re-extract with
the pre-norm capture if piece 2 has time, else drop `d_28`).

## 5. Gates before any number is read

1. **Reproduction.** Arm A at the logged layer and scale reproduces the logged ranks: role L20 s=1
   **1.69** (random 3.25); factor L14 **1.25 / 1.24 / 1.03** (random 2.00 / 2.25 / 1.44, no-patch
   2.00 / 2.00 / 1.50); composed **2.81** (no-patch 9.50); ±0.05. The role number is re-derived under
   mid-rank (ties are measure-zero; if it moves, say so).
2. **N** reads exactly chance with all gains 0.0.
3. **F_Δ ≡ A** to 1e-4 nats on every case, and the zero-vector final patch equals base.
4. **F_R** at null at `c = 1` and at `c_KL` (and the `c` at which it leaves null, recorded).
5. **P** reads as direct: `m_{P,F} ≫ 0` and `A − F_par ≤ τ` for P. **C_plumb** moves the model and is
   exactly 0.0 offline.
6. Deterministic re-run of 10 cases within 1e-3 nats.

If gate 5 fails the instrument cannot distinguish the hypotheses and **no verdict is issued**; the note
says so and stops. If gate 1 fails the logged number is the problem and that is reported first.

## 6. Registered predictions and the decision rule

Predictions at the logged layer, `s = 1`. Nats are margins `m`; the logged instrument never reported
margins, so `m_A` is predicted from the logged mean target/other gains (role L20: +1.10 / −0.76 →
`m_A ≈ 1.9`; factor grid gains were not logged, `m_A` guessed).

| claim | m_A (pred) | m_F_abs | m_F_par | **G_new** | c* (÷ skip term) | planner's belief the lens is substantially direct |
|---|---|---|---|---|---|---|
| role lens L20 | 1.9 | 0.4 | 0.8 | **+1.0**, LB > 0 | 3–5 | 0.30 |
| role lens L14 | 2.0 | 0.3 | 0.8 | +1.1 | 4–6 | 0.30 |
| era L14 | 0.8 | 0.3 | 0.5 | **+0.3**, LB ≈ 0 | 2–3 | 0.45 |
| voice L14 | 0.9 | 0.5 | 0.8 | +0.1, LB < 0 | 1.5–2 | 0.65 |
| tense L14 | 0.6 | 0.4 | 0.6 | 0.0, LB < 0 | ~1 | 0.80 |
| composed D, L14 | 2.0 | 0.9 | 1.6 | +0.4, LB ≈ 0 | 2 | 0.50 |
| X at final | — | — | diagonal, on-diagonal fractions 0.4–0.6 | — | — | the diagonal is mostly geometry: 0.6 |
| P, any L | — | large | ≈ same | ≤ τ | ≈ 1 | by construction |
| C, role L20 | 0.2–0.5 | ≡ 0 | ≡ 0 | = m_C | — | unambiguous computation if `LB > 0` |
| G_new(27), all | — | — | — | 0 ± τ | — | sanity |

Why these numbers: voice and tense are decodable at layer 0 (0.92, 1.00) and their directions are
mean differences between spans that differ in vocabulary; with tied embeddings those directions carry
the target vocabulary's unembedding rows at every layer, and tense in particular is a verb-suffix
direction. Era rises from 0.31 at layer 0 to 0.92 at 12, so its representation is computed, but its
direction still contains *Packard* and *speakeasy*. The role lens fading at L26 (h5: 2.67, target gain
−0.04) is the wrong shape for a pure direct path unless it is a norm artefact, which the transport
sweep settles. Expected `c*` well above 1 for role and era means the literal skip term is too weak to
be the lens — but §4.3 will say whether a rescaled `d` is.

**Decision rule, one primary per claim, no conjunctions.**

| outcome for `G_new` at the logged layer | verdict on the claim |
|---|---|
| 90% lower bound `> 0` **and** point estimate `≥ 2τ` | **computed** — the claim stands as written (with `c*` and survival reported beside it) |
| point estimate `≥ 2τ`, lower bound `≤ 0` | **not demonstrated** — claim reworded to "the direction selects the span; whether the stack contributes beyond the direct path is not demonstrated (G_new = …, LB …)". No stop. |
| point estimate `< 2τ` | **not demonstrated, and PROGRAM.md 0.1's stop fires** for that claim |

Primary per claim: claim 2 → role lens at **L20** (the writeup's number; L14 reported beside it);
claim 4 → the **composed patch D at L14** (the headline "three compose"), with each single factor
reworded individually by the same table. The stop fires if **either** primary lands in the third row.
Voice and tense in the third row while D is in the first is a rewording of claim 4 ("lexical factors
are read straight off the unembedding"), not a stop — that is the program's registered prediction.
The `c*` fit (§4.3) is reported with every verdict; a "computed" verdict whose residual RMS at `c*`
is within τ is downgraded to "not demonstrated" in the note, because a one-parameter direct family
that fits the cases is the more parsimonious account — this is the one place a second statistic can
demote, never promote.

## 7. Execution plan

**Environment.** `/home/user/latent-space-exploration/.venv/bin/python` (py3.11, CPU torch 2.14,
transformers 5.17 — record both in the json), `HF_HOME=/home/user/latent-space-exploration/cache/hf`,
`HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_XET=1`, `PYTHONPATH=src`. Qwen2.5-1.5B only, cached; download
nothing. Four cores, 15 GB.

**Stacks** (gitignored; present at time of writing; re-extract if missing; verify `(29, 6, 1536)` × 48
keys and `(29, 1536)` × 72; note that index 28 of every existing stack is post-norm and is not used by
this spec except as flagged in §4.5):
```
.venv/bin/python scripts/make_rotated.py prompts/holonic_v1.json          # only if prompts/holonic_v1_rotated.json is missing
.venv/bin/python scripts/extract_grid.py prompts/holonic_v1_rotated.json --model Qwen/Qwen2.5-1.5B
.venv/bin/python scripts/extract_factors.py prompts/narrative_factors_v2.json --model Qwen/Qwen2.5-1.5B
```

**New script** `scripts/selector_direct_path.py`, subcommands `role` and `factors`, reusing the
direction construction, prompt assembly, rng draw order and candidate sets of `stage4.py` and
`stage6_factors.py` verbatim (copy; do not import their module-level argparse). Options `--layers`,
`--transport-from`, `--scales`, `--doses`, `--out`. Candidates may be batched in one right-padded
forward as `stage4b_relation_v2.py` does **only if** `test_patch_reaches_every_sequence_in_a_batch`
covers the new final-norm hook and batched log-probs equal single-text log-probs to 1e-4 on the
shortest item; otherwise single-text.

**Illustrative offline arm** (the whole thing, so nothing is improvised):
```python
# base pass, once per scored text
pre28, r, logits_base = lm.pre_norm_residual(text)          # pre-hook on model.model.norm; NOT hidden_states[28]
# treatment pass A at layer L
pre28_A, _, logits_A = lm.pre_norm_residual(text, [Patch(L, add_vector(d, s))])
delta = pre28_A - pre28                                       # [seq, d]; F_delta must reproduce logits_A exactly
dhat = d / d.norm()
par  = (delta @ dhat)[:, None] * dhat[None, :]                # F_par vector, per position
def final_lp(v):                                              # v: [seq, d] or [d]
    z  = lm.model.model.norm(pre28 + v)
    lp = torch.log_softmax(lm.model.lm_head(z)[:-1].float(), -1)
    return float(lp[torch.arange(n_p - 1, len(tgt)), tgt[n_p - 1:]].sum())
kl_A = kl(logits_base, logits_A, scored_positions).mean()     # for F_KL bisection over c in final_lp(c * s * d)
```

**Cost.** Time 20 `lm.logprob` calls first and write the number in the note. Per layer: role
≈ 8·6·6·(A + A_span + 2R + P) + N ≈ 1.8k single-text forwards; factor B ≈ 4·8·18·(A + A_span + R + P) + N
≈ 2.4k; composed D ≈ 4·18·18 ≈ 1.3k. Final-residual arms are free after the base and A passes.
Piece 1 ≈ 8k forwards; at 0.5 s ≈ 1.1 h, at 1.5 s ≈ 3.5 h. Sweeps ≈ 25k more.

**Pieces.** Two, sequential. **Piece 1 (runs first, must stand alone):** §5 gates; role lens at L14,
L20; factors B and D at L14; X at final; every arm of §3; the §4.1–4.4 statistics; the §6 verdict.
**Piece 2:** §4.5 sweeps, mood and theme grids through the same script, §8 add-ons. Piece 2 does not
begin until piece 1's note is written, and cannot change piece 1's verdict.

**Deliverables** (executing agents write nothing outside these; RESULTS.md, VISION.md, README.md,
WRITEUP.md, docs/ALGEBRA.md are off limits):
- `scripts/selector_direct_path.py`; the `model.py` extension (hook at `n_layers`, `pre_norm_residual`)
  and its tests;
- `results/selector_direct_path_role.json`, `results/selector_direct_path_factors.json` — per-case rows
  (arm, layer, scale/dose, m, gain_target, gain_others, rank, survival, orth, kl, c_KL), the gate
  checks, τ, library versions, timing;
- `results/notes/selector_direct_path.md` — gate table; one table per claim in the layout of §6 with
  observed beside predicted; the `c*` fit and residual; the X matrix at final beside the logged one;
  the C result; the decision-rule row each primary landed in, stated in one sentence per claim.

**Rules carried from CLAUDE.md.** Every table shows A, R and N with their declared nulls (#1). Every
arm's null is written before the run (§3). Never select a layer on scoring data; the verdict is read
at the logged layers and the curve is reported (#4). The reviewer reads the diff of `model.py` and the
direction/ranking code, not the report (#5).

## 8. Piece 2 add-ons, in priority order

1. **Mood** (`narrative_mood_v1.json`, L14, last-token pooling as h9) and **theme**
   (`narrative_theme_v1.json`, L20) through the same script, B only.
2. **Relation as a patch** (h25): `dir_T + 0.5·pred` and `dir_T + 0.5·pred_wrong_source` at L16 versus
   their F_par at `pre_28`. If `G_new ≈ 0` for both, the h25 "output direction is useful" paragraph is
   arithmetic. Needs `holonic_v2_rotated` stacks (240 prompts, ~20 min) and `stage4c`'s operator fit.
3. Native and transport sweeps (§4.5).

## 9. What this spec does not cover

The 70B h37 battery and the Gemma/GPT-J factor runs (`ndif_factors.py`) have the same exposure and
the same free arm (pre-norm residual at the last block + norm + head, one cached job per text). They
wait for the local verdict: if the 1.5B lenses are not demonstrated, the remote ones are withdrawn
without a run; if computed, the remote arm belongs to `scale_vs_tuning_v1.md`, not Phase 0.

## 10. Changes from v1

1. `pre_28` is captured by a forward-pre-hook on the final norm everywhere (`hidden_states[28]` is
   post-norm on this environment); the identity test compares against that capture.
2. Primary statistic is `G_new` in nats with sign fraction and 90% lower bound; ranks descriptive.
3. §4.3's decomposition is the verdict machinery: `F_Δ` identity gate, `F_abs` literal term, `F_par`
   null, `A − F_par` new content; `F_best` replaced by a continuous dose family with a fitted `c*`;
   KL-matched `F_KL` added; relative-norm `F_rel` demoted to a robustness line.
4. Burden inverted: "computed" only with lower bound > 0 and point ≥ 2τ; otherwise "not
   demonstrated"; stop fires below threshold. One primary per claim; case-level bootstrap with
   clustering stated.
5. Composed patch, cross-talk at final and `A_span` moved into piece 1; mood and theme listed as
   exposed (piece 2).
6. C-arm note corrected: no scored position is patched, so its final arm is N for every span token.
