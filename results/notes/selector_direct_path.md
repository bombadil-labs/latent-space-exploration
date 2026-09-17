# Phase 0.1 — the direct-path test for the log-prob selectors (piece 1)

Spec: `docs/specs/selector_direct_path_v1.md` (v2). Script: `scripts/selector_direct_path.py`; `pre_28` captured by a forward-pre-hook on `model.model.norm` (never `hidden_states[28]`).

- `results/selector_direct_path_role.json` — role BD layers 20,14 scale 1.0; py 3.11.15, torch 2.14.0+cu130, transformers 5.17.0; tied embeddings **True**; r_L {'20': 83.3, '14': 56.6}, r_28 291.7; 49 min
- `results/selector_direct_path_factors.json` — factors BD layers 14 scale 1.0; py 3.11.15, torch 2.14.0+cu130, transformers 5.17.0; tied embeddings **True**; r_L {'14': 57.9}, r_28 339.8; 115 min
- `results/selector_direct_path_factors_D_gate5.json` — factors P layers 14 scale 1.0; py 3.11.15, torch 2.14.0+cu130, transformers 5.17.0; tied embeddings **True**; r_L {'14': 57.9}, r_28 339.8; 14 min
- `results/selector_direct_path_role_l27.json` — role BD layers 27 scale 1.0; py 3.11.15, torch 2.14.0+cu130, transformers 5.17.0; tied embeddings **True**; r_L {'27': 297.1}, r_28 291.7; 24 min

## Result in one paragraph

**The selector effect is not the direct path.** At every logged layer the literal skip term
(`F_abs`) and the skip term rescaled by everything the stack did along `d` (`F_par`, the spec's
null) move the margin by at most +0.31 nats, while the treatment moves it +0.83 to +3.83.
`G_new = m_A - m_F_par` is +0.62 to +3.61 nats with 90% lower bounds of +0.34 to +3.25 and paired
sign fractions of 0.92-1.00. No dose of the direction at `pre_28`, anywhere on a 0.25x-16x grid,
reproduces the per-case margins (residual RMS at `c*` is 0.59-4.04 nats against a noise floor `tau`
of 0.04-0.26). The mechanism is arithmetic: the injected direction is only 0.7-3.8% of the norm of
the residual entering the final norm (`|d|/r_28`), so the skip path delivers almost nothing, while
the blocks after `L` write orthogonal content of 2.3-5.1x `|d|` in response. **PROGRAM.md 0.1's stop
condition does not fire for either primary.** Writeup claims 2 and 4 describe computation rather
than vocabulary geometry — subject to the three qualifications in *What is still confounded*, of
which the sharpest is that the spec's own registered sanity check, `G_new(27) = 0 +/- tau`, **failed**.

## Doses, and how the run was made

`rho = |s d_L| / r_L` is the treatment's size relative to the residual it is added to; `|d|/r_28` is
its size relative to the residual that actually reaches the unembedding, which is what the direct
path gets to work with.

| claim | \|d\| | r_L | rho | \|d\|/r_28 |
|---|---|---|---|---|
| role L14 | 8.11 | 56.6 | 0.143 | 0.028 |
| role L20 | 11.08 | 83.3 | 0.133 | 0.038 |
| role L27 | 23.23 | 297.1 | 0.078 | 0.080 |
| era L14 | 4.27 | 57.9 | 0.074 | 0.013 |
| voice L14 | 6.06 | 57.9 | 0.105 | 0.018 |
| tense L14 | 2.28 | 57.9 | 0.039 | 0.007 |
| composed D L14 | 7.77 | 57.9 | 0.134 | 0.023 |

Timing, measured before the run as the spec asks: 20 `lm.logprob` calls took 0.457 s each
(Qwen2.5-1.5B, CPU fp32, 4 threads, 32-token text); `pre_norm_residual` 0.477 s; the offline
final-residual readout 0.059 s per candidate at 25 scored positions. Piece 1 ran as four sequential
jobs totalling 3 h 22 min of compute. Two concurrent 1.5B fp32 processes were tried first and the
OOM killer took one: 15 GB does not hold two copies, so everything ran sequentially.

```
PY=.venv/bin/python; ST=results/stacks_qwen2.5_1.5b
HF_HOME=cache/hf HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONPATH=src
$PY scripts/selector_direct_path.py role    prompts/holonic_v1_rotated.json   ${ST}_holonic_v1_rotated.npz   --layers 20,14 --out results/selector_direct_path_role.json
$PY scripts/selector_direct_path.py factors prompts/narrative_factors_v2.json ${ST}_narrative_factors_v2.npz --layers 14 --tests BD --out results/selector_direct_path_factors.json
$PY scripts/selector_direct_path.py factors prompts/narrative_factors_v2.json ${ST}_narrative_factors_v2.npz --layers 14 --tests P  --out results/selector_direct_path_factors_D_gate5.json
$PY scripts/selector_direct_path.py role    prompts/holonic_v1_rotated.json   ${ST}_holonic_v1_rotated.npz   --layers 20 --domains physics,biology --out results/selector_direct_path_role_rerun.json
$PY scripts/selector_direct_path.py role    prompts/holonic_v1_rotated.json   ${ST}_holonic_v1_rotated.npz   --layers 27 --out results/selector_direct_path_role_l27.json
$PY scripts/selector_direct_path_report.py results/selector_direct_path_{role,factors,factors_D_gate5,role_l27}.json --rerun results/selector_direct_path_role_rerun.json --out results/notes/selector_direct_path.md
```

The `--tests P` and `--layers 27` runs are additions to the spec's piece-1 list, both forced by it:
the first supplies test D's gate-5 positive control, which the first pass skipped because §3's row D
lists only the A and F arms; the second is §6's registered `G_new(27)` sanity check, which turned out
to be the most informative single number in this note.

## Gates (spec §5)

- **gate 2 (N)**: max |gain| over every no-patch candidate = 0.0e+00; mean rank reported per claim below. -> PASS
- **gate 3 (F_delta == A)**: max |m_F_delta - m_A| over all cases = 4.58e-05 nats (threshold 1e-4). -> PASS
- **gate 5, C_plumb**: over the plumbing cases, max |gain| in the model = 0.64, 1.88, 0.82, 0.87, 0.73, 0.80, 1.27, 0.93, 4.72, 2.00, 2.24, 0.53, 2.14, 1.99, 2.56, 1.25, 0.27, 0.49, 0.67, 0.26, 0.19, 0.35, 0.10, 0.50, 0.30, 0.55, 0.20, 0.32 nats (needs >> 0.1); the corresponding offline direct arm is identically 0.0 because no scored position is patched. -> PASS
- **gate 6 (determinism)**: 12 cases re-run from scratch; 252 rng-free arm margins compared (A, A_span, C, F_delta, F_abs, F_par, the dose grid, F_KL, N, U, P); max |m_rerun - m_first| = 0.0e+00 nats (threshold 1e-3). -> PASS

## Per-claim tables


### B/era @ L14 — n = 12 cases in 4 clusters

`tau` = **0.1687** nats (random arm: mean m +0.1687, 90% CI of the mean [+0.0510, +0.2914], n=24 draws); the threshold is `2 tau` = 0.3373.

| arm | mean m (nats) | mean rank | declared null / expectation |
|---|---|---|---|
| N | +0.0000 | 2.00 | rank exactly 2.00 = chance, every gain exactly 0.0 |
| A | +2.2451 | 1.25 | the logged instrument (logged rank 1.25) |
| A_span | +2.0334 | 1.29 | ~ A if the lens acts where it is read |
| R0 | +0.2008 | 1.78 | m ~ 0, rank chance 2.00 |
| R1 | +0.1365 | 1.94 | m ~ 0, rank chance 2.00 |
| C | +0.1980 | 1.06 | lead positions only: any effect is computed through attention |
| F_delta | +2.2451 | 1.25 | identical to A (gate 3) |
| F_abs | +0.1852 | 1.47 | the literal skip term, no help from any block |
| F_par | +0.3079 | 1.53 | **THE NULL**: skip term rescaled by the stack |
| F_KL | +0.9844 | 1.47 | dose matched on output perturbation |
| F_R_10 | +0.0369 | 1.92 | m ~ 0 (control on the control, c=1) |
| F_R_11 | +0.0757 | 1.69 | m ~ 0 |
| F_R_kl0 | +0.1128 | 1.93 | m ~ 0 (control on the control, c=c_KL) |
| F_R_kl1 | +0.2227 | 1.82 | m ~ 0 |
| U | +0.0057 | 1.36 | unembedding-only: rank ~ 1 means d points at the target vocabulary |
| P | +1.2092 | 1.15 | positive control (unembedding-built direction) |
| P_abs | +4.3891 | 1.00 | P's literal direct term |
| P_par | +1.4242 | 1.00 | P's null; P - P_par must be <= tau |

- **G_new = m_A - m_F_par** = +1.9372 nats, 90% CI [+1.3871, +2.4832], sign fraction 1.00 (n=12); cluster bootstrap [+1.2893, +2.5934]
- **G_abs = m_A - m_F_abs** = +2.0599 nats, 90% CI [+1.4991, +2.6230], sign fraction 1.00 (n=12); cluster bootstrap [+1.3629, +2.7219]
- **G_KL (dose-free)** = +1.2608 nats, 90% CI [+0.6716, +1.8633], sign fraction 0.75 (n=12); cluster bootstrap [+0.7935, +1.7280]
- **G_rel (v1's relative-norm dose c=5.87, robustness only)** = +1.2089 nats, 90% CI [+0.6204, +1.8132], sign fraction 0.75 (n=12); cluster bootstrap [+0.7385, +1.6793]
- **A - A_span (lead-position contribution)** = +0.2117 nats, 90% CI [+0.1682, +0.2537], sign fraction 1.00 (n=12); cluster bootstrap [+0.1922, +0.2298]
- **m_C (lead positions only: computation, no direct path)** = +0.1980 nats, 90% CI [+0.1596, +0.2342], sign fraction 1.00 (n=12); cluster bootstrap [+0.1754, +0.2198]
- survival = 1.524, orth = 5.072 (both / s|d_L|); mean c_KL = 5.50; KL(A) = 0.0188 nats/position; r_28/r_L = 5.87
- c* fit (§4.3): **c\* = 7.98** x the skip term (1.36 x v1's relative-norm dose), residual RMS 1.6782 nats vs tau 0.1687; 67% of cases inside the dose envelope
- dose curve, mean m_F(c): 0.25: +0.047, 0.5: +0.093, 1.0: +0.185, 2.0: +0.368, 3.5: +0.636, 5.0: +0.895, 8.0: +1.382, 12.0: +1.944, 16.0: +2.383
- **verdict: computed**

### B/tense @ L14 — n = 8 cases in 4 clusters

`tau` = **0.0432** nats (random arm: mean m -0.0001, 90% CI of the mean [-0.0422, +0.0442], n=16 draws); the threshold is `2 tau` = 0.0864.

| arm | mean m (nats) | mean rank | declared null / expectation |
|---|---|---|---|
| N | +0.0000 | 1.50 | rank exactly 1.50 = chance, every gain exactly 0.0 |
| A | +0.8344 | 1.03 | the logged instrument (logged rank 1.03) |
| A_span | +0.7568 | 1.06 | ~ A if the lens acts where it is read |
| R0 | -0.0010 | 1.53 | m ~ 0, rank chance 1.50 |
| R1 | +0.0008 | 1.44 | m ~ 0, rank chance 1.50 |
| C | +0.0669 | 1.00 | lead positions only: any effect is computed through attention |
| F_delta | +0.8344 | 1.03 | identical to A (gate 3) |
| F_abs | +0.1311 | 1.11 | the literal skip term, no help from any block |
| F_par | +0.2164 | 1.12 | **THE NULL**: skip term rescaled by the stack |
| F_KL | +0.6335 | 1.12 | dose matched on output perturbation |
| F_R_10 | -0.0124 | 1.78 | m ~ 0 (control on the control, c=1) |
| F_R_11 | -0.0065 | 1.56 | m ~ 0 |
| F_R_kl0 | -0.0372 | 1.60 | m ~ 0 (control on the control, c=c_KL) |
| F_R_kl1 | +0.0247 | 1.40 | m ~ 0 |
| U | +0.0128 | 1.00 | unembedding-only: rank ~ 1 means d points at the target vocabulary |
| P | +0.3057 | 1.03 | positive control (unembedding-built direction) |
| P_abs | +0.6876 | 1.04 | P's literal direct term |
| P_par | +0.4823 | 1.03 | P's null; P - P_par must be <= tau |

- **G_new = m_A - m_F_par** = +0.6179 nats, 90% CI [+0.3420, +0.9285], sign fraction 1.00 (n=8); cluster bootstrap [+0.3396, +0.8963]
- **G_abs = m_A - m_F_abs** = +0.7033 nats, 90% CI [+0.4046, +1.0357], sign fraction 1.00 (n=8); cluster bootstrap [+0.3752, +1.0314]
- **G_KL (dose-free)** = +0.2009 nats, 90% CI [-0.0101, +0.4332], sign fraction 0.62 (n=8); cluster bootstrap [+0.0334, +0.3683]
- **G_rel (v1's relative-norm dose c=5.87, robustness only)** = +0.0657 nats, 90% CI [-0.2616, +0.4224], sign fraction 0.38 (n=8); cluster bootstrap [-0.2503, +0.3816]
- **A - A_span (lead-position contribution)** = +0.0775 nats, 90% CI [+0.0652, +0.0894], sign fraction 1.00 (n=8); cluster bootstrap [+0.0657, +0.0893]
- **m_C (lead positions only: computation, no direct path)** = +0.0669 nats, 90% CI [+0.0587, +0.0746], sign fraction 1.00 (n=8); cluster bootstrap [+0.0571, +0.0766]
- survival = 0.866, orth = 3.550 (both / s|d_L|); mean c_KL = 4.94; KL(A) = 0.0063 nats/position; r_28/r_L = 5.87
- c* fit (§4.3): **c\* = 5.75** x the skip term (0.98 x v1's relative-norm dose), residual RMS 0.5945 nats vs tau 0.0432; 88% of cases inside the dose envelope
- dose curve, mean m_F(c): 0.25: +0.033, 0.5: +0.066, 1.0: +0.131, 2.0: +0.262, 3.5: +0.459, 5.0: +0.655, 8.0: +1.048, 12.0: +1.569, 16.0: +2.088
- **verdict: computed**

### B/voice @ L14 — n = 12 cases in 4 clusters

`tau` = **0.2606** nats (random arm: mean m +0.0085, 90% CI of the mean [-0.2634, +0.2577], n=24 draws); the threshold is `2 tau` = 0.5212.

| arm | mean m (nats) | mean rank | declared null / expectation |
|---|---|---|---|
| N | +0.0000 | 2.00 | rank exactly 2.00 = chance, every gain exactly 0.0 |
| A | +2.3815 | 1.24 | the logged instrument (logged rank 1.24) |
| A_span | +2.1021 | 1.31 | ~ A if the lens acts where it is read |
| R0 | -0.0327 | 2.07 | m ~ 0, rank chance 2.00 |
| R1 | +0.0498 | 1.97 | m ~ 0, rank chance 2.00 |
| C | +0.2752 | 1.08 | lead positions only: any effect is computed through attention |
| F_delta | +2.3815 | 1.24 | identical to A (gate 3) |
| F_abs | -0.1022 | 2.12 | the literal skip term, no help from any block |
| F_par | -0.0239 | 2.07 | **THE NULL**: skip term rescaled by the stack |
| F_KL | -0.3970 | 2.19 | dose matched on output perturbation |
| F_R_10 | +0.0286 | 2.06 | m ~ 0 (control on the control, c=1) |
| F_R_11 | -0.0819 | 2.12 | m ~ 0 |
| F_R_kl0 | +0.2226 | 1.92 | m ~ 0 (control on the control, c=c_KL) |
| F_R_kl1 | +0.1727 | 1.96 | m ~ 0 |
| U | +0.0055 | 1.31 | unembedding-only: rank ~ 1 means d points at the target vocabulary |
| P | +1.5460 | 1.18 | positive control (unembedding-built direction) |
| P_abs | +6.2452 | 1.00 | P's literal direct term |
| P_par | +1.8840 | 1.00 | P's null; P - P_par must be <= tau |

- **G_new = m_A - m_F_par** = +2.4054 nats, 90% CI [+1.8021, +3.0421], sign fraction 1.00 (n=12); cluster bootstrap [+2.2196, +2.5959]
- **G_abs = m_A - m_F_abs** = +2.4837 nats, 90% CI [+1.8933, +3.0958], sign fraction 1.00 (n=12); cluster bootstrap [+2.2727, +2.7018]
- **G_KL (dose-free)** = +2.7785 nats, 90% CI [+2.1679, +3.3876], sign fraction 1.00 (n=12); cluster bootstrap [+2.6261, +2.9309]
- **G_rel (v1's relative-norm dose c=5.87, robustness only)** = +3.1954 nats, 90% CI [+2.4088, +3.9870], sign fraction 1.00 (n=12); cluster bootstrap [+3.0426, +3.3556]
- **A - A_span (lead-position contribution)** = +0.2794 nats, 90% CI [+0.2136, +0.3443], sign fraction 1.00 (n=12); cluster bootstrap [+0.2521, +0.3066]
- **m_C (lead positions only: computation, no direct path)** = +0.2752 nats, 90% CI [+0.2145, +0.3352], sign fraction 1.00 (n=12); cluster bootstrap [+0.2507, +0.2994]
- survival = 0.941, orth = 4.395 (both / s|d_L|); mean c_KL = 4.49; KL(A) = 0.0326 nats/position; r_28/r_L = 5.87
- c* fit (§4.3): **c\* = 0.25** x the skip term (0.04 x v1's relative-norm dose), residual RMS 2.7824 nats vs tau 0.2606; 33% of cases inside the dose envelope
- dose curve, mean m_F(c): 0.25: -0.024, 0.5: -0.049, 1.0: -0.102, 2.0: -0.220, 3.5: -0.424, 5.0: -0.657, 8.0: -1.199, 12.0: -2.037, 16.0: -2.945
- **verdict: computed**

### D @ L14 — n = 72 cases in 4 clusters

`tau` = **0.2056** nats (random arm: mean m +0.0448, 90% CI of the mean [-0.1609, +0.2504], n=72 draws); the threshold is `2 tau` = 0.4113.

| arm | mean m (nats) | mean rank | declared null / expectation |
|---|---|---|---|
| N | +0.0000 | 9.50 | rank exactly 9.50 = chance, every gain exactly 0.0 |
| A | +3.8279 | 2.81 | the logged instrument (logged rank 2.81) |
| R0 | +0.0448 | 9.40 | m ~ 0, rank chance 9.50 |
| F_delta | +3.8279 | 2.81 | identical to A (gate 3) |
| F_abs | +0.1281 | 7.38 | the literal skip term, no help from any block |
| F_par | +0.2143 | 6.88 | **THE NULL**: skip term rescaled by the stack |
| F_KL | +0.7042 | 7.39 | dose matched on output perturbation |
| F_R_10 | -0.0727 | 10.75 | m ~ 0 (control on the control, c=1) |
| F_R_11 | +0.0290 | 9.04 | m ~ 0 |
| F_R_kl0 | +0.0693 | 8.96 | m ~ 0 (control on the control, c=c_KL) |
| F_R_kl1 | -0.0496 | 9.35 | m ~ 0 |
| U | +0.0074 | 2.83 | unembedding-only: rank ~ 1 means d points at the target vocabulary |
| P | +2.4606 | 2.29 | positive control (unembedding-built direction) |
| P_abs | +10.0767 | 1.07 | P's literal direct term |
| P_par | +3.0429 | 1.19 | P's null; P - P_par must be <= tau |

- **G_new = m_A - m_F_par** = +3.6135 nats, 90% CI [+3.2513, +3.9772], sign fraction 1.00 (n=72); cluster bootstrap [+3.0096, +4.1671]
- **G_abs = m_A - m_F_abs** = +3.6997 nats, 90% CI [+3.3364, +4.0684], sign fraction 1.00 (n=72); cluster bootstrap [+3.0824, +4.2806]
- **G_KL (dose-free)** = +3.1237 nats, 90% CI [+2.6607, +3.6130], sign fraction 0.89 (n=72); cluster bootstrap [+2.5384, +3.7089]
- **G_rel (v1's relative-norm dose c=5.87, robustness only)** = +3.2331 nats, 90% CI [+2.6969, +3.7989], sign fraction 0.89 (n=72); cluster bootstrap [+2.5982, +3.8680]
- survival = 1.099, orth = 4.577 (both / s|d_L|); mean c_KL = 4.85; KL(A) = 0.0576 nats/position; r_28/r_L = 5.87
- c* fit (§4.3): **c\* = 3.09** x the skip term (0.53 x v1's relative-norm dose), residual RMS 4.0422 nats vs tau 0.2056; 53% of cases inside the dose envelope
- dose curve, mean m_F(c): 0.25: +0.033, 0.5: +0.066, 1.0: +0.128, 2.0: +0.244, 3.5: +0.398, 5.0: +0.531, 8.0: +0.752, 12.0: +1.011, 16.0: +1.300
- **verdict: computed**

### role @ L14 — n = 48 cases in 8 clusters

`tau` = **0.2117** nats (random arm: mean m +0.0264, 90% CI of the mean [-0.1876, +0.2357], n=96 draws); the threshold is `2 tau` = 0.4233.

| arm | mean m (nats) | mean rank | declared null / expectation |
|---|---|---|---|
| N | +0.0000 | 3.50 | rank exactly 3.50 = chance, every gain exactly 0.0 |
| A | +2.1236 | 1.58 | the logged instrument (logged rank 1.33) |
| A_span | +1.5218 | 1.71 | ~ A if the lens acts where it is read |
| R0 | +0.2647 | 3.19 | m ~ 0, rank chance 3.50 |
| R1 | -0.2119 | 3.96 | m ~ 0, rank chance 3.50 |
| C | +0.6099 | 1.62 | lead positions only: any effect is computed through attention |
| F_delta | +2.1236 | 1.58 | identical to A (gate 3) |
| F_abs | -0.0192 | 3.50 | the literal skip term, no help from any block |
| F_par | -0.0159 | 3.46 | **THE NULL**: skip term rescaled by the stack |
| F_KL | -0.0219 | 3.60 | dose matched on output perturbation |
| F_R_10 | +0.0173 | 3.46 | m ~ 0 (control on the control, c=1) |
| F_R_11 | -0.0554 | 3.83 | m ~ 0 |
| F_R_kl0 | +0.1365 | 3.38 | m ~ 0 (control on the control, c=c_KL) |
| F_R_kl1 | +0.1057 | 3.35 | m ~ 0 |
| U | +0.0041 | 2.62 | unembedding-only: rank ~ 1 means d points at the target vocabulary |
| P | +3.4923 | 1.06 | positive control (unembedding-built direction) |
| P_abs | +9.9092 | 1.00 | P's literal direct term |
| P_par | +3.0598 | 1.00 | P's null; P - P_par must be <= tau |

- **G_new = m_A - m_F_par** = +2.1396 nats, 90% CI [+1.8287, +2.4483], sign fraction 0.98 (n=48); cluster bootstrap [+1.9268, +2.3189]
- **G_abs = m_A - m_F_abs** = +2.1428 nats, 90% CI [+1.8271, +2.4581], sign fraction 0.96 (n=48); cluster bootstrap [+1.9413, +2.3081]
- **G_KL (dose-free)** = +2.1455 nats, 90% CI [+1.7064, +2.5735], sign fraction 0.83 (n=48); cluster bootstrap [+1.8506, +2.4021]
- **G_rel (v1's relative-norm dose c=5.15, robustness only)** = +2.2521 nats, 90% CI [+1.7047, +2.8027], sign fraction 0.83 (n=48); cluster bootstrap [+1.8808, +2.5995]
- **A - A_span (lead-position contribution)** = +0.6018 nats, 90% CI [+0.4869, +0.7170], sign fraction 0.94 (n=48); cluster bootstrap [+0.4993, +0.6907]
- **m_C (lead positions only: computation, no direct path)** = +0.6099 nats, 90% CI [+0.4835, +0.7408], sign fraction 0.94 (n=48); cluster bootstrap [+0.5023, +0.7088]
- survival = 0.709, orth = 3.204 (both / s|d_L|); mean c_KL = 3.64; KL(A) = 0.0467 nats/position; r_28/r_L = 5.15
- c* fit (§4.3): **c\* = 0.46** x the skip term (0.09 x v1's relative-norm dose), residual RMS 2.5109 nats vs tau 0.2117; 42% of cases inside the dose envelope
- dose curve, mean m_F(c): 0.25: -0.004, 0.5: -0.009, 1.0: -0.019, 2.0: -0.042, 3.5: -0.081, 5.0: -0.124, 8.0: -0.208, 12.0: -0.266, 16.0: -0.190
- **verdict: **no verdict** — gate 5 failed here: a direction built from the target span's own unembedding rows is credited with +0.432 nats of 'new content' (> tau = 0.212), so this configuration cannot distinguish H_direct from H_computed (G_new would have been +2.140, LB +1.829)**

### role @ L20 — n = 48 cases in 8 clusters

`tau` = **0.1306** nats (random arm: mean m +0.0947, 90% CI of the mean [-0.0363, +0.2250], n=96 draws); the threshold is `2 tau` = 0.2613.

| arm | mean m (nats) | mean rank | declared null / expectation |
|---|---|---|---|
| N | +0.0000 | 3.50 | rank exactly 3.50 = chance, every gain exactly 0.0 |
| A | +1.8566 | 1.69 | the logged instrument (logged rank 1.69) |
| A_span | +1.5718 | 1.67 | ~ A if the lens acts where it is read |
| R0 | +0.2670 | 3.10 | m ~ 0, rank chance 3.50 |
| R1 | -0.0775 | 3.56 | m ~ 0, rank chance 3.50 |
| C | +0.3164 | 2.12 | lead positions only: any effect is computed through attention |
| F_delta | +1.8566 | 1.69 | identical to A (gate 3) |
| F_abs | +0.0369 | 3.25 | the literal skip term, no help from any block |
| F_par | +0.0199 | 3.23 | **THE NULL**: skip term rescaled by the stack |
| F_KL | +0.1478 | 3.38 | dose matched on output perturbation |
| F_R_10 | +0.0782 | 3.23 | m ~ 0 (control on the control, c=1) |
| F_R_11 | +0.0246 | 3.25 | m ~ 0 |
| F_R_kl0 | +0.0058 | 3.42 | m ~ 0 (control on the control, c=c_KL) |
| F_R_kl1 | +0.0050 | 3.48 | m ~ 0 |
| U | +0.0069 | 2.29 | unembedding-only: rank ~ 1 means d points at the target vocabulary |
| P | +9.7385 | 1.00 | positive control (unembedding-built direction) |
| P_abs | +13.4447 | 1.00 | P's literal direct term |
| P_par | +10.6933 | 1.00 | P's null; P - P_par must be <= tau |

- **G_new = m_A - m_F_par** = +1.8367 nats, 90% CI [+1.5630, +2.0953], sign fraction 0.92 (n=48); cluster bootstrap [+1.5650, +2.1063]
- **G_abs = m_A - m_F_abs** = +1.8197 nats, 90% CI [+1.5427, +2.0779], sign fraction 0.92 (n=48); cluster bootstrap [+1.5502, +2.0840]
- **G_KL (dose-free)** = +1.7088 nats, 90% CI [+1.3366, +2.0625], sign fraction 0.90 (n=48); cluster bootstrap [+1.3589, +2.0488]
- **G_rel (v1's relative-norm dose c=3.50, robustness only)** = +1.7801 nats, 90% CI [+1.2551, +2.3056], sign fraction 0.81 (n=48); cluster bootstrap [+1.3335, +2.2228]
- **A - A_span (lead-position contribution)** = +0.2848 nats, 90% CI [+0.2121, +0.3574], sign fraction 0.77 (n=48); cluster bootstrap [+0.2145, +0.3432]
- **m_C (lead positions only: computation, no direct path)** = +0.3164 nats, 90% CI [+0.2323, +0.4061], sign fraction 0.83 (n=48); cluster bootstrap [+0.2405, +0.3862]
- survival = 1.006, orth = 2.294 (both / s|d_L|); mean c_KL = 2.24; KL(A) = 0.0389 nats/position; r_28/r_L = 3.50
- c* fit (§4.3): **c\* = 0.88** x the skip term (0.25 x v1's relative-norm dose), residual RMS 2.1357 nats vs tau 0.1306; 48% of cases inside the dose envelope
- dose curve, mean m_F(c): 0.25: +0.010, 0.5: +0.020, 1.0: +0.037, 2.0: +0.061, 3.5: +0.077, 5.0: +0.072, 8.0: +0.045, 12.0: +0.173, 16.0: +0.745
- **verdict: computed**

### role @ L27 — n = 48 cases in 8 clusters

`tau` = **0.1244** nats (random arm: mean m +0.1158, 90% CI of the mean [-0.0070, +0.2417], n=96 draws); the threshold is `2 tau` = 0.2487.

| arm | mean m (nats) | mean rank | declared null / expectation |
|---|---|---|---|
| N | +0.0000 | 3.50 | rank exactly 3.50 = chance, every gain exactly 0.0 |
| A | +0.0135 | 3.46 | the logged instrument (logged rank nan) |
| A_span | -0.0173 | 3.46 | ~ A if the lens acts where it is read |
| R0 | +0.1405 | 3.27 | m ~ 0, rank chance 3.50 |
| R1 | +0.0911 | 3.33 | m ~ 0, rank chance 3.50 |
| C | +0.0339 | 2.62 | lead positions only: any effect is computed through attention |
| F_delta | +0.0135 | 3.46 | identical to A (gate 3) |
| F_abs | -0.4805 | 3.79 | the literal skip term, no help from any block |
| F_par | -0.4396 | 3.81 | **THE NULL**: skip term rescaled by the stack |
| F_KL | -0.3017 | 3.75 | dose matched on output perturbation |
| F_R_10 | +0.1712 | 3.23 | m ~ 0 (control on the control, c=1) |
| F_R_11 | +0.0581 | 3.29 | m ~ 0 |
| F_R_kl0 | -0.0032 | 3.44 | m ~ 0 (control on the control, c=c_KL) |
| F_R_kl1 | +0.0346 | 3.46 | m ~ 0 |
| U | +0.0083 | 2.42 | unembedding-only: rank ~ 1 means d points at the target vocabulary |
| P | +20.2190 | 1.00 | positive control (unembedding-built direction) |
| P_abs | +27.5063 | 1.00 | P's literal direct term |
| P_par | +25.7846 | 1.00 | P's null; P - P_par must be <= tau |

- **G_new = m_A - m_F_par** = +0.4531 nats, 90% CI [+0.2887, +0.6260], sign fraction 0.75 (n=48); cluster bootstrap [+0.3741, +0.5302]
- **G_abs = m_A - m_F_abs** = +0.4940 nats, 90% CI [+0.3149, +0.6845], sign fraction 0.71 (n=48); cluster bootstrap [+0.4246, +0.5625]
- **G_KL (dose-free)** = +0.3152 nats, 90% CI [+0.2057, +0.4257], sign fraction 0.77 (n=48); cluster bootstrap [+0.2479, +0.3842]
- **G_rel (v1's relative-norm dose c=0.98, robustness only)** = +0.4842 nats, 90% CI [+0.3102, +0.6687], sign fraction 0.73 (n=48); cluster bootstrap [+0.4175, +0.5493]
- **A - A_span (lead-position contribution)** = +0.0308 nats, 90% CI [+0.0193, +0.0423], sign fraction 0.77 (n=48); cluster bootstrap [+0.0203, +0.0415]
- **m_C (lead positions only: computation, no direct path)** = +0.0339 nats, 90% CI [+0.0206, +0.0473], sign fraction 0.71 (n=48); cluster bootstrap [+0.0241, +0.0440]
- survival = 1.014, orth = 0.591 (both / s|d_L|); mean c_KL = 0.72; KL(A) = 0.0269 nats/position; r_28/r_L = 0.98
- c* fit (§4.3): **c\* = 0.59** x the skip term (0.60 x v1's relative-norm dose), residual RMS 0.5596 nats vs tau 0.1244; 79% of cases inside the dose envelope
- dose curve, mean m_F(c): 0.25: -0.100, 0.5: -0.214, 1.0: -0.480, 2.0: -1.108, 3.5: -1.987, 5.0: -2.222, 8.0: +0.074, 12.0: +4.638, 16.0: +8.000
- **verdict: no verdict — the treatment itself is at its own null here (m_A = +0.014 < 2 tau = 0.249), so G_new measures only the sign of the direct term, not computation**

### Test X at the final residual (spec §3 row X)

Rows: the patched factor. Columns: fraction of the gain variance over the 18 candidates explained by that factor's main effect. Logged (at L14, arm A): era 0.58 / voice 0.64 / tense 0.47 on-diagonal, random 0.25 / 0.21 / 0.02.

**arm A**

| patched | era | voice | tense |
|---|---|---|---|
| era | 0.58 | 0.14 | 0.00 |
| voice | 0.16 | 0.64 | 0.01 |
| tense | 0.11 | 0.11 | 0.47 |

**arm F_par**

| patched | era | voice | tense |
|---|---|---|---|
| era | 0.40 | 0.23 | 0.01 |
| voice | 0.13 | 0.45 | 0.03 |
| tense | 0.06 | 0.12 | 0.49 |

**arm F_KL**

| patched | era | voice | tense |
|---|---|---|---|
| era | 0.38 | 0.27 | 0.01 |
| voice | 0.13 | 0.53 | 0.01 |
| tense | 0.13 | 0.14 | 0.43 |

**arm R0**

| patched | era | voice | tense |
|---|---|---|---|
| era | 0.27 | 0.23 | 0.03 |
| voice | 0.19 | 0.28 | 0.02 |
| tense | 0.26 | 0.17 | 0.04 |

## Gates 4 and 5, per claim

| claim | tau | F_R at c=1 | F_R at c=c_KL | gate 4 | m_P_abs | m_P - m_P_par | gate 5 |
|---|---|---|---|---|---|---|---|
| B/era @ L14 | 0.169 | +0.056 | +0.168 | PASS | +4.39 | -0.215 | PASS |
| B/tense @ L14 | 0.043 | -0.009 | -0.006 | PASS | +0.69 | -0.177 | PASS |
| B/voice @ L14 | 0.261 | -0.027 | +0.198 | PASS | +6.25 | -0.338 | PASS |
| D @ L14 | 0.206 | -0.022 | +0.010 | PASS | +10.08 | -0.582 | PASS |
| role @ L14 | 0.212 | -0.019 | +0.121 | PASS | +9.91 | +0.432 | FAIL |
| role @ L20 | 0.131 | +0.051 | +0.005 | PASS | +13.44 | -0.955 | PASS |
| role @ L27 | 0.124 | +0.115 | +0.016 | PASS | +27.51 | -5.566 | PASS |

Gate 4 asks that a random vector at the same dose sits at the noise floor; gate 5 asks that a direction built from the target span's unembedding rows reads as large and as **not computed** (`m_P - m_P_par <= tau`). If gate 5 fails, no verdict is issued (spec §5).

## Gate 1 — reproduction of the logged ranks

| claim | logged A | observed A | logged random | observed random | logged no-patch | observed no-patch |
|---|---|---|---|---|---|---|
| B/era @ L14 | 1.25 | 1.25 | 2.00 | 1.86 | 2.00 | 2.00 |
| B/tense @ L14 | 1.03 | 1.03 | 1.44 | 1.49 | 1.50 | 1.50 |
| B/voice @ L14 | 1.24 | 1.24 | 2.25 | 2.02 | 2.00 | 2.00 |
| D @ L14 | 2.81 | 2.81 | — | 9.40 | 9.50 | 9.50 |
| role @ L14 | 1.33 | 1.58 | — | 3.57 | — | 3.50 |
| role @ L20 | 1.69 | 1.69 | 3.25 | 3.33 | — | 3.50 |
| role @ L27 | — | 3.46 | — | 3.30 | — | 3.50 |

The logged role number at L14 (1.33) was measured on four domains only (biology, law, music, software); restricted to those four this run reads **1.33** (n=24). The 1.58 above is all eight domains. Every logged rank is therefore reproduced exactly under mid-rank.

## Verdicts

| claim | m_A | m_F_abs | m_F_par | G_new | 90% LB | sign | 2 tau | c* | verdict |
|---|---|---|---|---|---|---|---|---|---|
| B/era @ L14 | +2.245 | +0.185 | +0.308 | +1.937 | +1.387 | 1.00 | 0.337 | 7.98 | computed |
| B/tense @ L14 | +0.834 | +0.131 | +0.216 | +0.618 | +0.342 | 1.00 | 0.086 | 5.75 | computed |
| B/voice @ L14 | +2.382 | -0.102 | -0.024 | +2.405 | +1.802 | 1.00 | 0.521 | 0.25 | computed |
| D @ L14 | +3.828 | +0.128 | +0.214 | +3.614 | +3.251 | 1.00 | 0.411 | 3.09 | computed |
| role @ L14 | +2.124 | -0.019 | -0.016 | +2.140 | +1.829 | 0.98 | 0.423 | 0.46 | **no verdict** — gate 5 failed here: a direction built from the target span's own unembedding rows is credited with +0.432 nats of 'new content' |
| role @ L20 | +1.857 | +0.037 | +0.020 | +1.837 | +1.563 | 0.92 | 0.261 | 0.88 | computed |
| role @ L27 | +0.014 | -0.480 | -0.440 | +0.453 | +0.289 | 0.75 | 0.249 | 0.59 | no verdict — the treatment itself is at its own null here |

**PROGRAM.md 0.1 stop condition:** primaries are role @ L20 and composed D @ L14; the stop fires if either lands below 2 tau. Fired for: neither.

## Registered predictions, graded (spec §6)

Predictions were registered before any number existed. `m_A` was itself a guess; the planner's last
column was their probability that the lens is *substantially direct*.

| claim | m_A pred -> obs | m_F_abs pred -> obs | m_F_par pred -> obs | G_new pred -> obs (90% LB) | c* pred -> obs | belief direct | graded |
|---|---|---|---|---|---|---|---|
| role L20 | 1.9 -> **1.86** | 0.4 -> 0.04 | 0.8 -> 0.02 | +1.0, LB>0 -> **+1.84** (+1.56) | 3-5 -> 0.88, family does not fit | 0.30 | direction right, direct path far weaker than predicted |
| role L14 | 2.0 -> 2.12 | 0.3 -> -0.02 | 0.8 -> -0.02 | +1.1 -> +2.14 (+1.83) | 4-6 -> 0.46 | 0.30 | **no verdict** (gate 5 failed at this layer) |
| era L14 | 0.8 -> 2.25 | 0.3 -> 0.19 | 0.5 -> 0.31 | +0.3, LB~0 -> **+1.94** (+1.39) | 2-3 -> 7.98 | 0.45 | wrong: far more computed than predicted |
| voice L14 | 0.9 -> 2.38 | 0.5 -> -0.10 | 0.8 -> -0.02 | +0.1, LB<0 -> **+2.41** (+1.80) | 1.5-2 -> 0.25 | 0.65 | **wrong**: predicted lexical, reads computed |
| tense L14 | 0.6 -> 0.83 | 0.4 -> 0.13 | 0.6 -> 0.22 | 0.0, LB<0 -> **+0.62** (+0.34) | ~1 -> 5.75 | 0.80 | **wrong**: predicted pure vocabulary, reads computed |
| composed D L14 | 2.0 -> 3.83 | 0.9 -> 0.13 | 1.6 -> 0.21 | +0.4, LB~0 -> **+3.61** (+3.25) | 2 -> 3.09 | 0.50 | wrong in the same direction |
| X at final | — | — | on-diagonal 0.4-0.6 -> **0.40 / 0.45 / 0.49** | — | — | 0.6 | **correct**: the direct term alone reproduces most of the logged diagonal (A 0.58/0.64/0.47, random 0.27/0.28/0.04) |
| P, any L | — | large -> +0.69 to +27.5 | ~ same -> yes | <= tau -> yes at every layer except role L14 (+0.43) | ~1 | by construction | held, with one exception that cost role L14 its verdict |
| C, role L20 | 0.2-0.5 -> **+0.32** (LB +0.23 > 0) | == 0 | == 0 | = m_C | — | unambiguous computation if LB>0 | **correct**: lead-position-only patching, with no scored position touched, still selects |
| G_new(27) | — | — | — | 0 +/- tau -> **+0.45 (LB +0.29), 2 tau = 0.25** | — | sanity | **FAILED** — see below |

Six of the seven substantive predictions under-estimated how computed these lenses are. The two
predictions the planner was most confident about (tense 0.80, voice 0.65 probability of being
substantially direct) are the two that were most wrong. The two structural predictions — the
cross-talk diagonal being mostly vocabulary geometry, and the C arm showing unambiguous computation
— both held.

## What is still confounded

1. **The registered sanity check failed, and it bounds how small a `G_new` may be believed.**
   `G_new(27)` should have been `0 +/- tau`; it is **+0.453, LB +0.289, against 2 tau = 0.249**.
   The diagnosis is visible in the L27 table: the treatment at L27 does nothing at all
   (`m_A = +0.014`, rank 3.46 against chance 3.50 — the role lens is gone by then, which is the
   right shape for h5's fade at L26), while the null arm is strongly *negative*
   (`m_F_par = -0.440`). `G_new` is a difference, so it is large whenever the direct term **hurts**
   the target, with no computation involved. At L27 the skip term is 8.0% of `r_28` against 2.8-3.8%
   at L14/L20, which is the likely reason it is harmful there and neutral here. The practical
   consequence: `G_new` is not an unbiased estimate of computed content; its zero point is only as
   good as the direct term's neutrality, which must be measured per layer. It does **not** overturn
   the verdicts, because at the logged layers the entire direct family sits in [-0.10, +0.31] nats
   while `G_new` is +0.62 to +3.61 — but **no `G_new` below roughly 0.5 nats should be believed from
   this instrument until the offset is measured at that layer.**
2. **Gate 5 failed for the role lens at L14**, where a direction built from the target span's own
   unembedding rows is credited with +0.432 nats of "new content" (`tau` = 0.212). Per spec §5 no
   verdict is issued for role @ L14. Why L14 and not L20 (-0.955) or L27 (-5.566) is not explained
   here. Claim 2's primary is L20, which passes, so the claim itself is unaffected.
3. **The random arm is off its null for B/era** (`m_R` = +0.20 and +0.14, rank 1.78/1.94 against
   chance 2.00) and mildly for role L20 (`R0` +0.27) and L27 (+0.12). `tau` absorbs this by
   construction — that is what `tau` is for — but a random direction that systematically helps the
   target span means the margin carries a direction-independent component, probably a norm or
   entropy effect at the scored positions. Not diagnosed.
4. **"Computed" is a narrow claim.** It means: the blocks after `L` write content orthogonal to `d`
   that the unembedding reads in the target's favour, and an equal-norm random direction at the same
   layer does not produce it. It does not establish that the content is *about* the role or the
   factor, nor exclude a generic "sharpen whatever the patched direction favours" mechanism. The
   only specificity evidence here is the X matrix, and that matrix shows the **direct** term already
   reproducing most of the logged diagonal (0.40/0.45/0.49 against A's 0.58/0.64/0.47) — so the
   writeup's "diagonal cross-talk" paragraph remains substantially vocabulary geometry even though
   the lenses themselves are not.
5. **Dose, unevenly.** For era and tense the *mean* margin is reachable by the direct family at
   `c* ~ 6-8` (six to eight times the norm the skip connection actually delivers); only the per-case
   pattern resists (RMS 1.68 and 0.59 against `tau` 0.17 and 0.04). For role at L14/L20, voice and D
   no dose on the grid comes close — those dose curves are flat or non-monotone. The "the direct
   path cannot do this" conclusion is therefore much stronger for role, voice and D than for era and
   tense.
6. **`F_R` at the KL-matched dose** sits at +0.17 to +0.22 for era and voice: within `tau`, so gate 4
   passes, but not at zero. `F_KL` is reported; `F_par` is the null the verdict uses.
7. **Bookkeeping.** The base pass and the no-patch arm are computed once per candidate set and reused
   across the cases sharing it (both are deterministic and were verified bit-identical on re-run), so
   "N reads exactly chance with every gain exactly 0.0" is one measurement per candidate set rather
   than one per case. One scale (`s = 1`), one model, one prompt family per claim. The lexical-floor
   problem of INSTRUMENTS §5 is untouched by this test: reproducing the logged instrument exactly
   says nothing about whether the stimulus is confounded.

## What piece 2 should do

1. **Fix the statistic before extending it.** The L27 result shows the `G_new` offset is
   layer-dependent and can exceed `2 tau`. Piece 2 should measure the offset at every swept layer
   with the `P` arm, which is cheap, and report either `G_new` minus the P gap or the P gap as the
   per-layer floor in place of `tau`. Every sweep number below ~0.5 nats depends on this.
2. **Diagnose the role lens's gate-5 failure at L14.** The transport sweep of §4.5 (fixed `d_20` at
   every `L`, absolute norm held) is the instrument that separates "the direction estimated late is
   worse" from "less computation follows", and it is the same machinery that explains the L14/L20
   asymmetry in the P gap.
3. **Native and transport sweeps** (§4.5), with the corrected floor. The `L = 27` point is already
   measured and is worth reconciling with h5: the lens is completely gone at L27 (`m_A = +0.014`,
   rank 3.46) where h5 logged 2.67 at L26.
4. **Mood** (`narrative_mood_v1.json`, L14, last-token) and **theme** (`narrative_theme_v1.json`,
   L20) through the same script, test B only — both are exposed and neither has been tested.
5. **Relation as a patch** (h25): `dir_T + 0.5 pred` at L16 against its `F_par`, per §8.2.
6. **The remote batteries are unblocked, not withdrawn.** Spec §9 made the 70B h37 battery and the
   Gemma/GPT-J runs conditional on this verdict: since the 1.5B lenses read as computed, they are not
   withdrawn without a run, and the remote arm belongs to `scale_vs_tuning_v1.md` rather than Phase 0.
   The same free arm applies there — pre-norm residual at the last block, plus norm and head, one
   cached job per text.
7. **Diagnose the B/era random arm's positive margin** (item 3 above) before any era number is
   quoted with a tight interval.
