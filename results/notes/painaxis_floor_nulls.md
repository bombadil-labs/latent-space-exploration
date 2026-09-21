# Pain axis: the true embedding floor, and null arms at every layer

**Verdict up front.** Tier A's headline survives as a *signal* and does not survive as a *finding
about computation*. Both nulls sit exactly on 0.5 at every layer, so the AUCs are real — but the
static-embedding floor that nobody (them or us) had ever measured comes in at **0.872** for `mean`
on S2_1P, against a Tier A `mean` peak of **0.918**. The whole 28-block network buys **+0.046 AUC
over a bag of token embeddings**. For S2_3P it buys **+0.025**. The `final_token` extraction is
worse than the lexical floor for the first thirteen blocks and beats it by **+0.072** at its best.

Model `Qwen/Qwen2.5-1.5B-Instruct`, local CPU fp32, `.venv` py3.11, sklearn 1.9.1 (their calls).
Stimuli: the vendored `prompts/external/pain_axis/3.1_pain_and_control_datasets.json`, sets
`S1_1P S1_3P S2_1P S2_3P` (800 sentences). Code: `scripts/painaxis_floor_nulls.py`, which imports
every piece of arithmetic from `scripts/painaxis_analyze.py` and reimplements none of it. Tests:
`tests/test_painaxis_floor_nulls.py` (8). Outputs: `results/painaxis_floor_nulls/`
(`floor_null_curves.csv`, `floor_null_summary.json`, the eight `nulls_*.json` draw dumps, the
shared `embed_bag.npz`, and four gitignored `acts_*.npz`, 260 MB).

## 0. The degeneracy check, first, because everything else depends on it

The brief's instruction: at the embedding layer the `final_token` readout must be exactly 0.5,
because every prompt in every set ends with the identical token. It is.

| set | distinct last token ids | max abs deviation of the final-token embedding from row 0 | emb final_token AUC |
|---|---|---|---|
| S1_1P | 1 (id 25, `':'`) | **0.0** | **0.5000** |
| S1_3P | 1 (id 25, `':'`) | **0.0** | **0.5000** |
| S2_1P | 1 (id 25, `':'`) | **0.0** | **0.5000** |
| S2_3P | 1 (id 25, `':'`) | **0.0** | **0.5000** |

Not approximately constant — bit-identical, because `hidden_states[0]` is a pure embedding-table
lookup. Both null arms also return exactly 0.5000 there, on all four sets, as they must when every
row of the matrix is the same vector. Note the 3P sets share the degeneracy: the third-person
framing lives in the stimulus clause, the readout prompt `I feel:` is constant in all four sets.

The 0.5 is therefore a *property of the stimulus design*, not a measurement of the model, and it is
the reason the meaningful floor is the `mean` (bag-of-embeddings) one.

## 1. THE TABLE — at their chosen layer

Chosen layers are their own procedure, recomputed here: argmax of the held-out curve averaged over
S2_1P and S2_3P, over the block layers. `final_token` → 26, `mean` → 21, identical to Tier A.
Nulls are 500 draws each; the bracket is the 2.5–97.5 percentile **over draws**.
Floor = mean-pooled static embeddings (`hidden_states[0]`), same held-out K-fold, same AUC path.

| extraction | dataset | layer | treatment AUC | embedding floor | random-direction null [95%] | shuffled-label null [95%] | **gain over floor** |
|---|---|---|---|---|---|---|---|
| final_token | S2_1P | 26 | 0.944 | 0.872 | 0.498 [0.340, 0.642] | 0.498 [0.379, 0.599] | **+0.072** |
| final_token | S2_3P | 26 | 0.927 | 0.882 | 0.499 [0.380, 0.617] | 0.501 [0.370, 0.603] | **+0.045** |
| final_token | S1_1P | 26 | 0.864 | 0.775 | 0.501 [0.376, 0.616] | 0.499 [0.395, 0.601] | **+0.088** |
| final_token | S1_3P | 26 | 0.870 | 0.825 | 0.502 [0.395, 0.610] | 0.501 [0.402, 0.601] | **+0.045** |
| mean | S2_1P | 21 | 0.918 | 0.872 | 0.497 [0.406, 0.597] | 0.497 [0.374, 0.607] | **+0.046** |
| mean | S2_3P | 21 | 0.906 | 0.882 | 0.498 [0.405, 0.597] | 0.499 [0.382, 0.609] | **+0.025** |
| mean | S1_1P | 21 | 0.825 | 0.775 | 0.501 [0.426, 0.575] | 0.499 [0.387, 0.600] | **+0.050** |
| mean | S1_3P | 21 | 0.856 | 0.825 | 0.499 [0.409, 0.591] | 0.499 [0.387, 0.601] | **+0.031** |

The embedding floor is **0.775–0.882** depending on the sentence grid. The paper-style `mean`
headline is 0.906–0.918 on S2. Most of that number is vocabulary.

## 2. THE CURVE — every layer (`emb` = `hidden_states[0]`, 0–27 = residual after block i)

Nulls pooled over the four sets (mean of the four cells' draw-means; the bracket is the widest
single-cell 2.5–97.5 band, i.e. the most pessimistic). Gain column is S2_1P, the headline grid.

### `mean` (their extraction; the floor is directly comparable)

| layer | S2_1P | S2_3P | S1_1P | S1_3P | random null [95%] | shuffled null [95%] | gain over floor (S2_1P) |
|---|---|---|---|---|---|---|---|
| **emb** | **0.872** | **0.882** | **0.775** | **0.825** | 0.499 [0.419, 0.592] | 0.502 [0.385, 0.616] | — (floor) |
| 0 | 0.886 | 0.867 | 0.776 | 0.839 | 0.501 [0.395, 0.599] | 0.501 [0.386, 0.615] | +0.015 |
| 1 | 0.866 | 0.865 | 0.768 | 0.823 | 0.500 [0.405, 0.603] | 0.501 [0.385, 0.609] | −0.006 |
| 2 | 0.862 | 0.864 | 0.776 | 0.832 | 0.501 [0.407, 0.586] | 0.502 [0.384, 0.610] | −0.009 |
| 3 | 0.881 | 0.880 | 0.784 | 0.828 | 0.501 [0.404, 0.602] | 0.501 [0.375, 0.608] | +0.010 |
| 4 | 0.885 | 0.887 | 0.789 | 0.836 | 0.500 [0.406, 0.599] | 0.501 [0.381, 0.608] | +0.014 |
| 5 | 0.887 | 0.893 | 0.784 | 0.839 | 0.498 [0.407, 0.593] | 0.502 [0.382, 0.604] | +0.016 |
| 6 | 0.882 | 0.886 | 0.795 | 0.836 | 0.499 [0.409, 0.594] | 0.502 [0.384, 0.609] | +0.011 |
| 7 | 0.878 | 0.888 | 0.797 | 0.835 | 0.501 [0.407, 0.598] | 0.501 [0.383, 0.611] | +0.006 |
| 8 | 0.883 | 0.889 | 0.804 | 0.834 | 0.500 [0.403, 0.597] | 0.501 [0.378, 0.611] | +0.011 |
| 9 | 0.879 | 0.888 | 0.801 | 0.835 | 0.499 [0.402, 0.594] | 0.501 [0.378, 0.607] | +0.007 |
| 10 | 0.894 | 0.900 | 0.804 | 0.834 | 0.500 [0.405, 0.594] | 0.500 [0.377, 0.611] | +0.023 |
| 11 | 0.893 | 0.903 | 0.806 | 0.838 | 0.499 [0.409, 0.593] | 0.500 [0.374, 0.611] | +0.021 |
| 12 | 0.888 | 0.899 | 0.803 | 0.834 | 0.498 [0.401, 0.597] | 0.500 [0.375, 0.613] | +0.017 |
| 13 | 0.883 | 0.901 | 0.806 | 0.832 | 0.500 [0.405, 0.597] | 0.500 [0.375, 0.607] | +0.011 |
| 14 | 0.887 | 0.897 | 0.810 | 0.834 | 0.500 [0.399, 0.601] | 0.499 [0.370, 0.613] | +0.015 |
| 15 | 0.891 | 0.898 | 0.807 | 0.837 | 0.500 [0.405, 0.604] | 0.499 [0.374, 0.609] | +0.020 |
| 16 | 0.891 | 0.898 | 0.813 | 0.837 | 0.501 [0.405, 0.596] | 0.499 [0.372, 0.611] | +0.020 |
| 17 | 0.884 | 0.894 | 0.804 | 0.832 | 0.499 [0.402, 0.594] | 0.499 [0.374, 0.615] | +0.013 |
| 18 | 0.882 | 0.885 | 0.807 | 0.836 | 0.499 [0.395, 0.594] | 0.499 [0.368, 0.617] | +0.011 |
| 19 | 0.887 | 0.893 | 0.806 | 0.828 | 0.500 [0.405, 0.598] | 0.499 [0.374, 0.612] | +0.015 |
| 20 | 0.906 | 0.901 | 0.818 | 0.839 | 0.497 [0.398, 0.602] | 0.499 [0.371, 0.614] | +0.035 |
| **21** | **0.918** | 0.906 | 0.825 | 0.856 | 0.499 [0.405, 0.597] | 0.499 [0.374, 0.609] | **+0.046** |
| 22 | 0.907 | 0.900 | 0.820 | 0.851 | 0.499 [0.405, 0.599] | 0.498 [0.370, 0.611] | +0.035 |
| 23 | 0.911 | 0.904 | 0.813 | 0.851 | 0.500 [0.399, 0.601] | 0.498 [0.369, 0.605] | +0.040 |
| 24 | 0.905 | 0.901 | 0.810 | 0.842 | 0.500 [0.399, 0.602] | 0.498 [0.367, 0.606] | +0.034 |
| 25 | 0.903 | 0.899 | 0.814 | 0.849 | 0.501 [0.401, 0.608] | 0.498 [0.372, 0.607] | +0.032 |
| 26 | 0.912 | 0.897 | 0.887 | 0.898 | 0.501 [0.381, 0.607] | 0.500 [0.383, 0.606] | +0.041 |
| 27 | 0.897 | 0.893 | 0.842 | 0.886 | 0.500 [0.391, 0.610] | 0.500 [0.387, 0.599] | +0.026 |

The `mean` curve is **flat above its floor for twenty layers**. On S2_1P, blocks 1 and 2 are
*below* the bag of embeddings and blocks 3–19 sit within +0.023 of it. The only real structure is
a late rise at blocks 20–26 worth about +0.03.

### `final_token`

| layer | S2_1P | S2_3P | S1_1P | S1_3P | random null [95%] | shuffled null [95%] | gain over floor (S2_1P) |
|---|---|---|---|---|---|---|---|
| **emb** | **0.500** | **0.500** | **0.500** | **0.500** | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | degenerate |
| 0 | 0.795 | 0.800 | 0.731 | 0.778 | 0.499 [0.391, 0.610] | 0.499 [0.382, 0.607] | −0.076 |
| 1 | 0.799 | 0.804 | 0.778 | 0.791 | 0.499 [0.397, 0.608] | 0.499 [0.379, 0.602] | −0.072 |
| 2 | 0.794 | 0.837 | 0.779 | 0.745 | 0.498 [0.397, 0.601] | 0.501 [0.381, 0.616] | −0.077 |
| 3 | 0.809 | 0.829 | 0.799 | 0.790 | 0.500 [0.389, 0.606] | 0.498 [0.371, 0.607] | −0.062 |
| 4 | 0.807 | 0.829 | 0.798 | 0.753 | 0.499 [0.394, 0.613] | 0.498 [0.382, 0.600] | −0.065 |
| 5 | 0.822 | 0.826 | 0.791 | 0.758 | 0.498 [0.397, 0.596] | 0.499 [0.377, 0.615] | −0.050 |
| 6 | 0.812 | 0.785 | 0.770 | 0.791 | 0.501 [0.402, 0.606] | 0.501 [0.374, 0.610] | −0.059 |
| 7 | 0.848 | 0.814 | 0.791 | 0.808 | 0.502 [0.398, 0.609] | 0.502 [0.385, 0.607] | −0.024 |
| 8 | 0.841 | 0.825 | 0.776 | 0.839 | 0.496 [0.379, 0.606] | 0.501 [0.387, 0.610] | −0.030 |
| 9 | 0.835 | 0.850 | 0.785 | 0.829 | 0.500 [0.383, 0.604] | 0.501 [0.387, 0.603] | −0.037 |
| 10 | 0.864 | 0.877 | 0.820 | 0.836 | 0.502 [0.384, 0.614] | 0.501 [0.382, 0.611] | −0.008 |
| 11 | 0.864 | 0.870 | 0.809 | 0.856 | 0.500 [0.390, 0.615] | 0.500 [0.382, 0.605] | −0.007 |
| 12 | 0.863 | 0.872 | 0.793 | 0.852 | 0.499 [0.395, 0.605] | 0.500 [0.385, 0.604] | −0.009 |
| 13 | 0.882 | 0.868 | 0.814 | 0.892 | 0.500 [0.393, 0.607] | 0.501 [0.379, 0.610] | +0.011 |
| 14 | 0.874 | 0.871 | 0.825 | 0.874 | 0.502 [0.389, 0.620] | 0.501 [0.375, 0.608] | +0.002 |
| 15 | 0.904 | 0.870 | 0.851 | 0.902 | 0.499 [0.391, 0.612] | 0.501 [0.376, 0.612] | +0.033 |
| 16 | 0.906 | 0.889 | 0.849 | 0.891 | 0.501 [0.393, 0.623] | 0.501 [0.382, 0.610] | +0.035 |
| 17 | 0.911 | 0.873 | 0.842 | 0.880 | 0.500 [0.385, 0.612] | 0.501 [0.383, 0.608] | +0.040 |
| 18 | 0.898 | 0.866 | 0.843 | 0.868 | 0.499 [0.389, 0.611] | 0.501 [0.393, 0.608] | +0.026 |
| 19 | 0.893 | 0.869 | 0.840 | 0.856 | 0.499 [0.381, 0.625] | 0.501 [0.387, 0.609] | +0.021 |
| 20 | 0.914 | 0.880 | 0.862 | 0.874 | 0.499 [0.381, 0.624] | 0.502 [0.391, 0.608] | +0.043 |
| 21 | 0.912 | 0.897 | 0.858 | 0.866 | 0.500 [0.377, 0.636] | 0.501 [0.384, 0.609] | +0.041 |
| 22 | 0.916 | 0.898 | 0.857 | 0.858 | 0.500 [0.370, 0.630] | 0.501 [0.389, 0.606] | +0.044 |
| 23 | 0.920 | 0.898 | 0.860 | 0.870 | 0.499 [0.369, 0.636] | 0.501 [0.381, 0.606] | +0.049 |
| 24 | 0.934 | 0.916 | 0.863 | 0.889 | 0.500 [0.361, 0.638] | 0.501 [0.371, 0.605] | +0.063 |
| 25 | 0.942 | 0.917 | 0.868 | 0.891 | 0.506 [0.371, 0.639] | 0.500 [0.374, 0.600] | +0.070 |
| **26** | **0.944** | 0.927 | 0.864 | 0.870 | 0.500 [0.340, 0.642] | 0.500 [0.370, 0.603] | **+0.072** |
| 27 | 0.919 | 0.921 | 0.875 | 0.887 | 0.504 [0.361, 0.642] | 0.499 [0.375, 0.610] | +0.048 |

**43 of the 112 `final_token` (layer × set) cells sit BELOW the static-embedding bag**, including
every layer up to 12 on S2_1P. Tier A read the `final_token` curve as "the one that shows the model
building something" because it starts at 0.795 and adds 0.15. Against the floor that is not the
right reading: for half the network the final-token residual is a *worse* linear pain readout than
counting which embeddings are in the sentence. It overtakes the floor only at block 13, and its
whole lead is the last quarter of the network. The qualitative story ("the late layers build
something") survives; "the model builds it from 0.795" does not, because 0.795 is below 0.872.

## 3. Both nulls, as arms (non-negotiable 1)

Declared null for both: 0.5. Over all 8 cells × 29 layers × 500 draws:

| arm | min cell mean | max cell mean | mean 95% band width | worst single-draw 97.5th pct |
|---|---|---|---|---|
| random direction (redrawn per fold, matching the refit) | 0.4930 | 0.5079 | 0.193 | 0.642 |
| random direction (one direction per draw, all folds) | 0.4878 | 0.5094 | 0.298 | 0.759 |
| shuffled label (full pipeline, denoising included) | 0.4957 | 0.5049 | 0.213 | 0.617 |

Every arm sits on its declared null at every layer: across all 232 cells the arm means span
0.488–0.509, i.e. within ±0.012 of 0.5, with no layer-wise trend. Neither arm comes back
systematically below 0.5 (the leak signature the brief warned about) nor anywhere near the
treatment. **The Tier A numbers now have their arms and are a result in this project's sense.**

The second row of that table is the number worth keeping. **A single draw of either null can
return 0.62, and a single fixed random direction can return 0.76.** h47's lesson, quantified for
this grid: with 20 sentence sets and 5 folds, any AUC below ~0.64 from this pipeline is
indistinguishable from nothing at the single-draw level. Every treatment number here clears that,
but three Tier A cells (`final_token` S1_1P layer 0 = 0.731, S1_3P layers 2 and 4 = 0.745/0.753)
clear it by less than 0.1 — and they are all *below the embedding floor* anyway.

## 4. What I verified vs what I assumed

**Verified:**
1. **The embedding capture is `hidden_states[0]`**, bit-exactly (max abs deviation 0.0 on spot-
   checked prompts), and `hidden_states[0]` equals a pure `embed_tokens` lookup of the input ids,
   bit-exactly — so Qwen adds no positional term there and this really is a position-free bag of
   static token vectors, the object `docs/specs/conscription_instrument_v1.md` §3 asks for.
   My block-0 capture equals `hidden_states[1]`, bit-exactly, re-confirming Tier A's indexing.
2. **This re-extraction reproduces Tier A to machine precision.** 224 cells
   (2 extractions × 4 sets × 28 layers) compared against `results/painaxis_tierA/layer_curves.csv`:
   **max |diff| = 2.2e-16**. Two independent extraction runs, same numbers.
3. **My fold loop is their fold loop.** `curve_all_controls` equals
   `painaxis_analyze.kfold_curve`'s `auc_vs_all_controls` to 1e-12 (asserted in the run, and
   tested on synthetic data); it only skips the unused `auc_vs_neutral` arm.
4. **The identity permutation reproduces the treatment curve exactly** — so the shuffle machinery
   itself adds nothing; the null comes from the permutation and nothing else.
5. **The shuffle does not leak.** Tested: it preserves category marginals globally and per set;
   it preserves 5-pain/5-control in every set and hence in every fold; agreement with the true
   labels sits at chance (measured over the real run: mean 0.0996, chance 0.10, max 0.175); it
   does not mutate its input. Folds are built from `sets`, which carries no label information
   because every one of the 20 sets holds exactly one sentence per category.
6. **Both nulls return 0.5 on data with a real signal.** On synthetic activations whose treatment
   AUC is >0.95, the shuffled arm returns 0.500 ± 0.05 and the random arm 0.500 ± 0.05.
   A denoising fitted on the true controls, or folds built from the unshuffled assignment, would
   show up here; they do not.
7. **Constant activations give AUC exactly 0.5**, not NaN and not a near-miss, through their path.
   (sklearn's PCA does emit a `RuntimeWarning: invalid value encountered in divide` on the
   zero-variance control block — the degenerate `final_token` embedding cell. The warning is real
   and harmless: `vec` is identically zero, so the projection is zero and the AUC is all-ties.)
8. 223 passed, 3 skipped (`pytest -q tests/`), up from 215 + 3 before this work.

**Assumed, not verified:**
- **The BOS token, inherited from Tier A.** I prepend `tok.eos_token_id` = 151645 = `<|im_end|>`,
  the same id Tier A used, so the reproduction above is exact. Note that
  `scripts/painaxis_extract.py`'s docstring says TransformerLens aliases BOS to `<|endoftext|>`;
  the id actually used is `<|im_end|>`. The *code* is consistent between the two runs and the
  *comment* is wrong, but neither run checked what TransformerLens really does — no network.
  This affects `mean` (a prefix token is 1 of ~10 positions) and not `final_token`.
  It does **not** affect the embedding floor's status as a floor, but it does mean the floor's
  exact value includes one `<|im_end|>` embedding in every bag.
- **That the floor is the right floor.** A mean-pooled bag of static embeddings is a *lexical*
  representation of the same text, which is the h47 rule. It is not the only one; a word-count
  floor (`types._bag`, the sidecar's featurization) would be a second, and
  `conscription_instrument_v1.md` U3 records that the two can disagree in sign. I measured one.
- That CPU fp32 here matches their GPU bf16 + TransformerLens path. Unchanged from Tier A.

## 5. What failed, and what I got wrong

- **I nearly sized this run off an instrument measuring itself.** My first timing probe said one
  29-layer curve took 284 s, which would have capped the shuffled-label arm at ~50 draws. The probe
  was running while the 800-sentence extraction still had a set to go, on a 4-core box; re-timed
  single-threaded the same curve takes **3.3 s**, a factor of 86. I had been about to report a null
  banded on 50 draws because a stopwatch was measuring CPU contention rather than the algorithm —
  which is the `docs/INSTRUMENTS.md` failure mode one level down, in the scheduling rather than the
  statistics. Re-timed, then ran 500 draws of each arm on all eight cells.
- **My first null design would have under-stated its own band.** I drew a fresh random direction
  per fold, mirroring the per-fold refit, and averaged 5 of them per draw — which narrows the band
  by ~√5 relative to what a person means by "one random direction". Caught it while writing the
  loop and now report both: per-fold-redrawn (band width 0.193) and one-direction-per-draw (0.298).
  The honest single-draw noise floor is the second, and it reaches 0.76.
- **I expected the embedding floor to embarrass `mean` and leave `final_token` alone.** It did the
  first. It also did something I did not predict: `final_token` is *below* the lexical floor for
  the first 13 blocks, so Tier A's reading of the `final_token` curve as the one showing genuine
  computation is only true of its top quarter.
- **`sleep` does not advance wall-clock reliably in this sandbox**, so three of my early progress
  polls returned stale logs and I twice thought a healthy job had stalled. Cost: no damage, some
  wasted polling. I did not `pgrep -f`/`pkill -f` anything (non-negotiable 6); every watcher greps
  a log file for `EXTRACTION DONE|Traceback|Error|Killed`, never a process table.

## 6. What I did NOT do

- **No band on the gain.** `gain_over_floor` is a difference of two AUCs measured on the same
  sentences; its null needs a paired resample over sentence sets, which I did not run. So the
  +0.046 has no error bar here and I am not claiming it is significantly greater than zero — only
  that the *floor* is 0.872 and the treatment is 0.918, each with its own held-out K-fold.
  That paired band is the obvious next thing.
- **No word-count floor.** `conscription_instrument_v1.md` U3 wants both and has no rule for
  disagreement; I measured the embedding bag only.
- **No no-patch arm**, because there is no patch: this is a read-only decoding measurement, so
  non-negotiable 2 does not bite and non-negotiable 1's third arm has no referent here. Said
  plainly rather than silently dropped.
- **Did not touch** `scripts/painaxis_tierB.py`, `src/lsx/core/painaxis_remote.py`,
  `scripts/painaxis_extract.py` or `scripts/painaxis_analyze.py` (the concurrent NDIF agent's
  files); everything new is in `scripts/painaxis_floor_nulls.py` and
  `tests/test_painaxis_floor_nulls.py`. No NDIF, no remote model, no downloads.
- **Did not reuse the Tier A checkpoints** — they were gone (`.npz` is gitignored and
  `results/painaxis_tierA/` holds only the five CSV/JSON files). So this is a fresh 800-sentence
  extraction, 260 MB added, which is why the exact reproduction in §4.2 is a real cross-check
  rather than a tautology. `ControlSupplement_1P` was not re-extracted: it feeds only the control-
  vector cosines, which this task does not touch.
- **Did not re-run** the per-category breakdown, the unembedding readout or the cosines with nulls.
  Those Tier A sections still have no arms.
- Did not edit `RESULTS.md`, `WRITEUP.md`, `VISION.md`, `README.md`, `docs/` or `prompts/`.

## 7. The shared floor object

`results/painaxis_floor_nulls/embed_bag.npz` (3.8 MB, gitignored) holds, per set,
`{set}__mean` (the layer-0 mean-pooled static-embedding bag, `[200, 1536]`),
`{set}__final_token`, `{set}__categories`, `{set}__sets`. That is the same object
`conscription_instrument_v1.md` §3 names as `Floor.stimulus`, built once here so both uses share
it. Same tokenizer, same prompts, same `hidden_states[0]`. It is regenerable in ~15 min from
`python scripts/painaxis_floor_nulls.py extract`.
