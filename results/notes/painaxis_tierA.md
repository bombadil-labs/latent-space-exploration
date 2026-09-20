> **CORRECTION (Opus, on merge).** Section 3 below calls layer 0 "the bag-of-embeddings
> baseline" and "the embedding layer". **It is neither.** `scripts/painaxis_extract.py`'s own
> docstring is correct — layer index i is the residual *after block i* — so layer 0 is the output
> of the first transformer block, one full attention+MLP in. Checked directly: every prompt in
> S2_1P ends with the identical token "I feel:", so a true embedding readout at the final token
> would have zero spread across sentences; the measured spread is 1.13. The finding survives in
> weakened form — **after a single block, mean-pooled AUC is already 0.887** — but "before any
> computation" is not what was measured, and the true lexical floor (`hidden_states[0]`) has
> never been measured, by them or by us. That is the missing number.

# Pain axis, Tier A: a faithful port run on Qwen2.5-1.5B-Instruct

**Verdict up front: the port works, and the effect is present below 2B.** Held-out AUC peaks at
**0.944** (final_token, S2_1P, layer 26) / **0.936** averaged over S2_1P and S2_3P, and at
**0.918 / 0.912** for `mean` at layer 21. Nothing is near chance anywhere on the curve — the
*lowest* held-out AUC at any of the 28 layers, either extraction, either person, is 0.731 (layer 0).
So the question the brief posed ("if a faithful port returns AUC near 0.5, the port is wrong, not the
paper") does not arise: it returned a strong signal, and the internal consistency checks below all
hold. This is an **extension**, not a replication — Qwen2.5-1.5B is not among their 25 models and
there is no published number to hit.

Model: `Qwen/Qwen2.5-1.5B-Instruct`, 28 layers, d_model 1536, local CPU fp32, `.venv` py3.11.
Stimuli: their vendored `prompts/external/pain_axis/3.1_pain_and_control_datasets.json`, read-only,
sets `S1_1P S1_3P S2_1P S2_3P ControlSupplement_1P` (900 sentences, 0.8 s/sentence, ~13 min).
Code: `scripts/painaxis_extract.py`, `scripts/painaxis_analyze.py`, `scripts/painaxis_numpy_impl.py`,
`tests/test_painaxis_port.py`. Outputs: `results/painaxis_tierA/` (`layer_curves.csv`,
`s1_kfold_summary.csv`, `per_category_auc.csv`, `cosines.csv`, `summary.json`, and the five
gitignored `acts_*.npz` checkpoints, 413 MB).

## What ran on sklearn and what ran on my own code

sklearn 1.9.1 **is** installed in `.venv`, so every number below was produced by **their** calls:
`sklearn.decomposition.PCA`, `sklearn.metrics.roc_auc_score`, `sklearn.model_selection.KFold`, with
their constants read out of their file (`N_FOLDS = 5`, `RANDOM_SEED = 42`, `DENOISE_VARIANCE = 0.5`).
`scripts/painaxis_numpy_impl.py` holds independent numpy versions (PCA by SVD on the centred
controls; ROC-AUC by the Mann-Whitney rank identity with ties averaged) which are **not** on the
path that produced these numbers — they exist so `tests/test_painaxis_port.py` can check the
arithmetic against a second implementation, and so the port runs unchanged where sklearn is absent.
Both are verified in the test suite against sklearn *and* against a brute-force pairwise AUC on
random data with deliberate ties. `pandas` is **not** installed; CSVs are written with `csv`.

## 1. Held-out AUC layer curve, all 28 layers, both extraction types

Their K-fold: `KFold(5, shuffle=True, random_state=42)` over the 20 **sentence sets**, vector fitted
on the training sets, AUC scored on the held-out sets, averaged over folds. `auc_vs_all_controls`.

| layer | ft S2_1P | ft S2_3P | ft S1_1P | ft S1_3P | mean S2_1P | mean S2_3P | mean S1_1P | mean S1_3P |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.795 | 0.800 | 0.731 | 0.778 | 0.887 | 0.867 | 0.776 | 0.839 |
| 1 | 0.799 | 0.804 | 0.778 | 0.791 | 0.866 | 0.864 | 0.768 | 0.822 |
| 2 | 0.794 | 0.837 | 0.779 | 0.745 | 0.862 | 0.864 | 0.776 | 0.832 |
| 3 | 0.809 | 0.829 | 0.799 | 0.790 | 0.882 | 0.880 | 0.784 | 0.828 |
| 4 | 0.807 | 0.829 | 0.798 | 0.754 | 0.886 | 0.887 | 0.789 | 0.836 |
| 5 | 0.822 | 0.826 | 0.791 | 0.759 | 0.887 | 0.893 | 0.785 | 0.839 |
| 6 | 0.812 | 0.786 | 0.770 | 0.791 | 0.882 | 0.887 | 0.795 | 0.836 |
| 7 | 0.847 | 0.814 | 0.791 | 0.808 | 0.878 | 0.888 | 0.797 | 0.835 |
| 8 | 0.841 | 0.824 | 0.777 | 0.839 | 0.883 | 0.889 | 0.804 | 0.833 |
| 9 | 0.835 | 0.850 | 0.785 | 0.829 | 0.879 | 0.888 | 0.801 | 0.834 |
| 10 | 0.863 | 0.877 | 0.820 | 0.836 | 0.895 | 0.900 | 0.804 | 0.834 |
| 11 | 0.864 | 0.870 | 0.809 | 0.856 | 0.893 | 0.903 | 0.806 | 0.838 |
| 12 | 0.863 | 0.872 | 0.794 | 0.852 | 0.888 | 0.899 | 0.803 | 0.834 |
| 13 | 0.882 | 0.867 | 0.814 | 0.893 | 0.883 | 0.901 | 0.806 | 0.832 |
| 14 | 0.873 | 0.871 | 0.825 | 0.873 | 0.887 | 0.897 | 0.810 | 0.834 |
| 15 | 0.904 | 0.870 | 0.851 | 0.902 | 0.891 | 0.899 | 0.807 | 0.837 |
| 16 | 0.906 | 0.890 | 0.849 | 0.891 | 0.891 | 0.898 | 0.813 | 0.837 |
| 17 | 0.911 | 0.872 | 0.842 | 0.880 | 0.885 | 0.895 | 0.804 | 0.832 |
| 18 | 0.898 | 0.866 | 0.842 | 0.867 | 0.882 | 0.885 | 0.807 | 0.836 |
| 19 | 0.893 | 0.869 | 0.840 | 0.856 | 0.887 | 0.894 | 0.806 | 0.828 |
| 20 | 0.914 | 0.880 | 0.862 | 0.873 | 0.906 | 0.902 | 0.817 | 0.839 |
| 21 | 0.912 | 0.897 | 0.857 | 0.866 | **0.918** | 0.906 | 0.825 | 0.855 |
| 22 | 0.916 | 0.898 | 0.856 | 0.858 | 0.907 | 0.900 | 0.820 | 0.851 |
| 23 | 0.920 | 0.898 | 0.860 | 0.870 | 0.911 | 0.904 | 0.813 | 0.851 |
| 24 | 0.935 | 0.916 | 0.863 | 0.889 | 0.905 | 0.901 | 0.810 | 0.842 |
| 25 | 0.941 | 0.917 | 0.868 | 0.891 | 0.903 | 0.899 | 0.814 | 0.848 |
| 26 | **0.944** | **0.927** | 0.863 | 0.870 | 0.912 | 0.897 | 0.887 | 0.899 |
| 27 | 0.919 | 0.922 | 0.875 | 0.887 | 0.897 | 0.893 | 0.841 | 0.886 |

Layer *i* = residual stream **after** block *i* (their `blocks.{i}.hook_resid_post`), so layer 27 is
the pre-final-norm residual, captured as the block's own output — not `hidden_states[-1]`, which is
the norm output (see `src/lsx/model.py` and `tests/test_invariants.py`).

Shape: `final_token` rises monotonically-ish into the last quarter and peaks at L26; `mean` is
already at 0.87-0.89 at layer 0 (the bag-of-embeddings baseline: pain words are lexically distinct)
and gains only ~0.03 over 21 layers. That difference matters for interpretation — **the `mean`
curve's height is not evidence of a computed representation**, because the embedding layer alone
nearly matches it. The `final_token` curve, which starts at 0.795 and adds 0.15, is the one that
shows the model building something. Their paper's headline numbers are of the `mean` kind; on this
model I would not read `mean` as the stronger result, even though it is numerically comparable.

Full per-fold CSV: `results/painaxis_tierA/layer_curves.csv`.

## 2. Best layer by their own K-fold, in their `s1_kfold_summary.csv` shape

Columns as in `Pain-axis/scripts/3.3_validation/08_s1_auc.py` (their S1 summary), plus the two S2
columns so the S2 selection is visible in the same row. Best layer is chosen on the **held-out**
curve averaged over 1P and 3P, which is their procedure; I did not select on scoring data.

| model | extraction | s2_layer | s1_heldout_auc_at_s2_layer | s1_best_layer | s1_heldout_auc_at_best_layer | s1_1P_heldout_at_best | s1_3P_heldout_at_best | s2_1P_heldout_at_s2_best | s2_3P_heldout_at_s2_best |
|---|---|---|---|---|---|---|---|---|---|
| Qwen_2.5_1.5B_instruct | final_token | 26 | 0.8667 | 27 | 0.8813 | 0.8750 | 0.8875 | **0.9435** | **0.9275** |
| Qwen_2.5_1.5B_instruct | mean | 21 | 0.8400 | 26 | 0.8930 | 0.8875 | 0.8985 | **0.9180** | **0.9065** |

1P vs 3P: essentially no gap (final_token 0.944 vs 0.928; mean 0.918 vs 0.907; S1 0.875 vs 0.888).
Whatever this axis is, on this model it is **not** specific to first-person framing. That is a
negative for any self-referential reading of the vector and should be recorded as such.

S1 is consistently ~0.05-0.08 below S2 at every layer. Their S1 and S2 are different sentence grids,
so this is a stimulus-set difference, not a bug — but it does mean the "which layer" answer moves
(26/27 for final_token, 21 vs 26 for mean) depending on which grid you ask.

## 3. Per-control-category AUC at the chosen layer

`in_sample_S2_1P_vector` is exactly their Figure recipe (`create_auc_bars`): the S2_1P vector, fitted
on all of S2_1P, scored on the same sentences. `heldout_kfold` is the same breakdown computed inside
their K-fold, so the vector never sees the sentences it scores. **Their published per-category figure
is the in-sample one**; I report both because the in-sample row is not a held-out number and should
not be quoted as one.

| extraction | layer | dataset | kind | ALL | B fear | C1 neg-emotion | C2 neg-world | D neutral | E body-sensation |
|---|---|---|---|---|---|---|---|---|---|
| final_token | 26 | S2_1P | in-sample (theirs) | 0.966 | 0.980 | 0.922 | 0.988 | 0.987 | 0.952 |
| final_token | 26 | S2_3P | in-sample (theirs) | 0.933 | 0.963 | 0.872 | 0.931 | 0.952 | 0.945 |
| final_token | 26 | S2_1P | **held-out** | 0.944 | 0.965 | 0.893 | 0.978 | 0.978 | 0.905 |
| final_token | 26 | S2_3P | **held-out** | 0.928 | 0.950 | 0.903 | 0.915 | 0.953 | 0.918 |
| mean | 21 | S2_1P | in-sample (theirs) | 0.945 | 0.960 | 0.933 | 0.885 | 0.995 | 0.951 |
| mean | 21 | S2_3P | in-sample (theirs) | 0.935 | 0.959 | 0.936 | 0.831 | 0.994 | 0.957 |
| mean | 21 | S2_1P | **held-out** | 0.918 | 0.938 | 0.903 | 0.860 | 0.985 | 0.905 |
| mean | 21 | S2_3P | **held-out** | 0.907 | 0.923 | 0.905 | 0.790 | 0.990 | 0.925 |

The ordering matches their claim qualitatively: neutral (D) is separated best, negative emotion (C1)
worst — i.e. the hardest control is the semantically nearest one, which is what you would want if
the axis were pain-specific rather than valence-generic. But C1 at 0.89-0.90 held-out is still high,
and **C1 is the category in which the single token "shame" appears** (PROVENANCE.md: no shame,
humiliation or embarrassment control exists at all). The nearest-neighbour control the project
suspects is missing is missing here too; that is a separate experiment and I did not run it.

## 4. Unembedding readout, before and after denoising

Logit lens through the final norm and `lm_head` (`LM.unembed`), on the S2_1P vector at
final_token layer 26. Denoising removed **4 components** (cumulative control variance 0.534 at the
4th, their `searchsorted(cumvar, 0.5) + 1` rule), shrinking the norm 29.91 → 20.43 and turning the
vector by cos 0.683.

**Raw (pain mean − control mean, no denoising)**
- promoted: ` hurt, ashamed, painful, pain, lost, shame, guilt, crushed, shattered, worthless,
  dis, regret, isolated, conf, hopeless, deep, tortured, broken, betrayed, guilty`
- suppressed: `ibus, temperature, preload, /weather, 停车场, mapper, predicted, imap, svens,
  Cooler, 行车, плав, cco, 热水, MaxY, Temperature, robat, 室外, _CID, 防晒`

**Denoised (the actual pain vector)**
- promoted: `自卑(inferiority), rejection, rejected, 耻(shame), 学习成绩, 愚(foolish), perfection,
  flaws, shame, 失败(failure), 分手(break-up), self, inferior, failure, 贬(belittle), misunderstood,
  flawed, .self, foolish, FAILURE`
- suppressed: ` Cooler, ario, arios, /weather, Atmospheric, warmer, ibus, 气象, 气候, 降雨, 室外,
  保暖, 气温, ystate, eco, охран, Alert, cooler, 扬尘, SAFE`

**This contradicts the brief's premise, and I am reporting it rather than dressing it up.** The
brief said a raw residual's readout is common-mode junk — punctuation and digits — and that the
denoising is what makes the readout meaningful. On this model at this layer the *raw* readout is
already the cleanest match to the paper's reported list: it contains `hurt, shame, guilt, worthless,
pain` outright (5 of their 7; `rejected` and `hollow` are absent, though ` rejection`/` rejected`
top the denoised list). The denoised readout is *more specific* — it moves decisively toward social
and psychological injury (inferiority, shame, rejection, failure) and away from physical pain words
— but it is not "junk → meaning". What the denoising removes here is evidently not punctuation
common-mode but a broad negative-affect/physical-sensation component; what is left is the
social-evaluative part. A fair summary is: **denoising sharpened the vector's meaning and changed
it (cos 0.68), it did not rescue it from noise.**

Two caveats I will not paper over. (a) A large share of the denoised top tokens are Chinese; Qwen's
unembedding is heavily multilingual and this readout is partly a property of the tokenizer's
geometry, not only of the vector. (b) The suppressed lists for both raw and denoised are dominated
by weather/temperature tokens, which are exactly the semantic field of their neutral (D) and
body-sensation (E) stimuli — the readout is partly reading the control set back, which is expected
for a difference-of-means vector and is a reason not to over-read it.

## 5. Cosines with the control-category vectors (their orthogonality claim)

Control vectors built by their `02_build_control_vectors.py` recipe at the same layer: category mean
pooled over `S1_1P + S2_1P + ControlSupplement_1P`, minus the pooled neutral (D) mean, denoised
against the neutral cloud's SVD basis to 0.5 cumulative variance.

| pair | cosine |
|---|---|
| pain vs fear (B) | **0.063** |
| pain vs neg-emotion (C1) | **0.256** |
| pain vs neg-world (C2) | **−0.034** |
| pain vs body-sensation (E) | **0.140** |
| pain (S2) vs pain (S1) | 0.582 |
| pain raw vs pain denoised | 0.683 |
| fear vs neg-emotion | 0.783 |
| fear vs neg-world | 0.686 |
| neg-emotion vs neg-world | 0.792 |
| fear vs body-sensation | 0.161 |
| neg-emotion vs body-sensation | −0.017 |
| neg-world vs body-sensation | −0.167 |

Their orthogonality claim holds on this model, and it holds in an informative way: the four control
directions are strongly mutually correlated (0.69-0.79 among B/C1/C2 — a generic negative-valence
cluster), while pain sits nearly orthogonal to all of them (|cos| ≤ 0.26) yet at 0.58 with the pain
vector fitted on the *other* sentence grid. So the axis is not "negative valence": the negative-
valence directions agree with each other far more than any of them agrees with pain. Nearest of the
controls is C1 negative emotion (0.256), which is also the hardest control by AUC — the two measures
agree, which is a mild internal consistency check.

## What I verified vs what I assumed

**Verified (in `tests/test_painaxis_port.py`, 9 tests, all passing):**
1. numpy SVD-PCA matches `sklearn.PCA` on random data — components (up to sign) and
   explained-variance ratios.
2. My rank-identity ROC-AUC matches `roc_auc_score` **and** a brute-force pairwise
   `P(pos>neg) + 0.5 P(tie)` on 5 random seeds with deliberate ties.
3. The denoise component count obeys their `searchsorted(cumvar, 0.5) + 1` rule at the boundary
   (`cum[k-1] >= 0.5 and cum[k-2] < 0.5`), and the returned vector is numerically orthogonal to
   every component removed.
4. **The K-fold actually holds out.** Synthetic data where the pain direction is a *different*
   random direction in every sentence set: fitting and scoring on everything gives AUC 1.000; the
   K-fold held-out AUC lands in [0.3, 0.7]. This is the named trap and it is guarded by a test that
   would fail loudly if the vector ever saw its scoring sentences.
5. Fold masks are disjoint, partition the data, and every sentence is held out exactly once.
6. Extraction-time asserts: the hook fired exactly `n_layers` times per sentence and the captured
   sequence length equals the input length, on all 900 sentences.

**Assumed, not verified:**
- **The BOS token.** TransformerLens's `to_tokens` prepends BOS; Qwen2.5-Instruct has no BOS token,
  and TL's `set_tokenizer` aliases `bos_token = eos_token`, which for this model is `<|im_end|>`
  (id 151645, *not* `<|endoftext|>`). I prepend 151645 on that reasoning. I did not install
  transformer_lens to confirm — no network. If TL in fact prepends nothing or `<|endoftext|>`, the
  `mean` numbers shift (the prefix token is one of ~10 positions and carries a large-norm sink);
  `final_token` is unaffected. Cheap hedge I did take: `mean_nobos` (mean over positions ≥ 1) is
  stored in every `.npz`, so the sensitivity can be measured without re-running the model.
- **No chat template.** They feed the raw prompt string; so do I, on an instruct model. Faithful,
  and worth flagging as odd, but it is their specification.
- That their GPU bf16 + TransformerLens path and my CPU fp32 + HF-hooks path agree numerically. I
  cannot check this without their hardware; the layer-indexing convention is the piece I did verify
  by construction (block outputs, not `hidden_states[-1]`).

## Things in their code I disagreed with and implemented anyway

1. **`pca.fit(control_acts - control_mean)`** — sklearn's `PCA` centres internally, so subtracting
   the mean first is a no-op except when NaNs are present, where it silently differs from the
   `nanmean` used for the vector. Implemented verbatim.
2. **`np.searchsorted(cumvar, 0.5) + 1`** overshoots by one component whenever `cumvar` lands
   exactly on 0.5, and in general returns the first index *reaching* 0.5 plus one. Here it removed
   4 components at 0.534 cumulative variance where 3 would have given 0.49. Implemented verbatim.
3. **The per-category figure is in-sample** (`create_auc_bars` fits on all of S2_1P and scores the
   same sentences) while the layer curve beside it is held-out. Mixing the two in one figure invites
   the reader to compare numbers that are not comparable. I implemented theirs and added the
   held-out breakdown next to it rather than replacing it.
4. **`compute_auc` discards NaN scores after labelling**, so an arm with many NaNs silently reports
   an AUC over a shrunken, possibly unbalanced sample instead of failing. No NaNs occurred here.
5. **Best layer is `idxmax` over a held-out curve**, and the same held-out number is then quoted as
   the result at that layer. Selecting the argmax on the curve and reporting the max of the same
   curve is optimistic by roughly the curve's fold-noise, even though no scoring data was used to
   fit the vector. This is why the curve, not the peak, is the headline above — which is also this
   project's rule 4.

## At least one thing I got wrong

- **I broke non-negotiable 6 in this very run.** My first background watcher was
  `until ... || ! pgrep -f painaxis_extract.py`, whose own command string contains the pattern —
  so the liveness check matched itself and would never have fired on a crash (it would have hung
  silently until timeout, looking exactly like "still running"). Caught it before it cost anything,
  stopped that task, and re-armed on log signatures (`EXTRACTION DONE|Traceback|Error|Killed`)
  instead of on process existence. No false kill this time; the rule earned its place again.
- **I nearly shipped a mean-over-padding bug by omission.** The first design batched sentences for
  speed. I dropped batching entirely and ran batch size 1 — 900 forwards, 13 minutes — precisely so
  that the two named traps (mean over padding, final-token index before right-padding) cannot exist
  rather than being handled correctly-I-hope. That is a deliberate cost: the same run on our
  `src/lsx/core/extract.py` batched path would have needed the padding-equivalence assertions to
  carry the whole argument.
- **My initial expectation about the readout was wrong**, and it was wrong in the same direction the
  brief's was: I expected the raw vector's readout to be punctuation and digits. It was `hurt,
  ashamed, painful, pain, shame, guilt, worthless`. I have reported the observation over the
  expectation (§4).

## What I did NOT do

- **No NDIF, no remote model, no Gemma-2-9B.** Tier B is untouched. Nothing here is the replication;
  the 0.9473 / 0.9313 targets remain unmet and unattempted.
- **No baseline arms.** This note has no random-vector arm and no shuffled-label arm. Their method
  has no such arm either, so this is faithful, but by this project's non-negotiable 1 these numbers
  are not yet a result in our sense. **The obvious next thing is a random-direction and a
  label-shuffled null at each layer**, to say what AUC this K-fold returns when there is nothing to
  find. I estimated no noise floor and am not claiming one.
- **Did not extract `Random_*`, `Arousal_*`, `Numb_*` or the sadness set**, so none of their z-score
  or "numbness" analyses are reproduced — only the five sets the deliverables need.
- **Did not run the missing-shame-control experiment.** PROVENANCE flags that their control set has
  no shame/humiliation category; §4 here shows the denoised vector pointing straight at shame and
  inferiority, which makes that gap more interesting, not less. Deliberately not contaminated into
  this run.
- **Did not verify against their published outputs.** Their repo's `results/` was not consulted for
  a numerical cross-check of my port on a model they ran; the only cross-checks are internal.
- Did not edit `RESULTS.md`, `WRITEUP.md`, `VISION.md`, `README.md`, `docs/` or anything under
  `prompts/`.
