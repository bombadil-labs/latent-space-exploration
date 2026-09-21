# Pain axis, Tier B: the replication on `google/gemma-2-9b-it`, through NDIF

**Verdict up front: this is a replication in the strong sense — the whole curve, not the peak.**
Across all 84 published layer-points for this model (42 layers x 2 extractions, S1_1P and S1_3P
averaged), our held-out AUC differs from theirs by **mean |Δ| = 0.0006 (0.0008 final_token, 0.0004 mean) and max |Δ| = 0.0022**.
Pearson r between the two curves is **0.9997** (final_token) and **0.9998** (mean). The two
pre-registered targets are hit to within a thousandth:

| target | theirs (published) | ours | Δ |
|---|---|---|---|
| **mean, their layer 31** | **0.9473** | **0.9472** | −0.0001 |
| **final_token, their layer 10** | **0.9313** | **0.9323** | +0.0010 |

This is not a match at one layer with divergence elsewhere. It tracks the curve — including the
shape features that carry the interpretation: the sharp final_token rise from 0.73 to 0.93 by
L10, the dip and plateau at 0.88–0.91 for the remaining 31 layers, and the slow monotone climb
of `mean` to a broad plateau after L29.

Model: `google/gemma-2-9b-it` on NDIF, 42 blocks, d_model 3584, bf16 remote, `.venv312` py3.12,
nnsight 0.7.0. Stimuli: their vendored `prompts/external/pain_axis/3.1_pain_and_control_datasets.json`,
read-only, sets `S1_1P S1_3P S2_1P S2_3P` (800 sentences). Code: `scripts/painaxis_tierB.py`,
`src/lsx/core/painaxis_remote.py`; analysis functions imported **unchanged** from
`scripts/painaxis_analyze.py`. Outputs: `results/painaxis_tierB/` (`curve_comparison.csv`,
`s1_kfold_layer_curves.csv`, `s1_kfold_summary.csv`, `embed_layer_auc.csv`, `extract_meta.json`,
and 8 gitignored `shards/*.npz`, 560 MB).

## 0. Their layer indexing, determined rather than assumed

Two independent lines, both checked, neither assumed:

1. **From their CSV.** `s1_kfold_layer_curves.csv` has exactly **42 rows per (extraction, dataset)
   for `Gemma_2_9B_instruct`, layers 0..41**. gemma-2-9b-it has 42 blocks. The same holds for
   every other model in their table: Gemma_2_2B 26 rows / 26 blocks, Llama_3.1_8B 32 / 32,
   Llama_3.1_70B 80 / 80, Gemma_3_27B 62 / 62. There is **no embedding row** in their indexing.
2. **From their source.** `01_extract_activations_and_pain_vectors.py::extract_activations` reads
   `cache[f"blocks.{layer}.hook_resid_post"]`.

So **their layer n is the residual after block n**, and ours matches index for index. This is the
same convention the Tier A correction block established, and Tier B independently confirms it —
see §3, where the true embedding layer behaves in a way layer 0 provably cannot.

Their published curve is S1 only, so the layer-by-layer comparison below is against S1_1P and
S1_3P. Their `s1_heldout_auc_at_best_layer` is the mean of those two, which reproduces their
0.94725 and 0.93125 exactly from their own CSV — confirming I am comparing like with like.

## 1. Our curve against theirs, every layer

`auc_vs_all_controls`, 5-fold split by sentence set, vector fitted on training sets only.
`theirs`/`ours` are the mean of S1_1P and S1_3P. Full per-dataset CSV:
`results/painaxis_tierB/curve_comparison.csv`.

| L | ft theirs | ft ours | ft Δ | mean theirs | mean ours | mean Δ |
|---|---|---|---|---|---|---|
| 0 | 0.7290 | 0.7268 | −0.0022 | 0.8340 | 0.8342 | +0.0002 |
| 1 | 0.7943 | 0.7953 | +0.0010 | 0.8543 | 0.8548 | +0.0005 |
| 2 | 0.8022 | 0.8027 | +0.0005 | 0.8585 | 0.8590 | +0.0005 |
| 3 | 0.8055 | 0.8065 | +0.0010 | 0.8565 | 0.8575 | +0.0010 |
| 4 | 0.8187 | 0.8170 | −0.0017 | 0.8620 | 0.8618 | −0.0003 |
| 5 | 0.8002 | 0.8020 | +0.0018 | 0.8698 | 0.8700 | +0.0002 |
| 6 | 0.8493 | 0.8492 | −0.0000 | 0.8818 | 0.8820 | +0.0002 |
| 7 | 0.8535 | 0.8550 | +0.0015 | 0.8827 | 0.8827 | 0.0000 |
| 8 | 0.8740 | 0.8735 | −0.0005 | 0.8945 | 0.8955 | +0.0010 |
| 9 | 0.9275 | 0.9267 | −0.0008 | 0.9160 | 0.9160 | −0.0000 |
| **10** | **0.9313** | **0.9323** | +0.0010 | 0.9285 | 0.9285 | 0.0000 |
| 11 | 0.9230 | 0.9230 | 0.0000 | 0.9235 | 0.9230 | −0.0005 |
| 12 | 0.9115 | 0.9112 | −0.0003 | 0.9243 | 0.9243 | 0.0000 |
| 13 | 0.9073 | 0.9083 | +0.0010 | 0.9243 | 0.9240 | −0.0003 |
| 14 | 0.9110 | 0.9108 | −0.0003 | 0.9132 | 0.9135 | +0.0003 |
| 15 | 0.9045 | 0.9048 | +0.0002 | 0.9250 | 0.9250 | 0.0000 |
| 16 | 0.9073 | 0.9062 | −0.0010 | 0.9182 | 0.9182 | 0.0000 |
| 17 | 0.9108 | 0.9108 | 0.0000 | 0.9260 | 0.9275 | +0.0015 |
| 18 | 0.8965 | 0.8967 | +0.0002 | 0.9272 | 0.9272 | 0.0000 |
| 19 | 0.8962 | 0.8962 | 0.0000 | 0.9287 | 0.9283 | −0.0005 |
| 20 | 0.8958 | 0.8958 | 0.0000 | 0.9325 | 0.9307 | −0.0018 |
| 21 | 0.8925 | 0.8915 | −0.0010 | 0.9320 | 0.9338 | +0.0018 |
| 22 | 0.8970 | 0.8965 | −0.0005 | 0.9307 | 0.9313 | +0.0005 |
| 23 | 0.8917 | 0.8932 | +0.0015 | 0.9310 | 0.9305 | −0.0005 |
| 24 | 0.8878 | 0.8882 | +0.0005 | 0.9267 | 0.9265 | −0.0002 |
| 25 | 0.8900 | 0.8887 | −0.0012 | 0.9338 | 0.9325 | −0.0013 |
| 26 | 0.8950 | 0.8930 | −0.0020 | 0.9355 | 0.9358 | +0.0003 |
| 27 | 0.8813 | 0.8800 | −0.0013 | 0.9370 | 0.9372 | +0.0002 |
| 28 | 0.8962 | 0.8955 | −0.0008 | 0.9385 | 0.9380 | −0.0005 |
| 29 | 0.9048 | 0.9055 | +0.0008 | 0.9447 | 0.9445 | −0.0002 |
| 30 | 0.9010 | 0.8995 | −0.0015 | 0.9470 | **0.9472** | +0.0002 |
| **31** | 0.9002 | 0.8993 | −0.0010 | **0.9473** | **0.9472** | −0.0000 |
| 32 | 0.8903 | 0.8907 | +0.0005 | 0.9470 | 0.9470 | 0.0000 |
| 33 | 0.8970 | 0.8973 | +0.0003 | 0.9415 | 0.9418 | +0.0003 |
| 34 | 0.9002 | 0.8998 | −0.0005 | 0.9420 | 0.9417 | −0.0002 |
| 35 | 0.8952 | 0.8952 | 0.0000 | 0.9420 | 0.9420 | 0.0000 |
| 36 | 0.8932 | 0.8925 | −0.0007 | 0.9427 | 0.9427 | 0.0000 |
| 37 | 0.8920 | 0.8915 | −0.0005 | 0.9420 | 0.9425 | +0.0005 |
| 38 | 0.9000 | 0.8998 | −0.0002 | 0.9427 | 0.9432 | +0.0005 |
| 39 | 0.8977 | 0.8962 | −0.0015 | 0.9473 | 0.9470 | −0.0002 |
| 40 | 0.9055 | 0.9052 | −0.0002 | 0.9417 | 0.9417 | 0.0000 |
| 41 | 0.9012 | 0.9000 | −0.0012 | 0.9400 | 0.9400 | 0.0000 |

The largest disagreement anywhere is 0.0022, at L0 final_token — the layer with the least signal
and, being one block from the embeddings, the one most exposed to a tokenisation difference. Every
other point is within 0.002. **Nothing about the discrepancy pattern looks systematic**: the deltas
are centred on zero (mean signed Δ = −0.0002 final_token, +0.0001 mean), they do not drift with
depth, and they are the size you would expect from bf16 arithmetic on different hardware.

## 2. Summary row, in their `s1_kfold_summary.csv` shape

| model | extraction | s2_layer | s1_heldout@s2_layer | s1_best_layer | s1_heldout@best | s1_1P@best | s1_3P@best |
|---|---|---|---|---|---|---|---|
| **ours** Gemma_2_9B_instruct | final_token | 37 | 0.8915 | 10 | **0.93225** | 0.9545 | 0.9100 |
| **theirs** Gemma_2_9B_instruct | final_token | 37 | 0.8920 | 10 | **0.93125** | 0.9550 | 0.9075 |
| **ours** Gemma_2_9B_instruct | mean | 31 | **0.94725** | 30 | **0.94725** | 0.9430 | 0.9515 |
| **theirs** Gemma_2_9B_instruct | mean | 31 | **0.94725** | 31 | **0.94725** | 0.9395 | 0.9550 |

`s2_layer` is their `01` selection (argmax of the S2_1P/S2_3P held-out curve); `s1_best_layer` is
their `08` selection (argmax of the S1 curve). **Both `s2_layer` values reproduce exactly** (37 and
31), which is a stronger check than it looks — it is an argmax over a 42-point curve computed from
an entirely separate pair of datasets, and it landed on the same integer twice.

**The one place we differ is instructive and I am not smoothing it over.** Our `mean` S1 argmax is
L30, theirs is L31. The *value* is identical: our L30 and L31 are **both exactly 0.947250**, an
exact tie that `idxmax` breaks toward the lower index. Theirs are 0.9470 and 0.9473, so their
argmax breaks the other way. **This is a tie-break artefact, not a disagreement** — at their chosen
L31 our value is 0.94725 against their 0.94725.

It does, however, make the more general point: **the peak layer is not identified to better than
±1 by this procedure**, because the curve is flat to well within fold noise across the top —
L29–L39 spans 0.9417 to 0.9473, a spread of 0.0055 over eleven layers, against a per-fold std of
roughly 0.03. This is exactly why this project's rule 4 says report the curve, and why the curve,
not the peak, is the headline above. Quoting "layer 31" as though the method resolved a layer would
be over-reading it; quoting 0.9473 as the plateau height would not.

## 3. The embedding layer: the number nobody, including them, has measured

Their indexing has no embedding row, and Tier A's correction block flagged that the true lexical
floor had never been measured. It is cheap on this path, so I measured it. `layer_embed` is
`embed_tokens.output`, run through the same K-fold and the same functions.

| extraction | S1_1P | S1_3P | S2_1P | S2_3P |
|---|---|---|---|---|
| **final_token** | **0.500** | **0.500** | **0.500** | **0.500** |
| **mean** | 0.789 | 0.833 | 0.871 | 0.876 |

**final_token at the embedding layer is exactly 0.500, with fold std exactly 0.0.** Not
approximately — the measured variance of `embed_final_token` across all 200 sentences in a set is
**0.0 exactly**. Every prompt in a set ends with the identical suffix ("… I feel:" in 1P), so the
static embedding of the final token is the same vector for a pain sentence and a neutral one. The
AUC is chance because the readout is a constant.

Two things follow, and they cut in opposite directions:

- **It settles the Tier A correction, independently and by construction.** Layer 0 final_token is
  0.727 here. A true embedding readout at the final token is provably 0.500 on these stimuli. So
  the 0.727 at "layer 0" is produced *entirely* by one attention+MLP block moving context into the
  final position — it cannot be lexical, because the lexical content of that position is constant.
  "Before any computation" was never what layer 0 measured, and now there is a number showing it.
- **It substantially deflates the `mean` result, which is their headline extraction.** Mean-pooled
  AUC over *static embeddings alone* is 0.811 averaged over S1 (0.874 over S2). Their S1 peak is
  0.947. So on a chance-to-peak scale, **a bag of embeddings with no transformer at all already
  covers about 70% of the distance** (0.811 − 0.5) / (0.947 − 0.5). The 42 blocks add the
  remaining 30%. The `mean` curve is high at every layer largely because pain sentences use
  lexically distinct words, which is a fact about the stimulus set, not about a representation the
  model computes.

The `final_token` curve does not have this problem: it starts at a provable 0.500 and reaches
0.932, so all of its signal is computed. **On this model I would report final_token as the
scientifically stronger result even though `mean` is numerically higher**, which is the same
conclusion Tier A reached on Qwen2.5-1.5B by a weaker argument. Their paper leads with the `mean`
kind of number.

Caveat on the arithmetic: `embed_tokens.output` is the *unscaled* embedding; Gemma-2 multiplies by
`sqrt(d_model)` before block 0, so HF's `hidden_states[0]` is ours times a positive scalar.
Difference-in-means, PCA denoising, projection and ROC-AUC are all invariant to a positive scalar,
so the AUC is identical either way. Norms and cross-layer cosines are not, and must not be mixed.
This is recorded in `painaxis_remote.EMBED_SCALE_NOTE`. I did **not** capture HF `hidden_states[0]`
directly to confirm the scale factor empirically; I argued it from the Gemma-2 forward pass.

## 4. S2, for completeness (they published no S2 curve for this model)

| extraction | dataset | L0 | peak | L41 |
|---|---|---|---|---|
| final_token | S2_1P | 0.826 | **0.992** @L37 | 0.991 |
| final_token | S2_3P | 0.832 | 0.965 @L37 | 0.960 |
| mean | S2_1P | 0.889 | 0.984 @L34 | 0.978 |
| mean | S2_3P | 0.897 | 0.962 @L31 | 0.959 |

S2 runs 0.03–0.06 above S1 at every layer, the same offset and direction Tier A found on Qwen.
Their S1/S2 are different sentence grids, so this is a stimulus-set property, not a bug — but it
means "how separable is the pain axis" has no single answer, it has a grid-dependent one, and
0.992 on S2_1P should not be quoted beside 0.947 on S1 as though they measured the same thing.

## 5. NDIF loss

**Zero.** 80 jobs submitted, **80 returned, 0 failed, 0 retried, 800 of 800 sentences captured**
(`extract_meta.json`: `jobs_submitted: 80, jobs_failed: 0, shards_failed: []`). No shard was lost,
so the per-shard checkpointing never had to earn its keep on this run.

The brief warned of ~18% loss on NDIF *generation*. This is extraction, and it behaved like h39's
clean 720-text run, not like generation. What did happen was **latency, not loss**: one job sat
`QUEUED at position 1` for **6 min 58 s** before running, turning the `S2_1P[100]` shard from the
typical 67 s into 497 s. It completed on its first attempt — `retry_job`'s 420 s resubmit threshold
was reached but the job returned before the resubmit mattered. Every other shard ran at 67–72 s.
So the failure mode to budget for here is a multi-minute queue stall on a shared HOT deployment,
not dropped work. 40 batch jobs + 40 batched-vs-single equivalence jobs = 80.

## 6. What I verified vs what I assumed

**Verified, on the live deployment:**

1. **That the model was actually served, before committing to the run.** `google/gemma-2-9b-it`,
   `deployment_level: HOT`, `application_state: RUNNING`. Then a smoke forward returning
   `[2, 11, 3584]`, with `len(blocks) == 42` and `hidden_size == 3584` read off the served model.
2. **Padding side read back off the remote tokenizer** (`left`) and asserted against the indexing
   convention (`end_relative`) by `checks.assert_padding_convention` before any job. Under left
   padding the final real token *is* at index −1 and the masked mean covers exactly the real
   tokens. The h39 trap is structurally absent, not handled-I-hope.
3. **Batched-vs-single equivalence on the shortest (= maximally padded) item of every batch**,
   `checks.shortest_item_index` + `checks.assert_batch_equivalence`: **140 checks, min cosine
   0.99989, max 1.0000**. Four readouts per check (mean at mid-depth, mean at last layer,
   final_token at mid-depth, embed mean).
4. **That this guard actually fires.** I injected the bug it exists to catch — replaced the masked
   mean with `h.mean(dim=1)` over all positions including padding — and ran it on a real padded
   batch. Caught at cosine **0.99713 < 0.999**. Worth recording: with only 3 pad tokens in a
   13-token sequence the corrupted vector still sits at cosine 0.997 to the truth, so a 0.99
   threshold would have passed h39 straight through. The 0.999 threshold is doing real work.
5. **In-trace pooling against the audited path.** `cross_check_against_asserted_path` re-ran the
   reviewed single-layer `remote.remote_residuals`, pooled it offline with the same mask, and
   compared: mean cosine 0.9999968, final_token 0.9999480. This is what ties the fast multi-layer
   capture to the code that was actually reviewed. (Max relative elementwise difference 0.5–0.8%,
   which is bf16 batch-composition nondeterminism — the same text in a different batch shape gives
   slightly different activations. On an identically-shaped batch the agreement was exact, 0.0.)
6. **h36 refused by construction**: block output resolved by type (`o if isinstance(o, Tensor) else
   o[0]`), never `output[0]`; plus an explicit shape check that the returned batch dimension equals
   the requested batch size, and that 42 layers came back, on every one of the 80 jobs.
7. **Non-empty spans** on all 800 rows, for both readouts, never reaching padding.
8. **The k-fold holds out.** Not re-verified here — it is `scripts/painaxis_analyze.kfold_curve`,
   imported unchanged, and Tier A's `tests/test_painaxis_port.py` guards it with a synthetic test
   where in-sample AUC is 1.000 and held-out lands in [0.3, 0.7].
9. **Test suite: `pytest -q tests/` → 215 passed, 3 skipped.** Unchanged from before my work; my
   new module is importable from the py3.11 venv (nnsight imported inside functions only), so
   `lsx.core` stays one package.

**Assumed, not verified:**

- **BOS equivalence.** Their `model.to_tokens(prompt)` (TransformerLens) prepends BOS; I use HF
  `add_special_tokens=True`, which for Gemma-2 prepends `<bos>` (id 2). I did **not** install
  transformer_lens to confirm TL does nothing else. The 0.0006 mean agreement is strong indirect
  evidence it doesn't — a BOS difference would show up hardest in `mean`, which is the extraction
  that agrees best.
- **No chat template**, on an instruct model. That is their specification; faithful, and odd.
- **bf16 on NDIF vs whatever precision their GPU run used.** Not controllable. The measured
  batch-composition noise (§6.5) bounds it at ~0.5% elementwise, cosine > 0.9999.
- **`hidden_states[0] == embed_tokens.output * sqrt(d_model)`**, argued from the Gemma-2 forward
  rather than measured (§3).
- Their published CSV is the output of the code I read. I did not re-run their pipeline.

## 7. At least one thing I got wrong

- **I nearly reported that gemma-2-9b-it was not served, and I would have been wrong.** My first
  NDIF probe did `rows = body if isinstance(body, list) else list(body.values())` over
  `api.ndif.us/status`. That endpoint returns `{"deployments": {...}, "cluster": {...}}`, so
  `list(body.values())` yields those two mappings — neither has a `repo_id`. The probe printed
  "n_rows 2" and matched no gemma. Had I trusted it, I would have stopped and reported the model
  unreachable, which is the exact failure the brief told me to guard against. I caught it only
  because "NDIF serves 2 models total" was implausible on its face. The real listing has ~100
  deployments and gemma-2-9b-it HOT among them. **The same bug is live in
  `remote.RemoteLM.lib_versions`** — which is why every remote Stack this project has produced
  records `ndif_reported: None` while looking like a successful best-effort read. I did not fix it
  (my writable files were scoped to two) and filed it as a separate task instead.
- **My throwaway negative-control script did `res.get("mn") or next(...)`**, which raises
  `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`. The real module
  uses `if v is None`, so it was unaffected — but it is the same class of mistake.
- **A stray `inspect.py` left by another agent in the shared scratchpad shadowed the stdlib**
  and broke my script at `import numpy`. Scripts run out of the shared scratchpad get its
  directory on `sys.path`; I moved mine into a subdirectory. Worth knowing for anyone else using
  that directory.
- On non-negotiable 6: I deliberately never used `pgrep -f`/`pkill -f`. Liveness was checked by
  PID (`kill -0 <pid>`) and process listings filtered with `awk '/[p]attern/'`. The one process I
  killed (a redundant duplicate analysis competing for CPU) was killed by PID.

## 8. What I did NOT do

- **No null arms. By non-negotiable 1, these numbers are not yet a result in this project's
  sense.** There is no random-vector arm and no shuffled-label arm here, so I have not estimated
  what AUC this K-fold returns when there is nothing to find. Their method has no such arm either,
  so the port is faithful — but "faithful to a method with no null" is not the same as "has a
  null". `scripts/painaxis_floor_nulls.py` is owned by another agent and I did not touch it. The
  replication claim in this note is a claim about **agreement with their numbers**, which does not
  depend on a floor; any claim about what the axis *means* does.
- **No pass-through arm** — non-negotiable 2 does not apply, as nothing here is patched. This is
  pure extraction, no forward is modified.
- **No unit tests for the new code.** The brief scoped my writable files to `scripts/painaxis_tierB.py`
  and `src/lsx/core/painaxis_remote.py`, which excludes a test file. The assertions above are
  runtime assertions on the real run plus the live deliberate-bug check (§6.4), not pytest cases.
  That is a real gap: nothing in CI will catch a regression in `painaxis_remote.py`.
- Did not extract `ControlSupplement_1P`, `Random_*`, `Arousal_*`, `Numb_*` or the sadness set, so
  none of their z-score or numbness analyses are reproduced.
- **No per-category AUC breakdown, no unembedding readout, no cosines against the control-category
  vectors.** Tier A has all three for Qwen2.5-1.5B; none of them exist for gemma-2-9b-it. The
  §5 orthogonality claim is therefore untested on the model the paper actually used.
- Did not run the base model `google/gemma-2-9b` (their `Gemma_2_9B_base` row), so the
  instruct-vs-base contrast is unreplicated.
- Did not run the missing-shame-control experiment; deliberately not contaminated into this run.
- Did not edit `RESULTS.md`, `WRITEUP.md`, `VISION.md`, `README.md`, `docs/`, `prompts/`,
  `scripts/painaxis_extract.py`, `scripts/painaxis_analyze.py` or `scripts/painaxis_floor_nulls.py`.
