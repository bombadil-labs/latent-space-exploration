# Replication targets, recorded before running anything

Their published held-out AUCs (`results/3.2_pain_vectors/auc_tables/s1_kfold_summary.csv`),
for the models this environment can actually reach:

| model | extraction | their best layer | their held-out AUC |
|---|---|---|---|
| Gemma-2-9B-instruct | final_token | 10 | **0.9313** |
| Gemma-2-9B-instruct | mean | 31 | **0.9473** |
| Llama-3.1-8B-instruct | final_token | 30 | **0.9043** |
| Llama-3.1-8B-instruct | mean | 25 | **0.9318** |
| Llama-3.1-70B-instruct | final_token | 46 | **0.9255** |
| Llama-3.1-70B-instruct | mean | 47 | **0.9498** |
| Gemma-2-2B-instruct | mean | 20 | 0.9253 |

## Two tiers, and only one of them is a replication

- **Tier A, local, Qwen2.5-1.5B-Instruct.** This model is **NOT** among their 25, so this is an
  *extension* — does the axis appear below 2B? There is no target to hit. Its value is as a method
  check: if a faithful port returns AUC near 0.5 on a model where the effect should exist, the port
  is wrong, not the paper.
- **Tier B, NDIF, Gemma-2-9B-instruct.** Same model, same sentences, our implementation. This is the
  replication, and the number to hit is **0.9473 (mean) / 0.9313 (final_token)**.

## Their method, from their code, not their prose

`compute_pain_vector`: mean(pain) − mean(controls), then **denoise** — PCA on the centred control
activations, project out the components whose cumulative explained variance first reaches
`DENOISE_VARIANCE = 0.5`. `compute_auc`: `roc_auc_score` of projections, pain vs controls.
`compute_layer_curves_kfold`: `KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)`,
the vector fit on train folds and scored on held-out, averaged.

**Port it faithfully. Any deviation is a bug, not an improvement.** We think their control set is
missing its nearest neighbour; that is a separate experiment and must not contaminate this one.
