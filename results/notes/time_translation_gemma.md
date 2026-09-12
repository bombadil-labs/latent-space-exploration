# Time translation on Gemma-2-9B-it (hour 30, scale check)

Companion to `time_translation.md` (v1, Qwen2.5-1.5B) and `time_translation_v2.md` (vocabulary-
matched far-Δt states, Qwen2.5-1.5B). Same v2 grid (`prompts/time_translation_v2.json`, 480
passages: 240 experimental + 240 phrase-only control), same measurements 1-7
(`docs/specs/time_translation_v1.md`), no selector test (measurement 8), run on Gemma-2-9B-it via
NDIF at layers 9, 20, 31 (state span mean-pooled). Extraction: `scripts/ndif_time_translation_extract.py`.
Measurement: `scripts/time_translation.py prompts/time_translation_v2.json --model
google/gemma-2-9b-it --layers 9,20,31 --suffix gemma --stage measure`.

## Prediction (written before the run)

- Shared variance fraction at Gemma layer 20 lands in **0.4-0.6** (vs Qwen layer-14 v2: 0.545),
  Spearman(‖shared‖, log Δt) **≥ 0.6** (vs 0.68).
- The residual curves acquire subject structure: **at least 4 of 8 subjects have τ > 1 day**
  (vs all 8 flat at τ = 1 day in both Qwen runs), with **mayfly and street among the shortest**
  τ and **mountain and asteroid among the longest**.

Graded against the actual numbers below.
