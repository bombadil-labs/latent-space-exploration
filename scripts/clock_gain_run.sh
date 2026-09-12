#!/usr/bin/env bash
# Piece 1 of docs/specs/clock_depth_gain_v1.md, end to end, after the stacks exist.
set -euo pipefail
PY=/home/user/latent-space-exploration/.venv/bin/python
export HF_HOME=/home/user/latent-space-exploration/cache/hf HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1

$PY scripts/clock_gain_lex.py

# hour-35 continuity (sec 3.5): the discrimination script is run UNCHANGED on arm D
# (must reproduce 0.710 / 0.961) and on arm A (the difference is the copy-inflation estimate).
for arm in D A; do
  $PY scripts/time_translation_discrimination.py \
      "results/clock_gain_v1_stacks_${arm}.npz" prompts/clock_gain_v1.json \
      --layers 0,8,14,20,27 --out "results/clock_gain_v1_discrim_${arm}.json"
done

$PY scripts/clock_gain.py --nperm 200
