#!/usr/bin/env bash
# Hour 19 follow-up: the three controls the first commutator run lacked.
# Runs against the main checkout's stacks; appends into results/commutator_gemma9b_v2.json
# (seeded from results/commutator_gemma9b.json, so the news/road L14-L20 runs are reused).
set -u
cd "$(dirname "$0")/.."
MAIN=/home/user/latent-space-exploration
source $MAIN/.venv312/bin/activate
export HF_HOME=$MAIN/cache/hf HF_HUB_DISABLE_XET=1 PYTHONPATH=$PWD/src
OUT=results/commutator_gemma9b_v2.json
TH="--grid prompts/narrative_theme_v1.json --stacks $MAIN/results/stacks_gemma_2_9b_it_narrative_theme_v1.npz"
VO="--grid prompts/narrative_factors_v1.json --stacks $MAIN/results/stacks_gemma_2_9b_it_narrative_factors_v1.npz"

# 1. random-direction null, matched norm, same protocol (4 random pairs x 2 prompts per factor pair)
python scripts/ndif_commutator.py --pair era_theme $TH --null 4 --prompts news,road --out $OUT
python scripts/ndif_commutator.py --pair era_voice $VO --null 4 --prompts news,road --out $OUT
# 2. two more prompts at the original layer pair (dominance table already measured on news/road)
python scripts/ndif_commutator.py --pair era_theme $TH --prompts door,fire --no-single --out $OUT
python scripts/ndif_commutator.py --pair era_voice $VO --prompts door,fire --no-single --out $OUT
# 3. second layer pair, with the single patches needed for the dominance ordering
python scripts/ndif_commutator.py --pair era_theme $TH --layers 16,24 --prompts news,road --out $OUT
python scripts/ndif_commutator.py --analyze --out $OUT
