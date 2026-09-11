# latent-space-exploration

Tools for measuring and moving *relational shapes* in a transformer's residual stream.

The motivating idea: a prompt that describes a structure (thesis → antithesis → synthesis;
a Kegan subject/object transition; a plot) produces a point cloud in activation space whose
*internal relations* may be shared across domains even though the clouds sit in different regions.
If that shape can be measured (stage 2) it can be fit as an operator (stage 3) and pointed at a
domain where the corresponding parts are unknown (stage 4). Longer-term this is the activation-level
substrate for a narrative calculus: decompose a story into factors, transform them, recompose.

## Ladder

| stage | module | question | status |
|---|---|---|---|
| 1 | `lsx.model`, `lsx.extract` | capture residuals, pool by role, patch | working on Qwen2.5-0.5B |
| 2 | `lsx.compare` | does the same shape appear across domains, above baseline? | measures written, controls not yet meaningful |
| 3 | `lsx.operate` | can an affine map carry the relation to a held-out domain? | fit + holdout eval written, untested on real activations |
| 4 | `lsx.steer` | patch a target activation in and read it out by generation | patching works, no real experiment yet |

## Setup

```
uv venv .venv && . .venv/bin/activate
uv pip install -e ".[dev]"
export HF_HOME=$PWD/cache/hf HF_HUB_DISABLE_XET=1   # Xet transfer host is not reachable from the cloud env
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B', allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'])"
pytest                                   # synthetic tests on a tiny random model, no weights needed
python scripts/sweep.py prompts/dialectic_v0.json
```

CPU-only is fine for 0.5B–1.5B: ~0.5 s per extraction, ~4 tok/s generation on 4 cores.

## Prompt format

Roles are marked inline and pooled by mean over their tokens; unmarked text still runs through the
model as context.

```
In physics, the initial claim is that [[thesis: ...]]. The opposing claim is that [[antithesis: ...]]. ...
```

A grid is a JSON file with `roles` and `prompts` keyed `domain/framing`. `prompts/dialectic_v0.json`
is a draft 4-domain × 2-framing grid for the dialectic relation.

## Open problems (in order)

1. **Baseline is near ceiling.** Linear CKA between any two prompts' role stacks is ~0.95–0.99 at
   every layer on 0.5B. Three role points is too few for a shape, and the residual stream has a large
   shared component. Fixes to try: more roles per prompt, token-level clouds with Gromov-Wasserstein,
   subtracting the grand mean across many prompts, projecting out top shared PCs, and permutation
   nulls (shuffle which tokens are pooled into which role).
2. **Correspondence.** RSA/CKA assume role order lines up across prompts. GW does not, but is noisier.
3. **Domain is not one offset.** Centering removes the first-order address; contrastive domain
   directions estimated across many prompts remove more.
4. **Erosion.** A single-layer patch is repaired by later layers; `steer.steer_patches` re-imposes
   the direction at several layers. The "cohere" operator (repair projected onto the complement of
   the requested directions) is not implemented.
