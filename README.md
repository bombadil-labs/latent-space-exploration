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
| 2 | `lsx.compare` | does the same shape appear across domains, above baseline? | **mostly slot position, not content**: shuffled control kills the by-content signal (RESULTS.md) |
| 3 | `lsx.operate` | can an affine map carry the relation to a held-out domain? | role identity is a domain-independent direction (constant baseline rank 1.07); after removing it, a weak relation transfers (rank 3.0 vs 3.5 null, peaks layer 20) |
| 4 | `lsx.steer` | patch a target activation in and read it out | role direction selects the held-out domain's span (rank 1.3–1.7 vs 3.3 random, best at layers 14–20); relation-content patch is null |

## Setup

```
uv venv .venv && . .venv/bin/activate
uv pip install -e ".[dev]"
export HF_HOME=$PWD/cache/hf HF_HUB_DISABLE_XET=1   # Xet transfer host is not reachable from the cloud env
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B', allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'])"
pytest                                   # synthetic tests on a tiny random model, no weights needed
python scripts/extract_grid.py prompts/holonic_v1.json --model Qwen/Qwen2.5-1.5B
python scripts/stage2.py prompts/holonic_v1.json --model Qwen/Qwen2.5-1.5B --metric rsa --stacks results/stacks_qwen2.5_1.5b_holonic_v1.npz
python scripts/stage3.py results/stacks_qwen2.5_1.5b_holonic_v1.npz
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

| 5 | `scripts/stage5_*` | narrative factors (era, voice) as directions: lens, composition, order | both lenses work (1.3/3 vs 2.1 random), era+voice composes (2.0/9), order gap 0.5; replicates on Qwen 0.5B and Pythia 1.4B |
| 6 | `scripts/stage6_factors.py` | N factors: era × voice × tense; era × mood; era × theme (3-sentence passages) | three-way composition 2.8/18 (chance 9.5); mood lens 1.28/3 and composes with era 1.9/9; theme lens 1.25/3 but does not steer generation; cross-talk matrices diagonal |

## Remote models via NDIF

`src/lsx/ndif.py` runs traces on NDIF-hosted models (`nnsight`) through a credential-injecting
egress proxy: the API key header is added by the proxy (the client omits it), and because the proxy
does not carry WebSocket upgrades, jobs are submitted over HTTPS and polled. Requires Python 3.12
(`.venv312`). `python scripts/ndif_smoke.py EleutherAI/gpt-j-6b` round-trips in ~4 s; `ndif_extract.py`, `ndif_factors.py`, `ndif_generate.py` mirror the local pipeline. Only
"pinned" models are available on the free tier (`results/ndif_pinned.txt`); gated ones need an
HF token injected for `huggingface.co`.

## Documents

- `RESULTS.md`: the running log, every number with its control.
- `WRITEUP.md`: the draft writeup.
- `VISION.md`: the operator set, lexicon, coarse-graining mechanism, ecology notes.
- `docs/ALGEBRA.md`: the calculus as a closed many-sorted operator algebra with laws marked measured / hypothesized / conjectured.

## Results

See `RESULTS.md` for the running log, including open problems in priority order.
