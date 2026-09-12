"""NDIF extraction for the time-translation grid on a remote-hosted model (hour 30, Gemma-2-9B-it).

Copies the ndif_extract.py pattern (ProxyAuthBackend, non-blocking submit + poll, per-text
checkpoint, retry_job) but batches BATCH passages per remote job via tracer.invoke, following the
verified-working unrolled pattern in ndif_recompose_gen.py/ndif_recompose_sweep.py (a Python loop
around tracer.invoke is not used elsewhere in this repo, so this stays unrolled rather than risk
an unverified loop form).

Writes a stacks .npz with the SAME key layout scripts/time_translation.py's loader expects:
  "{tag}/{s}/{t}/p{p}"  -> [nL, d]   state-span pooled mean, per requested layer
  "{tag}!{s}/{t}/p{p}"  -> [nL, d]   interval-span pooled mean
tag in {"exp", "ctrl"}. Layers are residual-stream indices in the LM.residuals() convention
(resid[0] = embeddings, resid[i] = output of block i-1); block index used on the wire is l-1, so
layer 0 (embeddings) cannot be requested here -- pass only block-output layers (e.g. 9,20,31).
"""
import argparse, json, os, re, time

import numpy as np
import torch
from nnsight import LanguageModel

from lsx.extract import parse_roles, tokens_in_span
from lsx.ndif import ProxyAuthBackend, retry_job

ap = argparse.ArgumentParser()
ap.add_argument("grid", nargs="?", default="prompts/time_translation_v2.json")
ap.add_argument("--model", default="google/gemma-2-9b-it")
ap.add_argument("--layers", default="9,20,31", help="residual-stream indices (1-indexed vs blocks; 0 = embeddings, not supported here)")
ap.add_argument("--stacks", default=None, help="output .npz path; default derived from model+grid+suffix")
ap.add_argument("--suffix", default="", help="extra tag inserted into the default stacks filename")
ap.add_argument("--batch", type=int, default=6, help="passages per NDIF job (must be 6: unrolled invoke pattern)")
a = ap.parse_args()
assert a.batch == 6, "unrolled invoke pattern below is fixed at 6 per job"

LAYERS = [int(x) for x in a.layers.split(",")]
assert all(l >= 1 for l in LAYERS), "layer 0 (embeddings) is not a block output; use block-output layers only"
BLOCKS = [l - 1 for l in LAYERS]           # nnsight block index for each requested residual-stream layer

G = json.load(open(a.grid))

tag_model = re.sub(r"[^A-Za-z0-9.]+", "_", a.model.split("/")[-1]).lower()
tag_grid = a.grid.split("/")[-1].split(".")[0]
default_stacks = f"results/stacks_{tag_model}_{tag_grid}{('_' + a.suffix) if a.suffix else ''}.npz"
a.stacks = a.stacks or default_stacks
ckpt = a.stacks + ".partial.npz"

model = LanguageModel(a.model, device_map="auto", dispatch=False)
tok = model.tokenizer


def blocks(m):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = m
        try:
            for x in path.split("."):
                obj = getattr(obj, x)
            return obj
        except AttributeError:
            continue


B = blocks(model)

# ------------------------------------------------------------------ build the flat job list
# each item: (key, marked_prompt) where key is "{tag}/{s}/{t}/p{p}"
items = []
for k, marked in G["prompts"].items():
    items.append((f"exp/{k}", marked))
for k, marked in G["control_prompts"].items():
    items.append((f"ctrl/{k}", marked))

out = {}
try:
    _z = np.load(ckpt)
    out = {k: _z[k] for k in _z.files}
    print(f"resuming {len(out)} spans from {ckpt}", flush=True)
except Exception:
    pass

todo = [(k, m) for k, m in items if f"{k}::state" not in out]


def _spans(marked):
    """Parsed text plus token index lists for the 'interval' and 'state' roles."""
    parsed = parse_roles(marked)
    enc = tok(parsed.text, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    idx_state = tokens_in_span(offs, parsed.spans["state"][0])
    idx_interval = tokens_in_span(offs, parsed.spans["interval"][0])
    return parsed.text, idx_interval, idx_state


def _pooled_both(idx_iv, idx_st, block_idx, D):
    """[2, nL, d]: (interval, state) mean-pool of block outputs `block_idx`. Reads each block's
    output exactly once (a single forward pass), in increasing block order -- nnsight requires
    sequential-in-forward-order envoy access within a trace, so interval+state must be pooled
    together per block rather than in two separate passes over `block_idx`."""
    per_layer = []
    for bi in block_idx:
        o = B[bi].output[0]
        iv = o[..., idx_iv, :].mean(-2).reshape(-1, D)[-1]
        st = o[..., idx_st, :].mean(-2).reshape(-1, D)[-1]
        per_layer.append(torch.stack([iv, st]))          # [2, d]
    return torch.stack(per_layer, dim=1)                  # [2, nL, d]


def _inner_batch6(chunk):
    """chunk: list of 6 (key, marked) pairs. One remote job; two .save()'d tensors per invoke
    (interval-pooled [nL,d] and state-pooled [nL,d]), stacked as [2, nL, d] per text."""
    texts, ii, si = [], [], []
    for _, marked in chunk:
        text, idx_iv, idx_st = _spans(marked)
        texts.append(text); ii.append(idx_iv); si.append(idx_st)
    D = model.config.hidden_size
    backend = ProxyAuthBackend(model.to_model_key())
    with model.trace(backend=backend) as tracer:
        with tracer.invoke(texts[0]):
            r0 = _pooled_both(ii[0], si[0], BLOCKS, D).cpu().save()
        with tracer.invoke(texts[1]):
            r1 = _pooled_both(ii[1], si[1], BLOCKS, D).cpu().save()
        with tracer.invoke(texts[2]):
            r2 = _pooled_both(ii[2], si[2], BLOCKS, D).cpu().save()
        with tracer.invoke(texts[3]):
            r3 = _pooled_both(ii[3], si[3], BLOCKS, D).cpu().save()
        with tracer.invoke(texts[4]):
            r4 = _pooled_both(ii[4], si[4], BLOCKS, D).cpu().save()
        with tracer.invoke(texts[5]):
            r5 = _pooled_both(ii[5], si[5], BLOCKS, D).cpu().save()
    res = backend.wait(tracer)
    keys = ("r0", "r1", "r2", "r3", "r4", "r5")
    if not isinstance(res, dict) or any(k not in res for k in keys):
        raise TimeoutError("empty NDIF result for batched extraction job")
    return [res[k].float().cpu().numpy() for k in keys]


t0 = time.time()
n_total = len(items)
n_done = len(out) // 2 if out else 0
for i in range(0, len(todo), 6):
    chunk = todo[i:i + 6]
    if len(chunk) < 6:               # pad the tail with a repeat so the fixed-6 job shape holds
        chunk = chunk + [chunk[-1]] * (6 - len(chunk))
        real_n = len(todo[i:i + 6])
    else:
        real_n = 6
    results = retry_job(lambda: _inner_batch6(chunk), attempts=5, wait_s=10.0)
    for j in range(real_n):
        k, _ = chunk[j]
        r = results[j]                # [2, nL, d]
        out[f"{k}::interval"] = r[0]
        out[f"{k}::state"] = r[1]
    np.savez_compressed(ckpt, **out)
    n_done += real_n
    print(f"[{n_done}/{n_total}] {chunk[0][0]}..{chunk[real_n-1][0]}  {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------ re-key to match time_translation.py's loader
# loader reads z[f"{tag}/{s}/{t}/p{p}"][l] and z[f"{tag}!{s}/{t}/p{p}"][l] (role separator '!' vs '/')
final = {}
for k in [k for k, _ in items]:
    final[k] = out[f"{k}::state"]
    final[k.replace("/", "!", 1)] = out[f"{k}::interval"]
os.makedirs(os.path.dirname(a.stacks) or ".", exist_ok=True)
np.savez_compressed(a.stacks, **final)
print(f"wrote {a.stacks} ({len(final)} arrays) in {time.time()-t0:.0f}s", flush=True)
