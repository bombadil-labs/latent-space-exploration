"""Extract span vectors for a factor grid on an NDIF-hosted model. One remote job per span; saves
mean-pooled (or last-token) residuals after every block -> results/stacks_<model>_<grid>.npz, [L, d]
(index i = output of block i = residual index i+1 in the local convention)."""
import argparse, json, re, sys, time
import numpy as np, torch
from nnsight import LanguageModel
from lsx.ndif import ProxyAuthBackend, retry_job
ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("--model", default="EleutherAI/gpt-j-6b"); ap.add_argument("--pool", default="mean")
a = ap.parse_args()
g = json.load(open(a.grid)); lead = g["lead"]
model = LanguageModel(a.model, device_map="auto", dispatch=False); tok = model.tokenizer
def blocks(m):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = m
        try:
            for x in path.split("."): obj = getattr(obj, x)
            return obj
        except AttributeError: continue
B = blocks(model); L = len(B)
tag = re.sub(r"[^A-Za-z0-9.]+", "_", a.model.split("/")[-1]).lower() + "_" + a.grid.split("/")[-1].split(".")[0] + ("" if a.pool == "mean" else "_last")
ckpt = f"results/stacks_{tag}.partial.npz"
out = {}; t0 = time.time()
try:                                   # resume: a busy deployment can time a job out and kill the run
    _z = np.load(ckpt); out = {k: _z[k] for k in _z.files}; print("resuming", len(out), "spans")
except Exception: pass


def _one(text, sel):
    backend = ProxyAuthBackend(model.to_model_key())
    with model.trace(text, backend=backend) as tracer:
        D = model.config.hidden_size
        # tuple-or-tensor block output; .cpu() because a model-parallel host (70B) spreads blocks over GPUs
        hs = torch.stack([B[l].output[0][..., sel, :].mean(-2).reshape(-1, D)[-1].cpu() for l in range(L)]).save()
    res = backend.wait(tracer)
    return res["hs"] if isinstance(res, dict) and "hs" in res else next(x for x in res.values() if isinstance(x, torch.Tensor))


for i, (k, span) in enumerate(g["spans"].items()):
    if k in out: continue
    text = f"{lead} {span}"
    enc = tok(text, return_offsets_mapping=True); offs = enc["offset_mapping"]
    s0 = len(lead) + 1; idx = [j for j, (a_, b_) in enumerate(offs) if b_ > a_ and a_ < len(text) and b_ > s0]
    sel = idx if a.pool == "mean" else idx[-1:]
    out[k] = retry_job(lambda: _one(text, sel), attempts=5, wait_s=10.0).float().cpu().numpy()
    np.savez_compressed(ckpt, **out)
    print(f"[{i+1}/{len(g['spans'])}] {k} {out[k].shape} {time.time()-t0:.0f}s", flush=True)
np.savez_compressed(f"results/stacks_{tag}.npz", **out); print("saved", f"results/stacks_{tag}.npz")
