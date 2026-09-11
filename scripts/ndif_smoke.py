"""Smoke test on NDIF through the proxy: one remote forward pass, mid-layer residual + next-token logits."""
import sys, time, torch
from nnsight import LanguageModel
from lsx.ndif import ProxyAuthBackend
name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B"
t = time.time()
model = LanguageModel(name, device_map="auto", dispatch=False)
def blocks(m):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = m
        try:
            for a in path.split("."): obj = getattr(obj, a)
            return obj
        except AttributeError: continue
    raise ValueError("no blocks")
B = blocks(model); mid = B[len(B) // 2]
print(f"{name}: config+tokenizer {time.time()-t:.1f}s; layers={len(B)} d={model.config.hidden_size}")
t = time.time()
backend = ProxyAuthBackend(model.to_model_key())
with model.trace("A moment from a story: The door opened and", backend=backend) as tracer:
    h = mid.output[0][0, -1, :].save()
    logits = model.lm_head.output[0, -1, :].save()
print(f"submitted in {time.time()-t:.1f}s; polling...")
res = backend.wait(tracer)
print(f"completed in {time.time()-t:.1f}s; result type {type(res).__name__}", (list(res.keys())[:6] if isinstance(res, dict) else ""))
import torch as _t
def _get(x):
    if isinstance(res, dict):
        for k, v in res.items():
            if isinstance(v, _t.Tensor) and v.shape == getattr(x, "shape", None): return v
        vals = [v for v in res.values() if isinstance(v, _t.Tensor)]
        return vals
    return res
vals = [v for v in res.values() if isinstance(v, _t.Tensor)] if isinstance(res, dict) else []
print("tensors in result:", [tuple(v.shape) for v in vals])
h, logits = (vals[0], vals[1]) if len(vals) >= 2 else (h, logits)
v = h.detach().float()
print("mid-layer resid, last token: shape", tuple(v.shape), "norm", round(float(v.norm()), 2))
print("next-token top5:", [model.tokenizer.decode([i]) for i in torch.topk(logits.detach().float(), 5).indices])
