"""Fetch per-token residuals at one block for a set of texts from an NDIF model -> results/tokens_<tag>.npz
(keys = text ids; values [n_tokens, d]; plus '<id>__tokens' string arrays)."""
import argparse, json, re, time
import numpy as np, torch
from nnsight import LanguageModel
from lsx.ndif import ProxyAuthBackend, retry_job
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="google/gemma-2-9b-it"); ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--texts", required=True, help="json file: {id: text}"); ap.add_argument("--out", required=True)
a = ap.parse_args()
texts = json.load(open(a.texts)); model = LanguageModel(a.model, device_map="auto", dispatch=False); tok = model.tokenizer
def blocks(m):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = m
        try:
            for x in path.split("."): obj = getattr(obj, x)
            return obj
        except AttributeError: continue
B = blocks(model); D = model.config.hidden_size; out = {}; t0 = time.time()
for i, (k, text) in enumerate(texts.items()):
    backend = ProxyAuthBackend(model.to_model_key())
    with model.trace(text, backend=backend) as tracer:
        h = B[a.layer].output[0].reshape(-1, D).save()
    res = backend.wait(tracer); v = res["h"] if isinstance(res, dict) and "h" in res else next(x for x in res.values() if isinstance(x, torch.Tensor))
    out[k] = v.float().cpu().numpy(); out[k + "__tokens"] = np.array(tok.convert_ids_to_tokens(tok(text)["input_ids"]))
    print(f"[{i+1}/{len(texts)}] {k} {out[k].shape} {time.time()-t0:.0f}s", flush=True)
np.savez_compressed(a.out, **out); print("saved", a.out)
