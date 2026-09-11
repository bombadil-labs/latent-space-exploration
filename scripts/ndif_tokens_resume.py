"""Resumable variant of scripts/ndif_tokens.py: per-token residuals at one block for a set of texts.

Differences from ndif_tokens.py: each text's job is wrapped in retry_job (NDIF jobs hang often enough
that a 73-text run fails outright), and results are checkpointed to <out> after every text, so a
re-run skips what is already saved.
"""
import argparse, json, os, time
import numpy as np, torch
from nnsight import LanguageModel
from lsx.ndif import ProxyAuthBackend, retry_job

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="google/gemma-2-9b-it")
ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--texts", required=True, help="json file: {id: text}")
ap.add_argument("--out", required=True)
a = ap.parse_args()

texts = json.load(open(a.texts))
model = LanguageModel(a.model, device_map="auto", dispatch=False); tok = model.tokenizer


def blocks(m):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = m
        try:
            for x in path.split("."):
                obj = getattr(obj, x)
            return obj
        except AttributeError:
            continue


B = blocks(model); D = model.config.hidden_size
out = {}
if os.path.exists(a.out):
    z = np.load(a.out); out = {k: z[k] for k in z.files}
    print(f"resuming: {len([k for k in out if not k.endswith('__tokens')])} already saved", flush=True)
t0 = time.time()
for i, (k, text) in enumerate(texts.items()):
    if k in out:
        continue

    def job():
        backend = ProxyAuthBackend(model.to_model_key())
        with model.trace(text, backend=backend) as tracer:
            h = B[a.layer].output[0].reshape(-1, D).save()
        res = backend.wait(tracer)
        return res["h"] if isinstance(res, dict) and "h" in res else next(x for x in res.values() if isinstance(x, torch.Tensor))

    v = retry_job(job, attempts=4)
    out[k] = v.float().cpu().numpy()
    out[k + "__tokens"] = np.array(tok.convert_ids_to_tokens(tok(text)["input_ids"]))
    np.savez_compressed(a.out, **out)
    print(f"[{i+1}/{len(texts)}] {k} {out[k].shape} {time.time()-t0:.0f}s", flush=True)
print("saved", a.out)
