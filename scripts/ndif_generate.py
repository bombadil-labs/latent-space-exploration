"""Generate on an NDIF-hosted model with a factor direction added at one block's output on every step."""
import argparse, json, sys, time
import numpy as np, torch
from nnsight import LanguageModel
from lsx.ndif import ProxyAuthBackend
ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("stacks"); ap.add_argument("--model", default="google/gemma-2-9b-it")
ap.add_argument("--layer", type=int, default=20); ap.add_argument("--factor", default="theme"); ap.add_argument("--scales", default="1,2")
ap.add_argument("--tokens", type=int, default=60); ap.add_argument("--chat", action="store_true"); ap.add_argument("--prompts", default=None)
a = ap.parse_args()
g = json.load(open(a.grid)); z = np.load(a.stacks); X = {k: z[k][a.layer] for k in z.files}
F = g["factors"]; names = list(F); fi = names.index(a.factor); mu = np.mean(list(X.values()), axis=0)
dirs = {lvl: np.mean([v for k, v in X.items() if k.split("/")[1 + fi] == lvl], axis=0) - mu for lvl in F[a.factor]}
model = LanguageModel(a.model, device_map="auto", dispatch=False); tok = model.tokenizer
def blocks(m):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = m
        try:
            for x in path.split("."): obj = getattr(obj, x)
            return obj
        except AttributeError: continue
B = blocks(model); l = a.layer
prompts = a.prompts.split("|") if a.prompts else ["A passage from a story: It was late when the news reached her, and", "A passage from a story: The old man opened the box and"]
def fmt(p):
    if not a.chat: return p
    return tok.apply_chat_template([{"role": "user", "content": f"Write the next three sentences of this story passage, continuing in the same voice.\n\n{p.split(': ',1)[-1]}"}], tokenize=False, add_generation_prompt=True)
def gen(prompt, vec=None, scale=1.0):
    backend = ProxyAuthBackend(model.to_model_key())
    v = None if vec is None else torch.as_tensor(vec * scale, dtype=torch.float32)
    with model.generate(prompt, max_new_tokens=a.tokens, do_sample=False, backend=backend) as tracer:
        if v is not None:
            with tracer.all():
                B[l].output[0][:] = B[l].output[0] + v.to(B[l].output[0].dtype)
        out = model.generator.output.save()
    res = backend.wait(tracer)
    o = res["out"] if isinstance(res, dict) and "out" in res else next(x for x in res.values() if isinstance(x, torch.Tensor))
    ids = o[0] if o.dim() == 2 else o
    n_in = len(tok(prompt)["input_ids"])
    return tok.decode(ids[n_in:], skip_special_tokens=True)
for p in prompts:
    fp = fmt(p)
    print(f"\nPROMPT: {p!r}  chat={a.chat} layer={l}\nBASE: {gen(fp)!r}", flush=True)
    for sc in [float(x) for x in a.scales.split(",")]:
        for lvl, d in dirs.items():
            print(f"+{lvl:10s} x{sc}: {gen(fp, d, sc)!r}", flush=True)
