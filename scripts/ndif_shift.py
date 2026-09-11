"""Stage 7 on an NDIF model: era shift while reading a theme passage; read out era and theme at a layer.
One remote job per (passage, condition): patch dir_era[e2]-dir_era[e1] at block `layer` (all positions),
save the mean span vector at block `read`. Classification (nearest direction) is done locally."""
import argparse, json, time
import numpy as np, torch
from nnsight import LanguageModel
from lsx.ndif import ProxyAuthBackend, retry_job
ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("stacks"); ap.add_argument("--model", default="google/gemma-2-9b-it")
ap.add_argument("--layer", type=int, default=14); ap.add_argument("--read", type=int, default=20); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--out", default=None)
a = ap.parse_args()
g = json.load(open(a.grid)); z = np.load(a.stacks); X = {k: z[k] for k in z.files}
E, T = g["factors"]["era"], g["factors"]["theme"]; S = g["scenes"]; lead = g["lead"]; spans = g["spans"]; key = lambda s, e, t: f"{s}/{e}/{t}"
model = LanguageModel(a.model, device_map="auto", dispatch=False); tok = model.tokenizer
def blocks(m):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = m
        try:
            for x in path.split("."): obj = getattr(obj, x)
            return obj
        except AttributeError: continue
B = blocks(model); D = model.config.hidden_size
def dirs(l, train):
    allv = np.stack([X[key(s, e, t)][l] for s in train for e in E for t in T]); mu = allv.mean(0)
    return ({e: np.mean([X[key(s, e, t)][l] for s in train for t in T], axis=0) - mu for e in E},
            {t: np.mean([X[key(s, e, t)][l] for s in train for e in E], axis=0) - mu for t in T}, mu)
cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
def _inner_read_span(text, vec=None):
    enc = tok(text, return_offsets_mapping=True); offs = enc["offset_mapping"]; s0 = len(lead) + 1
    sel = [j for j, (x, y) in enumerate(offs) if y > x and y > s0]
    v = None if vec is None else torch.as_tensor(vec * a.scale, dtype=torch.float32)
    backend = ProxyAuthBackend(model.to_model_key())
    with model.trace(text, backend=backend) as tracer:
        if v is not None:
            B[a.layer].output[0][:] = B[a.layer].output[0] + v.to(B[a.layer].output[0])
        h = B[a.read].output[0][..., sel, :].mean(-2).reshape(-1, D)[-1].save()
    res = backend.wait(tracer); o = res["h"] if isinstance(res, dict) and "h" in res else next(x for x in res.values() if isinstance(x, torch.Tensor))
    return o.float().cpu().numpy()
read_span = lambda *a_, **k_: retry_job(lambda: _inner_read_span(*a_, **k_))
rng = np.random.default_rng(0); rows = []; t0 = time.time()
ckpt = (a.out or "results/ndif_shift_ckpt.json") + ".partial"
try: rows = json.load(open(ckpt)); print("resuming", len(rows))
except Exception: pass
done = {(r["scene"], r["e1"], r["t"], r["e2"], r["cond"]) for r in rows}
for s in S:
    train = [x for x in S if x != s]; deP, _, _ = dirs(a.layer, train); deR, dtR, muR = dirs(a.read, train)
    def classify(v):
        v = v - muR; return max(E, key=lambda e: cos(v, deR[e])), max(T, key=lambda tt: cos(v, dtR[tt]))
    for e1 in E:
        for t in T:
            text = f"{lead} {spans[key(s, e1, t)]}"
            conds = [("base", e1, None)]
            for e2 in E:
                if e2 == e1: continue
                shift = deP[e2] - deP[e1]; r = rng.normal(size=shift.shape); r *= np.linalg.norm(shift) / np.linalg.norm(r)
                conds += [("shift", e2, shift), ("rand", e2, r)]
            for cond, e2, vec in conds:
                if (s, e1, t, e2, cond) in done: continue
                er, tr = classify(read_span(text, vec))
                rows.append(dict(scene=s, e1=e1, t=t, e2=e2, cond=cond, era_read=er, theme_read=tr))
    json.dump(rows, open(ckpt, "w")); print(f"scene {s} done {time.time()-t0:.0f}s", flush=True)
def acc(cond, f):
    xs = [x for x in rows if x["cond"] == cond]; return np.mean([f(x) for x in xs]), len(xs)
print(f"\n=== {a.model} patch@{a.layer} read@{a.read} scale={a.scale} ===")
print(f"base : era read = e1 {acc('base', lambda x: x['era_read']==x['e1'])[0]:.2f} | theme read = t {acc('base', lambda x: x['theme_read']==x['t'])[0]:.2f}")
for c in ("shift", "rand"):
    print(f"{c:5s}: era read = e2 (moved) {acc(c, lambda x: x['era_read']==x['e2'])[0]:.2f} | era read = e1 (stayed) {acc(c, lambda x: x['era_read']==x['e1'])[0]:.2f} | theme read = t (kept) {acc(c, lambda x: x['theme_read']==x['t'])[0]:.2f}   (n={acc(c, lambda x: 1)[1]})")
if a.out: json.dump(rows, open(a.out, "w"), indent=1)

