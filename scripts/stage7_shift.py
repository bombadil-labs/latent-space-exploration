"""Stage 7: move the address, keep the form.  For each theme passage (era e1, theme t), patch an era
shift (dir_era[e2] - dir_era[e1]) at layer L at every position while the model READS the passage, and
read the representation out (mean over span tokens at layer L_read):
   era readout:   nearest era direction among 3      -> should become e2   (address moved)
   theme readout: nearest theme direction among 3    -> should stay t      (form kept)
Directions are leave-one-situation-out. Controls: no patch (base) and a random direction of equal norm.
"""
import argparse, json, itertools
import numpy as np, torch
from lsx import LM, extract
from lsx.model import Patch
from lsx.steer import add_vector
ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("stacks"); ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--layer", type=int, default=14); ap.add_argument("--read", type=int, default=20); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--out", default=None)
a = ap.parse_args()
g = json.load(open(a.grid)); z = np.load(a.stacks); X = {k: z[k] for k in z.files}
E, T = g["factors"]["era"], g["factors"]["theme"]; S = g["scenes"]; lead = g["lead"]; spans = g["spans"]; key = lambda s, e, t: f"{s}/{e}/{t}"
def dirs(l, train):
    allv = np.stack([X[key(s, e, t)][l] for s in train for e in E for t in T]); mu = allv.mean(0)
    de = {e: np.mean([X[key(s, e, t)][l] for s in train for t in T], axis=0) - mu for e in E}
    dt = {t: np.mean([X[key(s, e, t)][l] for s in train for e in E], axis=0) - mu for t in T}
    return de, dt, mu
cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
lm = LM.from_pretrained(a.model); rng = np.random.default_rng(0); rows = []
for s in S:
    train = [x for x in S if x != s]
    deP, _, _ = dirs(a.layer, train)                      # patch directions at the patch layer
    deR, dtR, muR = dirs(a.read, train)                   # readout directions at the read layer
    for e1 in E:
        for t in T:
            marked = f"{lead} [[span: {spans[key(s, e1, t)]}]]"
            def readout(patches):
                v = extract(lm, marked, keep_resid=False).roles["span"][a.read] if not patches else None
                if patches:
                    with lm.patched(patches):
                        v = extract(lm, marked, keep_resid=False).roles["span"][a.read]
                v = v - muR
                return max(E, key=lambda e: cos(v, deR[e])), max(T, key=lambda tt: cos(v, dtR[tt]))
            e_b, t_b = readout([])
            rows.append(dict(scene=s, e1=e1, t=t, e2=e1, cond="base", era_read=e_b, theme_read=t_b))
            for e2 in E:
                if e2 == e1: continue
                shift = deP[e2] - deP[e1]
                r = rng.normal(size=shift.shape); r *= np.linalg.norm(shift) / np.linalg.norm(r)
                for cond, vec in (("shift", shift), ("rand", r)):
                    e_r, t_r = readout([Patch(a.layer, add_vector(vec, a.scale))])
                    rows.append(dict(scene=s, e1=e1, t=t, e2=e2, cond=cond, era_read=e_r, theme_read=t_r))
    print("scene", s, "done", flush=True)
def acc(cond, f):
    xs = [x for x in rows if x["cond"] == cond]; return np.mean([f(x) for x in xs]), len(xs)
print(f"\n=== {a.model} patch@{a.layer} read@{a.read} scale={a.scale} ===")
print(f"base : era read = e1 {acc('base', lambda x: x['era_read']==x['e1'])[0]:.2f} | theme read = t {acc('base', lambda x: x['theme_read']==x['t'])[0]:.2f}")
for c in ("shift", "rand"):
    print(f"{c:5s}: era read = e2 (moved) {acc(c, lambda x: x['era_read']==x['e2'])[0]:.2f} | era read = e1 (stayed) {acc(c, lambda x: x['era_read']==x['e1'])[0]:.2f} | theme read = t (kept) {acc(c, lambda x: x['theme_read']==x['t'])[0]:.2f}   (n={acc(c, lambda x: 1)[1]})")
if a.out: json.dump(rows, open(a.out, "w"), indent=1)
