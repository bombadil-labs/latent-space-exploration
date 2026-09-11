"""Stage 6: N narrative factors. Grid keys scene/<f1>/<f2>/... ; leave one scene out.
(A) decodability per factor per layer (no model).
(B) each factor as a lens: rank of the patched level among that factor's levels, other factors fixed (chance (k+1)/2).
(D) all-factor composition: rank of the joint span among ALL variants of the scene (chance (N+1)/2).
(X) pairwise cross-talk matrix: under +factor_i patch, fraction of gain variance explained by each factor's main effect.
"""
import argparse, json, itertools
import numpy as np
from lsx import LM
from lsx.model import Patch
from lsx.steer import add_vector
ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("stacks"); ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--layer", type=int, default=14); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--out", default=None); ap.add_argument("--skip-model", action="store_true")
a = ap.parse_args()
g = json.load(open(a.grid)); z = np.load(a.stacks); X = {k: z[k] for k in z.files}
F = g["factors"]; names = list(F); S = g["scenes"]; lead = g["lead"]; spans = g["spans"]
combos = list(itertools.product(*[F[n] for n in names]))
key = lambda s, c: "/".join([s, *c])
def dirs(l, train):
    allv = {c: np.stack([X[key(s, c)][l] for s in train]).mean(0) for c in combos}; mu = np.mean(list(allv.values()), axis=0)
    return {n: {lvl: np.mean([allv[c] for c in combos if c[i] == lvl], axis=0) - mu for lvl in F[n]} for i, n in enumerate(names)}, mu
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
L = next(iter(X.values())).shape[0]
print("(A) leave-one-scene-out decodability (chance " + ", ".join(f"{n} 1/{len(F[n])}" for n in names) + ")")
for l in range(0, L, 4):
    acc = {n: 0 for n in names}; tot = 0
    for s in S:
        D, mu = dirs(l, [x for x in S if x != s])
        for c in combos:
            x = X[key(s, c)][l] - mu; tot += 1
            for i, n in enumerate(names): acc[n] += max(F[n], key=lambda v: cos(x, D[n][v])) == c[i]
    print(f"  layer {l:2d}: " + "  ".join(f"{n} {acc[n]/tot:.2f}" for n in names))
if a.skip_model: raise SystemExit
lm = LM.from_pretrained(a.model); rng = np.random.default_rng(0); l = a.layer; res = []
def decompose(gd):  # gd: combo -> gain ; returns fraction of variance per factor main effect
    G = np.array([gd[c] for c in combos]); gm = G.mean(); tot = ((G - gm) ** 2).sum() + 1e-12; out = {}
    for i, n in enumerate(names):
        means = {lvl: np.mean([gd[c] for c in combos if c[i] == lvl]) for lvl in F[n]}
        out[n] = float(sum((means[c[i]] - gm) ** 2 for c in combos) / tot)
    return out
for s in S:
    D, mu = dirs(l, [x for x in S if x != s])
    base = {c: lm.logprob(lead, f" {spans[key(s, c)]}") for c in combos}
    gains = lambda vec: {c: lm.logprob(lead, f" {spans[key(s, c)]}", [Patch(l, add_vector(vec, a.scale))]) - base[c] for c in combos}
    rank = lambda gd, t, cands: 1 + sum(gd[c] > gd[t] for c in cands if c != t)
    for i, n in enumerate(names):
        for lvl in F[n]:
            r = rng.normal(size=D[n][lvl].shape); r *= np.linalg.norm(D[n][lvl]) / np.linalg.norm(r)
            for cond, vec in (("factor", D[n][lvl]), ("rand", r)):
                gd = gains(vec); dec = decompose(gd)
                for c in combos:
                    if c[i] == lvl:
                        res.append(dict(scene=s, test="B", factor=n, cond=cond, rank=rank(gd, c, [cc for cc in combos if all(cc[j] == c[j] for j in range(len(names)) if j != i)])))
                res.append(dict(scene=s, test="X", factor=n, cond=cond, **{f"frac_{m}": dec[m] for m in names}))
    for c in combos:
        gd = gains(sum(D[n][c[i]] for i, n in enumerate(names))); res.append(dict(scene=s, test="D", rank=rank(gd, c, combos)))
    print("scene", s, "done")
print(f"\n=== summary layer={l} scale={a.scale} ===")
for n in names:
    k = len(F[n]); f = lambda c: np.mean([x["rank"] for x in res if x["test"] == "B" and x["factor"] == n and x["cond"] == c])
    print(f"(B) {n:6s} lens: rank/{k} factor-dir {f('factor'):.2f}  random {f('rand'):.2f}   (chance {(k+1)/2:.1f})")
print(f"(D) all {len(names)} factors composed: rank/{len(combos)} {np.mean([x['rank'] for x in res if x['test']=='D']):.2f}   (chance {(len(combos)+1)/2:.1f})")
print("(X) cross-talk: rows = patched factor, cols = fraction of gain variance explained by each factor")
print("        " + "".join(f"{m:>9s}" for m in names))
for n in names + ["rand"]:
    xs = [x for x in res if x["test"] == "X" and ((x["factor"] == n and x["cond"] == "factor") if n != "rand" else x["cond"] == "rand")]
    print(f"{n:>7s} " + "".join(f"{np.mean([x[f'frac_{m}'] for x in xs]):9.2f}" for m in names))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
