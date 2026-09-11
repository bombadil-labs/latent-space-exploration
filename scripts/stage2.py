"""Stage 2: does a cross-domain relational shape exist above chance?

For every pair of prompts, compute shape similarity at each layer AND a permutation null
(same vectors, scrambled role correspondence). Report per-bucket mean z-scores. Buckets:
  H  same framing (holonic), different domain   <- the hypothesis
  F  same framing (flat),    different domain   <- template/syntax control: flat prompts share connectives too
  A  same domain, different framing             <- address control
  X  different domain, different framing        <- floor
usage: python scripts/stage2.py prompts/holonic_v1.json --model Qwen/Qwen2.5-1.5B [--metric cka|rsa] [--nperm 200]
"""
import argparse, itertools, json, os, sys
import numpy as np
from lsx import LM, extract, compare

ap = argparse.ArgumentParser()
ap.add_argument("grid"); ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--metric", default="cka"); ap.add_argument("--nperm", type=int, default=200)
ap.add_argument("--step", type=int, default=2); ap.add_argument("--no-center", action="store_true")
ap.add_argument("--out", default=None); ap.add_argument("--stacks", default=None, help="cached npz from extract_grid.py")
a = ap.parse_args()
grid = json.load(open(a.grid)); roles = grid["roles"]
fn = compare.linear_cka if a.metric == "cka" else compare.rsa
ap_stacks = a.stacks
if ap_stacks:
    z = np.load(ap_stacks); stacks = {k: z[k] for k in z.files if k != "roles"}
else:
    lm = LM.from_pretrained(a.model)
    stacks = {k: compare.role_stack(extract(lm, p, keep_resid=False), roles) for k, p in grid["prompts"].items()}
if not a.no_center:
    stacks = compare.subtract_grand_mean(stacks)
keys = list(stacks); dom = {k: k.split("/")[0] for k in keys}; fr = {k: k.split("/")[1] for k in keys}
def bucket(x, y):
    if dom[x] == dom[y]: return "A" if fr[x] != fr[y] else None
    if fr[x] == fr[y]: return "H" if fr[x] == "holonic" else "F"
    return "X"
pairs = {b: [] for b in "HFAX"}
for x, y in itertools.combinations(keys, 2):
    b = bucket(x, y)
    if b: pairs[b].append((x, y))
L = next(iter(stacks.values())).shape[0]
layers = list(range(0, L, a.step))
res = {b: {"obs": [], "z": []} for b in pairs}
print(f"model={a.model} metric={a.metric} nperm={a.nperm} centered={not a.no_center} roles={len(roles)}")
print(f"pairs: " + ", ".join(f"{b}={len(p)}" for b, p in pairs.items()))
print("layer |   H obs    H z |   F obs    F z |   A obs    A z |   X obs    X z")
for l in layers:
    row = []
    for b, prs in pairs.items():
        o, z = [], []
        for i, (x, y) in enumerate(prs):
            obs, m, s = compare.permutation_null(stacks[x][l], stacks[y][l], fn, a.nperm, seed=l * 1000 + i)
            o.append(obs); z.append((obs - m) / s)
        res[b]["obs"].append(float(np.mean(o))); res[b]["z"].append(float(np.mean(z)))
        row.append(f"{np.mean(o):7.3f} {np.mean(z):6.2f}")
    print(f"{l:5d} | " + " | ".join(row))
if a.out:
    json.dump({"model": a.model, "metric": a.metric, "layers": layers, "buckets": res, "n_pairs": {b: len(p) for b, p in pairs.items()}}, open(a.out, "w"), indent=1)
