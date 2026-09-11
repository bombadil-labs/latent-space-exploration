"""Layer sweep of cross-prompt shape similarity over a prompt grid.

usage: python scripts/sweep.py prompts/dialectic_v0.json [--model Qwen/Qwen2.5-0.5B] [--metric cka|rsa]
Prints, per layer, mean similarity for: same framing across domains (the hypothesis),
same domain across framings (the address control), and different domain + different framing (floor).
"""
import argparse, json, itertools, os
import numpy as np
from lsx import LM, extract, compare

ap = argparse.ArgumentParser()
ap.add_argument("grid"); ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B"); ap.add_argument("--metric", default="cka")
ap.add_argument("--step", type=int, default=2)
a = ap.parse_args()
grid = json.load(open(a.grid)); roles = grid["roles"]
lm = LM.from_pretrained(a.model)
fn = compare.linear_cka if a.metric == "cka" else compare.rsa
stacks = {k: compare.role_stack(extract(lm, p, keep_resid=False), roles) for k, p in grid["prompts"].items()}
keys = list(stacks); dom = {k: k.split("/")[0] for k in keys}; fr = {k: k.split("/")[1] for k in keys}
buckets = {"same framing, diff domain": [], "same domain, diff framing": [], "diff both": []}
for x, y in itertools.combinations(keys, 2):
    if fr[x] == fr[y] and dom[x] != dom[y]: buckets["same framing, diff domain"].append((x, y))
    elif dom[x] == dom[y] and fr[x] != fr[y]: buckets["same domain, diff framing"].append((x, y))
    elif dom[x] != dom[y] and fr[x] != fr[y]: buckets["diff both"].append((x, y))
L = next(iter(stacks.values())).shape[0]
print(f"model={a.model} metric={a.metric} roles={roles}")
print("layer | " + " | ".join(f"{b:>26}" for b in buckets))
for l in range(0, L, a.step):
    row = [np.mean([fn(stacks[x][l], stacks[y][l]) for x, y in prs]) for prs in buckets.values()]
    print(f"{l:5d} | " + " | ".join(f"{v:26.3f}" for v in row))
