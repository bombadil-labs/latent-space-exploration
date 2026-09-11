"""Content-vs-position test. Shuffled stacks are indexed by SLOT; reorder rows so index = original
CONTENT role (using shuffle_perms). Then cross-domain RSA between content-relabeled shuffled prompts
measures shape that survives position scrambling; holonic~shuffled(content) compares true prompts
to their scrambled twins by content."""
import argparse, itertools, json
import numpy as np
from lsx import compare
ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("stacks"); ap.add_argument("--metric", default="rsa")
ap.add_argument("--nperm", type=int, default=200); ap.add_argument("--step", type=int, default=2); ap.add_argument("--out", default=None)
a = ap.parse_args()
g = json.load(open(a.grid)); perms = g["shuffle_perms"]; roles = g["roles"]
z = np.load(a.stacks); stacks = {k: z[k] for k in z.files if k != "roles"}
stacks = compare.subtract_grand_mean(stacks)
fn = compare.linear_cka if a.metric == "cka" else compare.rsa
# content relabel: slot i holds content role perms[d][i]  =>  content-ordered[c] = slot[ perm^-1[c] ]
cont = {}
for d, perm in perms.items():
    inv = np.argsort(np.array(perm))
    cont[d] = stacks[f"{d}/shuffled"][:, inv]
hol = {d: stacks[f"{d}/holonic"] for d in perms}
slot = {d: stacks[f"{d}/shuffled"] for d in perms}
doms = list(perms)
buckets = {
    "holonic~holonic (content=slot)": [(hol[x], hol[y]) for x, y in itertools.combinations(doms, 2)],
    "shuffled~shuffled by SLOT": [(slot[x], slot[y]) for x, y in itertools.combinations(doms, 2)],
    "shuffled~shuffled by CONTENT": [(cont[x], cont[y]) for x, y in itertools.combinations(doms, 2)],
    "holonic~shuffled by CONTENT": [(hol[x], cont[y]) for x in doms for y in doms if x != y],
    "holonic~shuffled(self) by CONTENT": [(hol[x], cont[x]) for x in doms],
}
L = next(iter(hol.values())).shape[0]
print(f"metric={a.metric} nperm={a.nperm}"); [print(f"  B{i} = {b} (n={len(p)})") for i, (b, p) in enumerate(buckets.items())]
print("layer | " + " | ".join(f"  B{i} obs  B{i} z" for i in range(len(buckets))))
res = {b: {"obs": [], "z": []} for b in buckets}
for l in range(0, L, a.step):
    row = []
    for b, prs in buckets.items():
        o, zz = [], []
        for i, (X, Y) in enumerate(prs):
            obs, m, s = compare.permutation_null(X[l], Y[l], fn, a.nperm, seed=l * 1000 + i); o.append(obs); zz.append((obs - m) / s)
        res[b]["obs"].append(float(np.mean(o))); res[b]["z"].append(float(np.mean(zz))); row.append(f"{np.mean(o):8.3f} {np.mean(zz):6.2f}")
    print(f"{l:5d} | " + " | ".join(row))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
