"""Stage 2 on the position-balanced rotated grid (roles labeled by CONTENT).
  avg:      per-domain stacks averaged over the 6 rotations (position fully balanced), cross-domain RSA
  samerot:  cross-domain pairs with the SAME rotation (position and content aligned)
  diffrot:  cross-domain pairs with DIFFERENT rotations (position decorrelated from content)
  slotlab:  same as diffrot but rows relabeled by SLOT (content decorrelated): the position signal alone
"""
import argparse, itertools, json
import numpy as np
from lsx import compare
ap = argparse.ArgumentParser(); ap.add_argument("stacks"); ap.add_argument("--nperm", type=int, default=200)
ap.add_argument("--step", type=int, default=2); ap.add_argument("--metric", default="rsa"); ap.add_argument("--out", default=None)
a = ap.parse_args()
z = np.load(a.stacks); stacks = compare.subtract_grand_mean({k: z[k] for k in z.files if k != "roles"})
fn = compare.linear_cka if a.metric == "cka" else compare.rsa
doms = sorted({k.split("/")[0] for k in stacks}); R = 6
avg = {d: np.mean([stacks[f"{d}/rot{r}"] for r in range(R)], axis=0) for d in doms}
def by_slot(d, r):  # content index c sits at slot (c - r) mod 6  =>  slot s holds content (s + r) mod 6
    return stacks[f"{d}/rot{r}"][:, [(s + r) % R for s in range(R)]]
B = {"avg (position balanced) by CONTENT": [(avg[x], avg[y]) for x, y in itertools.combinations(doms, 2)],
     "same rotation by CONTENT (=slot)": [(stacks[f"{x}/rot{r}"], stacks[f"{y}/rot{r}"]) for x, y in itertools.combinations(doms, 2) for r in range(R)],
     "diff rotation by CONTENT": [(stacks[f"{x}/rot{r}"], stacks[f"{y}/rot{(r + k) % R}"]) for x, y in itertools.combinations(doms, 2) for r in range(R) for k in (1, 3)],
     "diff rotation by SLOT": [(by_slot(x, r), by_slot(y, (r + k) % R)) for x, y in itertools.combinations(doms, 2) for r in range(R) for k in (1, 3)]}
L = avg[doms[0]].shape[0]
print(f"metric={a.metric} nperm={a.nperm}"); [print(f"  B{i} = {b} (n={len(p)})") for i, (b, p) in enumerate(B.items())]
print("layer | " + " | ".join(f"  B{i} obs  B{i} z" for i in range(len(B))))
res = {b: {"obs": [], "z": []} for b in B}
for l in range(0, L, a.step):
    row = []
    for b, prs in B.items():
        o, zz = [], []
        for i, (X, Y) in enumerate(prs):
            obs, m, s = compare.permutation_null(X[l], Y[l], fn, a.nperm, seed=l * 7919 + i); o.append(obs); zz.append((obs - m) / s)
        res[b]["obs"].append(float(np.mean(o))); res[b]["z"].append(float(np.mean(zz))); row.append(f"{np.mean(o):8.3f} {np.mean(zz):6.2f}")
    print(f"{l:5d} | " + " | ".join(row))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
