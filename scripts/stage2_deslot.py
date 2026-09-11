"""Project out the discourse-slot subspace, then re-test cross-domain shape by CONTENT.
Slot subspace at each layer = span of the 6 per-slot mean vectors over the shuffled prompts of the
domains NOT in the pair being compared (content is decorrelated from slot there). Then RSA on
content-labeled shuffled stacks and on holonic stacks, with permutation nulls."""
import argparse, itertools, json
import numpy as np
from lsx import compare
ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("stacks")
ap.add_argument("--nperm", type=int, default=200); ap.add_argument("--step", type=int, default=2); ap.add_argument("--out", default=None)
ap.add_argument("--extra-pcs", type=int, default=0, help="also remove this many top PCs of the pooled shuffled set")
a = ap.parse_args()
g = json.load(open(a.grid)); perms = g["shuffle_perms"]; doms = list(perms)
z = np.load(a.stacks); stacks = compare.subtract_grand_mean({k: z[k] for k in z.files if k != "roles"})
slot = {d: stacks[f"{d}/shuffled"] for d in doms}
cont = {d: slot[d][:, np.argsort(np.array(perms[d]))] for d in doms}
hol = {d: stacks[f"{d}/holonic"] for d in doms}
L = slot[doms[0]].shape[0]

def projector(l, exclude):
    others = [d for d in doms if d not in exclude]
    M = np.mean([slot[d][l] for d in others], axis=0)            # [6, d] slot means
    B = M - M.mean(0)
    if a.extra_pcs:
        pool = np.concatenate([slot[d][l] for d in others]); pool -= pool.mean(0)
        _, _, Vt = np.linalg.svd(pool, full_matrices=False); B = np.vstack([B, Vt[:a.extra_pcs]])
    Q, _ = np.linalg.qr(B.T)                                       # orthonormal basis of slot subspace
    return np.eye(Q.shape[0]) - Q @ Q.T

buckets = {"holonic~holonic": [(hol, x, y) for x, y in itertools.combinations(doms, 2)],
           "shuffled~shuffled by SLOT": [(slot, x, y) for x, y in itertools.combinations(doms, 2)],
           "shuffled~shuffled by CONTENT": [(cont, x, y) for x, y in itertools.combinations(doms, 2)],
           "holonic~shuffled by CONTENT": [((hol, cont), x, y) for x in doms for y in doms if x != y]}
print(f"deslot: leave-pair-out slot subspace (5 dims) + {a.extra_pcs} PCs; nperm={a.nperm}")
[print(f"  B{i} = {b} (n={len(p)})") for i, (b, p) in enumerate(buckets.items())]
print("layer | " + " | ".join(f"  B{i} obs  B{i} z" for i in range(len(buckets))))
res = {b: {"obs": [], "z": []} for b in buckets}
for l in range(0, L, a.step):
    row = []
    for b, prs in buckets.items():
        o, zz = [], []
        for i, (src, x, y) in enumerate(prs):
            P = projector(l, {x, y})
            sx, sy = (src if isinstance(src, dict) else src[0]), (src if isinstance(src, dict) else src[1])
            X, Y = sx[x][l] @ P, sy[y][l] @ P
            obs, m, s = compare.permutation_null(X, Y, compare.rsa, a.nperm, seed=l * 1000 + i); o.append(obs); zz.append((obs - m) / s)
        res[b]["obs"].append(float(np.mean(o))); res[b]["z"].append(float(np.mean(zz))); row.append(f"{np.mean(o):8.3f} {np.mean(zz):6.2f}")
    print(f"{l:5d} | " + " | ".join(row))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
