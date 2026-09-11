"""Stage 3: can an affine operator carry a role->role relation to a held-out domain?

For each ordered role pair (src -> dst) and each layer: fit on the holonic prompts of all domains
but one, predict the held-out domain's dst vector from its src vector.  Report cosine(pred, true)
against baselines: identity (pred = src), mean-target, and a single shared offset (king-queen).
Rank = where the true target lands among all domains' dst vectors by cosine to the prediction (1 = best).
usage: python scripts/stage3.py results/stacks_qwen2.5_1.5b_holonic_v1.npz [--pairs embedded:objectified,...] [--rank 4]
"""
import argparse, itertools, json
import numpy as np
from lsx import operate, compare

ap = argparse.ArgumentParser(); ap.add_argument("stacks"); ap.add_argument("--pairs", default=None)
ap.add_argument("--rank", type=int, default=4); ap.add_argument("--ridge", type=float, default=1.0)
ap.add_argument("--step", type=int, default=2); ap.add_argument("--framing", default="holonic"); ap.add_argument("--out", default=None)
ap.add_argument("--null-seed", type=int, default=None, help="permute dst rows within training folds (breaks src->dst pairing) to get a null")
ap.add_argument("--role-center", action="store_true", help="remove per-role cross-domain mean (train folds) before fitting/ranking")
ap.add_argument("--content-relabel", default=None, help="grid json with shuffle_perms; reorders --framing shuffled stacks so index = content role")
a = ap.parse_args()
z = np.load(a.stacks); roles = list(z["roles"]); stacks = {k: z[k] for k in z.files if k != "roles"}
stacks = compare.subtract_grand_mean(stacks)
keys = [k for k in stacks if k.endswith("/" + a.framing)] if a.framing != "all" else list(stacks)
if a.content_relabel:
    perms = json.load(open(a.content_relabel))["shuffle_perms"]
    for k in keys:
        inv = np.argsort(np.array(perms[k.split("/")[0]])); stacks[k] = stacks[k][:, inv]
groups = np.array([k.split("/")[0] for k in keys])
L = next(iter(stacks.values())).shape[0]
pairs = [tuple(p.split(":")) for p in a.pairs.split(",")] if a.pairs else list(itertools.permutations(roles, 2))
ri = {r: i for i, r in enumerate(roles)}
print(f"stacks={a.stacks} framing={a.framing} domains={len(keys)} low_rank={a.rank} ridge={a.ridge}")
print(f"{'src->dst':28s} layer | cos_pred cos_offset cos_ident cos_mean | role_rank: pred offset ident mean (of {len(roles)}) role_center={a.role_center}")
res = {}
for src, dst in pairs:
    best = None
    for l in range(0, L, a.step):
        S = np.stack([stacks[k][l, ri[src]] for k in keys]); O = np.stack([stacks[k][l, ri[dst]] for k in keys])
        C = np.stack([stacks[k][l] for k in keys])           # [n, m_roles, d]
        ev = operate.holdout_eval(S, O, groups, l, src, dst, cands=C, dst_idx=ri[dst], src_idx=ri[src], role_center=a.role_center, null_seed=a.null_seed, ridge=a.ridge, low_rank=(None if a.rank <= 0 else a.rank))
        res.setdefault(f"{src}->{dst}", {})[l] = ev
        if best is None or ev["cos_pred"] > best[1]["cos_pred"]:
            best = (l, ev)
    l, ev = best
    print(f"{src+'->'+dst:28s} {l:5d} | {ev['cos_pred']:8.3f} {ev['cos_offset']:10.3f} {ev['cos_identity']:9.3f} {ev['cos_mean']:8.3f} | {ev['role_rank']:10.1f} {ev['role_rank_offset']:6.1f} {ev['role_rank_identity']:5.1f} {ev['role_rank_mean']:4.1f}")
summ = {k: float(np.mean([res[p][l][k] for p in res for l in res[p]])) for k in ("role_rank", "role_rank_offset", "role_rank_identity", "role_rank_mean")}
best_l = {k: float(np.mean([min(res[p].values(), key=lambda e: e[k])[k] for p in res])) for k in ("role_rank", "role_rank_offset", "role_rank_identity", "role_rank_mean")}
print("mean role_rank over all pairs & layers:", {k: round(v, 2) for k, v in summ.items()})
print("mean role_rank at each pair's best layer:", {k: round(v, 2) for k, v in best_l.items()})
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
