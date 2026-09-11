"""Stage 4c (hour 25): refined relation patch test. Relation (true source) vs EVERY wrong source of the same prompt at NATURAL norm (no rescale), plus one random; fit at the given layer with ridge 10; lambda 0.5. Derived from stage4_relation.py.

Original docstring: Stage 4b: relation lens.  For a held-out domain d and role pair S->T:
  role-only patch:   v0 = dir_T                                  (hour-4 lens)
  relation patch:    v1 = dir_T + lam * pred,  pred = W s_d + b   (role-centered affine map fit on the other
                     7 domains x 6 rotations; s_d = d's position-balanced source residual)
  controls:          dir_T + lam * random (same norm as pred), x2
                     dir_T + lam * pred_from_wrong_source (map applied to a different role's residual of d)
Metric: logp(span_T | patch) - logp(span_T | dir_T only)  ("extra gain"), and rank of span_T among
the 6 spans by gain over base.  If the relation carries content, v1 beats the controls.
"""
import argparse, json, re
import numpy as np
from lsx import LM, operate
from lsx.model import Patch
from lsx.steer import add_vector

ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("stacks")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B"); ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--pairs", default="objectified:disturbance,disturbance:embedded,embedded:from_above,from_below:from_above,objectified:new_subject,disturbance:new_subject")
ap.add_argument("--lam", default="0.5"); ap.add_argument("--ridge", type=float, default=10.0); ap.add_argument("--domains", default=None); ap.add_argument("--out", default=None)
a = ap.parse_args()
g = json.load(open(a.grid)); roles = g["roles"]; ri = {r: i for i, r in enumerate(roles)}; pat = re.compile(r"\[\[(\w+):\s*(.*?)\]\]", re.S)
z = np.load(a.stacks); stacks = {k: z[k] for k in z.files if k != "roles"}
doms = sorted({k.split("/")[0] for k in stacks}); R = len(roles); l = a.layer
avg = {d: np.mean([stacks[f"{d}/rot{r}"][l] for r in range(R)], axis=0) for d in doms}   # [6, d]
rot = {d: [stacks[f"{d}/rot{r}"][l] for r in range(R)] for d in doms}                      # 6 x [6, d]
spans, lead = {}, {}
for d in doms:
    p = g["prompts"][f"{d}/rot0"]; lead[d] = p.split(" First,")[0]; spans[d] = {r: t for r, t in pat.findall(p)}
pairs = [tuple(x.split(":")) for x in a.pairs.split(",")]; lams = [float(x) for x in a.lam.split(",")]
test_doms = a.domains.split(",") if a.domains else doms
lm = LM.from_pretrained(a.model); rng = np.random.default_rng(0); res = []
print(f"model={a.model} layer={l} ridge={a.ridge} lams={lams}")
for d in test_doms:
    train = [x for x in doms if x != d]
    mu = np.mean([avg[x] for x in train], axis=0)                     # [6, d] role means (training)
    dirs = mu - mu.mean(0, keepdims=True)
    prefix = f"{lead[d]} First,"
    base = {r: lm.logprob(prefix, f" {spans[d][r]}.") for r in roles}
    for S, T in pairs:
        Si, Ti = ri[S], ri[T]
        Xs = np.stack([rot[x][k][Si] - mu[Si] for x in train for k in range(R)])   # 42 role-centered sources
        Ys = np.stack([rot[x][k][Ti] - mu[Ti] for x in train for k in range(R)])
        op = operate.fit_affine(Xs, Ys, l, S, T, ridge=a.ridge)
        s_d = avg[d][Si] - mu[Si]
        pred = op(s_d); pn = np.linalg.norm(pred)
        role_only = {r: lm.logprob(prefix, f" {spans[d][r]}.", [Patch(l, add_vector(dirs[Ti], 1.0))]) - base[r] for r in roles}
        for lam in lams:
            conds = {"relation": pred}
            for j in range(R):
                if j == Si: continue
                conds[f"wrong_source_{roles[j]}"] = op(avg[d][j] - mu[j])          # natural norm, no rescale
            v = rng.normal(size=pred.shape); conds["rand0"] = v / np.linalg.norm(v) * pn
            for cname, extra in conds.items():
                vec = dirs[Ti] + lam * extra
                gains = {r: lm.logprob(prefix, f" {spans[d][r]}.", [Patch(l, add_vector(vec, 1.0))]) - base[r] for r in roles}
                rank = 1 + sum(gains[r] > gains[T] for r in roles if r != T)
                res.append(dict(domain=d, src=S, dst=T, lam=lam, cond="rand" if cname.startswith("rand") else ("wrong_source" if cname.startswith("wrong") else cname), wrong_role=(cname.split("wrong_source_")[1] if cname.startswith("wrong") else None), norm_ratio=float(np.linalg.norm(extra) / (pn + 1e-9)),
                                extra_gain_T=gains[T] - role_only[T], rank=rank, role_only_rank=1 + sum(role_only[r] > role_only[T] for r in roles if r != T),
                                pred_over_dir=float(pn / np.linalg.norm(dirs[Ti]))))
    sub = [x for x in res if x["domain"] == d]
    for lam in lams:
        f = lambda c: np.mean([x["extra_gain_T"] for x in sub if x["cond"] == c and x["lam"] == lam])
        print(f"{d:12s} lam={lam:3.1f} | extra gain on target span: relation {f('relation'):+.2f}  wrong-source {f('wrong_source'):+.2f}  random {f('rand'):+.2f}  | role-only rank {np.mean([x['role_only_rank'] for x in sub if x['lam']==lam]):.2f}")
print("\n=== summary ===")
for lam in lams:
    for c in ("relation", "wrong_source", "rand"):
        xs = [x for x in res if x["cond"] == c and x["lam"] == lam]
        print(f"lam={lam:3.1f} {c:13s} | extra gain on target {np.mean([x['extra_gain_T'] for x in xs]):+.3f} (n={len(xs)}) | rank {np.mean([x['rank'] for x in xs]):.2f} (role-only {np.mean([x['role_only_rank'] for x in xs]):.2f})")
print("mean |pred|/|dir_T| =", round(float(np.mean([x['pred_over_dir'] for x in res])), 3))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
