"""Stage 4: is the role-identity direction a lens?  Leave one domain out. Build the role direction
dir_R(l) = mean_train(role R) - mean_train(all roles) from the position-balanced rotated stacks.
Add s*dir_R to the residual at layers L during a teacher-forced pass over
    "<lead> First, <span_r>."   for each of the held-out domain's 6 spans.
gain_r = logp(span_r | +dir_R) - logp(span_r | base).  If dir_R is a lens, gain is largest for r = R.
Report rank of gain_R among the 6 (1 = best, chance 3.5), against random directions of equal norm.
"""
import argparse, json, re
import numpy as np, torch
from lsx import LM, compare
from lsx.model import Patch
from lsx.steer import add_vector

ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("stacks")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B"); ap.add_argument("--layers", default="20")
ap.add_argument("--scales", default="1,2,4"); ap.add_argument("--nrand", type=int, default=2)
ap.add_argument("--domains", default=None); ap.add_argument("--out", default=None)
a = ap.parse_args()
g = json.load(open(a.grid)); roles = g["roles"]; pat = re.compile(r"\[\[(\w+):\s*(.*?)\]\]", re.S)
z = np.load(a.stacks); stacks = {k: z[k] for k in z.files if k != "roles"}
doms = sorted({k.split("/")[0] for k in stacks}); R = len(roles)
avg = {d: np.mean([stacks[f"{d}/rot{r}"] for r in range(R)], axis=0) for d in doms}   # [L+1, 6, d]
spans = {}; lead = {}
for d in doms:
    p = g["prompts"][f"{d}/rot0"]; lead[d] = p.split(" First,")[0]
    spans[d] = {r: t for r, t in pat.findall(p)}
layers = [int(x) for x in a.layers.split(",")]; scales = [float(x) for x in a.scales.split(",")]
test_doms = a.domains.split(",") if a.domains else doms
lm = LM.from_pretrained(a.model)
rng = np.random.default_rng(0)
res = []
print(f"model={a.model} layers={layers} scales={scales} nrand={a.nrand}")
for d in test_doms:
    train = [x for x in doms if x != d]
    M = np.mean([avg[x] for x in train], axis=0)                 # [L+1, 6, d] role means over training domains
    dirs = {l: M[l] - M[l].mean(0, keepdims=True) for l in layers}  # role-identity directions per layer
    prefix = f"{lead[d]} First,"
    base = {r: lm.logprob(prefix, f" {spans[d][r]}.") for r in roles}
    resid_norm = float(np.linalg.norm(avg[d][layers[0]], axis=1).mean())
    for Ri, Rname in enumerate(roles):
        dnorm = {l: float(np.linalg.norm(dirs[l][Ri])) for l in layers}
        for s in scales:
            conds = {"role": {l: dirs[l][Ri] for l in layers}}
            for k in range(a.nrand):
                conds[f"rand{k}"] = {l: (v := rng.normal(size=dirs[l].shape[1])) / np.linalg.norm(v) * dnorm[l] for l in layers}
            for cname, vecs in conds.items():
                patches = [Patch(l, add_vector(vecs[l], s)) for l in layers]
                gains = {r: lm.logprob(prefix, f" {spans[d][r]}.", patches) - base[r] for r in roles}
                rank = 1 + sum(gains[r] > gains[Rname] for r in roles if r != Rname)
                res.append(dict(domain=d, role=Rname, scale=s, cond="role" if cname == "role" else "rand", rank=rank,
                                gain_R=gains[Rname], gain_other=float(np.mean([gains[r] for r in roles if r != Rname]))))
    sub = [x for x in res if x["domain"] == d]
    for s in scales:
        rr = np.mean([x["rank"] for x in sub if x["cond"] == "role" and x["scale"] == s]); rd = np.mean([x["rank"] for x in sub if x["cond"] == "rand" and x["scale"] == s])
        print(f"{d:12s} scale={s:3.0f} | rank(role dir)={rr:4.2f}  rank(random)={rd:4.2f}  | dir/resid norm={dnorm[layers[0]]/resid_norm:.3f}")
print("\n=== summary (chance 3.5) ===")
for s in scales:
    rr = [x["rank"] for x in res if x["cond"] == "role" and x["scale"] == s]; rd = [x["rank"] for x in res if x["cond"] == "rand" and x["scale"] == s]
    gR = np.mean([x["gain_R"] for x in res if x["cond"] == "role" and x["scale"] == s]); gO = np.mean([x["gain_other"] for x in res if x["cond"] == "role" and x["scale"] == s])
    print(f"scale={s:3.0f} | role-dir rank {np.mean(rr):4.2f} (n={len(rr)})  random rank {np.mean(rd):4.2f} (n={len(rd)})  | mean logp gain: target span {gR:+.2f}, other spans {gO:+.2f}")
if a.out: json.dump(res, open(a.out, "w"), indent=1)
