"""Stage 4b (v2): the relation lens as a *patch*, with the forty-domain operator.

Hour 5 asked this at seven domains and got nothing (-0.02 nats vs -0.19 for a random direction).
Hour 16 showed the role-centered affine operator is real as a *selector* once 40 domains are
available (role_rank 2.21 vs 3.5 null, peaking at layer 16).  This script asks the generative half:

  role-only patch:  v0 = dir_T                                     (hour-4 lens)
  relation patch:   v1 = dir_T + lam * pred,  pred = W s_d + b      (role-centered affine map fit on
                    the other 39 domains x 6 rotations; s_d = d's position-balanced source residual)
  controls:         dir_T + lam * random  (same norm as pred, x2)
                    dir_T + lam * pred_from_wrong_source  (map applied to another role's residual)

Metric: logp(span_T | patch) - logp(span_T | dir_T only)  ("extra gain"), and the rank of span_T
among the six spans by gain over base.  If the relation carries usable content, v1 beats the controls.

Adapted from scripts/stage4_relation.py; the only mechanical change is that the six candidate spans
are scored in one padded batch per patch (right padding, so causal positions are untouched), which
cuts wall-clock roughly threefold on CPU.
"""
import argparse, json, re, time
import numpy as np
import torch
from lsx import LM, operate
from lsx.model import Patch
from lsx.steer import add_vector

ap = argparse.ArgumentParser()
ap.add_argument("grid"); ap.add_argument("stacks")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--layer", type=int, default=16)
ap.add_argument("--pairs", default="from_above:from_below,embedded:from_below,new_subject:from_below,"
                                   "embedded:from_above,embedded:disturbance,from_below:objectified")
ap.add_argument("--lam", default="0.5,1.0")
ap.add_argument("--ridge", type=float, default=10.0)
ap.add_argument("--domains", default="physics,psychology,music,law,software,biology,mathematics,narrative,"
                                     "cooking,grief,chess,urban_planning")
ap.add_argument("--out", default=None)
a = ap.parse_args()

g = json.load(open(a.grid)); roles = g["roles"]; ri = {r: i for i, r in enumerate(roles)}
pat = re.compile(r"\[\[(\w+):\s*(.*?)\]\]", re.S)
z = np.load(a.stacks); stacks = {k: z[k] for k in z.files if k != "roles"}
doms = sorted({k.split("/")[0] for k in stacks}); R = len(roles); l = a.layer
avg = {d: np.mean([stacks[f"{d}/rot{r}"][l] for r in range(R)], axis=0) for d in doms}   # [6, d]
rot = {d: [stacks[f"{d}/rot{r}"][l] for r in range(R)] for d in doms}                    # 6 x [6, d]
spans, lead = {}, {}
for d in doms:
    p = g["prompts"][f"{d}/rot0"]; lead[d] = p.split(" First,")[0]; spans[d] = {r: t for r, t in pat.findall(p)}
pairs = [tuple(x.split(":")) for x in a.pairs.split(",")]
lams = [float(x) for x in a.lam.split(",")]
test_doms = a.domains.split(",") if a.domains else doms
assert all(d in doms for d in test_doms), [d for d in test_doms if d not in doms]

lm = LM.from_pretrained(a.model)


@torch.no_grad()
def logprobs(prefix, conts, patches=None):
    """Summed log p(cont | prefix) for several continuations, in one padded batch."""
    n_p = lm.encode(prefix)[0]["input_ids"].shape[1]
    ids = [lm.encode(prefix + c)[0]["input_ids"][0] for c in conts]
    n = max(len(t) for t in ids)
    pad = lm.tok.pad_token_id
    inp = torch.full((len(ids), n), pad, dtype=torch.long)
    att = torch.zeros((len(ids), n), dtype=torch.long)
    for i, t in enumerate(ids):
        inp[i, :len(t)] = t; att[i, :len(t)] = 1
    with lm.patched(patches or []):
        logits = lm.model(input_ids=inp.to(lm.device), attention_mask=att.to(lm.device)).logits.float()
    lp = torch.log_softmax(logits[:, :-1], dim=-1)
    out = []
    for i, t in enumerate(ids):
        idx = torch.arange(n_p - 1, len(t) - 1)
        out.append(float(lp[i, idx, t[n_p:].to(lp.device)].sum().cpu()))
    return out


# sanity: the batched readout must match the one-at-a-time reference used in hour 5
d0 = test_doms[0]; pre0 = f"{lead[d0]} First,"
ref = [lm.logprob(pre0, f" {spans[d0][r]}.") for r in roles[:2]]
bat = logprobs(pre0, [f" {spans[d0][r]}." for r in roles])[:2]
assert max(abs(x - y) for x, y in zip(ref, bat)) < 1e-2, (ref, bat)

rng = np.random.default_rng(0); res = []; t0 = time.time()
print(f"model={a.model} layer={l} ridge={a.ridge} lams={lams} pairs={pairs} n_test={len(test_doms)}")
for d in test_doms:
    train = [x for x in doms if x != d]
    mu = np.mean([avg[x] for x in train], axis=0)                     # [6, d] role means (training)
    dirs = mu - mu.mean(0, keepdims=True)
    prefix = f"{lead[d]} First,"
    conts = [f" {spans[d][r]}." for r in roles]
    base = dict(zip(roles, logprobs(prefix, conts)))
    role_only = {}
    for T in sorted({t for _, t in pairs}):
        gg = logprobs(prefix, conts, [Patch(l, add_vector(dirs[ri[T]], 1.0))])
        role_only[T] = {r: gg[i] - base[r] for i, r in enumerate(roles)}
    for S, T in pairs:
        Si, Ti = ri[S], ri[T]
        Xs = np.stack([rot[x][k][Si] - mu[Si] for x in train for k in range(R)])   # 234 role-centered sources
        Ys = np.stack([rot[x][k][Ti] - mu[Ti] for x in train for k in range(R)])
        op = operate.fit_affine(Xs, Ys, l, S, T, ridge=a.ridge, n_spin=0)
        s_d = avg[d][Si] - mu[Si]
        pred = op(s_d); pn = float(np.linalg.norm(pred))
        wi = (Si + 3) % R
        pred_wrong = op(avg[d][wi] - mu[wi]); pred_wrong = pred_wrong * pn / (np.linalg.norm(pred_wrong) + 1e-9)
        ro = role_only[T]
        ro_rank = 1 + sum(ro[r] > ro[T] for r in roles if r != T)
        cos_pred_true = float(operate.cosine(pred, avg[d][Ti] - mu[Ti]))
        for lam in lams:
            conds = {"relation": pred, "wrong_source": pred_wrong}
            for k in range(2):
                v = rng.normal(size=pred.shape); conds[f"rand{k}"] = v / np.linalg.norm(v) * pn
            for cname, extra in conds.items():
                vec = dirs[Ti] + lam * extra
                gg = logprobs(prefix, conts, [Patch(l, add_vector(vec, 1.0))])
                gains = {r: gg[i] - base[r] for i, r in enumerate(roles)}
                rank = 1 + sum(gains[r] > gains[T] for r in roles if r != T)
                res.append(dict(domain=d, src=S, dst=T, lam=lam,
                                cond="rand" if cname.startswith("rand") else cname,
                                extra_gain_T=gains[T] - ro[T], gain_T=gains[T], rank=rank,
                                role_only_gain_T=ro[T], role_only_rank=ro_rank,
                                mean_other_gain=float(np.mean([gains[r] for r in roles if r != T])),
                                cos_pred_true=cos_pred_true,
                                pred_over_dir=float(pn / np.linalg.norm(dirs[Ti]))))
    sub = [x for x in res if x["domain"] == d]
    for lam in lams:
        f = lambda c: np.mean([x["extra_gain_T"] for x in sub if x["cond"] == c and x["lam"] == lam])
        print(f"{d:16s} lam={lam:3.1f} | extra gain on target: relation {f('relation'):+.2f}  "
              f"wrong-source {f('wrong_source'):+.2f}  random {f('rand'):+.2f}  | role-only rank "
              f"{np.mean([x['role_only_rank'] for x in sub if x['lam']==lam]):.2f}  [{time.time()-t0:.0f}s]",
              flush=True)

print("\n=== summary ===")
for lam in lams:
    for c in ("relation", "wrong_source", "rand"):
        xs = [x for x in res if x["cond"] == c and x["lam"] == lam]
        print(f"lam={lam:3.1f} {c:13s} | extra gain on target {np.mean([x['extra_gain_T'] for x in xs]):+.3f} "
              f"(n={len(xs)}) | rank {np.mean([x['rank'] for x in xs]):.2f} "
              f"(role-only {np.mean([x['role_only_rank'] for x in xs]):.2f})")
print("mean |pred|/|dir_T| =", round(float(np.mean([x['pred_over_dir'] for x in res])), 3),
      "| mean cos(pred, true target content) =", round(float(np.mean([x['cos_pred_true'] for x in res])), 3))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
