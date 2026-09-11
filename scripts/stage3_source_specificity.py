"""Stage 3 (hour 24): is the forty-domain relation operator *source-specific*?

PRE-REGISTERED DESIGN (fixed before any number was looked at; not changed afterwards).

Background.  Hour 16: the role-centered affine operator (dual form, ridge 10), fit on 39 domains
and evaluated leave-one-domain-out, puts a held-out domain's true target role at mean role_rank
2.21 of 6 (chance 3.5), peaking at layer 16 (1.73).  Hour 20: as a *patch* its prediction helps
(+0.18 nats at lambda 0.5) but no more than the same operator fed the WRONG source role (+0.21).
So the operator's output direction is useful while source-specificity is undemonstrated.  The
question here is whether source-specificity exists at all, tested at the selector level first
(cheap, numpy only) and only then under patching.

Data: results/stacks_qwen2.5_1.5b_holonic_v2_rotated.npz, keys "<domain>/rot<k>", values
[29, 6, 1536]; roles from prompts/holonic_v2_rotated.json; 40 domains x 6 rotations = 240 prompts.
Protocol reproduced from scripts/stage3.py: grand-mean subtraction, all 30 ordered role pairs,
leave-one-domain-out over the 40 domains, per-fold role-centering (subtract the TRAINING-fold
per-role mean from train, test and candidate vectors), full affine fit in dual form (ridge 10,
n_spin 0), role_rank = position of the true target role among the held-out prompt's own six
centered role vectors by cosine to the prediction (1 best, chance 3.5).

(1) SELECTOR-LEVEL SOURCE-SPECIFICITY TEST.  For every held-out prompt and every pair (S -> T),
    feed the SAME fitted operator the prompt's true source residual and, separately, each of the
    five OTHER role residuals of the same prompt (the same prompt, so domain address is held
    fixed).  Metrics:
      - paired win rate: fraction of (prompt, pair, wrong-role) triples where role_rank(true
        source) < role_rank(wrong source); ties count 0.5.
      - mean role_rank of the true-source prediction vs mean role_rank of wrong-source predictions.
      - also reported: cos(pred, true target residual) for both.
    Interpretation fixed in advance: if wrong sources do as well (win rate near 0.5 and equal
    role_ranks), the operator is a role -> "target signature" map and not a relation.  If the true
    source wins clearly, source-specificity exists at the selector level and only the patch fails.

(2) CONTRASTIVE-SOURCE FIT.  Identical, except the operator is fit on differences: learn
    Delta = f(s) from (s, t - s) on the training folds and predict t_hat = s + f(s).  Reported at
    layer 16 (and over the sweep layers) with the same metrics, to see whether it sharpens
    source-specificity.

(3) LAYER SWEEP of (1) at layers 8, 12, 16, 20, 24.

(4) PATCH RERUN, CONDITIONAL.  Only if (1) shows a clear true-source win at layer 16 -- fixed
    threshold: paired win rate >= 0.60 -- rerun the hour-20 patch test (scripts/stage4b_relation_v2.py
    logic, same 12 held-out domains, same 6 pairs, lambda = 0.5) with relation vs wrong-source only.
    Otherwise (4) is skipped and the note says why.  Run with --patch (this loads the model).
    One extension over hour 20, declared before the patch was run and after (1)-(3) were seen:
    hour 20 used a single fixed wrong source (role index (src+3) mod 6); here EVERY one of the five
    other roles of the held-out prompt is used as a wrong source, so the patch comparison is the
    same paired comparison as (1).  The hour-20 single wrong source is reported separately too.
    Conditions are relation vs wrong-source only (no random control: hour 20 already established
    that both beat random at lambda 0.5); lambda = 0.5; layer 16; the same 12 held-out domains and
    the same 6 role pairs as scripts/stage4b_relation_v2.py; readout identical (extra log-prob gain
    on the target span beyond the role-only patch).

usage: python scripts/stage3_source_specificity.py <stacks.npz> <grid.json> --out results/stage3_source_specificity.json
"""
import argparse, itertools, json, time
import numpy as np
from lsx import operate, compare

ap = argparse.ArgumentParser()
ap.add_argument("stacks"); ap.add_argument("grid")
ap.add_argument("--layers", default="8,12,16,20,24")
ap.add_argument("--ridge", type=float, default=10.0)
ap.add_argument("--gate", type=float, default=0.60)
ap.add_argument("--patch", action="store_true", help="run step (4), the gated patch rerun (loads the model)")
ap.add_argument("--patch-layer", type=int, default=16)
ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--patch-pairs", default="from_above:from_below,embedded:from_below,new_subject:from_below,"
                                         "embedded:from_above,embedded:disturbance,from_below:objectified")
ap.add_argument("--patch-domains", default="physics,psychology,music,law,software,biology,mathematics,narrative,"
                                           "cooking,grief,chess,urban_planning")
ap.add_argument("--out", default=None)
a = ap.parse_args()

g = json.load(open(a.grid)); roles = g["roles"]; R = len(roles)
ri = {r: i for i, r in enumerate(roles)}
z = np.load(a.stacks); stacks = {k: z[k] for k in z.files if k != "roles"}
stacks = compare.subtract_grand_mean(stacks)
keys = sorted(stacks)
groups = np.array([k.split("/")[0] for k in keys])
doms = sorted(set(groups))
layers = [int(x) for x in a.layers.split(",")]
pairs = list(itertools.permutations(roles, 2))
print(f"stacks={a.stacks} prompts={len(keys)} domains={len(doms)} pairs={len(pairs)} layers={layers} ridge={a.ridge}")


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def run_layer(l, mode):
    """mode: 'residual' (fit t ~ f(s)) or 'contrastive' (fit t-s ~ f(s), predict s + f(s)).
    Returns per-pair aggregates over all 240 held-out prompts."""
    C = np.stack([stacks[k][l] for k in keys])            # [240, 6, d]
    per_pair = {}
    for src, dst in pairs:
        si, di = ri[src], ri[dst]
        rr_true, rr_wrong, wins, cos_true, cos_wrong = [], [], [], [], []
        for gname in doms:
            tr, te = groups != gname, groups == gname
            mu = C[tr].mean(0)                            # [6, d] training-fold per-role means
            Cc = C - mu                                   # role-centered everything
            S_tr, O_tr = Cc[tr][:, si], Cc[tr][:, di]
            Y_tr = O_tr - S_tr if mode == "contrastive" else O_tr
            op = operate.fit_affine(S_tr, Y_tr, l, src, dst, ridge=a.ridge, n_spin=0)
            for j in np.flatnonzero(te):
                cand = unit(Cc[j])                        # [6, d]
                true_t = Cc[j, di]

                def rank_of(s_vec):
                    p = op(s_vec)
                    if mode == "contrastive":
                        p = s_vec + p
                    sims = cand @ unit(p)
                    return int((sims > sims[di]).sum()) + 1, operate.cosine(p, true_t)

                r_t, c_t = rank_of(Cc[j, si])
                rr_true.append(r_t); cos_true.append(c_t)
                for w in range(R):
                    if w == si:
                        continue
                    r_w, c_w = rank_of(Cc[j, w])
                    rr_wrong.append(r_w); cos_wrong.append(c_w)
                    wins.append(1.0 if r_t < r_w else (0.5 if r_t == r_w else 0.0))
        per_pair[f"{src}->{dst}"] = dict(
            role_rank_true=float(np.mean(rr_true)), role_rank_wrong=float(np.mean(rr_wrong)),
            paired_win=float(np.mean(wins)), cos_true=float(np.mean(cos_true)),
            cos_wrong=float(np.mean(cos_wrong)), n_pairs=len(wins))
    return per_pair


def agg(per_pair):
    ks = ("role_rank_true", "role_rank_wrong", "paired_win", "cos_true", "cos_wrong")
    return {k: float(np.mean([v[k] for v in per_pair.values()])) for k in ks}


res = {"design": __doc__, "layers": layers, "ridge": a.ridge, "residual": {}, "contrastive": {}}
t0 = time.time()
for mode in ("residual", "contrastive"):
    for l in layers:
        pp = run_layer(l, mode)
        s = agg(pp)
        res[mode][str(l)] = {"summary": s, "per_pair": pp}
        print(f"{mode:12s} L{l:<3d} | role_rank true {s['role_rank_true']:.3f}  wrong {s['role_rank_wrong']:.3f}"
              f" | paired win {s['paired_win']:.3f} | cos true {s['cos_true']:.3f} wrong {s['cos_wrong']:.3f}"
              f"  [{time.time()-t0:.0f}s]", flush=True)

# hour-16 reproduction check at layer 16 via lsx.operate.holdout_eval (the exact hour-16 call)
rep = {}
C16 = np.stack([stacks[k][16] for k in keys])
for src, dst in pairs:
    si, di = ri[src], ri[dst]
    ev = operate.holdout_eval(C16[:, si], C16[:, di], groups, 16, src, dst, cands=C16, dst_idx=di,
                              src_idx=si, role_center=True, ridge=a.ridge, low_rank=None)
    rep[f"{src}->{dst}"] = ev["role_rank"]
res["holdout_eval_repro_layer16_role_rank"] = float(np.mean(list(rep.values())))
print(f"\nhour-16 reproduction (holdout_eval, layer 16, role-centered): mean role_rank "
      f"{res['holdout_eval_repro_layer16_role_rank']:.3f}  (hour 16 reported 1.73 at L16)")

w16 = res["residual"]["16"]["summary"]["paired_win"]
res["gate"] = {"threshold": a.gate, "paired_win_layer16": w16, "run_patch": bool(w16 >= a.gate)}
print(f"GATE: layer-16 paired win {w16:.3f} vs threshold {a.gate} -> "
      f"{'RUN' if w16 >= a.gate else 'SKIP'} the patch rerun (step 4)")

# ---------------------------------------------------------------- step (4): the gated patch rerun
if a.patch and res["gate"]["run_patch"]:
    import re
    import torch
    from lsx import LM
    from lsx.model import Patch
    from lsx.steer import add_vector

    l = a.patch_layer
    raw = {k: z[k] for k in z.files if k != "roles"}          # NOT grand-mean subtracted (as in hour 20)
    avg = {d: np.mean([raw[f"{d}/rot{r}"][l] for r in range(R)], axis=0) for d in doms}
    rot = {d: [raw[f"{d}/rot{r}"][l] for r in range(R)] for d in doms}
    pat = re.compile(r"\[\[(\w+):\s*(.*?)\]\]", re.S)
    spans, lead = {}, {}
    for d in doms:
        p = g["prompts"][f"{d}/rot0"]
        lead[d] = p.split(" First,")[0]
        spans[d] = {r: t for r, t in pat.findall(p)}
    ppairs = [tuple(x.split(":")) for x in a.patch_pairs.split(",")]
    test_doms = a.patch_domains.split(",")
    lm = LM.from_pretrained(a.model)

    @torch.no_grad()
    def logprobs(prefix, conts, patches=None):
        n_p = lm.encode(prefix)[0]["input_ids"].shape[1]
        ids = [lm.encode(prefix + c)[0]["input_ids"][0] for c in conts]
        n = max(len(t) for t in ids)
        inp = torch.full((len(ids), n), lm.tok.pad_token_id, dtype=torch.long)
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

    prows = []; t1 = time.time()
    print(f"\n=== step 4: patch rerun, layer {l}, lam {a.lam}, {len(test_doms)} domains, {len(ppairs)} pairs ===")
    for d in test_doms:
        train = [x for x in doms if x != d]
        mu = np.mean([avg[x] for x in train], axis=0)
        dirs = mu - mu.mean(0, keepdims=True)
        prefix = f"{lead[d]} First,"
        conts = [f" {spans[d][r]}." for r in roles]
        base = dict(zip(roles, logprobs(prefix, conts)))
        role_only = {}
        for T in sorted({t for _, t in ppairs}):
            gg = logprobs(prefix, conts, [Patch(l, add_vector(dirs[ri[T]], 1.0))])
            role_only[T] = {r: gg[i] - base[r] for i, r in enumerate(roles)}
        for S, T in ppairs:
            Si, Ti = ri[S], ri[T]
            Xs = np.stack([rot[x][k][Si] - mu[Si] for x in train for k in range(R)])
            Ys = np.stack([rot[x][k][Ti] - mu[Ti] for x in train for k in range(R)])
            op = operate.fit_affine(Xs, Ys, l, S, T, ridge=a.ridge, n_spin=0)
            pred = op(avg[d][Si] - mu[Si]); pn = float(np.linalg.norm(pred))
            ro = role_only[T]
            conds = {"relation": (pred, Si)}
            for w in range(R):
                if w == Si:
                    continue
                pw = op(avg[d][w] - mu[w])
                conds[f"wrong_{roles[w]}"] = (pw * pn / (np.linalg.norm(pw) + 1e-9), w)
            for cname, (extra, widx) in conds.items():
                vec = dirs[Ti] + a.lam * extra
                gg = logprobs(prefix, conts, [Patch(l, add_vector(vec, 1.0))])
                gains = {r: gg[i] - base[r] for i, r in enumerate(roles)}
                prows.append(dict(domain=d, src=S, dst=T, cond=cname, src_idx=Si, fed_idx=widx,
                                  hour20_wrong=(widx == (Si + 3) % R),
                                  extra_gain_T=gains[T] - ro[T],
                                  rank=1 + sum(gains[r] > gains[T] for r in roles if r != T),
                                  role_only_rank=1 + sum(ro[r] > ro[T] for r in roles if r != T)))
        print(f"{d:16s} relation {np.mean([x['extra_gain_T'] for x in prows if x['domain']==d and x['cond']=='relation']):+.3f}"
              f"  wrong(all) {np.mean([x['extra_gain_T'] for x in prows if x['domain']==d and x['cond']!='relation']):+.3f}"
              f"  [{time.time()-t1:.0f}s]", flush=True)

    rel = {(x["domain"], x["src"], x["dst"]): x["extra_gain_T"] for x in prows if x["cond"] == "relation"}
    wrong = [x for x in prows if x["cond"] != "relation"]
    wins = [1.0 if rel[(x["domain"], x["src"], x["dst"])] > x["extra_gain_T"] else 0.0 for x in wrong]
    h20 = [x for x in wrong if x["hour20_wrong"]]
    psum = dict(n_forward=len(prows) + len(test_doms) * 5, lam=a.lam, layer=l,
                relation_extra_gain=float(np.mean(list(rel.values()))),
                wrong_all_extra_gain=float(np.mean([x["extra_gain_T"] for x in wrong])),
                wrong_hour20_extra_gain=float(np.mean([x["extra_gain_T"] for x in h20])),
                paired_win_vs_wrong=float(np.mean(wins)),
                paired_win_vs_hour20_wrong=float(np.mean(
                    [1.0 if rel[(x["domain"], x["src"], x["dst"])] > x["extra_gain_T"] else 0.0 for x in h20])),
                relation_rank=float(np.mean([x["rank"] for x in prows if x["cond"] == "relation"])),
                wrong_rank=float(np.mean([x["rank"] for x in wrong])),
                role_only_rank=float(np.mean([x["role_only_rank"] for x in prows])))
    res["patch"] = {"summary": psum, "rows": prows}
    print("\n=== patch summary ===")
    for k, v in psum.items():
        print(f"  {k:28s} {v}")

if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)
