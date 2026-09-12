"""Subject-relative clocks, piece 1 (activations).

Spec: docs/specs/subject_clocks_v1.md sections 3.1-3.5 + the section-5 power rule.

  --stage extract : 504 passages (C1 240 + C3 264), layers 0/8/14/20/27; pools the
                    state span (mean, last, per-token for C3), the interval last token
                    (C2 expectation readout) and the per-passage t0 span (state0).
  --stage measure : 3.1 paraphrase-floor diagnosis, 3.2 per-subject contrast direction
                    + cross-subject matrix, 3.3 trajectory alignment, 3.4 resolvability
                    and floor-referenced tau, 3.5 discrimination, section-5 power rule,
                    figures + results/subject_clocks_measures.json.
"""
import argparse
import json
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("grid", nargs="?", default="prompts/subject_clocks_v1.json")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--layers", default="0,8,14,20,27")
ap.add_argument("--stage", default="all", choices=["extract", "measure", "all"])
ap.add_argument("--stacks", default="results/subject_clocks_stacks.npz")
ap.add_argument("--out", default="results/subject_clocks_measures.json")
ap.add_argument("--figdir", default="results/figures")
ap.add_argument("--ntok", type=int, default=24, help="per-token positions kept for C3")
a = ap.parse_args()

rng = np.random.default_rng(0)
G = json.load(open(a.grid))
S, DT = G["subjects"], G["deltas"]
LAYERS = [int(x) for x in a.layers.split(",")]
NP_ = G["n_paraphrases"]
LOGDT = G["log10_dt_days"]
NULLK = list(G["null_phrases"])                      # ["null1", "null2"]
ORDER = ["t0"] + DT                                  # null row first, per the tau estimator
CHANGING = [s for s in S if s != "asteroid"]
FAR = ["1000years", "10000years", "1000000years"]

# ------------------------------------------------------------------ extraction
if a.stage in ("extract", "all"):
    from lsx import LM
    from lsx.extract import extract

    lm = LM.from_pretrained(a.model)
    out, t_start, i = {}, time.time(), 0
    todo = [("C1", G["C1"]), ("C3", G["C3"])]
    n = sum(len(d) for _, d in todo)
    for tag, d in todo:
        for k, marked in d.items():
            ra = extract(lm, marked, keep_resid=True)
            R = ra.resid[LAYERS]                                    # [nL, seq, d]
            st, s0, iv = ra.tokens["state"], ra.tokens["state0"], ra.tokens["interval"]
            out[f"{tag}/mean/{k}"] = R[:, st].mean(1).astype(np.float16)
            out[f"{tag}/last/{k}"] = R[:, st[-1]].astype(np.float16)
            out[f"{tag}/s0mean/{k}"] = R[:, s0].mean(1).astype(np.float16)
            out[f"{tag}/s0last/{k}"] = R[:, s0[-1]].astype(np.float16)
            out[f"{tag}/ilast/{k}"] = R[:, iv[-1]].astype(np.float16)
            if tag == "C3":
                out[f"C3/tok/{k}"] = R[:, st[:a.ntok]].astype(np.float16)
            i += 1
            if i % 25 == 0:
                el = time.time() - t_start
                print(f"  {i}/{n}  {el:.0f}s  eta {el / i * (n - i):.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.stacks), exist_ok=True)
    np.savez_compressed(a.stacks, **out)
    print(f"wrote {a.stacks} ({len(out)} arrays) in {time.time() - t_start:.0f}s", flush=True)

if a.stage == "extract":
    raise SystemExit

# ------------------------------------------------------------------ helpers
Z = np.load(a.stacks)
nL = len(LAYERS)
LI = {l: i for i, l in enumerate(LAYERS)}


def get(tag, pool, s, t, p, li):
    return Z[f"{tag}/{pool}/{s}/{t}/p{p}"][li].astype(np.float64)


def disp(tag, pool, s, t, p, li):
    """displacement of the post-phrase state span from that passage's own t0 span."""
    ref = "s0mean" if pool == "mean" else "s0last"
    return get(tag, pool, s, t, p, li) - get(tag, ref, s, t, p, li)


def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def rank(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def spearman(x, y):
    rx, ry = rank(x) - rank(x).mean(), rank(y) - rank(y).mean()
    return float(rx @ ry / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


R = {"_meta": dict(model=a.model, layers=LAYERS, subjects=S, deltas=DT, order=ORDER,
                   grid=a.grid, n_paraphrases=NP_, spec="docs/specs/subject_clocks_v1.md",
                   n_passages=len(G["C1"]) + len(G["C3"]))}

# ================================================== 3.1 paraphrase-floor diagnosis
# C1, mean pool, hour-28 style residual vs the within-cell paraphrase floor.
floor = {}
for l in LAYERS:
    li = LI[l]
    dbar = {s: {t: np.mean([disp("C1", "mean", s, t, p, li) for p in range(NP_)], axis=0)
                for t in DT} for s in S}
    shared = {t: np.mean([dbar[s][t] for s in S], axis=0) for t in DT}

    def trsig(vecs):                       # unbiased trace of within-cell covariance
        M = np.stack(vecs)
        return float(((M - M.mean(0)) ** 2).sum(1).sum() / (len(vecs) - 1))

    tr0 = {s: trsig([get("C1", "s0mean", s, "t0", p, li) for p in range(NP_)]) for s in S}
    per = {}
    for s in S:
        per[s] = {}
        for t in DT:
            trt = trsig([get("C1", "mean", s, t, p, li) for p in range(NP_)])
            F = float(np.sqrt((trt + tr0[s]) / NP_))
            rn = float(np.linalg.norm(dbar[s][t] - shared[t]))
            per[s][t] = dict(resid_norm=rn, floor=F, ratio=rn / (F + 1e-12),
                             d_norm=float(np.linalg.norm(dbar[s][t])))
    floor[str(l)] = dict(per_subject=per,
                         shared_norm={t: float(np.linalg.norm(shared[t])) for t in DT})
R["m31_floor"] = floor

# ================================================== 3.2 contrast direction + y readout
def displacement_table(tag, pool, li):
    """d[s][t][p] for a construction; C1/C3 use the state span, C2 the interval token."""
    tab = {}
    for s in S:
        tab[s] = {}
        if tag == "C2":
            base = {p: np.mean([get("C3", "ilast", s, nk, p, li) for nk in NULLK], axis=0)
                    for p in range(NP_)}
            for t in DT:
                tab[s][t] = [get("C3", "ilast", s, t, p, li) - base[p] for p in range(NP_)]
            # null row: alternate the two null phrases by p (C1's convention), baseline = mean
            tab[s]["t0"] = [get("C3", "ilast", s, NULLK[p % 2], p, li) - base[p]
                            for p in range(NP_)]
        elif tag == "C3":
            for t in DT:
                tab[s][t] = [disp("C3", pool, s, t, p, li) for p in range(NP_)]
            tab[s]["t0"] = [np.mean([disp("C3", pool, s, nk, p, li) for nk in NULLK], axis=0)
                            for p in range(NP_)]
        else:
            for t in ORDER:
                tab[s][t] = [disp("C1", pool, s, t, p, li) for p in range(NP_)]
    return tab


def u_dirs(tab, s):
    """u_s^{(-p)}: subject's own t0->far axis, fit without paraphrase p."""
    return [unit(np.mean([tab[s][t][q] for t in FAR for q in range(NP_) if q != p], axis=0))
            for p in range(NP_)]


def y_curve(tab, s):
    u = u_dirs(tab, s)
    return {t: np.array([float(tab[s][t][p] @ u[p]) for p in range(NP_)]) for t in ORDER}


def tau(y, z=2.5):
    n = len(next(iter(y.values())))
    ybar = {t: float(y[t].mean()) for t in ORDER}
    se = float(np.sqrt(np.mean([y[t].var(ddof=1) for t in ORDER]) * 2 / n))
    sig = float(np.sqrt(np.mean([y[t].var(ddof=1) for t in ORDER])))
    b = ybar[ORDER[0]]
    P = float(np.mean(sorted(ybar[t] - b for t in ORDER[1:])[-3:]))
    res = dict(ybar={t: ybar[t] - b for t in ORDER}, sigma=sig, se=se, P=P,
               P_over_sigma=P / (sig + 1e-12))
    if P < 2 * z * se:
        res["tau"] = "no signal"
        return res
    thr = max(P / 2, z * se)
    hits = [t for t in ORDER[1:] if ybar[t] - b >= thr]
    later = [t for t in ORDER[1:] if ybar[t] - b >= z * se]
    res["tau"] = (hits[0] if hits and len(later) >= 2 else "no signal")
    res["threshold"] = thr
    return res


def tau_loo(tab, s):
    """tau recomputed under each leave-one-paraphrase-out fit of u_s (the CI set)."""
    out = []
    for q in range(NP_):
        u = unit(np.mean([tab[s][t][p] for t in FAR for p in range(NP_) if p != q], axis=0))
        y = {t: np.array([float(tab[s][t][p] @ u) for p in range(NP_)]) for t in ORDER}
        out.append(tau(y)["tau"])
    return out


CONS = [("C1", "mean"), ("C1", "last"), ("C3", "last"), ("C2", "ilast")]
yres = {}
for tag, pool in CONS:
    key = f"{tag}_{pool}"
    yres[key] = {}
    for l in LAYERS:
        tab = displacement_table(tag, pool, LI[l])
        d = {}
        for s in S:
            y = y_curve(tab, s)
            t_ = tau(y)
            t_["tau_loo"] = tau_loo(tab, s)
            t_["y_mean"] = {t: float(y[t].mean()) for t in ORDER}
            d[s] = t_
        yres[key][str(l)] = d
R["m32_y_tau"] = yres

# --------- cross-subject matrix + diagonal dominance (C2, C3)
cross = {}
for tag, pool in [("C3", "last"), ("C2", "ilast")]:
    cross[f"{tag}_{pool}"] = {}
    for l in LAYERS:
        tab = displacement_table(tag, pool, LI[l])
        U = {s: u_dirs(tab, s) for s in S}
        sd = {}
        for s in S:                                   # column scale: within-cell SD along u_s
            v = [float(tab[s][t][p] @ U[s][p]) for t in ORDER for p in range(NP_)]
            v = np.array(v).reshape(len(ORDER), NP_)
            sd[s] = float(np.sqrt(np.mean(v.var(axis=1, ddof=1)))) + 1e-12
        # addendum (not pre-registered): the same table on dt-centred displacements, i.e. with
        # each subject's mean-over-dt displacement removed, so that diagonal dominance cannot be
        # produced by subject identity alone (which is dt-independent by construction).
        ctab = {s: {t: [tab[s][t][p] - np.mean([tab[s][tt][p] for tt in DT], axis=0)
                        for p in range(NP_)] for t in ORDER} for s in S}
        CU = {s: u_dirs(ctab, s) for s in S}
        csd = {}
        for s in S:
            v = np.array([float(ctab[s][t][p] @ CU[s][p]) for t in ORDER
                          for p in range(NP_)]).reshape(len(ORDER), NP_)
            csd[s] = float(np.sqrt(np.mean(v.var(axis=1, ddof=1)))) + 1e-12
        n = len(S)

        def dom(A):
            off = (A.sum() - np.trace(A)) / (n * n - n)
            D = float(np.trace(A) / n - off)
            null = np.empty(10000)
            for b in range(10000):
                Ap = A[rng.permutation(n)]
                null[b] = np.trace(Ap) / n - (Ap.sum() - np.trace(Ap)) / (n * n - n)
            return D, float((null >= D).mean())

        per_dt = {}
        for t in DT:
            A = np.array([[np.mean([float(tab[s][t][p] @ U[sp][p]) for p in range(NP_)]) / sd[sp]
                           for sp in S] for s in S])
            Ac = np.array([[np.mean([float(ctab[s][t][p] @ CU[sp][p]) for p in range(NP_)]) / csd[sp]
                            for sp in S] for s in S])
            D, pv = dom(A)
            Dc, pvc = dom(Ac)
            per_dt[t] = dict(D=D, p=pv, D_centred=Dc, p_centred=pvc, A=A.round(3).tolist())
        cross[f"{tag}_{pool}"][str(l)] = per_dt
R["m32_crosssubject"] = cross

# ================================================== 3.3 trajectory alignment kappa
kap = {}
for l in LAYERS:
    li = LI[l]
    tab = displacement_table("C1", "last", li)
    dbar = {s: {t: np.mean(tab[s][t], axis=0) for t in ORDER} for s in S}
    sh = {t: {s: np.mean([dbar[o][t] for o in S if o != s], axis=0) for s in S} for t in ORDER}
    per = {}
    for s in S:
        per[s] = {}
        for t in ORDER:
            vals = []
            for p in range(NP_):
                lhs = tab[s][t][p]
                rhs = np.mean([tab[s]["1000000years"][q] for q in range(NP_) if q != p], axis=0)
                a1, a2 = unit(sh[t][s]), unit(sh["1000000years"][s])
                lhs = lhs - (lhs @ a1) * a1
                rhs = rhs - (rhs @ a2) * a2
                vals.append(cos(lhs, rhs))
            per[s][t] = float(np.mean(vals))
    kap[str(l)] = per
R["m33_kappa"] = kap

# ================================================== 3.4 resolvability
def resolvability(vec, s):
    """vec[t][p] -> per-pair leave-one-paraphrase-out nearest-centroid accuracy (6 trials)."""
    Rm = np.zeros((len(ORDER), len(ORDER)))
    for i, ti in enumerate(ORDER):
        for j, tj in enumerate(ORDER):
            if j <= i:
                continue
            ok = 0
            for p in range(NP_):
                ci = np.mean([vec[ti][q] for q in range(NP_) if q != p], axis=0)
                cj = np.mean([vec[tj][q] for q in range(NP_) if q != p], axis=0)
                ok += np.linalg.norm(vec[ti][p] - ci) < np.linalg.norm(vec[ti][p] - cj)
                ok += np.linalg.norm(vec[tj][p] - cj) < np.linalg.norm(vec[tj][p] - ci)
            Rm[i, j] = Rm[j, i] = ok / 6
    np.fill_diagonal(Rm, np.nan)
    return Rm


# pre-registered block structure (spec 3.4 table): unresolved set from t0 up to and
# including the listed timepoint; orchard's "1 y ~ t0 again" is kept as written.
BLOCK = {"mayfly": [], "orchard": ["1day", "1week", "1year"], "street": ["1day", "1week", "6months", "1year"],
         "real_population": ["1day", "1week", "6months", "1year"],
         "fictional_population": ["1day", "1week", "6months", "1year"],
         "river": ["1day", "1week", "6months", "1year", "10years"],
         "mountain": ["1day", "1week", "6months", "1year", "10years", "100years", "1000years"],
         "asteroid": DT}
FIRST = {"mayfly": "1day", "orchard": "6months", "street": "10years", "real_population": "10years",
         "fictional_population": "10years", "river": "100years", "mountain": "10000years",
         "asteroid": None}

resolv = {}
for l in LAYERS:
    li = LI[l]
    tab = displacement_table("C1", "last", li)
    resolv[str(l)] = {}
    for readout in ["h_last", "y"]:
        per = {}
        for s in S:
            if readout == "h_last":
                vec = {t: [get("C1", "last", s, t, p, li) for p in range(NP_)] for t in ORDER}
            else:
                u = u_dirs(tab, s)
                vec = {t: [np.array([float(tab[s][t][p] @ u[p])]) for p in range(NP_)]
                       for t in ORDER}
            Rm = resolvability(vec, s)
            blk = set(["t0"] + BLOCK[s])
            lab, obs = [], []
            for i, ti in enumerate(ORDER):
                for j, tj in enumerate(ORDER):
                    if j <= i:
                        continue
                    inb_i, inb_j = ti in blk, tj in blk
                    if inb_i and inb_j:
                        lab.append(0)
                    elif inb_i != inb_j:
                        lab.append(1)
                    else:
                        continue
                    obs.append(1 if Rm[i, j] >= 1.0 else 0)
            agree = float(np.mean([l_ == o for l_, o in zip(lab, obs)]))
            firstres = None
            for t in DT:
                if Rm[0, ORDER.index(t)] >= 1.0:
                    firstres = t
                    break
            per[s] = dict(matrix=np.nan_to_num(Rm, nan=-1).round(3).tolist(),
                          first_resolved_from_t0=firstres, predicted_first=FIRST[s],
                          block_agreement=agree, n_labelled=len(lab),
                          frac_resolved=float(np.nanmean(Rm >= 1.0)))
        resolv[str(l)][readout] = per
R["m34_resolvability"] = resolv

# ================================================== 3.5 discrimination
disc = {}
for l in LAYERS:
    li = LI[l]
    tab = displacement_table("C1", "last", li)
    U = {s: u_dirs(tab, s) for s in S}
    per = {}
    for s in S:
        pw, ps, tr = [], [], []
        for i, ti in enumerate(DT):
            for p in range(NP_):
                h = get("C1", "last", s, ti, p, li)
                cent = [np.mean([get("C1", "last", s, tj, q, li) for q in range(NP_) if q != p],
                                axis=0) for tj in ORDER]
                pw.append(int(np.argmin([np.linalg.norm(h - c) for c in cent])))
                d = tab[s][ti][p]
                shd = [np.mean([tab[o][tj][q] for o in S if o != s for q in range(NP_)], axis=0)
                       for tj in ORDER]
                ps.append(int(np.argmin([np.linalg.norm(d - c) for c in shd])))
                tr.append(ORDER.index(ti))
        # matched-size shared: 2 other subjects, one paraphrase each (18 training vectors)
        msp, msm = [], []
        for draw in range(20):
            oth = rng.choice([o for o in S if o != s], 2, replace=False)
            par = rng.integers(0, NP_, 2)
            shd = [np.mean([tab[oth[k]][tj][par[k]] for k in range(2)], axis=0) for tj in ORDER]
            pred = []
            for ti in DT:
                for p in range(NP_):
                    d = tab[s][ti][p]
                    pred.append(int(np.argmin([np.linalg.norm(d - c) for c in shd])))
            msp.append(spearman(pred, tr))
            msm.append(float(np.mean(np.abs(np.array(pred) - np.array(tr)))))
        per[s] = dict(within_spearman=spearman(pw, tr),
                      within_mae=float(np.mean(np.abs(np.array(pw) - np.array(tr)))),
                      shared_spearman=spearman(ps, tr),
                      shared_mae=float(np.mean(np.abs(np.array(ps) - np.array(tr)))),
                      matched_spearman=float(np.mean(msp)), matched_mae=float(np.mean(msm)),
                      within_minus_shared=spearman(pw, tr) - spearman(ps, tr))
    disc[str(l)] = per
R["m35_discrimination"] = disc

# ================================================== per-token C3
pertok = {}
for l in LAYERS:
    li = LI[l]
    tab = displacement_table("C3", "last", li)
    U = {s: u_dirs(tab, s) for s in S}
    per = {}
    for s in S:
        K = Z[f"C3/tok/{s}/1day/p0"].shape[1]
        base = {p: np.mean([Z[f"C3/tok/{s}/{nk}/p{p}"][li].astype(np.float64) for nk in NULLK],
                           axis=0) for p in range(NP_)}
        curves = np.zeros((len(DT), K))
        for ti, t in enumerate(DT):
            for k in range(K):
                curves[ti, k] = np.mean([(Z[f"C3/tok/{s}/{t}/p{p}"][li][k].astype(np.float64)
                                          - base[p][k]) @ U[s][p] for p in range(NP_)])
        per[s] = dict(sd_over_dt=curves.std(axis=0).round(4).tolist(),
                      mean_abs=np.abs(curves).mean(axis=0).round(4).tolist())
    pertok[str(l)] = per
R["m3_pertoken_C3"] = pertok

# ================================================== section 5 power rule
pw = {}
for l in [8, 14, 20]:
    vals = [yres["C1_last"][str(l)][s]["P_over_sigma"] for s in CHANGING]
    pw[str(l)] = dict(per_subject={s: yres["C1_last"][str(l)][s]["P_over_sigma"] for s in CHANGING},
                      r_median=float(np.median(vals)))
r14 = pw["14"]["r_median"]
branch = ("adequate n=3 (r >= 4.1)" if r14 >= 4.1 else
          "underpowered: piece 2 writes 3 more paraphrases (n=6)" if r14 >= 2.2 else
          "r < 2.2: author-supplied readout null at this model; piece 2 runs decoding only")
pw["branch"] = branch
pw["r_layer14"] = r14
R["m5_power"] = pw

# ================================================== gate check (report only)
gate = {}
for key in ["C3_last", "C2_ilast"]:
    gate[key] = {str(l): int(sum(cross[key][str(l)][t]["p"] < 0.05 for t in DT)) for l in LAYERS}
    gate[key + "_centred"] = {str(l): int(sum(cross[key][str(l)][t]["p_centred"] < 0.05 for t in DT))
                              for l in LAYERS}
gate["dominance_gate_fires"] = bool(max(gate["C3_last"][str(l)] for l in LAYERS) >= 3
                                    or max(gate["C2_ilast"][str(l)] for l in LAYERS) >= 3)
gate["dominance_gate_fires_centred"] = bool(
    max(gate["C3_last_centred"][str(l)] for l in LAYERS) >= 3
    or max(gate["C2_ilast_centred"][str(l)] for l in LAYERS) >= 3)
R["m7_gate"] = gate

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(R, open(a.out, "w"), indent=1, default=float)
print("wrote", a.out)
print("r(layer14) =", round(r14, 2), "->", branch)

# ================================================== figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs(a.figdir, exist_ok=True)
X = [LOGDT[t] for t in DT]
XO = [-1.0] + X


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{a.figdir}/subject_clocks_{name}.png", dpi=110)
    plt.close(fig)


fig, ax = plt.subplots(figsize=(7, 4.5))
for s in S:
    ax.plot(X, [floor["14"]["per_subject"][s][t]["ratio"] for t in DT], marker="o", label=s)
ax.axhline(1, color="k", ls="--")
ax.set_xlabel("log10 dt (days)"), ax.set_ylabel("||resid|| / paraphrase floor")
ax.set_title("3.1 old probe (C1 mean pool, layer 14) vs its own noise floor")
ax.legend(fontsize=6)
save(fig, "floor")

for key, nm in [("C1_last", "ycurves_C1"), ("C3_last", "ycurves_C3"), ("C2_ilast", "ycurves_C2")]:
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, l in zip(axs, [8, 14, 20]):
        for s in S:
            d = yres[key][str(l)][s]
            ax.plot(XO, [d["ybar"][t] / (d["sigma"] + 1e-12) for t in ORDER], marker="o", label=s)
        ax.set_title(f"layer {l}"), ax.set_xlabel("log10 dt (days); -1 = null row")
    axs[0].set_ylabel("y (units of within-cell SD)")
    axs[2].legend(fontsize=6)
    fig.suptitle(f"3.2 y readout, {key}")
    save(fig, nm)

fig, ax = plt.subplots(figsize=(7, 4.5))
for s in S:
    ax.plot(XO, [kap["14"][s][t] for t in ORDER], marker="o", label=s)
ax.set_xlabel("log10 dt (days)"), ax.set_ylabel("kappa vs 1 My")
ax.set_title("3.3 trajectory alignment to terminal state (C1 last, layer 14)")
ax.legend(fontsize=6)
save(fig, "kappa")

fig, axs = plt.subplots(2, 4, figsize=(15, 7))
for ax, s in zip(axs.ravel(), S):
    M = np.array(resolv["14"]["h_last"][s]["matrix"])
    ax.imshow(np.where(M < 0, np.nan, M), vmin=0.5, vmax=1, cmap="viridis")
    ax.set_title(s, fontsize=8)
    ax.set_xticks(range(len(ORDER))), ax.set_xticklabels(ORDER, rotation=90, fontsize=5)
    ax.set_yticks(range(len(ORDER))), ax.set_yticklabels(ORDER, fontsize=5)
fig.suptitle("3.4 resolvability (C1 last, layer 14, LOO nearest centroid)")
save(fig, "resolv")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, key in zip(axs, ["C3_last", "C2_ilast"]):
    ax.plot(X, [cross[key]["14"][t]["D"] for t in DT], marker="o")
    ax2 = ax.twinx()
    ax2.plot(X, [cross[key]["14"][t]["p"] for t in DT], marker="x", color="r")
    ax2.axhline(0.05, color="r", ls="--")
    ax.set_title(f"{key}: diagonal dominance D (blue), perm p (red)")
    ax.set_xlabel("log10 dt (days)")
save(fig, "crosssubject")

fig, ax = plt.subplots(figsize=(7, 4.5))
for s in S:
    ax.plot(pertok["14"][s]["sd_over_dt"], marker="o", label=s)
ax.set_xlabel("token index in repeated state span"), ax.set_ylabel("SD over dt of y_k")
ax.set_title("C3 per-token dt-dependence (layer 14)")
ax.legend(fontsize=6)
save(fig, "pertoken")
print("figures ->", a.figdir)
