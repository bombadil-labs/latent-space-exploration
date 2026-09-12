"""The shared clock as a gain over the lexical floor, by depth.

Implements docs/specs/clock_depth_gain_v1.md sec 3-4 exactly:
  - LOSO ridge (dual form, kernel-centred) with inner-LOSO lambda selection
  - per-layer normalisation (grand-mean centre, divide by mean vector norm)
  - lexical floor F-emb / F-bow / F-bow+emb, rho_lex = max
  - primary residual gain G_res(L), secondary drho(L), structure gain G_struct(L)
  - within-subject permutation nulls (200 draws), A/B label-swap null for S2
  - S1 depth profile, S2 order dependence, S3 per-dt gain, S4 arm-C transfer + cosine
  - the calibration checks of sec 3.4.6

Usage: python scripts/clock_gain.py [--nperm 200]
"""
import argparse
import json
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="prompts/clock_gain_v1.json")
ap.add_argument("--nperm", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/clock_gain_v1_measures.json")
ap.add_argument("--figdir", default="results/figures")
a = ap.parse_args()

RNG = np.random.default_rng(a.seed)
G = json.load(open(a.grid))
S, DT, NP_ = G["subjects"], G["deltas"], G["n_paraphrases"]
LOGDT = G["log10_dt_days"]
NL = 29
ARMS = ["A", "B", "C", "D"]

CELLS = [(s, t, p) for s in S for t in DT for p in range(NP_)]          # 216 regression cells
ALL_CELLS = [(s, t, p) for s in S for t in ["t0"] + DT for p in range(NP_)]   # 240
y = np.array([LOGDT[t] for (_, t, _) in CELLS])
groups = np.array([S.index(s) for (s, _, _) in CELLS])
dt_index = np.array([DT.index(t) for (_, t, _) in CELLS])
GRIDVALS = np.array([LOGDT[t] for t in DT])
LAM_MULT = np.logspace(-3, 3, 7)


# ------------------------------------------------------------------ scoring
def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    # average ranks on ties (mid-rank; never rank-1-on-ties, see hour-36 diagnosis)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y_):
    rx, ry = rankdata(x), rankdata(y_)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float(rx @ ry / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def spearman_cols(P, target):
    """Spearman of each column of P [n, m] against target [n] (or [n, m] elementwise)."""
    T = np.asarray(target)
    out = np.empty(P.shape[1])
    for j in range(P.shape[1]):
        out[j] = spearman(P[:, j], T[:, j] if T.ndim == 2 else T)
    return out


def r2(pred, true):
    return float(max(0.0, 1.0 - np.sum((true - pred) ** 2) / (np.sum((true - true.mean()) ** 2) + 1e-12)))


def grid_mae(pred):
    """MAE in grid steps: snap each prediction to the nearest of the nine grid values."""
    pi = np.argmin(np.abs(pred[:, None] - GRIDVALS[None, :]), axis=1)
    return pi, np.abs(pi - dt_index)


def partial_spearman(pred, target, control):
    """Rank-partial correlation of pred with target, controlling for control.

    Used as the CALIBRATED companion to the spec's G_res (sec 3.3). G_res residualises
    the target on the floor and refits; because the layer's features overlap the floor's
    inputs, the two readouts' *errors* are correlated and G_res is biased negative by an
    amount that depends on that error correlation rather than on information. The partial
    rank correlation has no such bias: when the layer readout is the floor readout it is
    exactly 0, which is the identity the sec-3.4.6 calibration check is about.
    """
    a, b, c = rankdata(pred), rankdata(target), rankdata(control)
    c = c - c.mean()
    a = a - a.mean() - c * (a @ c) / (c @ c + 1e-12)
    b = b - b.mean() - c * (b @ c) / (c @ c + 1e-12)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def zscore(obs, null):
    null = np.asarray(null, float)
    return float((obs - null.mean()) / (null.std() + 1e-12))


# ------------------------------------------------------------------ ridge machinery
def _centre(K, tr, te):
    Ktr = K[np.ix_(tr, tr)]
    r = Ktr.mean(0)
    mm = Ktr.mean()
    Ktr_c = Ktr - r[None, :] - r[:, None] + mm
    Kte = K[np.ix_(te, tr)]
    Kte_c = Kte - Kte.mean(1, keepdims=True) - r[None, :] + mm
    return Ktr_c, Kte_c, r, mm


class FoldMachine:
    """LOSO ridge over a precomputed Gram matrix, with inner-LOSO lambda selection.

    All kernel centrings and eigendecompositions are cached, so predict_batch() can be
    called for hundreds of permuted targets at the cost of a few matmuls each.
    """

    def __init__(self, K, groups_, d_feat):
        self.K = K
        self.g = groups_
        self.d = d_feat
        self.folds = []
        for u in np.unique(groups_):
            te = np.where(groups_ == u)[0]
            tr = np.where(groups_ != u)[0]
            Ktr_c, Kte_c, _, _ = _centre(K, tr, te)
            w, V = np.linalg.eigh(Ktr_c)
            lam = LAM_MULT * (np.trace(Ktr_c) / len(tr) / d_feat)
            inner = []
            for u2 in np.unique(groups_[tr]):
                te2 = tr[groups_[tr] == u2]
                tr2 = tr[groups_[tr] != u2]
                Ktr2, Kte2, _, _ = _centre(K, tr2, te2)
                w2, V2 = np.linalg.eigh(Ktr2)
                lam2 = LAM_MULT * (np.trace(Ktr2) / len(tr2) / d_feat)
                inner.append(dict(tr=tr2, te=te2, w=w2, V=V2, Kte=Kte2, lam=lam2))
            self.folds.append(dict(tr=tr, te=te, w=w, V=V, Kte=Kte_c, lam=lam, inner=inner))

    @staticmethod
    def _solve(f, Y):
        """Predictions for every lambda: returns [n_lam, n_te, m]."""
        ybar = Y.mean(0, keepdims=True)
        Yc = Y - ybar
        A = f["V"].T @ Yc                                   # [n, m]
        out = np.empty((len(f["lam"]), f["Kte"].shape[0], Y.shape[1]))
        for li, lam in enumerate(f["lam"]):
            alpha = f["V"] @ (A / (f["w"][:, None] + lam))
            out[li] = f["Kte"] @ alpha + ybar
        return out

    def predict_batch(self, Y, cross=None):
        """Y: [N, m] targets (column 0 conventionally the real one). Returns [N, m].

        cross: optional Gram [N_other, N] of *other-arm* test features against this arm's
        features; when given, the held-out rows are taken from the other arm instead.
        """
        Y = np.asarray(Y, float)
        out = np.zeros((self.K.shape[0], Y.shape[1]))
        for f in self.folds:
            # inner-LOSO lambda selection, per target column
            mse = np.zeros((len(LAM_MULT), Y.shape[1]))
            for inn in f["inner"]:
                P = self._solve(inn, Y[inn["tr"]])
                mse += ((P - Y[inn["te"]][None, :, :]) ** 2).mean(1)
            best = np.argmin(mse, axis=0)                     # [m]
            if cross is None:
                P = self._solve(f, Y[f["tr"]])
            else:
                fx = dict(f)
                Ktr = self.K[np.ix_(f["tr"], f["tr"])]
                r = Ktr.mean(0)
                mm = Ktr.mean()
                Kx = cross[np.ix_(f["te"], f["tr"])]
                fx["Kte"] = Kx - Kx.mean(1, keepdims=True) - r[None, :] + mm
                P = self._solve(fx, Y[f["tr"]])
            out[f["te"]] = P[best, :, np.arange(Y.shape[1])].T
        return out


# ------------------------------------------------------------------ features
def normalise(Xall, idx):
    """Grand-mean centre over all rows, divide by the mean vector norm; return the subset."""
    Xc = Xall - Xall.mean(0, keepdims=True)
    scale = np.linalg.norm(Xc, axis=1).mean() + 1e-12
    return (Xc / scale)[idx]


def selftest():
    """Synthetic checks on the estimator itself, before it touches any real stack.

    (1) pure-noise features must score ~0 with a permutation null centred on ~0;
    (2) features carrying the target must score high;
    (3) a feature that is an invertible linear map of another must score identically
        (scale invariance, sec 3.4);
    (4) residual-gain of features that ARE the floor's own inputs must be ~0 (the
        G_res(0) calibration, reproduced synthetically).
    """
    rng = np.random.default_rng(7)
    n, d = len(CELLS), 64
    Ynoise = np.column_stack([y] + [RNG.permutation(y) for _ in range(40)])
    Xn = rng.normal(size=(n, d))
    Mn = FoldMachine(Xn @ Xn.T, groups, d)
    Pn = Mn.predict_batch(Ynoise)
    s_noise = spearman_cols(Pn, y)
    Xs = np.column_stack([y + rng.normal(scale=0.5, size=n), rng.normal(size=(n, d - 1))])
    Ms = FoldMachine(Xs @ Xs.T, groups, d)
    s_sig = spearman(Ms.predict_batch(Ynoise[:, :1])[:, 0], y)
    A = rng.normal(size=(d, d))
    Xr = Xs @ A * 37.0
    s_rot = spearman(FoldMachine(Xr @ Xr.T, groups, d).predict_batch(Ynoise[:, :1])[:, 0], y)
    # (4) residual gain: a floor of realistic strength (rho ~0.95), then three layers --
    #     one identical to the floor (must give ~0), one strictly more informative (must
    #     give >0), one pure noise (must give ~0). This is the synthetic twin of the
    #     G_res(0) calibration check.
    noise = rng.normal(size=(n, d - 1))
    Ffl = np.column_stack([y + rng.normal(scale=0.8, size=n), noise])
    Xbet = np.column_stack([y + rng.normal(scale=0.2, size=n), noise])
    Mf = FoldMachine(Ffl @ Ffl.T, groups, d)
    Pf = Mf.predict_batch(Ynoise[:, :1])
    rr = (Ynoise[:, :1] - Pf)
    g_self = spearman(Mf.predict_batch(rr)[:, 0], rr[:, 0])
    g_better = spearman(FoldMachine(Xbet @ Xbet.T, groups, d).predict_batch(rr)[:, 0], rr[:, 0])
    g_noise = spearman(Mn.predict_batch(rr)[:, 0], rr[:, 0])
    out = dict(noise_rho=float(s_noise[0]), noise_null_mean=float(s_noise[1:].mean()),
               noise_null_sd=float(s_noise[1:].std()), signal_rho=float(s_sig),
               rotated_rescaled_rho=float(s_rot), scale_invariance_diff=float(abs(s_rot - s_sig)),
               floor_rho=float(spearman(Pf[:, 0], y)),
               g_self=float(g_self), g_better=float(g_better), g_noise=float(g_noise))
    print("selftest:", json.dumps(out, indent=1), flush=True)
    ok = (abs(out["noise_rho"]) < 0.25 and abs(out["noise_null_mean"]) < 0.08
          and out["signal_rho"] > 0.8 and out["scale_invariance_diff"] < 0.05
          and abs(out["g_self"]) < 0.10 and abs(out["g_noise"]) < 0.10 and out["g_better"] > 0.2)
    print("selftest", "PASS" if ok else "FAIL", flush=True)
    return out, ok


SELFTEST, _st_ok = selftest()
if os.environ.get("CG_SELFTEST_ONLY"):
    raise SystemExit(0 if _st_ok else 1)

print("loading stacks", flush=True)
Z = {arm: np.load(f"results/clock_gain_v1_stacks_{arm}.npz") for arm in ARMS}
LEX = np.load("results/clock_gain_v1_lex.npz")

sub216 = [ALL_CELLS.index(c) for c in CELLS]
RAW = {}          # arm -> [240, 29, d] raw (unnormalised) state vectors
for arm in ARMS:
    RAW[arm] = np.stack([Z[arm][f"exp/{s}/{t}/p{p}"] for (s, t, p) in ALL_CELLS]).astype(np.float32)
NLAY, DMODEL = RAW["A"].shape[1], RAW["A"].shape[2]
assert NLAY == NL, NLAY

EMB = {arm: np.stack([LEX[f"emb/{arm}/{s}/{t}/p{p}"] for (s, t, p) in ALL_CELLS]).astype(np.float64)
       for arm in ("A", "B")}
BOW = {arm: np.stack([LEX[f"bow/{arm}/{s}/{t}/p{p}"] for (s, t, p) in ALL_CELLS]).astype(np.float64)
       for arm in ("A", "B")}
NVOC = BOW["A"].shape[1]

R = {"_meta": dict(model="Qwen/Qwen2.5-1.5B", grid=a.grid, n_layers=NL, d_model=DMODEL,
                   n_cells=len(CELLS), n_perm=a.nperm, seed=a.seed, vocab=int(NVOC),
                   lam_mult=list(LAM_MULT)), "selftest": SELFTEST}

# within-subject permutations of y (column 0 = the real target)
Y = np.empty((len(CELLS), a.nperm + 1))
Y[:, 0] = y
for j in range(1, a.nperm + 1):
    col = y.copy()
    for u in np.unique(groups):
        m = groups == u
        col[m] = RNG.permutation(y[m])
    Y[:, j] = col

# ------------------------------------------------------------------ the floor
print("floor", flush=True)
Femb = normalise(EMB["A"], sub216)
Fbow = normalise(BOW["A"], sub216)
floor_specs = {
    "F-emb": (Femb @ Femb.T, DMODEL),
    "F-bow": (Fbow @ Fbow.T, NVOC),
    "F-bow+emb": (Femb @ Femb.T + Fbow @ Fbow.T, DMODEL + NVOC),
}
floor = {}
floor_pred = {}
for name, (K, d) in floor_specs.items():
    M = FoldMachine(K, groups, d)
    P = M.predict_batch(Y)
    floor_pred[name] = P
    rho_all = spearman_cols(P, y)      # col 0 real, rest permutation null (vs real y)
    _, ae = grid_mae(P[:, 0])
    floor[name] = dict(rho=float(rho_all[0]), rho_null_mean=float(rho_all[1:].mean()),
                       rho_z=zscore(rho_all[0], rho_all[1:]), r2=r2(P[:, 0], y),
                       mae=float(ae.mean()))
    print(f"  {name:10s} rho {floor[name]['rho']:+.3f}  z {floor[name]['rho_z']:+.1f}  "
          f"MAE {floor[name]['mae']:.2f}", flush=True)
rho_lex = max(floor[n]["rho"] for n in floor)
R["floor"] = dict(per_readout=floor, rho_lex=float(rho_lex), best=max(floor, key=lambda n: floor[n]["rho"]))

# lexical residual, computed for the real target and for every permutation
# (the permuted-y floor is refit, so the null passes through the whole pipeline)
PFL = floor_pred["F-bow+emb"]
Rres = Y - PFL                                            # [216, nperm+1]
r_real = Rres[:, 0]
_, ae_lex = grid_mae(PFL[:, 0])
R["floor"]["residual_std"] = float(r_real.std())

# ------------------------------------------------------------------ per-layer sweep
def layer_features(arm, l, zscore_dims=False):
    Xall = RAW[arm][:, l, :].astype(np.float64)
    if zscore_dims:
        Xc = Xall - Xall.mean(0, keepdims=True)
        sd = Xc.std(0, keepdims=True) + 1e-9
        return (Xc / sd)[sub216]
    return normalise(Xall, sub216)


per_layer = {arm: {k: [] for k in
                   ("rho", "rho_null_mean", "rho_z", "mae", "G_res", "G_res_null_mean",
                    "G_res_z", "R2_res", "drho", "G_part", "G_part_null_mean", "G_part_z")}
             for arm in ("A", "B", "D")}
gain_pred = {}    # (arm, layer) -> residual-gain predictions, real column only
rho_pred = {}
t_start = time.time()
for l in range(NL):
    for arm in ("A", "B", "D"):
        X = layer_features(arm, l)
        M = FoldMachine(X @ X.T, groups, DMODEL)
        P = M.predict_batch(Y)                     # target = y (and its permutations)
        Q = M.predict_batch(Rres)                  # target = lexical residual
        rho_all = spearman_cols(P, y)
        g_all = spearman_cols(Q, Rres)             # each column scored against its own residual
        gp_all = np.array([partial_spearman(P[:, j], Y[:, j], PFL[:, j]) for j in range(P.shape[1])])
        d = per_layer[arm]
        d["G_part"].append(float(gp_all[0]))
        d["G_part_null_mean"].append(float(gp_all[1:].mean()))
        d["G_part_z"].append(zscore(gp_all[0], gp_all[1:]))
        d["rho"].append(float(rho_all[0]))
        d["rho_null_mean"].append(float(rho_all[1:].mean()))
        d["rho_z"].append(zscore(rho_all[0], rho_all[1:]))
        d["G_res"].append(float(g_all[0]))
        d["G_res_null_mean"].append(float(g_all[1:].mean()))
        d["G_res_z"].append(zscore(g_all[0], g_all[1:]))
        d["R2_res"].append(r2(Q[:, 0], r_real))
        d["drho"].append(float(rho_all[0] - rho_lex))
        _, ae = grid_mae(P[:, 0])
        d["mae"].append(float(ae.mean()))
        if arm == "A":
            gain_pred[l] = Q[:, 0]
            rho_pred[l] = P[:, 0]
    print(f"  layer {l:2d}  A rho {per_layer['A']['rho'][-1]:+.3f} G_res {per_layer['A']['G_res'][-1]:+.3f}"
          f" (z {per_layer['A']['G_res_z'][-1]:+.1f})  G_part {per_layer['A']['G_part'][-1]:+.3f}"
          f" (z {per_layer['A']['G_part_z'][-1]:+.1f})   B G_res {per_layer['B']['G_res'][-1]:+.3f}"
          f"   [{time.time() - t_start:.0f}s]", flush=True)
R["per_layer"] = per_layer

GA = np.array(per_layer["A"]["G_res"])
LSTAR = int(np.argmax(GA))
GPA = np.array(per_layer["A"]["G_part"])
LSTAR_P = int(np.argmax(GPA))
R["L_star"] = LSTAR
R["L_star_partial"] = LSTAR_P
R["floor_calibration"] = dict(
    slope_yhat_on_y=float(np.polyfit(y, PFL[:, 0], 1)[0]),
    slope_y_on_yhat=float(np.polyfit(PFL[:, 0], y, 1)[0]),
    corr_resid_y=float(np.corrcoef(r_real, y)[0, 1]))

# robustness: per-dimension z-scoring (sec 3.4.4)
gz = []
for l in range(NL):
    X = layer_features("A", l, zscore_dims=True)
    M = FoldMachine(X @ X.T, groups, DMODEL)
    Q = M.predict_batch(Rres[:, :1])
    gz.append(spearman(Q[:, 0], r_real))
R["robustness_zscored"] = dict(G_res=gz, L_star=int(np.argmax(gz)),
                               agrees_within_3=bool(abs(int(np.argmax(gz)) - LSTAR) <= 3))

# ------------------------------------------------------------------ calibration (sec 3.4.6)
cal = {}
cal["rho_A0_vs_Femb"] = dict(rho_A0=per_layer["A"]["rho"][0], rho_Femb=floor["F-emb"]["rho"],
                             diff=abs(per_layer["A"]["rho"][0] - floor["F-emb"]["rho"]),
                             pass_=abs(per_layer["A"]["rho"][0] - floor["F-emb"]["rho"]) <= 0.01)
cal["rho_B0_vs_A0"] = dict(rho_B0=per_layer["B"]["rho"][0], rho_A0=per_layer["A"]["rho"][0],
                           diff=abs(per_layer["B"]["rho"][0] - per_layer["A"]["rho"][0]),
                           pass_=abs(per_layer["B"]["rho"][0] - per_layer["A"]["rho"][0]) <= 0.03)
cal["G_res0"] = {arm: per_layer[arm]["G_res"][0] for arm in ("A", "B", "D")}
cal["G_res0"]["pass_"] = all(abs(per_layer[arm]["G_res"][0]) <= 0.05 for arm in ("A", "B", "D"))
cal["G_part0"] = {arm: per_layer[arm]["G_part"][0] for arm in ("A", "B", "D")}
cal["G_part0"]["pass_"] = all(abs(per_layer[arm]["G_part"][0]) <= 0.05 for arm in ("A", "B", "D"))
wp = max(abs(np.array(per_layer[arm]["G_part_null_mean"])).max() for arm in ("A", "B", "D"))
cal["null_mean_G_part"] = dict(worst_abs=float(wp), pass_=bool(wp <= 0.05))
worst = max(abs(np.array(per_layer[arm]["G_res_null_mean"])).max() for arm in ("A", "B", "D"))
cal["null_mean_G_res"] = dict(worst_abs=float(worst), pass_=bool(worst <= 0.05))
# extra: layer-0 arm A really is the static mean embedding (sec 0.1 identity)
c0 = [float(np.dot(RAW["A"][i, 0], EMB["A"][i]) /
            (np.linalg.norm(RAW["A"][i, 0]) * np.linalg.norm(EMB["A"][i]) + 1e-12))
      for i in range(len(ALL_CELLS))]
cal["layer0_is_mean_embedding_cos"] = dict(mean=float(np.mean(c0)), min=float(np.min(c0)))
cal["all_pass"] = bool(cal["rho_A0_vs_Femb"]["pass_"] and cal["rho_B0_vs_A0"]["pass_"]
                       and cal["G_res0"]["pass_"] and cal["null_mean_G_res"]["pass_"])
R["calibration"] = cal
print("calibration:", json.dumps({k: (v if not isinstance(v, dict) else
                                      {kk: vv for kk, vv in v.items() if kk == "pass_" or kk == "diff"})
                                  for k, v in cal.items()}), flush=True)

# ------------------------------------------------------------------ S2: order dependence
print("S2 label-swap null", flush=True)
XA = layer_features("A", LSTAR)
XB = layer_features("B", LSTAR)
G_struct = per_layer["A"]["rho"][LSTAR] - per_layer["B"]["rho"][LSTAR]
G_struct_res = per_layer["A"]["G_res"][LSTAR] - per_layer["B"]["G_res"][LSTAR]
G_struct_part = per_layer["A"]["G_part"][LSTAR] - per_layer["B"]["G_part"][LSTAR]
swap_rho, swap_res, swap_part = [], [], []
rng2 = np.random.default_rng(a.seed + 1)
for _ in range(a.nperm):
    m = rng2.random(len(CELLS)) < 0.5
    Xa = np.where(m[:, None], XB, XA)
    Xb = np.where(m[:, None], XA, XB)
    Ma = FoldMachine(Xa @ Xa.T, groups, DMODEL)
    Mb = FoldMachine(Xb @ Xb.T, groups, DMODEL)
    Pa, Pb = Ma.predict_batch(Y[:, :1]), Mb.predict_batch(Y[:, :1])
    Qa, Qb = Ma.predict_batch(Rres[:, :1]), Mb.predict_batch(Rres[:, :1])
    swap_rho.append(spearman(Pa[:, 0], y) - spearman(Pb[:, 0], y))
    swap_res.append(spearman(Qa[:, 0], r_real) - spearman(Qb[:, 0], r_real))
    swap_part.append(partial_spearman(Pa[:, 0], y, PFL[:, 0]) - partial_spearman(Pb[:, 0], y, PFL[:, 0]))
R["S2"] = dict(L_star=LSTAR, G_struct=float(G_struct), G_struct_res=float(G_struct_res),
               G_struct_part=float(G_struct_part),
               z_rho=zscore(G_struct, swap_rho), z_res=zscore(G_struct_res, swap_res),
               z_part=zscore(G_struct_part, swap_part),
               swap_null_sd_rho=float(np.std(swap_rho)), swap_null_sd_res=float(np.std(swap_res)),
               swap_null_sd_part=float(np.std(swap_part)),
               pass_=bool(G_struct >= 0.08 and G_struct_res >= 0.10 and zscore(G_struct_res, swap_res) >= 3))
print("  S2", json.dumps(R["S2"]), flush=True)

# ------------------------------------------------------------------ S3: per-dt gain
mae_lex_dt, mae_A_dt = [], []
_, ae_A = grid_mae(rho_pred[LSTAR])
for k in range(len(DT)):
    m = dt_index == k
    mae_lex_dt.append(float(ae_lex[m].mean()))
    mae_A_dt.append(float(ae_A[m].mean()))
a_lex = [1 - v / 8 for v in mae_lex_dt]
g_dt = [mae_lex_dt[k] - mae_A_dt[k] for k in range(len(DT))]
top3 = [DT[i] for i in np.argsort(g_dt)[::-1][:3]]
mid = {"6months", "1year", "10years", "100years"}
R["S3"] = dict(deltas=DT, mae_lex=mae_lex_dt, mae_A=mae_A_dt, a_lex=a_lex, g=g_dt,
               spearman_alex_g=spearman(a_lex, g_dt), top3=top3,
               top3_in_middle=bool(set(top3) <= mid),
               pass_=bool(spearman(a_lex, g_dt) <= -0.5 and set(top3) <= mid))
print("  S3", R["S3"]["spearman_alex_g"], top3, flush=True)

# ------------------------------------------------------------------ S4: arm C
print("S4 transfer + cosine", flush=True)
def shared_dirs(arm, l):
    Xall = RAW[arm][:, l, :]
    H = {}
    for s in S:
        for t in ["t0"] + DT:
            H[(s, t)] = np.mean([Xall[ALL_CELLS.index((s, t, p))] for p in range(NP_)], axis=0)
    return {t: np.mean([H[(s, t)] - H[(s, "t0")] for s in S], axis=0) for t in DT}


def cosv(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


cos_by_layer = {}
for l in range(NL):
    dA, dC = shared_dirs("A", l), shared_dirs("C", l)
    cos_by_layer[l] = {t: cosv(dC[t], dA[t]) for t in DT}
far = ["100years", "1000years", "10000years", "1000000years"]
cos_far = float(np.mean([cos_by_layer[LSTAR][t] for t in far]))

# transfer: arm-A ridge (LOSO) evaluated on arm-C features of the held-out subject.
XAall = RAW["A"][:, LSTAR, :]
mu = XAall.mean(0, keepdims=True)
sc = np.linalg.norm(XAall - mu, axis=1).mean() + 1e-12
XA_n = ((XAall - mu) / sc)[sub216]
XC_n = ((RAW["C"][:, LSTAR, :] - mu) / sc)[sub216]        # arm C mapped with arm A's normalisation
MA = FoldMachine(XA_n @ XA_n.T, groups, DMODEL)
PC = MA.predict_batch(Y, cross=XC_n @ XA_n.T)
rho_AC = spearman(PC[:, 0], y)
rho_AC_null = spearman_cols(PC[:, 1:], y)
per_sub = {s: spearman(PC[groups == S.index(s), 0], y[groups == S.index(s)]) for s in S}
per_sub_cos = {}
for s in S:
    dA = {t: np.mean([RAW["A"][ALL_CELLS.index((s, t, p)), LSTAR] for p in range(NP_)], axis=0)
             - np.mean([RAW["A"][ALL_CELLS.index((s, "t0", p)), LSTAR] for p in range(NP_)], axis=0)
          for t in DT}
    dC = {t: np.mean([RAW["C"][ALL_CELLS.index((s, t, p)), LSTAR] for p in range(NP_)], axis=0)
             - np.mean([RAW["C"][ALL_CELLS.index((s, "t0", p)), LSTAR] for p in range(NP_)], axis=0)
          for t in DT}
    per_sub_cos[s] = float(np.mean([cosv(dC[t], dA[t]) for t in far]))
R["S4"] = dict(L_star=LSTAR, rho_A_to_C=float(rho_AC), rho_A_to_C_z=zscore(rho_AC, rho_AC_null),
               rho_A_to_C_null_mean=float(rho_AC_null.mean()),
               cos_far=cos_far, cos_by_layer={str(l): cos_by_layer[l] for l in range(NL)},
               per_subject_rho=per_sub, per_subject_cos_far=per_sub_cos,
               verdict=("partial_positive" if (rho_AC >= 0.4 and cos_far >= 0.4)
                        else "hour32_null_stands" if (rho_AC <= 0.2 and cos_far <= 0.25)
                        else "indeterminate"))
print("  S4 rho_A->C", rho_AC, "cos_far", cos_far, R["S4"]["verdict"], flush=True)

# ------------------------------------------------------------------ copy inflation, S1, verdicts
R["copy_inflation"] = dict(rho_D14=per_layer["D"]["rho"][14], rho_A14=per_layer["A"]["rho"][14],
                           diff=per_layer["D"]["rho"][14] - per_layer["A"]["rho"][14],
                           rho_D_Lstar=per_layer["D"]["rho"][LSTAR], rho_A_Lstar=per_layer["A"]["rho"][LSTAR])
peak = float(GA[LSTAR])
R["S1"] = dict(L_star=LSTAR, frac_depth=LSTAR / 28.0, peak=peak, final=float(GA[-1]),
               final_over_peak=float(GA[-1] / (peak + 1e-12)),
               early_frac=float(GA[3] / (peak + 1e-12)),
               pass_=bool(8 <= LSTAR <= 20 and GA[-1] <= 0.75 * peak),
               flat=bool(GA[3] >= 0.9 * peak))
maxz = max(per_layer["A"]["G_res_z"])
R["S1_partial"] = dict(L_star=LSTAR_P, frac_depth=LSTAR_P / 28.0, peak=float(GPA[LSTAR_P]),
                       final=float(GPA[-1]), final_over_peak=float(GPA[-1] / (GPA[LSTAR_P] + 1e-12)),
                       early_frac=float(GPA[3] / (GPA[LSTAR_P] + 1e-12)),
                       pass_=bool(8 <= LSTAR_P <= 20 and GPA[-1] <= 0.75 * GPA[LSTAR_P]))
R["kill"] = dict(max_G_res_A=peak, max_z=float(maxz),
                 max_G_part_A=float(GPA.max()), max_G_part_z=float(max(per_layer["A"]["G_part_z"])),
                 kill=bool(peak < 0.10 or maxz < 2),
                 partial_kill=bool(peak >= 0.10 and maxz >= 3 and abs(G_struct) <= 0.03
                                   and not R["S2"]["pass_"]))
P2 = bool(0.85 <= per_layer["A"]["rho"][LSTAR] <= 0.95 and peak >= 0.30 and maxz >= 4
          and per_layer["A"]["drho"][LSTAR] >= 0.12)
R["predictions"] = dict(P2=P2, P3=R["S1"]["pass_"], P4=R["S2"]["pass_"], P5=R["S3"]["pass_"],
                        P0=cal["all_pass"],
                        P1=bool(0.65 <= rho_lex <= 0.80),
                        P6=bool(R["copy_inflation"]["diff"] >= 0.03),
                        P7=bool(R["S4"]["verdict"] == "partial_positive"))
R["gate_piece2"] = bool(P2 and R["S1"]["pass_"] and R["S2"]["pass_"])

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(R, open(a.out, "w"), indent=1)
print("wrote", a.out, flush=True)

# ------------------------------------------------------------------ figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.figdir, exist_ok=True)
    X = np.arange(NL)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for arm, c in (("A", "C0"), ("B", "C1"), ("D", "C2")):
        ax[0].plot(X, per_layer[arm]["G_res"], marker="o", ms=3, color=c, label=f"arm {arm} G_res")
        ax[0].plot(X, per_layer[arm]["G_part"], marker="s", ms=3, color=c, ls="--", label=f"arm {arm} G_part")
        nm = np.array(per_layer[arm]["G_res_null_mean"])
        ax[0].plot(X, nm, color=c, ls=":", lw=0.8)
    ax[0].axvline(LSTAR, color="k", ls="--", lw=0.8)
    ax[0].set_xlabel("residual point"); ax[0].set_ylabel("G_res (Spearman)")
    ax[0].set_title("residual gain over the lexical floor"); ax[0].legend(fontsize=8)
    for arm, c in (("A", "C0"), ("B", "C1"), ("D", "C2")):
        ax[1].plot(X, per_layer[arm]["rho"], marker="o", ms=3, color=c, label=f"arm {arm}")
    ax[1].axhline(rho_lex, color="k", ls="--", lw=0.8, label="rho_lex")
    ax[1].set_xlabel("residual point"); ax[1].set_ylabel("Spearman(yhat, y)")
    ax[1].set_title("LOSO ridge decoding of log dt"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.figdir}/clock_gain_depth.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(len(DT)), a_lex, marker="o", label="a_lex (floor accuracy)")
    ax[0].plot(range(len(DT)), g_dt, marker="s", label="g (gain, grid steps)")
    ax[0].set_xticks(range(len(DT))); ax[0].set_xticklabels(DT, rotation=60, fontsize=7)
    ax[0].legend(fontsize=8); ax[0].set_title(f"S3: floor accuracy vs gain (rho={R['S3']['spearman_alex_g']:+.2f})")
    ax[1].plot(X, [np.mean([cos_by_layer[l][t] for t in far]) for l in range(NL)], marker="o", ms=3,
               label="cos(shared_C, shared_A), far dt")
    ax[1].axhline(0.4, color="k", ls="--", lw=0.8)
    ax[1].set_xlabel("residual point"); ax[1].legend(fontsize=8); ax[1].set_title("S4: arm C vs arm A direction")
    fig.tight_layout(); fig.savefig(f"{a.figdir}/clock_gain_s3_s4.png", dpi=130); plt.close(fig)
    print("wrote figures", flush=True)
except Exception as e:                                        # figures are not load-bearing
    print("figure step failed:", e, flush=True)
