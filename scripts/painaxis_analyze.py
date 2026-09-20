"""Tier A analysis: their pain-vector arithmetic, ported verbatim, on Qwen2.5-1.5B-Instruct.

Sources (the authority, not their prose):
  Pain-axis/scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py
      compute_pain_vector, compute_auc, compute_layer_curves_kfold, create_auc_bars
  Pain-axis/scripts/3.3_validation/08_s1_auc.py
      auc_table, kfold_curve, and the s1_kfold_summary row shape
  Pain-axis/scripts/3.2_pain_vectors/02_build_control_vectors.py
      denoise_basis / control_vec for the control-category directions

sklearn (1.9.1) is installed here, so PCA, roc_auc_score and KFold are THEIR calls, not
re-implementations. scripts/painaxis_numpy_impl.py holds independent numpy versions used only
by tests/test_painaxis_port.py to check the arithmetic.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results/painaxis_tierA"

# their constants, read out of their file
N_FOLDS = 5
RANDOM_SEED = 42
DENOISE_VARIANCE = 0.5
PAIN_CATEGORIES = ["A1", "A2", "A3", "A4", "A5"]
CONTROL_CATEGORIES = ["B", "C1", "C2", "D", "E"]
NEUTRAL_CATEGORY = "D"
S_SETS = ["S1_1P", "S2_1P", "ControlSupplement_1P"]
EXTRACTIONS = ["final_token", "mean"]


# --------------------------------------------------------------------------- their functions
def compute_pain_vector(acts, cats, baseline="all_controls", denoise=True):
    acts_np = np.asarray(acts)
    cats_np = np.array(cats)
    if np.isnan(acts_np).any() or np.isinf(acts_np).any():
        denoise = False
        acts_np = np.where(np.isinf(acts_np), np.nan, acts_np)
    pain_mask = np.isin(cats_np, PAIN_CATEGORIES)
    pain_mean = np.nanmean(acts_np[pain_mask], axis=0)
    if baseline == "neutral":
        control_mask = cats_np == NEUTRAL_CATEGORY
    else:
        control_mask = np.isin(cats_np, CONTROL_CATEGORIES)
    control_acts = acts_np[control_mask]
    control_mean = np.nanmean(control_acts, axis=0)
    vec = pain_mean - control_mean
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    if denoise and len(control_acts) > 1:
        pca = PCA()
        pca.fit(control_acts - control_mean)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_comp = min(np.searchsorted(cumvar, DENOISE_VARIANCE) + 1, len(pca.components_))
        for d in pca.components_[:n_comp]:
            vec = vec - np.dot(vec, d) * d
    return vec


def compute_auc(acts, cats, pain_vector):
    acts_np = np.asarray(acts)
    cats_np = np.array(cats)
    vec_norm = pain_vector / (np.linalg.norm(pain_vector) + 1e-8)
    proj = acts_np @ vec_norm
    pain_mask = np.isin(cats_np, PAIN_CATEGORIES)
    control_mask = np.isin(cats_np, CONTROL_CATEGORIES)
    if pain_mask.sum() == 0 or control_mask.sum() == 0:
        return np.nan
    labels = np.concatenate([np.ones(pain_mask.sum()), np.zeros(control_mask.sum())])
    scores = np.concatenate([proj[pain_mask], proj[control_mask]])
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, scores)


def auc_table(acts, cats, vec):
    """08_s1_auc.auc_table: pain vs all controls, then pain vs each control category."""
    cats = np.array(cats)
    v = vec / (np.linalg.norm(vec) + 1e-8)
    proj = np.asarray(acts) @ v
    pain = proj[np.isin(cats, PAIN_CATEGORIES)]
    out = {"ALL": roc_auc_score(
        np.r_[np.ones(len(pain)), np.zeros((np.isin(cats, CONTROL_CATEGORIES)).sum())],
        np.r_[pain, proj[np.isin(cats, CONTROL_CATEGORIES)]])}
    for c in CONTROL_CATEGORIES:
        cp = proj[cats == c]
        if len(cp):
            out[c] = roc_auc_score(np.r_[np.ones(len(pain)), np.zeros(len(cp))], np.r_[pain, cp])
    return out


def kfold_curve(acts_all, cats, sets, layers, per_category=False):
    """01.compute_layer_curves_kfold / 08.kfold_curve: KFold over the SENTENCE SETS, vector fitted
    on the training sets only, AUC scored on the held-out sets, averaged over folds."""
    uniq = sorted(set(sets.tolist()))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for L in layers:
        acts = acts_all[:, L, :]
        f_all, f_neu = [], []
        f_cat = {c: [] for c in CONTROL_CATEGORIES}
        for tr, te in kf.split(uniq):
            trm = np.isin(sets, [uniq[i] for i in tr])
            tem = np.isin(sets, [uniq[i] for i in te])
            if trm.sum() == 0 or tem.sum() == 0:
                continue
            va = compute_pain_vector(acts[trm], cats[trm], "all_controls")
            vn = compute_pain_vector(acts[trm], cats[trm], "neutral")
            a = compute_auc(acts[tem], cats[tem], va)
            n = compute_auc(acts[tem], cats[tem], vn)
            if not np.isnan(a):
                f_all.append(a)
            if not np.isnan(n):
                f_neu.append(n)
            if per_category:
                t = auc_table(acts[tem], cats[tem], va)
                for c in CONTROL_CATEGORIES:
                    if c in t:
                        f_cat[c].append(t[c])
        row = dict(layer=int(L),
                   auc_vs_all_controls=float(np.mean(f_all)) if f_all else np.nan,
                   auc_vs_neutral=float(np.mean(f_neu)) if f_neu else np.nan,
                   auc_std=float(np.std(f_all)) if f_all else np.nan)
        if per_category:
            for c in CONTROL_CATEGORIES:
                row[f"heldout_vs_{c}"] = float(np.mean(f_cat[c])) if f_cat[c] else np.nan
        rows.append(row)
    return rows


def denoise_basis(neutral_acts, neutral_mean):
    """02_build_control_vectors.denoise_basis (numpy SVD, theirs)."""
    X = np.nan_to_num(neutral_acts - neutral_mean, nan=0.0, posinf=0.0, neginf=0.0)
    if len(X) < 2:
        return np.zeros((0, X.shape[1]))
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    cumvar = np.cumsum(var) / var.sum()
    n_comp = min(int(np.searchsorted(cumvar, DENOISE_VARIANCE)) + 1, len(Vt))
    return Vt[:n_comp]


def project_out(vec, basis):
    for d in basis:
        vec = vec - np.dot(vec, d) * d
    return vec


# --------------------------------------------------------------------------- io helpers
def load(name):
    z = np.load(OUT / f"acts_{name}.npz", allow_pickle=False)
    return {k: z[k] for k in ("final_token", "mean", "mean_nobos", "categories", "sets")}


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    data = {n: load(n) for n in ["S1_1P", "S1_3P", "S2_1P", "S2_3P", "ControlSupplement_1P"]}
    n_layers = data["S2_1P"]["final_token"].shape[1]
    layers = list(range(n_layers))
    summary = {"model": "Qwen/Qwen2.5-1.5B-Instruct", "n_layers": n_layers}

    # ---- 1. layer curves, S2 (their 01) and S1 (their 08), both extractions -------------
    curve_rows, best = [], {}
    for ext in EXTRACTIONS:
        for ds in ["S2_1P", "S2_3P", "S1_1P", "S1_3P"]:
            d = data[ds]
            for r in kfold_curve(d[ext], d["categories"], d["sets"], layers):
                curve_rows.append(dict(model="Qwen2.5-1.5B-Instruct", extraction=ext, dataset=ds, **r))
    write_csv(OUT / "layer_curves.csv", curve_rows)

    def mean_curve(ext, dss):
        out = {}
        for L in layers:
            vals = [r["auc_vs_all_controls"] for r in curve_rows
                    if r["extraction"] == ext and r["dataset"] in dss and r["layer"] == L]
            out[L] = float(np.mean(vals))
        return out

    for ext in EXTRACTIONS:
        s2 = mean_curve(ext, ["S2_1P", "S2_3P"])
        s1 = mean_curve(ext, ["S1_1P", "S1_3P"])
        best[ext] = {"s2_best_layer": int(max(s2, key=s2.get)), "s2_best_auc": max(s2.values()),
                     "s1_best_layer": int(max(s1, key=s1.get)), "s1_best_auc": max(s1.values())}
    summary["best"] = best
    print(json.dumps(best, indent=2), flush=True)

    # ---- 2. s1_kfold_summary shape (their 08) ------------------------------------------
    s1_rows = []
    for ext in EXTRACTIONS:
        L2 = best[ext]["s2_best_layer"]
        s1 = mean_curve(ext, ["S1_1P", "S1_3P"])
        bL = best[ext]["s1_best_layer"]

        def at(ds, L):
            return [r["auc_vs_all_controls"] for r in curve_rows
                    if r["extraction"] == ext and r["dataset"] == ds and r["layer"] == L][0]

        s1_rows.append(dict(
            model="Qwen_2.5_1.5B_instruct", extraction=ext, s2_layer=L2,
            s1_heldout_auc_at_s2_layer=s1[L2], s1_best_layer=bL,
            s1_heldout_auc_at_best_layer=s1[bL],
            s1_1P_heldout_at_best=at("S1_1P", bL), s1_3P_heldout_at_best=at("S1_3P", bL),
            s2_1P_heldout_at_s2_best=at("S2_1P", L2), s2_3P_heldout_at_s2_best=at("S2_3P", L2)))
    write_csv(OUT / "s1_kfold_summary.csv", s1_rows)

    # ---- 3. per-control-category AUC at the chosen layer ---------------------------------
    cat_rows = []
    for ext in EXTRACTIONS:
        L = best[ext]["s2_best_layer"]
        s2_vec = compute_pain_vector(data["S2_1P"][ext][:, L, :], data["S2_1P"]["categories"])
        for ds in ["S2_1P", "S2_3P"]:
            t = auc_table(data[ds][ext][:, L, :], data[ds]["categories"], s2_vec)
            cat_rows.append(dict(extraction=ext, layer=L, dataset=ds, kind="in_sample_S2_1P_vector", **t))
        # held-out variant: same categories, but the vector never sees the scored sentences
        for ds in ["S2_1P", "S2_3P"]:
            d = data[ds]
            hr = [r for r in kfold_curve(d[ext], d["categories"], d["sets"], [L], per_category=True)][0]
            cat_rows.append(dict(extraction=ext, layer=L, dataset=ds, kind="heldout_kfold",
                                 ALL=hr["auc_vs_all_controls"],
                                 **{c: hr[f"heldout_vs_{c}"] for c in CONTROL_CATEGORIES}))
    write_csv(OUT / "per_category_auc.csv", cat_rows,
              fields=["extraction", "layer", "dataset", "kind", "ALL"] + CONTROL_CATEGORIES)

    # ---- 4/5. vectors at the final_token layer: readout and cosines ----------------------
    ext = "final_token"
    L = best[ext]["s2_best_layer"]
    cats = data["S2_1P"]["categories"]
    A = data["S2_1P"][ext][:, L, :]
    raw = (np.nanmean(A[np.isin(cats, PAIN_CATEGORIES)], axis=0)
           - np.nanmean(A[np.isin(cats, CONTROL_CATEGORIES)], axis=0))
    denoised = compute_pain_vector(A, cats)
    ctrl_acts = A[np.isin(cats, CONTROL_CATEGORIES)]
    pca = PCA()
    pca.fit(ctrl_acts - np.nanmean(ctrl_acts, axis=0))
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(min(np.searchsorted(cum, DENOISE_VARIANCE) + 1, len(pca.components_)))
    summary["denoise"] = {"layer": L, "n_components_removed": n_comp,
                          "cumvar_at_n": float(cum[n_comp - 1]),
                          "norm_raw": float(np.linalg.norm(raw)),
                          "norm_denoised": float(np.linalg.norm(denoised)),
                          "cos_raw_denoised": float(raw @ denoised / (np.linalg.norm(raw) * np.linalg.norm(denoised)))}

    # control-category vectors, 02_build_control_vectors recipe, at this layer
    def rows_of(ds, cs=None):
        a = data[ds][ext][:, L, :]
        if cs is None:
            return a
        return a[np.isin(data[ds]["categories"], cs)]

    neutral = np.concatenate([rows_of(ds, ["D"]) for ds in S_SETS])
    neutral_mean = np.nanmean(np.where(np.isinf(neutral), np.nan, neutral), axis=0)
    basis = denoise_basis(neutral, neutral_mean)

    def control_vec(acts):
        v = np.nanmean(np.where(np.isinf(acts), np.nan, acts), axis=0) - neutral_mean
        return project_out(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), basis)

    ctrl_vecs = {
        "fear_B": control_vec(np.concatenate([rows_of(ds, ["B"]) for ds in S_SETS])),
        "negemotion_C1": control_vec(np.concatenate([rows_of(ds, ["C1"]) for ds in S_SETS])),
        "negworld_C2": control_vec(np.concatenate([rows_of(ds, ["C2"]) for ds in S_SETS])),
        "bodysens_E": control_vec(np.concatenate([rows_of(ds, ["E"]) for ds in S_SETS])),
    }
    s1_vec = compute_pain_vector(data["S1_1P"][ext][:, L, :], data["S1_1P"]["categories"])

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    cos_rows = [dict(a="s2_pain", b=k, cosine=cos(denoised, v)) for k, v in ctrl_vecs.items()]
    cos_rows.append(dict(a="s2_pain", b="s1_pain", cosine=cos(denoised, s1_vec)))
    cos_rows.append(dict(a="s2_pain_raw", b="s2_pain_denoised", cosine=cos(raw, denoised)))
    names = list(ctrl_vecs)
    for i, k in enumerate(names):
        for k2 in names[i + 1:]:
            cos_rows.append(dict(a=k, b=k2, cosine=cos(ctrl_vecs[k], ctrl_vecs[k2])))
    write_csv(OUT / "cosines.csv", cos_rows)

    # unembedding readout, before and after denoising
    from lsx.model import LM
    lm = LM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device="cpu", dtype=torch.float32)
    read = {}
    for nm, v in [("raw", raw), ("denoised", denoised), ("s1_denoised", s1_vec)]:
        t = torch.tensor(v, dtype=torch.float32)
        read[nm] = {"promoted": lm.unembed(t, k=20), "suppressed": lm.unembed(-t, k=20)}
    summary["readout_layer"] = L
    summary["readout"] = read
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    for nm in read:
        print(nm, "PROMOTED", [t for t, _ in read[nm]["promoted"]], flush=True)
        print(nm, "SUPPRESSED", [t for t, _ in read[nm]["suppressed"]], flush=True)
    print("ANALYSIS DONE", flush=True)


if __name__ == "__main__":
    main()
