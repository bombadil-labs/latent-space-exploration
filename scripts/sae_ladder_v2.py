"""Abstraction ladder v2: more layers, a broad reference corpus, and the abstraction-flow
(fixed-point) test with the corrected expectation (VISION.md coarse-graining note: RG fixed points
may be trivially general; the interesting structure lives in the flow).

Three parts, all on Gemma-2-9B-it residuals and Gemma Scope JumpReLU dictionaries
(pre = x @ W_enc + b_enc; act = pre * (pre > threshold); x_hat = act @ W_dec + b_dec):

A. LAYERS. Generality of the target description's features at layers 9 / 20 / 31, width 16k
   (hour 15 did layer 20 only), against the 72 narrative passages.
B. BROAD CORPUS. Recompute generality at layer 20 (16k and 131k) against ~120 passages spanning
   12 genres instead of the 72 narrative passages; repeat the hour-15 merge test under it.
C. FLOW. Abstraction level k = keep only 16k features with generality >= g_k; reconstruct each of
   the 72 narrative passages' mean residual from the survivors; track pairwise cosine distance,
   and the within-theme / across-theme and within-era / across-era ratios, as g_k rises. Report
   what the top-level survivors are anchored on.

Usage:
  python scripts/sae_ladder_v2.py --out results/sae_ladder_v2.json --fig results/figures/sae_ladder_v2.png
"""
import argparse, glob, json, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/user/latent-space-exploration")
ap.add_argument("--target", default="picard")
ap.add_argument("--thresholds", default="0.0,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.55,0.7,0.85,0.95")
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--out", default="results/sae_ladder_v2.json")
ap.add_argument("--fig", default=None)
a = ap.parse_args()
R = a.root
SAE_ROOT = glob.glob(f"{R}/cache/hf/hub/models--google--gemma-scope-9b-it-res/snapshots/*")[0]
LAYER_DICTS = {9: "layer_9/width_16k/average_l0_47", 20: "layer_20/width_16k/average_l0_47",
               31: "layer_31/width_16k/average_l0_43"}
WIDE_20 = "layer_20/width_131k/average_l0_43"


def load_sae(rel):
    return {k: v.astype(np.float32) for k, v in np.load(f"{SAE_ROOT}/{rel}/params.npz").items()}


def encode(sae, x):
    pre = x @ sae["W_enc"] + sae["b_enc"]
    return pre * (pre > sae["threshold"])


def load_tokens(path):
    z = np.load(path)
    ids = [k for k in z.files if not k.endswith("__tokens")]
    return {k: z[k] for k in ids}, {k: z[k + "__tokens"] for k in ids}


def generality(sae, X, keys):
    """fraction of reference passages on which each feature fires on >= 1 token."""
    fires = np.zeros(sae["W_enc"].shape[1])
    for k in keys:
        A = encode(sae, X[k][1:])
        fires += (A > 0).any(0)
    return fires / len(keys)


def alphabetic(t):
    return bool(re.search(r"[A-Za-z]", str(t).replace("▁", "")))


def target_profile(sae, X, toks, gen, target, top):
    """hour-15 readout: content features active on the target, excluding near-universal ones."""
    A = encode(sae, X[target][1:]); tk = toks[target][1:]
    mean_act = A.mean(0)
    active = np.flatnonzero((A > 0).any(0))
    # content = fires mostly on alphabetic tokens for this target, and not near-universal
    keep = []
    for f in active:
        j = int(np.argmax(A[:, f]))
        if alphabetic(tk[j]) and gen[f] <= 0.9:
            keep.append(f)
    keep = np.array(keep, dtype=int)
    g = gen[keep]
    order = keep[np.argsort(-mean_act[keep])][:top]
    rows = [dict(feature=int(f), mean_act=float(mean_act[f]), generality=float(gen[f]),
                 tokens=[str(tk[j]).replace("▁", " ") for j in np.argsort(-A[:, f])[:4] if A[j, f] > 0])
            for f in order]
    return dict(n_active=int(len(active)), n_content=int(len(keep)),
                mean=float(g.mean()), median=float(np.median(g)),
                frac_over_half=float((g > 0.5).mean()), frac_rare=float((g < 0.1).mean()), top=rows)


def label(k):
    p = k.split("/")
    return (p[1], p[2]) if p[0] == "mood" else (p[0], p[1])   # (theme, era)


report = {"sae_root": SAE_ROOT}

# ---------------------------------------------------------------- A. layers
Xl, Tl = {}, {}
for L, rel in LAYER_DICTS.items():
    path = f"{R}/results/tokens_gemma9b_l{L}.npz" if L != 20 else f"{R}/results/tokens_gemma9b_l20.npz"
    if not os.path.exists(path):
        print(f"[A] missing {path}, skipping layer {L}"); continue
    X, toks = load_tokens(path); Xl[L], Tl[L] = X, toks
    refs = [k for k in X if k != a.target]
    sae = load_sae(rel)
    gen = generality(sae, X, refs)
    prof = target_profile(sae, X, toks, gen, a.target, a.top)
    report.setdefault("A_layers", {})[str(L)] = dict(dict=rel, n_refs=len(refs), **prof)
    print(f"[A] layer {L}: content {prof['n_content']}/{prof['n_active']} active, "
          f"mean gen {prof['mean']:.3f} median {prof['median']:.3f} rare {prof['frac_rare']:.2f}", flush=True)
    del sae

# ---------------------------------------------------------------- B. broad corpus, layer 20
X20, T20 = Xl.get(20), Tl.get(20)
narr_refs = [k for k in X20 if k != a.target]
broad_path = f"{R}/results/tokens_gemma9b_broad_l20.npz"
if os.path.exists(broad_path):
    XB, TB = load_tokens(broad_path)
    XU = dict(X20); XU.update(XB); TU = dict(T20); TU.update(TB)
    broad_refs = list(XB)
    B = {}
    for tag, rel in (("16k", LAYER_DICTS[20]), ("131k", WIDE_20)):
        sae = load_sae(rel)
        g_narr = generality(sae, X20, narr_refs)
        g_broad = generality(sae, XU, broad_refs)
        B[tag] = dict(dict=rel,
                      narrative=target_profile(sae, X20, T20, g_narr, a.target, a.top),
                      broad=target_profile(sae, XU, TU, g_broad, a.target, a.top))
        np.save(f"/tmp/gen_{tag}_broad.npy", g_broad); np.save(f"/tmp/gen_{tag}_narr.npy", g_narr)
        print(f"[B] {tag}: narrative mean gen {B[tag]['narrative']['mean']:.3f} -> "
              f"broad {B[tag]['broad']['mean']:.3f} (rare {B[tag]['narrative']['frac_rare']:.2f} -> "
              f"{B[tag]['broad']['frac_rare']:.2f})", flush=True)
        del sae
    # merge test: 15 strongest 131k content features -> nearest 16k feature by decoder cosine
    s16, s131 = load_sae(LAYER_DICTS[20]), load_sae(WIDE_20)
    for corpus, gfile in (("narrative", "narr"), ("broad", "broad")):
        g16 = np.load(f"/tmp/gen_16k_{gfile}.npy"); g131 = np.load(f"/tmp/gen_131k_{gfile}.npy")
        D16 = s16["W_dec"] / np.linalg.norm(s16["W_dec"], axis=1, keepdims=True)
        rows = []
        for r in B["131k"][corpus]["top"][:15]:
            f = r["feature"]
            d = s131["W_dec"][f]; d = d / np.linalg.norm(d)
            cos = D16 @ d; j = int(np.argmax(cos))
            A16 = encode(s16, X20[a.target][1:])[:, j]
            tk = T20[a.target][1:]
            rows.append(dict(wide=f, wide_tokens=r["tokens"], wide_gen=float(g131[f]),
                             narrow=j, cos=float(cos[j]), narrow_gen=float(g16[j]),
                             narrow_tokens=[str(tk[i]).replace("▁", " ") for i in np.argsort(-A16)[:4] if A16[i] > 0],
                             more_general=bool(g16[j] > g131[f])))
        B.setdefault("merge_test", {})[corpus] = dict(n_more_general=int(sum(r["more_general"] for r in rows)),
                                                      n=len(rows), rows=rows)
        print(f"[B] merge test ({corpus}): {B['merge_test'][corpus]['n_more_general']}/15 nearest-narrow more general", flush=True)
    del s131
    report["B_broad"] = B
else:
    print(f"[B] missing {broad_path}, skipping"); s16 = load_sae(LAYER_DICTS[20])

# ---------------------------------------------------------------- C. abstraction flow, layer 20 / 16k
gen_for_flow = {}
sae = load_sae(LAYER_DICTS[20])
gen_for_flow["narrative"] = generality(sae, X20, narr_refs)
if os.path.exists(broad_path):
    gen_for_flow["broad"] = np.load("/tmp/gen_16k_broad.npy")

thr = [float(x) for x in a.thresholds.split(",")]
keys = narr_refs
themes = np.array([label(k)[0] for k in keys]); eras = np.array([label(k)[1] for k in keys])
# cache per-passage activation means and the raw mean residual
ACT = {k: encode(sae, X20[k][1:]) for k in keys}
ACTM = np.stack([ACT[k].mean(0) for k in keys])          # [72, F] mean activation per feature
RAW = np.stack([X20[k][1:].mean(0) for k in keys])


def pair_stats(V):
    Vn = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
    C = Vn @ Vn.T
    n = len(V); iu = np.triu_indices(n, 1)
    d = 1 - C[iu]
    same_t = (themes[:, None] == themes[None, :])[iu]
    same_e = (eras[:, None] == eras[None, :])[iu]
    Cm = C - np.eye(n) * 2
    nn_t, nn_e = [], []
    for i in range(n):
        j = np.argsort(-Cm[i])[:8]
        nn_t.append((themes[j] == themes[i]).mean()); nn_e.append((eras[j] == eras[i]).mean())
    nrm = np.linalg.norm(V, axis=1)
    E = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=-1)[iu]
    return dict(mean_cos=float(C[iu].mean()), mean_dist=float(d.mean()), sd_dist=float(d.std()),
                spread=float(E.mean() / max(nrm.mean(), 1e-9)),
                within_theme=float(d[same_t].mean()), across_theme=float(d[~same_t].mean()),
                theme_ratio=float(d[same_t].mean() / d[~same_t].mean()),
                within_era=float(d[same_e].mean()), across_era=float(d[~same_e].mean()),
                era_ratio=float(d[same_e].mean() / d[~same_e].mean()),
                knn8_theme=float(np.mean(nn_t)), knn8_era=float(np.mean(nn_e)))


flow = {}
for corpus, gen in gen_for_flow.items():
    levels = []
    for g_k in thr:
        surv = np.flatnonzero(gen >= g_k)
        if len(surv) == 0:
            levels.append(dict(g_k=g_k, n_features=0)); continue
        Vc = ACTM[:, surv] @ sae["W_dec"][surv] + sae["b_dec"]      # reconstruction from survivors
        row = dict(g_k=g_k, n_features=int(len(surv)),
                   frac_act_mass=float(ACTM[:, surv].sum() / max(ACTM.sum(), 1e-9)),
                   raw=pair_stats(Vc), centered=pair_stats(Vc - Vc.mean(0)))
        levels.append(row)
        print(f"[C/{corpus}] g>={g_k}: {len(surv):5d} feats  mass {row['frac_act_mass']:.2f}  "
              f"dist {row['raw']['mean_dist']:.4f} spread {row['raw']['spread']:.3f}  "
              f"theme {row['centered']['theme_ratio']:.3f}/knn {row['centered']['knn8_theme']:.2f}  "
              f"era {row['centered']['era_ratio']:.3f}/knn {row['centered']['knn8_era']:.2f}", flush=True)
    flow[corpus] = levels

# baseline: the un-reconstructed residual
flow["baseline_raw_residual"] = dict(raw=pair_stats(RAW), centered=pair_stats(RAW - RAW.mean(0)))

# anchors of the survivors at the top levels
anchors = {}
gen = gen_for_flow.get("broad", gen_for_flow["narrative"])
for g_k in thr[-3:]:
    surv = np.flatnonzero(gen >= g_k)
    if not len(surv):
        continue
    mass = np.zeros(ACTM.shape[1]); mass[surv] = ACTM[:, surv].sum(0)
    order = surv[np.argsort(-mass[surv])][:20]
    rows = []
    for f in order:
        best = []
        for k in keys:
            A = ACT[k][:, f]
            if A.max() > 0:
                i = int(np.argmax(A)); best.append((float(A[i]), str(T20[k][1:][i]).replace("▁", " ")))
        best.sort(reverse=True)
        rows.append(dict(feature=int(f), generality=float(gen[f]), total_act=float(mass[f]),
                         tokens=[t for _, t in best[:8]]))
    anchors[str(g_k)] = rows
    print(f"[C] anchors g>={g_k} ({len(surv)} feats): " +
          "; ".join(f"f{r['feature']}({r['generality']:.2f}){r['tokens'][:3]}" for r in rows[:6]), flush=True)

report["C_flow"] = dict(thresholds=thr, flow=flow, anchors=anchors,
                        generality_corpus=list(gen_for_flow))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(report, open(a.out, "w"), indent=1)
print("saved", a.out)

if a.fig:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    corpus = "broad" if "broad" in flow else "narrative"
    lv = [r for r in flow[corpus] if r.get("n_features")]
    x = [r["g_k"] for r in lv]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(x, [r["raw"]["mean_dist"] for r in lv], "o-", label="mean cosine distance")
    axes[0].plot(x, [r["raw"]["spread"] for r in lv], "s--", label="euclidean spread / norm")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("generality threshold $g_k$"); axes[0].set_ylabel("pairwise separation")
    axes[0].set_title("(a) do passages collapse?"); axes[0].legend(fontsize=8)
    axes[1].plot(x, [r["centered"]["knn8_theme"] for r in lv], "o-", label="theme (chance 0.11)")
    axes[1].plot(x, [r["centered"]["knn8_era"] for r in lv], "s-", label="era (chance 0.32)")
    axes[1].axhline(8 / 71, color="C0", lw=0.8, ls=":"); axes[1].axhline(23 / 71, color="C1", lw=0.8, ls=":")
    axes[1].set_xlabel("generality threshold $g_k$"); axes[1].set_ylabel("8-NN purity (mean-centered)")
    axes[1].set_title("(b) toward structure or one blob?"); axes[1].legend(fontsize=8)
    axes[2].semilogy(x, [r["n_features"] for r in lv], "o-")
    axes[2].set_xlabel("generality threshold $g_k$"); axes[2].set_ylabel("surviving features")
    axes[2].set_title("(c) size of the surviving dictionary")
    for ax in axes: ax.grid(alpha=0.3)
    fig.tight_layout(); os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)
    fig.savefig(a.fig, dpi=140)
    print("saved", a.fig)
