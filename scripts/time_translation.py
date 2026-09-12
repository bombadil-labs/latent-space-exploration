"""Parameterized time translation T(dt): extraction + measurements 1-7.

Spec: docs/specs/time_translation_v1.md

Grid: 8 subjects x 10 timepoints (t0 + 9 dt) x 3 paraphrases. Passages are
`[[interval: <phrase>]] [[state: <description>]]`; the control reattaches each interval
phrase to the same subject's/paraphrase's t0 state description.

  stage extract : pool the `state` span (mean) at layers 0,8,14,20,27 for all 480 passages
  stage measure : displacement, shared/residual decomposition, clock geometry, tau(s),
                  cyclic return, real vs fictional, phrase-only control; figures.
"""
import argparse, json, os, time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("grid", nargs="?", default="prompts/time_translation_v1.json")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--layers", default="0,8,14,20,27")
ap.add_argument("--stage", default="all", choices=["extract", "measure", "all"])
ap.add_argument("--suffix", default="", help="tag inserted into default --stacks/--out/figure names, e.g. 'gemma'")
ap.add_argument("--stacks", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--figdir", default="results/figures")
a = ap.parse_args()
_sfx = f"_{a.suffix}" if a.suffix else ""
a.stacks = a.stacks or f"results/time_translation{_sfx}_stacks.npz"
a.out = a.out or f"results/time_translation{_sfx}_measures.json"
FPFX = f"time_translation{_sfx}"     # figure filename prefix, so a suffixed run doesn't clobber the default figures

G = json.load(open(a.grid))
S, TP, DT = G["subjects"], G["timepoints"], G["deltas"]
LAYERS = [int(x) for x in a.layers.split(",")]
NP_ = G["n_paraphrases"]
LOGDT = G["log10_dt_days"]

# ---------------------------------------------------------------- extraction
if a.stage in ("extract", "all"):
    from lsx import LM
    from lsx.extract import extract
    lm = LM.from_pretrained(a.model)
    out, t0 = {}, time.time()
    todo = [("exp", G["prompts"]), ("ctrl", G["control_prompts"])]
    n = sum(len(d) for _, d in todo)
    i = 0
    for tag, d in todo:
        for k, marked in d.items():
            ra = extract(lm, marked, keep_resid=False)
            out[f"{tag}/{k}"] = ra.roles["state"][LAYERS]        # [nL, d]
            out[f"{tag}!{k}"] = ra.roles["interval"][LAYERS]
            i += 1
            if i % 40 == 0:
                print(f"  {i}/{n}  {time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.stacks), exist_ok=True)
    np.savez_compressed(a.stacks, **out)
    print(f"wrote {a.stacks} ({len(out)} arrays) in {time.time()-t0:.0f}s", flush=True)

if a.stage == "extract":
    raise SystemExit

# ---------------------------------------------------------------- helpers
z = np.load(a.stacks)
nL = len(LAYERS)
norm = lambda v: float(np.linalg.norm(v))
def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
def rank(x):
    o = np.argsort(np.argsort(np.asarray(x, float)))
    return o.astype(float)
def spearman(x, y):
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float(rx @ ry / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))

def H(tag, s, t, l):
    """mean over paraphrases of the pooled state vector at layer index l."""
    return np.mean([z[f"{tag}/{s}/{t}/p{p}"][l] for p in range(NP_)], axis=0)

def displacements(tag, l):
    return {s: {t: H(tag, s, t, l) - H(tag, s, "t0", l) for t in DT} for s in S}

R = {"_meta": {"model": a.model, "layers": LAYERS, "subjects": S, "deltas": DT,
               "grid": a.grid, "n_paraphrases": NP_}}

# ---------------- 1 & 2: displacement and shared/residual decomposition
dec = {}
for li, l in enumerate(LAYERS):
    d = displacements("exp", li)
    shared = {t: np.mean([d[s][t] for s in S], axis=0) for t in DT}
    resid = {s: {t: d[s][t] - shared[t] for t in DT} for s in S}
    loo = {t: {s: np.mean([d[o][t] for o in S if o != s], axis=0) for s in S} for t in DT}
    per_dt = {}
    for t in DT:
        num = sum(norm(resid[s][t]) ** 2 for s in S)
        den = sum(norm(d[s][t]) ** 2 for s in S)
        per_dt[t] = dict(shared_norm=norm(shared[t]),
                         mean_d_norm=float(np.mean([norm(d[s][t]) for s in S])),
                         mean_resid_norm=float(np.mean([norm(resid[s][t]) for s in S])),
                         frac_shared=float(1 - num / (den + 1e-12)),
                         resid_norm={s: norm(resid[s][t]) for s in S},
                         d_norm={s: norm(d[s][t]) for s in S},
                         cos_shared_loo=float(np.mean([cos(shared[t], loo[t][s]) for s in S])))
    num = sum(norm(resid[s][t]) ** 2 for s in S for t in DT)
    den = sum(norm(d[s][t]) ** 2 for s in S for t in DT)
    dec[l] = dict(per_dt=per_dt, frac_shared_all=float(1 - num / (den + 1e-12)))
    dec[l]["_arrays"] = (shared, resid, d)
R["m1_m2_decomposition"] = {str(l): {k: v for k, v in dec[l].items() if k != "_arrays"} for l in LAYERS}

# ---------------- 3: clock geometry
geo = {}
for li, l in enumerate(LAYERS):
    shared, _, _ = dec[l]["_arrays"]
    C = [[cos(shared[ti], shared[tj]) for tj in DT] for ti in DT]
    ns = [norm(shared[t]) for t in DT]
    lg = [LOGDT[t] for t in DT]
    adj = float(np.mean([C[i][i + 1] for i in range(len(DT) - 1)]))
    far = float(np.mean([C[i][j] for i in range(len(DT)) for j in range(len(DT)) if j - i >= 4]))
    geo[str(l)] = dict(cos_matrix=C, shared_norms=ns,
                       spearman_norm_logdt=spearman(ns, lg),
                       pearson_norm_logdt=float(np.corrcoef(ns, lg)[0, 1]),
                       mean_cos_adjacent=adj, mean_cos_distant=far)
R["m3_clock_geometry"] = geo

# ---------------- 4: subject timescale tau
tau = {}
for li, l in enumerate(LAYERS):
    _, resid, _ = dec[l]["_arrays"]
    cur = {}
    for s in S:
        ys = [norm(resid[s][t]) for t in DT]
        half = 0.5 * max(ys)
        idx = next(i for i, y in enumerate(ys) if y >= half)
        cur[s] = dict(tau=DT[idx], tau_index=idx, tau_log10_days=LOGDT[DT[idx]], curve=ys)
    tau[str(l)] = cur
R["m4_tau"] = tau

# ---------------- 5: cyclic return
cyc = {}
for li, l in enumerate(LAYERS):
    _, _, d = dec[l]["_arrays"]
    cyc[str(l)] = {s: dict(d_6months=norm(d[s]["6months"]), d_1year=norm(d[s]["1year"]),
                           ratio_1y_over_6mo=norm(d[s]["1year"]) / (norm(d[s]["6months"]) + 1e-12),
                           cos_6mo_1y=cos(d[s]["6months"], d[s]["1year"]))
                   for s in ("orchard", "mountain", "street", "river")}
R["m5_cyclic"] = cyc

# ---------------- 6: real vs fictional population
rvf = {}
for li, l in enumerate(LAYERS):
    _, resid, _ = dec[l]["_arrays"]
    per = {t: dict(cos=cos(resid["real_population"][t], resid["fictional_population"][t]),
                   norm_real=norm(resid["real_population"][t]),
                   norm_fict=norm(resid["fictional_population"][t]),
                   dist=norm(resid["real_population"][t] - resid["fictional_population"][t]))
           for t in DT}
    sep = min(DT, key=lambda t: per[t]["cos"])
    sep_d = max(DT, key=lambda t: per[t]["dist"])
    rvf[str(l)] = dict(per_dt=per, min_cos_at=sep, max_dist_at=sep_d)
R["m6_real_vs_fictional"] = rvf

# ---------------- 7: phrase-only control
ctl = {}
for li, l in enumerate(LAYERS):
    dc = displacements("ctrl", li)
    sh_c = {t: np.mean([dc[s][t] for s in S], axis=0) for t in DT}
    rs_c = {s: {t: dc[s][t] - sh_c[t] for t in DT} for s in S}
    sh_e, _, _ = dec[l]["_arrays"]
    per = {}
    for t in DT:
        num = sum(norm(rs_c[s][t]) ** 2 for s in S)
        den = sum(norm(dc[s][t]) ** 2 for s in S)
        per[t] = dict(shared_ctrl_norm=norm(sh_c[t]), shared_exp_norm=norm(sh_e[t]),
                      ratio=norm(sh_e[t]) / (norm(sh_c[t]) + 1e-12),
                      cos_exp_ctrl=cos(sh_e[t], sh_c[t]),
                      frac_shared_ctrl=float(1 - num / (den + 1e-12)))
    ctl[str(l)] = dict(per_dt=per,
                       mean_ratio=float(np.mean([per[t]["ratio"] for t in DT])),
                       mean_cos=float(np.mean([per[t]["cos_exp_ctrl"] for t in DT])))
R["m7_phrase_control"] = ctl

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(R, open(a.out, "w"), indent=1)

# ---------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs(a.figdir, exist_ok=True)
X = [LOGDT[t] for t in DT]
lab = ["1d", "1w", "6mo", "1y", "10y", "100y", "1ky", "10ky", "1My"]

fig, axes = plt.subplots(1, nL, figsize=(4 * nL, 3.6), sharex=True)
for li, l in enumerate(LAYERS):
    ax = axes[li]
    for s in S:
        ax.plot(X, R["m4_tau"][str(l)][s]["curve"], marker="o", ms=3, label=s)
    ax.set_title(f"layer {l}"); ax.set_xticks(X); ax.set_xticklabels(lab, rotation=60, fontsize=7)
    ax.set_ylabel("||resid(s, dt)||" if li == 0 else "")
axes[-1].legend(fontsize=6, loc="upper left")
fig.suptitle("m4: subject residual curves (knee = tau)"); fig.tight_layout()
fig.savefig(f"{a.figdir}/{FPFX}_resid_curves.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.6))
for l in LAYERS:
    ax.plot(X, R["m3_clock_geometry"][str(l)]["shared_norms"], marker="o", ms=3, label=f"L{l}")
ax.set_xticks(X); ax.set_xticklabels(lab, rotation=60, fontsize=7)
ax.set_ylabel("||shared(dt)||"); ax.set_title("m3: shared clock magnitude vs log dt"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{a.figdir}/{FPFX}_shared_norm.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, nL, figsize=(3.2 * nL, 3.4))
for li, l in enumerate(LAYERS):
    ax = axes[li]
    im = ax.imshow(np.array(R["m3_clock_geometry"][str(l)]["cos_matrix"]), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(DT))); ax.set_xticklabels(lab, rotation=90, fontsize=6)
    ax.set_yticks(range(len(DT))); ax.set_yticklabels(lab, fontsize=6)
    ax.set_title(f"layer {l}", fontsize=9)
fig.colorbar(im, ax=axes, shrink=0.8)
fig.suptitle("m3: cos(shared(dt_i), shared(dt_j))")
fig.savefig(f"{a.figdir}/{FPFX}_clock_cos.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
L14 = str(14 if 14 in LAYERS else LAYERS[len(LAYERS) // 2])
for s, c in (("real_population", "C0"), ("fictional_population", "C3")):
    axes[0].plot(X, R["m4_tau"][L14][s]["curve"], marker="o", ms=4, color=c, label=s)
axes[0].set_xticks(X); axes[0].set_xticklabels(lab, rotation=60, fontsize=7)
axes[0].set_ylabel("||resid||"); axes[0].set_title(f"m6: residual curves, layer {L14}"); axes[0].legend(fontsize=7)
for l in LAYERS:
    axes[1].plot(X, [R["m6_real_vs_fictional"][str(l)]["per_dt"][t]["cos"] for t in DT],
                 marker="o", ms=3, label=f"L{l}")
axes[1].axhline(0.6, ls="--", c="k", lw=0.8)
axes[1].set_xticks(X); axes[1].set_xticklabels(lab, rotation=60, fontsize=7)
axes[1].set_ylabel("cos(resid_real, resid_fict)"); axes[1].set_title("m6: real vs fictional"); axes[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{a.figdir}/{FPFX}_real_vs_fictional.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for l in LAYERS:
    axes[0].plot(X, [R["m7_phrase_control"][str(l)]["per_dt"][t]["ratio"] for t in DT], marker="o", ms=3, label=f"L{l}")
    axes[1].plot(X, [R["m7_phrase_control"][str(l)]["per_dt"][t]["cos_exp_ctrl"] for t in DT], marker="o", ms=3, label=f"L{l}")
axes[0].axhline(1.5, ls="--", c="k", lw=0.8); axes[0].set_ylabel("||shared_exp|| / ||shared_ctrl||")
axes[1].set_ylabel("cos(shared_exp, shared_ctrl)")
for ax in axes:
    ax.set_xticks(X); ax.set_xticklabels(lab, rotation=60, fontsize=7); ax.legend(fontsize=7)
fig.suptitle("m7: experimental vs phrase-only control"); fig.tight_layout()
fig.savefig(f"{a.figdir}/{FPFX}_phrase_control.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- console summary
print("\n=== m1/m2 decomposition: fraction of sum||d||^2 explained by shared(dt) ===")
for l in LAYERS:
    print(f"  layer {l:2d}: all-dt {dec[l]['frac_shared_all']:.3f}   " +
          " ".join(f"{t}:{dec[l]['per_dt'][t]['frac_shared']:.2f}" for t in DT))
print("\n=== m3 clock geometry ===")
for l in LAYERS:
    g = R["m3_clock_geometry"][str(l)]
    print(f"  layer {l:2d}: spearman(||shared||, log dt) {g['spearman_norm_logdt']:+.3f}  "
          f"pearson {g['pearson_norm_logdt']:+.3f}  cos adj {g['mean_cos_adjacent']:.3f} vs distant {g['mean_cos_distant']:.3f}")
print("\n=== m4 tau (smallest dt at half-max ||resid||) ===")
for l in LAYERS:
    print(f"  layer {l:2d}: " + "  ".join(f"{s}={R['m4_tau'][str(l)][s]['tau']}" for s in S))
print("\n=== m5 cyclic return ===")
for l in LAYERS:
    for s in ("orchard", "mountain"):
        c = R["m5_cyclic"][str(l)][s]
        print(f"  layer {l:2d} {s:8s}: ||d(6mo)||={c['d_6months']:.2f} ||d(1y)||={c['d_1year']:.2f} "
              f"ratio={c['ratio_1y_over_6mo']:.2f} cos={c['cos_6mo_1y']:+.3f}")
print("\n=== m6 real vs fictional ===")
for l in LAYERS:
    v = R["m6_real_vs_fictional"][str(l)]
    print(f"  layer {l:2d}: " + " ".join(f"{t}:{v['per_dt'][t]['cos']:+.2f}" for t in DT) +
          f"   min-cos at {v['min_cos_at']}, max-dist at {v['max_dist_at']}")
print("\n=== m7 phrase-only control ===")
for l in LAYERS:
    v = R["m7_phrase_control"][str(l)]
    print(f"  layer {l:2d}: mean ratio {v['mean_ratio']:.2f}  mean cos {v['mean_cos']:+.3f}  " +
          " ".join(f"{t}:{v['per_dt'][t]['ratio']:.2f}" for t in DT))
print(f"\nwrote {a.out} and figures in {a.figdir}")
