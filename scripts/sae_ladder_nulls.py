"""Phase 0.2 (docs/PROGRAM.md): the three nulls specified at hour 39 for the abstraction-ladder
claim (writeup claim 6 / RESULTS hours 15, 26), never run before now.

All three nulls are constructed from data already cached: Gemma Scope JumpReLU dictionaries at
layer 20 (16k, 131k) and the cached per-token residuals for the 72 narrative + 111 broad reference
passages plus the Picard target (`results/tokens_gemma9b_l20.npz`,
`results/tokens_gemma9b_broad_l20.npz`). No forward pass, no NDIF, no download: this is pure
offline linear algebra on cached activations, exactly like `scripts/sae_ladder_v2.py` which it
reuses the encode/generality/target_profile machinery from (duplicated here rather than imported,
since that script executes at import time via argparse).

Null 1 — merge-test null. "12 of 15" / "13 of 15" wide-to-narrow merges going up in generality is
currently compared against nothing. The two dictionaries have different marginal generality
distributions, so the correct null is not 50%. For each of the 15 wide (131k) content features
being matched, we ask: if the matched narrow (16k) feature had instead been a uniformly random
16k-dictionary feature (matching exactly what the nearest-cosine search ranges over: all 16384
features, not just the content-filtered ones), what is the probability it is more general than the
wide feature? That gives 15 Bernoulli(p_i) draws; their exact distribution is a Poisson-binomial,
computed by convolution (no Monte Carlo needed). We report where the observed count falls in that
exact null distribution, for both the narrative-corpus and broad-corpus merge tests.

Null 2 — size-matched random-feature control on the width effect. The 0.18-vs-0.06 (narrative) and
0.19-vs-0.10 (broad) comparisons contrast dictionaries that differ in total width (16384 vs
131072). We draw repeated random subsets of the 131k dictionary's own 131072 features, each subset
the same size as the 16k dictionary (16384), and recompute mean generality of the target's content
features that happen to fall in the subset. Because "is this feature active on the target /
content-filtered / how general is it" are all fixed per-feature properties (computed once over the
full 131072-wide encode), each repeat is just an index restriction — no re-encoding needed. If a
random size-matched subset reproduces ~0.18, the effect would be about size, not about what
narrowing (training a dictionary AT that width) preserves.

Null 3 — matched-count + label-permutation null on the "flow" ordering (era dies before theme).
The flow's 8-NN theme/era purity is computed at each generality threshold g_k using the true
theme/era labels of the 72 narrative passages. We rebuild the reconstruction Vc = sum(surviving
features' mean activation * decoder direction) + b_dec at each g_k exactly as
`scripts/sae_ladder_v2.py` part C does, then repeatedly permute the theme labels (preserving the
8 x 9 class-size structure) and the era labels (preserving the 3 x 24 class-size structure)
independently, recomputing 8-NN purity and the within/across cosine-distance ratio under each
permutation. Because a permutation of labels preserves each label's class sizes exactly, this one
procedure is simultaneously the "matched-count" null (same class-count structure as truth) and the
"label-permutation" null the task names separately. We then ask, at each g_k, whether the observed
statistic sits inside or outside the permutation null band (z-score, one-sided empirical p), and
find the g_k at which each label's structure first falls into its own null band (the "dies" point),
to test whether era's dies before theme's, as hour 26 claimed.

Usage: python scripts/sae_ladder_nulls.py --out results/sae_ladder_nulls.json
"""
import argparse, glob, json, re, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--target", default="picard")
ap.add_argument("--thresholds", default="0.0,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.55,0.7,0.85,0.95")
ap.add_argument("--n_subset_repeats", type=int, default=4000)
ap.add_argument("--n_label_perms", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/sae_ladder_nulls.json")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
R = a.root
t0 = time.time()

SAE_ROOT = glob.glob(f"{R}/cache/hf/hub/models--google--gemma-scope-9b-it-res/snapshots/*")[0]
DICT_16K = "layer_20/width_16k/average_l0_47"
DICT_131K = "layer_20/width_131k/average_l0_43"


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
    fires = np.zeros(sae["W_enc"].shape[1])
    for k in keys:
        A = encode(sae, X[k][1:])
        fires += (A > 0).any(0)
    return fires / len(keys)


def alphabetic(t):
    return bool(re.search(r"[A-Za-z]", str(t).replace("▁", "")))


def target_profile(A, tk, gen, top):
    """hour-15/26 readout: content features active on the target, excluding near-universal ones."""
    mean_act = A.mean(0)
    active = np.flatnonzero((A > 0).any(0))
    keep = []
    for f in active:
        j = int(np.argmax(A[:, f]))
        if alphabetic(tk[j]) and gen[f] <= 0.9:
            keep.append(f)
    keep = np.array(keep, dtype=int)
    g = gen[keep]
    order = keep[np.argsort(-mean_act[keep])][:top]
    rows = [dict(feature=int(f), mean_act=float(mean_act[f]), generality=float(gen[f]))
            for f in order]
    return dict(n_active=int(len(active)), n_content=int(len(keep)), content_mask_idx=keep,
                mean=float(g.mean()) if len(g) else float("nan"),
                median=float(np.median(g)) if len(g) else float("nan"),
                frac_over_half=float((g > 0.5).mean()) if len(g) else float("nan"),
                frac_rare=float((g < 0.1).mean()) if len(g) else float("nan"), top=rows)


def label(k):
    p = k.split("/")
    return (p[1], p[2]) if p[0] == "mood" else (p[0], p[1])


def poisson_binomial_exact(ps):
    """Exact distribution of sum of independent Bernoulli(p_i), by DP convolution."""
    dist = np.array([1.0])
    for p in ps:
        dist = np.convolve(dist, [1 - p, p])
    return dist  # dist[k] = P(sum == k)


print("[load] SAE dictionaries and cached token residuals", flush=True)
s16 = load_sae(DICT_16K)
s131 = load_sae(DICT_131K)
X20, T20 = load_tokens(f"{R}/results/tokens_gemma9b_l20.npz")
XB, TB = load_tokens(f"{R}/results/tokens_gemma9b_broad_l20.npz")
narr_refs = [k for k in X20 if k != a.target]
broad_refs = list(XB)
XU = dict(X20); XU.update(XB); TU = dict(T20); TU.update(TB)
print(f"[load] done at {time.time()-t0:.1f}s", flush=True)

report = {"sae_root": SAE_ROOT, "dict_16k": DICT_16K, "dict_131k": DICT_131K,
          "n_narrative_refs": len(narr_refs), "n_broad_refs": len(broad_refs)}

# ---------------------------------------------------------------- shared: generality + target encodes
print("[shared] encoding target + refs through both dictionaries", flush=True)
g16_narr = generality(s16, X20, narr_refs)
g16_broad = generality(s16, XU, broad_refs)
g131_narr = generality(s131, X20, narr_refs)
g131_broad = generality(s131, XU, broad_refs)
print(f"[shared] generality vectors done at {time.time()-t0:.1f}s "
      f"(16k mean narr {g16_narr.mean():.3f} broad {g16_broad.mean():.3f}; "
      f"131k mean narr {g131_narr.mean():.3f} broad {g131_broad.mean():.3f})", flush=True)

A16_target = encode(s16, X20[a.target][1:]); tk16 = T20[a.target][1:]
A131_target = encode(s131, X20[a.target][1:]); tk131 = T20[a.target][1:]

prof16_narr = target_profile(A16_target, tk16, g16_narr, 25)
prof16_broad = target_profile(A16_target, tk16, g16_broad, 25)
prof131_narr = target_profile(A131_target, tk131, g131_narr, 25)
prof131_broad = target_profile(A131_target, tk131, g131_broad, 25)
print(f"[shared] target profiles: 16k narr mean {prof16_narr['mean']:.3f} n={prof16_narr['n_content']}, "
      f"131k narr mean {prof131_narr['mean']:.3f} n={prof131_narr['n_content']}; "
      f"16k broad mean {prof16_broad['mean']:.3f}, 131k broad mean {prof131_broad['mean']:.3f}", flush=True)

report["replication"] = {
    "16k_narrative": {k: v for k, v in prof16_narr.items() if k not in ("top", "content_mask_idx")},
    "16k_broad": {k: v for k, v in prof16_broad.items() if k not in ("top", "content_mask_idx")},
    "131k_narrative": {k: v for k, v in prof131_narr.items() if k not in ("top", "content_mask_idx")},
    "131k_broad": {k: v for k, v in prof131_broad.items() if k not in ("top", "content_mask_idx")},
}

# ================================================================== NULL 1: merge-test null
print("\n[null1] merge test against the narrow dictionary's own generality distribution", flush=True)
D16 = s16["W_dec"] / np.linalg.norm(s16["W_dec"], axis=1, keepdims=True)


def merge_test(prof_wide, g_wide, g16_pop, X20, T20):
    rows = []
    for r in prof_wide["top"][:15]:
        f = r["feature"]
        d = s131["W_dec"][f]; d = d / np.linalg.norm(d)
        cos = D16 @ d; j = int(np.argmax(cos))
        rows.append(dict(wide=f, wide_gen=float(g_wide[f]), narrow=j, cos=float(cos[j]),
                          narrow_gen=float(g16_pop[j]), more_general=bool(g16_pop[j] > g_wide[f])))
    return rows


null1 = {}
for corpus, prof_wide, g_wide, g16_pop in (
        ("narrative", prof131_narr, g131_narr, g16_narr),
        ("broad", prof131_broad, g131_broad, g16_broad)):
    rows = merge_test(prof_wide, g_wide, g16_pop, X20, T20)
    observed = sum(r["more_general"] for r in rows)
    # p_i = P(a uniformly random one of the 16384 16k-dict features is strictly more general
    # than wide feature i). This is exactly the population the nearest-cosine search ranged over.
    ps = [float((g16_pop > r["wide_gen"]).mean()) for r in rows]
    dist = poisson_binomial_exact(ps)
    ks = np.arange(len(dist))
    mean_null = float((ks * dist).sum())
    var_null = float(((ks - mean_null) ** 2 * dist).sum())
    sd_null = var_null ** 0.5
    p_at_least_observed = float(dist[ks >= observed].sum())
    ci_lo, ci_hi = int(np.searchsorted(np.cumsum(dist), 0.025)), int(np.searchsorted(np.cumsum(dist), 0.975))
    null1[corpus] = dict(
        rows=rows, observed=int(observed), n=len(rows), per_feature_p=ps,
        null_mean=mean_null, null_sd=sd_null, null_pmf=dist.tolist(),
        null_95ci=[ci_lo, ci_hi], p_value_observed_or_more=p_at_least_observed,
    )
    print(f"[null1/{corpus}] observed {observed}/15 more general; exact Poisson-binomial null "
          f"mean {mean_null:.2f} sd {sd_null:.2f} 95% CI [{ci_lo},{ci_hi}]/15; "
          f"P(K>={observed}) = {p_at_least_observed:.4f}", flush=True)
report["null1_merge_test"] = null1

# ================================================================== NULL 2: size-matched random subset
print("\n[null2] size-matched random-feature control on the width effect", flush=True)
n_narrow_width = s16["W_enc"].shape[1]   # 16384
n_wide_width = s131["W_enc"].shape[1]    # 131072
assert n_narrow_width < n_wide_width


def subset_control(A_target, tk, g_pop, n_subset, n_repeats, rng):
    active = (A_target > 0).any(0)
    mean_act = A_target.mean(0)
    peak_idx = np.argmax(A_target, axis=0)
    alpha = np.array([alphabetic(tk[j]) for j in peak_idx])
    content_mask = active & alpha & (g_pop <= 0.9)
    n_content_full = int(content_mask.sum())
    full_mean = float(g_pop[content_mask].mean()) if n_content_full else float("nan")
    means, ns = [], []
    idx_all = np.arange(len(g_pop))
    for _ in range(n_repeats):
        subset = rng.choice(idx_all, size=n_subset, replace=False)
        m = content_mask[subset]
        n_in = int(m.sum())
        ns.append(n_in)
        means.append(float(g_pop[subset][m].mean()) if n_in else np.nan)
    means = np.array(means)
    return dict(n_content_full_dict=n_content_full, full_dict_mean=full_mean,
                subset_means=means, subset_n_content=ns)


null2 = {}
for corpus, A_t, tk_t, g_pop_wide, g_pop_narrow, prof_narrow in (
        ("narrative", A131_target, tk131, g131_narr, g16_narr, prof16_narr),
        ("broad", A131_target, tk131, g131_broad, g16_broad, prof16_broad)):
    ctrl = subset_control(A_t, tk_t, g_pop_wide, n_narrow_width, a.n_subset_repeats, rng)
    means = ctrl["subset_means"]
    valid = means[~np.isnan(means)]
    pct_above = float((valid >= prof_narrow["mean"]).mean()) if len(valid) else float("nan")
    z = float((prof_narrow["mean"] - valid.mean()) / valid.std()) if len(valid) else float("nan")
    null2[corpus] = dict(
        narrow_dict_mean_observed=prof_narrow["mean"],
        wide_full_dict_mean_observed=ctrl["full_dict_mean"],
        n_content_features_full_wide_dict=ctrl["n_content_full_dict"],
        size_matched_subset_null_mean=float(valid.mean()) if len(valid) else float("nan"),
        size_matched_subset_null_sd=float(valid.std()) if len(valid) else float("nan"),
        size_matched_subset_null_median=float(np.median(valid)) if len(valid) else float("nan"),
        size_matched_subset_n_content_mean=float(np.mean(ctrl["subset_n_content"])),
        n_repeats=a.n_subset_repeats,
        fraction_of_random_subsets_reaching_observed_narrow_mean=pct_above,
        z_of_observed_narrow_mean_vs_subset_null=z,
    )
    print(f"[null2/{corpus}] observed 16k mean {prof_narrow['mean']:.3f}; wide-dict full mean "
          f"{ctrl['full_dict_mean']:.3f}; {a.n_subset_repeats} size-matched ({n_narrow_width}) random "
          f"subsets of the wide dict: null mean {valid.mean():.3f} sd {valid.std():.3f}, "
          f"avg {np.mean(ctrl['subset_n_content']):.0f} content features/subset; "
          f"P(random subset mean >= observed) = {pct_above:.4f}, z = {z:.2f}", flush=True)
report["null2_size_matched_subset"] = null2

# ================================================================== NULL 3: flow ordering, matched-count + permutation
print("\n[null3] matched-count + label-permutation null on the flow ordering", flush=True)
thr = [float(x) for x in a.thresholds.split(",")]
keys = narr_refs
themes = np.array([label(k)[0] for k in keys]); eras = np.array([label(k)[1] for k in keys])
n = len(keys)
ACT = {k: encode(s16, X20[k][1:]) for k in keys}
ACTM = np.stack([ACT[k].mean(0) for k in keys])


def knn8_purity(V, labels):
    Vn = V - V.mean(0)
    Vn2 = Vn / np.clip(np.linalg.norm(Vn, axis=1, keepdims=True), 1e-9, None)
    C = Vn2 @ Vn2.T
    Cm = C - np.eye(n) * 2
    purity = []
    for i in range(n):
        j = np.argsort(-Cm[i])[:8]
        purity.append((labels[j] == labels[i]).mean())
    return float(np.mean(purity))


def within_across_ratio(V, labels):
    Vn2 = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
    C = Vn2 @ Vn2.T
    iu = np.triu_indices(n, 1)
    d = 1 - C[iu]
    same = (labels[:, None] == labels[None, :])[iu]
    return float(d[same].mean() / d[~same].mean())


for gen_name, gen_arr in (("broad", g16_broad), ("narrative", g16_narr)):
    levels = []
    for g_k in thr:
        surv = np.flatnonzero(gen_arr >= g_k)
        if len(surv) == 0:
            levels.append(dict(g_k=g_k, n_features=0)); continue
        Vc = ACTM[:, surv] @ s16["W_dec"][surv] + s16["b_dec"]
        obs_theme = knn8_purity(Vc, themes)
        obs_era = knn8_purity(Vc, eras)
        obs_theme_ratio = within_across_ratio(Vc - Vc.mean(0), themes)
        obs_era_ratio = within_across_ratio(Vc - Vc.mean(0), eras)
        null_theme, null_era = [], []
        null_theme_ratio, null_era_ratio = [], []
        for _ in range(a.n_label_perms):
            pt = rng.permutation(themes)
            pe = rng.permutation(eras)
            null_theme.append(knn8_purity(Vc, pt))
            null_era.append(knn8_purity(Vc, pe))
            null_theme_ratio.append(within_across_ratio(Vc - Vc.mean(0), pt))
            null_era_ratio.append(within_across_ratio(Vc - Vc.mean(0), pe))
        null_theme = np.array(null_theme); null_era = np.array(null_era)
        null_theme_ratio = np.array(null_theme_ratio); null_era_ratio = np.array(null_era_ratio)

        def z_and_p(obs, null, higher_is_more_structured=True):
            mu, sd = null.mean(), null.std()
            z = (obs - mu) / sd if sd > 0 else float("nan")
            # one-sided: is obs beyond what random relabeling gives, in the direction that means
            # "more structured than chance"? For knn8 purity, higher = more structured. For the
            # within/across cosine-distance ratio, LOWER = more structured (within-label pairs
            # closer than across-label pairs), so the tail is flipped.
            if higher_is_more_structured:
                p = float((null >= obs).mean())
            else:
                p = float((null <= obs).mean())
            return float(mu), float(sd), float(z), p

        tm, ts, tz, tp = z_and_p(obs_theme, null_theme, higher_is_more_structured=True)
        em, es, ez, ep = z_and_p(obs_era, null_era, higher_is_more_structured=True)
        rtm, rts, rtz, rtp = z_and_p(obs_theme_ratio, null_theme_ratio, higher_is_more_structured=False)
        rem, res, rez, rep = z_and_p(obs_era_ratio, null_era_ratio, higher_is_more_structured=False)
        levels.append(dict(
            g_k=g_k, n_features=int(len(surv)),
            theme_knn8=obs_theme, theme_null_mean=tm, theme_null_sd=ts, theme_z=tz, theme_p=tp,
            era_knn8=obs_era, era_null_mean=em, era_null_sd=es, era_z=ez, era_p=ep,
            theme_ratio=obs_theme_ratio, theme_ratio_null_mean=rtm, theme_ratio_z=rtz, theme_ratio_p=rtp,
            era_ratio=obs_era_ratio, era_ratio_null_mean=rem, era_ratio_z=rez, era_ratio_p=rep,
        ))
        print(f"[null3/{gen_name}] g>={g_k}: theme knn8 {obs_theme:.3f} (null {tm:.3f}+-{ts:.3f}, "
              f"z={tz:.2f}, p={tp:.3f})  era knn8 {obs_era:.3f} (null {em:.3f}+-{es:.3f}, "
              f"z={ez:.2f}, p={ep:.3f})", flush=True)

    # find first g_k (ascending) at which each label's structure falls into its own null band
    # (one-sided p >= 0.05, i.e. can no longer reject "indistinguishable from a random relabeling")
    def dies_at(levels, key_p):
        for lv in levels:
            if lv.get("n_features", 0) == 0:
                continue
            if lv[key_p] >= 0.05:
                return lv["g_k"]
        return None

    theme_dies = dies_at(levels, "theme_p")
    era_dies = dies_at(levels, "era_p")
    theme_ratio_dies = dies_at(levels, "theme_ratio_p")
    era_ratio_dies = dies_at(levels, "era_ratio_p")
    print(f"[null3/{gen_name}] knn8 purity: theme falls into permutation-null band at g_k>={theme_dies}, "
          f"era at g_k>={era_dies}  |  ratio: theme at g_k>={theme_ratio_dies}, era at g_k>={era_ratio_dies}",
          flush=True)
    report.setdefault("null3_flow_permutation", {})[gen_name] = dict(
        thresholds=thr, n_label_perms=a.n_label_perms, levels=levels,
        theme_knn8_dies_at_gk=theme_dies, era_knn8_dies_at_gk=era_dies,
        theme_ratio_dies_at_gk=theme_ratio_dies, era_ratio_dies_at_gk=era_ratio_dies,
        era_dies_before_theme_knn8=(era_dies is not None and theme_dies is not None and era_dies <= theme_dies),
        era_dies_before_theme_ratio=(era_ratio_dies is not None and theme_ratio_dies is not None
                                      and era_ratio_dies <= theme_ratio_dies),
    )

report["wall_time_s"] = time.time() - t0
import os
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(report, open(a.out, "w"), indent=1)
print(f"\nsaved {a.out} ({time.time()-t0:.1f}s total)", flush=True)
