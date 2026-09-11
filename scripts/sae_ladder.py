"""Deterritorialization via dictionary width (Gemma Scope JumpReLU SAEs, layer 20 of Gemma-2-9B-it).
For a target text and a reference set of per-token activations:
  - encode every token through each dictionary: act = pre * (pre > threshold), pre = x @ W_enc + b_enc
  - generality(feature) = fraction of reference passages on which the feature fires on >= 1 token
  - for the target: features active (mean over its tokens), ranked by activation; for each, the tokens it
    fires on most and its generality. Compare the generality distribution across widths.
Prediction: the narrower dictionary's surviving features are more general (fire on more passages)."""
import argparse, glob, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("tokens_npz"); ap.add_argument("--target", default="picard")
ap.add_argument("--sae_root", default=glob.glob("cache/hf/hub/models--google--gemma-scope-9b-it-res/snapshots/*/layer_20")[0])
ap.add_argument("--widths", default="width_16k/average_l0_47,width_131k/average_l0_43"); ap.add_argument("--top", type=int, default=25); ap.add_argument("--out", default=None)
a = ap.parse_args()
z = np.load(a.tokens_npz); ids = [k for k in z.files if not k.endswith("__tokens")]
refs = [k for k in ids if k != a.target]; X = {k: z[k] for k in ids}; toks = {k: z[k + "__tokens"] for k in ids}
def encode(sae, x):
    pre = x @ sae["W_enc"] + sae["b_enc"]; return pre * (pre > sae["threshold"])
report = {}
for w in a.widths.split(","):
    sae = dict(np.load(f"{a.sae_root}/{w}/params.npz"))
    fires = np.zeros(sae["W_enc"].shape[1]); n_active_per_tok = []
    for k in refs:
        A = encode(sae, X[k][1:]); fires += (A > 0).any(0); n_active_per_tok.append((A > 0).sum(1).mean())
    generality = fires / len(refs)
    A = encode(sae, X[a.target][1:]); tk = toks[a.target][1:]
    mean_act = A.mean(0); active = np.flatnonzero((A > 0).any(0))
    order = active[np.argsort(-mean_act[active])][: a.top]
    rows = []
    for f in order:
        top_tok = np.argsort(-A[:, f])[:4]
        rows.append(dict(feature=int(f), mean_act=float(mean_act[f]), generality=float(generality[f]),
                         tokens=[str(tk[j]).replace("▁", " ") for j in top_tok if A[j, f] > 0]))
    g_active = generality[active]
    report[w] = dict(n_active_features=int(len(active)), mean_active_per_token=float(np.mean(n_active_per_tok)),
                     generality_of_target_features=dict(mean=float(g_active.mean()), median=float(np.median(g_active)),
                                                        frac_general_over_half=float((g_active > 0.5).mean()), frac_rare_under_tenth=float((g_active < 0.1).mean())),
                     top=rows)
    print(f"\n=== {w}: {len(active)} features active on '{a.target}'; ref avg active/token {np.mean(n_active_per_tok):.1f} ===")
    print(f"generality of target's features: mean {g_active.mean():.3f}  median {np.median(g_active):.3f}  >0.5: {(g_active>0.5).mean():.2f}  <0.1: {(g_active<0.1).mean():.2f}")
    for r in rows: print(f"  f{r['feature']:6d} act {r['mean_act']:6.2f} gen {r['generality']:.2f}  {r['tokens']}")
if a.out: json.dump(report, open(a.out, "w"), indent=1)
