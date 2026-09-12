"""Layer-0/14 discrimination test (subject_clocks_v1.md sec 3.5) adapted to the
time_translation grid's stacks (mean-pooled `state` span, tag "exp").

For each subject s and each held-out (timepoint, paraphrase) cell, two leave-one-out
nearest-centroid classifiers predict the timepoint index (0..9, t0 first):
  within  -- centroids built from s's OWN other paraphrases at each timepoint.
  shared  -- centroids built from every OTHER subject's paraphrases at each timepoint
             (the subject-agnostic "shared clock" predictor).
Spearman(predicted index, true index) is reported per subject and averaged, at the
requested layers. A perfect within AND shared score at layer 0 (pure embeddings, no
attention) means the timepoint is lexically recoverable before the model computes
anything -- the hour-32 confound this v3 grid targets.

Usage: python scripts/time_translation_discrimination.py results/time_translation_v2_fresh_stacks.npz prompts/time_translation_v2.json --layers 0,14
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("stacks")
ap.add_argument("grid")
ap.add_argument("--layers", default="0,14")
ap.add_argument("--out", default=None)
a = ap.parse_args()

G = json.load(open(a.grid))
S, DT, NP_ = G["subjects"], G["deltas"], G["n_paraphrases"]
ALL_LAYERS = G.get("_layers") or [0, 8, 14, 20, 27]  # not stored in grid; matches extraction default
LAYERS = [int(x) for x in a.layers.split(",")]
ORDER = ["t0"] + DT

z = np.load(a.stacks)


def rank(x):
    o = np.argsort(np.argsort(np.asarray(x, float)))
    return o.astype(float)


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float(rx @ ry / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def get(s, t, p, li):
    return z[f"exp/{s}/{t}/p{p}"][li]


results = {}
for l in LAYERS:
    li = ALL_LAYERS.index(l)
    per = {}
    for s in S:
        pw, ps, tr = [], [], []
        for ti in ORDER:
            for p in range(NP_):
                h = get(s, ti, p, li)
                # within: leave-one-paraphrase-out centroid of s's own other paraphrases, per timepoint
                cent_w = [np.mean([get(s, tj, q, li) for q in range(NP_) if q != p], axis=0) for tj in ORDER]
                pw.append(int(np.argmin([np.linalg.norm(h - c) for c in cent_w])))
                # shared: centroid of every OTHER subject's paraphrases, per timepoint (subject held out entirely)
                cent_s = [np.mean([get(o, tj, q, li) for o in S if o != s for q in range(NP_)], axis=0) for tj in ORDER]
                ps.append(int(np.argmin([np.linalg.norm(h - c) for c in cent_s])))
                tr.append(ORDER.index(ti))
        per[s] = dict(
            within_spearman=spearman(pw, tr),
            shared_spearman=spearman(ps, tr),
            within_mae=float(np.mean(np.abs(np.array(pw) - np.array(tr)))),
            shared_mae=float(np.mean(np.abs(np.array(ps) - np.array(tr)))),
        )
    within_mean = float(np.mean([per[s]["within_spearman"] for s in S]))
    shared_mean = float(np.mean([per[s]["shared_spearman"] for s in S]))
    results[str(l)] = dict(per_subject=per, within_mean=within_mean, shared_mean=shared_mean)
    print(f"layer {l:2d}: within {within_mean:.3f}  shared {shared_mean:.3f}  "
          + "  ".join(f"{s}:{per[s]['within_spearman']:.2f}/{per[s]['shared_spearman']:.2f}" for s in S))

if a.out:
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
