"""Cosine between v2's shared(Dt) and v3's shared(Dt) at a given layer, per Dt.

shared(Dt) = mean_s [ H(s, Dt) - H(s, t0) ] using the state-span pooled vectors in
results/time_translation_v2_fresh_stacks.npz and results/time_translation_v3_fresh_stacks.npz.
"""
import json
import numpy as np

G2 = json.load(open("prompts/time_translation_v2.json"))
G3 = json.load(open("prompts/time_translation_v3.json"))
S, DT = G2["subjects"], G2["deltas"]
assert G3["subjects"] == S and G3["deltas"] == DT
NP2, NP3 = G2["n_paraphrases"], G3["n_paraphrases"]
LAYERS = [0, 8, 14, 20, 27]

z2 = np.load("results/time_translation_v2_fresh_stacks.npz")
z3 = np.load("results/time_translation_v3_fresh_stacks.npz")


def shared(z, NP_, li):
    H = lambda s, t: np.mean([z[f"exp/{s}/{t}/p{p}"][li] for p in range(NP_)], axis=0)
    d = {s: {t: H(s, t) - H(s, "t0") for t in DT} for s in S}
    return {t: np.mean([d[s][t] for s in S], axis=0) for t in DT}


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


out = {}
for li, l in enumerate(LAYERS):
    sh2 = shared(z2, NP2, li)
    sh3 = shared(z3, NP3, li)
    out[str(l)] = {t: cos(sh2[t], sh3[t]) for t in DT}

json.dump(out, open("results/time_translation_v2_v3_shared_cos.json", "w"), indent=1)
print("=== cos(shared_v2(dt), shared_v3(dt)) ===")
for l in LAYERS:
    row = out[str(l)]
    print(f"  layer {l:2d}: " + "  ".join(f"{t}:{row[t]:+.3f}" for t in DT))
