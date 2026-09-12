"""Cosine between v1's shared(Dt) and v2's shared(Dt) at a given layer, per Dt.

shared(Dt) = mean_s [ H(s, Dt) - H(s, t0) ] using the state-span pooled vectors
already extracted into results/time_translation_stacks.npz (v1) and
results/time_translation_v2_stacks.npz (v2).
"""
import json
import numpy as np

G1 = json.load(open("prompts/time_translation_v1.json"))
G2 = json.load(open("prompts/time_translation_v2.json"))
S, DT = G1["subjects"], G1["deltas"]
assert G2["subjects"] == S and G2["deltas"] == DT
NP1, NP2 = G1["n_paraphrases"], G2["n_paraphrases"]
LAYERS = [0, 8, 14, 20, 27]

z1 = np.load("results/time_translation_stacks.npz")
z2 = np.load("results/time_translation_v2_stacks.npz")

def shared(z, NP_, li):
    H = lambda s, t: np.mean([z[f"exp/{s}/{t}/p{p}"][li] for p in range(NP_)], axis=0)
    d = {s: {t: H(s, t) - H(s, "t0") for t in DT} for s in S}
    return {t: np.mean([d[s][t] for s in S], axis=0) for t in DT}

def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

out = {}
for li, l in enumerate(LAYERS):
    sh1 = shared(z1, NP1, li)
    sh2 = shared(z2, NP2, li)
    out[str(l)] = {t: cos(sh1[t], sh2[t]) for t in DT}

json.dump(out, open("results/time_translation_v1_v2_shared_cos.json", "w"), indent=1)
print("=== cos(shared_v1(dt), shared_v2(dt)) ===")
for l in LAYERS:
    row = out[str(l)]
    print(f"  layer {l:2d}: " + "  ".join(f"{t}:{row[t]:+.3f}" for t in DT))
