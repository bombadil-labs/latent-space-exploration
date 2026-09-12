"""Bar figure: layer-0 vs layer-14 discrimination Spearman, v2 vs v3, within vs shared."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d2 = json.load(open("results/time_translation_v2_discrimination.json"))
d3 = json.load(open("results/time_translation_v3_discrimination.json"))

layers = ["0", "14"]
labels = ["v2 within", "v2 shared", "v3 within", "v3 shared"]
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(layers))
w = 0.2
vals = {
    "v2 within": [d2[l]["within_mean"] for l in layers],
    "v2 shared": [d2[l]["shared_mean"] for l in layers],
    "v3 within": [d3[l]["within_mean"] for l in layers],
    "v3 shared": [d3[l]["shared_mean"] for l in layers],
}
for i, k in enumerate(labels):
    ax.bar(x + (i - 1.5) * w, vals[k], width=w, label=k)
ax.axhline(0.5, ls="--", c="k", lw=0.8, label="predicted v3 layer-0 ceiling")
ax.set_xticks(x); ax.set_xticklabels([f"layer {l}" for l in layers])
ax.set_ylabel("discrimination Spearman"); ax.set_ylim(0, 1.05)
ax.set_title("sec 3.5 discrimination: v2 vs v3")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig("results/figures/time_translation_v3_discrimination.png", dpi=130)
print("wrote results/figures/time_translation_v3_discrimination.png")
