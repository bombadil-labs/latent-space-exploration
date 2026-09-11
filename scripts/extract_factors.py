"""Extract span vectors for a factor grid: one prompt per span, '<lead> [[span: ...]]', pooled over span tokens."""
import argparse, json, re
import numpy as np
from lsx import LM, extract
ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B"); a = ap.parse_args()
g = json.load(open(a.grid)); lm = LM.from_pretrained(a.model)
out = {k: extract(lm, f"{g['lead']} [[span: {t}]]", keep_resid=False).roles["span"] for k, t in g["spans"].items()}
tag = re.sub(r"[^A-Za-z0-9.]+", "_", a.model.split("/")[-1]).lower() + "_" + a.grid.split("/")[-1].split(".")[0]
np.savez_compressed(f"results/stacks_{tag}.npz", **out); print("saved", f"results/stacks_{tag}.npz", len(out), next(iter(out.values())).shape)
