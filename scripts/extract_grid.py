"""Extract role stacks for a prompt grid and cache to results/stacks_<tag>.npz (keys = prompt names)."""
import argparse, json, re
import numpy as np
from lsx import LM, extract, compare
ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
a = ap.parse_args()
grid = json.load(open(a.grid)); roles = grid["roles"]
lm = LM.from_pretrained(a.model)
stacks = {k: compare.role_stack(extract(lm, p, keep_resid=False), roles) for k, p in grid["prompts"].items()}
tag = re.sub(r"[^A-Za-z0-9.]+", "_", a.model.split("/")[-1]).lower() + "_" + a.grid.split("/")[-1].split(".")[0]
np.savez_compressed(f"results/stacks_{tag}.npz", roles=np.array(roles), **stacks)
print("saved", f"results/stacks_{tag}.npz", {k: v.shape for k, v in list(stacks.items())[:1]})
