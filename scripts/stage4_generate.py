"""Qualitative readout: generate a continuation for a held-out domain with +role direction at layer L."""
import argparse, json, re
import numpy as np
from lsx import LM
from lsx.model import Patch
from lsx.steer import add_vector
ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("stacks"); ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--layer", type=int, default=20); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--domain", default="law")
ap.add_argument("--roles", default="from_below,from_above,disturbance"); ap.add_argument("--tokens", type=int, default=28)
a = ap.parse_args()
g = json.load(open(a.grid)); roles = g["roles"]; R = len(roles)
z = np.load(a.stacks); stacks = {k: z[k] for k in z.files if k != "roles"}; doms = sorted({k.split("/")[0] for k in stacks})
avg = {d: np.mean([stacks[f"{d}/rot{r}"] for r in range(R)], axis=0) for d in doms}
M = np.mean([avg[x] for x in doms if x != a.domain], axis=0)[a.layer]; dirs = M - M.mean(0, keepdims=True)
lead = g["prompts"][f"{a.domain}/rot0"].split(" First,")[0]
prefix = f"{lead} First,"
lm = LM.from_pretrained(a.model)
print(f"PROMPT: {prefix!r}\nBASE: {lm.generate(prefix, a.tokens)!r}\n")
for r in a.roles.split(","):
    v = dirs[roles.index(r)]; rnd = np.random.default_rng(1).normal(size=v.shape); rnd *= np.linalg.norm(v) / np.linalg.norm(rnd)
    print(f"+{r} x{a.scale}: {lm.generate(prefix, a.tokens, [Patch(a.layer, add_vector(v, a.scale))])!r}")
    print(f"+random   x{a.scale}: {lm.generate(prefix, a.tokens, [Patch(a.layer, add_vector(rnd, a.scale))])!r}\n")
