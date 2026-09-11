"""Qualitative: generate from a neutral story prompt with a factor direction added (all scenes used to build it)."""
import argparse, json
import numpy as np
from lsx import LM
from lsx.model import Patch
from lsx.steer import add_vector
ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("stacks"); ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--layer", type=int, default=14); ap.add_argument("--scale", type=float, default=1.5); ap.add_argument("--tokens", type=int, default=40)
ap.add_argument("--prompt", default="A moment from a story: The door opened and"); a = ap.parse_args()
g = json.load(open(a.grid)); z = np.load(a.stacks); X = {k: z[k][a.layer] for k in z.files}
S, E, V = g["scenes"], g["eras"], g["voices"]; key = lambda s, e, v: f"{s}/{e}/{v}"
mu = np.mean(list(X.values()), axis=0)
de = {e: np.mean([X[key(s, e, v)] for s in S for v in V], axis=0) - mu for e in E}
dv = {v: np.mean([X[key(s, e, v)] for s in S for e in E], axis=0) - mu for v in V}
lm = LM.from_pretrained(a.model)
print(f"PROMPT: {a.prompt!r}   (layer {a.layer}, scale {a.scale})\n")
print(f"BASE:        {lm.generate(a.prompt, a.tokens)!r}\n")
for name, d in list(de.items()) + list(dv.items()):
    print(f"+{name:10s} {lm.generate(a.prompt, a.tokens, [Patch(a.layer, add_vector(d, a.scale))])!r}")
print()
for e, v in (("medieval", "child"), ("farfuture", "ornate"), ("1920s", "terse")):
    print(f"+{e}+{v}: {lm.generate(a.prompt, a.tokens, [Patch(a.layer, add_vector(de[e] + dv[v], a.scale))])!r}")
