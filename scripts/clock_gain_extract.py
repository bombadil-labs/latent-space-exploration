"""29-layer extraction for the clock-gain arms (docs/specs/clock_depth_gain_v1.md sec 2).

One prompt per forward pass (no batch dimension -> immune to the hour-36 batch-row bug).
Stores the mean-pooled `state` span at all 29 residual points as `exp/{s}/{t}/p{p}` (the
key the hour-35 discrimination script expects) and, where the arm has one, the `interval`
span as `exp!{s}/{t}/p{p}`.

Usage: python scripts/clock_gain_extract.py A B C D
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")

ARMS = [x for x in sys.argv[1:] if x in ("A", "B", "C", "D")] or ["A", "B", "C", "D"]
MODEL = os.environ.get("CG_MODEL", "Qwen/Qwen2.5-1.5B")
G = json.load(open("prompts/clock_gain_v1.json"))

from lsx import LM
from lsx.extract import extract

lm = LM.from_pretrained(MODEL)
print(f"model {MODEL}  n_layers={lm.n_layers}  d={lm.d_model}", flush=True)

for arm in ARMS:
    d = G["arms"][arm]
    out, t0 = {}, time.time()
    for i, (k, marked) in enumerate(sorted(d.items()), 1):
        ra = extract(lm, marked, keep_resid=False)
        state = ra.roles["state"]
        assert state.shape == (lm.n_layers + 1, lm.d_model), state.shape
        out[f"exp/{k}"] = state.astype(np.float32)
        if "interval" in ra.roles:
            out[f"exp!{k}"] = ra.roles["interval"].astype(np.float32)
        if i % 60 == 0:
            print(f"  arm {arm}  {i}/{len(d)}  {time.time() - t0:.0f}s", flush=True)
    path = f"results/clock_gain_v1_stacks_{arm}.npz"
    np.savez_compressed(path, **out)
    print(f"wrote {path} ({len(out)} arrays) in {time.time() - t0:.0f}s", flush=True)

print("done", flush=True)
