"""Measurement 8: the shared clock as a selector.

For each subject and each dt >= 1 year: prefix = the subject's t0 passage (markers stripped)
plus the interval phrase for dt; candidates = that subject's nine state spans, one per dt.
Patch = leave-one-subject-out shared(dt) added at every position at the given layer.
Rank of the matching state among nine (chance 5.0), with the patch, without it, and under a
random direction of equal norm (fixed seed).

Two rankings are reported:
  raw  : candidates ordered by sum log p(state | prefix) under that condition
  gain : candidates ordered by logprob(patched) - logprob(unpatched)  (patch/rand only)
"""
import argparse, json, os, time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("grid", nargs="?", default="prompts/time_translation_v1.json")
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--stacks", default="results/time_translation_stacks.npz")
ap.add_argument("--extract-layers", default="0,8,14,20,27", help="layer order inside --stacks")
ap.add_argument("--layers", default="14,8", help="layers to test")
ap.add_argument("--paraphrase", type=int, default=0)
ap.add_argument("--min-dt", default="1year")
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/time_translation_selector.json")
a = ap.parse_args()

G = json.load(open(a.grid))
S, DT, PH, ST = G["subjects"], G["deltas"], G["phrases"], G["states"]
EL = [int(x) for x in a.extract_layers.split(",")]
TEST = [int(x) for x in a.layers.split(",")]
NP_ = G["n_paraphrases"]
TESTDT = DT[DT.index(a.min_dt):]

z = np.load(a.stacks)
def H(s, t, li):
    return np.mean([z[f"exp/{s}/{t}/p{p}"][li] for p in range(NP_)], axis=0)

# leave-one-subject-out shared(dt) per test layer
SHARED = {}
for l in TEST:
    li = EL.index(l)
    d = {s: {t: H(s, t, li) - H(s, "t0", li) for t in DT} for s in S}
    SHARED[l] = {t: {s: np.mean([d[o][t] for o in S if o != s], axis=0) for s in S} for t in DT}

from lsx import LM
from lsx.model import Patch
from lsx.steer import add_vector

lm = LM.from_pretrained(a.model)
rng = np.random.default_rng(a.seed)
p = a.paraphrase
t0_text = {s: ST[s]["t0"][p] for s in S}

def rank_of(score, target, cands):
    """Mid-rank on ties. The strict-'>' form scored a total tie as rank 1, so a patch that moves
    nothing (the no-patch arm, whose gains are all identically zero) looked like a perfect selector
    -- the second bug of hour 36, results/notes/random_control_diagnosis.md §4. Audit hour 39."""
    others = [c for c in cands if c != target]
    return 1 + sum(score[c] > score[target] for c in others) + 0.5 * sum(score[c] == score[target] for c in others)


res, t_start = [], time.time()
base_cache = {}
for l in TEST:
    for s in S:
        cands = {t: " " + ST[s][t][p] for t in DT}
        for t in TESTDT:
            prefix = f"{PH['t0']} {t0_text[s]} {PH[t]}"
            key = (s, t)
            if key not in base_cache:
                base_cache[key] = {c: lm.logprob(prefix, cands[c]) for c in DT}
            base = base_cache[key]
            v = SHARED[l][t][s]
            r = rng.normal(size=v.shape); r *= np.linalg.norm(v) / np.linalg.norm(r)
            for cond, vec in (("clock", v), ("rand", r), ("none", np.zeros_like(v))):
                patches = [Patch(l, add_vector(vec, a.scale))]
                lp = {c: lm.logprob(prefix, cands[c], patches) for c in DT}
                gain = {c: lp[c] - base[c] for c in DT}
                res.append(dict(layer=l, subject=s, dt=t, cond=cond,
                                rank_raw=rank_of(lp, t, DT), rank_gain=rank_of(gain, t, DT)))
        print(f"  layer {l} {s} done  {time.time()-t_start:.0f}s", flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(dict(_meta=dict(model=a.model, layers=TEST, paraphrase=p, scale=a.scale, seed=a.seed,
                          test_dts=TESTDT, chance=5.0), rows=res), open(a.out, "w"), indent=1)

m = lambda f, **kw: float(np.mean([x[f] for x in res if all(x[k] == v for k, v in kw.items())
                                   and x[f] is not None]))
print(f"\n=== m8 selector: mean rank of the matching state among nine (chance 5.0) ===")
for l in TEST:
    print(f"  layer {l:2d}  raw:  clock {m('rank_raw', layer=l, cond='clock'):.2f}  "
          f"none {m('rank_raw', layer=l, cond='none'):.2f}  rand {m('rank_raw', layer=l, cond='rand'):.2f}")
    print(f"           gain: clock {m('rank_gain', layer=l, cond='clock'):.2f}  "
          f"rand {m('rank_gain', layer=l, cond='rand'):.2f}  "
          f"no-patch {m('rank_gain', layer=l, cond='none'):.2f}   (no-patch must read 5.00)")
    for t in TESTDT:
        print(f"      {t:14s} raw clock {m('rank_raw', layer=l, cond='clock', dt=t):.2f} "
              f"none {m('rank_raw', layer=l, cond='none', dt=t):.2f} "
              f"rand {m('rank_raw', layer=l, cond='rand', dt=t):.2f} | "
              f"gain clock {m('rank_gain', layer=l, cond='clock', dt=t):.2f} "
              f"rand {m('rank_gain', layer=l, cond='rand', dt=t):.2f} "
              f"none {m('rank_gain', layer=l, cond='none', dt=t):.2f}")
print(f"\nwrote {a.out} in {time.time()-t_start:.0f}s")
