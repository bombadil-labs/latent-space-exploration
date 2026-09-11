"""Cross-talk between narrative factors, done properly.  Under a single-factor patch (+era_e or +voice_v,
or a random direction of the same norm), take the 3x3 gain matrix over (era, voice) spans of a held-out
scene and decompose its variance into an era main effect, a voice main effect, and residual.
   cross-talk(era patch) = fraction of gain variance explained by VOICE   (should be ~0 if factors are clean)
   on-target(era patch)  = fraction explained by ERA
Also report whether the on-target factor's gain is largest for the patched level (rank/3)."""
import argparse, json
import numpy as np
from lsx import LM
from lsx.model import Patch
from lsx.steer import add_vector
ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("stacks"); ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--layer", type=int, default=14); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--out", default=None)
a = ap.parse_args()
g = json.load(open(a.grid)); z = np.load(a.stacks); X = {k: z[k] for k in z.files}
S, E, V = g["scenes"], g["eras"], g["voices"]; lead = g["lead"]; spans = g["spans"]; key = lambda s, e, v: f"{s}/{e}/{v}"
def dirs(l, train):
    allv = np.stack([X[key(s, e, v)][l] for s in train for e in E for v in V]); mu = allv.mean(0)
    return ({e: np.mean([X[key(s, e, v)][l] for s in train for v in V], axis=0) - mu for e in E},
            {v: np.mean([X[key(s, e, v)][l] for s in train for e in E], axis=0) - mu for v in V})
def decompose(G):  # G [3 eras, 3 voices]
    gm = G.mean(); re_ = G.mean(1, keepdims=True) - gm; rv = G.mean(0, keepdims=True) - gm
    tot = ((G - gm) ** 2).sum() + 1e-12
    return float((re_ ** 2).sum() * 3 / tot), float((rv ** 2).sum() * 3 / tot)   # frac era, frac voice
lm = LM.from_pretrained(a.model); rng = np.random.default_rng(0); l = a.layer; rows = []
for s in S:
    de, dv = dirs(l, [x for x in S if x != s])
    base = {(e, v): lm.logprob(lead, f" {spans[key(s, e, v)]}") for e in E for v in V}
    def G(vec):
        return np.array([[lm.logprob(lead, f" {spans[key(s, e, v)]}", [Patch(l, add_vector(vec, a.scale))]) - base[(e, v)] for v in V] for e in E])
    for e in E:
        r = rng.normal(size=de[e].shape); r *= np.linalg.norm(de[e]) / np.linalg.norm(r)
        for cond, vec in (("era", de[e]), ("rand", r)):
            Gm = G(vec); fe, fv = decompose(Gm)
            rows.append(dict(scene=s, patch=cond, level=e, frac_on_target=fe, frac_crosstalk=fv, on_target_rank=1 + int((Gm.mean(1) > Gm.mean(1)[E.index(e)]).sum())))
    for v in V:
        r = rng.normal(size=dv[v].shape); r *= np.linalg.norm(dv[v]) / np.linalg.norm(r)
        for cond, vec in (("voice", dv[v]), ("rand", r)):
            Gm = G(vec); fe, fv = decompose(Gm)
            rows.append(dict(scene=s, patch=cond, level=v, frac_on_target=fv, frac_crosstalk=fe, on_target_rank=1 + int((Gm.mean(0) > Gm.mean(0)[V.index(v)]).sum())))
    print("scene", s, "done")
print(f"\n=== cross-talk summary, layer {l}, scale {a.scale} (fractions of gain variance; on-target rank/3, chance 2) ===")
for p in ("era", "voice", "rand"):
    xs = [x for x in rows if x["patch"] == p]
    print(f"{p:5s} patch | on-target {np.mean([x['frac_on_target'] for x in xs]):.2f}  cross-talk {np.mean([x['frac_crosstalk'] for x in xs]):.2f}  residual {1-np.mean([x['frac_on_target']+x['frac_crosstalk'] for x in xs]):.2f}  | on-target rank {np.mean([x['on_target_rank'] for x in xs]):.2f}  (n={len(xs)})")
if a.out: json.dump(rows, open(a.out, "w"), indent=1)
