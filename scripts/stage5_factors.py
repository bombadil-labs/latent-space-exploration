"""Stage 5: narrative factors as directions. Grid: scene x era x voice, one span each.
Leave one scene out. Factor directions from the other scenes:
   dir_era[e]   = mean(era=e) - mean(all);  dir_voice[v] = mean(voice=v) - mean(all)
(A) decodability (no model): cosine-nearest factor direction classifies held-out spans?
(B) lens: +dir_era[e] at layer L raises logp of the era-e span among the 3 era variants (voice fixed)? rank/3, chance 2.
    same for voice. Control: random direction of same norm.
(C) cross-talk: under +dir_era[e], is the VOICE ranking of the 3 voice variants (era fixed) disturbed vs base?
    measured as rank of the true-voice span by logp under the era patch (chance 2; base should be ~1).
(D) composition: +dir_era[e] + dir_voice[v] -> rank of the (e,v) span among all 9 (chance 5).
(E) commutator: era at layer L1 + voice at L2  vs  voice at L1 + era at L2 -> rank among 9 under each.
"""
import argparse, json, itertools
import numpy as np
from lsx import LM
from lsx.model import Patch
from lsx.steer import add_vector
ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("stacks"); ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--layer", type=int, default=14); ap.add_argument("--layer2", type=int, default=20); ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--scenes", default=None); ap.add_argument("--out", default=None); ap.add_argument("--skip-model", action="store_true")
a = ap.parse_args()
g = json.load(open(a.grid)); z = np.load(a.stacks); X = {k: z[k] for k in z.files}
S, E, V = g["scenes"], g["eras"], g["voices"]; lead = g["lead"]; spans = g["spans"]
key = lambda s, e, v: f"{s}/{e}/{v}"
def dirs(l, train):
    allv = np.stack([X[key(s, e, v)][l] for s in train for e in E for v in V]); mu = allv.mean(0)
    de = {e: np.mean([X[key(s, e, v)][l] for s in train for v in V], axis=0) - mu for e in E}
    dv = {v: np.mean([X[key(s, e, v)][l] for s in train for e in E], axis=0) - mu for v in V}
    return de, dv, mu
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
# (A) decodability, all layers
print("(A) leave-one-scene-out decodability by nearest factor direction (chance: era 1/3, voice 1/3)")
L = next(iter(X.values())).shape[0]; accA = {}
for l in range(0, L, 2):
    ce = cv = n = 0
    for s in S:
        de, dv, mu = dirs(l, [x for x in S if x != s])
        for e in E:
            for v in V:
                x = X[key(s, e, v)][l] - mu
                ce += max(E, key=lambda ee: cos(x, de[ee])) == e; cv += max(V, key=lambda vv: cos(x, dv[vv])) == v; n += 1
    accA[l] = (ce / n, cv / n); print(f"  layer {l:2d}: era acc {ce/n:.2f}  voice acc {cv/n:.2f}")
if a.skip_model: raise SystemExit
lm = LM.from_pretrained(a.model); rng = np.random.default_rng(0); l = a.layer; res = []
test_scenes = a.scenes.split(",") if a.scenes else S
P = lambda vec, layer=l: [Patch(layer, add_vector(vec, a.scale))]
for s in test_scenes:
    de, dv, mu = dirs(l, [x for x in S if x != s]); de2, dv2, _ = dirs(a.layer2, [x for x in S if x != s])
    txt = {(e, v): f" {spans[key(s, e, v)]}" for e in E for v in V}
    lp = lambda ev, patches=None: lm.logprob(lead, txt[ev], patches)
    base = {ev: lp(ev) for ev in txt}
    def gains(patches): return {ev: lp(ev, patches) - base[ev] for ev in txt}
    # mid-rank on ties: a patch that moves nothing scores chance, not 1.0. The strict-'>' form scored a
    # total tie as rank 1, i.e. "doing nothing is a perfect selector" -- the second bug of hour 36
    # (results/notes/random_control_diagnosis.md §4). Audit hour 39.
    def rank_in(gd, target, cands):
        others = [c for c in cands if c != target]
        return 1 + sum(gd[c] > gd[target] for c in others) + 0.5 * sum(gd[c] == gd[target] for c in others)
    # no-patch arm: the same scoring pass with a zero direction. Under mid-rank ties this must read
    # chance (2.0 for a 3-way lens, 5.0 composed); anything else is a stop-the-line plumbing failure.
    gd_none = gains(P(np.zeros_like(de[E[0]])))
    # (B) era lens + (C) cross-talk under era patch
    for e in E:
        for cond, vec in (("era", de[e]), ("rand", (r := rng.normal(size=de[e].shape)) / np.linalg.norm(r) * np.linalg.norm(de[e]))):
            gd = gains(P(vec)); ab = {ev: base[ev] + gd[ev] for ev in txt}
            for v in V:
                res.append(dict(scene=s, test="B_era", cond=cond, factor=e, rank=rank_in(gd, (e, v), [(ee, v) for ee in E])))
                if cond == "era":  # cross-talk: voice ranking by absolute logp under the era patch, era fixed
                    for ee in E:
                        res.append(dict(scene=s, test="C_voice_under_era", cond="patched", rank=rank_in(ab, (ee, v), [(ee, vv) for vv in V]),
                                        base_rank=rank_in(base, (ee, v), [(ee, vv) for vv in V])))
    for v in V:
        for cond, vec in (("voice", dv[v]), ("rand", (r := rng.normal(size=dv[v].shape)) / np.linalg.norm(r) * np.linalg.norm(dv[v]))):
            gd = gains(P(vec)); ab = {ev: base[ev] + gd[ev] for ev in txt}
            for e in E:
                res.append(dict(scene=s, test="B_voice", cond=cond, factor=v, rank=rank_in(gd, (e, v), [(e, vv) for vv in V])))
                if cond == "voice":
                    for vv in V:
                        res.append(dict(scene=s, test="C_era_under_voice", cond="patched", rank=rank_in(ab, (e, vv), [(ee, vv) for ee in E]),
                                        base_rank=rank_in(base, (e, vv), [(ee, vv) for ee in E])))
    for e in E:
        for v in V:
            res.append(dict(scene=s, test="B_era", cond="none", factor=e, rank=rank_in(gd_none, (e, v), [(ee, v) for ee in E])))
            res.append(dict(scene=s, test="B_voice", cond="none", factor=v, rank=rank_in(gd_none, (e, v), [(e, vv) for vv in V])))
            res.append(dict(scene=s, test="D_compose", cond="none", rank=rank_in(gd_none, (e, v), list(txt))))
    # (D) composition and (E) commutator
    for e in E:
        for v in V:
            gd = gains(P(de[e] + dv[v])); res.append(dict(scene=s, test="D_compose", cond="compose", rank=rank_in(gd, (e, v), list(txt))))
            g1 = gains([Patch(l, add_vector(de[e], a.scale)), Patch(a.layer2, add_vector(dv2[v], a.scale))])
            g2 = gains([Patch(l, add_vector(dv[v], a.scale)), Patch(a.layer2, add_vector(de2[e], a.scale))])
            res.append(dict(scene=s, test="E_era_then_voice", rank=rank_in(g1, (e, v), list(txt))))
            res.append(dict(scene=s, test="E_voice_then_era", rank=rank_in(g2, (e, v), list(txt))))
            res.append(dict(scene=s, test="E_order_gap", rank=abs(rank_in(g1, (e, v), list(txt)) - rank_in(g2, (e, v), list(txt))),
                            agree=float(np.corrcoef([g1[ev] for ev in txt], [g2[ev] for ev in txt])[0, 1])))
    print(f"scene {s} done")
print(f"\n=== summary  layer={l} layer2={a.layer2} scale={a.scale} ===")
m = lambda t, c=None: np.mean([x["rank"] for x in res if x["test"] == t and (c is None or x["cond"] == c)])
print(f"(B) era lens: rank/3 factor-dir {m('B_era','era'):.2f}  random {m('B_era','rand'):.2f}  no-patch {m('B_era','none'):.2f}   (chance 2.0)")
print(f"(B) voice lens: rank/3 factor-dir {m('B_voice','voice'):.2f}  random {m('B_voice','rand'):.2f}  no-patch {m('B_voice','none'):.2f}   (chance 2.0)")
print(f"(C) voice readout under +era: rank {m('C_voice_under_era'):.2f}  (base {np.mean([x['base_rank'] for x in res if x['test']=='C_voice_under_era']):.2f})")
print(f"(C) era readout under +voice: rank {m('C_era_under_voice'):.2f}  (base {np.mean([x['base_rank'] for x in res if x['test']=='C_era_under_voice']):.2f})")
print(f"(D) era+voice composed: rank/9 {m('D_compose','compose'):.2f}  no-patch {m('D_compose','none'):.2f}   (chance 5.0)")
print(f"(E) era@{l}+voice@{a.layer2}: {m('E_era_then_voice'):.2f}   voice@{l}+era@{a.layer2}: {m('E_voice_then_era'):.2f}   mean |rank gap| {m('E_order_gap'):.2f}   gain-profile corr {np.mean([x['agree'] for x in res if x['test']=='E_order_gap']):.2f}")
if a.out: json.dump(dict(decodability=accA, results=res), open(a.out, "w"), indent=1)
