"""Absential ring census (Deacon via groovy-commutator: "off but adjacent to live").

For a passage, with a Gemma Scope JumpReLU dictionary at layer 20 of Gemma-2-9B-it:
  active   A = features firing on >= 1 non-BOS token
  content  Ac = A minus formatting features (generality > 0.9 over the reference set)
  ring     R = inactive features f with max_{a in Ac} cos(W_dec[f], W_dec[a]) >= tau,
               minus formatting features
generality(f) = fraction of reference passages on which f fires at least once.

Census: |A|, |Ac|, |R|; generality profile of ring vs active; whether ring composition differs by
theme or era (Jaccard within vs across); and a size-matched random-active-set null for the ring, so
that "adjacency" is not just an artefact of A covering ~10% of the dictionary.

Local only (no NDIF). Writes results/absential_census.json.
"""
import argparse, glob, json, itertools
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tokens_npz", default="/home/user/latent-space-exploration/results/tokens_gemma9b_l20.npz")
ap.add_argument("--sae", default=None, help="path to params.npz (default: layer_20/width_16k/average_l0_47)")
ap.add_argument("--tau", type=float, default=0.40)
ap.add_argument("--fmt_gen", type=float, default=0.90, help="generality above which a feature is 'formatting'")
ap.add_argument("--grids", default="theme,mood")
ap.add_argument("--out", default="results/absential_census.json")
a = ap.parse_args()

if a.sae is None:
    a.sae = glob.glob("/home/user/latent-space-exploration/cache/hf/hub/models--google--gemma-scope-9b-it-res/"
                      "snapshots/*/layer_20/width_16k/average_l0_47/params.npz")[0]
sae = dict(np.load(a.sae))
W_enc, b_enc, thr, W_dec = sae["W_enc"], sae["b_enc"], sae["threshold"], sae["W_dec"]
F = W_enc.shape[1]
Wn = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)

z = np.load(a.tokens_npz)
ids = [k for k in z.files if not k.endswith("__tokens")]
refs = [k for k in ids if k != "picard"]            # 72 reference passages (theme + mood grids)
THEMES = ("betrayal", "sacrifice", "homecoming")
theme_ids = [k for k in refs if k.split("/")[-1] in THEMES]
other_ids = [k for k in refs if k not in theme_ids]
want = {"theme": theme_ids, "mood": other_ids}
targets = [k for g in a.grids.split(",") for k in want[g]]

def encode(x):
    pre = x @ W_enc + b_enc
    return pre * (pre > thr)

# --- generality over the full reference set -------------------------------------------------
fires = np.zeros(F)
act = {}
for k in refs:
    A = encode(z[k][1:])                            # skip BOS
    m = (A > 0).any(0)
    fires += m
    act[k] = np.flatnonzero(m)
gen = fires / len(refs)
fmt = gen > a.fmt_gen

def ring_of(active_idx, tau):
    Ac = active_idx[~fmt[active_idx]]
    mx = (Wn @ Wn[Ac].T).max(1)                     # [F]
    inactive = np.ones(F, bool); inactive[active_idx] = False
    R = np.flatnonzero(inactive & (mx >= tau) & ~fmt)
    return Ac, R, mx, inactive

# distribution of max-cos over inactive features, pooled over 6 passages (basis for the tau choice)
pool = []
for k in targets[:6]:
    _, _, mx, inactive = ring_of(act[k], a.tau)
    pool.append(mx[inactive])
pool = np.concatenate(pool)
qs = {str(q): float(np.percentile(pool, q)) for q in (50, 75, 90, 95, 97.5, 99, 99.5, 99.9)}
print("max-cos(inactive -> active) percentiles:", {k: round(v, 3) for k, v in qs.items()})
print(f"tau = {a.tau}  -> keeps {(pool >= a.tau).mean()*100:.2f}% of inactive features")

# --- per-passage census ----------------------------------------------------------------------
rng = np.random.default_rng(0)
rows, ring_sets = {}, {}
seen = np.flatnonzero((gen > 0) & ~fmt)
for k in targets:
    Ac, R, mx, inactive = ring_of(act[k], a.tau)
    ring_sets[k] = set(R.tolist())
    # size-matched random-active-set null: same |Ac|, sampled from features that fire somewhere
    Ar = rng.choice(seen, size=len(Ac), replace=False)
    Cr = (Wn @ Wn[Ar].T).max(1)
    inact_r = np.ones(F, bool); inact_r[Ar] = False
    Rr = np.flatnonzero(inact_r & (Cr >= a.tau) & ~fmt)
    prof = lambda idx: dict(n=int(len(idx)), mean=float(gen[idx].mean()), median=float(np.median(gen[idx])),
                            frac_rare=float((gen[idx] < 0.1).mean()), frac_general=float((gen[idx] > 0.5).mean()),
                            frac_never=float((gen[idx] == 0).mean()))
    rows[k] = dict(n_active=int(len(act[k])), n_active_content=int(len(Ac)), n_ring=int(len(R)),
                   n_formatting_active=int(fmt[act[k]].sum()),
                   active=prof(Ac), ring=prof(R), ring_null_random_active=prof(Rr),
                   top_ring=[dict(feature=int(f), maxcos=float(mx[f]), generality=float(gen[f]))
                             for f in R[np.argsort(-mx[R])][:12]])

agg = lambda field, sub: float(np.mean([rows[k][field][sub] for k in targets]))
print(f"\n{len(targets)} passages: |A| {np.mean([rows[k]['n_active'] for k in targets]):.0f}  "
      f"|Ac| {np.mean([rows[k]['n_active_content'] for k in targets]):.0f}  "
      f"|R| {np.mean([rows[k]['n_ring'] for k in targets]):.0f}  "
      f"(random-active null |R| {agg('ring_null_random_active','n'):.0f})")
print(f"generality  active: mean {agg('active','mean'):.3f} rare<0.1 {agg('active','frac_rare'):.2f}   "
      f"ring: mean {agg('ring','mean'):.3f} rare<0.1 {agg('ring','frac_rare'):.2f} never-fires {agg('ring','frac_never'):.2f}   "
      f"null ring: mean {agg('ring_null_random_active','mean'):.3f}")

# --- does ring composition differ by theme / era? ---------------------------------------------
def jac(x, y): return len(x & y) / max(1, len(x | y))
comp = {}
th_ids = [k for k in targets if k in theme_ids]
if th_ids:
    for axis, pos in (("theme", -1), ("era", 1), ("scene", 0)):
        w, b, wa, ba = [], [], [], []
        for k1, k2 in itertools.combinations(th_ids, 2):
            same = k1.split("/")[pos] == k2.split("/")[pos]
            (w if same else b).append(jac(ring_sets[k1], ring_sets[k2]))
            (wa if same else ba).append(jac(set(act[k1].tolist()), set(act[k2].tolist())))
        comp[axis] = dict(ring_within=float(np.mean(w)), ring_across=float(np.mean(b)),
                          ring_delta=float(np.mean(w) - np.mean(b)),
                          active_within=float(np.mean(wa)), active_across=float(np.mean(ba)),
                          active_delta=float(np.mean(wa) - np.mean(ba)))
        print(f"ring Jaccard by {axis:6s}: within {np.mean(w):.3f} across {np.mean(b):.3f} (delta {np.mean(w)-np.mean(b):+.3f})"
              f"   | active: within {np.mean(wa):.3f} across {np.mean(ba):.3f} (delta {np.mean(wa)-np.mean(ba):+.3f})")

    # features in the ring of every passage of one theme and of no passage of the others
    sig = {}
    for t in THEMES:
        mine = [ring_sets[k] for k in th_ids if k.split("/")[-1] == t]
        others = set().union(*[ring_sets[k] for k in th_ids if k.split("/")[-1] != t])
        core = set.intersection(*mine) - others
        sig[t] = sorted(int(x) for x in core)
        print(f"ring core exclusive to theme {t:11s}: {len(core)} features {sig[t][:10]}")
    comp["theme_exclusive_ring_core"] = sig

out = dict(sae=a.sae, tau=a.tau, fmt_gen=a.fmt_gen, n_reference=len(refs),
           maxcos_percentiles_inactive=qs, frac_inactive_kept=float((pool >= a.tau).mean()),
           n_formatting_features=int(fmt.sum()), passages=rows, composition=comp)
json.dump(out, open(a.out, "w"), indent=1)
print("wrote", a.out)
