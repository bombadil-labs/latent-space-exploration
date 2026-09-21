"""Null arms at EVERY layer for the Section 4.1 screen, and the gain over the lexical floor.

The three-layer nulls in `painaxis_scenarios_analyze.py` showed something the paper cannot see
from one layer: at their steering layer the gaslighting z clears the random-direction band by
about a quarter of a z, at the static-embedding bag it sits exactly ON that band, and at their
own extraction layer it is BELOW it. Which of those is the model and which is the layer is not
answerable from three points, so this computes both null arms at all 43.

Why a random direction is not a strawman here. Every category z is a mean over that category's 20
items, taken against the pool of 420. The 420 cluster by category in activation space, so ANY
direction -- including an arbitrary one -- separates categories to some degree. The question their
screen never asks is how much, and that is exactly what this arm measures. Declared null: 0.

Two arms, as CLAUDE.md 1 requires:
  random_direction  - r ~ N(0, I), projected and z-scored through the identical path.
  shuffled_labels   - the pain-vector recipe refit on category labels permuted within the
                      stimulus set, so the denoising PCA and the difference in means are fitted
                      on labels that carry no information. The scenarios are untouched.
There is no patch, so no-patch has no referent; said rather than dropped.

Usage: python scripts/painaxis_scenarios_nulls.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import painaxis_scenarios as S            # noqa: E402
import painaxis_scenarios_analyze as A    # noqa: E402

N_RAND = 500
N_SHUF = 200
OUT = S.OUT


def main() -> None:
    ds = json.loads(S.CORE.read_text())["datasets"]
    scen_all = json.loads(S.SCEN.read_text())
    order = json.loads((OUT / "scenario_order.json").read_text())
    scen = sorted([c for c in scen_all if c["id"] in set(order)], key=lambda c: order.index(c["id"]))
    cats = np.array([c["category"] for c in scen])
    ucats = sorted(set(cats))
    masks = {c: cats == c for c in ucats}

    core = {n: A.load_group(f"core_{n}", len(ds[n]["sentences"]), ("final_token", "mean"))
            for n in S.S_SETS}
    core_cats = {n: np.array([s["category"] for s in ds[n]["sentences"]]) for n in S.S_SETS}
    sc = A.load_group("scen_chat", len(scen), ("final_token", "mean"))
    n_layers = sc["final_token"].shape[1]
    names = ["emb"] + [str(i) for i in range(n_layers - 1)]

    rng = np.random.default_rng(770021)
    rows = []
    for ext in ("final_token", "mean"):
        for L, nm in enumerate(names):
            t0 = time.time()
            acts = sc[ext][:, L, :]
            pool_sd = float(np.std(acts @ S.unit(np.ones(acts.shape[1]))))

            # --- treatment: their s2 pain vector, rebuilt at this layer -------------------
            v = S.compute_pain_vector(core["S2_1P"][ext][:, L, :], core_cats["S2_1P"])
            tz = S.zscore_pool(acts @ S.unit(v))
            treat = {c: float(tz[masks[c]].mean()) for c in ucats}

            # --- random-direction arm ----------------------------------------------------
            rnd = np.zeros((N_RAND, len(ucats)))
            R = rng.standard_normal((N_RAND, acts.shape[1]))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            P = acts @ R.T                                     # [420, N_RAND]
            Z = (P - P.mean(0)) / (P.std(0) + 1e-8)
            for ci, c in enumerate(ucats):
                rnd[:, ci] = Z[masks[c]].mean(0)

            # --- shuffled-label arm ------------------------------------------------------
            shu = np.zeros((N_SHUF, len(ucats)))
            ca = core["S2_1P"][ext][:, L, :]
            for k in range(N_SHUF):
                perm = core_cats["S2_1P"][rng.permutation(len(core_cats["S2_1P"]))]
                vs = S.compute_pain_vector(ca, perm)
                zs = S.zscore_pool(acts @ S.unit(vs))
                shu[k] = [zs[masks[c]].mean() for c in ucats]

            for ci, c in enumerate(ucats):
                t = treat[c]
                rows.append({
                    "extraction": ext, "layer": nm, "category": c, "treat_z": round(t, 4),
                    "rand_lo": round(float(np.percentile(rnd[:, ci], 2.5)), 4),
                    "rand_hi": round(float(np.percentile(rnd[:, ci], 97.5)), 4),
                    "rand_p_two_sided": round(float((np.abs(rnd[:, ci]) >= abs(t)).mean()), 4),
                    "shuf_lo": round(float(np.percentile(shu[:, ci], 2.5)), 4),
                    "shuf_hi": round(float(np.percentile(shu[:, ci], 97.5)), 4),
                    "shuf_p_two_sided": round(float((np.abs(shu[:, ci]) >= abs(t)).mean()), 4),
                    "clears_both": int(abs(t) > np.percentile(np.abs(rnd[:, ci]), 97.5)
                                       and abs(t) > np.percentile(np.abs(shu[:, ci]), 97.5)),
                })
            print(f"  {ext} {nm}: gaslighting {treat['gaslighting']:+.3f} "
                  f"rand95 {np.percentile(np.abs(rnd[:, ucats.index('gaslighting')]), 95):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with (OUT / "nulls_by_layer.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    g = [r for r in rows if r["category"] == "gaslighting"]
    clears = [r for r in g if r["clears_both"]]
    print(f"\nwrote {len(rows)} rows. gaslighting clears BOTH nulls at "
          f"{len(clears)}/{len(g)} (layer, extraction) cells:", flush=True)
    for r in clears:
        print(f"   {r['extraction']:12s} L{r['layer']:>3s}  z={r['treat_z']:+.3f}  "
              f"rand[{r['rand_lo']:+.3f},{r['rand_hi']:+.3f}]  "
              f"shuf[{r['shuf_lo']:+.3f},{r['shuf_hi']:+.3f}]", flush=True)


if __name__ == "__main__":
    main()
