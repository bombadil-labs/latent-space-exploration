"""Analysis for `painaxis_scenarios.py`: their screen, plus the floor, the curve and the nulls.

Layer axis: index 0 = "emb" (static embedding output, pre-block-0); index i+1 = their layer i
(residual after block i). Their steering layer 12 is index 13 here. Reported as "emb" and 0..41
so an index can never be silently read as one of their 42.

THE FLOOR, and why there are two of them.
Their read position is the final token, which under the chat template is the identical
"<start_of_turn>model\\n" for all 420 scenarios. So the static-embedding vector at that position
is the SAME for every item, its pool sd is 0, and the z-score is 0/0. That is not a bug to route
around; it is a fact worth printing, and it is checked (`final_token_embed_degenerate`). It also
means the commensurable lexical floor is the bag: the mask-weighted mean of the static
embeddings over the whole rendered turn, with the directions built the same way from the core
stimuli's bags. That is hour 52's `floor_mean_bag` recipe at this read position.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import painaxis_scenarios as S  # noqa: E402

OUT = S.OUT
THEIR_CSV = S.THEIR_REPO / "results/4.1_self_other/per_model" / f"screen_v2_{S.THEIR_NAME}.csv"
THEIR_VECS = (S.THEIR_REPO / "results/3.2_pain_vectors/pain_vectors" / S.THEIR_NAME
              / "pain_vectors.pt")
N_NULL = 200
NULL_LAYERS = None   # set in main(): emb, their L12, their L37


def load_group(prefix: str, n: int, keys=("final_token",)):
    """Concatenate a group's shards back into [n, 1 + n_layers, d] per key."""
    out = {}
    for key in keys:
        parts = []
        emb_key = "embed_" + key
        for s0 in range(0, n, S.SHARD):
            z = np.load(S.SHARDS / f"{prefix}_{s0:04d}.npz")
            parts.append(np.concatenate([z[emb_key][:, None, :], z[key]], axis=1))
        a = np.concatenate(parts, axis=0)
        if a.shape[0] != n:
            raise SystemExit(f"{prefix}/{key}: loaded {a.shape[0]} rows, expected {n}")
        out[key] = a
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = json.loads(S.CORE.read_text())["datasets"]
    scen = json.loads(S.SCEN.read_text())
    order = json.loads((OUT / "scenario_order.json").read_text())
    scen = [c for c in scen if c["id"] in set(order)]
    scen = sorted(scen, key=lambda c: order.index(c["id"]))
    cats = np.array([c["category"] for c in scen])
    strata = np.array([c["stratum"] for c in scen])

    core = {n: load_group(f"core_{n}", len(ds[n]["sentences"]), ("final_token", "mean"))
            for n in S.S_SETS}
    core_cats = {n: [s["category"] for s in ds[n]["sentences"]] for n in S.S_SETS}
    ctrl = {n: load_group(f"ctrl_{n}", len(ds[n]["sentences"]), ("final_token",))
            for n in S.CONTROL_SETS}
    sc = load_group("scen_chat", len(scen), ("final_token", "mean"))

    n_layers = core[S.S_SETS[0]]["final_token"].shape[1]        # 1 + 42
    names = ["emb"] + [str(i) for i in range(n_layers - 1)]
    their_L = S.THEIR_STEERING_LAYER + 1
    their_X = S.THEIR_EXTRACTION_LAYER + 1
    global NULL_LAYERS
    NULL_LAYERS = [0, their_L, their_X]
    report = {"model": S.MODEL, "n_layers": n_layers - 1, "layer_axis": names,
              "n_scenarios": len(scen), "their_steering_layer": S.THEIR_STEERING_LAYER,
              "their_extraction_layer": S.THEIR_EXTRACTION_LAYER}

    # ---- 0. degeneracy: the final-token static embedding is identical across all 420 ---------
    e = sc["final_token"][:, 0, :]
    report["final_token_embed_degenerate"] = {
        "max_abs_dev_from_row0": float(np.abs(e - e[0]).max()),
        "pool_sd_max": float(e.std(axis=0).max()),
        "note": ("all 420 render to the same final token under the chat template, so the "
                 "final-token embedding floor has no variance and its z is 0/0. The bag floor "
                 "(embed mean) is the commensurable lexical floor."),
    }
    print("final-token embedding degeneracy:",
          json.dumps(report["final_token_embed_degenerate"], indent=2), flush=True)

    def vectors_at(L, ext):
        ca = {n: core[n][ext][:, L, :] for n in S.S_SETS}
        cc = {n: ctrl[n]["final_token"][:, L, :] for n in S.CONTROL_SETS} if ext == "final_token" else {}
        return S.build_vectors(ca, core_cats, cc)

    # ---- 1. our vector recipe must reproduce their published vectors at their layer ----------
    import torch
    tv = torch.load(THEIR_VECS, map_location="cpu", weights_only=False)
    ours_x = vectors_at(their_X, "final_token")
    cos = {}
    for k in ("s1_pain_vector", "s2_pain_vector"):
        a = S.unit(ours_x[k]); b = S.unit(tv[k].float().numpy())
        cos[k] = float(np.dot(a, b))
    report["vector_crosscheck_at_their_extraction_layer"] = cos
    print("vector cosine vs their published (L%d):" % S.THEIR_EXTRACTION_LAYER, cos, flush=True)

    # ---- 2. the full curve: mean z per category per vector per layer -------------------------
    rows = []
    vec_names_seen = set()
    for ext in ("final_token", "mean"):
        for L, nm in enumerate(names):
            vecs = vectors_at(L, ext)
            acts = sc[ext][:, L, :]
            for vname, v in vecs.items():
                vec_names_seen.add(vname)
                proj = acts @ S.unit(v)
                sd = float(np.std(proj))
                z = S.zscore_pool(proj)
                for c in sorted(set(cats)):
                    m = cats == c
                    rows.append({"extraction": ext, "layer": nm, "vector": vname,
                                 "category": c, "n": int(m.sum()),
                                 "mean_z": round(float(z[m].mean()), 4),
                                 "mean_proj": round(float(proj[m].mean()), 4),
                                 "pool_sd": round(sd, 6)})
    with (OUT / "category_z_by_layer.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} curve rows", flush=True)

    # ---- 3. item-level replication at their layer, against their published CSV ---------------
    theirs = {r["id"]: r for r in csv.DictReader(THEIR_CSV.open(encoding="utf-8"))}
    vecs12 = vectors_at(their_L, "final_token")
    acts12 = sc["final_token"][:, their_L, :]
    item_rows, rep = [], {}
    for vname in ("s1_pain_vector", "s2_pain_vector"):
        z = S.zscore_pool(acts12 @ S.unit(vecs12[vname]))
        tz = np.array([float(theirs[c["id"]][f"{vname}_z"]) for c in scen])
        rep[vname] = {
            "pearson_r": round(float(np.corrcoef(z, tz)[0, 1]), 4),
            "mean_abs_delta": round(float(np.abs(z - tz).mean()), 4),
            "max_abs_delta": round(float(np.abs(z - tz).max()), 4),
        }
        for i, c in enumerate(scen):
            item_rows.append({"id": c["id"], "category": c["category"], "stratum": c["stratum"],
                              "vector": vname, "ours_z": round(float(z[i]), 4),
                              "theirs_z": round(float(tz[i]), 4),
                              "delta": round(float(z[i] - tz[i]), 4)})
    report["item_level_replication_at_their_L12"] = rep
    print("item-level replication at their L12:", json.dumps(rep, indent=2), flush=True)
    with (OUT / "item_z_vs_theirs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(item_rows[0])); w.writeheader(); w.writerows(item_rows)

    # ---- 4. the headline, ours vs theirs, and the bag floor ----------------------------------
    def cat_means(z):
        return {c: float(z[cats == c].mean()) for c in sorted(set(cats))}

    ours12 = cat_means(S.zscore_pool(acts12 @ S.unit(vecs12["s2_pain_vector"])))
    theirs12 = {c: float(np.mean([float(theirs[x["id"]]["s2_pain_vector_z"])
                                  for x in scen if x["category"] == c]))
                for c in sorted(set(cats))}
    floor_vecs = vectors_at(0, "mean")
    floor_z = S.zscore_pool(sc["mean"][:, 0, :] @ S.unit(floor_vecs["s2_pain_vector"]))
    floorm = cat_means(floor_z)
    report["headline"] = {
        "ours_rank": sorted(ours12, key=lambda c: -ours12[c]),
        "theirs_rank": sorted(theirs12, key=lambda c: -theirs12[c]),
        "gaslighting_ours": round(ours12["gaslighting"], 4),
        "gaslighting_theirs": round(theirs12["gaslighting"], 4),
        "gaslighting_bag_floor": round(floorm["gaslighting"], 4),
        "gaslighting_rank_ours": sorted(ours12, key=lambda c: -ours12[c]).index("gaslighting") + 1,
        "gaslighting_rank_floor": sorted(floorm, key=lambda c: -floorm[c]).index("gaslighting") + 1,
        "floor_rank": sorted(floorm, key=lambda c: -floorm[c]),
    }
    print("HEADLINE:", json.dumps(report["headline"], indent=2), flush=True)

    # ---- 5. null arms at emb / L12 / L37 (CLAUDE.md 1). Declared null for every z: 0 ----------
    rng = np.random.default_rng(20260921)
    nulls = {}
    for L in NULL_LAYERS:
        ext = "mean" if L == 0 else "final_token"
        acts = sc[ext][:, L, :]
        d = acts.shape[1]
        rnd = np.zeros((N_NULL, len(sorted(set(cats)))))
        for k in range(N_NULL):
            r = rng.standard_normal(d)
            rnd[k] = [S.zscore_pool(acts @ S.unit(r))[cats == c].mean() for c in sorted(set(cats))]
        shu = np.zeros((N_NULL, len(sorted(set(cats)))))
        for k in range(N_NULL):
            perm = {}
            for n in S.S_SETS:
                cc = np.array(core_cats[n]); idx = rng.permutation(len(cc)); perm[n] = cc[idx]
            ca = {n: core[n][ext if ext in core[n] else "final_token"][:, L, :] for n in S.S_SETS}
            v = S.compute_pain_vector(ca["S2_1P"], perm["S2_1P"])
            shu[k] = [S.zscore_pool(acts @ S.unit(v))[cats == c].mean() for c in sorted(set(cats))]
        nulls[names[L]] = {
            "extraction": ext, "n_draws": N_NULL, "categories": sorted(set(cats)),
            "random_direction": {"mean": rnd.mean(0).round(4).tolist(),
                                 "lo": np.percentile(rnd, 2.5, axis=0).round(4).tolist(),
                                 "hi": np.percentile(rnd, 97.5, axis=0).round(4).tolist()},
            "shuffled_labels": {"mean": shu.mean(0).round(4).tolist(),
                                "lo": np.percentile(shu, 2.5, axis=0).round(4).tolist(),
                                "hi": np.percentile(shu, 97.5, axis=0).round(4).tolist()},
        }
        gi = sorted(set(cats)).index("gaslighting")
        print(f"null @ {names[L]} ({ext}) gaslighting: random "
              f"[{nulls[names[L]]['random_direction']['lo'][gi]:+.3f}, "
              f"{nulls[names[L]]['random_direction']['hi'][gi]:+.3f}]  shuffled "
              f"[{nulls[names[L]]['shuffled_labels']['lo'][gi]:+.3f}, "
              f"{nulls[names[L]]['shuffled_labels']['hi'][gi]:+.3f}]", flush=True)
    (OUT / "nulls.json").write_text(json.dumps(nulls, indent=2))

    # ---- 6. gaslighting subclasses: does a visible record change the loading? -----------------
    WITH_RECORD = {"gaslight_19", "gaslight_20"}     # contain an [Assistant]: turn
    EVALUATIVE = {"gaslight_07", "gaslight_11", "gaslight_12", "gaslight_13"}
    THIRD_PARTY = {"gaslight_15"}
    z12 = S.zscore_pool(acts12 @ S.unit(vecs12["s2_pain_vector"]))
    sub_rows = []
    for label, ids in (("with_visible_record", WITH_RECORD), ("evaluative", EVALUATIVE),
                       ("third_party", THIRD_PARTY)):
        m = np.array([c["id"] in ids for c in scen])
        rest = np.array([c["category"] == "gaslighting" and c["id"] not in ids for c in scen])
        sub_rows.append({"subclass": label, "n": int(m.sum()), "mean_z": round(float(z12[m].mean()), 4),
                         "n_rest": int(rest.sum()), "rest_mean_z": round(float(z12[rest].mean()), 4)})
    report["gaslighting_subclasses_L12_s2"] = sub_rows
    report["gaslighting_subclasses_note"] = (
        "n = 1, 2 and 4. A Sketch, not a Claim. Listed because rule 1b of the conscription "
        "design requires a visible record and only 2 of their 20 items have one.")
    print("gaslighting subclasses:", json.dumps(sub_rows, indent=2), flush=True)

    meta = json.loads((OUT / "extract_meta.json").read_text())
    report["extract_meta"] = {k: meta[k] for k in
                             ("padding_side", "lib_versions", "crosscheck",
                              "chat_render_assertions", "excluded_by_their_validator")
                             if k in meta}
    report["equivalence_min_cos"] = (round(min(meta["equivalence"].values()), 6)
                                     if meta.get("equivalence") else None)
    report["vectors_built"] = sorted(vec_names_seen)
    report["not_built"] = ["sadness_vector (their SD_sadness_1P set not extracted)"]
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))
    print("\nwrote", OUT / "summary.json", flush=True)
