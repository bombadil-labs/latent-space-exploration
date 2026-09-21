"""Tier B: the pain-axis replication on google/gemma-2-9b-it, through NDIF.

Extraction: `src/lsx/core/painaxis_remote.extract_pooled`, which carries the §7 assertions.
Analysis:  `scripts/painaxis_analyze`'s functions, IMPORTED UNCHANGED -- they are the faithful
           Tier A port of their `compute_pain_vector` / `compute_auc` / `kfold_curve`.

Layer indexing, determined and not assumed: their published curve
(`s1_kfold_layer_curves.csv`) has exactly 42 rows for Gemma_2_9B_instruct, layers 0..41, and
gemma-2-9b-it has 42 blocks; their `extract_activations` reads
`cache["blocks.{layer}.hook_resid_post"]`. So their layer n is the residual AFTER block n, with
no embedding row. Ours matches index for index. `embed` is OUR extra row and is reported
separately so it can never be mistaken for one of their 42.

Usage:
    python scripts/painaxis_tierB.py extract      # resumable, per-shard checkpoints
    python scripts/painaxis_tierB.py analyze
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL = "google/gemma-2-9b-it"
THEIR_NAME = "Gemma_2_9B_instruct"
SETS = ["S1_1P", "S1_3P", "S2_1P", "S2_3P"]
DATA = ROOT / "prompts/external/pain_axis/3.1_pain_and_control_datasets.json"
OUT = ROOT / "results/painaxis_tierB"
SHARDS = OUT / "shards"
THEIRS = pathlib.Path("/tmp/claude-0/Pain-axis/results/3.2_pain_vectors/auc_tables")

BATCH = 20          # per NDIF job
SHARD = 100         # per checkpoint file


# --------------------------------------------------------------------------------- extraction
def extract() -> None:
    from lsx.core import painaxis_remote as pr
    from lsx.core.remote import RemoteLM

    OUT.mkdir(parents=True, exist_ok=True)
    SHARDS.mkdir(parents=True, exist_ok=True)
    ds = json.loads(DATA.read_text())["datasets"]

    rlm = RemoteLM(MODEL)
    n_layers = len(rlm.blocks)
    print(f"model={MODEL} blocks={n_layers} d={rlm.model.config.hidden_size} "
          f"padding_side={rlm.padding_side} bos={rlm.tok.bos_token_id} "
          f"pad={rlm.tok.pad_token_id}", flush=True)
    if n_layers != 42:
        raise SystemExit(f"expected 42 blocks for gemma-2-9b-it, got {n_layers}")

    meta = {"model": MODEL, "n_layers": n_layers, "batch": BATCH, "shard": SHARD,
            "padding_side": rlm.padding_side, "embed_scale_note": pr.EMBED_SCALE_NOTE,
            "lib_versions": rlm.lib_versions(), "equivalence": {}, "crosscheck": None,
            "jobs_submitted": 0, "jobs_failed": 0, "shards_failed": []}
    mpath = OUT / "extract_meta.json"
    if mpath.exists():
        meta.update(json.loads(mpath.read_text()))

    done_crosscheck = meta.get("crosscheck") is not None
    for name in SETS:
        sents = ds[name]["sentences"]
        prompts = [s["prompt"] for s in sents]
        for s0 in range(0, len(prompts), SHARD):
            chunk = prompts[s0:s0 + SHARD]
            path = SHARDS / f"{name}_{s0:04d}.npz"
            if path.exists():
                print(f"  {name}[{s0}]: cached", flush=True)
                continue
            print(f"  {name}[{s0}:{s0 + len(chunk)}] ...", flush=True)
            t0 = time.time()
            try:
                p = pr.extract_pooled(rlm, chunk, batch_size=BATCH)
            except Exception as e:                      # noqa: BLE001
                # per-shard checkpointing: a lost job costs 100 sentences, not the run.
                meta["shards_failed"].append({"shard": path.name, "error": repr(e)[:300]})
                meta["jobs_failed"] = meta.get("jobs_failed", 0) + 1
                print(f"  !! {name}[{s0}] FAILED: {e!r}", flush=True)
                mpath.write_text(json.dumps(meta, indent=1))
                continue
            meta["jobs_submitted"] += p.n_jobs
            meta["equivalence"].update({f"{name}:{k}": v for k, v in p.equivalence.items()})
            if not done_crosscheck:
                # tie the fast multi-layer path to the reviewed single-layer one, once, live.
                meta["crosscheck"] = {
                    "layer": 17, "n_texts": 6,
                    **pr.cross_check_against_asserted_path(
                        rlm, chunk[:6], 17,
                        {"mean": p.mean[:6], "final_token": p.final_token[:6]})}
                done_crosscheck = True
                print(f"  crosscheck vs remote_residuals: {meta['crosscheck']}", flush=True)
            np.savez_compressed(
                path, final_token=p.final_token, mean=p.mean,
                embed_final_token=p.embed_final_token, embed_mean=p.embed_mean,
                categories=np.array([s["category"] for s in sents[s0:s0 + len(chunk)]]),
                sets=np.array([s["set"] for s in sents[s0:s0 + len(chunk)]]),
                prompts=np.array(chunk))
            print(f"  {name}[{s0}] {len(chunk)} sentences, {p.n_jobs} jobs, "
                  f"{time.time() - t0:.0f}s -> {path.name}", flush=True)
            mpath.write_text(json.dumps(meta, indent=1))

    mpath.write_text(json.dumps(meta, indent=1))
    print("EXTRACTION DONE", flush=True)


def load_set(name: str) -> dict:
    parts = sorted(SHARDS.glob(f"{name}_*.npz"))
    if not parts:
        raise SystemExit(f"no shards for {name}")
    zs = [np.load(p, allow_pickle=False) for p in parts]
    out = {k: np.concatenate([z[k] for z in zs], axis=0)
           for k in ("final_token", "mean", "embed_final_token", "embed_mean",
                     "categories", "sets", "prompts")}
    return out


# ----------------------------------------------------------------------------------- analysis
def analyze() -> None:
    import painaxis_analyze as pa                       # their functions, unchanged

    rows, summary_rows, embed_rows = [], [], []
    data = {name: load_set(name) for name in SETS}
    n_layers = data[SETS[0]]["final_token"].shape[1]

    for ext in ("final_token", "mean"):
        for name in SETS:
            d = data[name]
            acts = d[ext]                                # [n, L, d]
            cats = np.asarray(d["categories"])
            sets = np.asarray(d["sets"])
            assert acts.shape[0] == len(cats) == len(sets) == 200, acts.shape
            for r in pa.kfold_curve(acts, cats, sets, range(n_layers)):
                rows.append(dict(model=THEIR_NAME, extraction=ext, dataset=name, **r))
            # our extra row: the true static embeddings, same K-fold, same functions.
            ekey = "embed_final_token" if ext == "final_token" else "embed_mean"
            e = d[ekey][:, None, :]
            er = pa.kfold_curve(e, cats, sets, [0])[0]
            er.update(model=THEIR_NAME, extraction=ext, dataset=name, layer="embed")
            embed_rows.append(er)

    # ---- summary in their s1_kfold_summary.csv shape
    def curve_mean(ext, dss):
        out = {}
        for L in range(n_layers):
            vals = [r["auc_vs_all_controls"] for r in rows
                    if r["extraction"] == ext and r["dataset"] in dss and r["layer"] == L]
            out[L] = float(np.mean(vals))
        return out

    def at(ext, ds, L):
        return [r["auc_vs_all_controls"] for r in rows
                if r["extraction"] == ext and r["dataset"] == ds and r["layer"] == L][0]

    for ext in ("final_token", "mean"):
        s2 = curve_mean(ext, {"S2_1P", "S2_3P"})
        s1 = curve_mean(ext, {"S1_1P", "S1_3P"})
        s2_layer = int(max(s2, key=s2.get))              # 01.best_layer, from the S2 curve
        best_L = int(max(s1, key=s1.get))                # 08.best_L, from the S1 curve
        summary_rows.append(dict(
            model=THEIR_NAME, extraction=ext, s2_layer=s2_layer,
            s1_heldout_auc_at_s2_layer=s1[s2_layer],
            s1_best_layer=best_L, s1_heldout_auc_at_best_layer=s1[best_L],
            s1_1P_heldout_at_best=at(ext, "S1_1P", best_L),
            s1_3P_heldout_at_best=at(ext, "S1_3P", best_L)))

    OUT.mkdir(parents=True, exist_ok=True)
    pa.write_csv(OUT / "s1_kfold_layer_curves.csv", rows)
    pa.write_csv(OUT / "s1_kfold_summary.csv", summary_rows)
    pa.write_csv(OUT / "embed_layer_auc.csv", embed_rows)

    # ---- theirs, side by side (S1_1P and S1_3P are the datasets their CSV carries)
    import csv as _csv
    theirs = [r for r in _csv.DictReader(open(THEIRS / "s1_kfold_layer_curves.csv"))
              if r["model"] == THEIR_NAME]
    comp = []
    for ext in ("final_token", "mean"):
        for L in range(n_layers):
            t = {r["dataset"]: float(r["auc_vs_all_controls"]) for r in theirs
                 if r["extraction"] == ext and int(r["layer"]) == L}
            o1, o3 = at(ext, "S1_1P", L), at(ext, "S1_3P", L)
            tm = float(np.mean([t["S1_1P"], t["S1_3P"]]))
            om = float(np.mean([o1, o3]))
            comp.append(dict(extraction=ext, layer=L,
                             theirs_S1_1P=t["S1_1P"], ours_S1_1P=round(o1, 4),
                             theirs_S1_3P=t["S1_3P"], ours_S1_3P=round(o3, 4),
                             theirs_mean=round(tm, 4), ours_mean=round(om, 4),
                             delta=round(om - tm, 4)))
    pa.write_csv(OUT / "curve_comparison.csv", comp)

    for ext in ("final_token", "mean"):
        c = [r for r in comp if r["extraction"] == ext]
        t = np.array([r["theirs_mean"] for r in c])
        o = np.array([r["ours_mean"] for r in c])
        r = float(np.corrcoef(t, o)[0, 1])
        print(f"{ext}: mean|delta|={np.abs(o - t).mean():.4f} max|delta|={np.abs(o - t).max():.4f} "
              f"pearson_r={r:.4f} spearman-ish peak theirs=L{int(t.argmax())}({t.max():.4f}) "
              f"ours=L{int(o.argmax())}({o.max():.4f})", flush=True)
    for r in summary_rows:
        print(r, flush=True)
    for r in embed_rows:
        print({k: r[k] for k in ("extraction", "dataset", "layer", "auc_vs_all_controls")},
              flush=True)
    print("ANALYSIS DONE", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "extract"
    {"extract": extract, "analyze": analyze}[cmd]()
