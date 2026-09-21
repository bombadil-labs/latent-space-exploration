"""GOAL 2 instalment 2: a band on `gain_over_floor`.

`results/notes/painaxis_floor_nulls.md` §6 names the gap in its own words: the +0.046 gain of
the 28-block network over the static-embedding bag "is a difference of two AUCs measured on the
same sentences; its null needs a paired resample over sentence sets, which I did not run."

This runs it. Two bands, each with a stated meaning, because they answer different questions:

  (1) SPLIT BAND. Repeated K-fold over R fold seeds. Answers: how much does the published number
      move if you re-draw the arbitrary fold assignment? Says nothing about sentence sampling.

  (2) CLUSTER BOOTSTRAP BAND. Resample the sentence SETS with replacement (the sets are the
      exchangeable unit: each holds exactly one sentence per category, and the K-fold already
      splits on them). Answers: how much does it move if you re-draw the stimuli?

Leakage rule for (2), stated because it is the only thing that makes the bootstrap honest: a set
drawn twice occupies two slots, and if those slots landed in different folds the SAME sentences
would sit in train and test. So folds are assigned at the level of the set IDENTITY, before
multiplicity is applied -- every copy of a set is always on the same side of the split.

PAIRED throughout: within a replicate, the treatment layer and the embedding floor see the
identical resampled sentences and the identical folds, so the difference is a within-replicate
quantity and the shared stimulus noise cancels in it.

Reported at EVERY layer, not at the peak (CLAUDE.md non-negotiable 4). The peak row is printed
for continuity with the note, flagged as such, and is not how the layer was chosen.

Null: gain = 0 (the network buys nothing over the bag of token vectors).

Usage:  python scripts/painaxis_gain_band.py cell <extraction> <dataset>
        python scripts/painaxis_gain_band.py report
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

SETS = ["S1_1P", "S1_3P", "S2_1P", "S2_3P"]
EXTRACTIONS = ["final_token", "mean"]
IN = ROOT / "results/painaxis_floor_nulls"
OUT = ROOT / "results/painaxis_gain_band"
N_BOOT = 200
N_SEEDS = 200
CKPT = 10


def fold_masks(PA, set_ids, sets_arr, seed):
    """K-fold over set IDENTITIES. Returns [(train_mask, test_mask)] over sentence rows."""
    uniq = sorted(set(set_ids))
    kf = PA.KFold(n_splits=PA.N_FOLDS, shuffle=True, random_state=seed)
    out = []
    for tr, te in kf.split(uniq):
        out.append((np.isin(sets_arr, [uniq[i] for i in tr]),
                    np.isin(sets_arr, [uniq[i] for i in te])))
    return out


def curve(PA, acts_all, cats, splits, layers):
    """Fold-averaged held-out `auc_vs_all_controls` per layer, their functions unchanged."""
    out = np.full(len(layers), np.nan)
    for li, L in enumerate(layers):
        acts = acts_all[:, L, :]
        vals = []
        for trm, tem in splits:
            if trm.sum() < 2 or tem.sum() < 2:
                continue
            v = PA.compute_pain_vector(acts[trm], cats[trm], "all_controls")
            a = PA.compute_auc(acts[tem], cats[tem], v)
            if not np.isnan(a):
                vals.append(a)
        if vals:
            out[li] = float(np.mean(vals))
    return out


def cluster_resample(uniq, rows_by_set, rng):
    """One cluster-bootstrap replicate. Returns (row_index, set_label_per_row, distinct_ids).

    Separated out so the no-leak property is testable: every row carrying set id `s` gets the
    label `s` whatever its multiplicity, and folds are later built from the DISTINCT ids, so
    duplicate copies of a set cannot land on opposite sides of a split."""
    drawn = rng.choice(np.asarray(uniq, dtype=object), size=len(uniq), replace=True)
    idx = np.concatenate([rows_by_set[s] for s in drawn])
    labels = np.concatenate([np.full(len(rows_by_set[s]), s, dtype=object) for s in drawn])
    return idx, labels, sorted(set(drawn.tolist()))


def run_cell(ext: str, ds: str) -> None:
    import painaxis_analyze as PA
    import painaxis_floor_nulls as FN

    OUT.mkdir(parents=True, exist_ok=True)
    d = FN.load(ds)
    A = FN.stack_with_embedding(d, ext)          # index 0 = embedding, i+1 = block i
    cats, sets_ = np.asarray(d["categories"]), np.asarray(d["sets"])
    layers = list(range(A.shape[1]))
    t0 = time.time()

    # Checkpoint every CKPT replicates. The first run of this was killed by the sandbox 20
    # minutes in with nothing on disk, because results were only written at the end. Long CPU
    # jobs in this environment have to assume they will be interrupted.
    ck = OUT / f"gain_{ext}_{ds}.partial.npz"
    split_curves = np.full((N_SEEDS, len(layers)), np.nan)
    boot_curves = np.full((N_BOOT, len(layers)), np.nan)
    base = None
    k0_split = k0_boot = 0
    if ck.exists():
        z = np.load(ck)
        if z["split"].shape == split_curves.shape and z["boot"].shape == boot_curves.shape:
            split_curves, boot_curves, base = z["split"], z["boot"], z["base"]
            k0_split, k0_boot = int(z["k_split"]), int(z["k_boot"])
            print(f"  {ext} {ds} resuming at split={k0_split} boot={k0_boot}", flush=True)

    def save(k_split, k_boot):
        np.savez(ck, split=split_curves, boot=boot_curves, base=base,
                 k_split=k_split, k_boot=k_boot)

    # --- point estimate on the published split ------------------------------------------
    if base is None:
        base = curve(PA, A, cats, fold_masks(PA, sets_.tolist(), sets_, PA.RANDOM_SEED), layers)
        save(0, 0)

    # --- (1) split band: re-draw the fold assignment ------------------------------------
    for k in range(k0_split, N_SEEDS):
        split_curves[k] = curve(PA, A, cats, fold_masks(PA, sets_.tolist(), sets_, 1000 + k), layers)
        if (k + 1) % CKPT == 0 or k + 1 == N_SEEDS:
            save(k + 1, k0_boot)
            print(f"  {ext} {ds} split {k+1}/{N_SEEDS} {time.time()-t0:.0f}s", flush=True)

    # --- (2) cluster bootstrap over sets -------------------------------------------------
    uniq = sorted(set(sets_.tolist()))
    rows_by_set = {s: np.where(sets_ == s)[0] for s in uniq}
    for k in range(k0_boot, N_BOOT):
        # The rng is reseeded per replicate so a resume reproduces the same draw it would have
        # made in an uninterrupted run; a single stream advanced by the loop would not.
        rng = np.random.default_rng([abs(hash((ext, ds))) % (2**31), k])
        idx, bsets, present = cluster_resample(uniq, rows_by_set, rng)
        # folds assigned on the DISTINCT ids present, so duplicates never straddle the split
        if len(present) >= PA.N_FOLDS:
            splits = fold_masks(PA, present, bsets, PA.RANDOM_SEED)
            boot_curves[k] = curve(PA, A[idx], cats[idx], splits, layers)
        if (k + 1) % CKPT == 0 or k + 1 == N_BOOT:
            save(N_SEEDS, k + 1)
            print(f"  {ext} {ds} boot {k+1}/{N_BOOT} {time.time()-t0:.0f}s", flush=True)

    (OUT / f"gain_{ext}_{ds}.json").write_text(json.dumps({
        "extraction": ext, "dataset": ds, "n_boot": N_BOOT, "n_seeds": N_SEEDS,
        "layer_axis": ["emb"] + [str(i) for i in range(len(layers) - 1)],
        "base": base.tolist(),
        "split_curves": split_curves.tolist(),
        "boot_curves": boot_curves.tolist(),
    }))
    ck.unlink(missing_ok=True)
    print(f"CELL DONE {ext} {ds} {time.time()-t0:.0f}s", flush=True)


def _band(g):
    g = g[np.isfinite(g)]
    if g.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    return (float(np.mean(g)), float(np.percentile(g, 2.5)),
            float(np.percentile(g, 97.5)), float((g <= 0).mean()))


def report() -> None:
    import csv
    rows = []
    for ext in EXTRACTIONS:
        for ds in SETS:
            p = OUT / f"gain_{ext}_{ds}.json"
            if not p.exists():
                continue
            j = json.loads(p.read_text())
            names = j["layer_axis"]
            base = np.array(j["base"])
            sc = np.array(j["split_curves"])
            bc = np.array(j["boot_curves"])
            # gain is WITHIN-replicate: layer L minus the embedding column of the same replicate
            for li, nm in enumerate(names):
                if nm == "emb":
                    continue
                sm, slo, shi, _ = _band(sc[:, li] - sc[:, 0])
                bm, blo, bhi, pneg = _band(bc[:, li] - bc[:, 0])
                rows.append({
                    "extraction": ext, "dataset": ds, "layer": nm,
                    "treat_auc": round(float(base[li]), 4),
                    "floor_auc": round(float(base[0]), 4),
                    "gain": round(float(base[li] - base[0]), 4),
                    "split_mean": round(sm, 4), "split_lo": round(slo, 4), "split_hi": round(shi, 4),
                    "boot_mean": round(bm, 4), "boot_lo": round(blo, 4), "boot_hi": round(bhi, 4),
                    "boot_frac_le_0": round(pneg, 4),
                })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "gain_bands.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT/'gain_bands.csv'}")
    for r in rows:
        if r["dataset"] == "S2_1P" and r["layer"] == "21":
            print("the note's row:", json.dumps(r, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "cell":
        run_cell(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "report":
        report()
    else:
        raise SystemExit("cell <ext> <ds> | report")
