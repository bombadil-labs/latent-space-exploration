"""GOAL 2 instalment 1: the true embedding floor, and null arms at every layer.

Two things Tier A (and the paper it ports) left unmeasured on Qwen2.5-1.5B-Instruct:

  (A) THE EMBEDDING FLOOR. Their curves and ours start at the residual AFTER block 0. The static
      embedding output, `hidden_states[0]` (Qwen applies RoPE inside attention, so this really is
      a position-free bag of token vectors), has never been read. We re-extract capturing it and
      run the IDENTICAL held-out AUC pipeline on it.
      Degeneracy check that must pass first: every prompt ends with the identical token, so the
      final-token embedding is constant within a set and its AUC must be exactly 0.5.

  (B) NULL ARMS (CLAUDE.md non-negotiable 1). A random-direction arm and a shuffled-label arm at
      every layer. Declared null for both: 0.5.

Everything here imports the Tier A analysis functions from `scripts/painaxis_analyze.py`; none of
their arithmetic is reimplemented. Extraction mirrors `scripts/painaxis_extract.py` exactly (batch
size 1, no chat template, BOS prepended as `eos_token_id`) with ONE extra capture: the embedding.

Layer axis used throughout: index 0 = "emb" (embedding output, pre-block-0); index i+1 = residual
after block i, i.e. Tier A's layer i. Reported as "emb" and 0..27.

Usage:  python scripts/painaxis_floor_nulls.py extract
        python scripts/painaxis_floor_nulls.py analyze
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SETS = ["S1_1P", "S1_3P", "S2_1P", "S2_3P"]
DATA = ROOT / "prompts/external/pain_axis/3.1_pain_and_control_datasets.json"
OUT = ROOT / "results/painaxis_floor_nulls"
EXTRACTIONS = ["final_token", "mean"]


# ---------------------------------------------------------------- (A) extraction
def extract() -> None:
    from lsx.model import LM

    OUT.mkdir(parents=True, exist_ok=True)
    ds = json.loads(DATA.read_text())["datasets"]
    lm = LM.from_pretrained(MODEL, device="cpu", dtype=torch.float32)
    n_layers, d = lm.n_layers, lm.d_model
    bos = lm.tok.bos_token_id
    if bos is None:  # identical reasoning to painaxis_extract.py
        bos = lm.tok.eos_token_id
    print(f"model={MODEL} layers={n_layers} d={d} bos_id={bos}", flush=True)

    grabbed: list[torch.Tensor] = []
    emb_grab: list[torch.Tensor] = []

    def hook(mod, args, out):
        grabbed.append((out[0] if isinstance(out, tuple) else out).detach()[0])

    def emb_hook(mod, args, out):
        emb_grab.append((out[0] if isinstance(out, tuple) else out).detach()[0])

    handles = [b.register_forward_hook(hook) for b in lm.blocks]
    handles.append(lm.model.get_input_embeddings().register_forward_hook(emb_hook))
    try:
        for name in SETS:
            path = OUT / f"acts_{name}.npz"
            if path.exists():
                print(f"{name}: cached", flush=True)
                continue
            sents = ds[name]["sentences"]
            n = len(sents)
            ft = np.zeros((n, n_layers, d), dtype=np.float32)
            mn = np.zeros_like(ft)
            eft = np.zeros((n, d), dtype=np.float32)
            emn = np.zeros((n, d), dtype=np.float32)
            last_tok = []
            t0 = time.time()
            for i, s in enumerate(sents):
                ids = lm.tok(s["prompt"], add_special_tokens=False)["input_ids"]
                last_tok.append(int(ids[-1]))
                ids = torch.tensor([[bos] + ids])
                grabbed.clear()
                emb_grab.clear()
                with torch.no_grad():
                    lm.model(input_ids=ids)
                assert len(grabbed) == n_layers, (len(grabbed), n_layers)
                assert len(emb_grab) == 1, len(emb_grab)
                stack = torch.stack(grabbed)              # [L, seq, d]
                assert stack.shape[1] == ids.shape[1]
                e = emb_grab[0]                            # [seq, d]
                assert e.shape[0] == ids.shape[1]
                ft[i] = stack[:, -1, :].numpy()
                mn[i] = stack.mean(dim=1).numpy()
                eft[i] = e[-1, :].numpy()
                emn[i] = e.mean(dim=0).numpy()
                if i % 50 == 0:
                    print(f"  {name} {i}/{n} {time.time()-t0:.0f}s", flush=True)
            np.savez_compressed(
                path, final_token=ft, mean=mn,
                embed_final_token=eft, embed_mean=emn,
                last_token_id=np.array(last_tok),
                categories=np.array([s["category"] for s in sents]),
                sets=np.array([s["set"] for s in sents]),
                prompts=np.array([s["prompt"] for s in sents]))
            print(f"{name}: {n} sentences in {time.time()-t0:.0f}s -> {path}", flush=True)
    finally:
        for h in handles:
            h.remove()
    print("EXTRACTION DONE", flush=True)


# ---------------------------------------------------------------- shared helpers
def load(name):
    z = np.load(OUT / f"acts_{name}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def stack_with_embedding(d, ext):
    """[n, 1 + n_layers, dim]: index 0 = embedding output, index i+1 = Tier A layer i."""
    emb = d[f"embed_{ext}"][:, None, :]
    return np.concatenate([emb, d[ext]], axis=1)


LAYER_NAMES = None  # set in analyze()


def curve_all_controls(PA, acts_all, cats, sets, layers):
    """Their K-fold, `auc_vs_all_controls` only, calling THEIR compute_pain_vector /
    compute_auc unchanged. Verified in analyze() to reproduce PA.kfold_curve exactly."""
    uniq = sorted(set(sets.tolist()))
    kf = PA.KFold(n_splits=PA.N_FOLDS, shuffle=True, random_state=PA.RANDOM_SEED)
    splits = []
    for tr, te in kf.split(uniq):
        splits.append((np.isin(sets, [uniq[i] for i in tr]),
                       np.isin(sets, [uniq[i] for i in te])))
    out = []
    for L in layers:
        acts = acts_all[:, L, :]
        vals = []
        for trm, tem in splits:
            v = PA.compute_pain_vector(acts[trm], cats[trm], "all_controls")
            a = PA.compute_auc(acts[tem], cats[tem], v)
            if not np.isnan(a):
                vals.append(a)
        out.append(float(np.mean(vals)) if vals else np.nan)
    return np.array(out)


def curve_random_direction(PA, acts_all, cats, sets, layers, rng, n_draws):
    """Random-direction arm. The direction is drawn from N(0, I) with no reference to the
    labels, then norm-matched to the fitted pain vector of that fold (cosmetic: compute_auc
    normalises, so the AUC is scale-free -- the match is for the declared arm spec, not maths).
    Scored through PA.compute_auc, the identical path. Declared null: 0.5.
    Returns [n_draws, len(layers)] of fold-averaged AUCs."""
    uniq = sorted(set(sets.tolist()))
    kf = PA.KFold(n_splits=PA.N_FOLDS, shuffle=True, random_state=PA.RANDOM_SEED)
    splits = [(np.isin(sets, [uniq[i] for i in tr]), np.isin(sets, [uniq[i] for i in te]))
              for tr, te in kf.split(uniq)]
    dim = acts_all.shape[2]
    res = np.full((n_draws, len(layers)), np.nan)        # redrawn per fold, like the refit
    res_fixed = np.full((n_draws, len(layers)), np.nan)  # one direction per draw, all folds
    for li, L in enumerate(layers):
        acts = acts_all[:, L, :]
        norms = []
        for trm, _ in splits:
            v = PA.compute_pain_vector(acts[trm], cats[trm], "all_controls")
            norms.append(np.linalg.norm(v))
        for k in range(n_draws):
            vals, vals_f = [], []
            rf = rng.standard_normal(dim)
            rf = rf / (np.linalg.norm(rf) + 1e-12)
            for (trm, tem), nrm in zip(splits, norms):
                r = rng.standard_normal(dim)
                r = r / (np.linalg.norm(r) + 1e-12) * nrm
                a = PA.compute_auc(acts[tem], cats[tem], r)
                if not np.isnan(a):
                    vals.append(a)
                af = PA.compute_auc(acts[tem], cats[tem], rf * nrm)
                if not np.isnan(af):
                    vals_f.append(af)
            res[k, li] = float(np.mean(vals)) if vals else np.nan
            res_fixed[k, li] = float(np.mean(vals_f)) if vals_f else np.nan
    return res, res_fixed


def shuffle_labels_within_set(cats, sets, rng):
    """Permute the category labels WITHIN each sentence set. Each set holds exactly one
    sentence per category, so this preserves the 5-pain/5-control balance of every set and
    hence of every fold, and destroys only the sentence->label association.

    Nothing downstream sees the true labels: the denoising PCA inside compute_pain_vector is
    fitted on whichever sentences the PERMUTED labels call controls, and the folds are built
    from `sets`, which carries no label information (every set is identically composed)."""
    out = np.array(cats, copy=True)
    for s in np.unique(sets):
        m = np.where(sets == s)[0]
        out[m] = np.array(cats)[m][rng.permutation(len(m))]
    return out


def run_combo(ext: str, ds: str, n_rand: int, n_shuf: int) -> None:
    """One (extraction, dataset) cell: treatment curve + both null arms, all 29 layers.

    Run one process per cell (OMP_NUM_THREADS=1) -- the shuffled-label arm refits their
    denoising PCA for every draw, every layer, every fold, which is the whole cost.
    """
    import painaxis_analyze as PA

    d = load(ds)
    A = stack_with_embedding(d, ext)
    cats, sets_ = d["categories"], d["sets"]
    layers = list(range(A.shape[1]))
    t0 = time.time()
    treat = curve_all_controls(PA, A, cats, sets_, layers)
    print(f"{ext} {ds}: treatment {time.time()-t0:.0f}s", flush=True)
    rand, rand_fixed = curve_random_direction(
        PA, A, cats, sets_, layers, np.random.default_rng(20260921), n_rand)
    print(f"{ext} {ds}: random {time.time()-t0:.0f}s", flush=True)
    rng2 = np.random.default_rng(770021)
    shuf = np.full((n_shuf, len(layers)), np.nan)
    same = []
    for k in range(n_shuf):
        cp = shuffle_labels_within_set(cats, sets_, rng2)
        same.append(float((cp == cats).mean()))
        shuf[k] = curve_all_controls(PA, A, cp, sets_, layers)
        if k % 10 == 0:
            print(f"  {ext} {ds} shuffle {k}/{n_shuf} {time.time()-t0:.0f}s", flush=True)
    (OUT / f"nulls_{ext}_{ds}.json").write_text(json.dumps({
        "extraction": ext, "dataset": ds, "n_rand": n_rand, "n_shuf": n_shuf,
        "treat": treat.tolist(), "rand": rand.tolist(), "rand_fixed": rand_fixed.tolist(),
        "shuf": shuf.tolist(),
        "shuffle_frac_same": {"mean": float(np.mean(same)), "max": float(np.max(same))},
    }))
    print(f"COMBO DONE {ext} {ds} {time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------- (B) analysis
def analyze() -> None:
    import csv

    import painaxis_analyze as PA

    layers_idx = None
    data = {n: load(n) for n in SETS}
    n_layers = data["S2_1P"]["final_token"].shape[1]
    layers_idx = list(range(n_layers + 1))          # 0 = emb, 1.. = block 0..n-1
    names = ["emb"] + [str(i) for i in range(n_layers)]
    report = {"model": MODEL, "n_layers": n_layers, "layer_axis": names}

    # ---- 0. degeneracy check: the final-token embedding must be constant ------------------
    deg = {}
    for ds in SETS:
        d = data[ds]
        e = d["embed_final_token"]
        deg[ds] = {
            "n_distinct_last_token_ids": int(len(np.unique(d["last_token_id"]))),
            "last_token_ids": [int(x) for x in np.unique(d["last_token_id"])],
            "max_abs_dev_from_row0": float(np.abs(e - e[0]).max()),
            "std_across_sentences": float(e.std(axis=0).max()),
            "mean_resid_L0_std": float(d["final_token"][:, 0, :].std(axis=0).mean()),
        }
    report["degeneracy_check"] = deg
    print("DEGENERACY CHECK", json.dumps(deg, indent=2), flush=True)

    # ---- 1. self-check: my fold loop == their kfold_curve, on true labels -----------------
    d = data["S2_1P"]
    A = stack_with_embedding(d, "mean")
    mine = curve_all_controls(PA, A, d["categories"], d["sets"], [1, 10, 22])
    theirs = np.array([r["auc_vs_all_controls"] for r in
                       PA.kfold_curve(A, d["categories"], d["sets"], [1, 10, 22])])
    assert np.allclose(mine, theirs, atol=1e-12), (mine, theirs)
    report["selfcheck_fold_loop_matches_kfold_curve"] = True
    print("selfcheck ok:", mine, theirs, flush=True)

    # ---- 2. identity-permutation check: the null machinery on an identity shuffle must
    #         reproduce the treatment curve exactly (catches a null that is not really a null)
    class _Identity:
        def permutation(self, n):
            return np.arange(n)
    ident = shuffle_labels_within_set(d["categories"], d["sets"], _Identity())
    assert (ident == d["categories"]).all()
    ident_curve = curve_all_controls(PA, A, ident, d["sets"], [1, 10, 22])
    assert np.allclose(ident_curve, mine, atol=1e-12)
    report["selfcheck_identity_permutation_reproduces_treatment"] = True

    # ---- 3. treatment, floor, and both nulls, every layer, every set, both extractions ----
    #         computed per (extraction, dataset) by `run_combo` in parallel processes.
    rows, shuf_checks, n_rand, n_shuf = [], [], None, None
    for ext in EXTRACTIONS:
        for ds in SETS:
            p = OUT / f"nulls_{ext}_{ds}.json"
            if not p.exists():
                raise SystemExit(f"missing {p}; run `combo {ext} {ds} <n_rand> <n_shuf>` first")
            c = json.loads(p.read_text())
            n_rand, n_shuf = c["n_rand"], c["n_shuf"]
            shuf_checks.append(c["shuffle_frac_same"])
            treat = np.array(c["treat"])
            rand = np.array(c["rand"])
            rand_fixed = np.array(c["rand_fixed"])
            shuf = np.array(c["shuf"])
            for li, nm in enumerate(names):
                rows.append(dict(
                    extraction=ext, dataset=ds, layer=nm,
                    treatment=round(float(treat[li]), 4),
                    random_mean=round(float(np.mean(rand[:, li])), 4),
                    random_lo=round(float(np.percentile(rand[:, li], 2.5)), 4),
                    random_hi=round(float(np.percentile(rand[:, li], 97.5)), 4),
                    randfix_mean=round(float(np.mean(rand_fixed[:, li])), 4),
                    randfix_lo=round(float(np.percentile(rand_fixed[:, li], 2.5)), 4),
                    randfix_hi=round(float(np.percentile(rand_fixed[:, li], 97.5)), 4),
                    shuffled_mean=round(float(np.mean(shuf[:, li])), 4),
                    shuffled_lo=round(float(np.percentile(shuf[:, li], 2.5)), 4),
                    shuffled_hi=round(float(np.percentile(shuf[:, li], 97.5)), 4),
                ))
    report["n_draws"] = {"random_direction": n_rand, "shuffled_label": n_shuf}
    report["shuffle_fraction_labels_unchanged"] = {
        "mean": float(np.mean([s["mean"] for s in shuf_checks])),
        "max": float(np.max([s["max"] for s in shuf_checks])),
        "expected_chance": 0.1}

    # ---- 4. the floor and the gain -------------------------------------------------------
    floor = {}
    for ds in SETS:
        r = [x for x in rows if x["dataset"] == ds and x["layer"] == "emb"]
        floor[ds] = {"mean_bag": [x["treatment"] for x in r if x["extraction"] == "mean"][0],
                     "final_token_degenerate":
                         [x["treatment"] for x in r if x["extraction"] == "final_token"][0]}
    report["floor"] = floor
    for x in rows:
        x["floor_mean_bag"] = floor[x["dataset"]]["mean_bag"]
        x["gain_over_floor"] = round(x["treatment"] - x["floor_mean_bag"], 4)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "floor_null_curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 5. their chosen layer, by their own procedure, on the block layers only ----------
    chosen = {}
    for ext in EXTRACTIONS:
        s2 = {}
        for nm in names[1:]:
            v = [x["treatment"] for x in rows
                 if x["extraction"] == ext and x["dataset"] in ("S2_1P", "S2_3P") and x["layer"] == nm]
            s2[nm] = float(np.mean(v))
        bl = max(s2, key=s2.get)
        chosen[ext] = {"s2_best_layer": bl, "s2_best_auc": s2[bl]}
    report["chosen_layer"] = chosen

    # ---- 6. reproducibility against the Tier A extraction --------------------------------
    tierA = ROOT / "results/painaxis_tierA/layer_curves.csv"
    if tierA.exists():
        prev = {}
        with open(tierA) as f:
            for r in csv.DictReader(f):
                prev[(r["extraction"], r["dataset"], r["layer"])] = float(r["auc_vs_all_controls"])
        diffs = []
        for x in rows:
            k = (x["extraction"], x["dataset"], x["layer"])
            if k in prev:
                diffs.append(abs(x["treatment"] - prev[k]))
        report["tierA_reproduction"] = {"n_compared": len(diffs),
                                        "max_abs_diff": float(max(diffs)) if diffs else None}
        print("tierA reproduction max|diff| =", report["tierA_reproduction"], flush=True)

    (OUT / "floor_null_summary.json").write_text(json.dumps(report, indent=2))

    # ---- 7. the shared layer-0 static-embedding bag (conscription_instrument_v1 §3) -------
    bag = {}
    for ds in SETS:
        d = data[ds]
        bag[f"{ds}__mean"] = d["embed_mean"]
        bag[f"{ds}__final_token"] = d["embed_final_token"]
        bag[f"{ds}__categories"] = d["categories"]
        bag[f"{ds}__sets"] = d["sets"]
    np.savez_compressed(OUT / "embed_bag.npz", **bag)
    print("ANALYSIS DONE", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "extract":
        extract()
    elif cmd == "combo":
        run_combo(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    else:
        analyze()
