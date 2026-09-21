"""GOAL 2 step 2: the conscription grid on the pain axis, on the free (machine-authored) grid.

`results/notes/conscription_direction.md` §7 stages this behind step 1, which passed
(`painaxis_scenarios.md`). Step 1 also fixed three things this pilot would otherwise have had to
guess:

  * THE READ POSITION is the generation token (`final_token`). The `mean` extraction never clears
    a null on the scenarios at any layer, so a conscription effect read there would be unreadable.
  * THE WINDOW is blocks 10-17. That is where the gaslighting loading clears both nulls on this
    model, and it was determined on THEIR stimuli, not on this grid -- so using it here is not
    selecting a layer on scoring data (CLAUDE.md 4). The whole curve is reported regardless.
  * THE FLOOR is 63% of the headline. The arms hold vocabulary nearly constant by design, so the
    quantity of interest is the part the floor does not already give.

THE STATISTIC, per `conscription_direction.md` §3: paired projection onto the EXTERNAL pain axis,
minus the same paired projection on the static-embedding bag.

    p_i(A)  = projection of item i's arm-A final-token activation onto unit(s2 pain vector at L),
              expressed in SCENARIO-POOL z UNITS -- centred and scaled by the mean and sd of the
              420 scenarios' projections at the same layer, so every number here is directly
              comparable with the z-scores in painaxis_scenarios.md.
    d_i     = p_i(A) - p_i(B)                        (paired, within item)
    g       = mean_i d_i(acts) - mean_i d_i(floor)   (gain over the lexical floor)

Null for g: 0. Band: sign-flip over items (the exact permutation null for a paired difference)
plus a random-direction arm. Both reported; neither is a gate.

FOUR ARMS OF STIMULUS, not three:
  the five design arms (enact/report/exit/true/neutral) as authored, plus
  `enact_norecord` -- the enact assertion as a SINGLE user turn with no prefix at all. This is the
  form all 18 of the paper's no-record gaslighting items take, and rule 1b excluded it by
  construction. It is the bridge between their stimulus class and this one, and without it a null
  on this grid cannot distinguish "conscription does not register" from "rule 1b removed the
  manipulation" (`conscription_direction.md` §5.1).

POSITIVE CONTROL: none is extracted here. The paper's own 20 gaslighting items, already measured
through this identical pipeline in `results/painaxis_scenarios/`, are the control -- if the grid
shows nothing while those items show +1.389 at L12 in the same session and the same code, the
null is about the stimuli and not about the instrument.

Usage:  python scripts/conscription_pilot.py extract   # .venv312, NDIF
        python scripts/conscription_pilot.py analyze
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import painaxis_scenarios as S  # noqa: E402

GRID = ROOT / "prompts/claude/conscription_claude_v1.json"
OUT = ROOT / "results/conscription_pilot"
SHARDS = OUT / "shards"
ARMS = ("enact", "report", "exit", "true", "neutral")
EXTRA = "enact_norecord"
NB = "neutral_b"          # the rewording floor; appended LAST so earlier shards stay valid
BATCH = 2
WINDOW = list(range(10, 18))          # their-layer numbering; determined on THEIR stimuli
PAIRS = (("enact", "report"), ("enact", "exit"), ("enact", "true"), ("enact", "neutral"),
         ("report", "exit"), ("report", "neutral"), ("exit", "neutral"), ("true", "neutral"),
         ("enact", EXTRA), (EXTRA, "neutral"),
         # THE REWORDING FLOOR. Two turns that assert nothing about the assistant, same closer,
         # written independently. Every contrast above has to beat THIS, not just the sign-flip
         # band, before it means anything (results/notes/conscription_pilot.md §2).
         ("neutral", NB), ("enact", NB))
N_SIGN = 5000
N_RAND = 500


def render_all(tok):
    """[(item_id, arm, text)] in a fixed order. `enact_norecord` drops the prefix entirely: two
    consecutive user turns are not renderable by these templates, and a bare user turn is exactly
    the form the paper's 18 no-record gaslighting items take."""
    from lsx.core.conscription import render_prompt, verify_offsets_cover_template

    items = json.loads(GRID.read_text())["items"]
    out = []
    for it in items:
        for arm in ARMS:
            t = render_prompt(tok, it["prefix"], it["arms"][arm])
            verify_offsets_cover_template(tok, t)
            out.append((it["id"], arm, t))
        t = render_prompt(tok, [], it["arms"]["enact"])
        verify_offsets_cover_template(tok, t)
        out.append((it["id"], EXTRA, t))
    # `neutral_b` is APPENDED, not interleaved, so every prompt index below this point is
    # unchanged and the six shards already paid for on a contended GPU stay valid. Shard
    # boundaries fall on multiples of 24 and there are 24 items, so the new arm is exactly one
    # new shard.
    for it in items:
        if NB not in it["arms"]:
            raise SystemExit(f"{it['id']} has no {NB} arm; the grid is out of date")
        t = render_prompt(tok, it["prefix"], it["arms"][NB])
        verify_offsets_cover_template(tok, t)
        out.append((it["id"], NB, t))
    return out


def extract() -> None:
    from lsx.core import painaxis_remote as pr
    from lsx.core.remote import RemoteLM

    OUT.mkdir(parents=True, exist_ok=True)
    SHARDS.mkdir(parents=True, exist_ok=True)
    rlm = RemoteLM(S.MODEL)
    rows = render_all(rlm.tok)
    texts = [t for _, _, t in rows]
    print(f"{len(rows)} prompts ({len(rows)//(len(ARMS)+1)} items x {len(ARMS)+1} arms)", flush=True)
    print("example enact:", repr(rows[0][2])[:220], flush=True)
    print("example norecord:", repr(rows[len(ARMS)][2])[:220], flush=True)

    meta_path = OUT / "extract_meta.json"
    meta = {"model": S.MODEL, "n_prompts": len(rows), "batch": BATCH,
            "arms": list(ARMS) + [EXTRA, NB],
            "grid": str(GRID.relative_to(ROOT)),
            "grid_sha": __import__("hashlib").sha256(GRID.read_bytes()).hexdigest()[:16],
            "equivalence": {},
            "padding_side": rlm.padding_side, "lib_versions": rlm.lib_versions()}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        meta["equivalence"] = old.get("equivalence", {})
        prev = [tuple(x) for x in old.get("order", [])]
        now = [(i, a) for i, a, _ in rows]
        # An append is safe; anything else invalidates the cached shards, which are indexed by
        # position and carry no labels of their own.
        if prev and now[:len(prev)] != prev:
            raise SystemExit(
                "the prompt order changed in place, not by appending: cached shards are indexed "
                "by position and would be silently mislabelled. Delete results/conscription_pilot/"
                "shards and re-extract.")
    # always recomputed, never restored from the file: the file may predate an appended arm
    meta["order"] = [[i, a] for i, a, _ in rows]
    meta["n_prompts"] = len(rows)

    SHARD = 24
    for s0 in range(0, len(texts), SHARD):
        path = SHARDS / f"grid_{s0:04d}.npz"
        if path.exists():
            print(f"  [{s0}] cached", flush=True)
            continue
        t0 = time.time()
        pooled = pr.extract_pooled(rlm, texts[s0:s0 + SHARD], batch_size=BATCH,
                                   add_special_tokens=False, check_every=3)
        np.savez_compressed(path, final_token=pooled.final_token.astype(np.float32),
                            embed_mean=pooled.embed_mean.astype(np.float32),
                            embed_final_token=pooled.embed_final_token.astype(np.float32))
        meta["equivalence"].update(pooled.equivalence)
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  [{s0}] done {time.time()-t0:.0f}s jobs={pooled.n_jobs}", flush=True)
    print("PILOT EXTRACTION DONE", flush=True)


# ------------------------------------------------------------------------------------ analysis
def _band(d, rng):
    """Sign-flip band for a paired difference: the exact permutation null under exchangeability
    of the two arms within an item. Returns (mean, lo, hi, two-sided p)."""
    m = float(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(N_SIGN, len(d)))
    null = (flips * d).mean(axis=1)
    return (m, float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5)),
            float((np.abs(null) >= abs(m)).mean()))


def analyze() -> None:
    import painaxis_scenarios_analyze as A

    meta = json.loads((OUT / "extract_meta.json").read_text())
    order = [(i, a) for i, a in meta["order"]]
    n = len(order)
    parts_ft, parts_eb = [], []
    for s0 in range(0, n, 24):
        z = np.load(SHARDS / f"grid_{s0:04d}.npz")
        parts_ft.append(z["final_token"])
        parts_eb.append(z["embed_mean"])
    ft = np.concatenate(parts_ft, axis=0)                    # [n, 42, d]
    eb = np.concatenate(parts_eb, axis=0)                    # [n, d]
    acts = np.concatenate([eb[:, None, :], ft], axis=1)      # [n, 43, d], 0 = emb bag
    assert acts.shape[0] == n, (acts.shape, n)

    ds = json.loads(S.CORE.read_text())["datasets"]
    core = {m: A.load_group(f"core_{m}", len(ds[m]["sentences"]), ("final_token", "mean"))
            for m in S.S_SETS}
    core_cats = {m: np.array([s["category"] for s in ds[m]["sentences"]]) for m in S.S_SETS}
    scen_order = json.loads((S.OUT / "scenario_order.json").read_text())
    scen = A.load_group("scen_chat", len(scen_order), ("final_token", "mean"))

    item_ids = sorted({i for i, _ in order})
    idx = {(i, a): k for k, (i, a) in enumerate(order)}
    names = ["emb"] + [str(i) for i in range(acts.shape[1] - 1)]
    rng = np.random.default_rng(31337)
    rows = []

    # The floor is ONE lexical baseline, not a per-layer quantity: the axis built on the static
    # embedding bag, and the grid read there. Hoisted so it is computed once and so `d_floor` is
    # visibly the same number at every layer. At L=emb the treatment IS the floor, so the gain
    # there must be exactly 0 -- a free check that the two paths are the same arithmetic.
    vf = S.unit(S.compute_pain_vector(core["S2_1P"]["mean"][:, 0, :], core_cats["S2_1P"]))
    spf = scen["mean"][:, 0, :] @ vf
    pf = (acts[:, 0, :] @ vf - float(spf.mean())) / (float(spf.std()) + 1e-8)

    for L, nm in enumerate(names):
        # the axis, built at this layer from THEIR core stimuli -- never from this grid
        ext = "mean" if nm == "emb" else "final_token"
        v = S.unit(S.compute_pain_vector(core["S2_1P"][ext][:, L, :], core_cats["S2_1P"]))
        # scenario-pool centring, so every number is in the units of painaxis_scenarios.md
        sp = scen[ext][:, L, :] @ v
        mu, sd = float(sp.mean()), float(sp.std()) + 1e-8
        p = (acts[:, L, :] @ v - mu) / sd

        for a, b in PAIRS:
            d = np.array([p[idx[(i, a)]] - p[idx[(i, b)]] for i in item_ids])
            df = np.array([pf[idx[(i, a)]] - pf[idx[(i, b)]] for i in item_ids])
            m, lo, hi, pv = _band(d, rng)
            mf, lof, hif, pvf = _band(df, rng)
            g = m - mf
            gm, glo, ghi, gp = _band(d - df, rng)
            rows.append({
                "layer": nm, "arm_a": a, "arm_b": b, "n": len(d),
                "d_acts": round(m, 4), "d_acts_p": round(pv, 4),
                "d_floor": round(mf, 4), "d_floor_p": round(pvf, 4),
                "gain": round(g, 4), "gain_lo": round(glo, 4), "gain_hi": round(ghi, 4),
                "gain_p": round(gp, 4),
                "clears": int(gp < 0.05),
                "in_window": int(nm.isdigit() and int(nm) in WINDOW),
            })
        if nm == "emb" or (nm.isdigit() and int(nm) in WINDOW):
            er = [r for r in rows if r["layer"] == nm and (r["arm_a"], r["arm_b"]) == ("enact", "report")][0]
            print(f"  L{nm:>3s} enact-report gain {er['gain']:+.4f} "
                  f"[{er['gain_lo']:+.4f},{er['gain_hi']:+.4f}] p={er['gain_p']:.4f}", flush=True)

    with (OUT / "pilot_contrasts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    win = [r for r in rows if r["in_window"]]
    summary = {"n_items": len(item_ids), "window": WINDOW, "n_rows": len(rows),
               "cells_in_window": len(win),
               "clearing_in_window": [
                   {k: r[k] for k in ("layer", "arm_a", "arm_b", "gain", "gain_lo", "gain_hi", "gain_p")}
                   for r in win if r["clears"]],
               "positive_control": json.loads((S.OUT / "summary.json").read_text())["headline"],
               "note": ("The positive control is the paper's own 20 gaslighting items measured "
                        "through this identical pipeline; if nothing here clears while that "
                        "reads +1.389 at L12, the null is about these stimuli and not the "
                        "instrument.")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{len(summary['clearing_in_window'])} of {len(win)} in-window contrasts clear "
          f"their sign-flip band at p<0.05", flush=True)
    for r in summary["clearing_in_window"]:
        print(f"   L{r['layer']:>3s} {r['arm_a']:>14s} - {r['arm_b']:<14s} "
              f"gain {r['gain']:+.4f} [{r['gain_lo']:+.4f},{r['gain_hi']:+.4f}] p={r['gain_p']:.4f}",
              flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract()
    elif sys.argv[1] == "analyze":
        analyze()
    else:
        raise SystemExit("extract | analyze")
