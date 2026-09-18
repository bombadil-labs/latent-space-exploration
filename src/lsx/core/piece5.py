"""Piece 5: close phase 1's gate, or say which target cannot be closed.

Piece 4 ended with §1B passing 12 of 12 and §1A failing on **reporting**: every instrument existed,
every number that could be recomputed reproduced to the logged decimals, and five targets were
refused for how their batteries were run or summarised. This module re-runs those batteries.

    1. h8's three single-factor lenses -- `MissingArm('permutation')`. The arm now exists
       (`reproduce.permuted_level_directions`).
    2. h29's 3x re-imposed generation -- one arm. The three-arm battery is
       `lsx.core.rerun_h29`; this module grades it.
    3. h39's Gemma clock -- `discrimination` exists, the stacks do not.
    4. h16 -- reproduced by piece 4; confirmed here, with the 2.21 / 2.1692 aggregate settled.
    5. h8 composed -- gain over CHANCE where §6 requires gain over a MEASURED floor on a flagged
       grid. The floor is measured by `reproduce.h8_lexical_floor`.

Run:  .venv/bin/python -m lsx.core.piece5 --parts h8,h16       (local, CPU)
      .venv/bin/python -m lsx.core.piece5 --parts h29,h39      (grades what the remote run wrote)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from . import instruments, ledger, registry
from .checks import CoreError
from .reproduce import (DEFERRED, FAILED, LOCAL_TOLERANCE, REFUSED, REMOTE_TOLERANCE, REPRODUCED,
                        Row, data, table)
from . import reproduce as R
from .types import Arm, Floor, Measured, Selection

REPO = pathlib.Path(__file__).resolve().parents[3]
T0 = time.time()


def log(m: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def publish(led, claim, note, reproduces, logged) -> str | None:
    if claim is None:
        return "no claim"
    refusal = ledger.would_refuse(claim)
    if refusal:
        return refusal
    led.append(claim, note=note, reproduces=reproduces, logged=logged)
    return None


# ================================================================================================
# 1 + 5: h8 -- the permutation arm, and the measured lexical floor
# ================================================================================================
def h8_rows(res: dict, lex: dict, led) -> tuple[list[Row], dict]:
    rows: list[Row] = []
    summary: dict = {"lexical_floor": {"composed": lex["composed_mean"],
                                       "lenses": lex["lens_mean"]}}
    claims, _ = R.h8_claims(res, lexical=lex)
    V = res["n_variants"]

    comp = claims[0]
    ref = publish(led, comp, "h8 composed, re-run through the core against the MEASURED lexical "
                             "floor (piece 5)",
                  "h8 composed 2.81/18", {"composed": 2.81, "no_patch": 9.50,
                                          "piece4_floor": 9.50, "piece5_floor": lex["composed_mean"]})
    summary["composed"] = {"treatment": comp.treatment.value,
                           "reported_gain_over_measured_floor": comp.reported_value,
                           "floor": float(comp.floor.stimulus),
                           "arms": {k: {"value": a.value, "null": a.expected_null,
                                        "tol": a.resolved_tolerance, "off_null": a.off_null}
                                    for k, a in comp.arms.items()},
                           "refusal": ref, "id": comp.id, "render": comp.render()}
    rows.append(Row(
        target="h8 three-factor battery, composed", source="h8 (re-verified h39)",
        logged="2.81/18, no-patch 9.50",
        reproduced=f"{comp.treatment.value:.4f}/{V}; gain over the MEASURED lexical floor "
                   f"{comp.reported_value:+.4f} (floor {comp.floor.stimulus:.4f}/{V})",
        tolerance=f"+-{LOCAL_TOLERANCE} (local, §1A); arm band "
                  f"+-{registry.arm_tolerance('composition', comp.treatment.n, {'levels': (3, 3, 2)}):.4f}",
        verdict=REPRODUCED if abs(comp.treatment.value - 2.81) <= LOCAL_TOLERANCE else FAILED,
        detail=("PUBLISHED" if ref is None else f"ledger: {ref}")
               + "; §6 now met with a measured floor, and the floor is the finding: "
                 f"a bag-of-tokens predictor reaches {lex['composed_mean']:.3f}/{V} on the same "
                 f"candidates, so the gain is {comp.reported_value:+.4f} of a rank",
        claim_id=comp.id))

    logged_lens = {"era": 1.25, "voice": 1.24, "tense": 1.03}
    for claim in claims[1:]:
        name = claim.treatment.label.split()[1]
        k = int(claim.treatment.label.split("/")[1].split()[0])
        b = res["B"][name]
        t = float(np.mean(b["factor"]))
        ref = publish(led, claim,
                      f"h8 {name} lens, with the permutation arm and the measured lexical floor "
                      "(piece 5)",
                      f"h8 {name} lens {logged_lens[name]}/3",
                      {"lens": logged_lens[name], "piece5_floor": float(claim.floor.stimulus)})
        summary.setdefault("lenses", {})[name] = {
            "treatment": t, "candidates": k,
            "reported_gain_over_measured_floor": claim.reported_value,
            "floor": float(claim.floor.stimulus),
            "arms": {kk: {"value": a.value, "null": a.expected_null,
                          "tol": a.resolved_tolerance, "off_null": a.off_null}
                     for kk, a in claim.arms.items()},
            "refusal": ref, "id": claim.id, "render": claim.render()}
        off = [kk for kk, a in claim.arms.items() if a.off_null]
        rows.append(Row(
            target=f"h8 {name} lens", source="h8",
            logged=f"{logged_lens[name]}/3",
            reproduced=f"{t:.4f}/{k}; gain over the measured lexical floor "
                       f"{claim.reported_value:+.4f} (floor {claim.floor.stimulus:.4f}/{k})",
            tolerance=f"+-{LOCAL_TOLERANCE}; arm band "
                      f"+-{registry.arm_tolerance('selector', len(b['perm']), {'n_candidates': k}):.4f}",
            verdict=REPRODUCED if abs(t - logged_lens[name]) <= LOCAL_TOLERANCE else FAILED,
            detail=("PUBLISHED" if ref is None else f"ledger: {ref}")
                   + f"; arms random {np.mean(b['rand']):.3f} / no_patch {np.mean(b['none']):.3f} "
                     f"/ permutation {np.mean(b['perm']):.3f} vs null {(k + 1) / 2:.2f}"
                   + (f"; OFF NULL: {off}" if off else "; no arm off its null"),
            claim_id=claim.id))
    return rows, summary


# ================================================================================================
# 4: h16 -- confirm it still publishes, and settle which aggregate the record uses
# ================================================================================================
def h16_rows(res: dict, led) -> tuple[list[Row], dict]:
    claim, s = R.h16_claim(res)
    ref = publish(led, claim,
                  "h16 as the full layer curve, re-confirmed; the canonical aggregate is the "
                  "step-2 curve mean (piece 5)",
                  "h16 relation selector, §1A's restatement",
                  {"step2_mean_canonical": 2.1692, "step4_mean_as_published_in_RESULTS": 2.21,
                   "peak_layer_refused": 16, "role_identity_retained": 1.37})
    s["refusal_at_ledger"] = ref
    s["canonical_aggregate"] = "step-2 mean over the 15-layer curve"
    delta_step4 = abs(s["curve_step4_mean"] - 2.21)
    off = [k for k, a in claim.arms.items() if a.off_null] if claim is not None else []
    rows = [R.h16_as_logged(),
            Row(target="h16 relation selector, as the full layer curve (§1A's restatement)",
                source="h16",
                logged="§1A quotes 2.21; the repo's own stage-3 JSON is 2.1692",
                reproduced=f"{s['treatment']:.4f} over 15 layers (step 2) = the canonical value; "
                           f"{s['curve_step4_mean']:.4f} over the 8 step-4 layers = what "
                           "RESULTS.md published as 2.21",
                tolerance=f"+-{LOCAL_TOLERANCE} (local, §1A)",
                verdict=REPRODUCED if delta_step4 <= LOCAL_TOLERANCE else FAILED,
                detail=("PUBLISHED" if ref is None else f"ledger: {ref}")
                       + f"; graded against the aggregate it IS: |delta| {delta_step4:.4f} on the "
                         "step-4 mean against 2.21. The two numbers are one curve at two sweep "
                         "steps; piece 5 makes the step-2 mean canonical and records 2.21 as the "
                         "step-4 subsample"
                       + (f"; OFF NULL: {off}" if off else "; no arm off its null"),
                claim_id="" if claim is None else claim.id)]
    return rows, s


# ================================================================================================
# 2: h29 -- the three-arm battery, graded through `top1_accuracy`
# ================================================================================================
# Declared BEFORE any score in this file is computed, from h27's own published base/rand rows
# (`results/recompose_gen_gemma9b.json`, n=36 each, scale 1.0). See `lsx.core.rerun_h29`'s module
# docstring for why the registry's 1/k is the wrong null for these two arms and this is the right
# one.
def h27_control_rates() -> dict:
    """h27's base and rand arms, re-expressed as the statistic h29 reports.

    h27 logged `era_stayed` and `era_other` for arms with no target label. The statistic here is
    'era reads as the designated target e2', and for an arm BLIND to e2, with both non-e1 targets
    present, that is exactly `era_other / 2`. This is arithmetic on h27's published numbers, not a
    new measurement.
    """
    blob = json.loads(data("recompose_gen_gemma9b.json").read_text())
    out = {}
    for cond in ("base", "rand"):
        s = blob["summary"][cond]
        out[cond] = {"n": s["n"], "era_stayed": s["era_stayed"], "era_other": s["era_other"],
                     "era_as_target": s["era_other"] / 2.0,
                     "theme_kept": s["theme_kept"]}
    return out


def h29_rows(led, path: str = "results/h29_arms_reimpose3.0.json") -> tuple[list[Row], dict]:
    p = REPO / path
    if not p.exists():
        return [Row(target="h29 era shift in generation, 3x re-imposed", source="h29",
                    logged="0.84 / 0.91 / 0.53, lex 0.30, n=55/72", reproduced="not run",
                    tolerance=f"+-{REMOTE_TOLERANCE:.4f} remote", verdict=DEFERRED,
                    detail="the three-arm battery has not been run")], {}
    blob = json.loads(p.read_text())
    rows_all = blob["rows"]
    ctl = blob.get("controls", {})
    prior = h27_control_rates()

    def arm(cond):
        return [r for r in rows_all if r["cond"] == cond and "era_read" in r]

    shift, rand, base = arm("shift"), arm("rand"), arm("base")
    # the no-patch arm is scored against BOTH designated targets of its passage, because the
    # continuation is the same for both (greedy, unpatched) and the treatment's items are (passage,
    # target) pairs. Its INDEPENDENT unit is still the 36 generations, so its tolerance is taken at
    # that n and not at 72 -- the h16 lesson about an n that counts repetitions.
    e = [x for x in blob["meta"].get("eras", [])] or ["medieval", "1920s", "farfuture"]
    base_pairs, base_units = [], len(base)
    for r in base:
        for e2 in [x for x in e if x != r["e1"]]:
            base_pairs.append(float(r["era_read"] == e2))

    treat = np.array([float(r["era_read"] == r["e2"]) for r in shift])
    a_rand = np.array([float(r["era_read"] == r["e2"]) for r in rand])
    a_base = np.array(base_pairs)
    leaves = {"shift": float(np.mean([r["era_read"] != r["e1"] for r in shift])),
              "rand": float(np.mean([r["era_read"] != r["e1"] for r in rand])),
              "base": float(np.mean([r["era_read"] != r["e1"] for r in base]))}
    theme = {k: float(np.mean([r["theme_read"] == r["t"] for r in v]))
             for k, v in (("shift", shift), ("rand", rand), ("base", base))}
    lex_rows = [r for r in shift if r["lex_era"] not in (None, "none", "-")]
    lexical = float(np.mean([r["lex_era"] == r["e2"] for r in lex_rows])) if lex_rows else 0.0

    inst = instruments.build("top1_accuracy", n=400, d=64, n_candidates=3)
    null_base = prior["base"]["era_as_target"]
    prov = dict(ctl.get("asserted_stack", {}).get(str(blob["meta"]["read_layer"]), {})
                .get("provenance", {}))
    prov.update({"direction_held_out": "scene (leave-one-scene-out)",
                 "template": blob["meta"]["prompt_format"],
                 "patch_layer": blob["meta"]["patch_layer"],
                 "read_layer": blob["meta"]["read_layer"],
                 "scale": blob["meta"]["scale"]})

    detail, claim, ref = [], None, None
    try:
        claim = inst.claim(
            treatment=Measured(treat, label="h29 era reads as target, 3x re-imposed @ scale 3.0"),
            arms={"random": Arm(a_rand, expected_null=null_base,
                                tolerance=inst.tolerance(len(a_rand)),
                                justification="a matched-norm random direction re-imposed at the "
                                              "same scale. Declared at the NO-PATCH base rate "
                                              f"({null_base:.4f}, h27's own base arm), not at 1/k: "
                                              "a direction with no era content should leave the "
                                              "continuation's era where the unpatched model puts "
                                              "it, and the unpatched model does not put it at "
                                              "chance. h27 measured this arm at 0.156 at scale 1.0"),
                  "no_patch": Arm(a_base, expected_null=null_base,
                                  tolerance=inst.tolerance(base_units),
                                  justification="no patch at all, each continuation scored against "
                                                "both designated targets. Declared at h27's own "
                                                f"published base rate {null_base:.4f} (era_other "
                                                "0.2222 / 2), an INDEPENDENT prior measurement of "
                                                "the identical condition -- base has no scale. "
                                                f"Tolerance at n={base_units}, the number of "
                                                "independent generations, not at the 72 pairs")},
            floor=Floor(stimulus=null_base, estimator=1.0 / 3.0),
            selection=Selection(axis=None,
                                rule=f"pre-registered scale {blob['meta']['scale']}, patch "
                                     f"{blob['meta']['patch_layer']} / read "
                                     f"{blob['meta']['read_layer']}; no sweep"),
            provenance=prov, stage="h29", report_as="gain_over_floor",
            notes=[f"lexical era-word check on the treatment arm: {lexical:.4f} on "
                   f"{len(lex_rows)} of {len(shift)} continuations with any era word.",
                   "the patch vectors were computed from the cached direction npz; the same grid "
                   "re-extracted through `remote.build_remote_stack` reproduces every shift vector "
                   f"at cosine >= "
                   f"{ctl.get('asserted_stack', {}).get(str(blob['meta']['patch_layer']), {}).get('min_cos_shift_vector_vs_cached_npz')}"
                   ", and the readout directions ARE that stack's.",
                   "structural check on target-blindness: for an arm that cannot see e2, "
                   "era-as-target must equal leaves-e1 / 2. no_patch "
                   f"{np.mean(a_base):.4f} vs {leaves['base'] / 2:.4f}; random "
                   f"{np.mean(a_rand):.4f} vs {leaves['rand'] / 2:.4f}."])
        ref = publish(led, claim,
                      "h29 3x re-imposed era shift, re-run with the random and no-patch arms it "
                      "never had at this scale (piece 5)",
                      "h29 era->target 0.84 at scale 3.0",
                      {"era_target": 0.84, "leaves_e1": 0.91, "theme_kept": 0.53, "lex": 0.30})
        detail.append("PUBLISHED" if ref is None else f"ledger: {ref}")
    except CoreError as ex:
        detail.append(f"{type(ex).__name__} -- {str(ex).splitlines()[0][:160]}")

    off = [k for k, a in claim.arms.items() if a.off_null] if claim else []
    detail.append(f"arms: random {a_rand.mean():.4f}, no_patch {a_base.mean():.4f}, both declared "
                  f"at {null_base:.4f} +-{inst.tolerance(len(a_rand)):.4f}"
                  + (f"; OFF NULL: {off}" if off else "; neither off its null"))
    summary = {
        "n": {"shift": len(shift), "rand": len(rand), "base": len(base),
              "attempted": {c: sum(1 for r in rows_all if r["cond"] == c)
                            for c in ("shift", "rand", "base")}},
        "era_target": {"shift": float(treat.mean()), "rand": float(a_rand.mean()),
                       "no_patch": float(a_base.mean())},
        "leaves_e1": leaves, "theme_kept": theme, "lexical": lexical,
        "declared_null": null_base, "h27_prior": prior,
        "arm_tolerance": float(inst.tolerance(len(treat))),
        "controls": ctl,
        "claim": None if claim is None else {"id": claim.id, "reported": claim.reported_value,
                                             "render": claim.render(), "refusal": ref},
    }
    rows = [Row(target="h29 era shift in generation, 3x re-imposed", source="h29",
                logged="0.84 era->target / 0.91 leaves e1 / 0.53 theme kept, lex 0.30, n=55/72",
                reproduced=f"{treat.mean():.4f} / {leaves['shift']:.4f} / {theme['shift']:.4f}, "
                           f"lex {lexical:.2f} (n={len(shift)}/"
                           f"{summary['n']['attempted']['shift']})",
                tolerance=f"+-{REMOTE_TOLERANCE:.4f} remote (measured, piece 3); arm band "
                          f"+-{inst.tolerance(len(treat)):.4f} at n={len(treat)}",
                verdict=(REPRODUCED if abs(treat.mean() - 0.84) <= REMOTE_TOLERANCE else FAILED)
                        if claim is not None and ref is None else REFUSED,
                detail="; ".join(detail),
                claim_id="" if claim is None else claim.id, extra=summary)]
    return rows, summary


# ================================================================================================
# 3: h39
# ================================================================================================
def h39_rows() -> tuple[list[Row], dict]:
    row = R.h39_as_logged()
    return [row], row.extra


# ================================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="h8,h16")
    ap.add_argument("--out", default="results/repro_summary_p5.json")
    a = ap.parse_args()
    parts = [x.strip() for x in a.parts.split(",") if x.strip()]

    led = ledger.Ledger()
    rows: list[Row] = []
    summary: dict = {}

    lm = None
    if "h8" in parts or "h16" in parts:
        from lsx import LM as _LM
        lm = _LM.from_pretrained("Qwen/Qwen2.5-1.5B")
        log("model loaded")

    if "h8" in parts:
        log("h8: three-factor battery WITH the permutation arm")
        res8 = R.h8(lm, progress=log)
        np.savez(REPO / "results/repro_p5_h8_arrays.npz",
                 composed=np.asarray(res8["composed"]),
                 composed_none=np.asarray(res8["composed_none"]),
                 **{f"B_{n}_{c}": np.asarray(v) for n, d in res8["B"].items()
                    for c, v in d.items()})
        lex = R.h8_lexical_floor()
        log(f"lexical floor: composed {lex['composed_mean']:.4f}, lenses {lex['lens_mean']}")
        r, s = h8_rows(res8, lex, led)
        rows += r
        summary["h8"] = s

    if "h16" in parts:
        log("h16: the layer curve, recomputed")
        res16 = R.h16(lm, layers=tuple(range(0, 29, 2)), progress=log)
        r, s = h16_rows(res16, led)
        rows += r
        summary["h16"] = s

    if "h29" in parts:
        log("h29: grading the three-arm battery")
        r, s = h29_rows(led)
        rows += r
        summary["h29"] = s

    if "h39" in parts:
        r, s = h39_rows()
        rows += r
        summary["h39"] = s

    tbl = table(rows)
    print(tbl, flush=True)
    counts: dict = {}
    for r in rows:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    log(f"verdicts: {counts}")
    out = REPO / a.out
    prev = json.loads(out.read_text()) if out.exists() else {"rows": [], "summary": {}}
    merged_rows = {r["target"]: r for r in prev.get("rows", [])}
    for r in rows:
        merged_rows[r.target] = r.__dict__
    prev["summary"].update(summary)
    out.write_text(json.dumps({"rows": list(merged_rows.values()),
                               "summary": prev["summary"],
                               "table": table(rows),
                               "verdict_counts": counts,
                               "wall_seconds": round(time.time() - T0, 1)},
                              indent=1, default=str))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
