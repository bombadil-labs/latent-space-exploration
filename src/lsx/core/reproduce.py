"""§1A: the reproduction suite. This is where the core either proves itself or does not.

Each target is a row of spec §1A. A target is graded in one of four ways, and three of them are
results:

  * **reproduced** -- re-derived through the core and inside the per-target tolerance;
  * **REFUSED** -- the core will not publish the number as logged. Two §1A rows are restated in the
    spec precisely because of this, and a refusal here is the acceptance test passing, not failing;
  * **deferred** -- the target's stack is not cached (spec §11.3 says defer rather than re-extract)
    or its statistic has no built instrument (§8's `discrimination`, §11.4's deferrals);
  * **FAILED** -- re-derived and outside tolerance. The rule, pre-registered in §1A: the target is
    then withdrawn from the writeup. The tolerance is not loosened.

Tolerances are per target and measured, never one number for all:

  * local: ±0.02, from §1A (h39's re-run reproduced h8 to two decimals);
  * remote: measured in piece 3 by re-running one remote target twice; see `REMOTE_TOLERANCE`.

Nothing here reads a number out of a logged JSON and calls it reproduced. Where a target is graded
from logged per-item values rather than re-derived, the row says so and the verdict is `deferred`,
not `reproduced`.
"""
from __future__ import annotations

import itertools
import json
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import extract as ex
from . import instruments, ledger, registry
from .checks import CoreError
from .types import Arm, Claim, Direction, Floor, Grid, Item, Measured, Selection

REPO = pathlib.Path(__file__).resolve().parents[3]
# Stacks and result JSONs live in the MAIN checkout: `.npz` is gitignored and a worktree does not
# share untracked files. Both paths are tried, the worktree's first.
DATA_DIRS = [REPO / "results", pathlib.Path("/home/user/latent-space-exploration/results")]
PROMPT_DIRS = [REPO / "prompts", pathlib.Path("/home/user/latent-space-exploration/prompts")]

LOCAL_TOLERANCE = 0.02

# MEASURED in piece 3 (results/remote_tolerance.json, spec §1A). h29's re-imposed era shift at
# scale 3.0 on Gemma-2-9B-it was run twice, end to end, nothing changed. The two runs are
# **identical**: the same 55 of 72 generations survived, every continuation matched character for
# character, and not one era readout flipped. Measured spread: **0.000**.
#
# A tolerance of exactly zero is not usable -- it would refuse a re-run that differs by a single
# item -- so the number below is the statistic's own RESOLUTION, one item in 55, which is the
# smallest difference a fraction over 55 surviving generations can express. That is a consequence
# of the measurement rather than a choice: the spread is smaller than the resolution, so the
# resolution is the binding constraint.
#
# What it does NOT bound: both runs hit the same pinned deployment within one session. Cross-session
# or cross-deployment variation (a redeployment, different bf16 kernels) is unmeasured, and the next
# remote grade should re-measure rather than inherit this number.
REMOTE_TOLERANCE: float | None = 1.0 / 55.0        # 0.0182
REMOTE_SPREAD_MEASURED = 0.0
REMOTE_TOLERANCE_BASIS = ("two identical end-to-end re-runs of h29 reimpose@3.0 on NDIF "
                          "(spread 0.000 on era-target, leaves-e1 and theme-kept, 0 of 55 items "
                          "flipped, continuations character-identical); tolerance = 1/55, the "
                          "resolution of the statistic")


def data(name: str) -> pathlib.Path:
    for d in DATA_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} in none of {[str(d) for d in DATA_DIRS]}")


def prompt(name: str) -> pathlib.Path:
    for d in PROMPT_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


REPRODUCED, REFUSED, DEFERRED, FAILED = "reproduced", "REFUSED", "deferred", "FAILED"


@dataclass
class Row:
    """One §1A target's verdict."""
    target: str
    source: str
    logged: str
    reproduced: str
    tolerance: str
    verdict: str
    detail: str = ""
    claim_id: str = ""
    extra: dict = field(default_factory=dict)

    def render(self) -> str:
        return (f"| {self.target} | {self.source} | {self.logged} | {self.reproduced} | "
                f"{self.tolerance} | **{self.verdict}** | {self.detail} |")


def table(rows: Sequence[Row]) -> str:
    head = ("| target | source | logged | reproduced | tolerance | verdict | note |\n"
            "|---|---|---|---|---|---|---|")
    return "\n".join([head] + [r.render() for r in rows])


# ================================================================================================
# grids
# ================================================================================================
def factor_grid(path: pathlib.Path, *, leak_check: bool = True) -> tuple[Grid, dict]:
    """`prompts/narrative_factors_v*.json` -> Grid. One item per span, text `<lead> <span>`, which
    is exactly the string `scripts/extract_factors.py` extracted and `stage6_factors.py` scores."""
    g = json.loads(path.read_text())
    lead, order = g["lead"], g["key_order"][1:]
    items = []
    for key, span in g["spans"].items():
        parts = key.split("/")
        text = f"{lead} {span}"
        items.append(Item(text=text, factors={"scene": parts[0],
                                              **{f: v for f, v in zip(order, parts[1:])}},
                          spans={"span": (len(lead) + 1, len(text))}))
    return Grid(items=items, name=path.stem, leak_check=leak_check), g


# ================================================================================================
# directions, with the held-out witness the core requires
# ================================================================================================
def level_directions(stack, grid: Grid, layer: int, factors: Sequence[str], held_out_scene: str
                     ) -> tuple[dict, np.ndarray, dict]:
    """h8's `dirs()`: the grand mean over the training scenes, and each factor level's mean minus
    it -- re-expressed so that every direction carries a `fit_witness`.

    `Direction` has demanded a declared held-out axis since piece 1 and verified it since piece 2;
    the point of routing h8's directions through it is that the declaration is now checked against
    what the fit used, rather than believed.
    """
    vecs = stack.vectors("span", layer)
    train = [i for i, it in enumerate(grid.items) if it.factors["scene"] != held_out_scene]
    used = sorted({grid.items[i].factors["scene"] for i in train})
    mu = vecs[train].mean(axis=0)
    out = {}
    for f in factors:
        out[f] = {}
        for lvl in grid.levels(f):
            idx = [i for i in train if grid.items[i].factors[f] == lvl]
            out[f][lvl] = Direction(
                vecs={layer: vecs[idx].mean(axis=0) - mu},
                held_out={"axis": "scene", "unseen": {held_out_scene}},
                fit_witness={"axis": "scene", "fit_values": used},
                provenance=dict(stack.provenance, factor=f, level=lvl))
    return out, mu, {"axis": "scene", "unseen": {held_out_scene}, "fit_values": used}


# ================================================================================================
# target: h8, the three-factor battery on Qwen2.5-1.5B
# ================================================================================================
def h8(lm, *, layer: int = 14, scale: float = 1.0, scenes: Sequence[str] | None = None,
       progress: Callable[[str], None] = lambda s: None) -> dict:
    """Re-run h8 through the core: one asserted extraction, directions with a checked hold-out, and
    every candidate scored through `extract.asserted_patched_logprob` so the h34 moved-candidates
    assertion is on a local log-probability selector for the first time.

    Returns per-item ranks for the composed test (`composition`, 18 joint variants, null 9.50) and
    for each single-factor test (`selector`, 3 candidates, null 2.00), plus the no-patch and random
    arms of each.
    """
    from ..model import Patch
    from ..steer import add_vector

    path = prompt("narrative_factors_v2.json")
    grid, g = factor_grid(path)
    F = g["factors"]
    names = list(F)
    S = list(scenes) if scenes else g["scenes"]
    lead, spans = g["lead"], g["spans"]
    combos = list(itertools.product(*[F[n] for n in names]))

    progress("extracting the stack through build_stack (every §7 assertion)")
    stack = ex.build_stack(lm, grid, layers=[layer], batch_size=8)

    rng = np.random.default_rng(0)
    out = {"composed": [], "composed_none": [],
           "B": {n: {"factor": [], "rand": [], "none": []} for n in names},
           "stack": stack, "grid": grid, "n_variants": len(combos)}

    def rank(gd, target, cands):
        vals = [gd[c] for c in cands]
        from .checks import midrank
        return midrank(vals, cands.index(target))

    for s in S:
        D, mu, witness = level_directions(stack, grid, layer, names, s)
        texts = [f" {spans['/'.join([s, *c])]}" for c in combos]
        base = np.array([lm.logprob(lead, t) for t in texts])
        zero = np.zeros_like(D[names[0]][F[names[0]][0]].vec(layer))

        def gains(vec, *, expect_move: bool) -> dict:
            patches = [Patch(layer, add_vector(vec, scale))]
            scores = ex.asserted_patched_logprob(lm, lead, texts, patches,
                                                 base=base if expect_move else None)
            return dict(zip(combos, scores - base))

        gd_none = gains(zero, expect_move=False)
        for i, n in enumerate(names):
            for lvl in F[n]:
                d = D[n][lvl].vec(layer)
                r = rng.normal(size=d.shape)
                r *= np.linalg.norm(d) / np.linalg.norm(r)
                for cond, vec in (("factor", d), ("rand", r)):
                    gd = gains(vec, expect_move=True)
                    for c in combos:
                        if c[i] != lvl:
                            continue
                        cands = [cc for cc in combos
                                 if all(cc[j] == c[j] for j in range(len(names)) if j != i)]
                        out["B"][n][cond].append(rank(gd, c, cands))
                for c in combos:
                    if c[i] != lvl:
                        continue
                    cands = [cc for cc in combos
                             if all(cc[j] == c[j] for j in range(len(names)) if j != i)]
                    out["B"][n]["none"].append(rank(gd_none, c, cands))
        for c in combos:
            vec = sum(D[n][c[i]].vec(layer) for i, n in enumerate(names))
            gd = gains(vec, expect_move=True)
            out["composed"].append(rank(gd, c, combos))
            out["composed_none"].append(rank(gd_none, c, combos))
        progress(f"scene {s} done")
    return out


def h8_claims(res: dict, *, layer: int = 14) -> tuple[list[Claim], list[Row]]:
    """h8's composed test through `composition` against its declared null of 9.50 -- not three
    `selector` calls, which is what piece 2 asked piece 3 to stop doing. The single-factor tests
    stay `selector` claims, one per factor, because that is what they are."""
    rows, claims = [], []
    grid, stack = res["grid"], res["stack"]
    V = res["n_variants"]

    comp = instruments.build("composition", n=400, d=64, levels=(3, 3, 2))
    treat = np.asarray(res["composed"], dtype=float)
    none = np.asarray(res["composed_none"], dtype=float)
    rand = np.asarray(res["B"]["era"]["rand"], dtype=float)   # a matched-norm random direction
    claim = comp.claim(
        treatment=Measured(treat, label=f"h8 composed rank/{V} @L{layer}"),
        arms={"no_patch": none,
              "random": Arm(rand, expected_null=2.0,
                            tolerance=registry.arm_tolerance("selector", len(rand),
                                                             {"n_candidates": 3}),
                            justification="random direction of matched norm, ranked within a "
                                          "factor's 3 candidates, so its null is 2.00")},
        floor=Floor(stimulus=float((V + 1) / 2), estimator=float((V + 1) / 2)),
        selection=Selection(axis=None, rule=f"pre-registered patch layer {layer}; no sweep"),
        provenance=dict(stack.provenance, direction_held_out="scene (leave-one-scene-out)"),
        grid=grid, stage="h8", report_as="gain_over_floor",
        # PIECE 4: this said `report_as="raw"` as piece 3 committed it, and it cannot have been the
        # code that produced piece 3's ledger row. `narrative_factors_v2` is FLAGGED by its own leak
        # report (era, scene, tense and voice all recoverable from a bag of tokens well above their
        # permutation nulls), so §6 forbids a raw score and `Claim` raises `RawScoreOnLeakyGrid` --
        # which it duly did, twenty minutes into the re-run. The published row reports
        # -6.6944, which is 2.8056 - 9.50, i.e. the gain; the committed function could not have
        # produced it. A path with no test on it, in a file whose tests all need a 1.5B forward.
        #
        # What the gain is measured against is worth stating rather than glossing: `Floor.stimulus`
        # here is the joint midpoint 9.50, which is CHANCE and not a measured stimulus floor. A
        # measured one would need the grid's own lexical predictor over 18 joint variants, which
        # h8 never built. So this row reports "gain over chance" under the name gain_over_floor,
        # and that is weaker than §6 intends. Recorded, not fixed, in this piece.
        notes=["report_as=gain_over_floor is required here: narrative_factors_v2 is flagged leaky. "
               "The floor subtracted is the joint midpoint 9.50 (chance), NOT a measured lexical "
               "floor -- h8 never built one, so the gain is over chance and §6 is only half met."])
    claims.append(claim)
    return claims, rows


# ================================================================================================
# the two targets graded first: both are refusals as logged
# ================================================================================================
def h16_as_logged() -> Row:
    """h16's 'peak layer 16' (role_rank 1.73, quoted from a 0/4/10/16/20/24/28 sweep of the same
    held-out data that scores it). §1A restates this target as a full layer curve; the core must
    refuse the peak."""
    sel = instruments.build("selector", n=200, d=32, n_candidates=6)
    curve = {"0": 2.73, "4": 2.20, "10": 1.98, "16": 1.73, "20": 1.86, "24": 2.39, "28": 2.73}
    n = 40
    rng = np.random.default_rng(0)
    peak = np.full(n, 1.73)
    arms = {"random": np.full(n, 3.49), "no_patch": np.full(n, 3.45),
            "permutation": np.full(n, 3.53)}
    prov = {"grid_hash": "holonic_v2", "model": "Qwen/Qwen2.5-1.5B", "layers": sorted(curve),
            "pooling": "mean", "template": None, "tokenizer_padding": "right",
            "code_version": "scripts/stage3.py (frozen)", "lib_versions": {}}
    try:
        sel.claim(treatment=Measured(peak, label="h16 role_rank at the peak layer"),
                  arms=arms, floor=Floor(stimulus=3.5, estimator=3.5),
                  selection=Selection(axis="layer", rule="peak layer 16 (argmax over the sweep)",
                                      held_out=False),
                  provenance=prov, stage="h16")
        verdict, detail = FAILED, "the core ACCEPTED an argmax on the scoring data"
    except CoreError as e:
        verdict = REFUSED
        detail = f"{type(e).__name__} -- {str(e).splitlines()[0][:120]}"
    return Row(target="h16 relation selector, 'peak layer 16'", source="h16",
               logged="1.73/6 at layer 16 (curve mean 2.21, null 3.5)",
               reproduced="not published", tolerance="n/a -- refused before grading",
               verdict=verdict, detail=detail, extra={"curve": curve})


def h39_as_logged() -> Row:
    """h39's corrected Gemma clock: 0.501 / 0.767 / 2.50 are raw scores on a grid whose leak check
    flags 221 of 240 state spans. §1A says the core must not print them bare, and restates the
    target as **gain over the measured stimulus floor**.

    Piece 3 refused this row for two independent reasons. Piece 4 closes the first and cannot close
    the second:

      1. *`discrimination` was declared and not built*, so no Claim could be graded through it at
         all. It is built now, it passes the six-test battery, and §3 below shows it computing a
         real gain over a real floor from h38's cached per-subject values.
      2. *The grid is flagged leaky, so §6 requires gain over the measured floor* -- and measuring
         that floor needs the Gemma stacks, which are not cached. §11.3 says defer rather than
         re-extract, and 0.767 is a rank correlation over nine per-Δt shared norms with no
         per-subject breakdown in the logged JSON, so there is nothing to grade per item either.

    So the verdict moves from REFUSED-for-two-reasons to **deferred**: the instrument exists, the
    data does not. That is a smaller gap than it was and it is still a gap.
    """
    from .types import LeakReport
    # the leak report h35/h38 measured on this grid, restated as the core's own object
    leak = LeakReport(recoverability={"interval": 0.92}, null_value={"interval": 0.40},
                      position_corr={}, flagged=("interval",),
                      notes=["h32/h35: 221 of 240 v2 state spans restate the interval; h38 measured "
                             "layer-14 discrimination 0.961 -> 0.522 with the phrase removed"])
    logged = json.loads(data("time_translation_gemma_auditfix_measures.json").read_text())
    m1 = logged["m1_m2_decomposition"]["20"]["per_dt"]
    frac_shared = float(np.mean([v["frac_shared"] for v in m1.values()]))
    spearman_logged = float(logged["m3_clock_geometry"]["20"]["spearman_norm_logdt"])
    ratio = float(logged["m7_phrase_control"]["20"]["mean_ratio"])

    inst = instruments.build("discrimination", n=200, d=64, m=9)
    built = inst.calibration.passed
    demo = discrimination_on_h38_cache()

    return Row(target="h39 Gemma clock, corrected", source="h39",
               logged="0.501 shared-variance / 0.767 Spearman / 2.50 phrase-only ratio (raw)",
               reproduced=f"read back from the logged JSON: {frac_shared:.3f} / "
                          f"{spearman_logged:.3f} / {ratio:.3f} -- NOT re-derived",
               tolerance="n/a -- not gradable without the floor",
               verdict=DEFERRED,
               detail=("`discrimination` is BUILT and calibrated (piece 4), so reason 1 of piece "
                       "3's two is closed. The grid is flagged leaky "
                       f"({leak.summary()[:60]}...), so §6 requires gain over the MEASURED floor, "
                       "and the Gemma v2/v3 stacks are not cached (§11.3: defer, do not "
                       "re-extract). The logged JSON carries aggregates only -- nine per-Δt shared "
                       "norms -- so there are no per-item values to grade either. The instrument "
                       "is exercised on h38's cached per-subject values instead (see extra)."),
               extra={"instrument_built": built, "leak": leak.summary(),
                      "logged_reread": {"frac_shared": frac_shared,
                                        "spearman_norm_logdt": spearman_logged,
                                        "phrase_ratio": ratio},
                      "instrument_demo": demo})


def discrimination_on_h38_cache() -> dict:
    """`discrimination` run on real per-subject numbers: h38's clock arms A and D, Qwen2.5-1.5B.

    Not an §1A row -- §1A's discrimination target is h39's Gemma clock -- but the nearest real data
    the instrument can reach, and the point of doing it is that a calibrated instrument which has
    never touched a measurement is only half-checked.

    Arm D is the full state text and arm A is the same text with the interval phrase removed, which
    is a *measured stimulus floor* rather than a declared one: h38 built arm A precisely so the
    floor could be measured rather than argued. The gain is `D - A` per subject and the claim
    reports `gain_over_floor`.

    It is REFUSED, twice, and both refusals are about h38's battery rather than the instrument:
    there is no `shuffled_stimulus` arm in the cached discrimination JSONs (arm B was extracted but
    never run through the discrimination script), and the provenance is a frozen script's output
    rather than a `Stack`.
    """
    out: dict = {"model": "Qwen/Qwen2.5-1.5B", "source": "results/clock_gain_v1_discrim_{A,D}.json"}
    try:
        A = json.loads(data("clock_gain_v1_discrim_A.json").read_text())
        D = json.loads(data("clock_gain_v1_discrim_D.json").read_text())
    except FileNotFoundError as e:
        return {"unavailable": str(e)}
    layers = sorted(set(A) & set(D), key=int)
    subjects = sorted(D[layers[0]]["per_subject"])
    per_layer = {}
    for l in layers:
        treat = np.array([D[l]["per_subject"][s]["shared_spearman"] for s in subjects])
        floor = np.array([A[l]["per_subject"][s]["shared_spearman"] for s in subjects])
        per_layer[l] = {"treatment": float(treat.mean()), "floor": float(floor.mean()),
                        "gain": float((treat - floor).mean())}
    out["per_layer"] = per_layer
    out["subjects"] = len(subjects)

    l = "14"
    treat = np.array([D[l]["per_subject"][s]["shared_spearman"] for s in subjects])
    floor = np.array([A[l]["per_subject"][s]["shared_spearman"] for s in subjects])
    inst = instruments.build("discrimination", n=200, d=64, m=9, floor=float(floor.mean()))
    out["declared_null"] = inst.declared_null
    out["calibration_null"] = inst.calibration_null
    out["calibration_key"] = inst.key
    out["arm_tolerance_at_8_subjects"] = float(inst.tolerance(len(subjects)))
    try:
        claim = inst.claim(
            treatment=Measured(treat, label=f"h38 clock arm D, shared Spearman @L{l}"),
            arms={"floor": Arm(floor, expected_null=float(floor.mean()),
                               tolerance=inst.tolerance(len(floor)),
                               justification="arm A: the same state text with the interval phrase "
                                             "removed. A MEASURED stimulus floor, which is what h38 "
                                             "built arm A for")},
            floor=Floor(stimulus=float(floor.mean()), estimator=0.0),
            selection=Selection(axis=None, rule=f"pre-registered layer {l}; no sweep"),
            provenance={"model": "Qwen/Qwen2.5-1.5B", "layers": [int(l)], "pooling": "mean",
                        "grid_hash": "clock_gain_v1", "template": None,
                        "tokenizer_padding": "right",
                        "code_version": "scripts/time_translation_discrimination.py (frozen)",
                        "lib_versions": {}},
            stage="h38", report_as="gain_over_floor")
        out["claim"] = {"reported_gain": claim.reported_value, "id": claim.id,
                        "render": claim.render()}
        out["refusal"] = None
    except CoreError as e:
        out["refusal"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
        out["treatment"] = float(treat.mean())
        out["floor_value"] = float(floor.mean())
        out["gain"] = float((treat - floor).mean())
        out["logged"] = {"arm_D_L14": 0.961, "arm_A_L14": 0.522}
    return out


# ================================================================================================
# target: h14, the NEGATIVE target -- no gain over a norm-matched pass-through
# ================================================================================================
def theme_grid(path: pathlib.Path) -> tuple[Grid, dict]:
    g = json.loads(path.read_text())
    lead = g["lead"]
    items = []
    for key, span in g["spans"].items():
        scene, era, theme = key.split("/")
        text = f"{lead} {span}"
        items.append(Item(text=text, factors={"scene": scene, "era": era, "theme": theme},
                          spans={"span": (len(lead) + 1, len(text))}))
    return Grid(items=items, name=path.stem), g


def _theme_dirs(stack, grid: Grid, layer: int, train: Sequence[str]):
    """h14's `make_dirs`, re-expressed through `Direction` so the hold-out is checked."""
    vecs = stack.vectors("span", layer)
    idx = [i for i, it in enumerate(grid.items) if it.factors["scene"] in train]
    mu = vecs[idx].mean(axis=0)
    de, dt = {}, {}
    for f, out in (("era", de), ("theme", dt)):
        for lvl in grid.levels(f):
            sel = [i for i in idx if grid.items[i].factors[f] == lvl]
            out[lvl] = Direction(vecs={layer: vecs[sel].mean(axis=0) - mu},
                                 held_out={"axis": "scene",
                                           "unseen": set(grid.levels("scene")) - set(train)},
                                 fit_witness={"axis": "scene", "fit_values": sorted(train)}
                                 ).vec(layer)
    return de, dt, mu


def _cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def h14(lm, *, patch_layer: int = 14, read_layer: int = 20, scale: float = 1.0,
        progress: Callable[[str], None] = lambda s: None) -> dict:
    """h14/h23 as spec §1A now states it: a NEGATIVE target. The core must report no gain over a
    norm-matched pass-through (logged -0.111 / -0.125 / -0.028).

    Everything is re-derived here -- the model arm is re-run locally through patched forwards and
    the pass-through is computed offline from the SAME asserted stack -- so the row's provenance is
    a real `Stack` and the ledger will take it. The logged JSON is only compared against.
    """
    from ..model import Patch
    from ..steer import add_vector

    path = prompt("narrative_theme_v1.json")
    grid, g = theme_grid(path)
    E, T, S = g["factors"]["era"], g["factors"]["theme"], g["scenes"]
    lead, spans = g["lead"], g["spans"]

    progress("extracting the theme stack through build_stack")
    stack = ex.build_stack(lm, grid, layers=[patch_layer, read_layer], batch_size=8)
    by_key = {f"{it.factors['scene']}/{it.factors['era']}/{it.factors['theme']}": i
              for i, it in enumerate(grid.items)}
    v_patch = stack.vectors("span", patch_layer)
    v_read = stack.vectors("span", read_layer)

    rng = np.random.default_rng(0)
    cases = []
    for s in S:
        train = [x for x in S if x != s]
        deP, _, _ = _theme_dirs(stack, grid, patch_layer, train)
        for e1 in E:
            for t in T:
                for e2 in E:
                    if e2 == e1:
                        continue
                    sh = deP[e2] - deP[e1]
                    r = rng.normal(size=sh.shape)
                    r *= np.linalg.norm(sh) / np.linalg.norm(r)
                    cases.append((s, e1, t, e2, "shift", sh))
                    cases.append((s, e1, t, e2, "rand", r))

    dir_cache = {s: _theme_dirs(stack, grid, read_layer, [x for x in S if x != s]) for s in S}

    def era_reads_as(vs: np.ndarray, targets: Sequence[str], scenes: Sequence[str]) -> np.ndarray:
        out = np.zeros(len(targets))
        for i, (v, tgt, s) in enumerate(zip(vs, targets, scenes)):
            deR, _, muR = dir_cache[s]
            u = v - muR
            out[i] = float(max(E, key=lambda e: _cos(u, deR[e])) == tgt)
        return out

    def rows_for(cond: str):
        sel = [c for c in cases if c[4] == cond]
        base = np.stack([v_read[by_key[f"{s}/{e1}/{t}"]] for s, e1, t, _, _, _ in sel])
        b14 = np.stack([v_patch[by_key[f"{s}/{e1}/{t}"]] for s, e1, t, _, _, _ in sel])
        # norm-matched: ||shift|| / ||resid|| held at its patch-layer value (the arm h40 rests on)
        k = (np.linalg.norm(base, axis=-1) / (np.linalg.norm(b14, axis=-1) + 1e-9))[:, None]
        shift = scale * k * np.stack([c[5] for c in sel])
        return sel, base, shift

    out = {"stack": stack, "grid": grid, "read_layer": read_layer, "patch_layer": patch_layer}
    for cond in ("shift", "rand"):
        sel, base, shift = rows_for(cond)
        tgt = [c[3] for c in sel]
        scn = [c[0] for c in sel]
        arm = instruments.PassthroughArm.compute(
            base=base, shift=shift,
            readout=(lambda vs, tgt=tgt, scn=scn: era_reads_as(vs, tgt, scn)),
            expected_null=0.0, justification="computed; its declared null is set in h14_claim")
        out[f"pt_{cond}"] = arm.scores
        out[f"{cond}_cases"] = sel
        out[f"{cond}_zero_shift_error"] = arm.zero_shift_error

    progress(f"model arm: {len(cases)} patched forwards")
    for cond in ("shift", "rand"):
        sel = out[f"{cond}_cases"]
        vals = np.zeros(len(sel))
        for i, (s, e1, t, e2, _, vec) in enumerate(sel):
            gkey = f"{s}/{e1}/{t}"
            text = f"{lead} {spans[gkey]}"
            hs, mask = ex._forward(lm, [text], patches=[Patch(patch_layer, add_vector(vec, scale))])
            idx_map = ex._real_token_spans(lm, text, grid.items[by_key[gkey]].spans)
            v = ex._pool(hs[0], idx_map["span"], "mean")[read_layer]
            deR, _, muR = dir_cache[s]
            u = v - muR
            vals[i] = float(max(E, key=lambda e: _cos(u, deR[e])) == e2)
            if i % 24 == 0:
                progress(f"  {cond} {i}/{len(sel)}")
        out[f"model_{cond}"] = vals
    return out


def h14_claim(res: dict) -> tuple[Claim, dict]:
    """`readout_shift`: gain as a DIFFERENCE, never a ratio, with the pass-through arm declaring
    where the arithmetic actually sits -- which is what makes the negative result the only sayable
    one (piece 2's fourth harness stage, from the other side)."""
    gain = res["model_shift"] - res["pt_shift"]
    gain_rand = res["model_rand"] - res["pt_rand"]
    n = len(gain)
    # §2a's paraphrase-noise interval: within an (e1, e2, theme) cell the four scenes are four
    # wordings of the same content, so the within-cell spread of the gain IS paraphrase noise.
    cells: dict[tuple, list[float]] = {}
    for (s, e1, t, e2, _, _), gv in zip(res["shift_cases"], gain):
        cells.setdefault((e1, e2, t), []).append(float(gv))
    within = [float(np.var(v, ddof=1)) for v in cells.values() if len(v) > 1]
    paraphrase_sd = float(np.sqrt(np.mean(within))) if within else float("nan")

    inst = instruments.build("readout_shift", n=400, d=64)
    inst.config = {"d": int(res["stack"].acts.shape[-1]), "readout_sd": paraphrase_sd}
    pt_value = float(np.mean(res["pt_shift"]))
    arms = {
        "passthrough": instruments.PassthroughArm(
            scores=res["pt_shift"], expected_null=pt_value,
            tolerance=inst.tolerance(n), zero_shift_error=res["shift_zero_shift_error"],
            norm_matched=True,
            justification="the offline arithmetic readout(base_resid_at_read_layer + shift), shift "
                          "norm-matched to the read layer; it declares where the arithmetic sits"),
        "random": Arm(gain_rand, expected_null=0.0, tolerance=inst.tolerance(len(gain_rand)),
                      justification="matched-norm random direction, model minus its own "
                                    "pass-through"),
        "no_patch": Arm(np.zeros(n), expected_null=0.0, tolerance=inst.tolerance(n),
                        justification="zero shift: the model arm and the pass-through are the same "
                                      "object, so the gain is identically zero"),
    }
    claim = inst.claim(
        treatment=Measured(gain, label="h14 era shift: gain over the norm-matched pass-through"),
        arms=arms,
        floor=Floor(stimulus=0.0, estimator=0.0),
        selection=Selection(axis=None, rule="pre-registered patch 14 / read 20; no sweep"),
        provenance=dict(res["stack"].provenance, direction_held_out="scene"),
        grid=res["grid"], stage="h14", report_as="gain_over_floor",
        patch_layer=res["patch_layer"], readout_layer=res["read_layer"],
        notes=[f"paraphrase-noise sd of the gain (within era-pair x theme cell, 4 scenes): "
               f"{paraphrase_sd:.4f}; 3-sigma arm tolerance at n={n}: {inst.tolerance(n):.4f}",
               "the theme grid is FLAGGED leaky (era recoverable 0.89 against a permutation null "
               "of 0.31), so §6 forbids a raw score. The stimulus floor of this quantity is 0: the "
               "treatment and the pass-through read the SAME span text with the same readout, so "
               "whatever the wording gives away is in both arms and cancels in the difference. "
               "That is an argument, not a measurement, and it is recorded here as one."])
    return claim, {"paraphrase_sd": paraphrase_sd, "tolerance": inst.tolerance(n),
                   "gain": float(np.mean(gain)), "passthrough": pt_value,
                   "model": float(np.mean(res["model_shift"])),
                   "gain_rand": float(np.mean(gain_rand)),
                   "zero_shift_error": res["shift_zero_shift_error"]}


# ================================================================================================
# target: h4, the role lens on held-out domains
# ================================================================================================
def rotated_holonic_grid(path: pathlib.Path) -> tuple[Grid, dict]:
    """`prompts/holonic_v1_rotated.json` -> Grid. Six role spans per item, position-balanced by
    rotation, which is the grid h4's directions were estimated from."""
    from ..extract import parse_roles
    g = json.loads(path.read_text())
    items = []
    for key, marked in g["prompts"].items():
        domain, rot = key.split("/")
        p = parse_roles(marked)
        items.append(Item(text=p.text, factors={"domain": domain, "rot": rot},
                          spans={r: p.spans[r][0] for r in g["roles"]}))
    # `rot` is a position label, not a stimulus factor: its bag-of-tokens recoverability is 1.00 by
    # construction (the six rotations are the same six spans in six orders) and flagging it would
    # be the alarm-fatigue case piece 1 named. Declared, so the report records it rather than
    # firing on it.
    return Grid(items=items, name=path.stem, declared_leaks=("rot", "domain")), g


def h4(lm, *, layer: int = 20, scale: float = 1.0, n_rand: int = 2,
       progress: Callable[[str], None] = lambda s: None) -> dict:
    """h4's role lens, re-run through the core with the `permutation` arm it never had.

    `selector` requires random, no_patch AND permutation; h4 shipped a random control only. The
    permutation arm here permutes the ROLE LABELS within each training domain before the direction
    is averaged, so it is a direction fit by the same code on the same vectors carrying no role
    identity -- not a relabelling of the ranking, which is the h6 trap.
    """
    from ..model import Patch
    from ..steer import add_vector
    from ..extract import parse_roles

    path = prompt("holonic_v1_rotated.json")
    grid, g = rotated_holonic_grid(path)
    roles = g["roles"]
    R = len(roles)
    domains = sorted({it.factors["domain"] for it in grid.items})

    progress("extracting the rotated holonic stack through build_stack")
    stack = ex.build_stack(lm, grid, layers=[layer], batch_size=4)

    acts = np.stack([stack.vectors(r, layer) for r in roles], axis=1)      # [item, role, d]
    idx_of = {(it.factors["domain"], it.factors["rot"]): i for i, it in enumerate(grid.items)}
    avg = {d: np.mean([acts[idx_of[(d, f"rot{k}")]] for k in range(R)], axis=0) for d in domains}

    lead, spans = {}, {}
    for d in domains:
        marked = g["prompts"][f"{d}/rot0"]
        p = parse_roles(marked)
        lead[d] = marked.split(" First,")[0]
        spans[d] = {r: p.text[p.spans[r][0][0]:p.spans[r][0][1]] for r in roles}

    rng = np.random.default_rng(0)
    perm_rng = np.random.default_rng(17)
    out = {"role": [], "rand": [], "permutation": [], "no_patch": [],
           "stack": stack, "grid": grid, "layer": layer, "n_candidates": R}

    for d in domains:
        train = [x for x in domains if x != d]
        M = np.mean([avg[x] for x in train], axis=0)                 # [role, d]
        dirs = M - M.mean(0, keepdims=True)
        Mp = np.mean([avg[x][perm_rng.permutation(R)] for x in train], axis=0)
        dirs_perm = Mp - Mp.mean(0, keepdims=True)

        prefix = f"{lead[d]} First,"
        cands = [f" {spans[d][r]}." for r in roles]
        base = np.array([lm.logprob(prefix, c) for c in cands])

        def ranks(vec, *, expect_move: bool) -> float:
            patches = [Patch(layer, add_vector(vec, scale))]
            sc = ex.asserted_patched_logprob(lm, prefix, cands, patches,
                                             base=base if expect_move else None)
            return sc - base

        from .checks import midrank
        zero = ranks(np.zeros(dirs.shape[1]), expect_move=False)
        for Ri, Rname in enumerate(roles):
            out["no_patch"].append(midrank(zero, Ri))
            out["role"].append(midrank(ranks(dirs[Ri], expect_move=True), Ri))
            out["permutation"].append(midrank(ranks(dirs_perm[Ri], expect_move=True), Ri))
            nrm = float(np.linalg.norm(dirs[Ri]))
            for _ in range(n_rand):
                v = rng.normal(size=dirs.shape[1])
                v *= nrm / np.linalg.norm(v)
                out["rand"].append(midrank(ranks(v, expect_move=True), Ri))
        progress(f"domain {d} done ({len(out['role'])}/{len(domains) * R} role items)")
    return out


def h4_claim(res: dict) -> tuple[Claim, dict]:
    inst = instruments.build("selector", n=400, d=64, n_candidates=res["n_candidates"])
    t = np.asarray(res["role"], dtype=float)
    claim = inst.claim(
        treatment=Measured(t, label=f"h4 role lens rank/{res['n_candidates']} @L{res['layer']}"),
        arms={"random": np.asarray(res["rand"], float),
              "no_patch": np.asarray(res["no_patch"], float),
              "permutation": np.asarray(res["permutation"], float)},
        floor=Floor(stimulus=float((res["n_candidates"] + 1) / 2),
                    estimator=float((res["n_candidates"] + 1) / 2)),
        selection=Selection(axis=None, rule="pre-registered layer 20, scale 1.0; no sweep"),
        provenance=dict(res["stack"].provenance, direction_held_out="domain (leave-one-domain-out)"),
        grid=res["grid"], stage="h4")
    return claim, {"role": float(t.mean()), "random": float(np.mean(res["rand"])),
                   "no_patch": float(np.mean(res["no_patch"])),
                   "permutation": float(np.mean(res["permutation"])), "n": int(t.size)}


# ================================================================================================
# target: h16, the relation selector -- as the full layer curve §1A restates it
# ================================================================================================
def rotated_holonic_v2_grid(path: pathlib.Path) -> tuple[Grid, dict]:
    """`prompts/holonic_v2_rotated.json` -> Grid: 40 domains x 6 rotations, six role spans each."""
    from ..extract import parse_roles
    g = json.loads(path.read_text())
    items = []
    for key, marked in g["prompts"].items():
        domain, rot = key.split("/")
        p = parse_roles(marked)
        items.append(Item(text=p.text, factors={"domain": domain, "rot": rot},
                          spans={r: p.spans[r][0] for r in g["roles"]}))
    # as in h4's grid: `rot` is a position label, not a stimulus factor -- the six rotations are the
    # same six spans in six orders, so its bag-of-tokens recoverability is 1.00 by construction.
    return Grid(items=items, name=path.stem, declared_leaks=("rot", "domain")), g


def _role_rank_per_item(C: np.ndarray, pred: np.ndarray, dst_idx: int) -> np.ndarray:
    """Where the true target role lands among the held-out prompt's own six roles, by cosine to the
    prediction. MID-RANK on ties.

    `operate.holdout_eval` -- which is what h16 ran -- counts `(sims > sims[dst]).sum() + 1`, i.e.
    strict-greater, i.e. rank 1 on a wholly tied field. That is the h34 shape, and h16 predates the
    rule that retired it. Nothing in h16 was tied, so the numbers do not move (measured below), but
    the core does not get to use a ranking rule it refuses elsewhere.
    """
    from .checks import midrank
    out = np.zeros(len(pred))
    for j in range(len(pred)):
        q = pred[j] / (np.linalg.norm(pred[j]) + 1e-9)
        sims = C[j] @ q
        out[j] = midrank(sims, dst_idx)
    return out


def h16_role_rank(acts: np.ndarray, domains: np.ndarray, roles: Sequence[str], *,
                  ridge: float = 10.0, role_center: bool = True, null_seed: int | None = None,
                  arm: str = "operator", seed: int = 0, return_groups: bool = False):
    """h16's statistic, PER ITEM: leave-one-domain-out affine operator, role-centred, mid-ranked.

    `acts` is [n_prompts, n_roles, d] at ONE layer, grand-mean removed. Returns one rank per
    (held-out prompt, ordered role pair) -- which is what `Instrument.sweep` needs and what
    `operate.holdout_eval` does not return: it reports the fold means, so h16's layer curve could
    only ever be asserted to the core, never computed by it (piece 3's §8.3).

    `arm` selects which prediction is ranked, so every arm is the SAME code on the same folds:
      * `operator`  -- the fitted affine map (the treatment);
      * `mean`      -- the training-mean target, i.e. no relation applied (the no-patch arm);
      * `random`    -- a Gaussian prediction of matched norm (the random arm).
    `null_seed` permutes the src->dst pairing within the training fold only, which is h16's own
    null and the `permutation` arm.
    """
    from ..operate import fit_affine
    rng = np.random.default_rng(seed)
    R = len(roles)
    uniq = sorted(set(domains.tolist()))
    out: list[float] = []
    who: list = []                # which DOMAIN each score came from; see `_cluster_tolerance`
    for g in uniq:
        tr, te = domains != g, domains == g
        C = acts
        if role_center:
            mu = acts[tr].mean(0)                     # [R, d], TRAINING prompts only
            C = acts - mu
        te_idx = np.flatnonzero(te)
        Cn = C / (np.linalg.norm(C, axis=-1, keepdims=True) + 1e-9)
        for si in range(R):
            for di in range(R):
                if si == di:
                    continue
                S, O = C[:, si], C[:, di]
                S_tr, O_tr = S[tr], O[tr]
                if null_seed is not None:
                    perm = np.random.default_rng(null_seed * 7919 + si * 31 + di).permutation(
                        len(O_tr))
                    O_tr = O_tr[perm]
                if arm == "mean":
                    pred = np.repeat(O_tr.mean(0)[None, :], te.sum(), axis=0)
                elif arm == "random":
                    pred = rng.normal(size=(int(te.sum()), S.shape[1]))
                    pred *= (np.linalg.norm(O_tr.mean(0)) /
                             np.linalg.norm(pred, axis=-1, keepdims=True))
                else:
                    op = fit_affine(S_tr, O_tr, 0, roles[si], roles[di], n_spin=0, ridge=ridge,
                                    low_rank=None)
                    pred = op(S[te])
                vals = _role_rank_per_item(Cn[te_idx], pred, di)
                out.extend(vals.tolist())
                who.extend([g] * len(vals))
    values = np.asarray(out, dtype=np.float64)
    return (values, np.asarray(who)) if return_groups else values


def _cluster_tolerance(values: np.ndarray, groups: np.ndarray, z: float = 3.0) -> float:
    """A 3-sigma arm band from the spread of DOMAIN means, measured rather than assumed.

    `registry.arm_tolerance(n)` is `3 * sd_item / sqrt(n)`, which is right when the n items are
    independent. h16's arms are not: the same 240 prompts are re-ranked for 30 ordered role pairs
    at 15 layers, so the pooled arm has n = 108 000 scores over 40 independent fits and the
    i.i.d. band comes out at +-0.016. Every one of the three plumbing arms sits 0.02-0.03 from
    chance and is refused by it -- which is piece 2's finding in reverse. Piece 2 measured a flat
    0.15 firing on 55 % of clean arms because it was too loose for some shapes and too tight for
    others; this is a *measured* band that is too tight because the n counts repetitions rather
    than evidence.

    The independent unit of this design is the DOMAIN: leave-one-domain-out means all 6 rotations,
    30 pairs and 15 layers of one domain share a fit and a text. So the band is measured from the
    between-domain spread of the arm's own means. This is a cluster-robust standard error, and it
    is reported ALONGSIDE the i.i.d. band rather than instead of it, because the point is that the
    i.i.d. band is wrong and not merely inconvenient.
    """
    per = np.array([values[groups == g].mean() for g in sorted(set(np.asarray(groups).tolist()))])
    if len(per) < 2:
        return float("inf")
    return float(z * per.std(ddof=1) / np.sqrt(len(per)))


def h16(lm, *, layers: Sequence[int] = tuple(range(0, 29, 2)), ridge: float = 10.0,
        progress: Callable[[str], None] = lambda s: None) -> dict:
    """h16 re-run through the core: one asserted extraction, and the layer curve COMPUTED.

    The published value was `role_rank 1.73 at the peak layer 16`, an argmax over a sweep of the
    same held-out data that scores it, and the core refuses it (`h16_as_logged`). §1A restates the
    target as the full curve, mean 2.21 over all pairs and layers against a null of 3.5. This
    function computes that curve.
    """
    path = prompt("holonic_v2_rotated.json")
    grid, g = rotated_holonic_v2_grid(path)
    roles = list(g["roles"])

    progress(f"extracting {len(grid.items)} prompts x {len(roles)} roles through build_stack")
    stack = ex.build_stack(lm, grid, layers=list(layers), batch_size=4)
    domains = np.array([it.factors["domain"] for it in grid.items])

    def acts_at(layer: int) -> np.ndarray:
        a = np.stack([stack.vectors(r, layer) for r in roles], axis=1)     # [item, role, d]
        return a - a.mean(axis=(0, 1), keepdims=True)                      # h16's grand-mean removal

    out = {"stack": stack, "grid": grid, "roles": roles, "layers": list(layers),
           "n_candidates": len(roles), "ridge": ridge, "per_layer": {}}
    for layer in layers:
        a = acts_at(layer)
        row = {}
        for arm in ("operator", "mean", "random"):
            row[arm], groups = h16_role_rank(a, domains, roles, ridge=ridge, arm=arm,
                                             return_groups=True)
        row["permutation"] = h16_role_rank(a, domains, roles, ridge=ridge, null_seed=1)
        row["role_identity_retained"] = h16_role_rank(a, domains, roles, ridge=ridge,
                                                      role_center=False)
        row["groups"] = groups
        out["per_layer"][int(layer)] = row
        progress(f"layer {layer}: operator {row['operator'].mean():.3f} "
                 f"perm {row['permutation'].mean():.3f} mean {row['mean'].mean():.3f}")
    return out


def h16_claim(res: dict) -> tuple[Claim | None, dict]:
    """The restated §1A target: the curve, reported whole, with the semantic null §4 names.

    Three things the logged form did not have, all of them required by the core:
      * the curve is computed by `Instrument.sweep`, so `Selection.executed` is a fact rather than
        prose (piece 3 made that mandatory at the ledger);
      * the `permutation` arm is h16's own null, the `no_patch` arm is the mean-target baseline
        (the prediction with no relation applied) and the `random` arm is a matched-norm Gaussian
        prediction -- three arms where h16 published one;
      * the semantic null §4 names for exactly this target, "role identity retained" (1.37 against
        2.21), computed as a real arm and declared BEFORE the treatment, which is what the
        construction order enforces.
    """
    inst = instruments.build("selector", n=400, d=64, n_candidates=res["n_candidates"])
    layers = res["layers"]
    per = res["per_layer"]

    # declared first, by construction order: the semantic null exists before any treatment score.
    sem = Arm(np.concatenate([per[l]["role_identity_retained"] for l in layers]),
              expected_null=float((res["n_candidates"] + 1) / 2),
              tolerance=float("inf"),
              justification="role identity retained (no role-centering): the lens can read which "
                            "ROLE a vector is without carrying any relation, and h16 logs it at "
                            "1.37 against the role-centred 2.21. It is not a plumbing arm and is "
                            "not expected at chance -- it is the question 'is this a relation or "
                            "is it role identity?' as a number (spec §4)")

    selection = inst.sweep("layer", layers, lambda l: float(per[int(l)]["operator"].mean()))
    treat = np.concatenate([per[l]["operator"] for l in layers])
    arms = {name: np.concatenate([per[l][key] for l in layers])
            for name, key in (("random", "random"), ("no_patch", "mean"),
                              ("permutation", "permutation"))}
    groups = np.concatenate([per[l]["groups"] for l in layers])
    bands = {name: _cluster_tolerance(v, groups) for name, v in arms.items()}
    iid_band = inst.tolerance(len(treat))
    summary = {"curve": dict(selection.curve), "treatment": float(treat.mean()),
               "random": float(arms["random"].mean()),
               "no_patch": float(arms["no_patch"].mean()),
               "permutation": float(arms["permutation"].mean()),
               "role_identity_retained": float(sem.value), "n": int(treat.size),
               "arm_tolerance": float(iid_band),
               "cluster_bands": {k: float(v) for k, v in bands.items()},
               "n_domains": int(len(set(groups.tolist()))),
               "curve_step4_mean": float(np.mean([v for k, v in selection.curve.items()
                                                  if int(k) % 4 == 0])),
               "peak_layer": min(selection.curve, key=lambda k: selection.curve[k]),
               "peak_value": min(selection.curve.values())}

    def build() -> Claim:
        return inst.claim(
            treatment=Measured(treat, label="h16 role_rank/6, all pairs x all swept layers"),
            arms={"random": Arm(arms["random"],
                                expected_null=float((res["n_candidates"] + 1) / 2),
                                tolerance=bands["random"],
                                justification="a Gaussian prediction of matched norm, ranked by "
                                              "the same code against the same six candidates. "
                                              f"Band {bands['random']:.4f}: 3 sigma on the "
                                              "between-DOMAIN spread, because leave-one-domain-out "
                                              "makes the domain the independent unit and the "
                                              f"i.i.d. band at n={len(treat)} ({iid_band:.4f}) "
                                              "counts repetitions rather than evidence"),
                  "no_patch": Arm(arms["no_patch"],
                                  expected_null=float((res["n_candidates"] + 1) / 2),
                                  tolerance=bands["no_patch"],
                                  justification="the training-mean target: the prediction with no "
                                                "relation applied at all. Band "
                                                f"{bands['no_patch']:.4f}, between-domain"),
                  "permutation": Arm(arms["permutation"],
                                     expected_null=float((res["n_candidates"] + 1) / 2),
                                     tolerance=bands["permutation"],
                                     justification="h16's own null: the src->dst pairing permuted "
                                                   "within the training fold, held-out rows "
                                                   "untouched. Band "
                                                   f"{bands['permutation']:.4f}, between-domain")},
            semantic_null=sem,
            floor=Floor(stimulus=float((res["n_candidates"] + 1) / 2),
                        estimator=float(np.mean(arms["permutation"]))),
            selection=selection,
            provenance=dict(res["stack"].provenance,
                            direction_held_out="domain (leave-one-domain-out)"),
            grid=res["grid"], stage="h16",
            notes=[f"arm bands are CLUSTER-ROBUST: 3 sigma on the spread of {len(set(groups.tolist()))} "
                   f"domain means, {bands}. The i.i.d. band `registry.arm_tolerance` would give at "
                   f"n={len(treat)} is {iid_band:.4f}, and it is wrong here rather than merely "
                   "tight: the same 240 prompts are re-ranked for 30 role pairs at "
                   f"{len(layers)} layers, so n counts repetitions and not evidence.",
                   "the curve is reported whole; its step-2 mean is "
                   f"{float(treat.mean()):.4f} and its step-4 mean is "
                   f"{float(np.mean([v for k, v in selection.curve.items() if int(k) % 4 == 0])):.4f}. "
                   "§1A's target of 2.21 is the second aggregate and this repo's own "
                   "results/stage3_qwen1.5b_v2_rolecentered.json is the first, at 2.1692."])

    # The claim is built inside a try because h16's own baselines are what is under test here, and
    # one of them does not survive the contract. Reported, not caught-and-hidden: `summary` carries
    # the refusal and the numbers that produced it either way.
    try:
        claim = build()
        summary["refusal"] = None
    except CoreError as e:
        claim = None
        summary["refusal"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
    return claim, summary


# ================================================================================================
# target: h29, the era shift in generation -- the statistic that had no instrument
# ================================================================================================
def h29_from_logged() -> Row:
    """h29's 3x re-imposed era shift, re-derived per item and graded through `top1_accuracy`.

    Piece 3 reproduced these numbers twice, bit-identically, and could not put them anywhere: the
    registry shipped a rank, a joint rank and a projection gain, and h29's statistic is an
    ACCURACY. `top1_accuracy` exists now, so the number can be computed by the core -- and the row
    still cannot be published, for two reasons that are both about h29's battery rather than about
    the instrument:

      * the logged run carries ONE arm. `ndif_recompose_sweep.py` at scale 3.0 emits the `shift`
        condition and nothing else, so there is no random-direction arm and no no-patch arm.
        `CLAUDE.md`'s first non-negotiable -- "every battery reports treatment, random AND
        no-patch" -- is not met by a number that is in `RESULTS.md` today, and `MissingArm` is the
        core saying so without being told to look.
      * the provenance is a frozen script's JSON, not a `Stack`, so `ProvenanceNotFromStack` fires
        at the ledger even if the arms existed.

    The verdict is therefore REFUSED and no longer `deferred (no instrument)`. What changed is
    which of the two sentences is true: "the core cannot compute this" has become "the core
    computes it and will not publish it".
    """
    path = data("recompose_sweep_reimpose_3.0.json")
    blob = json.loads(path.read_text())
    scored = [r for r in blob["rows"] if "era_read" in r]
    n_attempted = len(blob["rows"])
    era = np.array([float(r["era_read"] == r["e2"]) for r in scored])
    leaves = np.array([float(r["era_read"] != r["e1"]) for r in scored])
    theme = np.array([float(r["theme_read"] == r["t"]) for r in scored])
    lex_rows = [r for r in scored if r["lex_era"] not in (None, "none", "-")]
    lex = np.array([float(r["lex_era"] == r["e2"]) for r in lex_rows])

    inst = instruments.build("top1_accuracy", n=400, d=64, n_candidates=3)
    prov = {"model": blob["meta"]["model"], "layers": [blob["meta"]["patch_layer"],
                                                       blob["meta"]["read_layer"]],
            "pooling": "generated text", "grid_hash": blob["meta"]["grid"],
            "template": blob["meta"].get("prompt_format"), "tokenizer_padding": "left",
            "code_version": "scripts/ndif_recompose_sweep.py (frozen)", "lib_versions": {},
            "direction_held_out": "scene"}
    detail_bits = []
    try:
        claim = inst.claim(
            treatment=Measured(era, label="h29 era reads as target, 3x re-imposed @ scale 3.0"),
            arms={},
            floor=Floor(stimulus=float(lex.mean()), estimator=1.0 / 3.0),
            selection=Selection(axis=None, rule="pre-registered scale 3.0, patch 14 / read 20"),
            provenance=prov, stage="h29", report_as="raw")
        verdict, claim_id = FAILED, claim.id
        detail_bits.append("the core ACCEPTED a one-armed battery")
    except CoreError as e:
        verdict, claim_id = REFUSED, ""
        detail_bits.append(f"{type(e).__name__} -- {str(e).splitlines()[0][:150]}")

    # ... and what it would still be refused for once the arms existed.
    prov_refusal = None
    try:
        ledger.check_provenance_from_stack(
            type("_P", (), {"provenance": prov, "instrument": "top1_accuracy"})())
    except CoreError as e:
        prov_refusal = type(e).__name__
    detail_bits.append(f"at the ledger, separately: {prov_refusal}")

    tol = inst.tolerance(len(era))
    return Row(target="h29 era shift in generation, 3x re-imposed", source="h29",
               logged="0.84 era->target / 0.91 leaves e1 / 0.53 theme kept, lex 0.30, n=55/72",
               reproduced=f"{era.mean():.4f} / {leaves.mean():.4f} / {theme.mean():.4f}, "
                          f"lex {lex.mean():.2f} (n={len(scored)}/{n_attempted})",
               tolerance=f"+-{REMOTE_TOLERANCE:.4f} remote (measured, piece 3); arm band "
                         f"+-{tol:.4f} at n={len(era)}",
               verdict=verdict, detail="; ".join(detail_bits),
               claim_id=claim_id,
               extra={"era_target": float(era.mean()), "leaves_e1": float(leaves.mean()),
                      "theme_kept": float(theme.mean()), "lexical_floor": float(lex.mean()),
                      "n": len(scored), "n_attempted": n_attempted,
                      "null_era_target": 1.0 / 3.0,
                      "null_leaves_e1": 2.0 / 3.0,
                      "null_theme_kept": 1.0 / 3.0,
                      "arm_tolerance": float(tol),
                      "arms_present": sorted({r["cond"] for r in blob["rows"]}),
                      "arms_required": list(registry.required_arms("top1_accuracy"))})
