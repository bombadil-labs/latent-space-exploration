"""Phase 2's derivable batch (`docs/specs/phase2_v1.md` §5), through the core.

Two targets, and one correctness fix that had to land first:

* **§5.1 -- writeup claims 2b and 4c**, h41's two skip-path rows, as `readout_shift` claims
  carrying a `PassthroughArm`. A direction added at block L reaches the unembedding by the
  residual skip path whether or not any block uses it, so the quantity that means "the stack
  computed this" is `treatment - passthrough`, never the treatment alone (spec §2a).
* **§5.2 -- writeup claim 4b**, h8's battery re-fit on the cached replication stacks, each row
  reporting **gain over its own MEASURED lexical floor** rather than over chance (spec §6, and
  h47: of h8's four reference rows only era clears its floor).

The fix is `Arm.n_independent` (see `registry.InstrumentSpec.arm_tolerance`), which everything
here depends on: several of these arms have items that are not independent draws.

**What this module does NOT do: append to `results/ledger.jsonl`.** Every number below comes from
a cached `.npz` stack or a cached JSON that a frozen script in `scripts/` wrote, not from
`extract.build_stack`, so none of these claims carries a stack signature and the ledger refuses
them with `ProvenanceNotFromStack` -- the same refusal h29 hit in piece 4, which piece 5 cleared
by re-extracting. Re-extraction is not available here: four of the five replication models are not
in the local HF cache at all, and h41's residuals were never cached. The claims are therefore
*built* -- which runs the whole §4 contract, every arm against its declared null, both floors, the
calibration gate -- rendered, and written to `results/`. They are candidates for the ledger and
the note says so rather than rounding it off.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import re
from typing import Sequence

import numpy as np

from . import instruments
from .checks import ArmUnitNotInDesign, CoreError, midrank
from .reproduce import (data, factor_grid, h8_lexical_floor, prompt, rotated_holonic_grid)
from .types import Arm, Floor, Grid, Item, Measured, Selection


# ================================================================================================
# shared helpers
# ================================================================================================
def margin(gains: Sequence[float], target: int) -> float:
    """The h41 readout in nats: the target candidate's log-probability gain minus the mean gain of
    its siblings.

    Recomputed here from the per-candidate readouts the run recorded, rather than read off the
    run's own `m`. That is an aggregation check, not independent evidence about the readout, and
    it is named as one: what it catches is a wrong candidate ordering or a wrong target index on
    this side, which would otherwise be invisible.
    """
    g = np.asarray(gains, dtype=np.float64)
    others = np.delete(g, target)
    return float(g[target] - others.mean())


def clustered_arm(scores, *, expected_null: float, justification: str, clusters, unit: str,
                  factory=None, **kw) -> tuple[Arm, str]:
    """Declare the arm's independent unit from the design, and take the TIGHTER band if the data
    do not show it.

    The design reason is real for every arm here -- the cases inside one domain (h41 role) or one
    scene (h41 composed, h8) share a leave-one-out fit and a candidate set, so they are not
    independent draws from the null. But `Arm` refuses a declaration whose clustering is not
    visible in the arm's own scores, and the fallback when it fires is the i.i.d. band at the item
    count, which is *narrower*: falling back can only make this check harsher, never laxer. Which
    of the two happened is returned, and recorded in the row.
    """
    make = factory if factory is not None else (lambda **a: Arm(scores, **a))
    try:
        arm = make(expected_null=expected_null, justification=justification,
                   clusters=tuple(clusters), unit=unit, **kw)
        ev = arm.unit_evidence or {}
        return arm, (f"banded on {arm.effective_n} units ({unit}); between-unit share "
                     f"{ev.get('between_share', float('nan')):.3f} vs "
                     f"{ev.get('null_mean_share', float('nan')):.3f} shuffled, p="
                     f"{ev.get('p', float('nan')):.3f}")
    except ArmUnitNotInDesign as e:
        arm = make(expected_null=expected_null, justification=justification, **kw)
        return arm, (f"declared unit {unit!r} REFUSED, banded on all {arm.n} items (the tighter "
                     f"band): {str(e).splitlines()[0][:160]}")


def bootstrap_lb(values, clusters=None, *, q: float = 0.05, draws: int = 10000,
                 seed: int = 0) -> float:
    """The record's "90% lower bound": the 5th percentile of the bootstrap mean (the lower end of
    a 90% two-sided interval, which is what `scripts/selector_direct_path_report.py` computed and
    `RESULTS.md` prints). With `clusters`, the resampling unit is the CLUSTER rather than the
    item.

    The two differ for the same reason the arm band did: resampling 72 cases that are four scenes
    pretends to four times the evidence the design has. Both are reported; the clustered one is
    the honest one, and it is *not* the number the record carries.
    """
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if clusters is None:
        means = v[rng.integers(0, len(v), size=(draws, len(v)))].mean(1)
        return float(np.percentile(means, q * 100))
    lab = np.asarray([str(c) for c in clusters])
    idx = [np.where(lab == g)[0] for g in sorted(set(lab))]
    k = len(idx)
    means = np.empty(draws)
    for b in range(draws):
        means[b] = v[np.concatenate([idx[p] for p in rng.integers(0, k, k)])].mean()
    return float(np.percentile(means, q * 100))


# ================================================================================================
# §5.1 -- h41's two skip-path rows (writeup claims 2b and 4c)
# ================================================================================================
def _rows_by_arm(path: pathlib.Path, *, claim: str, layer: int) -> tuple[dict, dict]:
    d = json.loads(path.read_text())
    out: dict = {}
    for r in d["rows"]:
        if r["claim"] == claim and r["layer"] == layer:
            out.setdefault(r["arm"], {})[r["case"]] = r
    return out, d["meta"]


def h41_case_table(which: str) -> dict:
    """Per-case readouts for one h41 target, recomputed from the cached per-candidate gains.

    `which` is 'role' (writeup claim 2b: the role lens at L20, six candidates) or 'composed'
    (claim 4c: the three-factor composition at L14, eighteen joint variants).
    """
    if which == "role":
        src = data("selector_direct_path_role.json")
        rows, meta = _rows_by_arm(src, claim="role", layer=20)
        g = json.loads(prompt("holonic_v1_rotated.json").read_text())
        keys = list(g["roles"])

        def target_of(case: str) -> int:
            return keys.index(case.split("/")[1])

        def cluster_of(case: str) -> str:
            return case.split("/")[0]                      # the domain: the leave-one-out unit
        unit = ("one held-out domain: the six role cases inside a domain share one "
                "leave-one-domain-out fit, one prompt and one candidate set")
    elif which == "composed":
        src = data("selector_direct_path_factors.json")
        rows, meta = _rows_by_arm(src, claim="D", layer=14)
        g = json.loads(prompt("narrative_factors_v2.json").read_text())
        F = g["factors"]
        keys = list(itertools.product(*[F[n] for n in F]))

        def target_of(case: str) -> int:
            return keys.index(tuple(case.split("/")[1:]))

        def cluster_of(case: str) -> str:
            return case.split("/")[0]                      # the scene
        unit = ("one held-out scene: the eighteen composed cases inside a scene share one "
                "leave-one-scene-out fit, one lead and one 18-candidate set")
    else:
        raise ValueError(which)

    cases = sorted(rows["A"])
    per: dict = {"case": cases, "cluster": [cluster_of(c) for c in cases], "unit": unit,
                 "meta": meta, "source": src.name, "n_candidates": len(keys)}
    disagreement = {}
    for arm in ("A", "F_par", "R0", "N"):
        vals, rks, logged = [], [], []
        for c in cases:
            r = rows[arm][c]
            t = target_of(c)
            vals.append(margin(r["gains"], t))
            rks.append(midrank(r["gains"], t))
            logged.append(float(r["m"]))
        per[arm] = np.array(vals)
        per[arm + "_rank"] = float(np.mean(rks))
        disagreement[arm] = float(np.max(np.abs(np.array(vals) - np.array(logged))))
    per["recompute_vs_logged_max_abs"] = disagreement
    per["identity_error"] = max(float(r.get("fdelta_err", np.inf))
                                for r in rows["F_delta"].values())
    per["no_patch_max_abs_gain"] = max(float(r.get("max_abs_gain", np.inf))
                                       for r in rows["N"].values())
    # the random arm's OWN direct path, for the record: an equal-norm Gaussian read offline at the
    # pre-norm residual. Not paired to R0's draw, so it is a diagnostic and not the arm.
    per["random_direct_path"] = float(np.mean([r["m"] for r in rows.get("F_R_10", {}).values()]))
    per["skip_norm_fraction"] = None
    return per


def h41_claim(which: str):
    """One `readout_shift` claim: gain over the skip-path pass-through, as a DIFFERENCE."""
    per = h41_case_table(which)
    gain = per["A"] - per["F_par"]
    n = len(gain)
    clusters = per["cluster"]

    # the readout's own noise at one case, MEASURED: the spread of the equal-norm random arm's
    # margin. `readout_shift`'s registry default is sqrt(2/d), which is the spread of a unit-scaled
    # projection readout and is meaningless for a log-probability margin in nats (0.036 against a
    # random arm that moves by +-0.8). Passed as config, which is where §1A says a measured spread
    # enters the registry.
    readout_sd = float(np.std(per["R0"], ddof=1))
    inst = instruments.build("readout_shift", n=400, d=64)
    inst.config = {"d": 1536, "readout_sd": readout_sd}

    pt_value = float(np.mean(per["F_par"]))
    passthrough, pt_note = clustered_arm(
        per["F_par"], expected_null=pt_value, clusters=clusters, unit=per["unit"],
        factory=(lambda scores=per["F_par"], **a:
                 instruments.PassthroughArm.from_recorded_offline(scores=scores, **a)),
        justification="the skip path: the OBSERVED displacement projected on the patched "
                      "direction and added to the base pre-norm residual, read through the same "
                      "final-norm-and-unembedding code as the treatment with no forward pass "
                      "(spec §2a). It declares where the arithmetic sits, as h14's does.",
        identity_error=per["identity_error"], identity_tol=1e-4,
        source=f"scripts/selector_direct_path.py -> results/{per['source']}")
    random_arm, rand_note = clustered_arm(
        per["R0"], expected_null=0.0, clusters=clusters, unit=per["unit"],
        justification="an equal-norm Gaussian patched at the same layer: it has no direction the "
                      "stack or the unembedding cares about, so its margin should be 0")
    no_patch = Arm(per["N"], expected_null=0.0,
                   justification="no patch at all, scored through the identical code path; the "
                                 "margin is identically 0 and the run's own max |gain| is "
                                 f"{per['no_patch_max_abs_gain']:.3g}")

    grid = (rotated_holonic_grid(prompt("holonic_v1_rotated.json"))[0] if which == "role"
            else factor_grid(prompt("narrative_factors_v2.json"))[0])
    lb_item = bootstrap_lb(gain)                    # the record's own form
    lb = bootstrap_lb(gain, clusters)               # the same, resampling units
    label = ("h41 role lens @L20: gain over the skip path" if which == "role"
             else "h41 three-factor composition @L14: gain over the skip path")
    claim = inst.claim(
        treatment=Measured(gain, label=label),
        arms={"passthrough": passthrough, "random": random_arm, "no_patch": no_patch},
        floor=Floor(stimulus=0.0, estimator=0.0),
        selection=Selection(axis=None, rule=("pre-registered patch layer "
                                             f"{20 if which == 'role' else 14}, readout at the "
                                             "unembedding; no sweep")),
        provenance={"model": per["meta"]["model"], "layers": per["meta"]["layers"],
                    "grid": per["meta"]["grid"], "scale": per["meta"]["scale"],
                    "lib_versions": {"torch": per["meta"]["torch"],
                                     "transformers": per["meta"]["transformers"]},
                    "tied_embeddings": per["meta"]["tied_embeddings"],
                    "source": f"results/{per['source']}",
                    "patched_direction": which,
                    "code_version": "lsx.core.phase2 (aggregation); "
                                    "scripts/selector_direct_path.py (readouts)"},
        grid=grid, stage="h41", report_as="gain_over_floor",
        patch_layer=(20 if which == "role" else 14), readout_layer=28,
        notes=[
            f"pass-through arm: {pt_note}",
            f"random arm: {rand_note}",
            "the pass-through arm is RECORDED, not recomputed here: §2a's arithmetic ran inside "
            "scripts/selector_direct_path.py and the residuals it needs were never cached. It is "
            "admitted against that run's own offline identity gate -- the same offline path fed "
            f"the FULL observed displacement reproduces the treatment's forward to "
            f"{per['identity_error']:.2g} nats, against the 1e-4 the run registered. That is the "
            "closest available analogue of the zero-shift assertion and it is weaker than it.",
            f"the stimulus floor of this quantity is 0 by cancellation: treatment and pass-through "
            "read the SAME candidate texts with the SAME readout, so whatever the wording gives "
            "away is in both arms and subtracts out. That is an argument, not a measurement, and "
            "it is recorded as one (h14's row carries the same one).",
            f"90% lower bound on the gain: {lb_item:+.4f} resampling the {n} cases, which is the "
            f"form the record carries, and {lb:+.4f} resampling the "
            f"{len(set(clusters))} independent units instead. The case bootstrap is the narrower "
            "interval because it pretends to more evidence than the design has.",
            f"per-case readout noise, measured from the random arm: sd {readout_sd:.4f} nats",
            f"the random direction's own direct path (offline Gaussian at the pre-norm residual, "
            f"unpaired): {per['random_direct_path']:+.4f} nats",
        ])
    detail = {"which": which, "n_cases": n, "n_units": len(set(clusters)),
              "treatment_margin": float(np.mean(per["A"])),
              "passthrough": pt_value, "gain": float(np.mean(gain)),
              "lb90_cluster": lb, "lb90_item": lb_item,
              "sign_fraction": float(np.mean(gain > 0)),
              "random": float(np.mean(per["R0"])), "no_patch": float(np.mean(per["N"])),
              "readout_sd": readout_sd, "identity_error": per["identity_error"],
              "recompute_vs_logged_max_abs": per["recompute_vs_logged_max_abs"],
              "ranks": {a: per[a + "_rank"] for a in ("A", "F_par", "R0", "N")},
              "arm_band": {k: a.resolved_tolerance for k, a in claim.arms.items()},
              "source": per["source"]}
    return claim, detail


# ================================================================================================
# §5.2 -- h8's battery re-fit on the cached replication stacks (writeup claim 4b)
# ================================================================================================
#
# **What is and is not the same as h8.** h8's battery is a PATCHED FORWARD: a level direction is
# added at layer 14 and the candidate continuations are re-scored by log probability. None of the
# replication models can be patched here -- four of the five are not in the local HF cache and one
# is a 9B -- and what IS cached is the activation stack. So these rows are the READOUT analogue of
# h8's lens: the same leave-one-scene-out level directions, the same candidate sets, the same
# `midrank`, with the candidate's residual scored by cosine against the direction instead of its
# continuation being re-scored by a patched model. That is a different instrument reading of the
# same design, and the note says so rather than calling it h8 re-run.
#
# It is the same arithmetic as the lexical floor with activations in place of word counts, which
# is the point: `reproduce.h8_lexical_floor` was built to be "the treatment with the activations
# swapped out and nothing else changed", so on THIS side of the comparison the two differ in
# exactly one thing.

GRID_OF_STACK = {
    # cached stack (results/*.npz)                      grid file                model tag
    "stacks_qwen2.5_1.5b_narrative_factors_v2":        ("narrative_factors_v2", "Qwen2.5-1.5B"),
    "stacks_qwen2.5_1.5b_narrative_factors_v1":        ("narrative_factors_v1", "Qwen2.5-1.5B"),
    "stacks_qwen2.5_0.5b_narrative_factors_v1":        ("narrative_factors_v1", "Qwen2.5-0.5B"),
    "stacks_pythia_1.4b_narrative_factors_v1":         ("narrative_factors_v1", "Pythia-1.4B"),
    "stacks_gemma_2_9b_it_narrative_factors_v1":       ("narrative_factors_v1", "Gemma-2-9B-it"),
    "stacks_gpt_j_6b_narrative_theme_v1":              ("narrative_theme_v1",   "GPT-J-6B"),
    "stacks_qwen2.5_1.5b_narrative_factors_gpt_v1":    ("narrative_factors_gpt_v1", "Qwen2.5-1.5B"),
    "stacks_qwen2.5_1.5b_narrative_theme_gpt_v1":      ("narrative_theme_gpt_v1",   "Qwen2.5-1.5B"),
}

# The five §5.2 targets, plus two anchors that are not replications and are labelled as such: the
# reference grid read out of activations instead of patched forwards (so the two instruments can
# be compared on the SAME model and grid), and the reference model on the v1 grid.
REPLICATIONS = ["stacks_qwen2.5_0.5b_narrative_factors_v1",
                "stacks_pythia_1.4b_narrative_factors_v1",
                "stacks_gemma_2_9b_it_narrative_factors_v1",
                "stacks_gpt_j_6b_narrative_theme_v1",
                "stacks_qwen2.5_1.5b_narrative_factors_gpt_v1",
                "stacks_qwen2.5_1.5b_narrative_theme_gpt_v1"]
ANCHORS = ["stacks_qwen2.5_1.5b_narrative_factors_v2",
           "stacks_qwen2.5_1.5b_narrative_factors_v1"]

# Block count per model, so that "layer 0" can be CHECKED rather than assumed to be the static
# embedding. `lsx.extract` writes [L+1, d] with the embedding at index 0 -- but two of these
# stacks are one row short of that (GPT-J-6B 28 rows for 28 blocks, Gemma-2-9B-it 42 for 42),
# so they were written by a path that returns block outputs only and their index 0 is the output
# of block 0, not the embedding. That changes what the shallowest row MEANS, and it is exactly the
# kind of thing a gitignored stack written by a frozen script cannot be asked about later.
# Qwen2.5-1.5B (28), Qwen2.5-0.5B (24), Gemma-2-9B-it (42) and GPT-J-6B (28) were read out of the
# local HF cache's config.json; Pythia-1.4B (24) is the published architecture and was NOT checked
# here, because the weights are not in the local cache.
MODEL_BLOCKS = {"Qwen2.5-1.5B": 28, "Qwen2.5-0.5B": 24, "Pythia-1.4B": 24,
                "Gemma-2-9B-it": 42, "GPT-J-6B": 28}


def normalized_grid(name: str) -> dict:
    """{factors, scenes, spans, lead} from any of the narrative grid files.

    `narrative_factors_v1` predates the 'factors'/'key_order' convention and spells its two
    factors 'eras' and 'voices'; the rest carry it. Normalising here rather than in the battery
    means the battery cannot quietly read the factors in a different ORDER than the span keys are
    written in, which would silently mislabel every level -- so the order is checked against every
    key rather than assumed.
    """
    g = json.loads(prompt(f"{name}.json").read_text())
    if "factors" in g:
        factors = {k: list(v) for k, v in g["factors"].items()}
        order = list(g.get("key_order", ["scene", *factors])[1:])
        factors = {k: factors[k] for k in order}
    else:                                   # narrative_factors_v1: 'eras' + 'voices', in key order
        factors = {"era": list(g["eras"]), "voice": list(g["voices"])}
    spans = g["spans"]
    for key in spans:
        parts = key.split("/")
        if len(parts) != 1 + len(factors):
            raise ValueError(f"span key {key!r} has {len(parts) - 1} factor levels, "
                             f"{len(factors)} declared")
        for (fname, levels), got in zip(factors.items(), parts[1:]):
            if got not in levels:
                raise ValueError(f"span key {key!r}: {got!r} is not a level of {fname!r} "
                                 f"({levels}) -- the factor order and the key order disagree")
    return {"factors": factors, "scenes": list(g["scenes"]), "spans": spans,
            "lead": g["lead"], "name": name}


def load_stack_matrix(stack_name: str, gridspec: dict) -> dict:
    """`{span key: [layer, d]}` from a cached `.npz`, checked against the grid it claims to be.

    The stacks are gitignored and were written by frozen scripts; the only things that can be
    verified from here are that the keys are exactly the grid's keys, that the shapes agree and
    that the vectors are finite. All three are checked, because a stack silently missing a scene
    would turn a leave-one-scene-out fit into a leave-nothing-out fit.
    """
    z = np.load(data(f"{stack_name}.npz"), allow_pickle=False)
    want, got = set(gridspec["spans"]), set(z.files)
    if want != got:
        raise ValueError(f"{stack_name} does not match grid {gridspec['name']}: "
                         f"{len(want - got)} keys missing, {len(got - want)} extra")
    X = {k: np.asarray(z[k], dtype=np.float64) for k in sorted(got)}
    shapes = {v.shape for v in X.values()}
    if len(shapes) != 1:
        raise ValueError(f"{stack_name}: vectors disagree about shape: {shapes}")
    if not all(np.isfinite(v).all() for v in X.values()):
        raise ValueError(f"{stack_name}: non-finite activations")
    return X


def _cos_rank(cand_vecs: np.ndarray, direction: np.ndarray, target: int) -> float:
    """The readout: cosine of each candidate's own vector against the direction, mid-ranked.

    Identical in form to `reproduce.h8_lexical_floor`'s scoring, down to the 1e-12 in the
    denominator, so that the floor and the treatment differ in the vectors and in nothing else.
    A zero direction therefore gives an exactly tied field, which mid-rank reads as chance -- that
    is what the `no_patch` arm is here.
    """
    d = np.asarray(direction, dtype=np.float64)
    scores = [float(c @ d / (np.linalg.norm(c) * np.linalg.norm(d) + 1e-12)) for c in cand_vecs]
    return midrank(scores, target)


def readout_battery(X: dict, gridspec: dict, layer: int, *, n_perm: int = 32, n_rand: int = 8,
                    seed: int = 0) -> dict:
    """h8's lens and composition tests as a readout on a cached stack, with all four arms.

    Arms, all fitted by the same code on the same vectors:

      * `treatment`  -- level directions, leave-one-scene-out;
      * `random`     -- `n_rand` independent passes, each drawing a Gaussian of the same norm as
        the level direction per item. Several passes for the same reason the permutation arm needs
        several draws, and it was not a formality: with ONE pass Gemma-2-9B-it's voice arm read
        1.583 against a declared 2.000 and the row was refused. The fix is more draws, never a
        wider band (spec §7);
      * `no_patch`   -- the ZERO direction. In a patched battery the no-patch arm is the unpatched
        forward; in a readout there is nothing to unpatch, and the arm that means the same thing
        is a readout carrying no direction at all. Every candidate then scores exactly 0.0, the
        field is wholly tied, and `midrank` must read the candidate-set midpoint. An arm that read
        1.0 there would be h34 exactly (rank-1-on-ties), so this arm puts the tie rule under test
        as well as the plumbing;
      * `permutation` -- `n_perm` INDEPENDENT draws, each shuffling the factor labels among the
        TRAINING items before the level means are taken (`reproduce.permuted_level_directions`'s
        arithmetic: the labels before the fit, never the ranking afterwards, which is h6). Not one
        draw: h47 measured a single draw's own spread at up to 0.35 of a rank against a 3-sigma
        band of 0.29, so one draw has almost no power.
    """
    F = gridspec["factors"]
    names = list(F)
    S = gridspec["scenes"]
    combos = list(itertools.product(*[F[n] for n in names]))
    rng = np.random.default_rng(seed)
    perm_rng = np.random.default_rng(1000 + seed)

    def key(s, c):
        return "/".join([s, *c])

    def blank():
        return {"treatment": [], "no_patch": [], "scene": [],
                "random": [[] for _ in range(n_rand)], "perm": [[] for _ in range(n_perm)]}

    out = {"composed": blank(), "B": {n: blank() for n in names},
           "n_variants": len(combos), "layer": layer,
           "levels": {n: len(F[n]) for n in names}}

    for s in S:
        train = [t for t in S if t != s]
        train_keys = [key(t, c) for t in train for c in combos]
        train_labels = [c for _t in train for c in combos]
        M = np.stack([X[k][layer] for k in train_keys])
        mu = M.mean(axis=0)

        def fit(labels) -> dict:
            return {n: {lvl: M[[i for i in range(len(labels))
                                if labels[i][names.index(n)] == lvl]].mean(axis=0) - mu
                        for lvl in F[n]} for n in names}

        D = fit(train_labels)
        Dperm = [fit([train_labels[i] for i in perm_rng.permutation(len(train_labels))])
                 for _ in range(n_perm)]
        cand = {c: X[key(s, c)][layer] for c in combos}
        all_vecs = np.stack([cand[c] for c in combos])

        for c in combos:
            # --- composed: rank the jointly correct variant among all V ---------------------
            dsum = sum(D[n][c[i]] for i, n in enumerate(names))
            t = combos.index(c)
            out["composed"]["treatment"].append(_cos_rank(all_vecs, dsum, t))
            out["composed"]["no_patch"].append(_cos_rank(all_vecs, np.zeros_like(dsum), t))
            out["composed"]["scene"].append(s)
            for j in range(n_rand):
                r = rng.normal(size=dsum.shape)
                r *= np.linalg.norm(dsum) / np.linalg.norm(r)
                out["composed"]["random"][j].append(_cos_rank(all_vecs, r, t))
            for p in range(n_perm):
                dp = sum(Dperm[p][n][c[i]] for i, n in enumerate(names))
                out["composed"]["perm"][p].append(_cos_rank(all_vecs, dp, t))
            # --- each factor as a lens, the others held fixed -------------------------------
            for i, n in enumerate(names):
                sub = [cc for cc in combos
                       if all(cc[j] == c[j] for j in range(len(names)) if j != i)]
                vecs = np.stack([cand[cc] for cc in sub])
                t_i = sub.index(c)
                d = D[n][c[i]]
                out["B"][n]["treatment"].append(_cos_rank(vecs, d, t_i))
                out["B"][n]["no_patch"].append(_cos_rank(vecs, np.zeros_like(d), t_i))
                out["B"][n]["scene"].append(s)
                for j in range(n_rand):
                    r = rng.normal(size=d.shape)
                    r *= np.linalg.norm(d) / np.linalg.norm(r)
                    out["B"][n]["random"][j].append(_cos_rank(vecs, r, t_i))
                for p in range(n_perm):
                    out["B"][n]["perm"][p].append(_cos_rank(vecs, Dperm[p][n][c[i]], t_i))
    return out


def mid_depth(n_layers: int) -> int:
    """The pre-registered layer, fixed by a rule and not by a score: the middle of the stack.

    h8's own patch layer is 14 of Qwen2.5-1.5B's 0..28, which IS this rule, so the rule reproduces
    the reference row's layer rather than being invented for the replications. The whole layer
    curve is computed and written into every row beside the reported value, so a reader can see
    what the rule cost -- but no value other than the pre-registered one is ever reported as the
    result (spec §4: h16's "peak layer 16" stays refused, and it was an argmax over exactly this
    curve).
    """
    return int(round(0.5 * (n_layers - 1)))


def pooled_draw_arm(draws: list, *, expected_null: float, unit: str, justification: str,
                    inst) -> tuple[Arm, dict]:
    """The permutation arm over several independent draws, banded on the BETWEEN-DRAW spread.

    Two bands exist for this arm now and they are not the same number:

      * the registry's, at the declared independent unit -- `3 * sd_item_null / sqrt(n_draws)`,
        which is what phase 2's `arm_tolerance` fix returns once the arm declares its unit;
      * the measured one -- `3 * sd(draw means) / sqrt(n_draws)`, h47's form, which is a direct
        estimate of the spread of THIS arm's mean rather than a closed form for an ideal one.

    The measured one is used, and on these rows it is the tighter of the two. That is deliberate
    and it is the whole of spec §7: the unit declaration must not become a way to buy a wider
    band. Both are recorded so a reader can see which was taken and by how much.
    """
    means = np.array([float(np.mean(d)) for d in draws])
    scores = np.concatenate([np.asarray(d, dtype=float) for d in draws])
    clusters = tuple(f"draw{i}" for i, d in enumerate(draws) for _ in range(len(d)))
    sd = float(np.std(means, ddof=1))
    measured = 3.0 * sd / np.sqrt(len(means))
    arm, note = clustered_arm(
        scores, expected_null=expected_null, clusters=clusters, unit=unit,
        tolerance=float(measured),
        justification=(f"{len(means)} independent draws pooled (means "
                       f"{np.round(means, 3).tolist()}); {justification}"))
    info = {"n_draws": len(means), "draw_means": means.tolist(), "sd_between_draws": sd,
            "band_measured": float(measured),
            "band_registry_at_units": float(inst.tolerance(len(scores), len(means))),
            "band_registry_iid": float(inst.tolerance(len(scores))), "unit_note": note}
    return arm, info


def phase2_grid(gridspec: dict) -> Grid:
    """The grid as a `Grid`, so its leak report is computed and the rows are forced onto
    `report_as='gain_over_floor'` by the contract rather than by my good intentions."""
    lead = gridspec["lead"]
    items = []
    for key, span in gridspec["spans"].items():
        parts = key.split("/")
        text = f"{lead} {span}"
        items.append(Item(text=text,
                          factors={"scene": parts[0],
                                   **{f: v for f, v in zip(gridspec["factors"], parts[1:])}},
                          spans={"span": (len(lead) + 1, len(text))}))
    return Grid(items=items, name=gridspec["name"])


def layer_curve(X: dict, gridspec: dict, *, n_layers: int, what: str, factor: str | None = None,
                step: int = 1) -> dict:
    """The treatment's whole depth curve, computed and recorded beside the reported value.

    Not used to choose anything -- `mid_depth` is fixed by a rule announced before any score -- and
    that is exactly why it can be computed at all: the refusal in the contract is on SELECTING with
    it (h16's peak layer 16, h38's L* = argmax), not on knowing it.
    """
    curve = {}
    for l in range(0, n_layers, step):
        b = readout_battery(X, gridspec, l, n_perm=0)
        v = b["composed"]["treatment"] if what == "composed" else b["B"][factor]["treatment"]
        curve[str(l)] = float(np.mean(v))
    return curve


def h8_replication_rows(stack_name: str, *, n_perm: int = 32, n_rand: int = 8, seed: int = 0,
                        curve_step: int = 2) -> list[dict]:
    """Every row this stack carries: one `selector` per factor, one `composition` over the joint
    variants. Each reports GAIN OVER ITS OWN MEASURED LEXICAL FLOOR (spec §6), never over chance.

    This is the CACHED path: `X` comes from a frozen script's `.npz`, not from `build_stack`, so
    none of these rows carries a stack signature and the ledger refuses them with
    `ProvenanceNotFromStack` (see the module docstring). `h8_replication_rows_live` is the same
    battery on a real `Stack` -- it shares `_replication_rows_core` with this function rather than
    reimplementing the battery, which is the thing spec phase2_qwen §"FIND THE EXISTING PATH"
    warns against.
    """
    grid_name, model = GRID_OF_STACK[stack_name]
    gridspec = normalized_grid(grid_name)
    X = load_stack_matrix(stack_name, gridspec)
    n_layers = next(iter(X.values())).shape[0]
    prov_base = {"model": model, "grid": grid_name, "stack": f"results/{stack_name}.npz",
                 "n_layers": n_layers, "pooling": "as cached by scripts/",
                 "direction_held_out": "scene (leave-one-scene-out)",
                 "code_version": "lsx.core.phase2.readout_battery",
                 "readout": "cosine of the candidate's own residual against the level direction, "
                            "mid-ranked -- the READOUT analogue of h8's patched log-prob selector"}
    rows, _claims = _replication_rows_core(
        X, n_layers, gridspec, model, stack_name, prov_base,
        is_replication=stack_name in REPLICATIONS,
        n_perm=n_perm, n_rand=n_rand, seed=seed, curve_step=curve_step)
    return rows


def stack_matrix_from_build_stack(lm, gridspec: dict, *, batch_size: int = 8) -> tuple[dict, object]:
    """`{span key: [layer, d]}` extracted THROUGH `extract.build_stack` -- every §7 assertion runs
    -- rather than read from a frozen script's cached `.npz`. This is the path `reproduce.h8`
    already uses for `narrative_factors_v2` (`stack = ex.build_stack(lm, grid, layers=[layer],
    batch_size=8)`); the only difference here is `layers=None`, which stores every layer so the
    depth curve and the layer-0 floor need no second extraction -- the forward pass already
    computes every hidden state regardless of how many `build_stack` is told to keep.

    Returns `(X, stack)`: `X` has exactly the shape `load_stack_matrix` returns, so
    `_replication_rows_core`, `readout_battery` and `layer_curve` run UNCHANGED on either path.
    """
    from . import extract as ex

    grid = phase2_grid(gridspec)
    stack = ex.build_stack(lm, grid, layers=None, batch_size=batch_size)
    span_idx = stack.span_names.index("span")
    X = {key: np.asarray(stack.acts[i, span_idx], dtype=np.float64)
         for i, key in enumerate(gridspec["spans"])}
    return X, stack


def h8_replication_rows_live(lm, grid_name: str, *, model: str = "Qwen2.5-1.5B",
                             n_perm: int = 32, n_rand: int = 8, seed: int = 0,
                             curve_step: int = 2, batch_size: int = 8) -> tuple[list[dict], dict]:
    """`h8_replication_rows`'s battery, extracted LIVE through `build_stack` so the rows carry a
    real stack signature and can reach `results/ledger.jsonl`.

    Returns `(rows, cosine_report)`: `cosine_report` compares this live extraction against the
    cached `.npz` for the same grid, item by item and layer by layer, wherever that cached stack
    exists -- the check phase2_qwen §CONSTRAINTS 1 asks for.
    """
    gridspec = normalized_grid(grid_name)
    X, stack = stack_matrix_from_build_stack(lm, gridspec, batch_size=batch_size)
    n_layers = next(iter(X.values())).shape[0]
    prov_base = dict(stack.provenance, direction_held_out="scene (leave-one-scene-out)")
    stack_tag = f"build_stack:{grid_name}:{model} (live, {stack.provenance.get('acts_digest')})"

    cosine_report = {"grid": grid_name, "compared": False}
    cached_name = f"stacks_qwen2.5_1.5b_{grid_name}"
    try:
        cached_X = load_stack_matrix(cached_name, gridspec)
    except FileNotFoundError:
        cached_X = None
    if cached_X is not None:
        cos = []
        for key in gridspec["spans"]:
            a, b = X[key], cached_X[key]
            if a.shape != b.shape:
                cosine_report["shape_mismatch"] = [key, list(a.shape), list(b.shape)]
                continue
            num = np.sum(a * b, axis=-1)
            den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
            cos.append(num / den)
        if cos:
            cos = np.concatenate(cos)
            cosine_report.update(compared=True, cached_stack=f"results/{cached_name}.npz",
                                 n_pairs=int(cos.size), mean_cosine=float(cos.mean()),
                                 min_cosine=float(cos.min()),
                                 max_cosine=float(cos.max()))

    rows, claims = _replication_rows_core(
        X, n_layers, gridspec, model, stack_tag, prov_base,
        is_replication=False, n_perm=n_perm, n_rand=n_rand, seed=seed, curve_step=curve_step)
    return rows, claims, cosine_report


def _replication_rows_core(X: dict, n_layers: int, gridspec: dict, model: str, stack_tag: str,
                           prov_base: dict, *, is_replication: bool, n_perm: int, n_rand: int,
                           seed: int, curve_step: int) -> list[dict]:
    """The battery and claim construction shared by the cached path (`h8_replication_rows`) and
    the live path (`h8_replication_rows_live`): everything downstream of having `X` in hand.
    """
    grid_name = gridspec["name"]
    layer = mid_depth(n_layers)
    names = list(gridspec["factors"])
    V = int(np.prod([len(v) for v in gridspec["factors"].values()]))

    lex = h8_lexical_floor(gridspec=gridspec)
    lex_perm = h8_lexical_floor(gridspec=gridspec, permute_seed=5)
    batt = readout_battery(X, gridspec, layer, n_perm=n_perm, n_rand=n_rand, seed=seed)
    # A SECOND stimulus floor, stricter than the lexical one on several rows: the same readout run
    # on the shallowest cached layer. Where that row is the static embedding (see MODEL_BLOCKS) it
    # is spec §6's layer-0 check computed by the identical arithmetic instead of by a ridge, and it
    # is a floor the bag-of-tokens predictor does not reach -- era is at chance for the bag and at
    # 1.50 for the embedding readout on the reference grid. Reported beside the lexical floor, not
    # in place of it: §5.2 asks for the gain over `h8_lexical_floor` and that is what the claims
    # carry.
    shallow = readout_battery(X, gridspec, 0, n_perm=0, n_rand=0, seed=seed)
    has_embedding = n_layers == MODEL_BLOCKS.get(model, -1) + 1
    grid = phase2_grid(gridspec)
    scenes = batt["composed"]["scene"]
    if lex["scene"] != scenes:
        raise ValueError(f"{stack_tag}: the lexical floor's items are not the battery's items "
                         "in the same order; a paired gain would be pairing the wrong rows")

    prov_base = dict(prov_base, layers=prov_base.get("layers", [layer]))
    rule = (f"pre-registered mid-depth layer {layer} of {n_layers} "
            f"(the rule round(0.5*(L-1)) was fixed before any score was computed and is h8's own "
            f"layer 14 of 29); the full depth curve is recorded in this row and no value from it "
            f"entered the reported number")

    rows, claims = [], []
    specs = [("composed", None, V)] + [("lens", n, len(gridspec["factors"][n])) for n in names]
    for kind, factor, k in specs:
        b = batt["composed"] if kind == "composed" else batt["B"][factor]
        b0 = shallow["composed"] if kind == "composed" else shallow["B"][factor]
        floor0_items = np.asarray(b0["treatment"], dtype=float)
        floor_items = np.asarray(lex["composed"] if kind == "composed" else lex["B"][factor],
                                 dtype=float)
        floor_perm = float(np.mean(lex_perm["composed"] if kind == "composed"
                                   else lex_perm["B"][factor]))
        chance = (k + 1) / 2.0
        if kind == "composed":
            inst = instruments.build("composition", n=200, d=32,
                                     levels=tuple(len(v) for v in gridspec["factors"].values()))
        else:
            inst = instruments.build("selector", n=200, d=32, n_candidates=k)
        treat = np.asarray(b["treatment"], dtype=float)
        unit = (f"one held-out scene: the {len(treat) // len(set(scenes))} items of a scene share "
                "one leave-one-scene-out fit and one candidate set")
        perm_arm, perm_info = pooled_draw_arm(
            b["perm"], expected_null=chance, unit="one permutation draw of the factor labels, "
            "shared by every item fitted from it", inst=inst,
            justification="each draw shuffles the factor labels among the TRAINING items before "
                          "the level means are taken, so the direction is built by the same code "
                          "out of the same activations and carries no level identity; the ranking "
                          "is not permuted, which is h6")
        rand_arm, rand_info = pooled_draw_arm(
            b["random"], expected_null=chance, unit="one pass of random directions, one per item",
            inst=inst,
            justification="each pass draws a fresh Gaussian direction per item, of the same norm "
                          "as that item's level direction, read out by the same code")
        arms = {"random": rand_arm,
                "no_patch": Arm(np.asarray(b["no_patch"], float), expected_null=chance,
                                justification="the ZERO direction: every candidate scores exactly "
                                              "0.0, the field is wholly tied, and mid-rank must "
                                              "read the candidate midpoint (h34's tie rule under "
                                              "test, not just plumbing)"),
                "permutation": perm_arm}
        gain_items = treat - floor_items
        row = {"stack": stack_tag, "model": model, "grid": grid_name, "kind": kind,
               "factor": factor or "composed", "k": k, "layer": layer, "n_layers": n_layers,
               "n_items": int(treat.size), "treatment": float(treat.mean()),
               "floor_lexical": float(floor_items.mean()),
               "floor_permutation_control": floor_perm, "chance": chance,
               # SIGN, because it is the opposite of the word: these are RANKS, so a gain over the
               # floor is NEGATIVE and a row clears its floor when it sits BELOW it. This is the
               # record's own convention (h47: era -0.7500 clears, voice +0.2083 does not) and
               # `Claim.reported_value` computes treatment - floor, so the row matches the claim.
               "gain_over_floor": float(treat.mean() - floor_items.mean()),
               "gain_over_chance": float(treat.mean() - chance),
               "gain_ci90_scene_bootstrap": [bootstrap_lb(gain_items, scenes, q=0.05),
                                             bootstrap_lb(gain_items, scenes, q=0.95)],
               "clears_floor": bool(bootstrap_lb(gain_items, scenes, q=0.95) < 0.0),
               "floor_shallowest_layer": float(floor0_items.mean()),
               "shallowest_layer_is_embedding": bool(has_embedding),
               "gain_over_shallowest": float(treat.mean() - floor0_items.mean()),
               "clears_shallowest": bool(bootstrap_lb(treat - floor0_items, scenes, q=0.95) < 0.0),
               "arms": {"random": float(rand_arm.value),
                        "no_patch": float(np.mean(b["no_patch"])),
                        "permutation": float(perm_arm.value)},
               "permutation_arm": perm_info, "random_arm": rand_info,
               "curve": layer_curve(X, gridspec, n_layers=n_layers, what=kind, factor=factor,
                                    step=curve_step),
               "is_replication": is_replication}
        try:
            claim = inst.claim(
                treatment=Measured(treat, label=f"{model} {grid_name} "
                                                f"{'composed' if kind == 'composed' else factor}"
                                                f" rank/{k} @L{layer} (readout)"),
                arms=arms,
                floor=Floor(stimulus=float(floor_items.mean()), estimator=chance),
                selection=Selection(axis=None, rule=rule),
                provenance=dict(prov_base, factor=factor or "composed"),
                grid=grid, stage="h8-replication", report_as="gain_over_floor",
                notes=[f"floor.stimulus = {floor_items.mean():.4f}/{k}, MEASURED on THIS grid by "
                       "`reproduce.h8_lexical_floor`: a bag-of-tokens predictor fit "
                       "leave-one-scene-out by the same arithmetic and ranked through the "
                       "identical `midrank` over the identical candidates. Its own permutation "
                       f"control sits at {floor_perm:.4f} against chance {chance:.2f}, so the "
                       "predictor is reading the words and not the procedure.",
                       f"permutation arm: {perm_info['unit_note']}",
                       f"random arm: {rand_info['n_draws']} passes, means "
                       f"{np.round(rand_info['draw_means'], 3).tolist()}, band "
                       f"{rand_info['band_measured']:.4f}",
                       f"permutation bands: measured between-draw {perm_info['band_measured']:.4f} "
                       f"(used), registry at {perm_info['n_draws']} declared units "
                       f"{perm_info['band_registry_at_units']:.4f}, registry i.i.d. at "
                       f"{int(treat.size) * perm_info['n_draws']} items "
                       f"{perm_info['band_registry_iid']:.4f}. The tighter one is used.",
                       f"paired gain over the floor (ranks, so BELOW the floor is better), 90% "
                       f"interval resampling the {len(set(scenes))} held-out scenes: "
                       f"[{row['gain_ci90_scene_bootstrap'][0]:+.4f}, "
                       f"{row['gain_ci90_scene_bootstrap'][1]:+.4f}] -- clears its floor: "
                       f"{row['clears_floor']}",
                       "a SECOND and stricter stimulus floor, the same readout on the shallowest "
                       "cached layer ("
                       + ("the static embedding" if has_embedding else
                          "block 0's output -- this stack carries no embedding row")
                       + f"): {floor0_items.mean():.4f}, gain "
                       f"{treat.mean() - floor0_items.mean():+.4f}"
                       f", clears it: {row['clears_shallowest']}. §5.2 asks for the gain over the "
                       "lexical floor and that is what this claim's Floor carries; this number is "
                       "reported beside it because on several rows it is the harder test and the "
                       "row does not pass it.",
                       "this is the READOUT analogue of h8's patched battery, not h8 re-run: the "
                       "replication models cannot be patched here, so the candidate's residual is "
                       "scored by cosine against the direction rather than its continuation being "
                       "re-scored by a patched model.",
                       f"depth curve (step {curve_step}) recorded in results/"
                       "phase2_h8_replications.json; the reported layer is the pre-registered "
                       "one and nothing was chosen from the curve"])
            row["claim_id"] = claim.id
            row["render"] = claim.render()
            row["status"] = "built"
            row["arm_bands"] = {kk: a.resolved_tolerance for kk, a in claim.arms.items()}
            row["arms_off_null"] = [kk for kk, a in claim.arms.items() if a.off_null]
            claims.append(claim)
        except CoreError as e:
            row["status"] = "REFUSED"
            row["refusal"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
        rows.append(row)
    return rows, claims


def run(out_dir: pathlib.Path | None = None, *, n_perm: int = 32, curve_step: int = 2) -> dict:
    """Both §5 targets, written to `results/`."""
    out_dir = pathlib.Path(out_dir) if out_dir else pathlib.Path(
        "/home/user/latent-space-exploration/.claude/worktrees/"
        "agent-a6566d620ec8091ee/results")
    stage41 = {}
    for which in ("role", "composed"):
        claim, detail = h41_claim(which)
        detail["claim_id"] = claim.id
        detail["render"] = claim.render()
        detail["notes"] = list(claim.notes)
        stage41[which] = detail
    (out_dir / "phase2_stage41_rows.json").write_text(json.dumps(stage41, indent=1, default=str))

    rows = []
    for stack in ANCHORS + REPLICATIONS:
        rows.extend(h8_replication_rows(stack, n_perm=n_perm, curve_step=curve_step))
    (out_dir / "phase2_h8_replications.json").write_text(json.dumps(rows, indent=1, default=str))
    return {"stage41": stage41, "replications": rows}
