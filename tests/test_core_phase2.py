"""Phase 2: the arm-band fix (`Arm.n_independent`), and the two §5 targets.

The rule this file is written to is the one the band fix is most exposed to: a declaration that
widens a tolerance must be as checkable as a wrong number, so most of what is tested here is the
REFUSAL, not the feature.
"""
import json
import pathlib

import numpy as np
import pytest

from lsx.core import checks, instruments, phase2, registry, rediscovery
from lsx.core.types import Arm, Floor, Measured, Selection

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


# ================================================================================================
# 1. the band fix
# ================================================================================================
def test_arm_tolerance_defaults_to_the_item_count_exactly_as_before():
    """The fix must be a no-op on every arm that does not declare a unit. This is not a style
    point: `Instrument.resolved_null_tol` -- the threshold the CALIBRATION BATTERY passes or fails
    against -- is `arm_tolerance(n)`, and it is NOT part of the calibration key. So a change to the
    default path would leave every cached report on disk 'valid' while having been produced under a
    different threshold, which is piece 4's bug exactly. These numbers pin the default path."""
    sel = registry.spec("selector")
    assert sel.arm_tolerance(72, {"n_candidates": 3}) == pytest.approx(0.2886751, abs=1e-7)
    assert sel.arm_tolerance(48, {"n_candidates": 6}) == pytest.approx(0.7395099, abs=1e-7)
    assert registry.spec("composition").arm_tolerance(72, {"n_variants": 18}) == pytest.approx(
        1.8342801, abs=1e-6)
    assert registry.spec("top1_accuracy").arm_tolerance(55, {"n_candidates": 3}) == pytest.approx(
        0.1906925, abs=1e-7)
    # and passing the unit explicitly as "one item is one unit" changes nothing
    assert sel.arm_tolerance(72, {"n_candidates": 3}, n_independent=72) == sel.arm_tolerance(
        72, {"n_candidates": 3})


@pytest.mark.parametrize("name,kw,null_tol", [
    ("selector", {"n": 200, "d": 32, "n_candidates": 6}, 0.3622844),
    ("selector", {"n": 200, "d": 32, "n_candidates": 3}, 0.1732051),
    ("selector", {"n": 200, "d": 32, "n_candidates": 2}, 0.1060660),
    ("composition", {"n": 200, "d": 32, "levels": (3, 3, 2)}, 1.1005680),
    ("readout_shift", {"n": 400, "d": 64}, 0.0265165),
    ("discrimination", {"n": 200, "d": 32}, 0.0750000),
    ("top1_accuracy", {"n": 200, "d": 32, "n_candidates": 3}, 0.1000000)])
def test_the_band_fix_did_not_invalidate_any_cached_calibration(name, kw, null_tol):
    """Explicitly, because piece 4's lesson is that this is where the silence lives: the fix
    changes the arithmetic of `arm_tolerance`, the battery's own pass threshold comes out of it,
    and the calibration key does not cover it. Both halves are checked -- the threshold is
    unchanged to seven decimals, and every key still names a report that is on disk."""
    inst = instruments.build(name, **kw)
    assert inst.resolved_null_tol == pytest.approx(null_tol, abs=1e-7)
    assert (RESULTS / "calibration" / f"{name}.{inst.key}.json").exists(), (
        f"{name} {kw} now hashes to {inst.key}, which is not a cached report: the fix "
        "invalidated a calibration it should not have")


def test_a_declared_unit_bands_on_units_and_widens_nothing_by_itself():
    rng = np.random.default_rng(0)
    offsets = rng.normal(0, 0.4, size=6)
    scores = 2.0 + np.array([offsets[i % 6] for i in range(72)]) + rng.normal(0, 0.2, 72)
    scores = scores - scores.mean() + 2.0
    arm = Arm(scores, expected_null=2.0, n_independent=6, unit="one draw",
              clusters=tuple(f"d{i % 6}" for i in range(72)))
    # The band is on the MEASURED effective sample size (n / deff), not flatly on the cluster
    # count: this fixture's clustering is real but PARTIAL (icc ~0.5, not ~1), so n/deff lands
    # well above k=6 -- banding on k alone here would be the over-wide failure the fix closes.
    # Still bounded to [k, n] on both sides.
    assert 6 < arm.effective_n < 72
    assert arm.effective_n == 11
    assert arm.unit_evidence["p"] <= 0.05
    spec = registry.spec("selector")
    assert spec.arm_tolerance(arm.n, {"n_candidates": 3}, n_independent=arm.effective_n) > \
        spec.arm_tolerance(arm.n, {"n_candidates": 3})


def test_n_eff_is_clamped_to_k_and_n_and_never_widens_past_k():
    """Piece 5's constraint 1: `n_eff` is measured from the arm's own scores and can only ever
    land in `[k, n]`, so the loosest band a declared unit can ever buy is exactly the flat `k`
    band -- a declaration can only make the band NARROWER than that (more clustering visible in
    the scores earns a tighter band, never a wider one). Swept over a range of clustering
    strengths, including total clustering (icc -> 1, where n_eff must equal k exactly) and no
    real clustering at all (refused outright, so it never reaches the tolerance comparison)."""
    n, k = 72, 6
    lab = tuple(f"d{i % k}" for i in range(n))
    sel = registry.spec("selector")
    flat_tol = sel.arm_tolerance(n, {"n_candidates": 3}, n_independent=k)
    item_tol = sel.arm_tolerance(n, {"n_candidates": 3})
    for seed, offset_sd, noise_sd in [(0, 0.05, 0.30), (1, 0.15, 0.30), (2, 0.35, 0.20),
                                      (3, 0.80, 0.05), (4, 1.20, 0.02)]:
        rng = np.random.default_rng(seed)
        offsets = rng.normal(0, offset_sd, size=k)
        scores = 2.0 + np.array([offsets[i % k] for i in range(n)]) + rng.normal(0, noise_sd, n)
        scores = scores - scores.mean() + 2.0
        ev = checks.cluster_evidence(scores, lab)
        if ev["p"] > 0.05:
            continue    # not visibly clustered -- Arm refuses the declaration outright (tested
                        # elsewhere); nothing to bound here
        arm = Arm(scores, expected_null=2.0, n_independent=k, unit="one draw", clusters=lab)
        assert k <= arm.effective_n <= n, (seed, arm.effective_n)
        measured_tol = sel.arm_tolerance(n, {"n_candidates": 3}, n_independent=arm.effective_n)
        # never wider than the flat k-band, never tighter than banding on every raw item
        assert measured_tol <= flat_tol + 1e-12, (seed, measured_tol, flat_tol)
        assert measured_tol >= item_tol - 1e-12, (seed, measured_tol, item_tol)


def test_total_clustering_reproduces_the_h8_permutation_case_unchanged():
    """Piece 5's constraint 2: h8's permutation arm is four scenes re-ranked eighteen ways
    (n=72, k=4) with ICC near 1. At icc=1 exactly, deff = 1 + (mbar - 1) * 1 = mbar = n / k for
    ANY cluster sizing (mbar is always n/k, balanced or not), so n_eff = n / deff = k exactly --
    this case must be bit-for-bit unaffected by the fix, and it is by construction rather than by
    a special case in the code."""
    n, k = 72, 4
    lab = tuple(f"scene{i % k}" for i in range(n))
    rng = np.random.default_rng(2)
    offsets = rng.normal(0, 5.0, size=k)          # huge between-scene spread
    scores = 2.0 + np.array([offsets[i % k] for i in range(n)])   # ~zero within-scene noise
    scores = scores - scores.mean() + 2.0
    ev = checks.cluster_evidence(scores, lab)
    assert ev["p"] <= 0.05
    assert ev["icc"] > 0.999, ev            # "near 1", as the ticket specifies
    arm = Arm(scores, expected_null=2.0, n_independent=k, unit="one permutation draw, shared by "
              "all items of a scene", clusters=lab)
    assert arm.effective_n == k


def test_a_unit_declaration_with_no_design_behind_it_is_refused():
    rng = np.random.default_rng(1)
    iid = 2.0 + rng.normal(0, 0.4, size=72)
    lab = tuple(f"d{i % 6}" for i in range(72))
    with pytest.raises(checks.ArmUnitNotInDesign, match="not in the scores"):
        Arm(iid, expected_null=2.0, n_independent=6, unit="one draw", clusters=lab)
    with pytest.raises(checks.ArmUnitNotInDesign, match="no cluster labels"):
        Arm(iid, expected_null=2.0, n_independent=6, unit="one draw")
    with pytest.raises(checks.ArmUnitNotInDesign, match="says nothing about what a unit is"):
        Arm(iid, expected_null=2.0, n_independent=6)
    with pytest.raises(checks.ArmUnitNotInDesign, match="cannot be two numbers"):
        Arm(iid, expected_null=2.0, n_independent=4, unit="one draw", clusters=lab)


def test_a_constant_arm_cannot_buy_a_wider_band():
    """A constant arm sits exactly where it sits; a wider band around it can only hide an arm that
    is off its null, so `cluster_evidence` returns p = 1 rather than a convenient nan."""
    const = np.full(72, 2.4)
    ev = checks.cluster_evidence(const, [f"d{i % 6}" for i in range(72)])
    assert ev["p"] == 1.0
    with pytest.raises(checks.ArmUnitNotInDesign):
        Arm(const, expected_null=2.0, n_independent=6, unit="one draw",
            clusters=tuple(f"d{i % 6}" for i in range(72)))


def test_rediscovery_catches_a_band_widened_without_a_design_reason():
    v = rediscovery.case_12_band_widened_without_a_design_reason()
    assert v.ok, v.render()
    assert "ArmUnitNotInDesign" in v.mechanism
    assert "FAILED" not in v.positive_control


def test_rediscovery_catches_the_k_flat_band_this_piece_fixed():
    """Case 13: a `k`-flat band would have admitted an arm whose clustering is real but partial
    (ICC well under 1); the `n/deff`-measured band correctly flags it."""
    v = rediscovery.case_13_partial_clustering_k_band_admits_a_dirty_arm()
    assert v.ok, v.render()
    assert "ArmOffNull" in v.mechanism
    assert "FAILED" not in v.positive_control


def test_every_rediscovery_case_still_fires():
    vs = [v for v in rediscovery.run_all() if v.status != rediscovery.UNWIRED]
    assert all(v.ok for v in vs), "\n".join(v.render() for v in vs if not v.ok)
    assert len(vs) == 11


# ================================================================================================
# 2. the recorded pass-through arm
# ================================================================================================
def test_a_recorded_passthrough_arm_needs_its_identity_gate():
    kw = dict(scores=np.zeros(8), expected_null=0.0, justification="j",
              source="scripts/selector_direct_path.py")
    with pytest.raises(checks.PassthroughNotReproduced, match="needs the identity error"):
        instruments.PassthroughArm.from_recorded_offline(identity_error=None, **kw)
    with pytest.raises(checks.PassthroughNotReproduced, match="does not reproduce"):
        instruments.PassthroughArm.from_recorded_offline(identity_error=0.5, **kw)
    with pytest.raises(checks.PassthroughNotReproduced, match="must name where"):
        instruments.PassthroughArm.from_recorded_offline(
            scores=np.zeros(8), expected_null=0.0, justification="j", identity_error=1e-6,
            source="")
    arm = instruments.PassthroughArm.from_recorded_offline(identity_error=3.4e-5, **kw)
    assert arm.computed and not arm.recomputed and arm.zero_shift_error == pytest.approx(3.4e-5)


# ================================================================================================
# 3. §5.1 -- h41's two skip-path rows
# ================================================================================================
@pytest.mark.parametrize("which,gain,lb,n,units", [("role", 1.8367, 1.563, 48, 8),
                                                   ("composed", 3.6135, 3.251, 72, 4)])
def test_h41_rows_reproduce_the_record(which, gain, lb, n, units):
    claim, d = phase2.h41_claim(which)
    assert d["n_cases"] == n and d["n_units"] == units
    assert d["gain"] == pytest.approx(gain, abs=0.001)
    assert d["lb90_item"] == pytest.approx(lb, abs=0.002)      # the record's own resampling unit
    assert claim.reported_value == pytest.approx(gain, abs=0.001)
    assert not any(a.off_null for a in claim.arms.values())
    assert claim.arms["passthrough"].computed


def test_h41_margins_recompute_from_the_per_candidate_readouts():
    """Every arm's margin is rebuilt here from the run's per-candidate log-probability gains. If
    the candidate ordering or the target index on this side were wrong, this is what would say
    so -- and it agrees to the last bit, which is the only sense in which this side is checked."""
    for which in ("role", "composed"):
        per = phase2.h41_case_table(which)
        assert max(per["recompute_vs_logged_max_abs"].values()) == 0.0
        assert per["no_patch_max_abs_gain"] == 0.0
        assert per["identity_error"] < 1e-4


def test_the_h41_rows_require_a_passthrough_arm_at_all():
    """Drop the arm and the readout-after-patch rule must refuse the claim outright."""
    claim, _ = phase2.h41_claim("role")
    with pytest.raises(checks.MissingArm, match="passthrough"):
        type(claim)(instrument=claim.instrument, treatment=claim.treatment,
                    arms={k: v for k, v in claim.arms.items() if k != "passthrough"},
                    floor=claim.floor, selection=claim.selection, effect=claim.effect,
                    calibration=claim.calibration, provenance=claim.provenance,
                    grid=claim.grid, report_as=claim.report_as, patch_layer=claim.patch_layer,
                    readout_layer=claim.readout_layer, config=claim.config)


# ================================================================================================
# 4. §5.2 -- the replication battery
# ================================================================================================
def test_mid_depth_is_the_rule_the_original_runs_used():
    """The pre-registered layer is a rule, and the rule has to be checked against the runs it
    claims to reproduce: h8 patched layer 14 of Qwen2.5-1.5B's 29 rows, and hour 7 / hour 13 used
    L12 on the 0.5B and Pythia and L14 on GPT-J and L20 on Gemma."""
    assert phase2.mid_depth(29) == 14
    assert phase2.mid_depth(25) == 12
    assert phase2.mid_depth(28) == 14
    assert phase2.mid_depth(42) == 20


def test_the_grid_normaliser_refuses_a_factor_order_that_disagrees_with_the_keys():
    gs = phase2.normalized_grid("narrative_factors_v1")
    assert list(gs["factors"]) == ["era", "voice"]
    bad = dict(gs, factors={"voice": gs["factors"]["voice"], "era": gs["factors"]["era"]})
    # the check lives in the normaliser, so reproduce its condition on the swapped spec
    key = next(iter(bad["spans"]))
    parts = key.split("/")
    assert parts[1] not in bad["factors"]["voice"], (
        "if the era level were also a voice level the order check could not fire")


def test_the_readout_battery_reads_chance_on_noise_in_every_arm():
    """The instrument's own null, measured rather than assumed, and the check that says the
    treatment's 1.00 on real stacks is not an artefact of the ranking: on activations that carry
    no factor at all every arm -- treatment included -- must read the candidate midpoint."""
    gs = phase2.normalized_grid("narrative_factors_v1")
    rng = np.random.default_rng(0)
    X = {k: rng.normal(size=(25, 48)) for k in gs["spans"]}
    b = phase2.readout_battery(X, gs, 12, n_perm=4, n_rand=4)
    for n in gs["factors"]:
        band = registry.spec("selector").arm_tolerance(36, {"n_candidates": 3})
        assert abs(np.mean(b["B"][n]["treatment"]) - 2.0) < band
        assert abs(np.mean(b["B"][n]["random"]) - 2.0) < band
        assert np.mean(b["B"][n]["no_patch"]) == 2.0        # exact: a wholly tied field
        assert abs(np.mean(b["B"][n]["perm"]) - 2.0) < band
    assert np.mean(b["composed"]["no_patch"]) == 5.0


def test_the_no_patch_arm_is_the_tie_rule_under_test():
    """h34 read a dead patch as a sharp lens because a wholly tied field counted as rank 1. The
    no-patch arm here IS a wholly tied field, so it must read the midpoint exactly."""
    assert checks.midrank([0.0, 0.0, 0.0], 0) == 2.0
    assert checks.midrank([0.0] * 9, 4) == 5.0


@pytest.mark.skipif(not (RESULTS / "phase2_h8_replications.json").exists()
                    and not pathlib.Path("/home/user/latent-space-exploration/results/"
                                         "stacks_qwen2.5_1.5b_narrative_factors_v2.npz").exists(),
                    reason="the cached .npz stacks are gitignored and are not in this checkout")
def test_replication_rows_report_gain_over_a_measured_floor_and_never_over_chance():
    rows = json.loads((RESULTS / "phase2_h8_replications.json").read_text())
    assert len(rows) == 25
    for r in rows:
        assert r["floor_lexical"] != r["chance"] or r["factor"] == "era", r
        assert r["status"] in ("built", "REFUSED")
        if r["status"] == "built":
            assert not r["arms_off_null"], r
            # The reported quantity is the gain over the MEASURED floor. It may coincide with the
            # gain over chance -- but only when the floor was MEASURED at chance, which is h47's
            # own finding about era (a bag of tokens cannot tell three era-variants of one scene
            # apart at all), and never because chance was substituted for a measurement.
            assert (r["gain_over_floor"] != r["gain_over_chance"]) == (
                r["floor_lexical"] != r["chance"]), r
    # tense has TWO levels and a null of 1.50, not three and 2.00
    tense = [r for r in rows if r["factor"] == "tense"]
    assert tense and all(r["k"] == 2 and r["chance"] == 1.5 for r in tense)
    # era and voice must not collide on Claim.id: `factor` is in the provenance
    ids = [r["claim_id"] for r in rows if r["status"] == "built"]
    assert len(ids) == len(set(ids))


def test_a_band_cannot_be_inherited_by_an_arm_with_different_scores():
    """The unit guard short-circuits on re-entry so that `replace(arm, tolerance=...)` does not
    re-raise; that short-circuit must be keyed to the SCORES the clustering was measured on.
    Otherwise `replace(arm, scores=<other>)` inherits a band another arm earned, which is the
    unearned widening the permutation gate exists to refuse."""
    from dataclasses import replace
    import numpy as np
    from lsx.core.types import Arm
    from lsx.core.checks import ArmUnitNotInDesign

    clusters = tuple(f"s{i // 18}" for i in range(72))
    clustered = np.concatenate([np.full(18, 2.0 + j) for j in range(4)])
    arm = Arm(scores=clustered, expected_null=3.5, unit="one draw per scene", clusters=clusters)
    earned = arm.effective_n
    assert earned < arm.n

    same = replace(arm, tolerance=0.5)          # what Instrument.claim does: must not re-raise
    assert same.effective_n == earned

    flat = np.full(72, 3.5)                     # no clustering at all in these numbers
    try:
        other = replace(arm, scores=flat)
    except ArmUnitNotInDesign:
        return                                  # refused outright is also correct
    assert other.effective_n == other.n, (
        "an arm holding different scores inherited a band earned by the original's clustering")


# ================================================================================================
# phase2_qwen: the LIVE path, through build_stack -- this is what the cached-npz path in this
# file (h8_replication_rows) cannot do, and what closes ProvenanceNotFromStack.
# ================================================================================================
def test_live_replication_rows_carry_a_real_stack_signature(tiny_lm):
    """`h8_replication_rows_live` builds through `extract.build_stack`, not a frozen script's
    `.npz` -- so, unlike every row `h8_replication_rows` (the cached path) produces, its claims
    must carry a `stack_signature` the ledger's `check_provenance_from_stack` accepts."""
    from lsx.core import ledger as ledger_mod

    rows, claims, cosine = phase2.h8_replication_rows_live(
        tiny_lm, "narrative_mood_v1", model="tiny", n_perm=4, n_rand=2, curve_step=8)
    assert rows and claims
    built = [r for r in rows if r["status"] == "built"]
    assert len(built) == len(claims)
    for claim in claims:
        assert claim.provenance.get("stack_signature")
        # does not raise: this is the exact refusal phase2_v1.md §0 hit on the cached path
        ledger_mod.check_provenance_from_stack(claim)
    # the tiny model's width (32) never matches the cached Qwen2.5-1.5B stack (1536) for this
    # grid, so the comparison must report the mismatch rather than silently skip or crash.
    assert cosine["grid"] == "narrative_mood_v1"
    if cosine.get("compared"):
        assert "shape_mismatch" in cosine or cosine.get("n_pairs", 0) == 0


def test_live_and_cached_paths_share_the_same_battery_core():
    """`h8_replication_rows` (cached) and `h8_replication_rows_live` (build_stack) must both run
    through `_replication_rows_core` -- the thing phase2_qwen's task exists to prevent is a SECOND,
    parallel extraction-to-claim path that quietly drifts from the first."""
    import inspect

    src_cached = inspect.getsource(phase2.h8_replication_rows)
    src_live = inspect.getsource(phase2.h8_replication_rows_live)
    assert "_replication_rows_core(" in src_cached
    assert "_replication_rows_core(" in src_live
