"""Piece 2: the instrument registry (§8), the calibration battery (§5) and the arm contract (§4).

The rule this file is written to: every number the registry DECLARES is checked against the
statistic it describes. A declared null that no test compares to the instrument is a declaration,
not a measurement, and this project has already published three of those.
"""
import numpy as np
import pytest

from lsx.core import checks, instruments, registry
from lsx.core.types import (Arm, Claim, Direction, EffectSize, Floor, Measured, Selection,
                            fit_leave_one_out)


# --------------------------------------------------------------------------------------------
# the registry declares, the statistic confirms
# --------------------------------------------------------------------------------------------
def test_every_instrument_declares_arms_a_null_and_a_null_spread():
    for name, spec in registry.REGISTRY.items():
        assert spec.null_doc and spec.null_sd_doc, name
        assert isinstance(spec.required_arms, tuple), name
        if spec.implemented:
            assert not np.isnan(spec.null_value()), name
            assert spec.null_item_sd({}) > 0, name


@pytest.mark.parametrize("name,cfg", [("selector", {"n_candidates": 3}),
                                      ("selector", {"n_candidates": 6}),
                                      ("composition", {"n_variants": 18})])
def test_declared_null_item_sd_matches_the_statistic(name, cfg):
    """The closed form in the registry against a Monte-Carlo draw from the real statistic."""
    kw = {"n_candidates": cfg["n_candidates"]} if name == "selector" else {"levels": (3, 3, 2)}
    inst = instruments.build(name, n=4000, **kw)
    measured = inst.measure_null_item_sd()
    declared = registry.spec(name).null_item_sd(inst.config)
    assert abs(measured - declared) / declared < 0.05, (name, cfg, measured, declared)


def test_arm_tolerance_is_per_instrument_and_per_arm_size():
    sel = registry.spec("selector")
    # h4's shape: 6 candidates over 40 items. Piece 1's flat 0.15 is FOUR TIMES too tight here...
    assert sel.arm_tolerance(40, {"n_candidates": 6}) == pytest.approx(0.810, abs=0.01)
    # ...and at h8's per-factor shape over 400 items it is too loose.
    assert sel.arm_tolerance(400, {"n_candidates": 3}) == pytest.approx(0.122, abs=0.01)
    assert (registry.spec("composition").arm_tolerance(40, {"n_variants": 18})
            > sel.arm_tolerance(40, {"n_candidates": 6}))


def test_a_clean_arm_at_h4_size_would_have_fired_piece_1s_flat_tolerance():
    """Not an opinion: draw clean null arms of h4's shape and count how often 0.15 fires."""
    rng = np.random.default_rng(0)
    means = [np.mean(rng.integers(1, 7, size=40)) for _ in range(2000)]
    flat = np.mean([abs(m - 3.5) > 0.15 for m in means])
    measured = np.mean([abs(m - 3.5) > registry.spec("selector").arm_tolerance(
        40, {"n_candidates": 6}) for m in means])
    assert flat > 0.4, flat            # piece 1's default fires on ~half of all clean arms
    assert measured < 0.01, measured   # the measured 3-sigma band, on ~none


# --------------------------------------------------------------------------------------------
# calibration (§5)
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    d = tmp_path_factory.mktemp("calib")
    return {n: instruments.build(n).calibrate(cache_dir=d, refresh=True) for n in instruments.BUILDERS}


@pytest.mark.parametrize("name", ["selector", "composition", "readout_shift"])
def test_all_three_instruments_pass_the_full_battery(reports, name):
    r = reports[name]
    assert r.passed, r.failures
    assert set(r.tests_run) == {"noise", "sensitivity", "degeneracy", "self_floor", "invariance",
                                "known_zero"}
    assert not r.hand_declared_report


@pytest.mark.parametrize("name", ["selector", "composition", "readout_shift"])
def test_sensitivity_is_monotone_in_the_planted_size(reports, name):
    v = np.asarray(reports[name].sensitivity_values)
    d = np.diff(v)
    assert np.all(d <= 1e-12) or np.all(d >= -1e-12), v
    assert abs(v[-1] - v[0]) > 0, v


def test_readout_shift_is_calibrated_in_the_literal_sense(reports):
    """Its sensitivity values ARE the planted amplitudes: the gain is in units of the planted
    effect, to under 1%."""
    r = reports["readout_shift"]
    assert np.allclose(r.sensitivity_values, r.sensitivity_amplitudes, rtol=0.01)


def test_known_zero_points_are_exact(reports):
    assert reports["selector"].known_zero_value == 3.5      # every candidate tied -> mid-rank
    assert reports["composition"].known_zero_value == 9.5
    assert reports["readout_shift"].known_zero_value == 0.0


def test_claimed_invariances_hold_to_float_error(reports):
    for name in ("selector", "composition", "readout_shift"):
        for r in reports[name].invariance:
            if r.claimed:
                assert r.delta < 1e-6, (name, r)


def test_the_battery_refuses_the_rejected_cosine_variant():
    """The cosine form of readout_shift reads -0.13 on its own null, because orthogonal block work
    dilutes a cosine. Calibration is what found it; this asserts it still would."""
    rs = instruments.build("readout_shift")
    bad = checks.run_calibration(
        instruments.cosine_readout_gain, declared_null=0.0, plant=rs.plant, name="readout_shift",
        amplitudes=rs.amplitudes, null_tol=rs.resolved_null_tol, noise_fn=rs.noise_fn,
        transforms=rs.transforms, self_floor=rs.self_floor, known_zero=rs.known_zero,
        invariances=("scale", "rotation"))
    assert bad.noise_failed and not bad.passed
    assert bad.noise_value < -0.1
    assert not bad.sensitivity_failed, "and it fails on the NOISE test, not the sensitivity one"


def test_a_statistic_pinned_at_its_null_is_flagged_degenerate():
    rep = checks.run_calibration(lambda x: 2.0, shape=(64, 3), declared_null=2.0,
                                 plant=lambda x, a: x + a, name="pinned")
    assert rep.degenerate and not rep.passed
    assert not rep.noise_failed, "its null is perfect; that is the whole point of h6"


# --------------------------------------------------------------------------------------------
# the cache: editing ONE instrument re-calibrates that one alone
# --------------------------------------------------------------------------------------------
def test_cache_is_keyed_by_the_instrument_and_hits_on_the_second_call(tmp_path):
    inst = instruments.build("selector")
    first = inst.calibrate(cache_dir=tmp_path, refresh=True)
    path = checks.calibration_path("selector", inst.key, tmp_path)
    assert path.exists()
    import json
    d = json.loads(path.read_text())
    d["noise_value"] = 99.0                   # tamper, so a cache HIT is visible
    path.write_text(json.dumps(d))
    again = instruments.build("selector").calibrate(cache_dir=tmp_path)
    assert again.key == first.key
    assert again.noise_value == 99.0, "a second calibrate() must be a cache hit, not a re-run"
    assert instruments.build("selector").calibrate(
        cache_dir=tmp_path, refresh=True).noise_value == first.noise_value
    # a DIFFERENT instrument's cache is untouched by any of this
    comp = instruments.build("composition")
    comp.calibrate(cache_dir=tmp_path, refresh=True)
    assert checks.calibration_path("composition", comp.key, tmp_path).exists()
    assert checks.load_calibration("selector", "0" * 16, tmp_path) is None


def test_the_key_changes_with_the_null_and_the_invariances_only():
    a = instruments.build("selector", n_candidates=6).key
    b = instruments.build("selector", n_candidates=3).key      # different declared null
    c = instruments.build("composition").key
    assert a != b and a != c
    assert instruments.build("selector", n_candidates=6, n=90).key == a, \
        "the fixture size is not part of the identity of the statistic"


def test_a_stale_measured_report_blocks_claim_construction(tmp_path):
    inst = instruments.build("selector", n_candidates=6)
    good = inst.calibrate(cache_dir=tmp_path, refresh=True)
    stale = checks.CalibrationReport.from_dict(dict(good.to_dict(), key="0" * 16))
    with pytest.raises(checks.CalibrationStale):
        Claim(instrument="selector", treatment=Measured(np.full(40, 2.2)),
              arms={"random": Arm(np.full(40, 3.5), expected_null=3.5),
                    "no_patch": Arm(np.full(40, 3.5), expected_null=3.5),
                    "permutation": Arm(np.full(40, 3.5), expected_null=3.5)},
              floor=Floor(stimulus=3.5, estimator=3.5),
              selection=Selection(axis=None, rule="pre-registered", held_out=True),
              effect=EffectSize.against(np.full(40, 2.2), 3.5, fallback_item_sd=1.71),
              calibration=stale, provenance={}, config={"n_candidates": 6})


# --------------------------------------------------------------------------------------------
# the computed pass-through arm (§2a)
# --------------------------------------------------------------------------------------------
def _shift_case(n=18, d=64, seed=3):
    rng = np.random.default_rng(seed)
    direction = instruments.unit(rng.normal(size=d))
    base = rng.normal(size=(n, d))
    shift = 0.9 * float(np.linalg.norm(base, axis=-1).mean()) * direction

    def readout(resid):
        return (resid @ direction) / np.linalg.norm(base, axis=-1)

    return base, shift, readout


def test_passthrough_arm_reproduces_the_unpatched_readout_exactly_at_zero_shift():
    base, shift, readout = _shift_case()
    arm = instruments.PassthroughArm.compute(base=base, shift=shift, readout=readout,
                                             expected_null=0.9, justification="norm-matched")
    assert arm.zero_shift_error == 0.0 and arm.computed


def test_a_passthrough_arm_whose_readout_differs_is_the_arm_being_wrong():
    base, shift, readout = _shift_case()
    with pytest.raises(checks.PassthroughNotReproduced):
        instruments.PassthroughArm.compute(
            base=base, shift=shift, readout=lambda r: readout(r) + 1e-12,
            unpatched=readout(base), expected_null=0.9, justification="norm-matched")


def test_a_declared_passthrough_arm_is_refused_where_piece_1_accepted_it():
    base, shift, readout = _shift_case()
    rs = instruments.build("readout_shift", d=64)
    kw = dict(instrument="readout_shift",
              treatment=Measured(np.zeros(18), label="gain"),
              floor=Floor(stimulus=0.5, estimator=0.0),
              selection=Selection(axis=None, rule="pre-registered pair", held_out=True),
              effect=EffectSize.against(np.zeros(18), 0.0, fallback_item_sd=0.18),
              calibration=rs.calibration, provenance={}, config={"d": 64},
              patch_layer=14, readout_layer=20)
    plumbing = {"random": Arm(np.zeros(18), expected_null=0.0),
                "no_patch": Arm(np.zeros(18), expected_null=0.0)}
    with pytest.raises(checks.PassthroughNotComputed):
        Claim(arms=dict(plumbing, passthrough=Arm(np.full(18, 0.9), expected_null=0.9)), **kw)
    arm = instruments.PassthroughArm.compute(base=base, shift=shift, readout=readout,
                                             expected_null=float(np.mean(readout(base + shift))),
                                             justification="the arithmetic reproduces it")
    Claim(arms=dict(plumbing, passthrough=arm), **kw)     # constructs


def test_gain_is_a_difference_and_never_a_ratio():
    src = (instruments.readout_shift_gain.__doc__ or "") + "".join(
        l for l in open(instruments.__file__).read().splitlines()
        if "passthrough" in l and "/" in l and "#" not in l)
    assert "DIFFERENCE" in instruments.readout_shift_gain.__doc__
    f = instruments.build("readout_shift").noise_fn(np.random.default_rng(0))
    near_ceiling = instruments.readout_shift_gain(f)
    assert abs(near_ceiling) < 1e-9, "no gain when the blocks add nothing along the direction"


# --------------------------------------------------------------------------------------------
# effect size, held-out axes, executed sweeps
# --------------------------------------------------------------------------------------------
def test_effect_size_against_the_null_is_computed_and_stamped():
    e = EffectSize.against(np.full(40, 2.0) + np.random.default_rng(0).normal(0, 1, 40), 3.5)
    assert e.computed and e.against_null == 3.5 and e.verdict == -1


def test_a_fabricated_effect_size_is_refused():
    inst = instruments.build("selector", n_candidates=6)
    kw = dict(instrument="selector", treatment=Measured(np.full(40, 3.52)),
              arms={"random": Arm(np.full(40, 3.5), expected_null=3.5),
                    "no_patch": Arm(np.full(40, 3.5), expected_null=3.5),
                    "permutation": Arm(np.full(40, 3.5), expected_null=3.5)},
              floor=Floor(stimulus=3.5, estimator=3.5),
              selection=Selection(axis=None, rule="pre-registered layer", held_out=True),
              calibration=inst.calibration, provenance={}, config={"n_candidates": 6})
    with pytest.raises(checks.EffectSizeUnverified):
        Claim(effect=EffectSize(size=-1.4, n=40, z=-6.0), **kw)       # a flat null arm, claimed huge
    Claim(effect=EffectSize(size=0.02, n=40, z=0.1), **kw)            # an honest nothing


def test_held_out_axis_is_verified_against_what_the_fit_actually_used():
    vecs = {f"domain{i}": np.random.default_rng(i).normal(size=8) for i in range(4)}
    d = fit_leave_one_out(vecs, unseen={"domain3"}, axis="domain")
    assert d.verified and "domain3" not in d.fit_witness["fit_values"]
    with pytest.raises(checks.HeldOutViolated):
        Direction(vecs={0: np.zeros(8)}, held_out={"axis": "domain", "unseen": {"domain3"}},
                  fit_witness={"axis": "domain", "fit_values": ["domain0", "domain3"]})
    with pytest.raises(checks.HeldOutViolated):
        Direction(vecs={0: np.zeros(8)}, held_out={"axis": "layer", "unseen": {"domain3"}},
                  fit_witness={"axis": "domain", "fit_values": ["domain0"]})


def test_a_sweep_executed_by_the_core_is_not_a_choice():
    inst = instruments.build("selector")
    sel = inst.sweep("layer", [10, 12, 14], lambda l: 3.5 - l / 100)
    assert sel.executed and sel.held_out and not sel.is_a_choice
    assert len(sel.curve) == 3
    # the prose path still refuses an argmax, and still cannot see through a story
    assert Selection(axis="layer", rule="argmax over layers").is_a_choice
    assert not Selection(axis="layer",
                         rule="we looked at the curve and quoted layer 16").is_a_choice


# --------------------------------------------------------------------------------------------
# the instrument as the path to a Claim
# --------------------------------------------------------------------------------------------
def test_instrument_claim_takes_its_nulls_and_tolerances_from_the_registry():
    inst = instruments.build("composition", levels=(3, 3, 2))
    rng = np.random.default_rng(0)
    claim = inst.claim(treatment=rng.normal(2.81, 1.0, 40),
                       arms={"random": rng.normal(9.5, 1.0, 40),
                             "no_patch": rng.normal(9.5, 1.0, 40)},
                       floor=Floor(stimulus=9.5, estimator=9.5),
                       selection=Selection(axis=None, rule="pre-registered layer", held_out=True),
                       provenance={"model": "qwen2.5-1.5b"}, stage="h8")
    assert claim.arms["no_patch"].expected_null == 9.5                    # registry, not caller
    assert claim.arms["random"].resolved_tolerance == pytest.approx(
        registry.spec("composition").arm_tolerance(40, claim.config))
    assert claim.effect.computed and claim.effect.verdict == -1
    assert claim.to_row()["provenance"]["calibration_key"] == inst.key
    assert "treatment" in claim.render() or "gain" in claim.render()


def test_an_unknown_instrument_has_no_claim_and_an_undeclared_null_has_no_instrument():
    with pytest.raises(registry.UnknownInstrument):
        Claim(instrument="vibes", treatment=Measured([1.0]), arms={},
              floor=Floor(stimulus=0, estimator=0),
              selection=Selection(axis=None, rule="x", held_out=True),
              effect=EffectSize(0, 1, 0),
              calibration=checks.CalibrationReport.hand_declared("vibes", True), provenance={})
    with pytest.raises(registry.InstrumentNotImplemented):
        Claim(instrument="generality", treatment=Measured([0.18]), arms={},
              floor=Floor(stimulus=0.06, estimator=0.06),
              selection=Selection(axis=None, rule="pre-registered", held_out=True),
              effect=EffectSize(0.12, 1, 2.0),
              calibration=checks.CalibrationReport.hand_declared("generality", True), provenance={})
    with pytest.raises(registry.InstrumentNotImplemented):
        instruments.build("depth_gain")
