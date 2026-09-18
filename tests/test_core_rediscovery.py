"""The rediscovery harness, run as a test (spec §1B).

`tests/test_invariants.py` asserts properties of the measurement machinery. This file asserts
something narrower and harder: that the CORE REFUSES the configurations this project has already
shipped. A core that cannot rediscover our own bugs has not earned trust.

Each case is also checked for the *mechanism* that fired, because being refused for the wrong
reason is how a check rots: it keeps passing after the thing it protected has moved.
"""
import numpy as np
import pytest

from lsx.core import checks, rediscovery
from lsx.core.extract import build_stack, capture_residual
from lsx.core.types import (Arm, Claim, Direction, EffectSize, Floor, Grid, Item, Measured,
                            Selection, Sketch)


@pytest.fixture(scope="module")
def verdicts(tiny_lm):
    return {v.bug: v for v in rediscovery.run_all(tiny_lm)}


# --- piece 1's contractual minimum: bugs 1, 2, 7 and 8 ----------------------------------------
@pytest.mark.parametrize("bug", [1, 2, 7, 8])
def test_piece1_required_bugs_are_caught(verdicts, bug):
    v = verdicts[bug]
    assert v.ok, f"bug {bug} not caught: {v.render()}"


def test_every_wired_case_is_caught(verdicts):
    missed = [v.render() for v in verdicts.values() if not v.ok]
    assert not missed, "\n".join(missed)


@pytest.mark.parametrize("bug,mechanism", [
    (1, "MovedCandidates"), (2, "BatchEquivalence"), (3, "MissingArm"), (4, "SelectionOnScoringData"),
    (5, "MissingFloor"), (6, "CalibrationFailed"), (7, "RawScoreOnLeakyGrid"),
    (7.5, "PostNormResidual"), (8, "MissingArm"),
    # piece 4: the three PUBLICATION refusals, which piece 3 added and tested directly rather than
    # through the harness -- and said so in its handover. §1B's standard is "refuses without being
    # told what to look for", and a test that names the exception it wants is being told.
    (9, "HandDeclaredCalibration"), (10, "ProvenanceNotFromStack"), (11, "SweepNotExecuted"),
])
def test_the_named_mechanism_fired(verdicts, bug, mechanism):
    assert mechanism in verdicts[bug].mechanism, verdicts[bug].render()


def test_positive_controls_pass(verdicts):
    """The harness must not be satisfiable by a core that refuses everything."""
    failed = [v.render() for v in verdicts.values() if "FAILED" in v.positive_control]
    assert not failed, "\n".join(failed)


# --- the equivalence check's own premise (spec §7.2) ------------------------------------------
def test_shortest_item_is_the_one_that_fails_first(verdicts):
    """Under the h39 bug the longest item in each batch is clean. Checking a random item would
    have passed; checking the shortest is what fails."""
    cos = verdicts[2].extra["cosines"]
    assert max(cos) > 0.999, "no item was clean; the reconstruction is not the h39 shape"
    assert min(cos) < 0.999, "no item was corrupted; the reconstruction did not reproduce the bug"
    assert checks.shortest_item_index([5, 2, 9]) == 1


# --- the one extraction path (spec §7) --------------------------------------------------------
def _grid():
    items = []
    for i, body in enumerate(["the thesis is that a particle has a definite position",
                              "measurement disturbs what it measures",
                              "position and momentum are complementary descriptions",
                              "the muppets take manhattan"]):
        items.append(Item(text=body, factors={"kind": "a" if i % 2 == 0 else "b"},
                          spans={"whole": (0, len(body))}))
    return Grid(items, name="tiny", leak_check=False)


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_batched_extraction_equals_single(tiny_lm, padding_side):
    """The assertion that would have caught h39, run in the clean configuration: it must PASS, and
    pass at every padding side, or the core is unusable for the thing it protects."""
    old = tiny_lm.tok.padding_side
    tiny_lm.tok.padding_side = padding_side
    try:
        st = build_stack(tiny_lm, _grid(), batch_size=4)
    finally:
        tiny_lm.tok.padding_side = old
    assert st.acts.shape == (4, 1, tiny_lm.n_layers + 1, tiny_lm.d_model)
    assert min(st.checks["batched_vs_single_min_cos"].values()) >= 0.999
    assert st.provenance["tokenizer_padding"] == padding_side


def test_stack_provenance_is_complete(tiny_lm):
    st = build_stack(tiny_lm, _grid(), batch_size=2)
    for k in ("model", "layers", "pooling", "grid_hash", "code_version", "tokenizer_padding",
              "lib_versions", "template"):
        assert k in st.provenance
    assert st.provenance["lib_versions"]["transformers"]
    checks.assert_provenance(st.provenance)


def test_empty_span_is_refused():
    with pytest.raises(checks.EmptySpan):
        checks.assert_nonempty_spans([], [1, 1, 1], item=0, span="s")
    with pytest.raises(checks.EmptySpan):
        checks.assert_nonempty_spans([0, 1], [0, 1, 1], item=0, span="s")


def test_resid_helper_refuses_to_index_blindly():
    import torch
    t = torch.zeros(2, 3, 4)
    assert checks.resid(t) is t
    assert checks.resid((t, None)) is t
    with pytest.raises(checks.LayerOutputShape):
        checks.resid({"hidden_states": t})


def test_capture_residual_at_final_layer_is_pre_norm(tiny_lm):
    """h40: `residuals()[-1]` is the final-norm OUTPUT. The sanctioned capture must not return it."""
    text = "the muppets take manhattan"
    got = capture_residual(tiny_lm, text, layer=tiny_lm.n_layers)
    hs, _ = tiny_lm.residuals(text)
    assert not np.allclose(got, hs[-1].numpy(), atol=1e-4)
    with pytest.raises(checks.PostNormResidual):
        checks.assert_pre_norm(tiny_lm, hs[-1].numpy(), layer=tiny_lm.n_layers)
    checks.assert_pre_norm(tiny_lm, got, layer=tiny_lm.n_layers)     # must NOT raise


# --- the types (spec §3, §4) -------------------------------------------------------------------
def test_sketch_is_unledgerable_and_prints():
    s = Sketch([1.0, 2.0, 3.0], "exploring")
    assert s.ledgerable is False
    assert "NOT a result" in repr(s)
    assert isinstance(s.measured(), Measured)


def test_direction_requires_a_declared_held_out_axis():
    with pytest.raises(checks.HeldOutNotDeclared):
        Direction(vecs={0: np.zeros(4)}, held_out={})
    Direction(vecs={0: np.zeros(4)}, held_out={"axis": "domain", "unseen": {"d8"}})


def test_arm_must_declare_where_it_should_sit():
    with pytest.raises(checks.ArmOffNull):
        Arm(np.ones(3))


def _ok_claim(**kw):
    base = dict(instrument="selector", treatment=Measured(np.full(10, 1.25), label="era"),
                arms={"random": Arm(np.full(10, 2.0), expected_null=2.0),
                      "no_patch": Arm(np.full(10, 2.0), expected_null=2.0),
                      "permutation": Arm(np.full(10, 2.0), expected_null=2.0)},
                floor=Floor(stimulus=2.0, estimator=2.0),
                selection=Selection(axis="layer", rule="full curve reported", held_out=True),
                effect=EffectSize(size=-0.75, n=10, z=-4.0),
                calibration=checks.CalibrationReport.hand_declared("selector", passed=True),
                provenance={"model": "tiny"})
    base.update(kw)
    return Claim(**base)


def test_claim_renders_everything_together():
    """There is no way to quote a treatment number alone (spec §4)."""
    c = _ok_claim()
    out = c.render()
    for bit in ("treatment", "arm random", "arm no_patch", "arm permutation", "floor", "effect",
                "selection", "calibration", c.id):
        assert bit in out


def test_claim_refuses_a_treatment_that_was_never_stamped():
    with pytest.raises(TypeError):
        _ok_claim(treatment=Sketch(np.full(10, 1.25)))


def test_semantic_null_must_precede_the_treatment():
    """Construction order, not discipline: the null is a real arm declared before any treatment
    score exists (spec §4)."""
    treatment_scored_first = Measured(np.full(10, 1.25))
    null_declared_second = Arm(np.full(10, 1.37), expected_null=1.37,
                               justification="role identity retained")
    with pytest.raises(checks.NullDeclaredLate):
        _ok_claim(treatment=treatment_scored_first, semantic_null=null_declared_second)
    # the right order: the null exists before the treatment is scored
    null_first = Arm(np.full(10, 1.37), expected_null=1.37, justification="role identity retained")
    _ok_claim(treatment=Measured(np.full(10, 1.25)), semantic_null=null_first)


def test_semantic_null_needs_a_justification():
    null = Arm(np.full(10, 1.37), expected_null=1.37)
    with pytest.raises(checks.NullDeclaredLate):
        _ok_claim(treatment=Measured(np.full(10, 1.25)), semantic_null=null)


def test_ledger_row_carries_every_arm_and_both_floors():
    row = _ok_claim().to_row()
    assert set(row["arms"]) == {"random", "no_patch", "permutation"}
    assert row["floor"] == {"stimulus": 2.0, "estimator": 2.0}
    assert row["provenance"]["calibration_key"]
    assert row["status"] == "standing"


def test_missing_calibration_blocks_construction():
    with pytest.raises(checks.MissingCalibration):
        _ok_claim(calibration=None)


def test_calibration_key_changes_with_the_statistic_only():
    def f(x):
        return float(x.mean())

    def g(x):
        return float(x.mean()) + 1

    assert checks.calibration_key(f, 0.0) == checks.calibration_key(f, 0.0)
    assert checks.calibration_key(f, 0.0) != checks.calibration_key(g, 0.0)
    assert checks.calibration_key(f, 0.0) != checks.calibration_key(f, 1.0)


def test_midrank_reads_a_tied_field_as_chance():
    assert checks.midrank([0.0] * 9, 0) == 5.0
    assert checks.midrank([1.0] + [0.0] * 8, 0) == 1.0


def test_leak_report_is_computed_at_construction_and_cached_by_hash():
    leaky = rediscovery._leaky_grid()
    assert leaky.leak.leaky and "interval" in leaky.leak.flagged
    assert leaky.hash and leaky.hash != rediscovery._clean_grid().hash
    assert not rediscovery._clean_grid().leak.leaky


# --- piece 4: the ledger is covered by the harness, not only by direct tests -------------------
def test_the_harness_covers_the_three_publication_refusals(verdicts):
    """Piece 3 closed with "there is no harness case for these, and there should be"."""
    for bug in (9, 10, 11):
        assert verdicts[bug].ok, verdicts[bug].render()
        assert "Ledger.append" in verdicts[bug].mechanism, verdicts[bug].render()


def test_the_harness_never_writes_to_the_real_ledger(tmp_path):
    """A harness case is a demonstration that a mechanism fires, not a result. If one of these ever
    appended to `results/ledger.jsonl` it would be a fabricated row, which is the one thing the
    ledger exists to make impossible."""
    from lsx.core import ledger
    real = ledger.DEFAULT_PATH
    before = real.read_text() if real.exists() else ""
    for bug in (9, 10, 11):
        rediscovery.PURE_CASES[bug]()
    after = real.read_text() if real.exists() else ""
    assert before == after


def test_a_swept_claim_with_a_curve_publishes_and_one_with_prose_does_not():
    """The positive control of case 11, asserted here too: the refusal must cost exactly the prose
    path and nothing else."""
    v = rediscovery.PURE_CASES[11]()
    assert "FAILED" not in v.positive_control, v.render()
    assert "curve publishes" in v.positive_control
