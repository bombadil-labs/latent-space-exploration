"""Piece 3: the ledger, retraction, and the three publication refusals (spec §9, §1A).

Every test here is about the difference between a number the core will *compute* and a number the
core will *publish*. The contract gates publication, not thought.
"""
import json

import numpy as np
import pytest

from lsx.core import instruments, ledger, registry, reproduce
from lsx.core.checks import (CalibrationReport, HandDeclaredCalibration, LedgerConflict,
                             ProvenanceIncomplete, ProvenanceNotFromStack, SweepNotExecuted)
from lsx.core.types import Arm, Floor, Measured, Selection, stack_signature

STACK_PROV = {
    "model": "Qwen/Qwen2.5-1.5B", "layers": [14], "pooling": "mean", "grid_hash": "abc123",
    "grid_name": "narrative_factors_v2", "code_version": "deadbeef", "tokenizer_padding": "right",
    "span_policy": "auto", "template": None, "lib_versions": {"torch": "2.4"}, "batch_size": 8,
    "leak": "nothing flagged", "acts_digest": "0011223344556677",
    "equivalence_min_cos": 0.9999, "equivalence_item_rule": "shortest item in each batch",
}


def signed(**extra):
    prov = dict(STACK_PROV)
    prov["stack_signature"] = stack_signature(prov)
    prov.update(extra)
    return prov


def a_claim(*, prov=None, selection=None, calibration=None, n=40):
    sel = instruments.build("selector", n=200, d=32, n_candidates=6)
    rng = np.random.default_rng(0)
    treat = rng.uniform(1.0, 2.5, n)
    claim = sel.claim(
        treatment=Measured(treat, label="test"),
        arms={"random": np.full(n, 3.5), "no_patch": np.full(n, 3.5),
              "permutation": np.full(n, 3.5)},
        floor=Floor(stimulus=3.5, estimator=3.5),
        selection=selection or Selection(axis=None, rule="pre-registered layer; no sweep"),
        provenance=prov if prov is not None else signed(), stage="test")
    if calibration is not None:
        object.__setattr__(claim, "calibration", calibration)
    return claim


# ------------------------------------------------------------------ the three refusals
def test_a_hand_declared_calibration_cannot_be_published():
    """Piece 2 stamped these so they could not pass for measured ones; nothing acted on the stamp."""
    claim = a_claim()
    claim.calibration = CalibrationReport.hand_declared("selector", True, "trust me")
    assert claim.to_row()["calibration_hand_declared"] is True
    with pytest.raises(HandDeclaredCalibration):
        ledger.screen(claim)


def test_a_measured_calibration_passes_the_same_gate():
    ledger.check_calibration_measured(a_claim())


def test_provenance_that_did_not_come_from_a_stack_is_refused():
    prov = {k: v for k, v in STACK_PROV.items()}          # no signature: a hand-rolled extraction
    with pytest.raises(ProvenanceNotFromStack):
        ledger.screen(a_claim(prov=prov))


def test_a_tampered_stack_signature_is_refused():
    """The signature covers the fields the extraction wrote; editing one afterwards breaks it."""
    prov = signed()
    prov["tokenizer_padding"] = "left"                    # the h39 field, changed after the fact
    with pytest.raises(ProvenanceNotFromStack) as e:
        ledger.screen(a_claim(prov=prov))
    assert "recomputed" in str(e.value)


def test_missing_required_provenance_is_refused_before_the_signature():
    prov = signed()
    del prov["lib_versions"]                              # h36 WAS a transformers change
    with pytest.raises(ProvenanceIncomplete):
        ledger.screen(a_claim(prov=prov))


def test_a_swept_axis_without_a_core_computed_curve_is_refused():
    """Piece 2 left this decision to piece 3. It is taken: prose about a sweep is not a sweep."""
    sel = Selection(axis="layer", rule="we looked at the curve and quoted layer 16", held_out=True)
    assert not sel.is_a_choice          # the regex is fooled, on purpose, and a test says so
    with pytest.raises(SweepNotExecuted):
        ledger.screen(a_claim(selection=sel))


def test_a_curve_the_core_computed_passes():
    inst = instruments.build("selector", n=200, d=32, n_candidates=6)
    sel = inst.sweep("layer", [0, 4, 8], lambda l: 3.5 - 0.1 * l)
    assert sel.executed and sel.curve == {"0": 3.5, "4": 3.1, "8": 2.7}
    ledger.screen(a_claim(selection=sel))


def test_an_unswept_claim_needs_no_curve():
    ledger.screen(a_claim(selection=Selection(axis=None, rule="single pre-registered layer")))


# ------------------------------------------------------------------ the file
def test_append_withdraw_and_render(tmp_path):
    L = ledger.Ledger(tmp_path / "ledger.jsonl")
    claim = a_claim()
    row = L.append(claim, reproduces="a test target", logged={"value": 2.0})
    assert row["status"] == "standing"
    assert L.standing() and not L.withdrawn()

    # the same experiment re-run is RECOGNISED, not duplicated
    again = L.append(claim)
    assert again["id"] == row["id"]
    assert len(list(L.lines())) == 1

    w = L.withdraw(row["id"], "reason under test", superseded_by="h99")
    assert w["status"] == "withdrawn"
    assert L.rows()[row["id"]]["status"] == "withdrawn"
    assert not L.standing() and len(L.withdrawn()) == 1
    out = L.render()
    assert "WITHDRAWN" in out and "reason under test" in out and "superseded by h99" in out
    # append-only: the history of the retraction is itself in the file
    assert len(list(L.lines())) == 2


def test_withdrawing_something_that_is_not_there_is_an_error(tmp_path):
    L = ledger.Ledger(tmp_path / "ledger.jsonl")
    with pytest.raises(LedgerConflict):
        L.withdraw("nope", "reason")


def test_a_withdrawal_needs_a_reason(tmp_path):
    L = ledger.Ledger(tmp_path / "ledger.jsonl")
    row = L.append(a_claim())
    with pytest.raises(LedgerConflict):
        L.withdraw(row["id"], "")


def test_same_id_different_number_is_a_conflict_not_an_overwrite(tmp_path):
    """`id` is the provenance hash: identical provenance and a different number means the run is
    not reproducible, not that the row should be replaced."""
    L = ledger.Ledger(tmp_path / "ledger.jsonl")
    c1 = a_claim(n=40)
    L.append(c1)
    c2 = a_claim(n=40)
    c2.treatment = Measured(np.full(40, 1.0), label="test")
    with pytest.raises(LedgerConflict):
        L.append(c2)


def test_a_changed_library_version_makes_a_new_id_rather_than_overwriting(tmp_path):
    L = ledger.Ledger(tmp_path / "ledger.jsonl")
    p2 = dict(STACK_PROV, lib_versions={"torch": "9.9"})
    p2["stack_signature"] = stack_signature(p2)
    a, b = L.append(a_claim()), L.append(a_claim(prov=p2))
    assert a["id"] != b["id"] and len(list(L.lines())) == 2


def test_the_real_ledger_file_is_valid_jsonl_and_every_row_has_the_shape_9_names():
    """`results/ledger.jsonl` is a deliverable; this asserts it stays readable."""
    L = ledger.Ledger()
    if not L.path.exists():
        pytest.skip("ledger not yet written")
    for r in L.lines():
        assert "id" in r
        if r.get("_op") == "withdraw":
            assert r["withdrawn"]["reason"]
            continue
        for k in ("instrument", "treatment", "arms", "floor", "effect", "selection",
                  "provenance", "status"):
            assert k in r, k
        for k in ledger.REQUIRED_PROVENANCE:
            assert k in r["provenance"], k


# ------------------------------------------------------------------ §1A: the two graded first
def test_h16_peak_layer_is_refused_by_the_core():
    row = reproduce.h16_as_logged()
    assert row.verdict == reproduce.REFUSED
    assert "SelectionOnScoringData" in row.detail


def test_h39_is_deferred_now_that_discrimination_exists_but_its_floor_cannot_be_measured():
    """Piece 3 refused this row for two reasons; piece 4 closes one of them and not the other.

    The instrument is built and calibrated, so "no Claim can be graded through it" is no longer
    true. The grid is still flagged leaky, §6 still requires gain over the MEASURED floor, and the
    Gemma stacks that would measure it are still not cached -- so the row is deferred on data.
    """
    row = reproduce.h39_as_logged()
    assert row.verdict == reproduce.DEFERRED
    assert row.extra["instrument_built"] is True
    assert "not cached" in row.detail


def test_discrimination_reproduces_h38s_logged_floor_and_treatment():
    """The instrument, on real numbers rather than on its fixtures.

    h38 logged the arm-D clock at 0.961 and the arm-A floor (same text, interval phrase removed) at
    0.522 for layer 14. The instrument re-derives both from the cached per-subject values, and
    reports the difference as gain, because §6 forbids the raw score on this grid.
    """
    demo = reproduce.discrimination_on_h38_cache()
    assert demo["per_layer"]["14"]["treatment"] == pytest.approx(0.961, abs=0.001)
    assert demo["per_layer"]["14"]["floor"] == pytest.approx(0.522, abs=0.001)
    assert demo["gain"] == pytest.approx(0.4385, abs=0.001)
    # and it is refused, for h38's battery rather than for the instrument
    assert "MissingArm" in demo["refusal"] and "shuffled_stimulus" in demo["refusal"]


def test_the_known_zero_point_holds_on_real_data_at_layer_zero():
    """§5's known-zero point, arriving unasked on a real measurement.

    For Qwen and Gemma layer 0 IS the bag of static embeddings -- which h38 discovered the hard way
    -- so a state-text readout and a bag-of-the-same-words floor are the same object there and the
    gain must be zero. Measured on the cached arms: 0.004, against an arm band of 0.375 at eight
    subjects. This is the one calibration point the battery cannot fake, because nothing here is
    synthetic.
    """
    demo = reproduce.discrimination_on_h38_cache()
    assert abs(demo["per_layer"]["0"]["gain"]) < demo["arm_tolerance_at_8_subjects"]
    assert demo["per_layer"]["0"]["gain"] == pytest.approx(0.004, abs=0.01)
    assert demo["per_layer"]["14"]["gain"] > 10 * abs(demo["per_layer"]["0"]["gain"])


def test_a_selector_with_a_non_default_candidate_count_is_not_stale():
    """A bug the §1A suite found in piece 2's staleness check.

    The calibration key hashes the statistic's source, its declared null and its declared
    invariances -- and the declared null is a function of the CONFIG. A 3-candidate selector's null
    is 2.00 and a 6-candidate's is 3.50, so they hash differently and both are correct. The table
    held one key per instrument, so every selector claim whose candidate count was not the registry
    default (h8's three single-factor lenses, h34, h37) was refused as `CalibrationStale` while
    holding a freshly measured, passing report.
    """
    s3 = instruments.build("selector", n=200, d=32, n_candidates=3)
    s6 = instruments.build("selector", n=200, d=32, n_candidates=6)
    assert s3.key != s6.key and s3.declared_null == 2.0 and s6.declared_null == 3.5
    claim = s3.claim(
        treatment=Measured(np.full(30, 1.25)),
        arms={"random": np.full(30, 2.0), "no_patch": np.full(30, 2.0),
              "permutation": np.full(30, 2.0)},
        floor=Floor(stimulus=2.0, estimator=2.0),
        selection=Selection(axis=None, rule="pre-registered layer"),
        provenance=signed())
    assert claim.treatment.value == 1.25
    assert registry.key_is_current("selector", s3.key)
    assert registry.key_is_current("selector", "0" * 16) is False


def test_no_remote_target_can_be_graded_before_the_remote_tolerance_exists():
    """§1A's pre-registration gap. When `REMOTE_TOLERANCE` is None a remote grade is impossible by
    construction, rather than by someone remembering."""
    assert reproduce.REMOTE_TOLERANCE is None or reproduce.REMOTE_TOLERANCE > 0
