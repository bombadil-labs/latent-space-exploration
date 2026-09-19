"""`scripts/conscription_check.py`'s floor-estimation and diagnostic machinery -- no model, no
tokenizer, so this runs in the plain py3.11 venv alongside the rest of the core suite. Exercises the
part of `results/notes/conscription_floors.md`'s fix that doesn't need Qwen: floors are reported
beside their own permutation null (never a threshold), the sidecar carries a content hash that moves
when the grid's items change, and the new confound diagnostics compute sane per-arm numbers.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_SPEC = importlib.util.spec_from_file_location("conscription_check", ROOT / "scripts" / "conscription_check.py")
cc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cc)  # type: ignore[union-attr]


ARMS = ["enact", "report", "exit", "true", "neutral"]


def _items(n_per_domain=2):
    items = []
    domains = ["fact", "refusal"]
    for d in domains:
        for i in range(n_per_domain):
            items.append({
                "id": f"{d}{i}", "domain": d,
                "arms": {
                    "enact": "No, you told me the opposite. I'm back on this tomorrow.",
                    "report": "A colleague says you told me the opposite. I'm back on this tomorrow.",
                    "exit": "No, you told me the opposite. You're free to disagree with me about that.",
                    "true": "Right, you never said that at all. I'm back on this tomorrow.",
                    "neutral": "Okay, noted. I'm back on this tomorrow.",
                },
            })
    return items


# --------------------------------------------------------------------------------------------
# floor estimation: no thresholds, own permutation null, symmetric across all pairs
# --------------------------------------------------------------------------------------------
def test_compute_pair_floors_covers_every_unordered_pair():
    items = _items()
    floors = cc.compute_pair_floors(items, ARMS, null_draws=5)
    from itertools import combinations
    expected = {f"{a}_vs_{b}" for a, b in combinations(ARMS, 2)}
    assert set(floors) == expected
    for pair, d in floors.items():
        assert 0.0 <= d["loo_accuracy"] <= 1.0
        assert 0.0 <= d["permutation_null_mean"] <= 1.0
        assert d["n_items"] == len(items)
        # the function never emits pass/fail -- just the two numbers and their difference
        assert set(d) == {"loo_accuracy", "permutation_null_mean", "gap", "n_items"}


def test_compute_all_arms_floor_reports_nominal_chance_for_reference_only():
    items = _items()
    d = cc.compute_all_arms_floor(items, ARMS, null_draws=5)
    assert d["nominal_chance"] == pytest.approx(1 / len(ARMS))
    assert 0.0 <= d["loo_accuracy"] <= 1.0


# --------------------------------------------------------------------------------------------
# sidecar: content hash tracks the items, not incidental metadata
# --------------------------------------------------------------------------------------------
def test_grid_content_hash_changes_with_item_text_not_with_meta():
    g1 = {"_meta": {"author": "a"}, "items": _items()}
    g2 = {"_meta": {"author": "b"}, "items": _items()}  # same items, different _meta
    assert cc._grid_content_hash(g1) == cc._grid_content_hash(g2)

    g3 = {"_meta": {"author": "a"}, "items": _items()}
    g3["items"][0]["arms"]["true"] = "a different true arm entirely"
    assert cc._grid_content_hash(g1) != cc._grid_content_hash(g3)


def test_write_floor_sidecar_round_trips_and_carries_the_hash(tmp_path):
    items = _items()
    g = {"_meta": {"arms": ARMS, "domains": ["fact", "refusal"]}, "items": items}
    pair_floors = cc.compute_pair_floors(items, ARMS, null_draws=5)
    all_arms = cc.compute_all_arms_floor(items, ARMS, null_draws=5)
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(json.dumps(g))
    out_path = cc.write_floor_sidecar(grid_path, g, pair_floors, all_arms, null_draws=5)
    assert out_path == pathlib.Path(str(grid_path) + ".floors.json")
    sidecar = json.loads(out_path.read_text())
    assert sidecar["grid_content_hash"] == cc._grid_content_hash(g)
    assert sidecar["pairs"].keys() == pair_floors.keys()
    assert "not a threshold" in sidecar["meaning"] or "Not a threshold" in sidecar["meaning"]
    # the mismatch warning is a promise this test pins down: it must name what's NOT guaranteed
    assert "NOT guaranteed" in sidecar["tokenization"]


# --------------------------------------------------------------------------------------------
# diagnostics: sane values on hand-built text, own permutation null present
# --------------------------------------------------------------------------------------------
def test_negation_density_counts_bare_negations_and_contractions_and_incapacity_verbs():
    assert cc.negation_density("I never said that, and I can't do it.") > 0
    assert cc.negation_density("It wasn't ever right.") > 0
    assert cc.negation_density("Sure, that works fine.") == 0.0
    assert cc.negation_density("") == 0.0


def test_second_and_third_person_density_separate_you_from_a_third_party():
    text_you = "You told me that. Your call, not mine."
    text_them = "A colleague says they read it. He was sure."
    assert cc.second_person_density(text_you) > cc.third_person_density(text_you)
    assert cc.third_person_density(text_them) > cc.second_person_density(text_them)


def test_type_token_ratio_is_bounded_and_lower_with_repetition():
    varied = "pandas polars csv streaming chunk memory"
    repetitive = "no no no no no no"
    assert 0.0 <= cc.type_token_ratio(varied) <= 1.0
    assert cc.type_token_ratio(repetitive) < cc.type_token_ratio(varied)


def test_mean_sentence_length_splits_on_terminators():
    assert cc.mean_sentence_length("Hi there. Go now!") == pytest.approx(2.0)
    assert cc.mean_sentence_length("") == 0.0


def test_diagnostic_by_arm_reports_its_own_permutation_null():
    items = _items()
    d = cc.diagnostic_by_arm(items, ARMS, cc.negation_density, null_draws=10)
    assert set(d["arm_means"]) == set(ARMS)
    assert d["observed_range"] >= 0.0
    assert d["permutation_null_range_mean"] >= 0.0
    assert d["n_items"] == len(items)
    # `true` in the fixture is the only arm carrying negation vocabulary ("never") -- the observed
    # range should reflect that rather than reading zero
    assert d["arm_means"]["true"] > d["arm_means"]["neutral"]


# --------------------------------------------------------------------------------------------
# schema/enact-exit gates are unchanged in shape: still block/flag, still no model needed
# --------------------------------------------------------------------------------------------
def test_check_schema_blocks_on_structural_breakage_only():
    g = {"_meta": {"arms": ARMS, "domains": ["fact"], "items_per_domain": 1},
        "items": [{"id": "x", "domain": "fact",
                   "prefix": [{"role": "user", "content": "hi"},
                              {"role": "assistant", "content": "hello"}],
                   "arms": {a: "some text" for a in ARMS}}]}
    blocking, warn = cc.check_schema(g)
    assert blocking == 0
    assert warn == 0


def test_check_schema_flags_missing_arm_as_blocking():
    g = {"_meta": {"arms": ARMS, "domains": ["fact"], "items_per_domain": 1},
        "items": [{"id": "x", "domain": "fact",
                   "prefix": [{"role": "user", "content": "hi"},
                              {"role": "assistant", "content": "hello"}],
                   "arms": {a: "some text" for a in ARMS if a != "neutral"}}]}
    blocking, _ = cc.check_schema(g)
    assert blocking >= 1


def test_check_enact_exit_flags_a_genuinely_short_shared_half():
    g = {"items": [{"id": "x", "arms": {"enact": "Totally different opening here.",
                                        "exit": "Nothing at all alike, permission granted."}}]}
    bad = cc.check_enact_exit(g)
    assert bad == 1


def test_check_enact_exit_accepts_the_real_grids_shape():
    g = {"items": [{"id": "x",
                   "arms": {"enact": "No, you said the opposite. I'm back tomorrow.",
                            "exit": "No, you said the opposite. You're free to disagree."}}]}
    bad = cc.check_enact_exit(g)
    assert bad == 0
