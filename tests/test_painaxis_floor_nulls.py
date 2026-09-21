"""Tests for the floor/null arms in scripts/painaxis_floor_nulls.py.

The named trap for this task is a shuffled-label null that still leaks the labels. These tests
guard the shuffle itself and the fold loop it runs through; the pipeline-level guards (my fold
loop reproducing their `kfold_curve`, and an identity permutation reproducing the treatment
curve) run inside the analysis and abort it on failure.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

fn = pytest.importorskip("painaxis_floor_nulls")
PA = pytest.importorskip("painaxis_analyze")

CATS = ["A1", "A2", "A3", "A4", "A5", "B", "C1", "C2", "D", "E"]


def _grid(n_sets=20):
    cats = np.array(CATS * n_sets)
    sets = np.repeat(np.arange(1, n_sets + 1), len(CATS))
    return cats, sets


def test_shuffle_preserves_marginals_globally_and_per_set():
    cats, sets = _grid()
    rng = np.random.default_rng(0)
    out = fn.shuffle_labels_within_set(cats, sets, rng)
    assert sorted(out.tolist()) == sorted(cats.tolist())
    for s in np.unique(sets):
        m = sets == s
        assert sorted(out[m].tolist()) == sorted(cats[m].tolist())


def test_shuffle_preserves_fold_balance():
    """Every set keeps 5 pain / 5 control, so every fold does too -- the AUC's class balance
    is identical under the null and under the treatment."""
    cats, sets = _grid()
    rng = np.random.default_rng(1)
    out = fn.shuffle_labels_within_set(cats, sets, rng)
    for s in np.unique(sets):
        m = sets == s
        assert np.isin(out[m], PA.PAIN_CATEGORIES).sum() == 5
        assert np.isin(out[m], PA.CONTROL_CATEGORIES).sum() == 5


def test_shuffle_actually_destroys_the_association():
    """Agreement with the true labels must sit at chance (1/10 per sentence), not near 1."""
    cats, sets = _grid()
    rng = np.random.default_rng(2)
    same = [float((fn.shuffle_labels_within_set(cats, sets, rng) == cats).mean())
            for _ in range(50)]
    assert 0.05 < float(np.mean(same)) < 0.16
    assert max(same) < 0.35


def test_shuffle_does_not_mutate_its_input():
    cats, sets = _grid()
    before = cats.copy()
    fn.shuffle_labels_within_set(cats, sets, np.random.default_rng(3))
    assert (cats == before).all()


def test_fold_loop_matches_their_kfold_curve():
    """curve_all_controls must equal PA.kfold_curve's auc_vs_all_controls exactly -- it is the
    same functions in the same order, only the unused neutral arm is skipped."""
    rng = np.random.default_rng(4)
    cats, sets = _grid()
    n, d = len(cats), 16
    acts = rng.standard_normal((n, 3, d)).astype(np.float64)
    acts[np.isin(cats, PA.PAIN_CATEGORIES), 1, :] += 0.8  # a real signal at layer 1 only
    mine = fn.curve_all_controls(PA, acts, cats, sets, [0, 1, 2])
    theirs = np.array([r["auc_vs_all_controls"]
                       for r in PA.kfold_curve(acts, cats, sets, [0, 1, 2])])
    assert np.allclose(mine, theirs, atol=1e-12)
    assert mine[1] > 0.8 and 0.3 < mine[0] < 0.7


def test_shuffled_label_null_lands_at_chance_on_data_with_a_real_signal():
    """The null's own validity check: on data where the treatment AUC is ~1, the shuffled-label
    arm must still return ~0.5. A leak (denoising fitted on the true controls, or folds built
    from the unshuffled assignment) shows up here as a null well away from 0.5."""
    rng = np.random.default_rng(5)
    cats, sets = _grid()
    n, d = len(cats), 16
    acts = rng.standard_normal((n, 1, d))
    acts[np.isin(cats, PA.PAIN_CATEGORIES), 0, :] += 3.0
    assert fn.curve_all_controls(PA, acts, cats, sets, [0])[0] > 0.95
    draws = np.array([fn.curve_all_controls(
        PA, acts, fn.shuffle_labels_within_set(cats, sets, rng), sets, [0])[0]
        for _ in range(40)])
    assert abs(draws.mean() - 0.5) < 0.05, draws.mean()


def test_random_direction_arm_lands_at_chance_on_data_with_a_real_signal():
    rng = np.random.default_rng(6)
    cats, sets = _grid()
    n, d = len(cats), 16
    acts = rng.standard_normal((n, 1, d))
    acts[np.isin(cats, PA.PAIN_CATEGORIES), 0, :] += 3.0
    res, res_fixed = fn.curve_random_direction(PA, acts, cats, sets, [0], rng, 60)
    assert abs(res[:, 0].mean() - 0.5) < 0.05, res[:, 0].mean()
    assert abs(res_fixed[:, 0].mean() - 0.5) < 0.06, res_fixed[:, 0].mean()
    # the fixed-direction arm reuses one draw across folds, so its spread must be the wider one
    assert res_fixed[:, 0].std() > res[:, 0].std()


def test_constant_activations_give_auc_exactly_one_half():
    """The embedding-layer final-token degeneracy, in miniature: identical rows for every
    sentence must score exactly 0.5 through their AUC path, not NaN and not a near-miss."""
    cats, sets = _grid()
    acts = np.tile(np.arange(16, dtype=np.float64), (len(cats), 1))[:, None, :]
    out = fn.curve_all_controls(PA, acts, cats, sets, [0])
    assert out[0] == 0.5
