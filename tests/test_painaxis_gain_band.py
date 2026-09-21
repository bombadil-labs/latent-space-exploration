"""The one property that makes the cluster bootstrap honest: no set straddles a split.

A bootstrap that resamples clusters with replacement and then folds on ROW position lets the
same sentences sit in train and test, which inflates held-out AUC. The gain is a difference, so
some of that cancels -- but not all, and "it mostly cancels" is not a control. These tests pin
the leak-free construction instead of trusting it.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

PA = pytest.importorskip("painaxis_analyze")
GB = pytest.importorskip("painaxis_gain_band")


def _toy(n_sets=20, per_set=10):
    sets_ = np.array([f"set{i}" for i in range(n_sets) for _ in range(per_set)], dtype=object)
    return sets_, sorted(set(sets_.tolist())), {s: np.where(sets_ == s)[0] for s in set(sets_.tolist())}


def test_fold_masks_partition_every_row_exactly_once_per_fold():
    sets_, uniq, _ = _toy()
    splits = GB.fold_masks(PA, uniq, sets_, PA.RANDOM_SEED)
    assert len(splits) == PA.N_FOLDS
    for trm, tem in splits:
        assert not (trm & tem).any(), "a row is in both train and test"
        assert (trm | tem).all(), "a row is in neither"
    # every row is held out exactly once across the folds
    held = np.sum([tem for _, tem in splits], axis=0)
    assert (held == 1).all()


def test_fold_masks_keep_a_set_whole():
    sets_, uniq, _ = _toy()
    for trm, tem in GB.fold_masks(PA, uniq, sets_, PA.RANDOM_SEED):
        for s in uniq:
            rows = sets_ == s
            assert trm[rows].all() or tem[rows].all(), f"{s} straddles the split"


def test_cluster_resample_no_leak_across_many_draws():
    sets_, uniq, rows_by_set = _toy()
    rng = np.random.default_rng(0)
    saw_duplicate = False
    for _ in range(200):
        idx, labels, present = GB.cluster_resample(uniq, rows_by_set, rng)
        assert len(idx) == len(sets_)
        counts = {s: (labels == s).sum() // len(rows_by_set[s]) for s in present}
        saw_duplicate |= any(c > 1 for c in counts.values())
        splits = GB.fold_masks(PA, present, labels, PA.RANDOM_SEED)
        for trm, tem in splits:
            train_sets = set(labels[trm].tolist())
            test_sets = set(labels[tem].tolist())
            assert not (train_sets & test_sets), "a resampled set appears in train AND test"
    assert saw_duplicate, "resample drew no duplicate in 200 draws -- it is not with replacement"


def test_cluster_resample_labels_track_rows():
    sets_, uniq, rows_by_set = _toy()
    rng = np.random.default_rng(7)
    idx, labels, _ = GB.cluster_resample(uniq, rows_by_set, rng)
    assert (sets_[idx] == labels).all(), "row label does not match the row it was drawn for"


def test_gain_is_within_replicate():
    """The band must come from (layer - emb) inside each replicate, not from differencing two
    independently-banded means: the two share stimulus noise and it has to cancel."""
    rng = np.random.default_rng(3)
    shared = rng.normal(0, 0.05, 500)
    emb = 0.87 + shared
    treat = 0.92 + shared + rng.normal(0, 0.002, 500)
    paired = GB._band(treat - emb)
    naive_width = (np.percentile(treat, 97.5) - np.percentile(treat, 2.5))
    assert (paired[2] - paired[1]) < naive_width / 5, "paired band is not cancelling shared noise"
    assert paired[3] == 0.0, "gain should never be <= 0 here"
