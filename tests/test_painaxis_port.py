"""Checks on the Tier A pain-axis port (scripts/painaxis_analyze.py).

Three things are checked here, in the order they could bite:
  1. the numpy stand-ins (PCA by SVD, ROC-AUC by the rank identity) agree with sklearn AND with a
     brute-force pairwise AUC on random data -- so the arithmetic is the same whichever runs;
  2. compute_pain_vector's component count follows their `searchsorted(cumvar, 0.5) + 1` rule and
     the result is orthogonal to every component it removes;
  3. the K-fold really holds out: a fitted-and-scored-on-everything vector reaches AUC 1.0 on
     synthetic data whose pain direction is different in every sentence set, while the K-fold
     held-out AUC stays at chance. This is the leakage trap the brief names.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

sk = pytest.importorskip("sklearn")
from sklearn.decomposition import PCA                      # noqa: E402
from sklearn.metrics import roc_auc_score                  # noqa: E402

import painaxis_analyze as pa                              # noqa: E402
import painaxis_numpy_impl as ni                           # noqa: E402


def test_numpy_pca_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 12)) @ rng.normal(size=(12, 12))
    comps, ratio = ni.pca_components(X)
    ref = PCA().fit(X)
    assert np.allclose(ratio[:len(ref.explained_variance_ratio_)],
                       ref.explained_variance_ratio_, atol=1e-10)
    for a, b in zip(comps, ref.components_):
        assert np.allclose(a, b, atol=1e-8) or np.allclose(a, -b, atol=1e-8)


@pytest.mark.parametrize("seed", range(5))
def test_numpy_auc_matches_sklearn_and_bruteforce(seed):
    rng = np.random.default_rng(seed)
    n = 60
    labels = rng.integers(0, 2, size=n)
    if labels.sum() in (0, n):
        labels[0], labels[-1] = 0, 1
    # discrete scores so ties actually occur
    scores = rng.integers(0, 6, size=n).astype(float) + rng.normal(scale=0.01, size=n) * (seed % 2)
    mine = ni.roc_auc(labels, scores)
    assert mine == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)
    assert mine == pytest.approx(ni.roc_auc_bruteforce(labels, scores), abs=1e-12)


def test_denoise_component_count_and_orthogonality():
    rng = np.random.default_rng(1)
    d = 24
    cats = np.array(["A1"] * 20 + ["B", "C1", "C2", "D", "E"] * 8)
    acts = rng.normal(size=(len(cats), d))
    acts[:, 0] += 8 * rng.normal(size=len(cats))          # one dominant control direction
    vec = pa.compute_pain_vector(acts, cats)
    ctrl = acts[np.isin(cats, pa.CONTROL_CATEGORIES)]
    pca = PCA().fit(ctrl - np.nanmean(ctrl, axis=0))
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = min(np.searchsorted(cum, pa.DENOISE_VARIANCE) + 1, len(pca.components_))
    assert cum[k - 1] >= pa.DENOISE_VARIANCE and (k == 1 or cum[k - 2] < pa.DENOISE_VARIANCE)
    for c in pca.components_[:k]:
        assert abs(float(vec @ c)) < 1e-8 * max(1.0, float(np.linalg.norm(vec)))


def _set_specific_data(rng, n_sets=20, d=32):
    """Pain and control differ along a DIFFERENT direction in every sentence set, so nothing
    generalizes across sets: in-sample AUC 1.0, honest held-out AUC ~ chance."""
    cats, sets, rows = [], [], []
    per = ["A1", "A2", "A3", "A4", "A5", "B", "C1", "C2", "D", "E"]
    for s in range(1, n_sets + 1):
        u = rng.normal(size=d)
        u /= np.linalg.norm(u)
        for c in per:
            sign = 1.0 if c.startswith("A") else -1.0
            rows.append(20.0 * sign * u + rng.normal(scale=0.05, size=d))
            cats.append(c)
            sets.append(s)
    return np.array(rows)[:, None, :], np.array(cats), np.array(sets)


def test_kfold_holds_out_and_insample_does_not():
    rng = np.random.default_rng(3)
    acts, cats, sets = _set_specific_data(rng)
    in_sample = pa.compute_auc(acts[:, 0, :], cats,
                               pa.compute_pain_vector(acts[:, 0, :], cats))
    heldout = pa.kfold_curve(acts, cats, sets, [0])[0]["auc_vs_all_controls"]
    assert in_sample > 0.99, in_sample
    assert 0.3 < heldout < 0.7, heldout


def test_kfold_train_and_test_sets_are_disjoint():
    from sklearn.model_selection import KFold
    sets = np.repeat(np.arange(1, 21), 10)
    uniq = sorted(set(sets.tolist()))
    kf = KFold(n_splits=pa.N_FOLDS, shuffle=True, random_state=pa.RANDOM_SEED)
    seen = []
    for tr, te in kf.split(uniq):
        trm = np.isin(sets, [uniq[i] for i in tr])
        tem = np.isin(sets, [uniq[i] for i in te])
        assert not (trm & tem).any()
        assert trm.sum() + tem.sum() == len(sets)
        seen.append(tem)
    assert np.stack(seen).sum(axis=0).max() == 1          # every sentence held out exactly once
