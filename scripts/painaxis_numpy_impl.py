"""Independent numpy versions of the two sklearn pieces the port relies on.

sklearn IS installed in `.venv` (1.9.1), so the Tier A numbers were produced with THEIR calls
(`sklearn.decomposition.PCA`, `sklearn.metrics.roc_auc_score`, `sklearn.model_selection.KFold`).
These functions exist so tests/test_painaxis_port.py can check that arithmetic against a second,
independent implementation -- and so the same port runs unchanged in an environment without
sklearn.
"""
from __future__ import annotations

import numpy as np


def pca_components(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal components and explained-variance ratios of X, by SVD on the centred data.

    Matches sklearn.decomposition.PCA().fit(X): sklearn centres X itself, so we do too.
    Returns (components [k, d], explained_variance_ratio [k]).
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / max(len(X) - 1, 1)
    return Vt, var / var.sum()


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC by the Mann-Whitney U / rank identity, ties averaged."""
    labels = np.asarray(labels).astype(float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0       # 1-based, ties get the average rank
        i = j + 1
    r = np.empty(len(s), dtype=float)
    r[order] = ranks
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_auc_bruteforce(labels: np.ndarray, scores: np.ndarray) -> float:
    """Definition: P(score_pos > score_neg) + 0.5 P(tie), over all pos/neg pairs."""
    pos = np.asarray(scores, dtype=float)[np.asarray(labels) == 1]
    neg = np.asarray(scores, dtype=float)[np.asarray(labels) != 1]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
