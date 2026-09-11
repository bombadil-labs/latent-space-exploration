"""Stage 3: carry the shape.  Affine relational operators between roles, fit across domains.

Given role vectors (s_i, o_i) at one layer from many prompts, fit o ~= W s + b.  Fitting across
several domains cancels domain-specific content; what survives is the relation.  Hold out a domain
to test whether the operator generalizes.  The residuals of the fit define the "spin": the directions
along which valid targets vary for a fixed source.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AffineOp:
    W: np.ndarray        # [d, d]
    b: np.ndarray        # [d]
    layer: int
    src: str
    dst: str
    spin_basis: np.ndarray | None = None   # [k, d] principal directions of fit residuals
    spin_scale: np.ndarray | None = None   # [k] std along each

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return s @ self.W.T + self.b

    def spin(self, s: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Move around the ring of candidates: coords are in units of residual std along spin_basis."""
        return self(s) + (coords * self.spin_scale) @ self.spin_basis


def fit_affine(S: np.ndarray, O: np.ndarray, layer: int, src: str, dst: str,
               ridge: float = 1e-2, n_spin: int = 8, low_rank: int | None = None) -> AffineOp:
    """Ridge-regularized least squares. With few examples and d in the thousands this is heavily
    underdetermined; `low_rank` fits W as identity + rank-k correction, which is the usual remedy."""
    n, d = S.shape
    Sb = np.hstack([S, np.ones((n, 1))])
    if low_rank is None:
        A = np.linalg.solve(Sb.T @ Sb + ridge * np.eye(d + 1), Sb.T @ O)      # [d+1, d]
        W, b = A[:d].T, A[d]
    else:
        # fit residual after identity: (O - S) ~= S U V^T + b, via truncated SVD of the LS solution
        A = np.linalg.solve(Sb.T @ Sb + ridge * np.eye(d + 1), Sb.T @ (O - S))
        U, s, Vt = np.linalg.svd(A[:d], full_matrices=False)
        k = low_rank
        W = np.eye(d) + (U[:, :k] * s[:k]) @ Vt[:k]
        W = W.T
        b = A[d]
    op = AffineOp(W, b, layer, src, dst)
    R = O - op(S)
    if n > 1:
        U, s, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
        k = min(n_spin, len(s))
        op.spin_basis, op.spin_scale = Vt[:k], s[:k] / np.sqrt(max(n - 1, 1))
    return op


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def holdout_eval(S: np.ndarray, O: np.ndarray, groups: np.ndarray, layer: int, src: str, dst: str, **fit_kw) -> dict:
    """Leave-one-group-out: fit on all domains but one, predict the held-out domain's target vectors.
    Reports cosine(pred, true) vs. two baselines: identity (pred = source) and mean-target."""
    out = {"cos_pred": [], "cos_identity": [], "cos_mean": [], "rank": []}
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        op = fit_affine(S[tr], O[tr], layer, src, dst, **fit_kw)
        pred = op(S[te])
        mean_o = O[tr].mean(0)
        for p, s, o in zip(pred, S[te], O[te]):
            out["cos_pred"].append(cosine(p, o))
            out["cos_identity"].append(cosine(s, o))
            out["cos_mean"].append(cosine(mean_o, o))
            # rank of the true target among all targets by cosine to the prediction (1 = best)
            sims = np.array([cosine(p, o2) for o2 in O])
            out["rank"].append(int((sims > cosine(p, o)).sum()) + 1)
    return {k: (float(np.mean(v)) if k != "rank" else float(np.median(v))) for k, v in out.items()}
