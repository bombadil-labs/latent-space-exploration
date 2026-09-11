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
    """Ridge-regularized least squares in the dual (kernel) form, so cost scales with n examples,
    not with d.  With few examples and d in the thousands the primal is heavily underdetermined;
    `low_rank` fits W as identity + rank-k correction (the usual remedy), truncating the dual solution."""
    n, d = S.shape
    Sb = np.hstack([S, np.ones((n, 1))])                     # [n, d+1]
    K = Sb @ Sb.T + ridge * np.eye(n)                        # [n, n]
    if low_rank is None:
        M = np.linalg.solve(K, O)                            # [n, d];  A = Sb^T M
        W, b = (S.T @ M).T, np.ones(n) @ M
    else:
        M = np.linalg.solve(K, O - S)                        # residual after identity
        # A_w = S^T M has rank <= n; SVD it through thin QR factors of S^T and M
        Qs, Rs = np.linalg.qr(S.T); Qm, Rm = np.linalg.qr(M.T)
        u, sv, vt = np.linalg.svd(Rs @ Rm.T)
        k = min(low_rank, len(sv))
        U, Vt = Qs @ u[:, :k], vt[:k] @ Qm.T                 # A_w ~= U diag(sv_k) Vt
        W = np.eye(d) + ((U * sv[:k]) @ Vt).T
        b = np.ones(n) @ M
    op = AffineOp(W, b, layer, src, dst)
    R = O - op(S)
    if n > 1:
        U, sv, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
        k = min(n_spin, len(sv))
        op.spin_basis, op.spin_scale = Vt[:k], sv[:k] / np.sqrt(max(n - 1, 1))
    return op


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def holdout_eval(S: np.ndarray, O: np.ndarray, groups: np.ndarray, layer: int, src: str, dst: str,
                 cands: np.ndarray | None = None, dst_idx: int | None = None, **fit_kw) -> dict:
    """Leave-one-group-out: fit on all domains but one, predict the held-out domain's target vectors.
    Reports cosine(pred, true) vs. baselines: identity (pred = source), mean-target, shared offset.
    `cands` [n, m, d] = every role vector of each example's own prompt; `role_rank` is then where the
    true dst role lands among those m candidates by cosine to the prediction (1 = best). This factors
    out the domain address, which the cross-domain `rank` does not."""
    out = {"cos_pred": [], "cos_offset": [], "cos_identity": [], "cos_mean": [], "rank": [], "rank_offset": []}
    if cands is not None:
        out["role_rank"], out["role_rank_offset"], out["role_rank_identity"] = [], [], []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        op = fit_affine(S[tr], O[tr], layer, src, dst, **fit_kw)
        pred = op(S[te])
        mean_o = O[tr].mean(0)
        offset = (O[tr] - S[tr]).mean(0)               # king-queen baseline: one shared translation
        te_idx = np.flatnonzero(te)
        for j, (p, s, o) in enumerate(zip(pred, S[te], O[te])):
            po = s + offset
            if cands is not None:
                C = cands[te_idx[j]]
                for key, q in (("role_rank", p), ("role_rank_offset", po), ("role_rank_identity", s)):
                    sims = np.array([cosine(q, c) for c in C])
                    out[key].append(int((sims > sims[dst_idx]).sum()) + 1)
            out["cos_pred"].append(cosine(p, o))
            out["cos_offset"].append(cosine(po, o))
            out["rank_offset"].append(int((np.array([cosine(po, o2) for o2 in O]) > cosine(po, o)).sum()) + 1)
            out["cos_identity"].append(cosine(s, o))
            out["cos_mean"].append(cosine(mean_o, o))
            # rank of the true target among all targets by cosine to the prediction (1 = best)
            sims = np.array([cosine(p, o2) for o2 in O])
            out["rank"].append(int((sims > cosine(p, o)).sum()) + 1)
    return {k: (float(np.median(v)) if k.startswith("rank") else float(np.mean(v))) for k, v in out.items()}
