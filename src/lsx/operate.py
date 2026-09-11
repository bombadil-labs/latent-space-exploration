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
    W: np.ndarray | None   # [d, d] (materialized) or None when kept in dual form
    b: np.ndarray          # [d]
    layer: int
    src: str
    dst: str
    spin_basis: np.ndarray | None = None   # [k, d] principal directions of fit residuals
    spin_scale: np.ndarray | None = None   # [k] std along each
    S_tr: np.ndarray | None = None         # dual form: W = M^T S_tr  (so  s W^T = (s S_tr^T) M)
    M: np.ndarray | None = None

    def __call__(self, s: np.ndarray) -> np.ndarray:
        if self.W is None:
            return (s @ self.S_tr.T) @ self.M + self.b
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
        b = np.ones(n) @ M
        op = AffineOp(None, b, layer, src, dst, S_tr=S, M=M)   # dual form; never build the d x d matrix
        if n_spin <= 0:
            return op
        R = O - op(S)
        if n > 1:
            U, sv, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
            k = min(n_spin, len(sv)); op.spin_basis, op.spin_scale = Vt[:k], sv[:k] / np.sqrt(max(n - 1, 1))
        return op
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
    if n_spin <= 0:
        return op
    R = O - op(S)
    if n > 1:
        U, sv, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
        k = min(n_spin, len(sv))
        op.spin_basis, op.spin_scale = Vt[:k], sv[:k] / np.sqrt(max(n - 1, 1))
    return op


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def holdout_eval(S: np.ndarray, O: np.ndarray, groups: np.ndarray, layer: int, src: str, dst: str,
                 cands: np.ndarray | None = None, dst_idx: int | None = None, role_center: bool = False,
                 src_idx: int | None = None, null_seed: int | None = None, **fit_kw) -> dict:
    """Leave-one-group-out: fit on all domains but one, predict the held-out domain's target vectors.
    Reports cosine(pred, true) vs. baselines: identity (pred = source), mean-target, shared offset.
    `cands` [n, m, d] = every role vector of each example's own prompt; `role_rank` is then where the
    true dst role lands among those m candidates by cosine to the prediction (1 = best). This factors
    out the domain address, which the cross-domain `rank` does not."""
    out = {"cos_pred": [], "cos_offset": [], "cos_identity": [], "cos_mean": [], "rank": [], "rank_offset": []}
    if cands is not None:
        out["role_rank"], out["role_rank_offset"], out["role_rank_identity"], out["role_rank_mean"] = [], [], [], []
    S0, O0, C0 = S, O, cands
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if role_center:
            # per-role mean over TRAINING prompts only; subtract from everything (train, test, candidates).
            # what remains is domain-specific content per role; role identity is gone.
            mu = C0[tr].mean(0)                                  # [m_roles, d]
            cands = C0 - mu
            S, O = S0 - mu[src_idx], O0 - mu[dst_idx]
        S_tr, O_tr = S[tr], O[tr]
        if null_seed is not None:   # break the src->dst pairing among TRAINING rows only; held-out rows untouched
            O_tr = O_tr[np.random.default_rng(null_seed * 7919 + layer).permutation(len(O_tr))]
        op = fit_affine(S_tr, O_tr, layer, src, dst, n_spin=0, **fit_kw)   # no spin basis needed for evaluation
        pred = op(S[te])
        mean_o = O_tr.mean(0)
        offset = (O_tr - S_tr).mean(0)               # king-queen baseline: one shared translation
        te_idx = np.flatnonzero(te)
        def unit(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)
        On = unit(O)
        for j, (p, s, o) in enumerate(zip(pred, S[te], O[te])):
            po = s + offset
            if cands is not None:
                C = unit(cands[te_idx[j]])                       # [m, d]
                for key, q in (("role_rank", p), ("role_rank_offset", po), ("role_rank_identity", s), ("role_rank_mean", mean_o)):
                    sims = C @ unit(q); out[key].append(int((sims > sims[dst_idx]).sum()) + 1)
            out["cos_pred"].append(cosine(p, o)); out["cos_offset"].append(cosine(po, o))
            out["cos_identity"].append(cosine(s, o)); out["cos_mean"].append(cosine(mean_o, o))
            sp, spo = On @ unit(p), On @ unit(po); t = te_idx[j]
            out["rank"].append(int((sp > sp[t]).sum()) + 1); out["rank_offset"].append(int((spo > spo[t]).sum()) + 1)
    return {k: (float(np.median(v)) if k.startswith("rank") else float(np.mean(v))) for k, v in out.items()}
