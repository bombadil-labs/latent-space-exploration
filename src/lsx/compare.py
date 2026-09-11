"""Stage 2: does the shape exist?  Compare intra-prompt relational structure across prompts.

All measures here are invariant to *where* the cloud sits (translation) and most to how it is
oriented (rotation), so they compare Gestalt, not Ort.
"""
from __future__ import annotations

import numpy as np
import ot
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


# ---------- point-set helpers ----------
def center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(0, keepdims=True)


def rdm(X: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Representational dissimilarity matrix of the rows of X after centering."""
    return squareform(pdist(center(X), metric=metric))


# ---------- shape comparison (rows correspond) ----------
def rsa(X: np.ndarray, Y: np.ndarray, metric: str = "cosine") -> float:
    """Spearman correlation of upper-triangular RDMs. Requires the same n rows in the same role order."""
    a, b = rdm(X, metric), rdm(Y, metric)
    iu = np.triu_indices_from(a, k=1)
    return float(spearmanr(a[iu], b[iu]).correlation)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered kernel alignment with a linear kernel; invariant to orthogonal transforms and isotropic scale."""
    Xc, Yc = center(X), center(Y)
    Kx, Ky = Xc @ Xc.T, Yc @ Yc.T                      # n x n grams; tr(Kx Ky) == ||Xc^T Yc||_F^2
    return float(np.sum(Kx * Ky) / (np.linalg.norm(Kx, "fro") * np.linalg.norm(Ky, "fro")))


def procrustes_residual(X: np.ndarray, Y: np.ndarray) -> float:
    """Normalized residual after optimal rotation+scale of X onto Y (0 = identical shape)."""
    Xc, Yc = center(X), center(Y)
    Xc /= np.linalg.norm(Xc); Yc /= np.linalg.norm(Yc)
    U, s, Vt = np.linalg.svd(Xc.T @ Yc)
    return float(1 - s.sum() ** 2)


# ---------- shape comparison (no correspondence) ----------
def gromov_wasserstein(X: np.ndarray, Y: np.ndarray, metric: str = "cosine") -> tuple[float, np.ndarray]:
    """GW distance between two clouds with no shared coordinates and no known correspondence.
    Returns (gw2, coupling). Coupling[i, j] is how much point i of X is matched to point j of Y."""
    C1, C2 = rdm(X, metric), rdm(Y, metric)
    p, q = ot.unif(len(X)), ot.unif(len(Y))
    T, log = ot.gromov.gromov_wasserstein(C1, C2, p, q, "square_loss", log=True)
    return float(log["gw_dist"]), T


# ---------- layer sweeps ----------
def sweep(A: np.ndarray, B: np.ndarray, fn=rsa, **kw) -> np.ndarray:
    """A, B: [L+1, n, d] stacks (e.g. role vectors stacked in a fixed role order). Returns [L+1] scores."""
    return np.array([fn(A[l], B[l], **kw) for l in range(A.shape[0])])


def role_stack(acts, roles: list[str]) -> np.ndarray:
    """RoleActivations -> [L+1, n_roles, d] in the given role order."""
    return np.stack([acts.roles[r] for r in roles], axis=1)


def similarity_matrix(stacks: dict[str, np.ndarray], layer: int, fn=rsa) -> tuple[list[str], np.ndarray]:
    """Pairwise shape similarity between named prompts at one layer."""
    names = list(stacks)
    M = np.eye(len(names))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j > i:
                M[i, j] = M[j, i] = fn(stacks[a][layer], stacks[b][layer])
    return names, M


# ---------- making the numbers interpretable ----------
def subtract_grand_mean(stacks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remove the per-layer mean role vector across all prompts (the shared 'English paragraph' component)."""
    mu = np.mean(np.stack(list(stacks.values())), axis=(0, 2), keepdims=True)[0]  # [L+1, 1, d]
    return {k: v - mu for k, v in stacks.items()}


def permutation_null(X: np.ndarray, Y: np.ndarray, fn=linear_cka, n: int = 200, seed: int = 0) -> tuple[float, float, float]:
    """Observed fn(X,Y), and the mean/std of fn(X, Y[perm]) over random role permutations.
    Returns (observed, null_mean, null_std). z = (obs - mean)/std says how far above chance the
    shape match is, given these exact vectors but scrambled correspondence."""
    rng = np.random.default_rng(seed)
    obs = fn(X, Y)
    null = np.array([fn(X, Y[rng.permutation(len(Y))]) for _ in range(n)])
    return float(obs), float(null.mean()), float(null.std() + 1e-12)


def zscore(X, Y, fn=linear_cka, n=200, seed=0) -> float:
    o, m, s = permutation_null(X, Y, fn, n, seed)
    return (o - m) / s
