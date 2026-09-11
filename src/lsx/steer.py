"""Stage 4: use the lens.  Steering vectors, residual patches, and readout by generation."""
from __future__ import annotations

import numpy as np
import torch

from .model import LM, Patch


def contrast_vector(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Mean-difference steering direction: mean(pos) - mean(neg), each [n, d]."""
    return pos.mean(0) - neg.mean(0)


def add_vector(vec: np.ndarray, scale: float = 1.0, positions: slice | list[int] | None = None) -> callable:
    """Patch fn: add scale*vec to the residual at the given positions (default: all)."""
    v = torch.as_tensor(vec, dtype=torch.float32)

    def fn(hidden: torch.Tensor) -> torch.Tensor:
        h = hidden.clone()
        vv = (scale * v).to(h.dtype).to(h.device)
        if positions is None:
            h += vv
        else:
            h[:, positions] += vv
        return h
    return fn


def replace_vector(vec: np.ndarray, positions: slice | list[int]) -> callable:
    """Patch fn: overwrite the residual at positions with vec."""
    v = torch.as_tensor(vec, dtype=torch.float32)

    def fn(hidden: torch.Tensor) -> torch.Tensor:
        h = hidden.clone()
        h[:, positions] = v.to(h.dtype).to(h.device)
        return h
    return fn


def steer_patches(vec: np.ndarray, layers: list[int], scale: float = 1.0, positions=None) -> list[Patch]:
    """Re-impose the same direction at several layers so downstream 'repair' cannot erode it."""
    return [Patch(l, add_vector(vec, scale, positions)) for l in layers]


def readout(lm: LM, prompt: str, patches: list[Patch], max_new_tokens: int = 24) -> dict:
    """Generate with and without patches; return both so the effect is visible side by side."""
    return {
        "base": lm.generate(prompt, max_new_tokens=max_new_tokens),
        "patched": lm.generate(prompt, max_new_tokens=max_new_tokens, patches=patches),
    }


def logit_shift(lm: LM, prompt: str, patches: list[Patch], k: int = 10) -> dict:
    base, pat = lm.next_token_logits(prompt), lm.next_token_logits(prompt, patches)
    delta = pat - base
    top = torch.topk(delta, k)
    return {
        "base_top": [lm.tok.decode([i]) for i in torch.topk(base, k).indices],
        "patched_top": [lm.tok.decode([i]) for i in torch.topk(pat, k).indices],
        "most_boosted": [(lm.tok.decode([i]), float(s)) for s, i in zip(top.values, top.indices)],
    }
