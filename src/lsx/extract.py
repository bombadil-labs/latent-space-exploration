"""Role-marked prompts -> per-role, per-layer pooled residual vectors.

Prompt syntax:  "... the initial claim is [[thesis: a particle has a definite position]] ..."
Roles are pooled by mean over the tokens whose character span overlaps the marked span.
Un-marked text still runs through the model (context matters); only marked spans are pooled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch

from .model import LM

_ROLE = re.compile(r"\[\[(\w+):\s*(.*?)\]\]", re.S)


@dataclass
class Parsed:
    text: str                                  # prompt with markers stripped
    spans: dict[str, list[tuple[int, int]]]    # role -> [(char_start, char_end), ...]


def parse_roles(marked: str) -> Parsed:
    text, spans, pos = [], {}, 0
    cursor = 0
    for m in _ROLE.finditer(marked):
        text.append(marked[cursor:m.start()])
        pos += m.start() - cursor
        content = m.group(2)
        spans.setdefault(m.group(1), []).append((pos, pos + len(content)))
        text.append(content)
        pos += len(content)
        cursor = m.end()
    text.append(marked[cursor:])
    return Parsed("".join(text), spans)


def tokens_in_span(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    s, e = span
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < e and b > s]


@dataclass
class RoleActivations:
    """roles[role] -> array [L+1, d]; tokens[role] -> token indices used; resid -> full [L+1, seq, d]."""
    roles: dict[str, np.ndarray]
    tokens: dict[str, list[int]]
    resid: np.ndarray
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return self.resid.shape[0]


def extract(lm: LM, marked_prompt: str, pool: str = "mean", keep_resid: bool = True, **meta) -> RoleActivations:
    parsed = parse_roles(marked_prompt)
    hs, offsets = lm.residuals(parsed.text)        # [L+1, seq, d]
    roles, toks = {}, {}
    for role, spans in parsed.spans.items():
        idx = sorted({i for sp in spans for i in tokens_in_span(offsets, sp)})
        if not idx:
            raise ValueError(f"role {role!r} matched no tokens")
        sel = hs[:, idx]                             # [L+1, k, d]
        roles[role] = (sel.mean(1) if pool == "mean" else sel[:, -1]).numpy()
        toks[role] = idx
    return RoleActivations(roles, toks, hs.numpy() if keep_resid else np.empty(0), parsed.text, dict(meta))
