"""lsx: tools for measuring and moving relational shapes in transformer residual streams.

Stages:
  model    - load a causal LM, capture per-layer residual stream, patch it during forward.
  extract  - run role-marked prompts, pool token activations per role per layer.
  compare  - RSA / CKA / Gromov-Wasserstein across prompts, per layer (does the shape exist?).
  operate  - fit affine relational maps between roles across domains (carry the shape).
  steer    - build steering vectors, patch them in, read out by generation (use the lens).
"""
from .model import LM
from .extract import parse_roles, extract, RoleActivations
from . import compare, operate, steer

__all__ = ["LM", "parse_roles", "extract", "RoleActivations", "compare", "operate", "steer"]
