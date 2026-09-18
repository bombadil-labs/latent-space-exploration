"""Refusals and assertions: the mechanisms the rediscovery harness names.

Every exception here is tied to a retraction in `docs/INSTRUMENTS.md` or to a near-miss the spec
records. The rule of the module: an assertion raises with the *numbers* that made it fire, because
the failures this project has had all rendered as plausible numbers and a bare `AssertionError`
would have been read as a flaky test.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np


# --------------------------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------------------------
class CoreError(Exception):
    """Base of every refusal. Catching this catches the whole contract."""


class PaddingConvention(CoreError):
    """The tokenizer's padding side disagrees with the span-indexing convention. (h39)"""


class BatchEquivalence(CoreError):
    """A batched extraction does not reproduce the batch-of-one extraction. (h39)"""


class MovedCandidates(CoreError):
    """A patched forward moved a number of sequences other than the batch size. (h34/h36)"""


class EmptySpan(CoreError):
    """A span resolved to zero real tokens, or reached into padding. (h39)"""


class LayerOutputShape(CoreError):
    """A decoder block's output was neither a tensor nor a tuple whose first element is one. (h36)"""


class PostNormResidual(CoreError):
    """A post-final-norm hidden state was passed where a residual was required. (h40)"""


class MissingArm(CoreError):
    """A Claim is missing an arm its instrument requires. (h34, spec §4)"""


class ArmOffNull(CoreError):
    """An arm is sitting away from the value it declared it should sit at. (h34, spec §4)"""


class MissingFloor(CoreError):
    """A Claim has no floor, or only one of the two required floors. (h28, spec §4)"""


class RawScoreOnLeakyGrid(CoreError):
    """A Claim on a grid with a measured leak reports a raw score. (h35/h38, spec §6)"""


class SelectionOnScoringData(CoreError):
    """The reported value was chosen along an axis swept on the scoring data. (h3/h16/h38)"""


class MissingCalibration(CoreError):
    """No calibration report attached to the instrument. (spec §5)"""


class CalibrationFailed(CoreError):
    """The instrument failed its own calibration and has no passing companion. (h6, spec §5)"""


class NullDeclaredLate(CoreError):
    """The semantic null was computed after the treatment score. (spec §4)"""


class HeldOutNotDeclared(CoreError):
    """A Direction was fitted without declaring what it never saw. (spec §3)"""


class ProvenanceIncomplete(CoreError):
    """Provenance is missing a field that makes a stack or claim re-identifiable. (spec §7.6)"""


# --------------------------------------------------------------------------------------------
# plumbing assertions (spec §7)
# --------------------------------------------------------------------------------------------
def resid(output):
    """Tuple-or-tensor, resolved by type and never by index. (h36: `output[0]` is batch row 0 when
    a decoder block returns a bare tensor, which transformers >= 4.54 does.)"""
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise LayerOutputShape(f"block output of type {type(output).__name__} is neither a tensor nor a "
                           "tuple whose first element is one; refusing to index it")


def assert_padding_convention(padding_side: str, span_policy: str, declared: str | None = None) -> None:
    """The tokenizer's actual padding side, the convention the caller indexes with, and (if given)
    the padding side the caller *believes* it has, must agree. (h39: nnsight's LanguageModel loads
    the tokenizer with padding_side='left' while the script assumed right.)"""
    if declared is not None and declared != padding_side:
        raise PaddingConvention(
            f"tokenizer pads {padding_side!r} but the caller declared {declared!r}; span indices "
            "computed on the unpadded text are shifted by n_pad for every item shorter than the "
            "longest in its batch (h39: 363 of 480 passages)")
    if padding_side not in ("left", "right"):
        raise PaddingConvention(f"unknown padding side {padding_side!r}")
    if span_policy == "auto":
        return
    if padding_side == "left" and span_policy != "end_relative":
        raise PaddingConvention(
            f"tokenizer pads left but span_policy is {span_policy!r}; under left padding absolute "
            "indices read n_pad positions too early. Use end-relative indices (identical at batch 1).")
    if padding_side == "right" and span_policy != "absolute":
        raise PaddingConvention(
            f"tokenizer pads right but span_policy is {span_policy!r}; under right padding the "
            "unpadded indices ARE the padded ones and end-relative indices read n_pad too late.")


def shortest_item_index(lengths: Sequence[int]) -> int:
    """The item a batched-vs-single check must use. Never a random sample: the shortest item is the
    maximally padded one, so it is the one that fails first. (spec §7.2)"""
    return int(np.argmin(np.asarray(lengths)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def assert_batch_equivalence(batched: np.ndarray, single: np.ndarray, *, item: int, where: str,
                             min_cos: float = 0.999) -> float:
    """Vectors from a batched job must match a batch-of-one extraction at cosine >= min_cos."""
    c = min(cosine(batched[k], single[k]) for k in range(len(np.atleast_2d(batched))))
    if c < min_cos:
        raise BatchEquivalence(
            f"batched extraction of the shortest item (index {item}, {where}) matches its "
            f"batch-of-one extraction at cosine {c:.5f} < {min_cos}; the batched vectors are not "
            "the vectors you think they are (h39)")
    return c


def assert_moved_candidates(base: np.ndarray, patched: np.ndarray, *, batch: int,
                            atol: float = 1e-6) -> int:
    """The number of sequences whose score changed under a patch must equal the batch size. (h34:
    nine candidates in one padded batch, one moved, and rank-1-on-ties read that as a sharp lens.)"""
    base, patched = np.asarray(base, dtype=np.float64), np.asarray(patched, dtype=np.float64)
    moved = int(np.sum(np.abs(patched - base) > atol))
    if moved != batch:
        raise MovedCandidates(
            f"patch moved {moved}/{batch} sequences; a patch that reaches one row of a padded batch "
            f"is the h34 signature (deltas {np.round(patched - base, 6).tolist()})")
    return moved


def assert_nonempty_spans(indices: Iterable[int], attention_mask: Sequence[int], *, item: int,
                          span: str) -> None:
    """Every span resolves to >= 1 token, and never into padding."""
    idx = list(indices)
    if not idx:
        raise EmptySpan(f"item {item} span {span!r} resolved to zero tokens")
    pad = [i for i in idx if not attention_mask[i]]
    if pad:
        raise EmptySpan(f"item {item} span {span!r} reaches padding at positions {pad} "
                        f"(of {idx}); the span is being pooled out of the pad block (h39)")


def assert_pre_norm(lm, vec: np.ndarray, layer: int, rel_tol: float = 0.01) -> None:
    """Refuse a post-final-norm hidden state where a residual is required. (h40)

    HF applies the final norm to the last hidden state, so `residuals()[-1]` is the norm OUTPUT
    (mean position norm 190.5 vs 283.2 on Qwen2.5-1.5B). `LM.pre_norm_residual` is the sanctioned
    capture; this assertion is how anything else gets rejected.
    """
    import torch

    if layer != lm.n_layers:
        return
    norm = lm.final_norm()
    if norm is None:
        return
    v = torch.as_tensor(np.asarray(vec, dtype=np.float32))
    with torch.no_grad():
        rel = float((norm(v) - v).norm() / v.norm())
    pn = v.norm(dim=-1) if v.ndim > 1 else v.norm()
    cv = float(pn.std() / pn.mean()) if v.ndim > 1 and pn.numel() > 1 else 1.0
    if rel < rel_tol or cv < 1e-3:
        raise PostNormResidual(
            "vector at layer == n_layers is (near) unchanged by the final norm, i.e. it IS the norm "
            f"output, not the last block's residual: applying the norm again moves it by {rel:.2%} "
            f"and its per-position norms vary by {cv:.2%} (a residual's do not). Mean position norm "
            f"{float(pn.mean()):.1f}. Use LM.pre_norm_residual (h40; "
            "tests/test_invariants.py::test_last_hidden_state_is_post_norm)")


def assert_provenance(prov: dict, required: Sequence[str] = ()) -> None:
    """Every field must be PRESENT; the ones that identify the run must also be non-null.

    `template` is recorded but may legitimately be None (a base model has no chat template) -- the
    record of "no template" is itself what stops the untracked raw-text/chat-template mismatch of
    spec §2a. A missing KEY is a different thing from a recorded None.
    """
    required = tuple(required) or ("model", "layers", "pooling", "grid_hash", "code_version",
                                   "tokenizer_padding", "lib_versions", "template")
    absent = [k for k in required if k not in prov]
    empty = [k for k in required if k != "template" and prov.get(k) is None]
    if absent or empty:
        raise ProvenanceIncomplete(
            f"provenance missing {sorted(set(absent + empty))}; a stale stack could be silently reused")


def midrank(scores: Sequence[float], target: int) -> float:
    """Mid-rank on ties. Strict '>' counting awards rank 1 to a wholly tied field, which reads as a
    perfect result; mid-rank reads as chance, which is what nothing-happened should look like."""
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float(ranks[target])


# --------------------------------------------------------------------------------------------
# calibration (spec §5) -- the minimum battery piece 1 needs to make `Claim` honest.
# Piece 2 owns the full five-test battery (self-floor, scale/rotation invariance, known-zero) and
# the cache. What is here is the part without which the Claim contract cannot refuse bug 6.
# --------------------------------------------------------------------------------------------
@dataclass
class CalibrationReport:
    instrument: str
    key: str
    declared_null: float | None = None
    noise_value: float | None = None
    noise_failed: bool = False
    sensitivity_amplitudes: tuple[float, ...] = ()
    sensitivity_values: tuple[float, ...] = ()
    sensitivity_failed: bool = False
    tests_run: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)
    hand_declared_passed: bool | None = None

    @property
    def passed(self) -> bool:
        if self.hand_declared_passed is not None:
            return self.hand_declared_passed
        return not (self.noise_failed or self.sensitivity_failed)

    @property
    def failures(self) -> list[str]:
        out = []
        if self.noise_failed:
            out.append(f"noise: returned {self.noise_value:.4f} against declared null {self.declared_null}")
        if self.sensitivity_failed:
            out.append(f"sensitivity: {np.round(self.sensitivity_values, 4).tolist()} over planted "
                       f"sizes {list(self.sensitivity_amplitudes)} -- not monotone and moving")
        return out

    @classmethod
    def hand_declared(cls, instrument: str, passed: bool, note: str = "") -> "CalibrationReport":
        """An explicitly hand-asserted report, for tests and for instruments whose calibration piece
        2 owns. It is recorded as hand-declared in the ledger so it cannot pass as a measured one."""
        return cls(instrument=instrument, key=f"hand:{instrument}", hand_declared_passed=passed,
                   tests_run=("hand_declared",), notes=[note] if note else [])

    def summary(self) -> str:
        if self.hand_declared_passed is not None:
            return f"hand-declared {'pass' if self.hand_declared_passed else 'FAIL'} ({self.instrument})"
        return (f"{self.instrument}: noise {self.noise_value:.4f} vs null {self.declared_null}"
                f"{' FAIL' if self.noise_failed else ''}; sensitivity "
                f"{np.round(self.sensitivity_values, 4).tolist()}"
                f"{' FAIL' if self.sensitivity_failed else ''}")


def calibration_key(stat: Callable, declared_null, invariances=()) -> str:
    """Hash of the statistic's own source, its declared null and its declared invariances -- not a
    repo-wide version, so that editing one instrument re-calibrates that one only. (spec §5)"""
    try:
        src = inspect.getsource(stat)
    except (OSError, TypeError):
        src = getattr(stat, "__qualname__", repr(stat))
    h = hashlib.sha256()
    h.update(src.encode())
    h.update(repr(declared_null).encode())
    h.update(repr(tuple(invariances)).encode())
    return h.hexdigest()[:16]


def run_calibration(stat: Callable[[np.ndarray], float], *, shape: tuple[int, ...],
                    declared_null: float, plant: Callable[[np.ndarray, float], np.ndarray],
                    name: str = "", amplitudes: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
                    null_tol: float = 0.1, seed: int = 0,
                    invariances: Sequence[str] = ()) -> CalibrationReport:
    """Noise test AND synthetic-signal sensitivity test.

    Noise alone is not a calibration: the broken cross-talk rank returned its chance value 2.0 on
    noise *and* on everything else (h6). A statistic that cannot distinguish planted signal from
    noise fails calibration even when its null is perfect.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=shape)
    noise_value = float(stat(noise))
    noise_failed = abs(noise_value - declared_null) > null_tol

    vals = tuple(float(stat(plant(noise, a))) for a in amplitudes)
    moved = abs(vals[-1] - noise_value) > null_tol
    monotone = all(
        (vals[i + 1] - vals[i]) * (vals[-1] - vals[0]) >= -1e-12 for i in range(len(vals) - 1))
    sensitivity_failed = not (moved and monotone)

    return CalibrationReport(
        instrument=name or getattr(stat, "__name__", "anonymous"),
        key=calibration_key(stat, declared_null, invariances),
        declared_null=declared_null, noise_value=noise_value, noise_failed=noise_failed,
        sensitivity_amplitudes=tuple(amplitudes), sensitivity_values=vals,
        sensitivity_failed=sensitivity_failed,
        tests_run=("noise", "sensitivity"),
        notes=["self-floor, scale/rotation invariance and the known-zero point are piece 2"])
