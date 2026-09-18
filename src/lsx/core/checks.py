"""Refusals and assertions: the mechanisms the rediscovery harness names.

Every exception here is tied to a retraction in `docs/INSTRUMENTS.md` or to a near-miss the spec
records. The rule of the module: an assertion raises with the *numbers* that made it fire, because
the failures this project has had all rendered as plausible numbers and a bare `AssertionError`
would have been read as a flaky test.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import pathlib
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


class UnassertedForward(CoreError):
    """A `Probe` was built on a function that does not run the §7 assertions. Pieces 1-3 each
    recorded `Probe` as a shell whose only job was to wrap a callable; this is the gate that makes
    it a type rather than a label. (spec §3, §7)"""


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


class HeldOutViolated(CoreError):
    """A Direction's fit actually saw something it declared it had never seen. (spec §3)"""


class PassthroughNotComputed(CoreError):
    """A pass-through arm was declared rather than computed from the residual arithmetic. (§2a)"""


class PassthroughNotReproduced(CoreError):
    """A pass-through arm does not reproduce the unpatched readout at zero shift; the ARM is wrong,
    not the claim. (spec §2a)"""


class EffectSizeUnverified(CoreError):
    """A caller-supplied effect size disagrees with the one recomputed from the scores. (spec §4)"""


class CalibrationStale(CoreError):
    """The calibration report was produced by a different version of the statistic, its null or its
    declared invariances. (spec §5)"""


class ProvenanceIncomplete(CoreError):
    """Provenance is missing a field that makes a stack or claim re-identifiable. (spec §7.6)"""


# --- piece 3: refusals that happen at PUBLICATION, not at Claim construction -----------------
# The contract gates publication, not thought (spec §3). These three fire when a Claim tries to
# reach `results/ledger.jsonl`, which is the only place a number becomes a result.
class ProvenanceNotFromStack(CoreError):
    """The Claim's provenance did not come from `extract.build_stack`. Pieces 1 and 2 both flagged
    this as the last structural hole: every assertion in §7 guards the one extraction path, and
    nothing stopped a hand-rolled extraction being wrapped in a well-formed `Claim`."""


class HandDeclaredCalibration(CoreError):
    """The instrument's calibration report was hand-declared rather than measured. Piece 2 stamped
    these so they could not pass for measured ones; refusing them at the ledger is what makes the
    stamp mean something (spec §5)."""


class SweepNotExecuted(CoreError):
    """A Claim whose selection axis was swept carries no core-computed curve, only the caller's
    prose about how the value was chosen. Piece 2 built `Instrument.sweep` and left the decision
    to make it mandatory to piece 3 (spec §4)."""


class LedgerConflict(CoreError):
    """An append would overwrite a different result under the same id, or withdraw an id that is
    not in the ledger. `id` is the provenance hash, so the same experiment re-run is recognised
    rather than duplicated (spec §9)."""


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
# calibration (spec §5) -- the full battery.
#
# Piece 1 shipped noise + sensitivity, which is what the `Claim` contract needed to refuse bug
# 6. Piece 2 EXTENDS that rather than replacing it: the two original tests are unchanged and
# are still the ones that fail first, joined by self-floor, scale/rotation invariance, a
# known-zero point, an explicit degeneracy flag, and the on-disk cache keyed by
# `calibration_key` (source + declared null + declared invariances), so that editing one
# instrument re-calibrates that one alone.
# --------------------------------------------------------------------------------------------

@dataclass
class InvarianceResult:
    name: str
    claimed: bool
    before: float
    after: float

    @property
    def delta(self) -> float:
        return abs(self.after - self.before)

    def failed(self, tol: float) -> bool:
        """Only a CLAIMED invariance can fail. An invariance the instrument explicitly does not
        claim is measured and recorded (so "we never tested it" and "it is genuinely not
        invariant" stay distinguishable), but it cannot fail a calibration."""
        return self.claimed and self.delta > tol


@dataclass
class CalibrationReport:
    instrument: str
    key: str
    declared_null: float | None = None
    noise_value: float | None = None
    noise_sd: float = 0.0
    noise_failed: bool = False
    sensitivity_amplitudes: tuple[float, ...] = ()
    sensitivity_values: tuple[float, ...] = ()
    sensitivity_failed: bool = False
    self_floor_value: float | None = None
    self_floor_failed: bool = False
    invariance: tuple[InvarianceResult, ...] = ()
    invariance_failed: bool = False
    invariance_tol: float = 1e-6
    known_zero_value: float | None = None
    known_zero_failed: bool = False
    degenerate: bool = False
    tests_run: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)
    hand_declared_passed: bool | None = None

    # ---- verdict -------------------------------------------------------------------------
    @property
    def passed(self) -> bool:
        if self.hand_declared_passed is not None:
            return self.hand_declared_passed
        return not bool(self.failures)

    @property
    def hand_declared_report(self) -> bool:
        return self.hand_declared_passed is not None

    @property
    def failures(self) -> list[str]:
        out = []
        if self.noise_failed:
            out.append(f"noise: returned {self.noise_value:.4f} against declared null "
                       f"{self.declared_null}")
        if self.sensitivity_failed:
            out.append(f"sensitivity: {np.round(self.sensitivity_values, 4).tolist()} over planted "
                       f"sizes {list(self.sensitivity_amplitudes)} -- not monotone and moving")
        if self.degenerate:
            out.append("degenerate: the statistic returns the same value on noise and on every "
                       "planted signal, so it cannot distinguish them (h6)")
        if self.self_floor_failed:
            out.append(f"self-floor: the instrument on its own floor as treatment returned "
                       f"{self.self_floor_value:.4f}, not its null {self.declared_null}")
        if self.invariance_failed:
            bad = [f"{r.name} moved the statistic by {r.delta:.4g}"
                   for r in self.invariance if r.failed(self.invariance_tol)]
            out.append("invariance: claimed but not held -- " + "; ".join(bad))
        if self.known_zero_failed:
            out.append(f"known-zero: the analytically-zero configuration returned "
                       f"{self.known_zero_value:.6g}, not {self.declared_null}")
        return out

    @classmethod
    def hand_declared(cls, instrument: str, passed: bool, note: str = "") -> "CalibrationReport":
        """An explicitly hand-asserted report, for tests and for instruments whose calibration is
        not built. It is recorded as hand-declared in the ledger so it cannot pass as a measured
        one -- and piece 3's ledger should refuse to grade a §1A target on one."""
        return cls(instrument=instrument, key=f"hand:{instrument}", hand_declared_passed=passed,
                   tests_run=("hand_declared",), notes=[note] if note else [])

    def summary(self) -> str:
        if self.hand_declared_passed is not None:
            return f"hand-declared {'pass' if self.hand_declared_passed else 'FAIL'} ({self.instrument})"
        inv = ", ".join(f"{r.name}{'' if r.claimed else ' (not claimed)'} d={r.delta:.3g}"
                        for r in self.invariance) or "none"
        return (f"{self.instrument} [{self.key}] {len(self.tests_run)} tests "
                f"{'PASS' if self.passed else 'FAIL'}: noise {self.noise_value:.4f}"
                f"+-{self.noise_sd:.4f} vs null {self.declared_null}"
                f"{' FAIL' if self.noise_failed else ''}; sensitivity "
                f"{np.round(self.sensitivity_values, 4).tolist()}"
                f"{' FAIL' if self.sensitivity_failed else ''}; self-floor "
                f"{'n/a' if self.self_floor_value is None else format(self.self_floor_value, '.4f')}"
                f"{' FAIL' if self.self_floor_failed else ''}; invariance {inv}"
                f"{' FAIL' if self.invariance_failed else ''}; known-zero "
                f"{'n/a' if self.known_zero_value is None else format(self.known_zero_value, '.3g')}"
                f"{' FAIL' if self.known_zero_failed else ''}"
                f"{'; DEGENERATE' if self.degenerate else ''}")

    # ---- cache shape ---------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "invariance"}
        d["invariance"] = [dict(r.__dict__) for r in self.invariance]
        for k in ("sensitivity_amplitudes", "sensitivity_values", "tests_run"):
            d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationReport":
        d = dict(d)
        d["invariance"] = tuple(InvarianceResult(**r) for r in d.get("invariance", []))
        for k in ("sensitivity_amplitudes", "sensitivity_values", "tests_run"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def _source_closure(fn: Callable, _seen: set | None = None) -> list[str]:
    """The source of `fn` AND of every function it calls that this package defines, transitively.

    Piece 4 found `calibration_key` hashing only the top-level source, and the failure is the one
    the key exists to prevent. `discrimination_rho` is one line -- `mean(discrimination_per_item(f))`
    -- so changing what `discrimination_per_item` computes, or changing `spearman` underneath it,
    left the key identical and every cached report valid. It happened in this piece: the tie
    tolerance that fixed a dead readout scoring +-0.55 changed the arithmetic and not the key. The
    same hole covers `midrank` and `cosine_scores`, which three shipped instruments delegate to.

    Still NOT a repo-wide version, which §5 forbids for good reason: the closure is only what this
    statistic actually calls, so editing `selector`'s ranking re-calibrates the instruments that
    use it and no others. Functions from outside `lsx` (numpy) are not followed -- their version is
    provenance, recorded in `lib_versions`, not a calibration key.
    """
    seen = _seen if _seen is not None else set()
    key = getattr(fn, "__module__", ""), getattr(fn, "__qualname__", repr(fn))
    if key in seen:
        return []
    seen.add(key)
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return [f"{key[0]}.{key[1]}"]
    out = [src]
    globs = getattr(fn, "__globals__", {})
    for name in sorted(_referenced_names(getattr(fn, "__code__", None))):
        dep = globs.get(name)
        if callable(dep) and str(getattr(dep, "__module__", "")).startswith("lsx."):
            out.extend(_source_closure(dep, seen))
    return out


def _referenced_names(code, _depth: int = 0) -> set:
    """Every global name a code object reads, INCLUDING inside its nested code objects.

    Comprehensions, lambdas and inner functions each compile to their own code object, and their
    names do not appear in the enclosing `co_names`. This bit the first version of the closure
    within the hour: `selector_rank` calls `midrank` inside a list comprehension, so a walker that
    read only the top-level `co_names` found `cosine_scores` and missed `midrank` -- i.e. editing
    the tie rule that h34 turned on would still not have re-calibrated the instrument that depends
    on it. The fix for a hole that hid a dependency was itself hiding a dependency.
    """
    if code is None or _depth > 8:
        return set()
    names = set(getattr(code, "co_names", ()))
    for const in getattr(code, "co_consts", ()):
        if hasattr(const, "co_names"):
            names |= _referenced_names(const, _depth + 1)
    return names


def calibration_key(stat: Callable, declared_null, invariances=()) -> str:
    """Hash of the statistic's source CLOSURE, its declared null and its declared invariances --
    not a repo-wide version, so that editing one instrument re-calibrates that one only. (spec §5)

    "Closure" rather than "own source" since piece 4: see `_source_closure` for the hole that was,
    and for why the boundary is `lsx.` rather than everything importable.
    """
    h = hashlib.sha256()
    for src in _source_closure(stat):
        h.update(src.encode())
    h.update(repr(declared_null).encode())
    h.update(repr(tuple(invariances)).encode())
    return h.hexdigest()[:16]


def run_calibration(stat: Callable, *, shape: tuple[int, ...] | None = None,
                    declared_null: float, plant: Callable,
                    name: str = "", amplitudes: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
                    null_tol: float = 0.1, seed: int = 0,
                    invariances: Sequence[str] = (), not_invariances: Sequence[str] = (),
                    noise_fn: Callable | None = None,
                    transforms: dict | None = None,
                    self_floor: Callable | None = None,
                    known_zero: Callable | None = None,
                    noise_repeats: int = 8,
                    invariance_tol: float = 1e-6,
                    known_zero_tol: float = 1e-9) -> CalibrationReport:
    """The §5 battery. Piece 1's two tests, unchanged, plus three more and a degeneracy flag.

    * **noise** -- `noise_repeats` independent draws of matched shape and scale; the statistic must
      return its declared null within `null_tol`. The spread of those draws is also the measured
      run-to-run noise of the statistic and is reported.
    * **sensitivity** -- a planted effect of known size must move the statistic, monotonically in
      the planted size. Noise alone is not enough: h6's cross-talk rank returned its chance value
      2.0 on noise *and* on everything else.
    * **degeneracy** -- if the statistic returns the *same number* across noise draws and across
      every planted amplitude, it is flagged degenerate outright. This is spec §1B bug 6's word
      ("flags it as degenerate") made a field, rather than an inference from the sensitivity line.
    * **self-floor** -- the instrument applied to its own floor as treatment must read its null.
    * **invariance** -- each claimed invariance is applied to a fixture WITH planted signal (a
      statistic sitting at its null is trivially invariant to everything) and must not move the
      statistic; each explicitly *unclaimed* invariance is applied and recorded, so "untested" and
      "not invariant" stay distinguishable in the report.
    * **known-zero** -- a configuration whose answer is analytically the null, checked at 1e-9.

    `noise_fn`/`transforms`/`self_floor`/`known_zero` let the fixture be something richer than a
    bare array (activations *and* candidate directions, say). Without them this is exactly piece
    1's call signature running exactly piece 1's two tests.
    """
    rng = np.random.default_rng(seed)
    if noise_fn is None:
        if shape is None:
            raise ValueError("run_calibration needs either a `shape` or a `noise_fn`")

        def noise_fn(r, _shape=shape):
            return r.normal(size=_shape)

    draws = [noise_fn(np.random.default_rng(seed + 1000 * i)) for i in range(max(noise_repeats, 1))]
    noise_values = [float(stat(x)) for x in draws]
    noise = draws[0]
    noise_value = float(np.mean(noise_values))
    noise_sd = float(np.std(noise_values))
    noise_failed = abs(noise_value - declared_null) > null_tol

    vals = tuple(float(stat(plant(noise, a))) for a in amplitudes)
    moved = abs(vals[-1] - noise_values[0]) > null_tol
    monotone = all(
        (vals[i + 1] - vals[i]) * (vals[-1] - vals[0]) >= -1e-12 for i in range(len(vals) - 1))
    sensitivity_failed = not (moved and monotone)
    degenerate = bool(np.ptp(np.asarray(noise_values + list(vals))) < 1e-12)

    tests = ["noise", "sensitivity", "degeneracy"]

    self_floor_value = None
    self_floor_failed = False
    if self_floor is not None:
        self_floor_value = float(stat(self_floor(plant(noise, amplitudes[-1]), rng)))
        self_floor_failed = abs(self_floor_value - declared_null) > null_tol
        tests.append("self_floor")

    results: list[InvarianceResult] = []
    if transforms:
        base_signal = plant(noise, amplitudes[-1])
        before = float(stat(base_signal))
        for inv_name, fn in transforms.items():
            after = float(stat(fn(base_signal, np.random.default_rng(seed + 7))))
            results.append(InvarianceResult(inv_name, inv_name in tuple(invariances), before, after))
        tests.append("invariance")
    invariance_failed = any(r.failed(invariance_tol) for r in results)

    known_zero_value = None
    known_zero_failed = False
    if known_zero is not None:
        known_zero_value = float(stat(known_zero(noise)))
        known_zero_failed = abs(known_zero_value - declared_null) > known_zero_tol
        tests.append("known_zero")

    notes = []
    for r in results:
        if not r.claimed:
            verdict = ("genuinely not invariant, as declared" if r.delta > invariance_tol else
                       "no measurable effect -- the NOT-invariance declaration may be wrong")
            notes.append(f"{r.name}: not claimed as an invariance; measured change "
                         f"{r.delta:.4g} ({verdict})")
    absent = [t for t, present in (("self_floor", self_floor is not None),
                                   ("invariance", bool(transforms)),
                                   ("known_zero", known_zero is not None)) if not present]
    if absent:
        notes.append("battery incomplete, tests not run: " + ", ".join(absent))

    return CalibrationReport(
        instrument=name or getattr(stat, "__name__", "anonymous"),
        key=calibration_key(stat, declared_null, invariances),
        declared_null=declared_null, noise_value=noise_value, noise_sd=noise_sd,
        noise_failed=noise_failed,
        sensitivity_amplitudes=tuple(amplitudes), sensitivity_values=vals,
        sensitivity_failed=sensitivity_failed,
        self_floor_value=self_floor_value, self_floor_failed=self_floor_failed,
        invariance=tuple(results), invariance_failed=invariance_failed,
        invariance_tol=invariance_tol,
        known_zero_value=known_zero_value, known_zero_failed=known_zero_failed,
        degenerate=degenerate,
        tests_run=tuple(tests), notes=notes)


# --------------------------------------------------------------------------------------------
# the calibration cache (spec §5): keyed by the instrument's own source + null + invariances, so
# editing one instrument re-calibrates that one and only that one.
# --------------------------------------------------------------------------------------------
CALIBRATION_DIR = pathlib.Path(__file__).resolve().parents[3] / "results" / "calibration"


def calibration_path(instrument: str, key: str, cache_dir=None) -> pathlib.Path:
    return pathlib.Path(cache_dir or CALIBRATION_DIR) / f"{instrument}.{key}.json"


def load_calibration(instrument: str, key: str, cache_dir=None) -> "CalibrationReport | None":
    """The cached report for exactly this key, or None. A report under a DIFFERENT key is not
    returned and not deleted: it is the record of what the previous version of the statistic did,
    and a stale report must never be silently reused (spec §5)."""
    p = calibration_path(instrument, key, cache_dir)
    if not p.exists():
        return None
    try:
        rep = CalibrationReport.from_dict(json.loads(p.read_text()))
    except Exception:  # noqa: BLE001 -- a corrupt cache must re-calibrate, never crash
        return None
    return rep if rep.key == key else None


def save_calibration(report: CalibrationReport, cache_dir=None) -> pathlib.Path:
    p = calibration_path(report.instrument, report.key, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), indent=1, sort_keys=True, default=float))
    return p


def cached_calibration(instrument: str, key: str, run: Callable, *, cache_dir=None,
                       refresh: bool = False) -> CalibrationReport:
    if not refresh:
        hit = load_calibration(instrument, key, cache_dir)
        if hit is not None:
            return hit
    rep = run()
    if rep.key != key:
        raise CalibrationStale(
            f"{instrument}: the report came back under key {rep.key} but the instrument hashes to "
            f"{key}; the statistic, its null or its invariances changed mid-calibration")
    save_calibration(rep, cache_dir)
    return rep


def top1_hit(scores: Sequence[float], target: int) -> float:
    """Did the argmax land on `target`? TIES SPLIT THE HIT (1/T), never winner-takes-first.

    This is `midrank`'s rule in the accuracy family, and it matters for the same reason. Written as
    `int(np.argmax(s) == target)`, a wholly tied field awards a full hit to whichever candidate the
    sort returns first -- so a dead readout scores 1.0 on the items whose target happens to sort
    first and 0.0 on the rest, and the mean is 1/k only by luck of the labelling. h34 is exactly
    this bug in the rank family. Splitting the hit makes a tied field read 1/k *per item*, which is
    what nothing-happened should look like.
    """
    s = np.asarray(scores, dtype=np.float64)
    top = np.flatnonzero(s == s.max())
    return float(1.0 / len(top)) if int(target) in set(top.tolist()) else 0.0


def dot_tie_atol(d: int, scale: float, safety: float = 4.0) -> float:
    """Below what difference are two float64 dot products of length `d` the SAME number?

    A measured bound, not a chosen epsilon. The rounding error of a length-`d` float64 dot product
    is bounded by about `d * eps * sum|x_i y_i|`, which `scale` stands in for; `safety` is the only
    free quantity and it is a small integer, not a threshold tuned to make anything pass. At
    Gemma's d=3584 with residual projections of order 20 this is ~1e-11 -- eleven orders of
    magnitude below any difference a readout could mean -- so it can only ever merge values that
    arithmetic, not the model, separated.

    Why this exists: see `spearman`. A flat tolerance anywhere in this core is a bug (piece 2), and
    this is the alternative -- a tolerance computed from the arithmetic that produced the numbers.
    """
    return float(safety) * int(d) * float(np.finfo(np.float64).eps) * abs(float(scale))


def _midranks(x: np.ndarray, atol: float = 0.0) -> np.ndarray:
    """Ranks of `x` with ties averaged -- the vector form of `midrank`'s tie rule.

    `atol` merges values that differ by less than a numerical-noise bound (see `dot_tie_atol`).
    Merging is over CONSECUTIVE values in sorted order, so it chains; at the magnitudes `atol` is
    ever set to that is a distinction without a difference, and at atol=0 (the default) this is
    exact equality and identical to `midrank`.
    """
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(x[order[j + 1]] - x[order[j]]) <= atol:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float], atol: float = 0.0) -> float:
    """Spearman rank correlation, with ties mid-ranked and a DEAD READOUT READING ZERO.

    Two things here are not standard, and both are the h34 lesson in this statistic's costume.

    **A constant vector reads 0, not NaN.** If either vector is constant its ranks are all equal,
    the Pearson correlation of the ranks is 0/0, and numpy returns NaN. A statistic that returns
    NaN on a dead readout is one `nanmean` away from being a statistic that quietly drops its dead
    items, and one that returned 1.0 there would be h34 exactly. Zero is the honest value.

    **`atol` decides what "constant" means, and it must not be zero on real numbers.** This is a
    bug the known-zero test caught in piece 4 before anything was published. The fixture whose
    answer is analytically zero gives every level of a subject the IDENTICAL activation vector --
    and the identical vectors, run through one batched matmul, came back differing by 4.4e-16,
    because BLAS does not promise the same summation order for every row of a batch. Spearman does
    not care how small a difference is: it ranks it. So a wholly dead readout scored +0.55 on some
    subjects and -0.55 on others, and the instrument read ~0 only because the signs happened to
    cancel in the mean. With an asymmetric rounding pattern -- or with real near-tied activations,
    which is what a dead readout looks like on real data -- it would have reported a clean
    correlation out of pure floating-point noise. `dot_tie_atol` is the bound that closes it.
    """
    rx = _midranks(np.asarray(x, dtype=np.float64), atol)
    ry = _midranks(np.asarray(y, dtype=np.float64))
    sx, sy = rx.std(), ry.std()
    if sx <= 0 or sy <= 0:
        return 0.0
    return float(((rx - rx.mean()) @ (ry - ry.mean())) / (len(rx) * sx * sy))
