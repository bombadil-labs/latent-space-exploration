"""The §3 types and the §4 `Claim` contract.

`Claim` is the only exportable type. Everything else either feeds it or is explicitly
un-ledgerable. The contract gates *publication*, not thought: `Sketch` prints freely and can never
reach the ledger, because a `Readout` that could not be printed at all would be bypassed with
`print(scores.mean())` within a day.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import registry
from .checks import (ArmOffNull, CalibrationFailed, CalibrationReport, CalibrationStale,
                     EffectSizeUnverified, HeldOutNotDeclared, HeldOutViolated, MissingArm,
                     MissingCalibration, MissingFloor, NullDeclaredLate, PassthroughNotComputed,
                     RawScoreOnLeakyGrid, SelectionOnScoringData)

# Plumbing arms are declared in the REGISTRY, not by the caller and no longer by a placeholder
# table here (spec §4, §8). This name is kept because the harness and the tests read it, but it is
# now a view onto `registry.REGISTRY`, which also carries each instrument's null, the measured
# tolerance around it, and its claimed invariances.
REQUIRED_ARMS: dict[str, tuple[str, ...]] = {
    name: spec.required_arms for name, spec in registry.REGISTRY.items()}

_SEQ = itertools.count()

# Piece 1's default, kept ONLY as the fallback for arms whose instrument has no measured null
# spread. Every registry instrument supplies its own, from `registry.InstrumentSpec.arm_tolerance`.
LEGACY_TOLERANCE = 0.15


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Item:
    text: str
    factors: dict[str, str]
    spans: dict[str, tuple[int, int]]

    def __post_init__(self):
        for name, (i, j) in self.spans.items():
            if not (0 <= i < j <= len(self.text)):
                raise ValueError(f"span {name!r} = ({i}, {j}) is not inside a text of "
                                 f"{len(self.text)} characters")


@dataclass
class LeakReport:
    """What the stimulus gives away before any model runs (spec §6)."""
    recoverability: dict[str, float] = field(default_factory=dict)
    null_value: dict[str, float] = field(default_factory=dict)   # permutation null of the same LOO
    position_corr: dict[str, float] = field(default_factory=dict)
    flagged: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)
    computed: bool = True

    @property
    def leaky(self) -> bool:
        return bool(self.flagged)

    def summary(self) -> str:
        if not self.computed:
            return "leak report not computed"
        bits = [f"{k} recoverable {v:.2f} (permutation null {self.null_value[k]:.2f})"
                for k, v in sorted(self.recoverability.items())]
        pos = [f"{k} position r={v:+.2f}" for k, v in sorted(self.position_corr.items())]
        return "; ".join(bits + pos) + (f"; FLAGGED {list(self.flagged)}" if self.flagged else
                                        "; nothing flagged")


_WORD = re.compile(r"[a-z0-9']+")


def _bag(texts: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    vocab = sorted({w for t in texts for w in _WORD.findall(t.lower())})
    idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(texts), len(vocab)))
    for r, t in enumerate(texts):
        for w in _WORD.findall(t.lower()):
            X[r, idx[w]] += 1.0
    return X, vocab


def _loo_accuracy(K: np.ndarray, y: np.ndarray, n_levels: int, ridge: float) -> float:
    """Leave-one-item-out kernel ridge classification accuracy. Kernel (n x n) form because the
    vocabulary is much larger than the grid."""
    n = len(y)
    Y = np.eye(n_levels)[y]
    correct = 0
    for i in range(n):
        tr = np.array([k for k in range(n) if k != i])
        A = K[np.ix_(tr, tr)] + ridge * np.eye(n - 1)
        alpha = np.linalg.solve(A, Y[tr])
        correct += int(np.argmax(K[i, tr] @ alpha) == y[i])
    return correct / n


def bag_of_tokens_recoverability(texts: Sequence[str], labels: Sequence[str], ridge: float = 1.0,
                                 null_draws: int = 20, seed: int = 0) -> tuple[float, float]:
    """Leave-one-item-out ridge on one-hot token counts: how well the label is predicted from the
    bag alone -- reported BESIDE its own permutation null, never against nominal chance.

    The null matters. Leave-one-out on balanced labels is anti-predictive by construction: removing
    an item tips the remaining majority the other way, so a grid carrying no information at all
    scores 0.00, not 0.50. Reading that 0.00 against a nominal chance of 0.50 would flag a clean
    grid and, on an unbalanced one, hide a real leak. This is h32's lesson applied to the leak
    detector itself: estimate the instrument's floor before believing its reading.

    For RoPE models layer 0 *is* the static embedding bag, so this is also the layer-0 check and the
    report says so rather than pretending to two measurements. (spec §6)

    Returns (accuracy, permutation null mean).
    """
    X, _ = _bag(texts)
    X = np.hstack([X, np.ones((len(X), 1))])
    K = X @ X.T
    levels = sorted(set(labels))
    y = np.array([levels.index(l) for l in labels])
    acc = _loo_accuracy(K, y, len(levels), ridge)
    rng = np.random.default_rng(seed)
    null = [_loo_accuracy(K, rng.permutation(y), len(levels), ridge) for _ in range(null_draws)]
    return acc, float(np.mean(null))


@dataclass
class Grid:
    """Stimuli + factors/levels + marked spans + a leak report computed at construction."""
    items: list[Item]
    name: str = ""
    leak_check: bool = True
    leak_margin: float = 0.2
    null_draws: int = 20
    position_margin: float = 0.6
    declared_leaks: tuple[str, ...] = ()
    leak: LeakReport = field(init=False)
    hash: str = field(init=False)

    def __post_init__(self):
        if not self.items:
            raise ValueError("a Grid needs items")
        names = {tuple(sorted(it.factors)) for it in self.items}
        if len(names) != 1:
            raise ValueError(f"items disagree about their factors: {names}")
        spans = {tuple(sorted(it.spans)) for it in self.items}
        if len(spans) != 1:
            raise ValueError(f"items disagree about their spans: {spans}")
        self.hash = _hash([[it.text, it.factors, it.spans] for it in self.items])
        self.leak = self._leak_report() if self.leak_check else LeakReport(computed=False)

    @property
    def factors(self) -> list[str]:
        return sorted(self.items[0].factors)

    @property
    def span_names(self) -> list[str]:
        return sorted(self.items[0].spans)

    def levels(self, factor: str) -> list[str]:
        return sorted({it.factors[factor] for it in self.items})

    def _leak_report(self) -> LeakReport:
        texts = [it.text for it in self.items]
        rec, nulls, pos, flagged = {}, {}, {}, []
        notes = ["layer 0 is the static embedding bag for RoPE models, so bag-of-tokens IS the "
                 "layer-0 check; one measurement, not two (spec §6)"]
        notes.append("recoverability is reported against a permutation null of the same "
                     "leave-one-out procedure, not against nominal chance: LOO on balanced labels "
                     "is anti-predictive by construction")
        for f in self.factors:
            labels = [it.factors[f] for it in self.items]
            acc, null = bag_of_tokens_recoverability(texts, labels, null_draws=self.null_draws)
            rec[f], nulls[f] = acc, null
            if acc - null > self.leak_margin and f not in self.declared_leaks:
                flagged.append(f)
            elif null - acc > self.leak_margin:
                # An anti-predictive bag is invertible, so it IS information -- but under
                # leave-one-out an exactly paired, exactly balanced grid is anti-predictive by
                # construction (the held-out item's twin carries the opposite label), which is the
                # shape of this project's own lexical repair (h35's v3). Recorded, not flagged:
                # flagging every repaired grid would retire the check by alarm fatigue. What it
                # means is that the LABEL STRUCTURE, not the prose, is recoverable, and any probe
                # trained across items -- including a layer-0 one -- can exploit it.
                notes.append(f"{f}: bag is anti-predictive ({acc:.2f} vs null {null:.2f}); the grid "
                             "is exactly paired/balanced, so the label is recoverable by inversion. "
                             "Not flagged as a stimulus leak; check the design before a per-item "
                             "cross-validated claim.")
            # position balance: does the level correlate with span start / item length? (h2)
            lv = sorted(set(labels))
            y = np.array([lv.index(l) for l in labels], dtype=float)
            starts = np.array([min((s for s, _ in it.spans.values()), default=0)
                               for it in self.items], dtype=float)
            lens = np.array([len(it.text) for it in self.items], dtype=float)
            for tag, x in (("start", starts), ("length", lens)):
                if x.std() > 0 and y.std() > 0:
                    r = float(np.corrcoef(x, y)[0, 1])
                    pos[f"{f}~{tag}"] = r
                    if abs(r) > self.position_margin and f not in flagged and f not in self.declared_leaks:
                        flagged.append(f)
        return LeakReport(rec, nulls, pos, tuple(dict.fromkeys(flagged)), notes)


# --------------------------------------------------------------------------------------------
# Stack / Direction / Readout / Probe / Sketch
# --------------------------------------------------------------------------------------------
# The exact provenance fields `extract.build_stack` writes. The stack signature is computed over
# THESE keys only, so that a Claim may add its own (calibration_key, the direction's held-out axis)
# without invalidating the binding between the Claim and the extraction it came from.
STACK_PROV_KEYS: tuple[str, ...] = (
    "model", "layers", "pooling", "grid_hash", "grid_name", "code_version", "tokenizer_padding",
    "span_policy", "template", "lib_versions", "batch_size", "leak", "acts_digest",
    "equivalence_min_cos", "equivalence_item_rule")


def stack_signature(prov: dict) -> str:
    """A hash over the stack's own provenance fields, including a digest of the activations.

    This is not a security boundary -- anyone who reads this file can call it -- and it is not
    meant to be one. It is the thing that was missing: a Claim that reaches the ledger must carry
    provenance that a real `build_stack` wrote, so an extraction that never ran the §7 assertions
    cannot be wrapped in a well-formed Claim by accident. Editing any recorded field afterwards
    (the model, the padding side, the library versions) breaks the signature rather than the
    silence.
    """
    core = {k: prov.get(k) for k in STACK_PROV_KEYS if k in prov}
    return _hash({"stack_v1": core})


@dataclass
class Stack:
    """Grid x Model x layers -> activations. Built only by `lsx.core.extract.build_stack`."""
    acts: np.ndarray                  # [item, span, layer, d]
    grid_hash: str
    span_names: tuple[str, ...]
    layers: tuple[int, ...]
    provenance: dict
    checks: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.acts.ndim != 4:
            raise ValueError(f"acts must be [item, span, layer, d], got shape {self.acts.shape}")

    def vectors(self, span: str, layer: int) -> np.ndarray:
        return self.acts[:, self.span_names.index(span), self.layers.index(layer)]


@dataclass
class Direction:
    """Stack -> vector, ALWAYS leave-one-out over a declared axis. There is no constructor that
    fits on everything (spec §3)."""
    vecs: dict[int, np.ndarray]
    held_out: dict
    provenance: dict = field(default_factory=dict)
    fit_witness: dict | None = None   # {'axis': ..., 'fit_values': [...]} recorded by the fitter

    def __post_init__(self):
        if not self.held_out or "axis" not in self.held_out or "unseen" not in self.held_out:
            raise HeldOutNotDeclared(
                "Direction requires held_out={'axis': ..., 'unseen': ...}: what this direction "
                "never saw. h1-h3 fitted on everything and scored on the same data.")
        # Piece 1 declared the held-out axis and never verified it. A witness is what the fitter
        # actually used; when one is present the declaration is CHECKED against it, and a fit that
        # saw what it said it had not seen is refused rather than recorded.
        if self.fit_witness:
            axis = self.fit_witness.get("axis")
            if axis != self.held_out["axis"]:
                raise HeldOutViolated(
                    f"the fit was left-one-out over {axis!r} but the direction declares its "
                    f"held-out axis as {self.held_out['axis']!r}")
            seen = set(self.fit_witness.get("fit_values", ()))
            overlap = sorted(seen & set(self.held_out["unseen"]))
            if overlap:
                raise HeldOutViolated(
                    f"the direction declares it never saw {sorted(self.held_out['unseen'])} along "
                    f"{axis!r}, but the fit used {overlap}. The declaration was the only thing "
                    "standing between this and h1-h3, and it was never checked until now.")

    def vec(self, layer: int) -> np.ndarray:
        return self.vecs[layer]

    @property
    def verified(self) -> bool:
        """True when the fit recorded a witness and the declaration was checked against it."""
        return bool(self.fit_witness)


def fit_leave_one_out(vectors_by_value: dict, unseen, axis: str, layer: int = 0) -> Direction:
    """The sanctioned fitter: mean of the vectors of every value of `axis` EXCEPT `unseen`, with a
    witness recording exactly which values were used. There is no constructor that fits on
    everything (spec §3), and now no way to claim a hold-out the fit did not honour.
    """
    unseen = set(np.atleast_1d(np.asarray(list(unseen), dtype=object)).tolist())
    used = [v for v in vectors_by_value if v not in unseen]
    if not used:
        raise HeldOutViolated(f"holding out {sorted(unseen)} along {axis!r} leaves nothing to fit on")
    vec = np.mean([np.mean(np.atleast_2d(vectors_by_value[v]), axis=0) for v in used], axis=0)
    return Direction(vecs={layer: vec}, held_out={"axis": axis, "unseen": set(unseen)},
                     fit_witness={"axis": axis, "fit_values": sorted(used, key=str)})


@dataclass
class Measured:
    """Scores with a construction-order stamp, so 'declared before the treatment was computed' is
    enforced by construction rather than by discipline (spec §4)."""
    values: np.ndarray
    label: str = ""
    seq: int = field(default_factory=lambda: next(_SEQ))

    def __post_init__(self):
        self.values = np.atleast_1d(np.asarray(self.values, dtype=np.float64))

    @property
    def value(self) -> float:
        return float(np.mean(self.values))

    @property
    def n(self) -> int:
        return int(self.values.size)


class Sketch:
    """Explicitly UN-LEDGERABLE. Prints freely, can never reach the ledger or a generated document.

    The contract gates publication, not thought.
    """
    ledgerable = False

    def __init__(self, values, label: str = ""):
        self.values = np.atleast_1d(np.asarray(values, dtype=np.float64))
        self.label = label

    def __repr__(self) -> str:
        v = self.values
        return (f"<Sketch {self.label or 'unnamed'} n={v.size} mean={v.mean():.4f} "
                f"sd={v.std():.4f} -- NOT a result, cannot be ledgered>")

    def measured(self, label: str = "") -> Measured:
        """Promote to a ledgerable measurement. Deliberately explicit: a stamp happens here."""
        return Measured(self.values, label or self.label)


@dataclass
class Readout:
    """Direction x Stack -> scores. No model call."""
    direction: Direction
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    layer: int

    def __call__(self, stack: Stack, span: str) -> Sketch:
        return Sketch(self.fn(stack.vectors(span, self.layer), self.direction.vec(self.layer)),
                      label=f"readout@{self.layer}")


@dataclass
class Probe:
    """Direction x Model -> scores (patched forward or generation). Goes through the one asserted
    path in `lsx.core.extract`, never through a bare model call (spec §7)."""
    direction: Direction
    patch_layer: int
    readout_layer: int
    fn: Callable

    def __call__(self, *a, **kw) -> Sketch:
        return Sketch(self.fn(*a, **kw), label=f"probe {self.patch_layer}->{self.readout_layer}")


# --------------------------------------------------------------------------------------------
# Claim
# --------------------------------------------------------------------------------------------
@dataclass
class Arm:
    """A control arm. It declares WHERE IT SHOULD SIT; an arm off its null raises (spec §4).

    Under h34 the missing arm was only half the failure: no_patch read a respectable 2.00 and what
    actually screamed was the *random* arm at 1.22, with nobody looking.
    """
    scores: np.ndarray
    expected_null: float | None = None
    tolerance: float | None = None     # None -> the instrument's MEASURED tolerance at this arm's n
    justification: str = ""
    seq: int = field(default_factory=lambda: next(_SEQ))

    def __post_init__(self):
        self.scores = np.atleast_1d(np.asarray(self.scores, dtype=np.float64))
        if self.expected_null is None:
            raise ArmOffNull("every arm must declare where it should sit (expected_null); "
                             "an arm with no declared null cannot be off it")

    @property
    def value(self) -> float:
        return float(np.mean(self.scores))

    @property
    def n(self) -> int:
        return int(self.scores.size)

    @property
    def resolved_tolerance(self) -> float:
        """Piece 1's unmeasured 0.15 survives only as the fallback for an arm on an instrument the
        registry does not know, and it is named as such wherever it is used."""
        return LEGACY_TOLERANCE if self.tolerance is None else float(self.tolerance)

    @property
    def off_null(self) -> bool:
        return abs(self.value - self.expected_null) > self.resolved_tolerance


@dataclass
class Floor:
    """BOTH fields required. Stage 28 would have passed a stimulus-only check while sitting at its
    estimator floor, which is exactly how it published three false negatives."""
    stimulus: float | None
    estimator: float | None

    def __post_init__(self):
        missing = [k for k in ("stimulus", "estimator") if getattr(self, k) is None]
        if missing:
            raise MissingFloor(f"Floor requires both stimulus and estimator; missing {missing}. "
                               "A stimulus floor says what the text gives away; an estimator floor "
                               "says what the statistic returns on its own noise (h28).")


_CHOOSING = re.compile(r"\b(argmax|argmin|best|peak|max over|min over|chosen|choose|select)\b", re.I)


@dataclass
class Selection:
    """What was swept and how the reported value was chosen (spec §4).

    `rule` is prose, and a regex over prose is the weakest check in the contract: a caller who
    writes "we looked at the curve and quoted layer 16" passes it. `from_sweep` is the honest
    path -- the core runs the sweep and records the curve it computed, so the rule is a FACT about
    what happened rather than a description of it. `curve` is what piece 3's ledger should require
    of any claim whose axis was swept.
    """
    axis: str | None
    rule: str
    held_out: bool = False
    curve: dict | None = None

    @classmethod
    def from_sweep(cls, axis: str, curve: dict) -> "Selection":
        """Recorded by `Instrument.sweep`, never written by hand."""
        return cls(axis=axis, curve=dict(curve), held_out=True,
                   rule=f"full curve reported: {len(curve)} points on {axis!r}, swept and recorded "
                        f"by the core (not selected)")

    @property
    def executed(self) -> bool:
        return self.curve is not None

    @property
    def is_a_choice(self) -> bool:
        if self.executed:
            return False      # the core ran the sweep and reported all of it; there is no choice
        return bool(self.axis) and bool(_CHOOSING.search(self.rule))


@dataclass
class EffectSize:
    """Piece 1 recorded this and asserted nothing about it: the z could be anything.

    `against()` computes it from the scores and the declared null and stamps `computed=True`. A
    hand-supplied effect size is still accepted -- there are legacy shapes the core cannot
    recompute -- but for a registry instrument the `Claim` now RECOMPUTES the z from the treatment
    scores and refuses if the two disagree about the verdict: a different sign, or significance
    claimed where the recomputation finds none. The exact value may legitimately differ (a caller's
    standard error can come from paraphrases rather than from items); the verdict may not.
    """
    size: float
    n: int
    z: float
    computed: bool = False
    against_null: float | None = None

    @classmethod
    def against(cls, values, null: float, *, fallback_item_sd: float | None = None
                ) -> "EffectSize":
        v = np.atleast_1d(np.asarray(values, dtype=np.float64))
        n = int(v.size)
        size = float(v.mean() - null)
        sd = float(v.std(ddof=1)) if n > 1 else 0.0
        # A constant array's std is not reliably 0.0 -- cancellation leaves ~1e-16, which turns a
        # zero-spread arm into a z of 1e14. Anything that small IS zero spread.
        if sd <= 1e-12 * max(1.0, abs(float(v.mean()))):
            sd = 0.0
        if sd <= 0:
            # A constant arm has no spread of its own; fall back to the instrument's measured
            # per-item null spread rather than reporting an infinite z.
            sd = float(fallback_item_sd) if fallback_item_sd else 0.0
        se = sd / np.sqrt(n) if sd > 0 and n > 0 else 0.0
        z = 0.0 if se == 0 else size / se
        return cls(size=size, n=n, z=float(z), computed=True, against_null=float(null))

    @property
    def verdict(self) -> int:
        """-1 / 0 / +1: significantly below its null, indistinguishable, significantly above."""
        return 0 if abs(self.z) < 2.0 else int(np.sign(self.z))


@dataclass
class Claim:
    """The ONLY exportable type. Construction is the contract."""
    instrument: str
    treatment: Measured
    arms: dict[str, Arm]
    floor: Floor | None
    selection: Selection
    effect: EffectSize
    calibration: CalibrationReport | None
    provenance: dict
    semantic_null: Arm | None = None
    grid: Grid | None = None
    report_as: str = "raw"
    patch_layer: int | None = None
    readout_layer: int | None = None
    companion: "Claim | None" = None
    stage: str = ""
    config: dict = field(default_factory=dict)   # e.g. {"n_candidates": 6}: fixes the null
    notes: list[str] = field(default_factory=list)
    id: str = field(init=False, default="")

    def __post_init__(self):
        if not isinstance(self.treatment, Measured):
            raise TypeError("treatment must be a Measured (a Sketch is deliberately un-ledgerable; "
                            "call .measured() to stamp it)")

        # --- the registry knows this instrument, or the Claim does not exist ----------------
        spec = registry.spec(self.instrument)         # raises UnknownInstrument
        null = spec.null_value(self.config)
        if np.isnan(null):
            raise registry.InstrumentNotImplemented(
                f"{self.instrument} has no declared null ({spec.null_doc}); the core must not ship "
                "an instrument that cannot state its own (spec §8)")
        if not spec.implemented:
            self.notes.append(f"{self.instrument} is DECLARED but not built (spec §11.4): "
                              + "; ".join(spec.notes))

        # --- required arms, from the registry, not from the caller -------------------------
        required = list(spec.required_arms)
        if (self.patch_layer is not None and self.readout_layer is not None
                and self.readout_layer >= self.patch_layer and "passthrough" not in required):
            required.append("passthrough")
        missing = [a for a in required if a not in self.arms]
        if missing:
            why = ""
            if "passthrough" in missing:
                why = (f" -- readout layer {self.readout_layer} is at or after patch layer "
                       f"{self.patch_layer}, so the readout moves by residual arithmetic whether or "
                       "not the blocks between them compute anything (h14/h40)")
            raise MissingArm(f"{self.instrument} requires arms {required}; missing {missing}{why}")

        # --- the pass-through arm must have been COMPUTED, not declared ---------------------
        pt = self.arms.get("passthrough")
        if pt is not None and not getattr(pt, "computed", False):
            raise PassthroughNotComputed(
                "the `passthrough` arm is a hand-declared number. §2a's arm is OFFLINE ARITHMETIC "
                "-- readout(base_resid_at_read_layer + shift), scored with the same readout code as "
                "the treatment -- and it must reproduce the unpatched readout exactly at zero "
                "shift. Build it with instruments.PassthroughArm.compute(...); a declared one is "
                "the h14 failure with a label on it.")

        # --- arm tolerances: MEASURED, per instrument and per arm size ----------------------
        for name, a in self.arms.items():
            if a.tolerance is None:
                a.tolerance = spec.arm_tolerance(a.n, self.config)
                if a.n == 1:
                    self.notes.append(
                        f"arm {name!r} was handed in as a single pooled number, so its tolerance is "
                        f"the full one-item null spread ({a.tolerance:.3f}); per-item scores would "
                        "make this check n times tighter")

        # --- every arm at its declared null ------------------------------------------------
        off = {k: (a.value, a.expected_null) for k, a in self.arms.items() if a.off_null}
        if off:
            raise ArmOffNull(
                "arm(s) off their declared null: "
                + "; ".join(f"{k} reads {v:.3f} where it declared {e:.3f} "
                            f"(tolerance {self.arms[k].resolved_tolerance:.3f}, measured from "
                            f"{self.instrument}'s own null at n={self.arms[k].n})"
                            for k, (v, e) in off.items())
                + ". An arm off its null is a bug until proven otherwise (h34: no_patch 2.00 looked "
                  "fine, random 1.22 was the tell).")

        # --- floor, both halves ------------------------------------------------------------
        if self.floor is None:
            raise MissingFloor("a Claim needs a Floor(stimulus=..., estimator=...); estimate the "
                               "noise floor before believing a null (h28/h32)")

        # --- leaky grid must report gain over the measured floor ---------------------------
        if self.grid is not None and self.grid.leak.leaky and self.report_as != "gain_over_floor":
            raise RawScoreOnLeakyGrid(
                f"grid {self.grid.name or self.grid.hash} leaks: {self.grid.leak.summary()}. "
                "A leaky grid is allowed; a raw score on one is not. Set "
                "report_as='gain_over_floor' and report the gain (h35/h38, spec §6).")

        # --- selection ---------------------------------------------------------------------
        if self.selection.is_a_choice and not self.selection.held_out:
            raise SelectionOnScoringData(
                f"the reported value was chosen along {self.selection.axis!r} by "
                f"{self.selection.rule!r} on the scoring data. Report the curve (h3: a constant "
                "predictor went 3.5 -> 2.4 by selection alone).")

        # --- calibration -------------------------------------------------------------------
        if self.calibration is None:
            raise MissingCalibration(f"no calibration report for {self.instrument}; a missing report "
                                     "blocks Claim construction (spec §5)")
        if (self.calibration.instrument != self.instrument
                and not self.calibration.hand_declared_report):
            raise CalibrationStale(
                f"the calibration report is for {self.calibration.instrument!r}, not "
                f"{self.instrument!r}")
        # Piece 2 left this open: `registry.CALIBRATION_KEYS` is populated on import of
        # `lsx.core.instruments`, so `from lsx.core.types import Claim` alone left a STALE measured
        # report acceptable. `types` cannot import `instruments` at module level (cycle), but it
        # can here, once, when the table is empty -- which is the only case the hole existed in.
        if not registry.CALIBRATION_KEYS:
            try:
                from . import instruments as _instruments   # noqa: F401  (registers the keys)
            except Exception:  # noqa: BLE001 -- a partially-imported package must not break Claim
                pass
        expected_key = registry.CALIBRATION_KEYS.get(self.instrument)
        if (expected_key is not None and not self.calibration.hand_declared_report
                and self.calibration.key != expected_key):
            raise CalibrationStale(
                f"{self.instrument}'s calibration report was produced under key "
                f"{self.calibration.key} but the instrument now hashes to {expected_key}: its "
                "source, its declared null or its declared invariances changed since. Re-calibrate "
                "(spec §5: a stale report blocks Claim construction).")
        if not self.calibration.passed and self.companion is None:
            raise CalibrationFailed(
                f"{self.instrument} failed its own calibration ({'; '.join(self.calibration.failures)}). "
                "A failing statistic may still be reported, but only alongside a passing companion, "
                "and the Claim records the failure (h6/h38, spec §5).")

        # --- the effect size has something behind it ---------------------------------------
        if not self.effect.computed and spec.implemented:
            recomputed = EffectSize.against(self.treatment.values, null,
                                            fallback_item_sd=spec.null_item_sd(self.config))
            if recomputed.verdict != self.effect.verdict:
                raise EffectSizeUnverified(
                    f"the declared effect (size {self.effect.size:+.3f}, z {self.effect.z:+.2f}) "
                    f"and the one recomputed from the treatment scores against {self.instrument}'s "
                    f"declared null {null:.3f} (size {recomputed.size:+.3f}, z {recomputed.z:+.2f}) "
                    "disagree about the verdict. EffectSize was the one number in the contract with "
                    "no assertion behind it; the assertion is on the VERDICT, not the value, "
                    "because a caller's standard error may legitimately come from paraphrases "
                    "rather than items.")
            self.notes.append(f"effect size was caller-supplied; recomputed z {recomputed.z:+.2f} "
                              f"agrees on the verdict")

        # --- semantic null declared before the treatment was computed ----------------------
        if self.semantic_null is not None:
            if not self.semantic_null.justification:
                raise NullDeclaredLate("the semantic null needs a one-line justification, computed "
                                       "from data as a real arm and never asserted in prose")
            if self.semantic_null.seq > self.treatment.seq:
                raise NullDeclaredLate(
                    "the semantic null was constructed after the treatment score; it must be "
                    "declared before any treatment score is computed (spec §4)")

        self.id = _hash({"instrument": self.instrument, "provenance": self.provenance,
                         "grid": self.grid.hash if self.grid else None,
                         "selection": [self.selection.axis, self.selection.rule],
                         "config": self.config,
                         "calibration": self.calibration.key})

    # ---------------------------------------------------------------------------------------
    @property
    def reported_value(self) -> float:
        if self.report_as == "gain_over_floor":
            return self.treatment.value - float(self.floor.stimulus)
        return self.treatment.value

    def render(self) -> str:
        """The only path to a printable number, and it always prints everything together."""
        head = (f"{self.instrument} [{self.id}] {self.treatment.label or ''}\n"
                f"  {'gain over stimulus floor' if self.report_as == 'gain_over_floor' else 'treatment'}"
                f" = {self.reported_value:.4f}  (raw {self.treatment.value:.4f}, n={self.treatment.n})")
        lines = []
        for k, a in sorted(self.arms.items()):
            line = (f"  arm {k:<18} {a.value:.4f}   declared null {a.expected_null:.4f}"
                    f"  +-{a.resolved_tolerance:.4f}")
            if a.off_null:
                line += "  OFF NULL"
            if getattr(a, "computed", False):
                line += ("  [COMPUTED offline arithmetic, zero-shift error "
                         f"{getattr(a, 'zero_shift_error', 0.0):.1g}"
                         f"{', norm-matched' if getattr(a, 'norm_matched', False) else ''}]")
            lines.append(line)
        arms = "\n".join(lines)
        if self.semantic_null is not None:
            arms += (f"\n  semantic null      {self.semantic_null.value:.4f}   "
                     f"({self.semantic_null.justification})")
        tail = (f"  floor              stimulus {self.floor.stimulus:.4f} / estimator "
                f"{self.floor.estimator:.4f}\n"
                f"  effect             {self.effect.size:.4f}  n={self.effect.n}  z={self.effect.z:.2f}\n"
                f"  selection          axis={self.selection.axis} rule={self.selection.rule!r} "
                f"held_out={self.selection.held_out} executed={self.selection.executed}\n"
                f"  calibration        {self.calibration.summary()}")
        if self.grid is not None:
            tail += f"\n  grid leak          {self.grid.leak.summary()}"
        if not self.calibration.passed:
            tail += "\n  CALIBRATION FAILED -- reported only beside its companion"
        return "\n".join([head, arms, tail])

    def to_row(self) -> dict:
        """The ledger row (spec §9). Piece 3 owns the ledger file; this is its shape."""
        return {"id": self.id, "stage": self.stage, "instrument": self.instrument,
                "treatment": self.treatment.value, "reported": self.reported_value,
                "report_as": self.report_as,
                "arms": {k: {"value": a.value, "expected_null": a.expected_null,
                             "tolerance": a.resolved_tolerance, "off_null": a.off_null,
                             "computed": bool(getattr(a, "computed", False))}
                         for k, a in self.arms.items()},
                "semantic_null": (None if self.semantic_null is None else
                                  {"value": self.semantic_null.value,
                                   "justification": self.semantic_null.justification}),
                "floor": {"stimulus": self.floor.stimulus, "estimator": self.floor.estimator},
                "effect": {"size": self.effect.size, "n": self.effect.n, "z": self.effect.z,
                           "computed": self.effect.computed},
                "selection": {"axis": self.selection.axis, "rule": self.selection.rule,
                              "held_out": self.selection.held_out,
                              "executed": self.selection.executed, "curve": self.selection.curve},
                "provenance": dict(self.provenance,
                                   grid_hash=self.grid.hash if self.grid else
                                   self.provenance.get("grid_hash"),
                                   calibration_key=self.calibration.key),
                "calibration_passed": self.calibration.passed,
                "calibration_hand_declared": self.calibration.hand_declared_report,
                "config": dict(self.config), "notes": list(self.notes),
                "status": "standing"}
