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

from .checks import (ArmOffNull, CalibrationFailed, CalibrationReport, HeldOutNotDeclared,
                     MissingArm, MissingCalibration, MissingFloor, NullDeclaredLate,
                     RawScoreOnLeakyGrid, SelectionOnScoringData)

# Plumbing arms are declared here, NOT by the caller (spec §4). This is the minimum declaration
# piece 1 needs; piece 2's instrument registry supersedes it and adds nulls and invariances.
REQUIRED_ARMS: dict[str, tuple[str, ...]] = {
    "selector": ("random", "no_patch", "permutation"),
    "composition": ("random", "no_patch"),
    "readout_shift": ("random", "no_patch", "passthrough"),
    "crosstalk": ("permutation",),
    "discrimination": ("shuffled_stimulus", "floor"),
    "depth_gain": ("shuffled_stimulus", "floor"),
}

_SEQ = itertools.count()


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

    def __post_init__(self):
        if not self.held_out or "axis" not in self.held_out or "unseen" not in self.held_out:
            raise HeldOutNotDeclared(
                "Direction requires held_out={'axis': ..., 'unseen': ...}: what this direction "
                "never saw. h1-h3 fitted on everything and scored on the same data.")

    def vec(self, layer: int) -> np.ndarray:
        return self.vecs[layer]


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
    tolerance: float = 0.15
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
    def off_null(self) -> bool:
        return abs(self.value - self.expected_null) > self.tolerance


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
    """What was swept and how the reported value was chosen (spec §4)."""
    axis: str | None
    rule: str
    held_out: bool = False

    @property
    def is_a_choice(self) -> bool:
        return bool(self.axis) and bool(_CHOOSING.search(self.rule))


@dataclass
class EffectSize:
    size: float
    n: int
    z: float


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
    notes: list[str] = field(default_factory=list)
    id: str = field(init=False, default="")

    def __post_init__(self):
        if not isinstance(self.treatment, Measured):
            raise TypeError("treatment must be a Measured (a Sketch is deliberately un-ledgerable; "
                            "call .measured() to stamp it)")

        # --- required arms, from the registry, not from the caller -------------------------
        required = list(REQUIRED_ARMS.get(self.instrument, ()))
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

        # --- every arm at its declared null ------------------------------------------------
        off = {k: (a.value, a.expected_null) for k, a in self.arms.items() if a.off_null}
        if off:
            raise ArmOffNull(
                "arm(s) off their declared null: "
                + "; ".join(f"{k} reads {v:.3f} where it declared {e:.3f}" for k, (v, e) in off.items())
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
        if not self.calibration.passed and self.companion is None:
            raise CalibrationFailed(
                f"{self.instrument} failed its own calibration ({'; '.join(self.calibration.failures)}). "
                "A failing statistic may still be reported, but only alongside a passing companion, "
                "and the Claim records the failure (h6/h38, spec §5).")

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
        arms = "\n".join(f"  arm {k:<18} {a.value:.4f}   declared null {a.expected_null:.4f}"
                         f"{'  OFF NULL' if a.off_null else ''}"
                         for k, a in sorted(self.arms.items()))
        if self.semantic_null is not None:
            arms += (f"\n  semantic null      {self.semantic_null.value:.4f}   "
                     f"({self.semantic_null.justification})")
        tail = (f"  floor              stimulus {self.floor.stimulus:.4f} / estimator "
                f"{self.floor.estimator:.4f}\n"
                f"  effect             {self.effect.size:.4f}  n={self.effect.n}  z={self.effect.z:.2f}\n"
                f"  selection          axis={self.selection.axis} rule={self.selection.rule!r} "
                f"held_out={self.selection.held_out}\n"
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
                             "off_null": a.off_null} for k, a in self.arms.items()},
                "semantic_null": (None if self.semantic_null is None else
                                  {"value": self.semantic_null.value,
                                   "justification": self.semantic_null.justification}),
                "floor": {"stimulus": self.floor.stimulus, "estimator": self.floor.estimator},
                "effect": {"size": self.effect.size, "n": self.effect.n, "z": self.effect.z},
                "selection": {"axis": self.selection.axis, "rule": self.selection.rule,
                              "held_out": self.selection.held_out},
                "provenance": dict(self.provenance,
                                   grid_hash=self.grid.hash if self.grid else
                                   self.provenance.get("grid_hash"),
                                   calibration_key=self.calibration.key),
                "calibration_passed": self.calibration.passed,
                "status": "standing"}
