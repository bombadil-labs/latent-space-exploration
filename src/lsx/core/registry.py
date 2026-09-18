"""The instrument registry (spec §8) and the arm contract behind it (spec §4).

Piece 1 shipped `types.REQUIRED_ARMS` as a placeholder table of arm *names* with a comment saying
so. This is the real thing: each instrument declares

  * the arms its `Claim` must carry -- plumbing arms are declared HERE, never by the caller;
  * where each arm should sit (its null), as a function of the configuration rather than a
    constant, because a selector's null is the midpoint of *its* candidate set and nothing else;
  * **how far off that null an arm may sit before it is a bug**, derived from the measured
    per-item spread of the statistic's own null distribution at the arm's own n -- not piece 1's
    single unmeasured 0.15;
  * which invariances it claims, and which it explicitly does NOT claim;
  * what it reproduces from spec §1A.

`null_item_sd` is the one number here that could be wrong quietly, so it is written as a closed
form AND checked against a Monte-Carlo draw from the real statistic in
`tests/test_core_registry.py::test_declared_null_item_sd_matches_the_statistic`. A closed form no
test compares to the instrument is a declaration, not a measurement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .checks import CoreError


class UnknownInstrument(CoreError):
    """A Claim names an instrument that is not in the registry. (spec §8)"""


class InstrumentNotImplemented(CoreError):
    """The instrument is declared in the registry but its statistic is not built. (spec §11.4)"""


# `n=1` arms: a caller who hands in one pooled number instead of per-item scores cannot be
# tolerance-checked at the same tightness, and the Claim says so rather than pretending.
Z_ARM = 3.0   # an arm 3 null-sigmas off its null is a bug until proven otherwise


@dataclass(frozen=True)
class InstrumentSpec:
    name: str
    required_arms: tuple[str, ...]
    null: Callable[[dict], float]                  # config -> where every plumbing arm should sit
    null_doc: str
    null_item_sd: Callable[[dict], float]          # per-ITEM sd of the statistic under its null
    null_sd_doc: str
    invariances: tuple[str, ...] = ()              # claimed, and therefore tested
    not_invariances: tuple[str, ...] = ()          # explicitly NOT claimed, and therefore not tested
    not_invariance_doc: str = ""
    reproduces: str = ""
    implemented: bool = False
    config_keys: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def null_value(self, config: dict | None = None) -> float:
        return float(self.null(config or {}))

    def arm_tolerance(self, n: int, config: dict | None = None, z: float = Z_ARM) -> float:
        """z * the standard error of an n-item arm under this instrument's own null.

        This is the measurement piece 1 asked for. It is per-instrument *and* per-arm-size: a
        6-candidate selector arm over 40 items has a null standard error of 0.27, so piece 1's flat
        0.15 would have raised `ArmOffNull` on roughly one clean arm in three; the same arm over
        400 items has 0.085, where 0.15 would have let a real 0.14 drift through in silence. Both
        directions of that error are failures of the same kind.
        """
        sd = float(self.null_item_sd(config or {}))
        return z * sd / math.sqrt(max(int(n), 1))


def _mid(config: dict, key: str, default: int) -> float:
    return (int(config.get(key, default)) + 1) / 2


def _rank_sd(config: dict, key: str, default: int) -> float:
    """sd of one item's rank when every candidate is equally likely to win: the uniform
    distribution on {1..k}, sd = sqrt((k^2 - 1)/12)."""
    k = int(config.get(key, default))
    return math.sqrt((k * k - 1) / 12) if k > 1 else 0.0


REGISTRY: dict[str, InstrumentSpec] = {}


def register(spec: InstrumentSpec) -> InstrumentSpec:
    REGISTRY[spec.name] = spec
    return spec


# --------------------------------------------------------------------------------------------
# the three instruments piece 2 ships (spec §11, rescoped)
# --------------------------------------------------------------------------------------------
register(InstrumentSpec(
    name="selector",
    required_arms=("random", "no_patch", "permutation"),
    null=lambda c: _mid(c, "n_candidates", 6),
    null_doc="midpoint of the candidate set, (k+1)/2: the mean rank of an arbitrary candidate "
             "when the readout knows nothing. Mid-rank on ties, so a wholly tied field reads as "
             "chance rather than as rank 1 (h34).",
    null_item_sd=lambda c: _rank_sd(c, "n_candidates", 6),
    null_sd_doc="uniform rank on {1..k}: sqrt((k^2-1)/12). k=6 -> 1.708, k=3 -> 0.816.",
    invariances=("scale", "rotation"),
    reproduces="h4 (1.7/6), h8 (era 1.25 / voice 1.24 / tense 1.03), h16 (2.21/6), h37 (1.50/1.06)",
    implemented=True,
    config_keys=("n_candidates",),
))

register(InstrumentSpec(
    name="composition",
    required_arms=("random", "no_patch"),
    null=lambda c: _mid(c, "n_variants", 18),
    null_doc="midpoint over the JOINT variant set, (V+1)/2. h8's composed patch ranks the "
             "jointly-correct variant among 18, so the null is 9.50 -- which is exactly where h8's "
             "no-patch arm sat.",
    null_item_sd=lambda c: _rank_sd(c, "n_variants", 18),
    null_sd_doc="uniform rank on {1..V}: sqrt((V^2-1)/12). V=18 -> 5.188.",
    invariances=("scale", "rotation"),
    reproduces="h8 composed 2.81/18 against no-patch 9.50",
    implemented=True,
    config_keys=("n_variants",),
))

register(InstrumentSpec(
    name="readout_shift",
    required_arms=("random", "no_patch", "passthrough"),
    null=lambda c: 0.0,
    null_doc="ZERO GAIN over the norm-matched pass-through -- not chance. The quantity is a "
             "DIFFERENCE, treatment_score - passthrough_score, never a ratio: a ratio is unstable "
             "exactly where the pass-through already sits near the ceiling, which is the h14 "
             "regime (spec §2a).",
    null_item_sd=lambda c: float(c.get("readout_sd", math.sqrt(2.0 / int(c.get("d", 1536))))),
    null_sd_doc="sqrt(2/d) -- the spread of the difference of two INDEPENDENT unit-scaled readouts "
                "in d dimensions (d=1536 -> 0.0361), used deliberately as a conservative stand-in. "
                "The instrument's own null spread is 0 by construction (orthogonal block work "
                "cancels exactly in the projection), which would make an arm tolerance of 0, and "
                "the spread §2a actually wants is the PARAPHRASE-noise interval on the difference. "
                "That needs real paraphrases and is piece 3's to measure; pass "
                "config={'readout_sd': ...} once it exists.",
    invariances=("scale", "rotation"),
    not_invariance_doc="the COSINE form of this gain is not scale-invariant and is biased negative "
                       "whenever the blocks do work orthogonal to the readout direction (measured: "
                       "-0.129 at a full residual-norm of orthogonal work). That variant is kept in "
                       "`instruments.cosine_readout_gain` as a statistic the battery REFUSES, and "
                       "is why this instrument reads a projection.",
    reproduces="h14 as a NEGATIVE target: no gain over pass-through (logged -0.111/-0.125/-0.028)",
    implemented=True,
    config_keys=("d", "readout_sd"),
))

# --------------------------------------------------------------------------------------------
# declared, not implemented (spec §11.4). They keep their arm requirements so a Claim naming one
# is still refused for the right reason, and `implemented=False` refuses it for the honest one.
# --------------------------------------------------------------------------------------------
register(InstrumentSpec(
    name="crosstalk",
    required_arms=("permutation",),
    null=lambda c: 0.0,
    null_doc="variance-decomposition zero. NOT a rank: the h6 rank averaged to 2.0 by construction.",
    null_item_sd=lambda c: float(c.get("null_sd", 0.15)),
    null_sd_doc="unmeasured; deferred with the instrument.",
    reproduces="h8 gain matrix",
    implemented=False,
    notes=("deferred to piece 4; the h6 failure is reproduced in the harness as a calibration "
           "case, not as a shipped instrument",),
))

register(InstrumentSpec(
    name="discrimination",
    required_arms=("shuffled_stimulus", "floor"),
    null=lambda c: float(c.get("floor", 0.5)),
    null_doc="the measured floor value, not chance (h38).",
    null_item_sd=lambda c: float(c.get("null_sd", 0.15)),
    null_sd_doc="unmeasured; deferred with the instrument.",
    reproduces="h39 Gemma clock, as gain over the measured stimulus floor",
    implemented=False,
    notes=("piece 1's harness uses this instrument for the floor and leak cases; it is declared so "
           "those keep working, and unimplemented so no Claim can be graded through it",),
))

register(InstrumentSpec(
    name="depth_gain",
    required_arms=("shuffled_stimulus", "floor"),
    null=lambda c: 0.0,
    null_doc="zero at the known-zero point (layer 0 against a bag-of-embeddings floor, which for "
             "RoPE models is the same object).",
    null_item_sd=lambda c: float(c.get("null_sd", 0.15)),
    null_sd_doc="unmeasured; deferred with the instrument.",
    reproduces="h38",
    implemented=False,
    notes=("deliberately not gating the core: its known-zero calibration is the one that FAILED at "
           "h38 (spec §11.2)",),
))

register(InstrumentSpec(
    name="generality",
    required_arms=(),
    null=lambda c: float("nan"),
    null_doc="UNDECLARED. The abstraction ladder has never had a null; three were specified at h39 "
             "and none measured. The core must not ship an instrument that cannot state its own.",
    null_item_sd=lambda c: float("nan"),
    null_sd_doc="undeclared",
    reproduces="h15 (0.18 vs 0.06) -- ungraded until the null exists",
    implemented=False,
    notes=("the registry records the absence of a null as a first-class fact: a Claim naming "
           "`generality` is refused for having no null, not for being unwritten",),
))


# --------------------------------------------------------------------------------------------
def spec(instrument: str) -> InstrumentSpec:
    try:
        return REGISTRY[instrument]
    except KeyError:
        raise UnknownInstrument(
            f"{instrument!r} is not in the instrument registry; the registry is the only place "
            f"plumbing arms and nulls are declared (known: {sorted(REGISTRY)})") from None


def required_arms(instrument: str) -> tuple[str, ...]:
    return spec(instrument).required_arms


def arm_tolerance(instrument: str, n: int, config: dict | None = None) -> float:
    return spec(instrument).arm_tolerance(n, config)


def table() -> str:
    """The §8 table, printed from the declarations themselves rather than kept in a doc by hand."""
    rows = ["| instrument | arms | null | null sd (per item) | invariances | built |",
            "|---|---|---|---|---|---|"]
    for s in REGISTRY.values():
        inv = ", ".join(s.invariances) or "-"
        if s.not_invariances:
            inv += " (not: " + ", ".join(s.not_invariances) + ")"
        rows.append(f"| `{s.name}` | {', '.join(s.required_arms) or '-'} | {s.null_value():.4g} | "
                    f"{s.null_item_sd({}):.4g} | {inv} | {'yes' if s.implemented else 'no'} |")
    return "\n".join(rows)


# Populated by `instruments.register_calibration_keys()` at import of `lsx.core.instruments`, and
# added to by `Instrument.calibrate()`: the calibration keys currently VALID for each built
# instrument, so that `Claim` can refuse a stale measured report without importing the instruments
# module (spec §5).
#
# Piece 3 found this holding one key per instrument, which was wrong and refused good claims. The
# key is a hash of the statistic's source, its declared null and its declared invariances -- and the
# declared null is a function of the CONFIG: a 3-candidate selector's null is 2.00 and a
# 6-candidate's is 3.50, so they hash differently and both are correct. With a single key, any
# selector claim whose candidate count was not the registry default (h8's three single-factor
# lenses, h34, h37) was refused as `CalibrationStale` while holding a freshly measured, passing
# report. It is a set per instrument now. A genuinely stale report -- one produced by an older
# version of the statistic -- is still refused, because its key is in none of the sets.
CALIBRATION_KEYS: dict[str, set[str]] = {}


def register_key(instrument: str, key: str) -> None:
    CALIBRATION_KEYS.setdefault(instrument, set()).add(key)


def key_is_current(instrument: str, key: str) -> bool | None:
    """True/False, or None when nothing is registered for this instrument (nothing to check against)."""
    keys = CALIBRATION_KEYS.get(instrument)
    return None if not keys else key in keys
