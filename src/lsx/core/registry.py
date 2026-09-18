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
    # The null the CALIBRATION BATTERY is run at, when that is not the same object as the null the
    # arms are held to. Default None -> they are the same, which is true of every piece-2
    # instrument: a 3-candidate selector's null of 2.00 is a property of the statistic and the
    # battery can check it.
    #
    # `discrimination` is the case that forced the distinction, and piece 3's warning is the reason
    # to write it down rather than quietly reuse `null`. Its declared null is the MEASURED FLOOR
    # (spec §8), which is a number the caller measures on their own grid. Feeding that into
    # `calibrate()` would run the battery against a synthetic fixture that knows nothing about the
    # caller's grid, so the noise test would "fail" for every floor except zero -- and, worse, the
    # calibration key hashes the declared null, so every distinct measured floor would demand its
    # own re-calibration of an unchanged statistic. That is piece 3's config-dependent-key bug
    # reappearing inside the code written after it. The battery therefore bounds the STATISTIC at
    # its chance null, and what it does NOT check is the caller's floor measurement. Stated in the
    # report rather than left implicit.
    calibration_null: Callable[[dict], float] | None = None
    calibration_null_doc: str = ""
    invariances: tuple[str, ...] = ()              # claimed, and therefore tested
    not_invariances: tuple[str, ...] = ()          # explicitly NOT claimed, and therefore not tested
    not_invariance_doc: str = ""
    reproduces: str = ""
    implemented: bool = False
    config_keys: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def null_value(self, config: dict | None = None) -> float:
        return float(self.null(config or {}))

    def calibration_null_value(self, config: dict | None = None) -> float:
        """Where the battery expects the statistic to sit. Same as `null_value` unless the
        instrument declares otherwise (see `calibration_null`)."""
        fn = self.calibration_null or self.null
        return float(fn(config or {}))

    def arm_tolerance(self, n: int, config: dict | None = None, z: float = Z_ARM,
                      n_independent: int | None = None) -> float:
        """z * the standard error of an arm under this instrument's own null, over the arm's
        INDEPENDENT UNITS -- which default to its items and are not always its items.

        This is the measurement piece 1 asked for. It is per-instrument *and* per-arm-size: a
        6-candidate selector arm over 40 items has a null standard error of 0.27, so piece 1's flat
        0.15 would have raised `ArmOffNull` on roughly one clean arm in three; the same arm over
        400 items has 0.085, where 0.15 would have let a real 0.14 drift through in silence. Both
        directions of that error are failures of the same kind.

        **Phase 2's correction, and the third failure of the same kind.** Dividing by the ITEM
        count assumes the items are independent draws from the null, and for any arm whose
        randomness is a draw rather than an item they are not: h8's permutation arm has 72 items
        that are four scenes re-ranked eighteen ways, and h16's pooled sweep is one curve read at
        fifteen layers. `n` counts repetitions, not evidence, and the band comes out too TIGHT --
        so the failure is a refusal of a clean arm, which is the dangerous direction: two targets
        were bitten (core_p5 §4, core_p4 on h16) and both were caught by hand. The arm now declares
        its unit (`types.Arm.n_independent`) and this bands on that.

        What stops the declaration from becoming a way to widen a band until a row publishes is
        NOT here -- a number passed in is a number -- it is at `Arm`, which requires the unit to be
        named and the per-item cluster labels handed in, and refuses a reduction whose clustering
        is not visible in the arm's own scores (`checks.ArmUnitNotInDesign`).
        """
        sd = float(self.null_item_sd(config or {}))
        units = n if n_independent is None else int(n_independent)
        return z * sd / math.sqrt(max(int(units), 1))


def _mid(config: dict, key: str, default: int) -> float:
    return (int(config.get(key, default)) + 1) / 2


def _rank_sd(config: dict, key: str, default: int) -> float:
    """sd of one item's rank when every candidate is equally likely to win: the uniform
    distribution on {1..k}, sd = sqrt((k^2 - 1)/12)."""
    k = int(config.get(key, default))
    return math.sqrt((k * k - 1) / 12) if k > 1 else 0.0


def _bernoulli_sd(config: dict, key: str, default: int) -> float:
    """sd of one item's hit/miss when an argmax over k exchangeable candidates is at chance:
    Bernoulli(1/k), sd = sqrt(p(1-p)). Not a rank's sd -- the statistic is bounded in [0, 1]."""
    k = int(config.get(key, default))
    if k <= 1:
        return 0.0
    p = 1.0 / k
    return math.sqrt(p * (1 - p))


def _spearman_sd(config: dict, key: str, default: int) -> float:
    """sd of a Spearman rho over m points under the null of no association: 1/sqrt(m-1)."""
    m = int(config.get(key, default))
    return 1.0 / math.sqrt(m - 1) if m > 1 else 0.0


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
    null=lambda c: float(c.get("floor", 0.0)),
    null_doc="the MEASURED FLOOR value, not chance (h38). A plumbing arm on this instrument sits "
             "where the stimulus alone puts it, and the reported quantity is the gain over that "
             "floor (spec §6), never the raw score. `config['floor']` is that measurement and it "
             "is the caller's: the battery does not and cannot verify it (see `calibration_null`). "
             "With no floor declared the null is 0, which is chance for a rank correlation.",
    null_item_sd=lambda c: _spearman_sd(c, "m", 9),
    null_sd_doc="sd of one subject's Spearman rho over m ordered levels under H0: 1/sqrt(m-1). "
                "m=9 (h39's nine intervals) -> 0.3536. Checked against a Monte-Carlo draw from the "
                "statistic in tests/test_core_registry.py.",
    calibration_null=lambda c: 0.0,
    calibration_null_doc="the battery runs at CHANCE (rho = 0), never at the caller's measured "
                         "floor: the synthetic fixture knows nothing about the caller's grid, and "
                         "hashing a measured floor into the calibration key would demand a fresh "
                         "battery for every floor of an unchanged statistic -- piece 3's "
                         "config-dependent-key bug, reappearing.",
    invariances=("scale", "rotation", "monotone_target"),
    not_invariances=("cell_rescale",),
    not_invariance_doc="rescaling INDIVIDUAL cells by different positive scalars changes the "
                       "projection and therefore may change the order; the instrument claims "
                       "invariance to one scalar on the whole stack (a layer or model scale "
                       "change), not to per-cell rescaling, and the battery measures the "
                       "difference rather than leaving it untested.",
    reproduces="h39 Gemma clock, as gain over the measured stimulus floor",
    implemented=True,
    config_keys=("floor", "m"),
    notes=("`monotone_target` is the invariance this statistic actually leans on and nobody had "
           "tested: h39's target is log Δt, and the choice of log base -- or of Δt, or of grid "
           "index -- must not move the number.",),
))

register(InstrumentSpec(
    name="top1_accuracy",
    required_arms=("random", "no_patch"),
    null=lambda c: 1.0 / max(int(c.get("n_candidates", 3)), 1),
    null_doc="1/k, the chance rate of an argmax over k exchangeable candidates -- NOT a rank "
             "midpoint. This instrument's statistic is bounded in [0, 1] and its null moves the "
             "other way from a rank's: more candidates make chance SMALLER, where a rank's "
             "midpoint grows. h29's era readout is `argmax_e cos(u, d_e)` over three era "
             "directions, so its null is 0.3333 and not 2.00. Ties split the hit over the tied "
             "set (1/T), so a wholly tied field reads exactly 1/k -- the h34 rank-1-on-ties rule "
             "restated for an argmax, where the same bug would award a full hit to whichever "
             "candidate the sort happened to return first.",
    null_item_sd=lambda c: _bernoulli_sd(c, "n_candidates", 3),
    null_sd_doc="one item is Bernoulli(1/k): sd = sqrt(p(1-p)) = sqrt(k-1)/k. k=3 -> 0.4714, "
                "k=2 -> 0.5, k=6 -> 0.3727. Checked against a Monte-Carlo draw from the statistic.",
    invariances=("scale", "rotation"),
    not_invariances=("candidate_count",),
    not_invariance_doc="the value is not comparable across different k, because the null moves: "
                       "0.53 over three candidates and 0.53 over six are different results, and "
                       "the instrument refuses to pretend otherwise by taking k from config.",
    reproduces="h29 (era->target 0.84, leaves-e1 0.91, theme-kept 0.53 at n=55, Gemma-2-9B-it)",
    implemented=True,
    config_keys=("n_candidates",),
    notes=("shares `cosine_scores` and the rank fixtures with `selector` and `composition`, so "
           "three instruments now rest on one scoring routine and their calibrations are not "
           "independent evidence (piece 2 named this hazard for two)",),
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


def arm_tolerance(instrument: str, n: int, config: dict | None = None,
                  n_independent: int | None = None) -> float:
    return spec(instrument).arm_tolerance(n, config, n_independent=n_independent)


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
