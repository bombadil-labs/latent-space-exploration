"""The three instruments piece 2 ships: `selector`, `composition`, `readout_shift` (spec §8, §11).

Each one is the same object: a statistic, the planted-signal generator it has to find (written
first, in `planted.py`), a declared null and tolerance from the registry, a declared set of
invariances, and a `calibrate()` that must pass before the instrument can build a `Claim`.

`crosstalk`, `depth_gain` and `generality` are declared in `registry.py` and deliberately not
built here (spec §11.4). `depth_gain`'s known-zero calibration is the one that *failed* at h38 and
must not gate the core; `generality` has no null at all.

The calibration report is cached under `results/calibration/<instrument>.<key>.json`, where the key
is the hash of the statistic's own source, its declared null and its declared invariances -- so
editing one instrument re-calibrates that one alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from . import planted, registry
from .checks import (CalibrationReport, CalibrationStale, PassthroughNotReproduced,
                     cached_calibration, calibration_key, dot_tie_atol, midrank,
                     spearman, top1_hit)
from .planted import OrderedFixture, RankFixture, ShiftFixture, unit
from .types import Arm, Claim, EffectSize, Measured, Selection


# --------------------------------------------------------------------------------------------
# the statistics
# --------------------------------------------------------------------------------------------
def cosine_scores(acts: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """[n, k] cosine of every activation against every candidate direction. Cosine, not dot: the
    dot product would make the statistic a function of the residual's norm, and both rank
    instruments claim scale invariance."""
    return unit(acts) @ unit(dirs).T


def selector_rank(f: RankFixture) -> float:
    """Mean rank of the correct candidate among k, MID-RANK on ties.

    Mid-rank is not a detail. Strict '>' counting awards rank 1 to a wholly tied field, which is
    how h34 read a dead patch as a sharp lens; mid-rank reads a tie as chance, which is what
    nothing-happened should look like. The known-zero test below is exactly this case.
    """
    s = cosine_scores(f.acts, f.dirs)
    return float(np.mean([midrank(s[i], int(f.target[i])) for i in range(f.n)]))


def composition_rank(f: RankFixture) -> float:
    """Mean rank of the JOINTLY correct variant among all V joint variants.

    The variants are composed by ADDING the factor-level directions (`compose_variants`), which is
    the operation h8's composed patch performs; the statistic is then the same mid-rank. What makes
    this a different instrument from `selector` is not the ranking, it is the null -- (V+1)/2 over
    the joint set, 9.50 for h8's 18 variants, where a per-factor null would be 2.0 -- and that it
    requires no `permutation` arm but does require `no_patch` to sit at that joint midpoint.
    """
    s = cosine_scores(f.acts, f.dirs)
    return float(np.mean([midrank(s[i], int(f.target[i])) for i in range(f.n)]))


def readout_shift_gain(f: ShiftFixture) -> float:
    """Gain of the patched readout over the pass-through, as a DIFFERENCE, per item, unit-free.

    `treatment - passthrough`, never `treatment / passthrough`: a ratio is unstable exactly where
    the pass-through already sits near the ceiling, which is the h14 regime this instrument exists
    to refuse (spec §2a).

    The readout is the PROJECTION on the unit direction, divided by the item's own base residual
    norm -- not a cosine, and the difference is why this instrument exists in this form. Calibration
    found the cosine version reading **-0.129 on its own null**: adding block work orthogonal to
    the readout direction dilutes a cosine, because a cosine is the FRACTION of the vector lying
    along the direction, so a model that is merely busy registers as negative gain. A projection
    difference cancels everything the pass-through already carries and reads exactly 0.0 when the
    blocks contribute nothing along the direction, however much they contribute elsewhere.
    `cosine_readout_gain` below keeps the rejected version, and a test asserts the battery refuses
    it. Anything RELATIVE -- a rank among candidate directions, a margin between two of them, which
    is what h14's "nearest era direction" readout actually was -- is unbiased in the same way; a
    bare scalar cosine is the shape to avoid, and that is a finding for piece 3's graders.
    """
    d = unit(f.direction)
    per_item = ((f.treatment - f.passthrough) @ d) / np.linalg.norm(f.base, axis=-1)
    return float(np.mean(per_item))


def cosine_readout_gain(f: ShiftFixture) -> float:
    """The rejected variant, kept because a rejected instrument is evidence. Cosine in, cosine out;
    fails the noise test at -0.129 under orthogonal block work. Not registered, not buildable."""
    d = unit(f.direction)
    return float(np.mean(unit(f.treatment) @ d - unit(f.passthrough) @ d))


def top1_accuracy_rate(f: RankFixture) -> float:
    """Fraction of items whose argmax over k candidates lands on the target. TIES SPLIT (1/T).

    This is h29's statistic. Its null is 1/k, NOT a rank midpoint, and the difference is not
    cosmetic: a rank's null grows with k and a rate's shrinks, so an accuracy read against a rank's
    null would look like a large effect at every k > 2. `selector` and this instrument read the
    same scores and answer different questions -- "how far down the list is the right answer" and
    "is the right answer first" -- and only the first survives a near-miss.

    Ties split the hit for the reason `midrank` mid-ranks: see `checks.top1_hit`.
    """
    s = cosine_scores(f.acts, f.dirs)
    return float(np.mean([top1_hit(s[i], int(f.target[i])) for i in range(f.n)]))


def top1_per_item(f: RankFixture) -> np.ndarray:
    s = cosine_scores(f.acts, f.dirs)
    return np.array([top1_hit(s[i], int(f.target[i])) for i in range(f.n)])


def naive_top1_accuracy(f: RankFixture) -> float:
    """The rejected variant, kept because a rejected instrument is evidence: `argmax == target`,
    winner-takes-first on ties. It passes noise, sensitivity, self-floor and both invariances, and
    it is separated from the shipped statistic by exactly one fixture -- `planted.rank_partial_tie`,
    where it reads whatever the tied candidates' labelling happens to give and the shipped statistic
    reads 1/T. Not registered, not buildable."""
    s = cosine_scores(f.acts, f.dirs)
    return float(np.mean([float(int(np.argmax(s[i])) == int(f.target[i])) for i in range(f.n)]))


def discrimination_rho(f: OrderedFixture) -> float:
    """Mean over subjects of the Spearman rank correlation between a scalar read off the residual
    and the ordered target. h39's `shared_mean`, and h38's 0.961 -> 0.522.

    RANK correlation, not Pearson, for a reason the battery tests: the target is `log Δt`, and the
    number must not depend on the choice of log base or on whether the axis is Δt, log Δt or grid
    index. A Pearson correlation moves under all three.

    The statistic is the RAW discrimination. The gain over the measured floor is computed at the
    `Claim`, through `report_as='gain_over_floor'`, because §6 makes that a reporting rule and not
    a property of the statistic -- and because a floor is a measurement of a GRID, which a
    synthetic calibration fixture cannot stand in for (registry.calibration_null).
    """
    return float(np.mean(discrimination_per_item(f)))


def discrimination_per_item(f: OrderedFixture) -> np.ndarray:
    """One Spearman per subject, with a tie tolerance computed from the arithmetic that produced
    the scores rather than chosen (`checks.dot_tie_atol`). Without it a dead readout scores +-0.55
    per subject off 4e-16 of BLAS rounding; see `checks.spearman`."""
    scores = f.acts @ unit(f.direction)                 # [S, m]
    atol = dot_tie_atol(f.acts.shape[-1], float(np.abs(scores).max()))
    return np.array([spearman(scores[s], f.y, atol) for s in range(f.n)])


def discrimination_floor_rho(f: OrderedFixture) -> float:
    """The same statistic on the fixture's FLOOR readout: what the stimulus alone orders. This is
    the `floor` arm, computed rather than declared -- the h28 lesson (estimate the floor before
    believing the null) applied to the arm rather than to the claim."""
    return float(np.mean([spearman(f.floor_scores[s], f.y) for s in range(f.n)]))


# --------------------------------------------------------------------------------------------
# fixtures (built from `planted.py`, which was written before these statistics existed)
# --------------------------------------------------------------------------------------------
def compose_variants(level_dirs: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """All joint variants of a factorial design, each direction the unit sum of its level
    directions. `level_dirs[f]` is [levels_f, d]."""
    import itertools
    combos = list(itertools.product(*[range(len(L)) for L in level_dirs]))
    V = np.stack([unit(sum(level_dirs[f][c[f]] for f in range(len(level_dirs)))) for c in combos])
    return V, combos


def composition_fixture(n: int, d: int, levels: Sequence[int], rng: np.random.Generator
                        ) -> RankFixture:
    level_dirs = [unit(rng.normal(size=(L, d))) for L in levels]
    V, combos = compose_variants(level_dirs)
    target = rng.integers(0, len(combos), size=n)
    return RankFixture(acts=rng.normal(size=(n, d)), dirs=V, target=target)


# --------------------------------------------------------------------------------------------
# the computed pass-through arm (spec §2a)
# --------------------------------------------------------------------------------------------
@dataclass
class PassthroughArm(Arm):
    """`readout(base_resid_at_read_layer + shift)`, computed offline with no forward pass.

    Piece 1 could only check that an arm named `passthrough` was PRESENT. This is the arm actually
    computed, and it carries two things a declared arm cannot:

    * `zero_shift_error` -- the arm reproduces the unpatched readout EXACTLY at zero shift, asserted
      in the constructor. If it does not, the arm is wrong, not the claim (spec §2a). Exact, not
      `allclose`: `base + 0*shift` scaled by `||base||/||base||` is bitwise `base`, so any
      discrepancy is a real difference in the readout path, which is the thing being checked --
      that the pass-through is scored with the SAME readout code as the treatment.
    * `norm_matched` -- whether the shifted residual was rescaled to the treatment's norms, and to
      what. A pass-through that is bigger than the treatment is not a control.
    """
    computed: bool = True
    zero_shift_error: float = 0.0
    norm_matched: bool = False

    @staticmethod
    def _shifted(base: np.ndarray, shift: np.ndarray, target_norms: np.ndarray | None
                 ) -> np.ndarray:
        v = base + shift
        if target_norms is None:
            return v
        return v * (np.asarray(target_norms, dtype=v.dtype)[:, None]
                    / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12))

    @classmethod
    def compute(cls, *, base: np.ndarray, shift: np.ndarray,
                readout: Callable[[np.ndarray], np.ndarray],
                expected_null: float, justification: str,
                match_norms: np.ndarray | None = None,
                unpatched: np.ndarray | None = None,
                tolerance: float | None = None) -> "PassthroughArm":
        base = np.asarray(base, dtype=np.float64)
        shift = np.asarray(shift, dtype=np.float64)
        own_norms = np.linalg.norm(base, axis=-1)

        # the arm's own assertion: at zero shift, through the SAME code path (norm-matching to the
        # base's own norms, whose ratio is exactly 1.0), the arm must reproduce the unpatched
        # readout exactly.
        ref = np.asarray(readout(base) if unpatched is None else unpatched, dtype=np.float64)
        zero = np.asarray(readout(cls._shifted(base, np.zeros_like(shift), own_norms)),
                          dtype=np.float64)
        err = float(np.max(np.abs(zero - ref))) if zero.shape == ref.shape else float("inf")
        if not (zero.shape == ref.shape and np.array_equal(zero, ref)):
            raise PassthroughNotReproduced(
                f"the pass-through arm does not reproduce the unpatched readout at zero shift "
                f"(max |error| {err:.3g}, shapes {zero.shape} vs {ref.shape}). The ARM is wrong, "
                "not the claim: it is not scoring with the same readout code as the treatment, or "
                "it is not reading the same base residual (spec §2a).")

        scores = np.asarray(readout(cls._shifted(base, shift, match_norms)), dtype=np.float64)
        return cls(scores=scores, expected_null=expected_null, tolerance=tolerance,
                   justification=justification, zero_shift_error=err,
                   norm_matched=match_norms is not None)


# --------------------------------------------------------------------------------------------
# the instrument
# --------------------------------------------------------------------------------------------
@dataclass
class Instrument:
    """A statistic that cannot produce a `Claim` without a passing, non-stale calibration report."""
    name: str
    stat: Callable
    noise_fn: Callable
    plant: Callable
    transforms: dict
    self_floor: Callable
    known_zero: Callable
    config: dict = field(default_factory=dict)
    per_item: Callable | None = None   # fixture -> per-ITEM scores; see `_per_item`
    n: int = 400                       # items in the calibration fixture
    amplitudes: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    null_tol: float | None = None      # None -> MEASURED, from this instrument's own null spread
    invariance_tol: float = 1e-6
    known_zero_tol: float = 1e-9
    _report: CalibrationReport | None = field(default=None, repr=False)

    # ---- declarations, all from the registry -------------------------------------------
    @property
    def spec(self) -> registry.InstrumentSpec:
        return registry.spec(self.name)

    @property
    def declared_null(self) -> float:
        """Where this instrument's ARMS should sit, given the configuration."""
        return self.spec.null_value(self.config)

    @property
    def calibration_null(self) -> float:
        """Where the BATTERY expects the statistic to sit. The same number for every piece-2
        instrument; for `discrimination` it is chance rather than the caller's measured floor, and
        `registry.InstrumentSpec.calibration_null` says why at length."""
        return self.spec.calibration_null_value(self.config)

    @property
    def invariances(self) -> tuple[str, ...]:
        return self.spec.invariances

    def tolerance(self, n: int) -> float:
        return self.spec.arm_tolerance(n, self.config)

    @property
    def resolved_null_tol(self) -> float:
        """How far off its null the statistic may read during calibration.

        MEASURED, not chosen: it is the same 3-sigma band as an arm's, at the calibration
        fixture's own n. A flat 0.1 is the mistake this whole piece exists to stop -- with 400
        6-candidate items the null itself has a standard error of 0.085, so a flat 0.1 fails a
        perfectly good statistic about one run in four. It did: the first run of this battery
        failed `selector`'s and `composition`'s self-floor test at 1.3 sigma.
        """
        return self.tolerance(self.n) if self.null_tol is None else float(self.null_tol)

    @property
    def key(self) -> str:
        """Hash of the statistic's source, the null THE BATTERY RAN AT, and the declared
        invariances. Not the arms' null: on `discrimination` that is a floor the caller measured on
        their own grid, and hashing it would demand a fresh battery for an unchanged statistic
        every time the grid changed -- piece 3's config-dependent-key bug, one level along."""
        return calibration_key(self.stat, self.calibration_null, self.invariances)

    # ---- calibration --------------------------------------------------------------------
    def calibrate(self, *, refresh: bool = False, cache_dir=None, seed: int = 0
                  ) -> CalibrationReport:
        """Run (or load) the §5 battery. Cached under this instrument's own key."""
        def run():
            from .checks import run_calibration
            return run_calibration(
                self.stat, declared_null=self.calibration_null, plant=self.plant,
                name=self.name, amplitudes=self.amplitudes, null_tol=self.resolved_null_tol,
                seed=seed,
                invariances=self.invariances, not_invariances=self.spec.not_invariances,
                noise_fn=self.noise_fn, transforms=self.transforms, self_floor=self.self_floor,
                known_zero=self.known_zero, invariance_tol=self.invariance_tol,
                known_zero_tol=self.known_zero_tol)

        self._report = cached_calibration(self.name, self.key, run, cache_dir=cache_dir,
                                          refresh=refresh)
        # This configuration's key is now a CURRENT key for this instrument. The declared null is a
        # function of the config (a 3-candidate selector's null is 2.00, a 6-candidate's is 3.50),
        # so one instrument legitimately has several current keys; piece 3 found a single-key table
        # refusing good claims as stale (see registry.CALIBRATION_KEYS).
        registry.register_key(self.name, self.key)
        return self._report

    @property
    def calibration(self) -> CalibrationReport:
        if self._report is None or self._report.key != self.key:
            self.calibrate()
        return self._report

    def measure_null_item_sd(self, n: int = 4000, seed: int = 11) -> float:
        """The per-ITEM spread of the statistic under its own null, measured rather than assumed.

        This is what the registry's `null_item_sd` closed form is checked against, and it is where
        the arm tolerance comes from: tolerance(n) = 3 * sd / sqrt(n).
        """
        rng = np.random.default_rng(seed)
        f = self.noise_fn(rng)
        per_item = self._per_item(f)
        return float(np.std(per_item))

    def _per_item(self, f) -> np.ndarray:
        """Per-item scores of THIS instrument's statistic on a fixture.

        Dispatched on the instrument, via `per_item`, and no longer on the fixture's TYPE. That
        change is piece 4's, and it is a bug fixed before it could be published: `top1_accuracy`
        reuses `RankFixture`, so the old type-sniffing version would have measured the per-item
        spread of a mid-RANK (0.816 at k=3) and handed it to an accuracy whose real per-item spread
        is Bernoulli (0.4714) -- a tolerance 1.7x too loose on every arm of every h29-shaped claim,
        arrived at by the fixture and the statistic disagreeing silently about what an item is.
        """
        if self.per_item is not None:
            return np.asarray(self.per_item(f), dtype=np.float64)
        if isinstance(f, RankFixture):
            s = cosine_scores(f.acts, f.dirs)
            return np.array([midrank(s[i], int(f.target[i])) for i in range(f.n)])
        d = unit(f.direction)
        return ((f.treatment - f.passthrough) @ d) / np.linalg.norm(f.base, axis=-1)

    # ---- the only way to a Claim ----------------------------------------------------------
    def claim(self, *, treatment, arms: dict, floor, selection: Selection, provenance: dict,
              grid=None, semantic_null: Arm | None = None, report_as: str = "raw",
              patch_layer: int | None = None, readout_layer: int | None = None, stage: str = "",
              effect: EffectSize | None = None, companion: Claim | None = None,
              notes: Sequence[str] = ()) -> Claim:
        """Build a `Claim` through the instrument, which is what makes the plumbing arms the
        REGISTRY's rather than the caller's: an arm handed in as a bare array of scores gets its
        declared null and its measured tolerance from the instrument, and a caller cannot quietly
        widen either.
        """
        report = self.calibration
        if report.key != self.key:
            raise CalibrationStale(f"{self.name}: calibration {report.key} != instrument {self.key}")

        built = {}
        for arm_name, a in arms.items():
            if isinstance(a, Arm):
                if a.expected_null is None:
                    raise ValueError("unreachable: Arm refuses a missing null at construction")
                built[arm_name] = a if a.tolerance is not None else replace(
                    a, tolerance=self.tolerance(a.n))
            else:
                scores = np.atleast_1d(np.asarray(a, dtype=np.float64))
                built[arm_name] = Arm(scores, expected_null=self.declared_null,
                                      tolerance=self.tolerance(len(scores)),
                                      justification=f"{arm_name}: null and tolerance from the "
                                                    f"registry ({self.spec.null_doc})")

        t = treatment if isinstance(treatment, Measured) else Measured(treatment, label=self.name)
        if effect is None:
            effect = EffectSize.against(t.values, self.declared_null,
                                        fallback_item_sd=self.spec.null_item_sd(self.config))
        return Claim(instrument=self.name, treatment=t, arms=built, floor=floor,
                     selection=selection, effect=effect, calibration=report,
                     provenance=dict(provenance, calibration_key=report.key),
                     semantic_null=semantic_null, grid=grid, report_as=report_as,
                     patch_layer=patch_layer, readout_layer=readout_layer, companion=companion,
                     stage=stage, config=dict(self.config), notes=list(notes))

    # ---- selection executed by the core, not described by the caller ----------------------
    def sweep(self, axis: str, values: Sequence, score: Callable[[object], float]) -> Selection:
        """Run the sweep HERE so the rule is recorded rather than described.

        Piece 1's selection catch is a regex over the caller's own prose: `rule="we looked at the
        curve and quoted layer 16"` passes it. A `Selection` that carries the curve the core itself
        computed cannot be a story about what the caller did.
        """
        curve = {str(v): float(score(v)) for v in values}
        return Selection.from_sweep(axis, curve)


# --------------------------------------------------------------------------------------------
# the three, wired to their generators
# --------------------------------------------------------------------------------------------
def _rank_per_item(f: RankFixture) -> np.ndarray:
    s = cosine_scores(f.acts, f.dirs)
    return np.array([midrank(s[i], int(f.target[i])) for i in range(f.n)])


def _shift_per_item(f: ShiftFixture) -> np.ndarray:
    d = unit(f.direction)
    return ((f.treatment - f.passthrough) @ d) / np.linalg.norm(f.base, axis=-1)


def selector(n: int = 400, d: int = 64, n_candidates: int = 6, **kw) -> Instrument:
    cfg = {"n_candidates": n_candidates}
    # Small planted sizes on purpose: pushed a full residual-norm along the target direction the
    # rank saturates at 1.0 and the "monotone in the planted size" curve is three tied points.
    kw.setdefault("amplitudes", (0.05, 0.1, 0.2, 0.4))
    return Instrument(
        name="selector", stat=selector_rank, config=cfg, n=n,
        noise_fn=lambda rng: planted.rank_noise((n, d, n_candidates), rng),
        plant=planted.rank_plant, per_item=_rank_per_item,
        transforms={"scale": planted.rank_rescale, "rotation": planted.rank_rotate},
        self_floor=planted.rank_floor, known_zero=planted.rank_known_zero, **kw)


def composition(n: int = 400, d: int = 64, levels: Sequence[int] = (3, 3, 2), **kw) -> Instrument:
    n_variants = int(np.prod(levels))
    cfg = {"n_variants": n_variants, "levels": tuple(levels)}
    kw.setdefault("amplitudes", (0.05, 0.1, 0.2, 0.4))
    return Instrument(
        name="composition", stat=composition_rank, config=cfg, n=n,
        noise_fn=lambda rng: composition_fixture(n, d, levels, rng),
        plant=planted.rank_plant, per_item=_rank_per_item,
        transforms={"scale": planted.rank_rescale, "rotation": planted.rank_rotate},
        self_floor=planted.rank_floor, known_zero=planted.rank_known_zero, **kw)


def _shift_noise_with_working_blocks(n: int, d: int, rng: np.random.Generator) -> ShiftFixture:
    """The null fixture for `readout_shift` is NOT "the blocks do nothing" -- that would make the
    treatment bitwise identical to the pass-through and the noise test a tautology reading exactly
    0.0000. The blocks here do a full residual-norm of work, all of it ORTHOGONAL to the readout
    direction: the model is busy, and the readout still must not register a gain.
    """
    f = planted.shift_noise((n, d), rng)
    return replace(f, amplitude=float(np.linalg.norm(f.base, axis=-1).mean()))


def _rescale_whole_stream(f: ShiftFixture, rng: np.random.Generator) -> ShiftFixture:
    """The "per-layer rescaling" of §5 for a shift fixture: one scalar on the WHOLE residual
    stream -- base, shift and block contribution together -- which is what changing layer or model
    scale does. `planted.shift_rescale` rescales the base ALONE, which is a different and much
    harsher transform; it is what the rejected cosine variant is measured against in the tests.
    """
    c = float(rng.uniform(0.2, 5.0))
    return replace(f, base=f.base * c, shift=f.shift * c, amplitude=f.amplitude * c)


def readout_shift(n: int = 400, d: int = 64, **kw) -> Instrument:
    cfg = {"d": d}
    kw.setdefault("amplitudes", (0.25, 0.5, 1.0, 2.0))
    return Instrument(
        name="readout_shift", stat=readout_shift_gain, config=cfg, n=n,
        noise_fn=lambda rng: _shift_noise_with_working_blocks(n, d, rng),
        plant=planted.shift_plant, per_item=_shift_per_item,
        transforms={"scale": _rescale_whole_stream, "rotation": planted.shift_rotate},
        self_floor=planted.shift_floor, known_zero=planted.shift_known_zero, **kw)


def top1_accuracy(n: int = 400, d: int = 64, n_candidates: int = 3, **kw) -> Instrument:
    """h29's instrument (spec §8, piece 4). Null 1/k, NOT a rank midpoint.

    The fixture is `RankFixture`, the same object `selector` calibrates on, because an accuracy
    over k candidate directions and a rank over k candidate directions read the same scores. The
    per-item function is NOT the same, and it is passed explicitly for that reason: the fixture
    cannot be allowed to decide what an item's score is (see `Instrument._per_item`).
    """
    cfg = {"n_candidates": n_candidates}
    # MEASURED, not guessed. The first set tried was (0.1, 0.2, 0.4, 0.8) and it read
    # [0.630, 0.808, 0.985, 1.000]: the top of the curve is ON the ceiling, so the last step tests
    # nothing and "monotone in the planted size" is satisfied by a statistic that has saturated.
    # The shipped amplitudes span the part of the curve where the statistic is still moving
    # (0.390 / 0.465 / 0.630 / 0.808), which is the only part where monotonicity is evidence.
    kw.setdefault("amplitudes", (0.02, 0.05, 0.1, 0.2))
    return Instrument(
        name="top1_accuracy", stat=top1_accuracy_rate, config=cfg, n=n,
        noise_fn=lambda rng: planted.rank_noise((n, d, n_candidates), rng),
        plant=planted.rank_plant, per_item=top1_per_item,
        transforms={"scale": planted.rank_rescale, "rotation": planted.rank_rotate},
        self_floor=planted.rank_floor, known_zero=planted.rank_known_zero, **kw)


def discrimination(n: int = 200, d: int = 64, m: int = 9, floor: float | None = None,
                   **kw) -> Instrument:
    """h39's instrument (spec §8, piece 4). `n` is the number of SUBJECTS, `m` the ordered levels.

    `floor` is the caller's measured stimulus floor and it sets where the ARMS sit; it does not
    change the battery, which always runs at chance. A `discrimination` claim reports
    `report_as='gain_over_floor'` -- §6's rule for a leaky grid, and h39's grid is flagged.
    """
    cfg = {"m": m}
    if floor is not None:
        cfg["floor"] = float(floor)
    # As for `top1_accuracy`: (0.1, 0.2, 0.4, 0.8) ran to [0.638, 0.861, 0.964, 0.994], which is a
    # saturating curve, and a saturating curve makes the monotonicity test vacuous at the top.
    kw.setdefault("amplitudes", (0.02, 0.05, 0.1, 0.2))
    return Instrument(
        name="discrimination", stat=discrimination_rho, config=cfg, n=n,
        noise_fn=lambda rng: planted.ordered_noise((n, m, d), rng),
        plant=planted.ordered_plant, per_item=discrimination_per_item,
        transforms={"scale": planted.ordered_layer_rescale,
                    "rotation": planted.ordered_rotate,
                    "monotone_target": planted.ordered_monotone_target,
                    "cell_rescale": planted.ordered_rescale},
        self_floor=planted.ordered_floor, known_zero=planted.ordered_known_zero, **kw)


BUILDERS: dict[str, Callable[..., Instrument]] = {
    "selector": selector, "composition": composition, "readout_shift": readout_shift,
    "top1_accuracy": top1_accuracy, "discrimination": discrimination}


def build(name: str, **kw) -> Instrument:
    if name not in BUILDERS:
        raise registry.InstrumentNotImplemented(
            f"{name!r} is declared in the registry but not built (spec §11.4): "
            f"{'; '.join(registry.spec(name).notes) or 'deferred'}")
    return BUILDERS[name](**kw)


def calibrate_all(*, refresh: bool = False, cache_dir=None) -> dict[str, CalibrationReport]:
    """Every shipped instrument's report, for the note and for the test suite."""
    return {name: build(name).calibrate(refresh=refresh, cache_dir=cache_dir) for name in BUILDERS}


def register_calibration_keys() -> dict[str, set[str]]:
    """Publish each built instrument's DEFAULT-configuration key so `Claim` can refuse a STALE
    measured report without importing this module (spec §5: a stale report blocks construction).
    Other configurations register their own keys when they calibrate."""
    for name in BUILDERS:
        registry.register_key(name, build(name).key)
    return {k: set(v) for k, v in registry.CALIBRATION_KEYS.items()}


register_calibration_keys()
