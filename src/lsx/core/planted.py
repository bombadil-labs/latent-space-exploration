"""Planted-signal generators for the calibration battery (spec §5).

Written and committed BEFORE the statistics they test, on piece 1's advice: *expect the
sensitivity test, not the noise test, to be the one that fails*. A generator written after its
statistic plants whatever that statistic happens to see. These are written from the geometry the
instruments claim to measure -- "the residual moves toward the target direction", "the blocks add
something the arithmetic did not" -- and the statistics have to find it or fail.

Each fixture also carries the transforms the calibration battery needs, in one place, so that a
statistic's claimed invariance is tested against the *same* object its sensitivity is:

  * `plant(f, a)`          -- a planted effect of known size `a`
  * `rescale(f, c)`        -- per-item (i.e. per-layer, per-position) rescaling
  * `rotate(f, Q)`         -- a random orthogonal rotation of the whole space
  * `floor(f)`             -- the instrument's own floor as a treatment (self-floor test)
  * `known_zero(f)`        -- a configuration whose answer is analytically zero
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-ish orthogonal matrix: QR of a Gaussian, with the sign fix that makes it uniform."""
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.sign(np.diag(r))[None, :]


# --------------------------------------------------------------------------------------------
# rank fixtures: `selector` and `composition`
# --------------------------------------------------------------------------------------------
@dataclass
class RankFixture:
    """`n` items, each scored against `k` candidate directions; `target[i]` is the right one.

    This is the shape of every selector in the repo (h4 role lens, h8 factors, h16 relation
    selector, h37 70B matched pair): a pooled span vector read against a set of candidate
    directions, one of which is correct.
    """
    acts: np.ndarray      # [n, d]
    dirs: np.ndarray      # [k, d], unit rows
    target: np.ndarray    # [n], index into dirs

    @property
    def n(self) -> int:
        return len(self.acts)

    @property
    def k(self) -> int:
        return len(self.dirs)


def rank_noise(shape: tuple[int, int, int], rng: np.random.Generator) -> RankFixture:
    """(n, d, k) of pure Gaussian activations: the target direction is assigned at random and the
    activations know nothing about it. Every rank statistic must read its declared null here."""
    n, d, k = shape
    dirs = unit(rng.normal(size=(k, d)))
    return RankFixture(acts=rng.normal(size=(n, d)), dirs=dirs,
                       target=rng.integers(0, k, size=n))


def rank_plant(f: RankFixture, a: float) -> RankFixture:
    """Push each item's activation `a` units along ITS OWN target direction, in the units of the
    activation's own scale. Nothing else changes: same candidate set, same targets, same noise."""
    scale = float(np.linalg.norm(f.acts, axis=-1).mean())
    return replace(f, acts=f.acts + a * scale * f.dirs[f.target])


def rank_rescale(f: RankFixture, rng: np.random.Generator) -> RankFixture:
    """Per-item positive rescaling -- the "per-layer rescaling" of spec §5, applied per row so a
    statistic that secretly compares magnitudes across items is caught too."""
    c = rng.uniform(0.2, 5.0, size=(f.n, 1))
    return replace(f, acts=f.acts * c)


def rank_rotate(f: RankFixture, rng: np.random.Generator) -> RankFixture:
    """The same orthogonal rotation applied to activations and candidate directions alike."""
    Q = random_rotation(f.acts.shape[1], rng)
    return replace(f, acts=f.acts @ Q, dirs=f.dirs @ Q)


def rank_floor(f: RankFixture, rng: np.random.Generator) -> RankFixture:
    """The instrument's own floor used as a treatment: activations redrawn with no relation to the
    target. A gain over this must be ~0 (spec §5, self-floor)."""
    return replace(f, acts=rng.normal(size=f.acts.shape) * np.linalg.norm(f.acts, axis=-1,
                                                                          keepdims=True).mean())


def rank_known_zero(f: RankFixture) -> RankFixture:
    """Analytically zero: every candidate direction is the SAME direction, so every candidate gets
    the identical score and the correct answer is a tie. A rank statistic must return exactly the
    midpoint here -- which is also the h34 rank-1-on-ties check, from the other side."""
    return replace(f, dirs=np.repeat(f.dirs[:1], f.k, axis=0))


# --------------------------------------------------------------------------------------------
# shift fixtures: `readout_shift`
# --------------------------------------------------------------------------------------------
@dataclass
class ShiftFixture:
    """Stage 14's shape. A shift is added at the patch layer and a readout is taken at a LATER
    layer, so `resid_read = base + shift + (whatever the blocks in between added)`.

    `extra` is the only thing a model can contribute that residual arithmetic cannot; `amplitude`
    is how much of it the planted model contributes. At amplitude 0 the treatment IS the
    pass-through, by construction, and the instrument must read exactly zero gain.
    """
    base: np.ndarray       # [n, d] base residual at the READOUT layer, unpatched
    shift: np.ndarray      # [d] the patched shift, propagated arithmetically
    direction: np.ndarray  # [d] unit readout direction
    extra: np.ndarray      # [n, d] the blocks' own contribution, orthogonalised against `shift`
    amplitude: float = 0.0

    @property
    def treatment(self) -> np.ndarray:
        return self.base + self.shift + self.amplitude * self.extra

    @property
    def passthrough(self) -> np.ndarray:
        return self.base + self.shift

    @property
    def n(self) -> int:
        return len(self.base)


def shift_noise(shape: tuple[int, int], rng: np.random.Generator, *, shift_norm: float = 1.0
                ) -> ShiftFixture:
    """A shift that IS decodable by the readout -- the h14 regime, where the pass-through already
    sits near the ceiling -- and blocks that add nothing. Declared null: zero gain."""
    n, d = shape
    direction = unit(rng.normal(size=d))
    base = rng.normal(size=(n, d))
    shift = shift_norm * float(np.linalg.norm(base, axis=-1).mean()) * direction
    extra = rng.normal(size=(n, d))
    extra = extra - np.outer(extra @ direction, direction)   # orthogonal to the readout direction
    return ShiftFixture(base=base, shift=shift, direction=direction, extra=unit(extra))


def shift_plant(f: ShiftFixture, a: float) -> ShiftFixture:
    """The blocks do something: add `a` units along the readout direction that the arithmetic did
    not carry. This is the ONLY thing a gain over pass-through is allowed to mean (spec §2a)."""
    scale = float(np.linalg.norm(f.base, axis=-1).mean())
    extra = f.direction[None, :].repeat(f.n, axis=0)
    return replace(f, extra=extra, amplitude=a * scale)


def shift_rescale(f: ShiftFixture, rng: np.random.Generator) -> ShiftFixture:
    """Rescale the base residual ONLY. `readout_shift` does not claim invariance to this and the
    battery records that it is genuinely not invariant, rather than quietly not testing it: a
    cosine against a fixed shift moves when the thing the shift is added to changes size."""
    c = rng.uniform(0.2, 5.0, size=(f.n, 1))
    return replace(f, base=f.base * c)


def shift_rotate(f: ShiftFixture, rng: np.random.Generator) -> ShiftFixture:
    Q = random_rotation(f.base.shape[1], rng)
    return replace(f, base=f.base @ Q, shift=f.shift @ Q, direction=f.direction @ Q,
                   extra=f.extra @ Q)


def shift_floor(f: ShiftFixture, rng: np.random.Generator) -> ShiftFixture:
    """Self-floor: the pass-through used as the treatment. Gain must be exactly zero."""
    return replace(f, extra=np.zeros_like(f.extra), amplitude=0.0)


def shift_known_zero(f: ShiftFixture) -> ShiftFixture:
    """Analytically zero: no shift at all and no block contribution, so the treatment is the
    unpatched readout and the pass-through is the unpatched readout."""
    return replace(f, shift=np.zeros_like(f.shift), extra=np.zeros_like(f.extra), amplitude=0.0)


# --------------------------------------------------------------------------------------------
# accuracy fixtures: `top1_accuracy` (piece 4)
#
# `top1_accuracy` reuses `RankFixture` deliberately -- an accuracy over k candidate directions and
# a rank over k candidate directions read the SAME object, and h29's era readout is literally
# `argmax_e cos(u, d_e)`. The generators above therefore apply unchanged, which is worth stating as
# a limitation and not only as economy: `selector`, `composition` and `top1_accuracy` now share
# `rank_noise`/`rank_plant` and `instruments.cosine_scores`, so an error in either breaks three
# instruments at once and their calibrations are not independent evidence. Piece 2 named that
# hazard for two instruments; piece 4 adds a third to the same family and does not pretend
# otherwise.
#
# One generator IS new, because the accuracy statistic has a failure mode the rank statistic does
# not: the tied field.
# --------------------------------------------------------------------------------------------
def rank_partial_tie(f: RankFixture, rng: np.random.Generator, n_tied: int = 2,
                     amplitude: float = 3.0) -> RankFixture:
    """The target ties with `n_tied - 1` other candidates AT THE TOP, for every item.

    A top-1 accuracy written as `argmax == target` awards a full hit to whichever candidate numpy's
    argmax happens to return first, which is the h34 rank-1-on-ties failure wearing the other hat:
    on a wholly tied field it reads 1.0 or 0.0 depending on nothing but which tied candidate the
    labelling calls correct. The shipped statistic splits the hit over the tied set (1/T), so this
    fixture must read exactly `n_tied ** -1`.

    Written before the statistic, like everything else in this file -- and then corrected by it,
    which is the ordering working rather than failing. The first version made the first `n_tied`
    directions identical and stopped there, so the tied pair did not necessarily WIN: four
    unrelated candidates were still in the race and the statistic read 0.091, not 0.5. A tie that
    is not at the top is not the h34 configuration. The activations are now pushed along the shared
    direction so the tied block takes the top places outright.
    """
    dirs = f.dirs.copy()
    dirs[:n_tied] = dirs[0]
    scale = float(np.linalg.norm(f.acts, axis=-1).mean())
    acts = f.acts + amplitude * scale * dirs[0]
    target = rng.integers(0, n_tied, size=f.n)
    return replace(f, acts=acts, dirs=dirs, target=target)


# --------------------------------------------------------------------------------------------
# ordered-target fixtures: `discrimination` (piece 4)
#
# h39's shape. Each SUBJECT (street, mountain, mayfly, ...) is described at m ordered intervals
# (1 day ... 1 000 000 years), and the question is whether a scalar read off the residual orders
# those intervals. The per-item statistic is therefore per SUBJECT, and the aggregate is the mean
# over subjects -- which is what `time_translation_discrimination.py` reports as `shared_mean`.
#
# The fixture carries a `floor` readout beside the treatment one because this instrument's null is
# the MEASURED FLOOR and not chance (spec §8): on a leaky grid the words alone order the intervals,
# and §6 requires the claim to report gain over that, never the raw score.
# --------------------------------------------------------------------------------------------
@dataclass
class OrderedFixture:
    """`S` subjects x `m` ordered levels of activation, one readout direction, one ordered target.

    `acts[s, j]` is the pooled residual for subject `s` at level `j`; `y[j]` is the level's ordered
    value (log Δt). `floor_scores[s, j]` is what a stimulus-only predictor gives for the same cell:
    the thing the treatment has to beat.
    """
    acts: np.ndarray            # [S, m, d]
    direction: np.ndarray       # [d]
    y: np.ndarray               # [m], strictly increasing
    floor_scores: np.ndarray    # [S, m]

    @property
    def n(self) -> int:
        return len(self.acts)

    @property
    def m(self) -> int:
        return self.acts.shape[1]


def ordered_noise(shape: tuple[int, int, int], rng: np.random.Generator) -> OrderedFixture:
    """(S, m, d) of pure Gaussian activations: the residual knows nothing about the level order.

    The floor readout is drawn the same way, so on this fixture BOTH the treatment and the floor are
    at chance and the declared calibration null is 0. That is deliberate and it is the one thing
    this battery cannot check for a caller: see `registry`'s `calibration_null` note.
    """
    S, m, d = shape
    return OrderedFixture(acts=rng.normal(size=(S, m, d)), direction=unit(rng.normal(size=d)),
                          y=np.arange(m, dtype=np.float64),
                          floor_scores=rng.normal(size=(S, m)))


def ordered_plant(f: OrderedFixture, a: float) -> OrderedFixture:
    """Push each cell `a` units along the readout direction IN PROPORTION to its level value.

    This is the only geometry a "the residual carries the interval" claim can mean: the component
    along the readout direction is monotone in the target. Nothing else changes -- same subjects,
    same levels, same noise, same direction.
    """
    scale = float(np.linalg.norm(f.acts, axis=-1).mean())
    z = (f.y - f.y.mean()) / (np.std(f.y) + 1e-12)              # [m]
    return replace(f, acts=f.acts + a * scale * z[None, :, None] * f.direction[None, None, :])


def ordered_rescale(f: OrderedFixture, rng: np.random.Generator) -> OrderedFixture:
    """Per-cell positive rescaling. A rank correlation of a projection is NOT invariant to this in
    general -- rescaling a cell rescales its projection -- so this is the transform that tells us
    whether the statistic reads the direction or the magnitude. `discrimination` claims scale
    invariance in the per-LAYER sense (one scalar for the whole stack), which is `ordered_rescale`
    with a single draw; the per-cell form is declared NOT claimed and measured anyway.
    """
    c = rng.uniform(0.2, 5.0, size=(f.n, f.m, 1))
    return replace(f, acts=f.acts * c)


def ordered_layer_rescale(f: OrderedFixture, rng: np.random.Generator) -> OrderedFixture:
    """One scalar on the whole stack: what changing layer or model scale does."""
    return replace(f, acts=f.acts * float(rng.uniform(0.2, 5.0)))


def ordered_rotate(f: OrderedFixture, rng: np.random.Generator) -> OrderedFixture:
    Q = random_rotation(f.acts.shape[-1], rng)
    return replace(f, acts=f.acts @ Q, direction=f.direction @ Q)


def ordered_monotone_target(f: OrderedFixture, rng: np.random.Generator) -> OrderedFixture:
    """Reparameterise the target by a strictly increasing nonlinear map.

    This is the invariance the h39 statistic actually leans on and nobody has ever tested: the
    target is `log Δt`, and the choice of log base -- or of Δt itself, or of grid index -- must not
    change the number. A Pearson correlation would move here; a rank correlation may not.
    """
    return replace(f, y=np.exp((f.y - f.y.mean()) / (np.std(f.y) + 1e-12)))


def ordered_floor(f: OrderedFixture, rng: np.random.Generator) -> OrderedFixture:
    """Self-floor: the instrument's own floor used as the treatment. Activations redrawn with no
    relation to the level order, so the statistic must read its null (spec §5)."""
    scale = float(np.linalg.norm(f.acts, axis=-1, keepdims=True).mean())
    return replace(f, acts=rng.normal(size=f.acts.shape) * scale)


def ordered_known_zero(f: OrderedFixture) -> OrderedFixture:
    """Analytically the null: within each subject every level has the IDENTICAL activation, so the
    readout is constant and every level is tied. A rank correlation against a constant is 0/0, and
    a statistic that returns 1.0, or NaN, on a dead readout is the h34 failure in this instrument's
    costume. Must read exactly 0."""
    return replace(f, acts=np.repeat(f.acts[:, :1], f.m, axis=1))
