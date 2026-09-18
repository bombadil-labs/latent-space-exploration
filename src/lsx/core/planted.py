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
