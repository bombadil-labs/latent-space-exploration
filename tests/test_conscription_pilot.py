"""The paired statistic and its sign-flip null.

A sign-flip band is the exact permutation null for a paired difference under exchangeability of
the two arms within an item. Two things have to hold or it is decoration: it must be centred on
zero whatever the data, and it must widen with the spread of the per-item differences rather than
with their mean. Both are pinned here, along with the property that makes the gain meaningful --
that the floor is subtracted WITHIN each item, so a shift shared by both arms cancels.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

P = pytest.importorskip("conscription_pilot")


def test_sign_flip_null_is_centred_on_zero_whatever_the_data():
    rng = np.random.default_rng(0)
    for shift in (0.0, 5.0, -20.0):
        d = rng.normal(shift, 1.0, 24)
        m, lo, hi, p = P._band(d, np.random.default_rng(1))
        assert abs(m - d.mean()) < 1e-12
        assert lo < 0 < hi, (shift, lo, hi)
        assert abs(lo + hi) < 0.25 * (hi - lo), "band is not centred on zero"


def test_sign_flip_band_scales_with_the_rms_of_the_differences_not_their_spread():
    """This is the band's real limitation and it is worth pinning rather than assuming away.

    Flipping signs leaves |d_i| untouched, so the null's width is set by rms(d) -- NOT by the
    spread of d around its mean. A contrast where every item moves by the same large amount has a
    WIDER band than one where items move erratically about zero. Two consequences:

      * the band cannot be read as "how much two arms differ by rewording alone". That is a
        different quantity and needs a second no-claim arm to measure (the `neutral_b` question in
        conscription_direction.md §4); the sign-flip band is not a substitute for it.
      * the band is a property of the treatment's own magnitudes, so it would look identical on a
        pipeline that had silently broken -- which is why the random-direction and floor arms are
        not optional here.
    """
    rng = np.random.default_rng(2)
    big_mean = rng.normal(0.0, 0.1, 24) + 10.0   # huge |d|, tiny spread
    small_rms = rng.normal(0.0, 3.0, 24)         # zero mean, larger spread, smaller |d|
    _, lo_b, hi_b, _ = P._band(big_mean, np.random.default_rng(3))
    _, lo_s, hi_s, _ = P._band(small_rms, np.random.default_rng(3))
    assert np.sqrt((big_mean ** 2).mean()) > np.sqrt((small_rms ** 2).mean())
    assert (hi_b - lo_b) > (hi_s - lo_s), "band does not track rms"
    # and it tracks it closely: width ~ 2 * 1.96 * rms / sqrt(n)
    for d, lo, hi in ((big_mean, lo_b, hi_b), (small_rms, lo_s, hi_s)):
        expect = 2 * 1.96 * np.sqrt((d ** 2).mean()) / np.sqrt(len(d))
        assert abs((hi - lo) - expect) / expect < 0.12, (hi - lo, expect)


def test_a_consistent_per_item_difference_is_detected_and_noise_is_not():
    rng = np.random.default_rng(4)
    real = rng.normal(0.5, 0.2, 24)             # every item shifted the same way
    noise = rng.normal(0.0, 0.2, 24)
    assert P._band(real, np.random.default_rng(5))[3] < 0.01
    assert P._band(noise, np.random.default_rng(5))[3] > 0.05


def test_p_value_is_two_sided():
    rng = np.random.default_rng(6)
    d = rng.normal(-0.5, 0.2, 24)               # a consistent NEGATIVE difference
    assert P._band(d, np.random.default_rng(7))[3] < 0.01, "a negative effect must also be detected"


def test_gain_cancels_a_shift_shared_by_both_arms():
    """The floor is subtracted per item. If an item is simply louder on the axis in both arms,
    that must not show up as conscription."""
    rng = np.random.default_rng(8)
    per_item = rng.normal(0, 5.0, 24)           # item-level nuisance, identical in both arms
    d_acts = per_item + 0.3
    d_floor = per_item
    gm, lo, hi, p = P._band(d_acts - d_floor, np.random.default_rng(9))
    assert abs(gm - 0.3) < 1e-9, "the item-level nuisance did not cancel"
    _, lo_a, hi_a, _ = P._band(d_acts, np.random.default_rng(9))
    assert (hi - lo) < (hi_a - lo_a) / 10, "subtracting the floor did not narrow the band"
    assert p < 0.001, "a consistent +0.3 over 24 items should clear its own band"


def test_pairs_cover_every_arm_and_include_the_no_record_variant():
    arms = set(P.ARMS) | {P.EXTRA}
    seen = {a for pair in P.PAIRS for a in pair}
    assert seen == arms, seen ^ arms
    assert ("enact", "report") in P.PAIRS, "the study's central contrast is not measured"
    assert ("enact", P.EXTRA) in P.PAIRS, "rule 1b's exclusion is not testable without this pair"


def test_window_is_the_one_determined_on_their_stimuli():
    """10..17 comes from painaxis_scenarios.md, measured on the paper's 420 items. If this ever
    gets tuned to the grid's own contrasts it becomes layer selection on scoring data."""
    assert P.WINDOW == list(range(10, 18))
