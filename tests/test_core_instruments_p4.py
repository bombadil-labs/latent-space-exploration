"""Piece 4's two instruments, and the three bugs building them exposed (spec §5, §8).

`tests/test_core_registry.py` covers piece 2's three instruments. This file covers
`top1_accuracy` and `discrimination`, and -- more to the point -- pins the three things that went
wrong on the way, each of which passed every test that existed before it was written:

  * a known-zero point that read zero for the wrong reason;
  * a calibration key that did not change when the arithmetic underneath the statistic did;
  * a per-item spread taken from the fixture's TYPE rather than from the instrument.
"""
import numpy as np
import pytest

from lsx.core import checks, instruments, planted, registry


# ---------------------------------------------------------------------------------------------
# the battery
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["top1_accuracy", "discrimination"])
def test_the_six_test_battery_passes(name):
    rep = instruments.build(name).calibrate()
    assert rep.passed, rep.summary()
    assert set(rep.tests_run) == {"noise", "sensitivity", "degeneracy", "self_floor",
                                  "invariance", "known_zero"}
    assert not rep.degenerate


@pytest.mark.parametrize("name", ["top1_accuracy", "discrimination"])
def test_sensitivity_is_monotone_and_does_not_saturate(name):
    """A curve that has hit its ceiling satisfies "monotone in the planted size" while testing
    nothing at the top. Both instruments' first amplitude set did exactly that (1.000 and 0.994),
    and the shipped ones were chosen to span the moving part of the curve."""
    rep = instruments.build(name).calibrate()
    v = list(rep.sensitivity_values)
    assert v == sorted(v), v
    assert v[-1] < 0.99, f"the sensitivity curve saturates at {v[-1]}: the top step tests nothing"
    assert v[-1] - v[0] > 0.2, v


@pytest.mark.parametrize("name,n,expected", [("top1_accuracy", 4000, 0.4714),
                                             ("discrimination", 4000, 0.3536)])
def test_declared_null_item_sd_matches_the_statistic(name, n, expected):
    """The closed form, checked against a Monte-Carlo draw from the real statistic. A closed form
    no test compares to the instrument is a declaration, not a measurement."""
    inst = instruments.build(name)
    declared = inst.spec.null_item_sd(inst.config)
    measured = inst.measure_null_item_sd(n=n)
    assert declared == pytest.approx(expected, abs=0.001)
    assert measured == pytest.approx(declared, rel=0.06), (measured, declared)


def test_top1_null_is_one_over_k_and_moves_the_other_way_from_a_rank():
    """The null this instrument exists for. A rank's midpoint GROWS with k; an accuracy's chance
    SHRINKS. Reading h29's 0.84 against a rank's null would be a category error, not a rounding
    one."""
    for k in (2, 3, 6):
        acc = registry.spec("top1_accuracy").null_value({"n_candidates": k})
        rank = registry.spec("selector").null_value({"n_candidates": k})
        assert acc == pytest.approx(1.0 / k)
        assert rank == pytest.approx((k + 1) / 2)
    assert (registry.spec("top1_accuracy").null_value({"n_candidates": 6})
            < registry.spec("top1_accuracy").null_value({"n_candidates": 3}))
    assert (registry.spec("selector").null_value({"n_candidates": 6})
            > registry.spec("selector").null_value({"n_candidates": 3}))


# ---------------------------------------------------------------------------------------------
# ties: h34's rule, in the accuracy family
# ---------------------------------------------------------------------------------------------
def test_a_tied_field_reads_exactly_chance_and_the_naive_variant_reads_the_labelling():
    """`planted.rank_partial_tie` is the fixture that separates the shipped statistic from the
    obvious one. Written before either.

    The separation is sharper than "the naive one is wrong on average": on a DEAD field of tied
    candidates, `argmax == target` returns whatever the labelling happens to be -- 1.0 if the
    target is the first of the tied block and 0.0 if it is the second -- while the same data read
    by the shipped statistic is 0.5 either way. Nothing about the model changed between those two
    numbers. That is h34: a dead patch reading as a perfect lens.
    """
    rng = np.random.default_rng(0)
    f = planted.rank_noise((400, 64, 6), rng)
    tied = planted.rank_partial_tie(f, rng, n_tied=2)
    # the tie is AT THE TOP: every item's two winners are the identical pair
    s = instruments.cosine_scores(tied.acts, tied.dirs)
    assert np.all(np.argsort(-s, axis=1)[:, :2].max(axis=1) <= 1)
    assert instruments.top1_accuracy_rate(tied) == pytest.approx(0.5, abs=1e-12)
    tied3 = planted.rank_partial_tie(f, rng, n_tied=3)
    assert instruments.top1_accuracy_rate(tied3) == pytest.approx(1 / 3, abs=1e-12)

    first = planted.replace(tied, target=np.zeros(tied.n, dtype=int))
    second = planted.replace(tied, target=np.ones(tied.n, dtype=int))
    assert instruments.naive_top1_accuracy(first) == pytest.approx(1.0)
    assert instruments.naive_top1_accuracy(second) == pytest.approx(0.0)
    assert instruments.top1_accuracy_rate(first) == pytest.approx(0.5)
    assert instruments.top1_accuracy_rate(second) == pytest.approx(0.5)


def test_top1_hit_splits_a_tie_and_midrank_mid_ranks_it():
    assert checks.top1_hit([1.0, 1.0, 1.0], 0) == pytest.approx(1 / 3)
    assert checks.top1_hit([2.0, 1.0, 1.0], 0) == 1.0
    assert checks.top1_hit([1.0, 2.0, 1.0], 0) == 0.0
    assert checks.midrank([1.0, 1.0, 1.0], 0) == 2.0


# ---------------------------------------------------------------------------------------------
# the known-zero point that read zero for the wrong reason
# ---------------------------------------------------------------------------------------------
def test_a_dead_readout_reads_zero_per_subject_not_plus_or_minus_one():
    """The bug the known-zero test caught, pinned.

    Identical activation vectors run through one batched matmul come back differing by ~4e-16,
    because BLAS does not promise the same summation order for every row of a batch. Spearman ranks
    that difference: before `dot_tie_atol`, a wholly dead readout scored +-0.548 PER SUBJECT and the
    instrument read ~0 only because the signs cancelled in the mean. The aggregate was fine and the
    statistic was wrong.
    """
    rng = np.random.default_rng(0)
    f = planted.ordered_noise((200, 9, 64), rng)
    kz = planted.ordered_known_zero(f)

    scores = kz.acts @ planted.unit(kz.direction)
    spread = float(np.max(scores.max(axis=1) - scores.min(axis=1)))
    assert 0 < spread < 1e-12, (
        "the fixture no longer exercises the hazard: identical rows are now bit-identical, so this "
        f"test would pass for a statistic with no tie tolerance at all (spread {spread})")

    per_item = instruments.discrimination_per_item(kz)
    assert np.all(per_item == 0.0), np.unique(per_item)

    # and without the tolerance -- which is what the first version did
    naive = np.array([checks.spearman(scores[s], kz.y) for s in range(len(scores))])
    assert np.abs(naive).max() > 0.5, (
        "the no-tolerance version no longer misreads a dead readout, so this regression test is "
        "no longer testing anything")
    assert abs(naive.mean()) < 1e-9, "the old bug hid in the MEAN, which is why it survived"


def test_the_tie_tolerance_is_computed_from_the_arithmetic_not_chosen():
    """A flat tolerance anywhere in this core is a bug (piece 2). This one is a float64 dot-product
    error bound: it scales with the accumulation length and with the numbers' own magnitude, and at
    Gemma's width it is eleven orders of magnitude below anything a readout could mean."""
    small = checks.dot_tie_atol(64, 0.3)
    gemma = checks.dot_tie_atol(3584, 20.0)
    assert small < 1e-13 and gemma < 1e-10
    assert checks.dot_tie_atol(7168, 20.0) == pytest.approx(2 * gemma)
    assert checks.dot_tie_atol(3584, 40.0) == pytest.approx(2 * gemma)
    # a real difference survives it
    assert checks.spearman([1.0, 1.0 + 1e-6, 2.0], [1, 2, 3], atol=gemma) == pytest.approx(1.0)


def test_spearman_matches_the_textbook_where_there_are_no_ties():
    rng = np.random.default_rng(5)
    for _ in range(5):
        x, y = rng.normal(size=25), rng.normal(size=25)
        rx = np.argsort(np.argsort(x)) + 1.0
        ry = np.argsort(np.argsort(y)) + 1.0
        want = np.corrcoef(rx, ry)[0, 1]
        assert checks.spearman(x, y) == pytest.approx(want, abs=1e-10)


# ---------------------------------------------------------------------------------------------
# the calibration key that did not move when the arithmetic did
# ---------------------------------------------------------------------------------------------
def test_the_calibration_key_covers_the_statistics_dependencies():
    """`discrimination_rho` is one line -- `mean(discrimination_per_item(f))` -- so a key over its
    own source alone cannot see a change to what it delegates to. It happened: the tie-tolerance
    fix changed the arithmetic and left every cached report valid.

    Checked on the real statistics rather than on a toy, because the closure only follows callees
    whose module is inside `lsx.` and a toy defined in this file is not.
    """
    for stat, dep in ((instruments.discrimination_rho, "def discrimination_per_item"),
                      (instruments.discrimination_rho, "def spearman"),
                      (instruments.selector_rank, "def midrank"),
                      (instruments.top1_accuracy_rate, "def top1_hit"),
                      (instruments.composition_rank, "def cosine_scores")):
        src = "".join(checks._source_closure(stat))
        assert dep in src, f"{stat.__name__}'s key does not cover {dep}"

    inst = instruments.build("discrimination")
    assert checks.calibration_key(instruments.discrimination_rho, inst.calibration_null,
                                  inst.invariances) == inst.key


def test_the_closure_follows_calls_made_inside_comprehensions():
    """The hole in the fix for the hole.

    `selector_rank` calls `midrank` inside a LIST COMPREHENSION, which compiles to its own code
    object whose names do not appear in the enclosing `co_names`. The first closure walker read
    only the top level, found `cosine_scores` and missed `midrank` -- so editing the tie rule that
    h34 turned on would still not have re-calibrated the instrument resting on it.
    """
    names = checks._referenced_names(instruments.selector_rank.__code__)
    assert "midrank" in names and "cosine_scores" in names
    assert "midrank" not in set(instruments.selector_rank.__code__.co_names), (
        "the call moved out of the comprehension; this regression test no longer tests anything")


def test_the_key_is_not_a_repo_wide_version():
    """§5 is explicit: editing one instrument re-calibrates that one and only that one. The closure
    must not sweep in the whole package."""
    keys = {n: instruments.build(n).key for n in instruments.BUILDERS}
    assert len(set(keys.values())) == len(keys), keys
    # `readout_shift` does not call the ranking code, so the rank instruments' primitives are not
    # in its closure
    src = "".join(checks._source_closure(instruments.readout_shift_gain))
    assert "def midrank" not in src and "def top1_hit" not in src
    rank_src = "".join(checks._source_closure(instruments.selector_rank))
    assert "def midrank" in rank_src and "def cosine_scores" in rank_src


# ---------------------------------------------------------------------------------------------
# the per-item spread taken from the fixture instead of the instrument
# ---------------------------------------------------------------------------------------------
def test_two_instruments_sharing_a_fixture_do_not_share_a_per_item_spread():
    """`top1_accuracy` and `selector` read the SAME `RankFixture`. Dispatching the per-item score on
    the fixture's type -- which is what the code did before this piece -- handed an accuracy the
    per-item spread of a mid-rank: 0.816 where the truth is 0.471 at k=3, so every arm band on
    every h29-shaped claim would have been 1.7x too loose."""
    acc = instruments.build("top1_accuracy", n_candidates=3)
    sel = instruments.build("selector", n_candidates=3)
    assert isinstance(acc.noise_fn(np.random.default_rng(0)), planted.RankFixture)
    assert isinstance(sel.noise_fn(np.random.default_rng(0)), planted.RankFixture)
    assert acc.measure_null_item_sd() == pytest.approx(0.4714, abs=0.02)
    assert sel.measure_null_item_sd() == pytest.approx(0.8165, abs=0.02)
    assert acc.tolerance(55) < sel.tolerance(55)


# ---------------------------------------------------------------------------------------------
# the calibration null that is not the arms' null
# ---------------------------------------------------------------------------------------------
def test_discriminations_battery_runs_at_chance_and_its_arms_sit_at_the_measured_floor():
    """The distinction piece 4 had to introduce, and the bug it avoids.

    `discrimination`'s declared null IS the caller's measured floor (§8). Running the battery at
    that number would test a synthetic fixture against a measurement of someone else's grid, and --
    because the key hashes the null -- would demand a fresh battery for every floor of an unchanged
    statistic. That is piece 3's config-dependent-key bug one level along.
    """
    a = instruments.build("discrimination", floor=0.522)
    b = instruments.build("discrimination", floor=0.728)
    assert a.declared_null == pytest.approx(0.522)
    assert b.declared_null == pytest.approx(0.728)
    assert a.calibration_null == b.calibration_null == 0.0
    assert a.key == b.key, "a different measured floor re-keyed an unchanged statistic"
    assert a.calibrate().passed and b.calibrate().passed


def test_the_battery_does_not_verify_the_callers_floor_and_says_so():
    """Stated as a limitation rather than left implicit: nothing in the battery can check that a
    declared floor is the floor of the caller's grid."""
    doc = registry.spec("discrimination").calibration_null_doc
    assert "chance" in doc.lower() and "floor" in doc.lower()


def test_discrimination_claims_invariance_to_a_monotone_reparameterisation_of_the_target():
    """The invariance h39's statistic leans on and nobody had tested: log Δt, Δt and grid index
    must give the same number."""
    rep = instruments.build("discrimination").calibrate()
    inv = {r.name: r for r in rep.invariance}
    assert inv["monotone_target"].claimed and inv["monotone_target"].delta < 1e-9
    # and the one it explicitly does NOT claim is measured rather than skipped
    assert not inv["cell_rescale"].claimed
    assert inv["cell_rescale"].delta > 1e-6
    assert any("cell_rescale" in n for n in rep.notes)


def test_an_unbuilt_instrument_is_still_refused_for_the_honest_reason():
    """`generality` stays unbuilt: it has no null, and piece 4 did not invent one."""
    assert registry.spec("generality").implemented is False
    assert np.isnan(registry.spec("generality").null_value())
    with pytest.raises(registry.InstrumentNotImplemented):
        instruments.build("generality")
    for name in ("crosstalk", "depth_gain"):
        assert registry.spec(name).implemented is False


# ---------------------------------------------------------------------------------------------
# the positive controls: both instruments can actually produce a Claim
#
# Every §1A row these two touch comes back REFUSED or deferred, for reasons that belong to h29's
# and h39's batteries rather than to the instruments. That makes it possible to ship an instrument
# that refuses everything and call it rigorous, so each one is also shown BUILDING a claim.
# ---------------------------------------------------------------------------------------------
def test_a_complete_top1_accuracy_claim_constructs_and_reports_against_one_over_k():
    from lsx.core.types import Floor, Measured, Selection

    inst = instruments.build("top1_accuracy", n_candidates=3)
    rng = np.random.default_rng(0)
    n = 55
    treat = (rng.random(n) < 0.84).astype(float)
    claim = inst.claim(
        treatment=Measured(treat, label="era reads as target"),
        arms={"random": (rng.random(n) < 1 / 3).astype(float),
              "no_patch": (rng.random(n) < 1 / 3).astype(float)},
        floor=Floor(stimulus=0.30, estimator=1 / 3),
        selection=Selection(axis=None, rule="pre-registered scale 3.0"),
        provenance={"model": "google/gemma-2-9b-it"}, stage="synthetic")
    assert claim.arms["random"].expected_null == pytest.approx(1 / 3)
    assert claim.effect.computed and claim.effect.verdict == 1
    assert "arm random" in claim.render() and "arm no_patch" in claim.render()

    # an arm that is NOT at chance is refused, which is the h34 rule
    with pytest.raises(checks.ArmOffNull):
        inst.claim(treatment=Measured(treat, label="x"),
                   arms={"random": np.full(n, 0.8), "no_patch": np.full(n, 1 / 3)},
                   floor=Floor(stimulus=0.30, estimator=1 / 3),
                   selection=Selection(axis=None, rule="pre-registered"),
                   provenance={"model": "m"}, stage="synthetic")


def test_a_complete_discrimination_claim_reports_gain_over_the_measured_floor():
    from lsx.core.types import Floor, Measured, Selection

    floor_value = 0.52
    inst = instruments.build("discrimination", m=9, floor=floor_value)
    rng = np.random.default_rng(1)
    treat = np.clip(rng.normal(0.96, 0.03, size=8), -1, 1)
    floor = np.clip(rng.normal(floor_value, 0.05, size=8), -1, 1)
    shuffled = np.clip(rng.normal(floor_value, 0.05, size=8), -1, 1)
    claim = inst.claim(
        treatment=Measured(treat, label="delta-t discrimination"),
        arms={"floor": floor, "shuffled_stimulus": shuffled},
        floor=Floor(stimulus=float(floor.mean()), estimator=0.0),
        selection=Selection(axis=None, rule="pre-registered layer 14"),
        provenance={"model": "Qwen/Qwen2.5-1.5B"}, stage="synthetic",
        report_as="gain_over_floor")
    assert claim.reported_value == pytest.approx(treat.mean() - floor.mean(), abs=1e-9)
    assert claim.reported_value < claim.treatment.value, "a gain must not be the raw score"
    assert "gain over stimulus floor" in claim.render()
    # both arms took their null from the registry, i.e. from the MEASURED floor
    assert claim.arms["floor"].expected_null == pytest.approx(floor_value)
    assert claim.arms["shuffled_stimulus"].expected_null == pytest.approx(floor_value)
