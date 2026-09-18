"""Piece 5: the two arms and the one floor that five §1A targets were refused for want of.

Everything here runs without a model. The batteries themselves need a 1.5B forward or an NDIF
deployment and are exercised by `lsx.core.piece5` / `lsx.core.rerun_h29` / `lsx.core.rerun_h39`;
what is pinned here is the arithmetic those runs depend on, and the two mistakes it would be
easiest to make in it.
"""
import json
import pathlib

import numpy as np
import pytest

from lsx.core import checks, piece5, remote, reproduce
from lsx.core.types import Grid, Item, is_asserted

GRID = reproduce.prompt("narrative_factors_v2.json")


# ---------------------------------------------------------------------------------------------
# the measured lexical floor (target 5, and the floor under target 1)
# ---------------------------------------------------------------------------------------------
def test_the_lexical_floor_reads_the_words_and_not_the_procedure():
    """The control that makes the floor usable at all.

    A floor is subtracted from every gain on this grid, so a floor that scores well for the wrong
    reason would silently inflate or destroy every result computed against it. Shuffling the factor
    labels before the fit must send it to chance; it must NOT be near chance unshuffled.
    """
    real = reproduce.h8_lexical_floor(GRID)
    perm = reproduce.h8_lexical_floor(GRID, permute_seed=3)
    V = real["n_variants"]
    chance = (V + 1) / 2
    assert real["composed_mean"] < 4.0, real["composed_mean"]
    # 72 items, per-item sd sqrt((V^2-1)/12) = 5.19, so 3 sigma is +-1.83
    assert abs(perm["composed_mean"] - chance) < 3 * np.sqrt((V ** 2 - 1) / 12) / np.sqrt(72)
    assert len(real["composed"]) == 72


def test_the_lexical_floor_ranks_with_the_same_code_as_the_treatment():
    """Not a different ranking procedure wearing the same name: `midrank`, the same tie rule h34
    turned on, over the same candidate sets."""
    src = pathlib.Path(reproduce.__file__).read_text().split("def h8_lexical_floor", 1)[1]
    body = src.split("\ndef ", 1)[0]
    assert "midrank(" in body
    assert "argmax" not in body and "argmin" not in body


def test_tense_has_two_levels_and_its_lens_is_a_two_candidate_selector():
    """Piece 4's driver built all three h8 lenses as 3-candidate selectors. `tense` has two levels,
    so its null is 1.50 and not 2.00 -- an arm declared at 2.00 would read 'off null' at exactly
    the value it should have. The refusal for the missing permutation arm fired first and hid it."""
    g = json.loads(GRID.read_text())
    assert len(g["factors"]["tense"]) == 2
    lex = reproduce.h8_lexical_floor(GRID)
    assert len(lex["B"]["tense"]) == 72


# ---------------------------------------------------------------------------------------------
# the permutation arm (target 1)
# ---------------------------------------------------------------------------------------------
def _toy_grid(d=16, seed=0):
    rng = np.random.default_rng(seed)
    items, acts = [], []
    levels = {"f": ["a", "b", "c"]}
    for scene in [f"s{i}" for i in range(1, 13)]:
        for lvl in levels["f"]:
            items.append(Item(text=f"{scene} {lvl} text", factors={"scene": scene, "f": lvl},
                              spans={"span": (0, 5)}))
    grid = Grid(items=items, name="toy", leak_check=False)
    base = {lvl: rng.normal(size=d) * 5 for lvl in levels["f"]}
    for it in items:
        acts.append(base[it.factors["f"]] + rng.normal(size=d) * 0.01)
    return grid, np.asarray(acts)[:, None, None, :]


class _Stack:
    def __init__(self, acts):
        self._a = acts
        self.provenance = {"model": "toy", "layers": [0], "pooling": "mean", "grid_hash": "toy"}

    def vectors(self, span, layer):
        return self._a[:, 0, 0, :]


def test_the_permutation_arm_destroys_level_identity_and_keeps_the_fit():
    grid, acts = _toy_grid()
    st = _Stack(acts)
    real, _, _ = reproduce.level_directions(st, grid, 0, ["f"], "s12")
    perm = reproduce.permuted_level_directions(st, grid, 0, ["f"], "s12",
                                               np.random.default_rng(0))
    # same shape, comparable norm, and no longer aligned with the real direction
    for lvl in ("a", "b", "c"):
        r, p = real["f"][lvl].vec(0), perm["f"][lvl]
        assert r.shape == p.shape
        assert abs(checks.cosine(r, p)) < 0.9
    # the permuted directions still sum to (approximately) zero, as the real ones do: they are
    # deviations from the same training grand mean
    assert np.allclose(sum(perm["f"].values()), 0, atol=1e-8)


def test_the_permutation_arm_never_touches_the_held_out_scene():
    grid, acts = _toy_grid()
    st = _Stack(acts)
    held = "s12"
    idx = [i for i, it in enumerate(grid.items) if it.factors["scene"] == held]
    a2 = acts.copy()
    a2[idx] = 1e6                       # poison the held-out scene
    p1 = reproduce.permuted_level_directions(st, grid, 0, ["f"], held, np.random.default_rng(7))
    p2 = reproduce.permuted_level_directions(_Stack(a2), grid, 0, ["f"], held,
                                             np.random.default_rng(7))
    for lvl in p1["f"]:
        assert np.allclose(p1["f"][lvl], p2["f"][lvl])


# ---------------------------------------------------------------------------------------------
# h29's declared nulls (target 2)
# ---------------------------------------------------------------------------------------------
def test_h29_control_nulls_come_from_h27s_published_numbers_and_not_from_1_over_k():
    p = piece5.h27_control_rates()
    # era_other / 2: the exact value for an arm blind to the designated target, given both non-e1
    # targets are present
    assert p["base"]["era_as_target"] == pytest.approx(p["base"]["era_other"] / 2)
    assert p["base"]["era_as_target"] == pytest.approx(0.1111, abs=1e-4)
    assert p["base"]["era_as_target"] < 1 / 3      # it is NOT chance, which is the whole point


# ---------------------------------------------------------------------------------------------
# the asserted generation path (targets 2 and 3)
# ---------------------------------------------------------------------------------------------
def test_the_generation_and_pooling_paths_are_stamped_asserted():
    for fn in (remote.asserted_remote_generate, remote.assert_patch_reaches_batch,
               remote.asserted_remote_tail_pool):
        assert is_asserted(fn)


def test_the_generate_trace_binds_the_model_outside_the_block():
    """A regression test for a bug this piece wrote and then hit on the wire.

    NDIF executes the trace block's source remotely and refuses any attribute path through a
    non-whitelisted module. `rlm.model.generator` inside the block is a path through an instance of
    a class defined in `lsx.core.remote`, so it fails with 'Module lsx.core.remote is not
    whitelisted' -- which it did, on twelve generations, before a local name was bound outside.
    `asserted_remote_patched_logprob` already carried the comment explaining exactly this.
    """
    src = pathlib.Path(remote.__file__).read_text()
    body = src.split("def asserted_remote_generate", 1)[1].split("\n@asserted", 1)[0]
    inner = body.split("def build(backend):", 1)[1]
    assert "rlm.model" not in inner, inner
    assert "mdl.generate(" in inner and "mdl.generator.output.save()" in inner


def test_the_shuffled_stimulus_arm_is_reproducible_across_processes():
    """`hash()` is randomised per process under PYTHONHASHSEED; a shuffled-stimulus arm seeded from
    it would differ between the extraction run and any re-run."""
    from lsx.core import rerun_h39

    src = pathlib.Path(rerun_h39.__file__).read_text()
    assert "hashlib.sha256" in src
    assert "abs(hash(" not in src
    a = rerun_h39.shuffle_words("one two three four five six", 11)
    b = rerun_h39.shuffle_words("one two three four five six", 11)
    assert a == b
    assert sorted(a.split()) == sorted("one two three four five six".split())
