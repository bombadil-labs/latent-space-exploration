"""Offline checks on the Section 4.1 port: rendering, the vector recipe, and the z-pool.

None of this needs a model. What it pins is the part of the pipeline that a remote run cannot
check for you: that the scenarios parse the way their validator demands, that our TEXT rendering
carries exactly one BOS (Gemma's chat template emits its own, and the tokenizer default would add
a second), and that the ported vector arithmetic is the arithmetic and not a paraphrase of it.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

S = pytest.importorskip("painaxis_scenarios")
SCEN = json.loads(S.SCEN.read_text())


# ------------------------------------------------------------------ their rendering
def test_all_420_pass_their_validator():
    """Their screen silently drops items that fail this. If any fail, our pool is not their
    pool and every z-score is taken against a different denominator."""
    assert S.validate_candidates(SCEN) == []
    assert len(SCEN) == 420


def test_parse_turns_alternates_and_ends_on_an_empty_assistant_turn():
    for c in SCEN:
        turns = S.parse_turns(c["text"])
        roles = [r for r, _ in turns]
        assert roles[0] == "user"
        assert turns[-1] == ("assistant", "")
        assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


def test_the_two_gaslighting_items_with_a_visible_record_parse_as_three_turns():
    """gaslight_19/20 include an assistant turn that denies the claim; the other 18 do not.
    That split is the whole bridge to the conscription design's rule 1b, so it is pinned here
    rather than left to a comment."""
    with_record = [c["id"] for c in SCEN
                   if c["category"] == "gaslighting" and len(S.parse_turns(c["text"])) > 2]
    assert with_record == ["gaslight_19", "gaslight_20"]
    assert sum(c["category"] == "gaslighting" for c in SCEN) == 20


def test_as_ids_normalises_every_shape_apply_chat_template_can_return():
    assert S._as_ids([1, 2, 3]) == [1, 2, 3]
    assert S._as_ids([[1, 2, 3]]) == [1, 2, 3]
    assert S._as_ids({"input_ids": [1, 2, 3]}) == [1, 2, 3]
    assert S._as_ids({"input_ids": [[1, 2, 3]]}) == [1, 2, 3]

    class T:
        def tolist(self):
            return [[1, 2, 3]]
    assert S._as_ids(T()) == [1, 2, 3]
    with pytest.raises(TypeError):
        S._as_ids(["a", "b"])


def test_as_ids_rejects_the_shape_that_silently_yielded_an_int():
    """`obj["input_ids"][0]` on an UNBATCHED encoding is the first token, not the sequence.
    That is what the first version of the render assertion did, and an int compares unequal to
    a list without ever looking like a bug in the rendering."""
    assert S._as_ids({"input_ids": [2, 106, 1645]}) == [2, 106, 1645]


def test_render_chat_carries_exactly_one_bos(monkeypatch):
    class Tok:
        bos_token_id = 2

        def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=False):
            body = "".join(f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>\n"
                           for m in msgs)
            return "<bos>" + body + ("<start_of_turn>model\n" if add_generation_prompt else "")

    txt = S.render_chat(SCEN[0], Tok())
    assert txt.count("<bos>") == 1
    assert txt.endswith("<start_of_turn>model\n")
    assert "[User]:" not in txt and "[Assistant]:" not in txt


def test_render_chat_drops_only_the_trailing_empty_assistant_turn():
    class Tok:
        bos_token_id = 2

        def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=False):
            Tok.seen = list(msgs)
            return "x"

    g19 = next(c for c in SCEN if c["id"] == "gaslight_19")
    S.render_chat(g19, Tok())
    roles = [m["role"] for m in Tok.seen]
    assert roles == ["user", "assistant", "user"], roles
    assert all(m["content"] for m in Tok.seen), "an empty turn survived into the messages"


# ------------------------------------------------------------------ their vector arithmetic
def test_denoise_basis_takes_components_up_to_half_the_variance():
    rng = np.random.default_rng(0)
    # variances 100, 1, 1, ...: the first component alone passes 0.5, so the basis is size 1
    X = rng.normal(size=(60, 8))
    X[:, 0] *= 10
    X -= X.mean(0)
    basis = S.denoise_basis(X, np.zeros(8))
    assert basis.shape[0] == 1
    # and projecting it out kills that direction
    v = np.ones(8)
    out = S.project_out(v.copy(), basis)
    assert abs(float(np.dot(out, basis[0]))) < 1e-10


def test_denoise_basis_degenerate_input_returns_empty_basis():
    assert S.denoise_basis(np.zeros((1, 5)), np.zeros(5)).shape == (0, 5)


def test_pain_vector_is_the_difference_in_means_with_the_control_pcs_removed():
    rng = np.random.default_rng(1)
    d = 12
    cats = S.PAIN_CATEGORIES * 5 + S.CONTROL_CATEGORIES * 5
    acts = rng.normal(size=(len(cats), d))
    v = S.compute_pain_vector(acts, cats)
    cats_a = np.array(cats)
    raw = (acts[np.isin(cats_a, S.PAIN_CATEGORIES)].mean(0)
           - acts[np.isin(cats_a, S.CONTROL_CATEGORIES)].mean(0))
    # the denoised vector is the raw difference minus its own component along each removed PC
    ctrl = acts[np.isin(cats_a, S.CONTROL_CATEGORIES)]
    basis = S.denoise_basis(ctrl, ctrl.mean(0))
    assert np.allclose(v, S.project_out(raw.copy(), basis), atol=1e-8)


def test_unit_is_norm_one_and_safe_on_zero():
    v = S.unit(np.array([3.0, 4.0]))
    assert abs(np.linalg.norm(v) - 1.0) < 1e-12
    assert np.allclose(S.unit(np.zeros(4)), np.zeros(4))


def test_zscore_pool_centres_and_scales_against_the_pool():
    x = np.array([1.0, 2.0, 3.0, 10.0])
    z = S.zscore_pool(x)
    assert abs(float(z.mean())) < 1e-9
    assert abs(float(z.std()) - 1.0) < 1e-6


def test_build_vectors_uses_the_pooled_neutral_of_all_three_core_sets():
    """Their control directions are (category mean - POOLED neutral mean), denoised against the
    pooled neutral cloud. Using one set's neutral instead is a different direction, and the
    difference is invisible in any single number the pipeline prints."""
    rng = np.random.default_rng(2)
    d = 10
    cats = ["A1", "A2", "A3", "A4", "A5", "B", "C1", "C2", "D", "E"] * 4
    core = {n: rng.normal(size=(len(cats), d)) for n in S.S_SETS}
    core_cats = {n: list(cats) for n in S.S_SETS}
    vecs = S.build_vectors(core, core_cats, {})
    assert set(vecs) == {"s1_pain_vector", "s2_pain_vector", "fear_vector",
                         "negemotion_vector", "negworld_vector", "bodysens_vector"}
    pooled_neutral = np.concatenate([core[n][np.array(cats) == "D"] for n in S.S_SETS])
    nm = pooled_neutral.mean(0)
    basis = S.denoise_basis(pooled_neutral, nm)
    fear_rows = np.concatenate([core[n][np.array(cats) == "B"] for n in S.S_SETS])
    expect = S.project_out(fear_rows.mean(0) - nm, basis)
    assert np.allclose(vecs["fear_vector"], expect, atol=1e-8)


def test_build_vectors_adds_the_optional_control_sets_only_when_present():
    rng = np.random.default_rng(3)
    d = 10
    cats = ["A1", "A2", "A3", "A4", "A5", "B", "C1", "C2", "D", "E"] * 4
    core = {n: rng.normal(size=(len(cats), d)) for n in S.S_SETS}
    core_cats = {n: list(cats) for n in S.S_SETS}
    vecs = S.build_vectors(core, core_cats, {"Arousal_1P": rng.normal(size=(8, d))})
    assert "arousal_vector" in vecs
    assert "random_vector" not in vecs and "numb_vector" not in vecs
