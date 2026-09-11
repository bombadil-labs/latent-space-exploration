import numpy as np
import torch

from lsx import compare, extract, operate, parse_roles, steer
from lsx.model import Patch

P1 = "first, [[thesis: a particle has a definite position]]. then [[antithesis: measurement disturbs what it measures]]. finally [[synthesis: position and momentum are complementary descriptions]]."
P2 = "first, [[thesis: a child is embedded in its impulses]]. then [[antithesis: later the impulses become objects]]. finally [[synthesis: the child can hold what once held it]]."
ROLES = ["thesis", "antithesis", "synthesis"]


def test_parse_roles():
    p = parse_roles("a [[x: bb]] c [[y: dd]] e [[x: ff]]")
    assert p.text == "a bb c dd e ff"
    assert p.spans == {"x": [(2, 4), (12, 14)], "y": [(7, 9)]}
    for role, spans in p.spans.items():
        for s, e in spans:
            assert p.text[s:e] in ("bb", "dd", "ff")


def test_extract_shapes(tiny_lm):
    a = extract(tiny_lm, P1)
    assert set(a.roles) == set(ROLES)
    assert a.resid.shape[0] == tiny_lm.n_layers + 1
    for r in ROLES:
        assert a.roles[r].shape == (tiny_lm.n_layers + 1, tiny_lm.d_model)
        assert len(a.tokens[r]) > 0


def test_compare_measures(tiny_lm):
    a, b = extract(tiny_lm, P1), extract(tiny_lm, P2)
    A, B = compare.role_stack(a, ROLES), compare.role_stack(b, ROLES)
    s = compare.sweep(A, B, compare.rsa)
    assert s.shape == (tiny_lm.n_layers + 1,) and np.all(np.abs(s[~np.isnan(s)]) <= 1)
    assert 0 <= compare.linear_cka(A[1], B[1]) <= 1
    assert compare.linear_cka(A[1], A[1]) > 0.999
    assert compare.procrustes_residual(A[1], A[1]) < 1e-6
    gw, T = compare.gromov_wasserstein(a.resid[2], b.resid[2])
    assert gw >= 0 and T.shape == (a.resid.shape[1], b.resid.shape[1])
    names, M = compare.similarity_matrix({"p1": A, "p2": B}, layer=2, fn=compare.linear_cka)
    assert names == ["p1", "p2"] and np.allclose(M, M.T)


def test_operator_recovers_planted_relation():
    rng = np.random.default_rng(0)
    d, n = 16, 40
    S = rng.normal(size=(n, d))
    Wtrue = np.eye(d) + 0.3 * rng.normal(size=(d, d))
    btrue = rng.normal(size=d)
    O = S @ Wtrue.T + btrue + 0.01 * rng.normal(size=(n, d))
    groups = np.repeat(np.arange(4), n // 4)
    op = operate.fit_affine(S, O, layer=1, src="thesis", dst="antithesis", ridge=1e-3)
    assert np.allclose(op(S), O, atol=0.2)
    assert op.spin_basis.shape[1] == d
    ev = operate.holdout_eval(S, O, groups, 1, "thesis", "antithesis", ridge=1e-3)
    assert ev["cos_pred"] > ev["cos_identity"] and ev["cos_pred"] > ev["cos_mean"]
    assert ev["rank"] <= 2


def test_patching_changes_output(tiny_lm):
    a = extract(tiny_lm, P1)
    v = a.roles["synthesis"][2] - a.roles["thesis"][2]
    prompt = "first, a particle has a definite position. then"
    base = tiny_lm.next_token_logits(prompt)
    pat = tiny_lm.next_token_logits(prompt, steer.steer_patches(v, layers=[1, 2], scale=5.0))
    assert not torch.allclose(base, pat)
    # patch handles are removed after the context: a clean forward matches the original
    assert torch.allclose(base, tiny_lm.next_token_logits(prompt))
    out = steer.readout(tiny_lm, prompt, [Patch(1, steer.add_vector(v, 5.0))], max_new_tokens=3)
    assert set(out) == {"base", "patched"}
    lens = tiny_lm.unembed(torch.as_tensor(a.roles["thesis"][-1]), k=3)
    assert len(lens) == 3
