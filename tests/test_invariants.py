"""Semantic invariants: properties that must hold if the measurement machinery is sound.

Each test names the retraction that bought it (see docs/INSTRUMENTS.md). These are not unit tests of
code paths; they are the error signal that measurement code otherwise lacks, because a wrong number
renders identically to a right one. Fast by construction (tiny random LM, CPU) so a hook can run
them on every edit to measurement code.
"""
import numpy as np
import pytest
import torch

from lsx import extract, steer
from lsx.model import Patch

P = "first, [[thesis: a particle has a definite position]]. then [[antithesis: measurement disturbs what it measures]]. finally [[synthesis: position and momentum are complementary descriptions]]."
ROLES = ["thesis", "antithesis", "synthesis"]


# --- h40: a readout at or after the patch layer moves by residual arithmetic ------------------

def test_zero_patch_is_identity(tiny_lm):
    """A zero-magnitude patch must leave every downstream residual bit-identical.

    If this fails, the patch machinery is perturbing something other than the vector it was given,
    and no 'the shift moved the readout' result means anything.
    """
    base, _ = tiny_lm.residuals(P)
    zero = np.zeros(tiny_lm.d_model, dtype=np.float32)
    with tiny_lm.patched(steer.steer_patches(zero, layers=[1], scale=1.0)):
        after, _ = tiny_lm.residuals(P)
    assert torch.allclose(base, after, atol=0, rtol=0), "zero patch changed the residual stream"


def test_patch_constant_dominates_downstream_delta(tiny_lm):
    """The pass-through structure that killed stage 14, stated correctly.

    A constant added at every position at layer L passes into that block's output ALMOST verbatim:
    the delta is the constant plus the block's own nonlinear response to its changed input. The
    additive part dominates; the remainder is the only thing the model contributed.

    That remainder is exactly what a pass-through arm measures. Stage 14 reported 0.89 "address
    moved" where the arithmetic alone gave 0.79-0.97 and the norm-matched arm gave 1.00, so the
    model's contribution was zero or negative. Written as an assertion: if this test ever shows the
    remainder dominating instead, the pass-through arm has stopped being the right control.
    """
    L = 1
    base, _ = tiny_lm.residuals(P)
    c = 0.37
    v = np.ones(tiny_lm.d_model, dtype=np.float32) * c
    with tiny_lm.patched(steer.steer_patches(v, layers=[L], scale=1.0)):
        after, _ = tiny_lm.residuals(P)
    delta = (after[L + 1] - base[L + 1]).numpy()
    remainder = np.abs(delta - c).mean() / c
    assert abs(delta.mean() - c) / c < 0.05, (
        f"patch constant did not pass through the patched layer (mean delta {delta.mean():.4f} vs {c})"
    )
    assert remainder < 0.25, (
        f"block response ({remainder:.2%} of the constant) is large enough that pass-through is a "
        "poor control here; re-derive the arm before trusting any shift result"
    )


# --- h34: no-patch was never run; doing nothing scored as a perfect selector ------------------

def test_no_patch_gains_are_exactly_zero(tiny_lm):
    """With no patch, the teacher-forced log-prob gain of every candidate must be exactly 0.

    Under the h34 bug, 'doing nothing' ranked as a perfect selector because rank-1-on-ties awarded
    rank 1 to every untouched candidate. Any battery must carry this arm.
    """
    cands = ["a definite position", "measurement disturbs", "complementary descriptions"]
    prefix = "first, a particle has"
    base = [tiny_lm.logprob(prefix, c) for c in cands]
    again = [tiny_lm.logprob(prefix, c, patches=[]) for c in cands]
    gains = [b - a for a, b in zip(base, again)]
    assert all(g == 0.0 for g in gains), f"no-patch produced nonzero gains {gains}"


def test_tie_ranking_is_mid_rank_not_one(tiny_lm):
    """All-equal gains must rank at the midpoint, never at 1.

    This is the second half of h34: strict '>' counting gives rank 1 to a wholly tied field, which
    reads as a perfect result. Mid-rank gives chance, which reads as nothing happened.
    """
    def rank(scores, target):
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        ranks = [0.0] * len(scores)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks[target]
    n = 9
    assert rank([0.0] * n, 0) == (n + 1) / 2, "tied field did not rank at chance"
    assert rank([1.0] + [0.0] * (n - 1), 0) == 1.0


# --- h36: the patch reached one row of a padded batch, not the hidden states ------------------

def test_patch_reaches_every_sequence_in_a_batch(tiny_lm):
    """Every candidate's score must change under a patch, not just the first.

    The h36 signature is 8 of 9 candidates identical to base. Assert the count, not the effect.
    """
    texts = ["a particle has a definite position",
             "measurement disturbs what it measures",
             "position and momentum are complementary"]
    v = np.ones(tiny_lm.d_model, dtype=np.float32) * 0.5
    base = [tiny_lm.residuals(t)[0][2].mean().item() for t in texts]
    with tiny_lm.patched(steer.steer_patches(v, layers=[1], scale=1.0)):
        after = [tiny_lm.residuals(t)[0][2].mean().item() for t in texts]
    moved = sum(1 for a, b in zip(base, after) if abs(a - b) > 1e-6)
    assert moved == len(texts), f"patch moved {moved}/{len(texts)} sequences"


# --- h39: spans were read out of left-padding ------------------------------------------------

def test_spans_are_nonempty_and_inside_the_text(tiny_lm):
    """Every marked span must resolve to at least one real token of its own text."""
    a = extract(tiny_lm, P)
    for r in ROLES:
        assert len(a.tokens[r]) > 0, f"role {r} resolved to zero tokens"
    assert not np.allclose(a.roles[ROLES[0]], a.roles[ROLES[1]]), "distinct roles gave identical vectors"


def test_pooling_is_length_invariant_for_a_constant_span(tiny_lm):
    """Mean pooling of a span must not depend on tokens outside it.

    A padding or off-by-one error shows up here as a shift when the surrounding text changes while
    the span itself does not.
    """
    short = "x [[a: position and momentum are complementary descriptions]] y"
    long = "x y z w q r s t u v " * 3 + "[[a: position and momentum are complementary descriptions]] tail tail"
    va = extract(tiny_lm, short).roles["a"][0]
    vb = extract(tiny_lm, long).roles["a"][0]
    assert np.allclose(va, vb, atol=1e-4), "layer-0 span pooling changed with surrounding context"


# --- h32: a statistic sitting at its own noise floor -----------------------------------------

def test_statistic_returns_its_null_on_noise_and_moves_on_signal(tiny_lm):
    """Calibration in miniature: a statistic must be flat on noise AND respond to planted signal.

    The broken cross-talk rank passed the first half and failed the second: it returned its chance
    value on everything. Noise alone is not a calibration.
    """
    rng = np.random.default_rng(0)
    d, n = 64, 40
    noise = rng.normal(size=(n, d))
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)

    def readout(X):
        return float(np.mean(X @ direction))

    assert abs(readout(noise)) < 0.5, "readout not ~null on noise"
    for amp in (1.0, 2.0, 4.0):
        planted = noise + amp * direction
        assert readout(planted) > amp * 0.5, f"readout did not respond to planted signal at {amp}"
    vals = [readout(noise + a * direction) for a in (1.0, 2.0, 4.0)]
    assert vals[0] < vals[1] < vals[2], "readout not monotone in planted effect size"


# --- phase 0.1 review finding: the last hidden state is post-final-norm --------------------

def test_last_hidden_state_is_post_norm(tiny_lm):
    """`residuals()[-1]` is the final-norm OUTPUT, not the last block's residual.

    Found while reviewing the direct-path spec: an offline arm written as `norm(hs[-1] + v)` would
    double-norm, and any 'layer N' number read from `hs[-1]` is a normed vector, not a residual.
    Assert the relationship so the convention cannot be silently misremembered again.
    """
    pre = {}
    h = tiny_lm.model.model.norm.register_forward_pre_hook(
        lambda mod, args: pre.setdefault("x", args[0].detach().clone()))
    res, _ = tiny_lm.residuals(P)
    h.remove()
    assert "x" in pre, "final norm never ran; capture point is wrong"
    post = tiny_lm.model.model.norm(pre["x"])[0]
    assert torch.allclose(res[-1], post, atol=1e-4), "last hidden state is not the final-norm output"
    assert not torch.allclose(res[-1], pre["x"][0], atol=1e-4), (
        "last hidden state equals the pre-norm residual; the convention changed, update the docstring"
    )


# --- phase 0.1: the final-residual patch point and the offline direct-path arm ----------------

def _final_norm_and_head(lm):
    return lm.final_norm(), lm.head()


def test_final_layer_zero_patch_is_identity(tiny_lm):
    """A zero vector added at `pre_28` must leave the log-probs exactly unchanged."""
    prefix, cont = "first, a particle has", " a definite position"
    zero = np.zeros(tiny_lm.d_model, dtype=np.float32)
    base = tiny_lm.logprob(prefix, cont)
    got = tiny_lm.logprob(prefix, cont, [Patch(tiny_lm.n_layers, steer.add_vector(zero, 1.0))])
    assert got == base, f"zero final patch moved the log-prob by {got - base}"


def test_final_layer_patch_is_offline_unembed(tiny_lm):
    """`Patch(n_layers, +v)` must equal the offline arithmetic on the captured pre-norm residual.

    This identity is what makes every final-residual arm computable without a forward pass. If it
    fails the arm is wrong, not the claim (docs/specs/selector_direct_path_v1.md §2). Note the
    offline form norms `pre + v` ONCE: using `residuals()[-1]` here would double-norm.
    """
    prefix, cont = "first, a particle has", " a definite position"
    rng = np.random.default_rng(0)
    v = rng.normal(size=tiny_lm.d_model).astype(np.float32)
    v /= np.linalg.norm(v)
    v *= 2.0

    enc_p, _ = tiny_lm.encode(prefix)
    enc_f, _ = tiny_lm.encode(prefix + cont)
    n_p = enc_p["input_ids"].shape[1]
    ids = enc_f["input_ids"][0]

    pre, r, logits = tiny_lm.pre_norm_residual(prefix + cont)
    norm, head = _final_norm_and_head(tiny_lm)
    with torch.no_grad():
        z = norm(pre + torch.as_tensor(v))
        lp = torch.log_softmax(head(z)[:-1].float(), dim=-1)
    tgt = ids[1:]
    offline = float(lp[torch.arange(n_p - 1, len(tgt)), tgt[n_p - 1:]].sum())
    online = tiny_lm.logprob(prefix, cont, [Patch(tiny_lm.n_layers, steer.add_vector(v, 1.0))])
    assert abs(online - offline) < 1e-4, f"offline final-residual arm off by {online - offline:.2e} nats"

    with torch.no_grad():
        lp0 = torch.log_softmax(head(norm(pre))[:-1].float(), dim=-1)
    off0 = float(lp0[torch.arange(n_p - 1, len(tgt)), tgt[n_p - 1:]].sum())
    assert abs(off0 - tiny_lm.logprob(prefix, cont)) < 1e-4, "base offline arm does not reproduce base"


def test_pre_norm_capture_differs_from_hidden_states_last(tiny_lm):
    """`pre_norm_residual` returns the residual, not the normed hidden state."""
    pre, r, _ = tiny_lm.pre_norm_residual(P)
    hs, _ = tiny_lm.residuals(P)
    norm, _ = _final_norm_and_head(tiny_lm)
    with torch.no_grad():
        assert torch.allclose(norm(pre), hs[-1], atol=1e-4), "norm(pre_28) != hidden_states[-1]"
    assert not torch.allclose(pre, hs[-1], atol=1e-4), "pre_28 == hidden_states[-1]; capture is post-norm"
    assert abs(float(r[-1]) - float(pre[1:].norm(dim=-1).mean())) < 1e-4, "r[-1] is not the pre-norm norm"


def test_final_layer_patch_sees_upstream_patch(tiny_lm):
    """A patch at block L is visible in the captured `pre_28` (the norm hook fires after it)."""
    v = np.ones(tiny_lm.d_model, dtype=np.float32) * 0.3
    pre0, _, _ = tiny_lm.pre_norm_residual(P)
    pre1, _, _ = tiny_lm.pre_norm_residual(P, [Patch(1, steer.add_vector(v, 1.0))])
    assert not torch.allclose(pre0, pre1), "upstream patch not visible in pre-norm capture"
