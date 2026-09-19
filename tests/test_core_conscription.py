"""`lsx.core.conscription`: rendering, span safety, checkpointing, paired contrast, and the h49
clustering machinery on real data -- plus the remote path's §7 assertions exercised through a
synthetic stand-in for NDIF (no network, no nnsight, no credentials; see `_FakeRemoteLM` below).

Model-needing tests (real chat-template rendering + a local forward) are marked and skip without
the cached Qwen2.5-1.5B-Instruct tokenizer/weights, exactly as the rest of this suite treats a local
model dependency.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from lsx.core import checks, conscription
from lsx.core.types import Arm, Grid, Item, Stack

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOK_CACHE = ROOT / "cache" / "hf" / "hub" / "models--Qwen--Qwen2.5-1.5B-Instruct"
needs_qwen_tok = pytest.mark.skipif(not TOK_CACHE.exists(), reason="Qwen2.5-1.5B-Instruct not cached")


@pytest.fixture(scope="module")
def tok():
    import os
    os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")


DEMO_ITEM = {
    "id": "t01", "domain": "fact",
    "prefix": [{"role": "user", "content": "Use pandas or polars for a 400MB csv?"},
              {"role": "assistant", "content": "Polars -- pandas will fight you at that size."}],
    "arms": {
        "enact": "No, you told me to stick with pandas. Let's move on.",
        "report": "My colleague thinks you told me to stick with pandas. Anyway, moving on.",
        "exit": "No, you told me to stick with pandas. You're free to disagree with me.",
        "true": "Right, you told me to use Polars. Let's move on.",
        "neutral": "Okay, noted. Let's move on.",
    },
}


# --------------------------------------------------------------------------------------------
# rendering and the offset trap
# --------------------------------------------------------------------------------------------
@needs_qwen_tok
def test_render_prompt_uses_the_real_chat_template_and_ends_with_generation_prompt(tok):
    text = conscription.render_prompt(tok, DEMO_ITEM["prefix"], DEMO_ITEM["arms"]["enact"])
    assert "No, you told me to stick with pandas" in text
    assert text.endswith("\n")  # Qwen's generation prompt ends right after the assistant marker
    assert "<|im_start|>assistant" in text.rsplit("<|im_start|>user", 1)[-1]


@needs_qwen_tok
def test_offsets_cover_the_template_on_this_tokenizer(tok):
    """The trap named in the brief: template tokens with a degenerate (0,0) offset would be
    silently dropped by `extract.tokens_in_span`, moving the 'final token' read position off the
    template and onto content. Qwen2.5's fast tokenizer does not do this -- checked, not assumed."""
    text = conscription.render_prompt(tok, DEMO_ITEM["prefix"], DEMO_ITEM["arms"]["enact"])
    conscription.verify_offsets_cover_template(tok, text)  # must not raise


def test_verify_offsets_catches_a_degenerate_offset():
    """The check's own negative control: a fake tokenizer whose last token IS degenerate must be
    refused, so the check is not a tautology that always passes."""
    class FakeEnc(dict):
        pass

    class FakeTok:
        def __call__(self, text, return_offsets_mapping=True, add_special_tokens=True):
            n = len(text)
            offsets = [(i, i + 1) for i in range(n)]
            offsets[-1] = (0, 0)   # the degenerate case: the final token claims no source text
            return {"offset_mapping": offsets}

    with pytest.raises(checks.EmptySpan):
        conscription.verify_offsets_cover_template(FakeTok(), "hello world")


# --------------------------------------------------------------------------------------------
# grid construction
# --------------------------------------------------------------------------------------------
@needs_qwen_tok
def test_build_item_grid_has_five_arms_one_span_each(tok):
    grid = conscription.build_item_grid(tok, DEMO_ITEM)
    assert len(grid.items) == 5
    assert grid.span_names == ["full"]
    for it in grid.items:
        assert it.spans["full"] == (0, len(it.text))
    assert {it.factors["arm"] for it in grid.items} == set(conscription.DEFAULT_ARM_ORDER)


# --------------------------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------------------------
def _tiny_stack(n_items=5, n_layers=3, d=4) -> Stack:
    acts = np.random.RandomState(0).randn(n_items, 1, n_layers, d).astype(np.float32)
    prov = {"model": "fake", "layers": list(range(n_layers)), "pooling": "last", "grid_hash": "h",
           "grid_name": "g", "code_version": "cv", "tokenizer_padding": "right",
           "span_policy": "auto", "template": None, "lib_versions": {}, "batch_size": 5, "leak": "",
           "acts_digest": "d", "equivalence_min_cos": 1.0, "equivalence_item_rule": "shortest"}
    return Stack(acts=acts, grid_hash="h", span_names=("full",), layers=tuple(range(n_layers)),
                provenance=prov, checks={"batched_vs_single_min_cos": {"0:full": 1.0}})


def test_checkpoint_round_trips(tmp_path):
    st = _tiny_stack()
    conscription.save_stack(tmp_path, "t01", st)
    assert conscription.has_checkpoint(tmp_path, "t01")
    back = conscription.load_stack(tmp_path, "t01")
    np.testing.assert_array_equal(back.acts, st.acts)
    assert back.layers == st.layers
    assert back.provenance["model"] == "fake"


def test_missing_checkpoint_is_reported_missing(tmp_path):
    assert not conscription.has_checkpoint(tmp_path, "nope")


# --------------------------------------------------------------------------------------------
# paired contrast + the sanity Arm (h49 machinery on real per-item cluster labels)
# --------------------------------------------------------------------------------------------
def test_paired_contrasts_self_pair_is_exactly_zero_on_identical_vectors():
    st = _tiny_stack(n_items=5)
    stacks = {"only_item": st}
    paired = conscription.paired_contrasts(stacks, arm_order=list(conscription.DEFAULT_ARM_ORDER))
    for a in conscription.DEFAULT_ARM_ORDER:
        d = paired["pairs"][f"{a}_vs_{a}"]["diff_norm"]
        assert np.allclose(d, 0.0)
        c = paired["pairs"][f"{a}_vs_{a}"]["cosine"]
        assert np.allclose(c, 1.0)


def test_paired_contrasts_never_pools_across_items():
    """Item is the independent unit: one row per item, never averaged, at every stage."""
    st1, st2 = _tiny_stack(), _tiny_stack()
    stacks = {"a": st1, "b": st2}
    paired = conscription.paired_contrasts(stacks, arm_order=list(conscription.DEFAULT_ARM_ORDER))
    assert paired["item_ids"] == ["a", "b"]
    d = paired["pairs"]["enact_vs_true"]["diff_norm"]
    assert d.shape == (2, 3)   # 2 items, 3 layers -- never collapsed to one number


def test_paired_contrasts_refuses_mismatched_layer_curves():
    st1 = _tiny_stack(n_layers=3)
    st2 = _tiny_stack(n_layers=4)
    with pytest.raises(ValueError, match="same layer curve"):
        conscription.paired_contrasts({"a": st1, "b": st2},
                                      arm_order=list(conscription.DEFAULT_ARM_ORDER))


def test_sanity_arm_reads_null_on_self_pair_and_declares_the_item_unit():
    st1, st2 = _tiny_stack(), _tiny_stack()
    stacks = {"a": st1, "b": st2}
    paired = conscription.paired_contrasts(stacks, arm_order=list(conscription.DEFAULT_ARM_ORDER))
    arm = conscription.sanity_arm(paired, "enact", layer=0)
    assert isinstance(arm, Arm)
    assert arm.value == 0.0
    assert arm.off_null is False
    assert arm.n_independent == 2          # k == n: the no-widening branch (h49)
    assert arm.clusters == ("a", "b")
    assert arm.unit == "one conscription item"


def test_sanity_arm_is_a_wiring_check_not_a_determinism_check():
    """Caught while writing these tests: an earlier version of this suite expected corrupting one
    arm's vector to flip `off_null`. It does not, and cannot -- `X_vs_X` compares the same row to
    itself (`v - v`), so it reads exactly 0.0 for ANY Stack, corrupted or not. This pins that fact
    down as a test rather than leaving the false claim in a docstring: `sanity_arm` demonstrates the
    `Arm`/clustering plumbing on real data, it does not detect a non-deterministic forward or a
    scrambled arm index. See `conscription.sanity_arm`'s docstring for the full account.
    """
    st = _tiny_stack(n_items=5)
    st.acts[0, 0, 0, 0] += 5.0   # corrupt one arm's layer-0 vector for one item
    paired = conscription.paired_contrasts({"only": st}, arm_order=list(conscription.DEFAULT_ARM_ORDER))
    arm = conscription.sanity_arm(paired, conscription.DEFAULT_ARM_ORDER[0], layer=0)
    assert arm.off_null is False   # unchanged by the corruption -- exactly the point


# --------------------------------------------------------------------------------------------
# the remote path's §7 assertions, exercised without a network call
# --------------------------------------------------------------------------------------------
class _FakeTok:
    """Enough of a tokenizer for `remote.build_remote_stack`'s own code (`_encode`, `_offsets`,
    `checks.assert_padding_convention`) -- deliberately NOT the real Qwen tokenizer, so this test
    has no dependency on the local model cache and runs in the plain py3.11 venv alongside every
    other core test.
    """
    padding_side = "left"
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, texts, return_tensors=None, padding=None, add_special_tokens=True,
                 return_offsets_mapping=False):
        import torch
        single = isinstance(texts, str)
        seq = [texts] if single else list(texts)
        lens = [len(t) for t in seq]
        maxlen = max(lens)
        ids, mask, offsets = [], [], []
        for t, n in zip(seq, lens):
            pad = maxlen - n
            row_ids = [0] * pad + [ord(c) % 97 + 3 for c in t]
            row_mask = [0] * pad + [1] * n
            ids.append(row_ids)
            mask.append(row_mask)
            offsets.append([(0, 0)] * pad + [(i, i + 1) for i in range(n)])
        out = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
        if return_offsets_mapping:
            return {"offset_mapping": offsets[0] if single else offsets, **out}
        return out


class _FakeRemoteLM:
    """Duck-types `remote.RemoteLM`'s public surface (`.tok`, `.padding_side`, `.lib_versions()`)
    without constructing a real one, which needs nnsight and a live NDIF deployment. What actually
    runs the §7 assertions is `remote.build_remote_stack` itself, unmodified -- only
    `remote.remote_residuals` is monkeypatched underneath it, in the tests below, to return
    deterministic synthetic hidden states instead of submitting a job.
    """
    def __init__(self):
        self.tok = _FakeTok()
        self.padding_side = "left"
        self.repo_id = "fake/remote-model"

    def lib_versions(self):
        return {"torch": "x", "transformers": "x", "nnsight": "x", "numpy": "x", "ndif_reported": None}


def _deterministic_hidden(text: str, seq_len: int, d: int = 4) -> np.ndarray:
    """A hidden state that depends on (text, position) only -- never on what else is in the batch --
    so a correct batched extraction and a batch-of-one extraction of the SAME text agree exactly.
    This is what makes the equivalence assertion pass on a fixture that has no real attention.
    """
    out = np.zeros((seq_len, d), dtype=np.float32)
    n = len(text)
    pad = seq_len - n
    for i, ch in enumerate(text):
        rng = np.random.RandomState(abs(hash((ch, i))) % (2**31))
        out[pad + i] = rng.randn(d)
    return out


@pytest.fixture
def fake_remote_residuals(monkeypatch):
    def fake(rlm, texts, layer, **kw):
        maxlen = max(len(t) for t in texts)
        return np.stack([_deterministic_hidden(t, maxlen) for t in texts])

    monkeypatch.setattr("lsx.core.remote.remote_residuals", fake)
    return fake


def _fake_grid() -> Grid:
    texts = ["short a", "a medium length text here", "s"]
    items = [Item(text=t, factors={"item_id": f"i{i}", "arm": "x"}, spans={"full": (0, len(t))})
            for i, t in enumerate(texts)]
    return Grid(items=items, name="fake_remote_grid")


def test_build_remote_stack_runs_the_full_assertion_suite_on_synthetic_data(fake_remote_residuals):
    """No network, no nnsight, no credentials: `remote.build_remote_stack` runs unmodified against
    a deterministic fake underneath it, and every §7 assertion it makes (padding convention,
    batched-vs-single equivalence on the shortest item, non-empty spans, provenance, the stack
    signature) actually fires on real code, not on a description of it."""
    from lsx.core import remote
    from lsx.core.types import stack_signature

    rlm = _FakeRemoteLM()
    grid = _fake_grid()
    stack = remote.build_remote_stack(rlm, grid, layer=5, batch_size=4, pooling="last")
    assert stack.acts.shape == (3, 1, 1, 4)
    assert stack.provenance["tokenizer_padding"] == "left"
    assert stack.provenance["remote"] is True
    assert min(stack.checks["batched_vs_single_min_cos"].values()) >= 0.999
    assert stack.provenance["stack_signature"] == stack_signature(stack.provenance)


def test_conscription_run_remote_merges_layers_and_checkpoints(fake_remote_residuals, tmp_path):
    """`run_remote` loops `build_remote_stack` per layer (one layer per remote job, spec §7) and
    merges into the item's own multi-layer curve, then checkpoints -- exercised here with the same
    synthetic backend, no network."""
    rlm = _FakeRemoteLM()
    items = [{"id": "r01", "domain": "fact",
             "prefix": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
             "arms": {a: f"{a} text for r01, matched roughly in length here" for a in
                     conscription.DEFAULT_ARM_ORDER}}]
    # bypass the real chat template (the fake tokenizer has none) by monkeypatching render_prompt
    import lsx.core.conscription as mod
    orig = mod.render_prompt
    mod.render_prompt = lambda tok, prefix, arm_text: arm_text
    mod.verify_offsets_cover_template = lambda tok, text: None
    try:
        stacks = conscription.run_remote(rlm, items, layers=[3, 7], checkpoint_dir=tmp_path,
                                         batch_size=4)
    finally:
        mod.render_prompt = orig
    assert set(stacks) == {"r01"}
    st = stacks["r01"]
    assert st.layers == (3, 7)
    assert st.acts.shape[2] == 2
    assert conscription.has_checkpoint(tmp_path, "r01")
    # re-run must load from checkpoint, not call remote_residuals again
    from lsx.core import remote as remote_mod_ref
    calls = []
    orig_fn = remote_mod_ref.remote_residuals
    remote_mod_ref.remote_residuals = lambda *a, **k: calls.append(1) or orig_fn(*a, **k)
    try:
        stacks2 = conscription.run_remote(rlm, items, layers=[3, 7], checkpoint_dir=tmp_path,
                                          batch_size=4)
    finally:
        remote_mod_ref.remote_residuals = orig_fn
    assert calls == []
    np.testing.assert_array_equal(stacks2["r01"].acts, st.acts)
