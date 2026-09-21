"""The `add_special_tokens` flag on the Tier B extractor, which is not cosmetic.

Gemma-2's chat template emits `<bos>` itself. Tokenising that text with the default
`add_special_tokens=True` prepends a SECOND one, shifting every position by one and putting a
duplicated BOS inside the masked mean. It is invisible: the capture succeeds, the shapes are
right, every §7 assertion passes, and the numbers are quietly taken on a different input.

These tests use a stub tokenizer so they pin the plumbing without a model or NDIF.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pr = pytest.importorskip("lsx.core.painaxis_remote")


class StubTok:
    """Mimics the one behaviour under test: a leading BOS added iff add_special_tokens."""
    bos_token_id = 2

    def __init__(self):
        self.calls = []

    def __call__(self, texts, return_tensors=None, padding=None, add_special_tokens=True):
        import torch
        self.calls.append(add_special_tokens)
        ids = []
        for t in texts:
            body = [100 + i for i in range(len(t.split()))]
            ids.append(([self.bos_token_id] if add_special_tokens else []) + body)
        width = max(len(x) for x in ids)
        # left padding, the convention this module indexes against
        padded = [[0] * (width - len(x)) + x for x in ids]
        mask = [[0] * (width - len(x)) + [1] * len(x) for x in ids]
        return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(mask)}


class StubRLM:
    def __init__(self):
        self.tok = StubTok()


def test_encode_defaults_to_adding_specials():
    rlm = StubRLM()
    ids, mask = pr._encode(rlm, ["a b c"])
    assert rlm.tok.calls == [True]
    assert int(ids[0][0]) == StubTok.bos_token_id
    assert ids.shape[1] == 4


def test_encode_can_be_told_not_to():
    rlm = StubRLM()
    ids, mask = pr._encode(rlm, ["a b c"], add_special_tokens=False)
    assert rlm.tok.calls == [False]
    assert ids.shape[1] == 3
    assert int(ids[0][0]) != StubTok.bos_token_id


def test_the_two_paths_differ_by_exactly_one_leading_token():
    """This is the whole hazard, stated as a number: the same string tokenises to two
    different sequences, and nothing downstream would notice."""
    rlm = StubRLM()
    with_specials, _ = pr._encode(rlm, ["a b c d"], add_special_tokens=True)
    without, _ = pr._encode(rlm, ["a b c d"], add_special_tokens=False)
    assert with_specials.shape[1] - without.shape[1] == 1
    assert list(with_specials[0][1:]) == list(without[0])


def test_mask_is_left_padded_so_final_token_is_real():
    rlm = StubRLM()
    ids, mask = pr._encode(rlm, ["a", "a b c d e"], add_special_tokens=False)
    for row in mask:
        assert int(row[-1]) == 1, "final position is padding; end-relative indexing would be wrong"


def test_cross_check_refuses_chat_rendered_text():
    """The offline path re-tokenises with specials on. Comparing it against a capture taken
    without them compares two different inputs, so the check is refused rather than run."""
    with pytest.raises(ValueError, match="chat-rendered"):
        pr.cross_check_against_asserted_path(
            StubRLM(), ["x"], 0, {"mean": None, "final_token": None},
            add_special_tokens=False)


def test_extract_pooled_threads_the_flag_to_every_encode(monkeypatch):
    """Three call sites encode: the job, the span check, and the equivalence single. If any one
    of them keeps the default, the assertions are computed against a different tokenisation
    than the capture."""
    import numpy as np

    rlm = StubRLM()
    seen = []

    def fake_job(r, texts, *, add_special_tokens=True):
        seen.append(("job", add_special_tokens, len(texts)))
        n, d = len(texts), 4
        return {"final_token": np.ones((n, 2, d), dtype=np.float32),
                "mean": np.ones((n, 2, d), dtype=np.float32),
                "embed_final_token": np.ones((n, d), dtype=np.float32),
                "embed_mean": np.ones((n, d), dtype=np.float32)}

    monkeypatch.setattr(pr, "_pooled_job", fake_job)
    monkeypatch.setattr(pr.checks, "assert_padding_convention", lambda *a, **k: None)
    monkeypatch.setattr(pr.checks, "assert_nonempty_spans", lambda *a, **k: None)
    monkeypatch.setattr(pr.checks, "assert_batch_equivalence", lambda *a, **k: 1.0)

    class Cfg:
        hidden_size = 4

    class Model:
        config = Cfg()

    rlm.model = Model()
    rlm.blocks = [object(), object()]
    rlm.padding_side = "left"

    pr.extract_pooled(rlm, ["a b", "a b c"], batch_size=2, add_special_tokens=False,
                      check_every=1, verbose=False)
    assert all(flag is False for _, flag, _ in seen), seen
    assert all(c is False for c in rlm.tok.calls), rlm.tok.calls
    assert ("job", False, 1) in seen, "the equivalence single did not run with the flag"
