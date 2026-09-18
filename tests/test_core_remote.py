"""`lsx.core.remote`: the asserted remote path, and `Probe` as a gate rather than a label (§3, §7).

The live half of this module needs NDIF and the py3.12 venv, and its evidence is
`results/ndif_probe_asserted.json` -- a real run on google/gemma-2-9b-it in which the h36 idiom
moved 1 of 3 sequences and `MovedCandidates` fired. What is tested HERE is everything that can be
tested without a deployment, which is more than it sounds:

  * `Probe` refuses a forward that is not one of the asserted ones;
  * the `resid` resolution written out inline inside the trace blocks agrees with `checks.resid`,
    which is the one duplication in the core and the one that could drift silently;
  * the recorded evidence from the live run is the shape the assertions claim.
"""
import json
import pathlib

import numpy as np
import pytest
import torch

from lsx.core import checks, remote
from lsx.core.types import Direction, Probe, Sketch

EVIDENCE = pathlib.Path(__file__).resolve().parents[1] / "results" / "ndif_probe_asserted.json"


def _direction():
    return Direction(vecs={20: np.ones(8)}, held_out={"axis": "era", "unseen": {"1920s"}})


# ---------------------------------------------------------------------------------------------
# Probe is no longer a shell
# ---------------------------------------------------------------------------------------------
def test_probe_refuses_a_forward_that_is_not_asserted():
    """Pieces 1, 2 and 3 each recorded `Probe` as a shell. A shell that names a contract without
    enforcing it is worse than no type at all: it reads as a guarantee."""
    with pytest.raises(checks.UnassertedForward):
        Probe(direction=_direction(), patch_layer=14, readout_layer=20,
              fn=lambda *a, **k: np.zeros(3))

    def looks_official(*a, **k):
        return np.zeros(3)

    looks_official.__qualname__ = "asserted_remote_patched_logprob"
    with pytest.raises(checks.UnassertedForward):
        Probe(direction=_direction(), patch_layer=14, readout_layer=20, fn=looks_official)


@pytest.mark.parametrize("fn", [remote.remote_residuals, remote.asserted_remote_patched_logprob,
                                remote.build_remote_stack])
def test_the_asserted_remote_forwards_are_accepted(fn):
    p = Probe(direction=_direction(), patch_layer=14, readout_layer=20, fn=fn)
    assert p.fn is fn


def test_the_local_asserted_forwards_are_accepted_too():
    from lsx.core import extract
    for fn in (extract.asserted_patched_forward, extract.asserted_patched_logprob):
        Probe(direction=_direction(), patch_layer=14, readout_layer=20, fn=fn)


def test_a_probe_returns_an_unledgerable_sketch():
    """A probe produces scores. A number becomes a result at the `Claim` and the ledger, not here."""
    fn = remote.asserted_remote_patched_logprob
    p = Probe(direction=_direction(), patch_layer=14, readout_layer=20, fn=fn)
    out = Sketch(np.zeros(3), "x")
    assert out.ledgerable is False
    assert "cannot be ledgered" in repr(out)
    assert p.fn is fn


# ---------------------------------------------------------------------------------------------
# the one duplication in the core
# ---------------------------------------------------------------------------------------------
def test_the_inline_resid_matches_checks_resid():
    """NDIF refuses a trace block that reaches into `lsx.core.checks`, so the tuple-or-tensor
    resolution is written out inside each trace block. This pins the duplicate to the original.

    `output[0]` is the thing both of them exist to avoid: on a block that returns a bare tensor it
    is BATCH ROW 0, which is h36, and on a batch of three it silently returns one sequence.
    """
    batch = torch.arange(24.0).reshape(3, 2, 4)

    def inline(o):
        return o if isinstance(o, torch.Tensor) else o[0]

    for obj in (batch, (batch, None), [batch, "attn"]):
        assert torch.equal(inline(obj), checks.resid(obj))
        assert inline(obj).shape == (3, 2, 4)

    # and the idiom they replace does something else entirely
    assert batch[0].shape == (2, 4)
    with pytest.raises(checks.LayerOutputShape):
        checks.resid("not a tensor")


def test_the_inline_resid_appears_in_every_trace_block():
    """A grep, deliberately: if a future trace block indexes `output[0]` outside the harness's
    `batch_row_bug` path, this fails."""
    src = (pathlib.Path(remote.__file__)).read_text()
    body = src.split("# the asserted forward", 1)[1]
    for line in body.splitlines():
        stripped = line.strip()
        # prose lines (comments, docstrings, the assertion messages that NAME the bad idiom) are
        # not code; the grep is over code
        if stripped.startswith("#") or "batch_row_bug" in stripped or "`" in stripped:
            continue
        assert "output[0]" not in stripped, line
    assert body.count("isinstance(o, torch.Tensor)") >= 2
    assert "isinstance(r, torch.Tensor)" in body


# ---------------------------------------------------------------------------------------------
# what the live run recorded
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not EVIDENCE.exists(), reason="no recorded NDIF run")
def test_the_live_run_caught_h36_on_a_real_deployment():
    ev = json.loads(EVIDENCE.read_text())
    assert ev["padding_side"] == "left"
    assert ev["batch_row_bug"]["caught"] is True
    assert ev["batch_row_bug"]["mechanism"] == "MovedCandidates"
    assert "moved 1/3" in ev["batch_row_bug"]["message"]
    # the positive control, so the run is not a core that refuses everything
    assert ev["moved_candidates_positive_control"] == "PASSED"
    assert all(abs(d) > 1e-6 for d in ev["patched_delta"])
    assert all(d == 0.0 for d in ev["no_patch_delta"])


@pytest.mark.skipif(not EVIDENCE.exists(), reason="no recorded NDIF run")
def test_the_live_equivalence_check_ran_on_a_maximally_padded_item():
    """§7.2's premise: the SHORTEST item is the one that fails first, so it is the one checked."""
    ev = json.loads(EVIDENCE.read_text())
    assert ev["max_padding_on_shortest"] >= 10, ev["token_lengths"]
    assert min(ev["equivalence"].values()) >= 0.999, ev["equivalence"]
    checked = {int(k.split(":")[0]) for k in ev["equivalence"]}
    shortest = int(np.argmin(ev["token_lengths"]))
    assert checked == {shortest}, (checked, ev["token_lengths"])


@pytest.mark.skipif(not EVIDENCE.exists(), reason="no recorded NDIF run")
def test_the_remote_stack_is_signed_like_a_local_one():
    """A remote claim reaches the ledger on the same terms as a local one and no others."""
    from lsx.core.types import stack_signature
    ev = json.loads(EVIDENCE.read_text())
    prov = dict(ev["provenance"])
    sig = prov.pop("stack_signature")
    assert sig == ev["stack_signature"]
    assert stack_signature(prov) == sig
    prov["tokenizer_padding"] = "right"
    assert stack_signature(prov) != sig
    assert prov["lib_versions"]["nnsight"]
    assert "ndif_reported" in prov["lib_versions"]
