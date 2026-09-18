"""§7 on the remote path: every NDIF forward, `Probe` included, through one asserted function.

Pieces 1, 2 and 3 each closed with the same sentence in their handover notes: *`Probe` is a shell;
nothing NDIF-side goes through the asserted path.* Spec §7 is explicit that this is where it
matters most -- "the stage-36 bug lived in a patched forward, not in an extraction, so assertions
scoped to `Stack` alone would have missed it" -- and h34, h36, h37 and h39 are all remote. Until
this module existed, the moved-candidates assertion that caught h34 guarded no NDIF call at all.

What is asserted here is what `extract.build_stack` asserts locally, on the same code:

  1. **padding side** read from the remote tokenizer and compared to the indexing convention
     (`checks.assert_padding_convention`). nnsight's `LanguageModel` loads Gemma's tokenizer with
     `padding_side='left'`, which is the h39 trap.
  2. **batched-vs-single equivalence on the SHORTEST item** of each batch
     (`checks.assert_batch_equivalence`), never a random sample -- the shortest item is the
     maximally padded one.
  3. **moved-candidates on every patched forward** (`checks.assert_moved_candidates`): the number
     of sequences whose score changed equals the batch size. This is h34/h36.
  4. **non-empty spans**, never resolving into padding.
  5. **the block output resolved by `checks.resid`**, by type and never by index. `output[0]` is
     batch row 0 whenever a decoder block returns a bare tensor, which Gemma-2 on today's NDIF
     deployment does -- measured, not assumed, in `results/ndif_probe_asserted.json`.
  6. **provenance** carrying the NDIF-reported library versions beside the local ones, and signed
     by `types.stack_signature` exactly as a local stack is, so a remote claim can reach the ledger
     on the same terms as a local one and no others.

nnsight is imported inside the functions: this module must be importable from the py3.11 venv that
runs the test suite, where there is no nnsight, so that `lsx.core` stays one package rather than
two. The tests that need a live deployment are marked and skipped there.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from . import checks
from .types import Grid, Stack, stack_signature


def asserted(fn: Callable) -> Callable:
    """Mark a function as one that runs the §7 assertions. `types.Probe` refuses anything else."""
    fn._lsx_asserted = True
    return fn


# --------------------------------------------------------------------------------------------
# the connection
# --------------------------------------------------------------------------------------------
@dataclass
class RemoteLM:
    """An NDIF-hosted model, with the fields the assertions need read from it rather than assumed.

    Deliberately thin. Spec §10 lists "a rewrite of the NDIF layer" as a non-goal -- it works, and
    its bugs were in the callers -- so this wraps `lsx.ndif`'s proxy backend and adds the
    assertions, rather than replacing anything.
    """
    repo_id: str
    padding_side: str = "left"
    _model: object = field(default=None, repr=False)
    _versions: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        from nnsight import LanguageModel

        self._model = LanguageModel(self.repo_id, device_map="auto", dispatch=False)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = self.padding_side
        # what the tokenizer ACTUALLY does, read back after setting it: the h39 bug was a script
        # believing one thing while nnsight had done another.
        self.padding_side = self.tok.padding_side

    @property
    def model(self):
        return self._model

    @property
    def tok(self):
        return self._model.tokenizer

    @property
    def blocks(self):
        m = self._model
        for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
            obj = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise AttributeError(f"no decoder block list found on {self.repo_id}")

    def lib_versions(self) -> dict:
        """Local versions, plus whatever the NDIF status endpoint reports for the deployment.

        §7.6 asks for "transformers and nnsight versions both locally and as reported by the NDIF
        server". The server half is best-effort and recorded as `None` when the endpoint does not
        say; a recorded None is a different thing from a missing key, and `assert_provenance`
        treats it that way.
        """
        if self._versions:
            return dict(self._versions)
        import nnsight
        import torch
        import transformers
        out = {"torch": torch.__version__, "transformers": transformers.__version__,
               "nnsight": nnsight.__version__, "numpy": np.__version__,
               "ndif_reported": None}
        try:
            import httpx
            r = httpx.get("https://api.ndif.us/status", timeout=10.0)
            if r.status_code == 200:
                body = r.json()
                rows = body if isinstance(body, list) else list(body.values())
                for row in rows:
                    if isinstance(row, dict) and row.get("repo_id") == self.repo_id:
                        out["ndif_reported"] = {k: row.get(k) for k in
                                                ("repo_id", "deployment_level", "pinned",
                                                 "nnsight_version", "status")
                                                if k in row}
                        break
        except Exception:  # noqa: BLE001 -- provenance must never fail a measurement
            pass
        self._versions = out
        return dict(out)

    # ---- the one remote call ----------------------------------------------------------
    def _run(self, build: Callable, key: str = "out") -> np.ndarray:
        """Submit one traced job through the credential-injecting proxy and wait for it.

        `build(backend)` opens the trace and returns the TRACER -- never a value computed inside
        the block. nnsight ships the block's source to the deployment and does not run its body
        locally, so a name bound inside the `with` is unbound when the `with` returns; the saved
        tensors come back through `backend.wait(tracer)` keyed by the name they were saved under.
        Getting this wrong fails loudly (`UnboundLocalError`), which is the good case, and it is
        the only reason this indirection exists.

        Everything else here is the retry the environment needs. No other function in this module
        talks to NDIF.
        """
        import torch
        from lsx.ndif import ProxyAuthBackend, retry_job

        def once():
            backend = ProxyAuthBackend(self._model.to_model_key())
            tracer = build(backend)
            res = backend.wait(tracer)
            v = res
            if isinstance(res, dict):
                v = res.get(key)
                if v is None:
                    v = next(x for x in res.values() if isinstance(x, torch.Tensor))
            return v.float().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

        return retry_job(once)


# --------------------------------------------------------------------------------------------
# `checks.resid`, inline.
#
# The trace block's source is shipped to the deployment and executed there, and NDIF refuses a
# block that reaches into a non-whitelisted module: calling `checks.resid(...)` inside a trace
# fails with "Module lsx.core.checks is not whitelisted". Measured, not guessed -- it is what the
# first run of `results/ndif_probe_asserted.json` did.
#
# So the tuple-or-tensor resolution is written out inside each trace block. It is two lines and it
# is `checks.resid`'s body verbatim, and `tests/test_core_remote.py::
# test_the_inline_resid_matches_checks_resid` asserts the two agree on a tensor, on a tuple and on
# a batch, so the duplicate cannot drift. What must NOT happen is the thing the duplication is
# there to avoid: `output[0]`, which is batch row 0 on any block that returns a bare tensor, and
# which is h36.
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# the asserted forward
# --------------------------------------------------------------------------------------------
def _encode(rlm: RemoteLM, texts: Sequence[str]):
    enc = rlm.tok(list(texts), return_tensors="pt", padding=True, add_special_tokens=True)
    return enc["input_ids"], enc["attention_mask"]


@asserted
def remote_residuals(rlm: RemoteLM, texts: Sequence[str], layer: int, *,
                     patch_layer: int | None = None, patch_vec: np.ndarray | None = None,
                     scale: float = 1.0, batch_row_bug: bool = False) -> np.ndarray:
    """Residuals at `layer` for a padded batch, optionally under a patch. [B, S, d].

    `batch_row_bug` reproduces h36 on purpose -- it writes the patch through `output[0]`, which is
    batch row 0 whenever the block returns a bare tensor -- so that the moved-candidates assertion
    can be shown catching it on a live NDIF call rather than on a local reconstruction. Nothing but
    the rediscovery harness should pass it.
    """
    import torch

    ids, mask = _encode(rlm, texts)
    blocks = rlm.blocks
    v = None if patch_vec is None else torch.as_tensor(
        np.asarray(patch_vec, dtype=np.float32) * float(scale))

    def build(backend):
        with rlm.model.trace({"input_ids": ids, "attention_mask": mask}, backend=backend) as tracer:
            if v is not None:
                o = blocks[int(patch_layer)].output
                if batch_row_bug:
                    h = o[0]                         # h36: batch row 0, not "the hidden states"
                else:
                    h = o if isinstance(o, torch.Tensor) else o[0]      # by type, never by index
                h[:] = h + v.to(h.device, h.dtype)
            r = blocks[int(layer)].output
            out = (r if isinstance(r, torch.Tensor) else r[0]).float().save()
        return tracer

    arr = np.asarray(rlm._run(build), dtype=np.float32)
    if arr.ndim == 2:
        # A [S, d] return from a [B, ...] request is the h36 signature on the CAPTURE side: the
        # save has silently taken batch row 0. Refuse it here rather than reshape it away.
        raise checks.LayerOutputShape(
            f"remote capture returned shape {arr.shape} for a batch of {len(texts)}; the saved "
            "object is one sequence, not the batch -- `output[0]` on a block that returns a bare "
            "tensor is batch row 0 (h36). Capture through checks.resid().")
    return arr


# --------------------------------------------------------------------------------------------
# a remote Stack, with the same assertions a local one gets
# --------------------------------------------------------------------------------------------
def _offsets(rlm: RemoteLM, text: str, spans: dict) -> dict:
    from ..extract import tokens_in_span
    enc = rlm.tok(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    return {name: sorted(set(tokens_in_span(offsets, sp))) for name, sp in spans.items()}


def _pool(hs_item: np.ndarray, idx: Sequence[int], pooling: str) -> np.ndarray:
    sel = hs_item[list(idx)]
    return sel.mean(axis=0) if pooling == "mean" else sel[-1]


@asserted
def build_remote_stack(rlm: RemoteLM, grid: Grid, layer: int, *, batch_size: int = 4,
                       span_policy: str = "auto", pooling: str = "mean",
                       declared_padding_side: str | None = None,
                       equivalence_min_cos: float = 0.999,
                       bypass: Sequence[str] = ()) -> Stack:
    """Grid x remote model x one layer -> `Stack`, with every §7 assertion run on the way.

    One layer rather than all of them because a remote job returns its saved tensors over the wire:
    the local path can afford `output_hidden_states=True`, and this cannot. That is a budget
    decision and it is recorded in the provenance (`layers`), not hidden.
    """
    if "padding" not in bypass:
        checks.assert_padding_convention(rlm.padding_side, span_policy, declared_padding_side)

    span_names = tuple(grid.span_names)
    acts = np.zeros((len(grid.items), len(span_names), 1, 0), dtype=np.float32)
    equivalence: dict[str, float] = {}
    rows: list[np.ndarray] = []

    for start in range(0, len(grid.items), batch_size):
        batch = grid.items[start:start + batch_size]
        hs = remote_residuals(rlm, [it.text for it in batch], layer)
        ids, mask = _encode(rlm, [it.text for it in batch])
        mask = np.asarray(mask)
        seq_len = hs.shape[1]
        n_real = mask.sum(axis=1)
        pooled = []
        for b, item in enumerate(batch):
            idx_map = _offsets(rlm, item.text, item.spans)
            off = (seq_len - int(n_real[b])) if (
                span_policy == "end_relative"
                or (span_policy == "auto" and rlm.padding_side == "left")) else 0
            per_span = []
            for name in span_names:
                idx = [i + off for i in idx_map[name]]
                if "spans" not in bypass:
                    checks.assert_nonempty_spans(idx, mask[b], item=start + b, span=name)
                per_span.append(_pool(hs[b], idx, pooling))
            pooled.append(np.stack(per_span))
        rows.extend(pooled)

        # batched-vs-single on the SHORTEST item of this batch, never a random sample
        j = checks.shortest_item_index([int(n) for n in n_real])
        item = batch[j]
        single = remote_residuals(rlm, [item.text], layer)
        s_ids, s_mask = _encode(rlm, [item.text])
        idx_map = _offsets(rlm, item.text, item.spans)
        soff = (single.shape[1] - int(np.asarray(s_mask).sum())) if (
            span_policy == "end_relative"
            or (span_policy == "auto" and rlm.padding_side == "left")) else 0
        for s, name in enumerate(span_names):
            u = _pool(single[0], [i + soff for i in idx_map[name]], pooling)
            if "equivalence" in bypass:
                continue
            equivalence[f"{start + j}:{name}"] = checks.assert_batch_equivalence(
                pooled[j][s][None, :], u[None, :], item=start + j,
                where=f"span {name!r} (remote)", min_cos=equivalence_min_cos)

    acts = np.stack(rows)[:, :, None, :]        # [item, span, layer=1, d]
    prov = {"model": rlm.repo_id, "layers": [int(layer)], "pooling": pooling,
            "grid_hash": grid.hash, "grid_name": grid.name,
            "code_version": _code_version(), "tokenizer_padding": rlm.padding_side,
            "span_policy": span_policy, "template": None, "lib_versions": rlm.lib_versions(),
            "batch_size": batch_size, "leak": grid.leak.summary(), "remote": True,
            "acts_digest": hashlib.sha256(np.ascontiguousarray(acts).tobytes()).hexdigest()[:16],
            "equivalence_min_cos": (min(equivalence.values()) if equivalence else None),
            "equivalence_item_rule": "shortest item in each batch"}
    checks.assert_provenance(prov)
    prov["stack_signature"] = stack_signature(prov)
    return Stack(acts=acts, grid_hash=grid.hash, span_names=span_names, layers=(int(layer),),
                 provenance=prov,
                 checks={"batched_vs_single_min_cos": equivalence,
                         "equivalence_item_rule": "shortest item in each batch"})


def _code_version() -> str:
    from .extract import code_version
    return code_version()


# --------------------------------------------------------------------------------------------
# the patched forward: h34's assertion, finally on an NDIF call
# --------------------------------------------------------------------------------------------
@asserted
def asserted_remote_patched_logprob(rlm: RemoteLM, lead: str, candidates: Sequence[str], *,
                                    patch_layer: int | None = None,
                                    patch_vec: np.ndarray | None = None, scale: float = 1.0,
                                    base: np.ndarray | None = None, atol: float = 1e-6,
                                    batch_row_bug: bool = False) -> np.ndarray:
    """Teacher-forced log p(candidate | lead) for every candidate in ONE padded remote job, with
    the h34/h36 moved-candidates assertion on the way.

    This is `extract.asserted_patched_logprob`'s remote twin, and it differs in the one way that
    matters: the local version scores candidates one per forward, exactly as the frozen local
    scripts do, so the assertion is over the set. The remote scripts batch, which is what made h34
    and h36 possible at all, so here the assertion is over the batch -- the number of candidates
    whose score changed must equal the batch size, and a patch that reaches one row of a padded
    batch fails it.

    `base` may be omitted when the patch is the zero vector (the no-patch arm): nothing is expected
    to move and the assertion is skipped rather than inverted.
    """
    import torch

    texts = [f"{lead}{c}" for c in candidates]
    ids, mask = _encode(rlm, texts)
    n_lead = len(rlm.tok(lead, add_special_tokens=True)["input_ids"])
    blocks = rlm.blocks
    # Bound OUTSIDE the trace on purpose. The block body's source is shipped to the deployment, and
    # any attribute path through a non-whitelisted module fails there -- `rlm.model.lm_head.output`
    # inside the block is refused with "Module lsx.core.remote is not whitelisted", because `rlm`
    # is an instance of a class defined in this file. Measured, not guessed: it is what the second
    # run of `results/ndif_probe_asserted.json` did. Locals are fine; module paths are not.
    lm_head = rlm.model.lm_head
    v = None if patch_vec is None else torch.as_tensor(
        np.asarray(patch_vec, dtype=np.float32) * float(scale))

    # Score only the candidate's own tokens. Under LEFT padding the lead does not start at position
    # 0 of every row -- it starts at `n_pad` -- so the lead mask is built per row from the
    # attention mask rather than from a single `n_lead` offset. A script that used a fixed offset
    # here would score part of the pad block on every short row, which is h39 in the scoring code
    # rather than in the span indexing.
    tgt = ids[:, 1:]
    score_mask = mask[:, 1:].clone().float()
    n_real = mask.sum(dim=1)
    for r in range(len(texts)):
        if rlm.padding_side == "left":
            first = int(mask.shape[1] - n_real[r])
        else:
            first = 0
        score_mask[r, : first + n_lead - 1] = 0.0
    if float(score_mask.sum()) <= 0:
        raise checks.EmptySpan("no candidate tokens left to score after masking the lead")

    def build(backend):
        with rlm.model.trace({"input_ids": ids, "attention_mask": mask}, backend=backend) as tracer:
            if v is not None:
                o = blocks[int(patch_layer)].output
                if batch_row_bug:
                    h = o[0]
                else:
                    h = o if isinstance(o, torch.Tensor) else o[0]
                h[:] = h + v.to(h.device, h.dtype)
            logits = lm_head.output[:, :-1, :]
            picked = (logits.gather(-1, tgt.unsqueeze(-1).to(logits.device)).squeeze(-1).float()
                      - torch.logsumexp(logits, dim=-1).float())
            out = (picked * score_mask.to(picked.device)).sum(-1).save()
        return tracer

    scores = np.asarray(rlm._run(build), dtype=np.float64).ravel()
    if scores.size != len(candidates):
        raise checks.MovedCandidates(
            f"the remote job returned {scores.size} scores for {len(candidates)} candidates; the "
            "saved tensor is not the batch (h36)")
    if base is not None and v is not None:
        checks.assert_moved_candidates(np.asarray(base, dtype=np.float64), scores,
                                       batch=len(candidates), atol=atol)
    return scores
