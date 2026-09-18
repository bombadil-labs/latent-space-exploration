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


# --------------------------------------------------------------------------------------------
# generation: h29's path, which had no asserted twin until piece 5
# --------------------------------------------------------------------------------------------
def _run_saved(rlm: RemoteLM, build: Callable) -> dict:
    """Submit one traced job and return the raw dict of saved values.

    `RemoteLM._run` coerces its result to a float array, which is right for residuals and wrong for
    token ids. Generation needs the ids themselves, so it gets its own thin wrapper rather than a
    lossy cast.
    """
    from lsx.ndif import ProxyAuthBackend, retry_job

    def once():
        backend = ProxyAuthBackend(rlm._model.to_model_key())
        tracer = build(backend)
        res = backend.wait(tracer)
        if not isinstance(res, dict) or not res:
            raise TimeoutError("empty NDIF result")
        return res

    return retry_job(once)


@asserted
def asserted_remote_generate(rlm: RemoteLM, prompt: str, *, max_new_tokens: int = 48,
                             patch_layer: int | None = None, patch_vec: np.ndarray | None = None,
                             scale: float = 1.0, reimpose: bool = True,
                             batch_row_bug: bool = False) -> str:
    """Greedy continuation of `prompt`, optionally under a re-imposed residual patch.

    This is the forward h29 ran through `scripts/ndif_recompose_sweep.py`, and the two differences
    from that script are the two §7 clauses it never had:

      * the block output is resolved **by type** (`checks.resid`'s body, inlined because a trace
        block may not reach into a non-whitelisted module), never by `output[0]` -- which on
        today's Gemma-2 deployment is batch row 0, i.e. h36;
      * the padding side is read back off the remote tokenizer and checked against the indexing
        convention before anything is submitted.

    The **moved-candidates** clause cannot be asserted inside a generation job -- there is no second
    arm inside one forward to compare against -- so it is asserted by the caller in two places
    instead, and neither is optional: `assert_patch_reaches_batch` runs the same patch through the
    asserted *residual* path on a real padded batch (the h34/h36 configuration), and the battery
    compares each patched continuation against its own no-patch continuation.
    """
    import torch

    checks.assert_padding_convention(rlm.padding_side, "auto")
    blocks = rlm.blocks
    v = None if patch_vec is None else torch.as_tensor(
        np.asarray(patch_vec, dtype=np.float32) * float(scale))
    n_in = len(rlm.tok(prompt)["input_ids"])
    # Bound OUTSIDE the trace, for the reason `asserted_remote_patched_logprob` records above and
    # this function's first draft ignored: the block body's source is shipped to the deployment,
    # and `rlm.model.generator` is an attribute path through an instance of a class defined in
    # THIS module, which NDIF refuses with "Module lsx.core.remote is not whitelisted". Measured,
    # not guessed -- it failed 12 generations that way before the first continuation came back.
    mdl = rlm.model

    def build(backend):
        with mdl.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False,
                          backend=backend) as tracer:
            if v is not None:
                if reimpose:
                    with tracer.all():
                        o = blocks[int(patch_layer)].output
                        if batch_row_bug:
                            h = o[0]
                        else:
                            h = o if isinstance(o, torch.Tensor) else o[0]
                        h[:] = h + v.to(h.device, h.dtype)
                else:
                    o = blocks[int(patch_layer)].output
                    if batch_row_bug:
                        h = o[0]
                    else:
                        h = o if isinstance(o, torch.Tensor) else o[0]
                    h[:] = h + v.to(h.device, h.dtype)
            out = mdl.generator.output.save()
        return tracer

    res = _run_saved(rlm, build)
    o = res.get("out")
    if o is None:
        o = next(x for x in res.values() if isinstance(x, torch.Tensor))
    ids = o[0] if o.dim() == 2 else o
    return rlm.tok.decode(ids[n_in:], skip_special_tokens=True)


@asserted
def assert_patch_reaches_batch(rlm: RemoteLM, texts: Sequence[str], layer: int,
                               patch_vec: np.ndarray, *, scale: float = 1.0,
                               atol: float = 1e-3, batch_row_bug: bool = False) -> int:
    """§7.3 on a generation battery's patch: run it through the asserted residual path on a padded
    batch and require that **every** row moved.

    h34/h36 is a patch that reaches one row of a padded batch. A generation job traces one prompt,
    so the bug cannot show there -- which is exactly why h29's script never tripped it, and why the
    check has to be run explicitly against the same patch tensor the generations use.
    """
    base = remote_residuals(rlm, texts, layer)
    patched = remote_residuals(rlm, texts, layer, patch_layer=layer, patch_vec=patch_vec,
                               scale=scale, batch_row_bug=batch_row_bug)
    # per-row summary: the mean absolute change over the row's own positions
    b = np.abs(patched - base).mean(axis=(1, 2))
    return checks.assert_moved_candidates(np.zeros_like(b), b, batch=len(texts), atol=atol)


@asserted
def asserted_remote_tail_pool(rlm: RemoteLM, texts: Sequence[str], layer: int,
                              n_tail: Sequence[int], *, equivalence_min_cos: float = 0.999,
                              check_equivalence: bool = True) -> tuple[np.ndarray, dict]:
    """Mean-pool the last `n_tail[i]` REAL tokens of each text at `layer`, in one padded job.

    h29's scoring read `B[read].output[0][..., -sel:, :]` inside six `tracer.invoke` blocks -- the
    h36 idiom, saved only by the fact that each invoke held one text. Here the batch is a real
    batch, the output is resolved by type, and the batched-vs-single equivalence check runs on the
    **shortest** item of the batch (§7.2), which under left padding is the maximally padded one.
    """
    hs = remote_residuals(rlm, texts, layer)
    ids, mask = _encode(rlm, texts)
    mask = np.asarray(mask)
    n_real = mask.sum(axis=1)
    if rlm.padding_side != "left":
        raise checks.PaddingConvention(
            f"tail pooling indexes end-relative and the tokenizer pads {rlm.padding_side!r}")
    pooled = []
    for i in range(len(texts)):
        k = int(min(max(1, n_tail[i]), int(n_real[i])))
        checks.assert_nonempty_spans(list(range(hs.shape[1] - k, hs.shape[1])), mask[i],
                                     item=i, span="continuation tail")
        pooled.append(hs[i, hs.shape[1] - k:, :].mean(axis=0))
    pooled = np.stack(pooled)

    info = {"equivalence_item": None, "equivalence_cos": None,
            "equivalence_item_rule": "shortest item in each batch"}
    if check_equivalence and len(texts) > 1:
        j = checks.shortest_item_index([int(x) for x in n_real])
        single = remote_residuals(rlm, [texts[j]], layer)
        s_ids, s_mask = _encode(rlm, [texts[j]])
        s_real = int(np.asarray(s_mask).sum())
        k = int(min(max(1, n_tail[j]), s_real))
        u = single[0, single.shape[1] - k:, :].mean(axis=0)
        info["equivalence_item"] = int(j)
        info["equivalence_cos"] = checks.assert_batch_equivalence(
            pooled[j][None, :], u[None, :], item=j, where="continuation tail (remote)",
            min_cos=equivalence_min_cos)
    return pooled, info
