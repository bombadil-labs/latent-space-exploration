"""Tier B extraction: pain-axis stimuli through an NDIF-hosted model, §7-asserted.

Why this is not a call to `remote.build_remote_stack`
----------------------------------------------------
`build_remote_stack` captures ONE layer per call and runs two jobs per batch. The Tier B
deliverable is a 42-layer curve on 800 sentences, i.e. 42 x (800/B) x 2 jobs -- thousands of
NDIF round trips. So the multi-layer capture lives here, and it carries the SAME assertions,
on the same helpers, rather than a second convention:

  1. `checks.assert_padding_convention` on the side read back off the remote tokenizer.
     We index END-RELATIVE (final token at -1, mean over the attention mask), so left padding
     is the correct convention and right padding must fail. This is the h39 trap.
  2. batched-vs-single equivalence on the SHORTEST item of each batch (`checks.shortest_item_index`,
     `checks.assert_batch_equivalence`) -- the maximally padded row under left padding, and the
     row a mean-over-padding bug corrupts most.
  3. non-empty spans (`checks.assert_nonempty_spans`) for both pooled readouts of every row.
  4. the block output resolved BY TYPE, `o if isinstance(o, torch.Tensor) else o[0]`, written
     inline because a trace block may not reach into a non-whitelisted module. `output[0]` is
     batch row 0 on a block that returns a bare tensor -- h36.
  5. a cross-check (`cross_check_against_asserted_path`) that this module's in-trace pooling
     reproduces, to 1e-3 cosine, the offline pooling of `remote.remote_residuals` -- the audited
     single-layer path. That is what ties the fast path to the reviewed one.

Faithfulness to `Pain-axis/scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py`
-----------------------------------------------------------------------------------------------
`extract_activations` does, per prompt, `model.to_tokens(prompt)` (TransformerLens: prepends BOS)
then `cache["blocks.{i}.hook_resid_post"]`, taking `resid[0, -1, :]` and `resid.mean(dim=1)`.

  * layer i = residual AFTER block i, i in 0..n_layers-1. Their published curve for
    gemma-2-9b-it has exactly 42 rows, 0..41, for a 42-block model, so there is no embedding
    row in their indexing. Ours matches index for index.
  * BOS: Gemma-2's tokenizer adds `<bos>` under `add_special_tokens=True`, which is what TL's
    `to_tokens` does. No chat template, as in theirs.
  * their mean is over ALL tokens of an UNPADDED single prompt, BOS included. Ours is the
    attention-mask-weighted mean, which is the same set of positions. Assertion 2 is what makes
    that a measured claim rather than a hopeful one.

`layer_embed` is ours, not theirs: the output of `embed_tokens`, i.e. the true static embedding
table lookup before any block. Note it is the UNSCALED embedding; Gemma-2 multiplies by
`sqrt(d_model)` before block 0, so HF's `hidden_states[0]` is ours times a positive scalar.
Every quantity in the Tier A analysis (difference in means, PCA denoising, projection, ROC-AUC)
is invariant to a positive scalar, so the AUC is identical either way; `EMBED_SCALE_NOTE` records
this so nobody has to rederive it.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import checks

EMBED_SCALE_NOTE = (
    "layer_embed is embed_tokens.output (unscaled). Gemma-2 scales by sqrt(d_model) before "
    "block 0, so HF hidden_states[0] == layer_embed * sqrt(d_model). Difference-in-means, PCA "
    "denoising, projection and ROC-AUC are all invariant to a positive scalar, so AUC is "
    "unchanged. Norms and cosines-with-other-layers are NOT; do not mix them."
)


def _embed_module(rlm):
    m = rlm.model
    for path in ("model.embed_tokens", "transformer.wte", "gpt_neox.embed_in"):
        obj = m
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError(f"no embedding module found on {rlm.repo_id}")


@dataclass
class Pooled:
    """final_token and mean pooling at every block output, plus the embedding layer."""
    final_token: np.ndarray         # [n, n_layers, d]
    mean: np.ndarray                # [n, n_layers, d]
    embed_final_token: np.ndarray   # [n, d]
    embed_mean: np.ndarray          # [n, d]
    equivalence: dict
    n_jobs: int


def _encode(rlm, texts: Sequence[str], *, add_special_tokens: bool = True):
    """Tokenise for capture.

    `add_special_tokens=False` is for text ALREADY rendered through the chat template: Gemma-2's
    template emits `<bos>` itself, so letting the tokenizer add a second one shifts every
    position and puts a duplicated BOS at the start of the mean. The flag is threaded rather
    than inferred because "does this string already carry a BOS" is not decidable from the
    string, and guessing it is exactly the class of error the §7 assertions exist to catch.
    """
    enc = rlm.tok(list(texts), return_tensors="pt", padding=True,
                  add_special_tokens=add_special_tokens)
    return enc["input_ids"], enc["attention_mask"]


def _pooled_job(rlm, texts: Sequence[str], *, add_special_tokens: bool = True) -> dict:
    """ONE NDIF job: masked-mean and final-token pooling at all block outputs + embeddings.

    Pooling happens INSIDE the trace on purpose. Returning [B, S, d] for 43 layers is ~300 MB a
    batch over the wire; returning [B, 43, d] is ~25 MB. That is a bandwidth decision and it is
    the reason the equivalence assertion (2) exists -- it is the only thing standing between this
    optimisation and a silent mean-over-padding.
    """
    import torch

    from .remote import _run_saved

    ids, mask = _encode(rlm, texts, add_special_tokens=add_special_tokens)
    blocks = rlm.blocks
    embed = _embed_module(rlm)
    n_layers = len(blocks)

    def build(backend):
        with rlm.model.trace({"input_ids": ids, "attention_mask": mask}, backend=backend) as tracer:
            m = mask.float()
            fts, mns = [], []
            eo = embed.output
            eh = (eo if isinstance(eo, torch.Tensor) else eo[0]).float()
            w = m.to(eh.device).unsqueeze(-1)
            denom = w.sum(dim=1).clamp(min=1.0)
            e_ft = eh[:, -1, :].save()
            e_mn = ((eh * w).sum(dim=1) / denom).save()
            for i in range(n_layers):
                o = blocks[i].output
                # BY TYPE, never by index: `o[0]` is batch row 0 on a bare-tensor block (h36).
                h = (o if isinstance(o, torch.Tensor) else o[0]).float()
                fts.append(h[:, -1, :])
                mns.append((h * w).sum(dim=1) / denom)
            ft = torch.stack(fts, dim=1).save()
            mn = torch.stack(mns, dim=1).save()
        return tracer

    res = _run_saved(rlm, build)

    def grab(name, want_dim):
        v = res.get(name)
        if v is None:
            cands = [x for x in res.values() if hasattr(x, "dim") and x.dim() == want_dim]
            if not cands:
                raise checks.LayerOutputShape(
                    f"remote job returned no {want_dim}-D tensor for {name!r}")
            v = cands[0]
        return np.asarray(v.float().cpu().numpy(), dtype=np.float32)

    out = {"final_token": grab("ft", 3), "mean": grab("mn", 3),
           "embed_final_token": grab("e_ft", 2), "embed_mean": grab("e_mn", 2)}

    b = len(texts)
    for k, arr in out.items():
        if arr.shape[0] != b:
            # The h36 signature on the CAPTURE side: the save silently took batch row 0.
            raise checks.LayerOutputShape(
                f"{k}: remote capture returned shape {arr.shape} for a batch of {b}; the saved "
                "object is not the batch (h36).")
    if out["final_token"].shape[1] != n_layers:
        raise checks.LayerOutputShape(
            f"captured {out['final_token'].shape[1]} layers, model has {n_layers} blocks")
    for k, arr in out.items():
        if not np.isfinite(arr).all():
            raise checks.LayerOutputShape(f"{k}: non-finite values in remote capture")
    return out


def extract_pooled(rlm, texts: Sequence[str], *, batch_size: int = 20,
                   equivalence_min_cos: float = 0.999, check_every: int = 1,
                   add_special_tokens: bool = True, verbose: bool = True) -> Pooled:
    """Pooled residuals at every layer for `texts`, with the §7 assertions on the way."""
    # (1) padding convention. We index end-relative; right padding would put pads at -1.
    checks.assert_padding_convention(rlm.padding_side, "end_relative")

    n_layers = len(rlm.blocks)
    d = int(rlm.model.config.hidden_size)
    ft = np.zeros((len(texts), n_layers, d), dtype=np.float32)
    mn = np.zeros_like(ft)
    e_ft = np.zeros((len(texts), d), dtype=np.float32)
    e_mn = np.zeros_like(e_ft)
    equivalence: dict[str, float] = {}
    n_jobs = 0

    for bi, start in enumerate(range(0, len(texts), batch_size)):
        batch = list(texts[start:start + batch_size])
        t0 = time.time()
        out = _pooled_job(rlm, batch, add_special_tokens=add_special_tokens)
        n_jobs += 1
        ids, mask = _encode(rlm, batch, add_special_tokens=add_special_tokens)
        mask = np.asarray(mask)
        n_real = mask.sum(axis=1)
        seq_len = mask.shape[1]

        # (3) non-empty spans. Under left padding the real tokens are the LAST n_real positions;
        # both readouts (final token, masked mean) must live entirely inside them.
        for b in range(len(batch)):
            idx = list(range(seq_len - int(n_real[b]), seq_len))
            checks.assert_nonempty_spans(idx, mask[b], item=start + b, span="all_real_tokens")
            checks.assert_nonempty_spans([seq_len - 1], mask[b], item=start + b, span="final_token")

        ft[start:start + len(batch)] = out["final_token"]
        mn[start:start + len(batch)] = out["mean"]
        e_ft[start:start + len(batch)] = out["embed_final_token"]
        e_mn[start:start + len(batch)] = out["embed_mean"]

        # (2) batched-vs-single on the SHORTEST item: the maximally padded row.
        if check_every and bi % check_every == 0 and len(batch) > 1:
            j = checks.shortest_item_index([int(x) for x in n_real])
            single = _pooled_job(rlm, [batch[j]], add_special_tokens=add_special_tokens)
            n_jobs += 1
            for name, bat, sing in (
                ("mean@mid", out["mean"][j, n_layers // 2], single["mean"][0, n_layers // 2]),
                ("mean@last", out["mean"][j, -1], single["mean"][0, -1]),
                ("final_token@mid", out["final_token"][j, n_layers // 2],
                 single["final_token"][0, n_layers // 2]),
                ("embed_mean", out["embed_mean"][j], single["embed_mean"][0]),
            ):
                equivalence[f"{start + j}:{name}"] = checks.assert_batch_equivalence(
                    bat[None, :], sing[None, :], item=start + j,
                    where=f"{name} (tierB remote)", min_cos=equivalence_min_cos)
        if verbose:
            print(f"    batch {bi} [{start}:{start + len(batch)}] seq={seq_len} "
                  f"{time.time() - t0:.1f}s", flush=True)

    return Pooled(final_token=ft, mean=mn, embed_final_token=e_ft, embed_mean=e_mn,
                  equivalence=equivalence, n_jobs=n_jobs)


def cross_check_against_asserted_path(rlm, texts: Sequence[str], layer: int,
                                      pooled: dict, *, min_cos: float = 0.999,
                                      add_special_tokens: bool = True) -> dict:
    """(5) This module's IN-TRACE pooling vs offline pooling of `remote.remote_residuals`.

    `remote_residuals` is the reviewed single-layer path that returns the full [B, S, d]. Pooling
    it here, locally, with the same mask, must reproduce what the trace computed on the server.
    If it does not, the in-trace arithmetic is wrong -- which is the whole risk this optimisation
    takes on.
    """
    from .remote import remote_residuals

    if not add_special_tokens:
        # remote_residuals does its own tokenisation with add_special_tokens=True. On text that
        # already carries a template BOS the two paths would encode DIFFERENT token sequences,
        # and the comparison would be between two different inputs -- a check that can only
        # mislead. Run this cross-check on the raw stimuli instead; the in-trace arithmetic it
        # validates does not depend on which string went in.
        raise ValueError(
            "cross_check_against_asserted_path cannot run on chat-rendered text: the offline "
            "path re-tokenises with add_special_tokens=True and would compare two different "
            "token sequences. Cross-check on the raw (untemplated) stimuli.")
    hs = remote_residuals(rlm, texts, layer)
    ids, mask = _encode(rlm, texts)
    mask = np.asarray(mask).astype(np.float32)
    w = mask[:, :, None]
    off_mean = (hs * w).sum(axis=1) / w.sum(axis=1).clip(min=1.0)
    off_ft = hs[:, -1, :]
    out = {}
    for name, off, got in (("mean", off_mean, pooled["mean"][:, layer]),
                           ("final_token", off_ft, pooled["final_token"][:, layer])):
        cos = min(checks.cosine(off[i], got[i]) for i in range(len(texts)))
        rel = float(np.max(np.abs(off - got)) / (np.max(np.abs(off)) + 1e-8))
        if cos < min_cos:
            raise checks.BatchEquivalence(
                f"in-trace {name} pooling at layer {layer} disagrees with offline pooling of the "
                f"asserted remote_residuals path: min cos {cos:.6f} < {min_cos}")
        out[f"{name}_min_cos"] = float(cos)
        out[f"{name}_max_rel"] = rel
    return out


def digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
