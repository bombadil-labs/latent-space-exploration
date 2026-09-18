"""§7: ONE extraction path, with assertions -- and every remote forward, `Probe` included, goes
through it. The h36 bug lived in a patched forward, not in an extraction, so assertions scoped to
`Stack` alone would have missed it.

Asserted on every run:

  1. padding side read from the tokenizer, compared to the indexing convention (h39);
  2. batched-vs-single equivalence on the SHORTEST item of each batch (h39);
  3. moved-candidates on every patched forward (h34/h36);
  4. non-empty spans, never resolving into padding (h39);
  5. layer output resolved by the `resid()` helper, never by index (h36);
  6. provenance written alongside the stack: grid hash, code version, library versions, padding
     side and chat template (h36 was a transformers change and nothing was pinned).

One thing this file does that no script in `scripts/` does: it passes explicit `position_ids`
derived from the attention mask. Without them a left-padded batch gives every short row shifted
RoPE positions, and batched-vs-single equivalence fails for a reason that has nothing to do with
span indexing -- a second, independent left-padding bug that the equivalence assertion surfaces.
"""
from __future__ import annotations

import hashlib
import pathlib
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from ..extract import tokens_in_span
from ..model import LM, Patch
from . import checks
from .types import Grid, Stack

_CORE_DIR = pathlib.Path(__file__).parent


def code_version() -> str:
    """Hash of the core's own source, so a stale stack cannot be silently reused."""
    h = hashlib.sha256()
    for p in sorted(_CORE_DIR.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def lib_versions() -> dict:
    import numpy
    import transformers
    out = {"torch": torch.__version__, "transformers": transformers.__version__,
           "numpy": numpy.__version__}
    try:  # only present in the py3.12 remote venv
        import nnsight
        out["nnsight"] = nnsight.__version__
    except Exception:  # noqa: BLE001
        out["nnsight"] = None
    return out


def chat_template(lm: LM) -> str | None:
    """Directions fit on raw `lead + span` text and applied inside a chat template is an untracked
    mismatch today (spec §2a). Record the template rather than check it."""
    t = getattr(lm.tok, "chat_template", None)
    return hashlib.sha256(t.encode()).hexdigest()[:16] if isinstance(t, str) and t else None


# --------------------------------------------------------------------------------------------
# the forward
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def _forward(lm: LM, texts: Sequence[str], patches: list[Patch] | None = None):
    """Padded batch forward. Returns hidden states [B, L+1, S, d] (float32 numpy) and the mask."""
    enc = lm.tok(list(texts), return_tensors="pt", padding=True, add_special_tokens=True)
    mask = enc["attention_mask"]
    # explicit position ids: without them left-padded rows get shifted RoPE positions
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    kw = {k: v.to(lm.device) for k, v in enc.items()}
    with lm.patched(patches or []):
        out = lm.model(**kw, position_ids=pos.to(lm.device), output_hidden_states=True)
    hs = torch.stack([checks.resid(h) for h in out.hidden_states], dim=0)   # [L+1, B, S, d]
    return hs.permute(1, 0, 2, 3).float().cpu().numpy(), mask.numpy()


def _real_token_spans(lm: LM, text: str, spans: dict[str, tuple[int, int]]) -> dict[str, list[int]]:
    """Token indices of each span in the UNPADDED tokenization of this text."""
    _, offsets = lm.encode(text)
    return {name: sorted(set(tokens_in_span(offsets, sp))) for name, sp in spans.items()}


def _offset(policy: str, padding_side: str, seq_len: int, n_real: int) -> int:
    if policy == "auto":
        policy = "end_relative" if padding_side == "left" else "absolute"
    if policy == "end_relative":
        return seq_len - n_real      # correct under left padding, a no-op at batch 1
    return 0                          # "absolute": the h39 bug when padding is left


def _pool(hs_item: np.ndarray, idx: Sequence[int], pooling: str) -> np.ndarray:
    sel = hs_item[:, list(idx)]                       # [L+1, k, d]
    return sel.mean(axis=1) if pooling == "mean" else sel[:, -1]


# --------------------------------------------------------------------------------------------
# the one extraction path
# --------------------------------------------------------------------------------------------
def build_stack(lm: LM, grid: Grid, layers: Iterable[int] | None = None, *, batch_size: int = 8,
                span_policy: str = "auto", declared_padding_side: str | None = None,
                pooling: str = "mean", equivalence_min_cos: float = 0.999,
                bypass: Sequence[str] = ()) -> Stack:
    """Grid x Model x layers -> Stack, with every §7 assertion run on the way.

    `bypass` exists for the rediscovery harness only: it lets a known-bad configuration past one
    assertion so the NEXT assertion can be shown catching it on its own. Nothing else should pass it.
    """
    padding_side = lm.tok.padding_side
    if "padding" not in bypass:
        checks.assert_padding_convention(padding_side, span_policy, declared_padding_side)

    layer_list = tuple(range(lm.n_layers + 1) if layers is None else layers)
    span_names = tuple(grid.span_names)
    acts = np.zeros((len(grid.items), len(span_names), len(layer_list), lm.d_model), dtype=np.float32)
    equivalence = {}

    for start in range(0, len(grid.items), batch_size):
        batch = grid.items[start:start + batch_size]
        hs, mask = _forward(lm, [it.text for it in batch])
        seq_len = hs.shape[2]
        n_real = mask.sum(axis=1)
        for b, item in enumerate(batch):
            idx_map = _real_token_spans(lm, item.text, item.spans)
            off = _offset(span_policy, padding_side, seq_len, int(n_real[b]))
            for s, name in enumerate(span_names):
                idx = [i + off for i in idx_map[name]]
                if "spans" not in bypass:
                    checks.assert_nonempty_spans(idx, mask[b], item=start + b, span=name)
                acts[start + b, s] = _pool(hs[b], idx, pooling)[list(layer_list)]

        # batched-vs-single, on the SHORTEST item of this batch, never a random sample
        j = checks.shortest_item_index([int(n) for n in n_real])
        item = batch[j]
        single_hs, single_mask = _forward(lm, [item.text])
        idx_map = _real_token_spans(lm, item.text, item.spans)
        soff = _offset(span_policy, padding_side, single_hs.shape[2], int(single_mask.sum()))
        for s, name in enumerate(span_names):
            single = _pool(single_hs[0], [i + soff for i in idx_map[name]], pooling)[list(layer_list)]
            if "equivalence" in bypass:
                continue
            equivalence[f"{start + j}:{name}"] = checks.assert_batch_equivalence(
                acts[start + j, s], single, item=start + j, where=f"span {name!r}",
                min_cos=equivalence_min_cos)

    prov = {"model": getattr(lm.model.config, "_name_or_path", None) or type(lm.model).__name__,
            "layers": list(layer_list), "pooling": pooling, "grid_hash": grid.hash,
            "grid_name": grid.name, "code_version": code_version(),
            "tokenizer_padding": padding_side, "span_policy": span_policy,
            "template": chat_template(lm), "lib_versions": lib_versions(),
            "batch_size": batch_size, "leak": grid.leak.summary()}
    checks.assert_provenance(prov)
    return Stack(acts=acts, grid_hash=grid.hash, span_names=span_names, layers=layer_list,
                 provenance=prov,
                 checks={"batched_vs_single_min_cos": equivalence,
                         "equivalence_item_rule": "shortest item in each batch"})


def per_item_equivalence(lm: LM, grid: Grid, *, span_policy: str = "auto", batch_size: int = 8,
                         pooling: str = "mean") -> np.ndarray:
    """Diagnostic: the batched-vs-single cosine for EVERY item, so it is visible that the items a
    random sample would have drawn are the clean ones. Raises nothing."""
    padding_side = lm.tok.padding_side
    out = np.ones(len(grid.items))
    for start in range(0, len(grid.items), batch_size):
        batch = grid.items[start:start + batch_size]
        hs, mask = _forward(lm, [it.text for it in batch])
        n_real = mask.sum(axis=1)
        for b, item in enumerate(batch):
            idx_map = _real_token_spans(lm, item.text, item.spans)
            off = _offset(span_policy, padding_side, hs.shape[2], int(n_real[b]))
            single_hs, single_mask = _forward(lm, [item.text])
            soff = _offset(span_policy, padding_side, single_hs.shape[2], int(single_mask.sum()))
            worst = 1.0
            for name in grid.span_names:
                idx = [i + off for i in idx_map[name]]
                if any(i < 0 or i >= hs.shape[2] for i in idx):
                    worst = 0.0
                    continue
                v = _pool(hs[b], idx, pooling)
                u = _pool(single_hs[0], [i + soff for i in idx_map[name]], pooling)
                worst = min(worst, min(checks.cosine(v[l], u[l]) for l in range(len(v))))
            out[start + b] = worst
    return out


# --------------------------------------------------------------------------------------------
# patched forwards go through the same path
# --------------------------------------------------------------------------------------------
def asserted_patched_forward(lm: LM, texts: Sequence[str], *, patch_layer: int,
                             patch_fn: Callable[[torch.Tensor], torch.Tensor],
                             readout: Callable[[np.ndarray], np.ndarray],
                             atol: float = 1e-6) -> np.ndarray:
    """Run a patched forward over a batch and assert the patch reached every sequence.

    `readout` receives mean-pooled real-token activations of shape [batch, n_layers+1, d] and
    returns one score per sequence. The moved-candidates assertion (h34/h36) counts how many of
    those scores changed: it must equal the batch size.
    """
    base_hs, mask = _forward(lm, texts)
    patched_hs, _ = _forward(lm, texts, patches=[Patch(patch_layer, patch_fn)])

    def pooled(hs):
        m = mask[:, None, :, None]
        return (hs * m).sum(axis=2) / np.maximum(m.sum(axis=2), 1)

    base, after = readout(pooled(base_hs)), readout(pooled(patched_hs))
    checks.assert_moved_candidates(base, after, batch=len(texts), atol=atol)
    return np.asarray(after) - np.asarray(base)


def capture_residual(lm: LM, text: str, layer: int) -> np.ndarray:
    """The sanctioned residual capture. At `layer == n_layers` this is `LM.pre_norm_residual`, NOT
    `residuals()[-1]`: HF applies the final norm to the last hidden state, so that index is the
    norm output (mean position norm 190.5 vs 283.2 on Qwen2.5-1.5B, h40)."""
    if layer == lm.n_layers:
        pre, _, _ = lm.pre_norm_residual(text)
        return pre.numpy()
    if not 0 <= layer < lm.n_layers + 1:
        raise ValueError(f"layer {layer} out of range 0..{lm.n_layers}")
    hs, _ = lm.residuals(text)
    return hs[layer].numpy()
