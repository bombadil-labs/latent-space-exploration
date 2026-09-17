"""Model wrapper: residual-stream capture and patching via plain forward hooks.

Layer indexing convention (matches HF `output_hidden_states`):
  resid[0]   = embedding output (before any transformer block)
  resid[i]   = residual stream after block i-1, i.e. input to block i
  resid[L]   = the model's FINAL-NORM OUTPUT, not the raw residual (HF applies the final norm to
               the last hidden state). Verified on Qwen2.5-1.5B: hidden_states[28] == norm(pre),
               mean position norm 190.5 vs 283.2 for the true pre-norm residual. To obtain the raw
               last-block residual, register a forward-pre-hook on `model.model.norm`. See
               tests/test_invariants.py::test_last_hidden_state_is_post_norm.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def find_blocks(model: PreTrainedModel) -> nn.ModuleList:
    """Locate the decoder block list across common HF architectures."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError(f"cannot find decoder blocks on {type(model).__name__}")


# A patch function receives the residual tensor [batch, seq, d] entering block `layer`
# and returns a replacement. `positions` are the token indices it may touch.
PatchFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class Patch:
    layer: int              # residual stream index (input to block `layer`); 0..L-1 patchable
    fn: PatchFn


class LM:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.device = device
        self.blocks = find_blocks(self.model)
        self.n_layers = len(self.blocks)
        self.d_model = self.model.config.hidden_size

    @classmethod
    def from_pretrained(cls, name: str, device: str = "cpu", dtype=torch.float32, **kw) -> "LM":
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw)
        return cls(model, tok, device)

    # ---------- encoding ----------
    def encode(self, text: str):
        """Tokenize with character offsets so role spans can be mapped to tokens."""
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        return {k: v.to(self.device) for k, v in enc.items()}, offsets

    # ---------- capture ----------
    @torch.no_grad()
    def residuals(self, text: str, layers: Iterable[int] | None = None) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Return residual stream [L+1, seq, d] (float32, cpu) and token char offsets."""
        enc, offsets = self.encode(text)
        out = self.model(**enc, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]  # [L+1, seq, d]
        if layers is not None:
            hs = hs[list(layers)]
        return hs.float().cpu(), offsets

    # ---------- patching ----------
    @contextlib.contextmanager
    def patched(self, patches: list[Patch]):
        """Context manager: apply residual-stream patches at the input of the given blocks."""
        handles = []
        by_layer: dict[int, list[PatchFn]] = {}
        for p in patches:
            by_layer.setdefault(p.layer, []).append(p.fn)

        def make_pre_hook(fns):
            def hook(module, args, kwargs):
                hidden = args[0] if args else kwargs["hidden_states"]
                for fn in fns:
                    hidden = fn(hidden)
                if args:
                    return (hidden, *args[1:]), kwargs
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = hidden
                return args, kwargs
            return hook

        for layer, fns in by_layer.items():
            handles.append(self.blocks[layer].register_forward_pre_hook(make_pre_hook(fns), with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def generate(self, text: str, max_new_tokens: int = 24, patches: list[Patch] | None = None, **kw) -> str:
        enc, _ = self.encode(text)
        with self.patched(patches or []):
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id, **kw
            )
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def next_token_logits(self, text: str, patches: list[Patch] | None = None) -> torch.Tensor:
        enc, _ = self.encode(text)
        with self.patched(patches or []):
            out = self.model(**enc)
        return out.logits[0, -1].float().cpu()

    def unembed(self, vec: torch.Tensor, k: int = 10) -> list[tuple[str, float]]:
        """Logit lens: push a residual vector through final norm + lm_head, return top-k tokens."""
        m = self.model
        norm = None
        for base_name, norm_name in (("model", "norm"), ("transformer", "ln_f"), ("gpt_neox", "final_layer_norm")):
            base = getattr(m, base_name, None)
            if base is not None and getattr(base, norm_name, None) is not None:
                norm = getattr(base, norm_name); break
        head = getattr(m, "lm_head", None) or getattr(m, "embed_out", None) or m.get_output_embeddings()
        with torch.no_grad():
            v = vec.to(self.device, dtype=next(m.parameters()).dtype)
            if norm is not None:
                v = norm(v)
            logits = head(v).float().cpu()
        top = torch.topk(logits, k)
        return [(self.tok.decode([i]), float(s)) for s, i in zip(top.values, top.indices)]


def _logprob_continuation(lm: "LM", prefix: str, continuation: str, patches=None) -> float:
    """Sum of log p(continuation tokens | prefix) under optional residual patches (applied at all positions)."""
    enc_p, _ = lm.encode(prefix)
    enc_f, _ = lm.encode(prefix + continuation)
    n_p = enc_p["input_ids"].shape[1]
    with torch.no_grad(), lm.patched(patches or []):
        logits = lm.model(**enc_f).logits[0].float()
    ids = enc_f["input_ids"][0]
    lp = torch.log_softmax(logits[:-1], dim=-1)
    tgt = ids[1:]
    return float(lp[torch.arange(n_p - 1, len(tgt)), tgt[n_p - 1:]].sum().cpu())


LM.logprob = _logprob_continuation
