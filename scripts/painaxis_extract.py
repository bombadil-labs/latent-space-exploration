"""Tier A extraction: pain-axis stimuli through Qwen2.5-1.5B-Instruct on CPU.

Faithful to Pain-axis/scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py
`extract_activations`: residual stream AFTER each block (transformer_lens
`blocks.{i}.hook_resid_post`), one prompt per forward, final-token and mean-over-all-token
pooling, raw prompt text with NO chat template.

Two deliberate faithfulness choices, both recorded in the note:
  * TransformerLens `model.to_tokens(prompt)` prepends BOS; for Qwen2.5 TL aliases the missing
    BOS to <|endoftext|>. We prepend that token id explicitly.
  * batch size 1, so there is no padding at all: the "mean over padding" and "final token index
    under right padding" traps cannot arise. Extra array `mean_nobos` (mean over positions >= 1)
    is stored for free as a robustness check; it is NOT the faithful arithmetic.

Layer index i in the output = residual after block i, i in 0..n_layers-1. That equals
hidden_states[i+1] for i < n_layers-1; for i = n_layers-1 it is the PRE-final-norm residual
(hidden_states[-1] is the norm OUTPUT -- see src/lsx/model.py), which is what hook_resid_post
means. We take block outputs directly with forward hooks, so this is right by construction.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from lsx.model import LM  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SETS = ["S1_1P", "S1_3P", "S2_1P", "S2_3P", "ControlSupplement_1P"]
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "prompts/external/pain_axis/3.1_pain_and_control_datasets.json"
OUT = ROOT / "results/painaxis_tierA"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = json.loads(DATA.read_text())["datasets"]
    lm = LM.from_pretrained(MODEL, device="cpu", dtype=torch.float32)
    n_layers, d = lm.n_layers, lm.d_model
    bos = lm.tok.bos_token_id
    if bos is None:  # Qwen2.5: TransformerLens aliases BOS to eos (<|endoftext|>)
        bos = lm.tok.eos_token_id
    print(f"model={MODEL} layers={n_layers} d={d} bos_id={bos} "
          f"({lm.tok.convert_ids_to_tokens([bos])[0]!r})", flush=True)

    grabbed: list[torch.Tensor] = []

    def hook(mod, args, out):
        grabbed.append((out[0] if isinstance(out, tuple) else out).detach()[0])

    handles = [b.register_forward_hook(hook) for b in lm.blocks]
    try:
        for name in SETS:
            path = OUT / f"acts_{name}.npz"
            if path.exists():
                print(f"{name}: cached", flush=True)
                continue
            sents = ds[name]["sentences"]
            ft = np.zeros((len(sents), n_layers, d), dtype=np.float32)
            mn = np.zeros_like(ft)
            nb = np.zeros_like(ft)
            t0 = time.time()
            for i, s in enumerate(sents):
                ids = lm.tok(s["prompt"], add_special_tokens=False)["input_ids"]
                ids = torch.tensor([[bos] + ids])
                grabbed.clear()
                with torch.no_grad():
                    lm.model(input_ids=ids)
                assert len(grabbed) == n_layers, (len(grabbed), n_layers)
                stack = torch.stack(grabbed)              # [L, seq, d]
                assert stack.shape[1] == ids.shape[1]
                ft[i] = stack[:, -1, :].numpy()
                mn[i] = stack.mean(dim=1).numpy()
                nb[i] = stack[:, 1:, :].mean(dim=1).numpy()
                if i % 50 == 0:
                    print(f"  {name} {i}/{len(sents)} {time.time()-t0:.0f}s", flush=True)
            np.savez_compressed(
                path, final_token=ft, mean=mn, mean_nobos=nb,
                categories=np.array([s["category"] for s in sents]),
                sets=np.array([s["set"] for s in sents]),
                prompts=np.array([s["prompt"] for s in sents]))
            print(f"{name}: {len(sents)} sentences in {time.time()-t0:.0f}s -> {path}", flush=True)
    finally:
        for h in handles:
            h.remove()
    print("EXTRACTION DONE", flush=True)


if __name__ == "__main__":
    main()
