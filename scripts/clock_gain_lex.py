"""Context-free lexical features for the clock-gain floor (spec sec 3.2).

For every cell of arms A and B, tokenize the prompt exactly as the extractor does, take
the token ids whose char span overlaps the `state` role span, and store
  emb  : mean static embedding of those ids (= model.get_input_embeddings()(ids).mean(0))
  bow  : one-hot count vector over the ids observed anywhere in the grid

Written to results/clock_gain_v1_lex.npz. No forward pass: tokenizer + embedding table only.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "src")

from lsx import LM
from lsx.extract import parse_roles, tokens_in_span

G = json.load(open("prompts/clock_gain_v1.json"))
S, TP, NP_ = G["subjects"], G["timepoints"], G["n_paraphrases"]

lm = LM.from_pretrained("Qwen/Qwen2.5-1.5B")
E = lm.model.get_input_embeddings()

ids_by = {}
for arm in ("A", "B"):
    for k, marked in sorted(G["arms"][arm].items()):
        parsed = parse_roles(marked)
        enc, offsets = lm.encode(parsed.text)
        idx = sorted({i for sp in parsed.spans["state"] for i in tokens_in_span(offsets, sp)})
        ids_by[f"{arm}/{k}"] = enc["input_ids"][0][idx].cpu().numpy()

vocab = sorted({int(i) for v in ids_by.values() for i in v})
vpos = {v: j for j, v in enumerate(vocab)}
print(f"{len(ids_by)} cells, vocabulary {len(vocab)} distinct ids", flush=True)

out = {"_vocab": np.array(vocab, dtype=np.int64)}
with torch.no_grad():
    for key, ids in ids_by.items():
        t = torch.tensor(ids, dtype=torch.long)
        out[f"emb/{key}"] = E(t).mean(0).float().numpy()
        b = np.zeros(len(vocab), dtype=np.float32)
        for i in ids:
            b[vpos[int(i)]] += 1.0
        out[f"bow/{key}"] = b / (np.linalg.norm(b) + 1e-12)
        out[f"ntok/{key}"] = np.array([len(ids)], dtype=np.int64)

np.savez_compressed("results/clock_gain_v1_lex.npz", **out)
print("wrote results/clock_gain_v1_lex.npz", flush=True)
