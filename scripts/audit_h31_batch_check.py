"""Audit (hour 39): does the hour-31 Gemma time-grid extraction read the right passage?

Reproduces `ndif_time_translation_extract.py`'s exact read idiom (`B[bi].output[0]`, span pooling)
for three deliberately different real grid passages of different token lengths, in two ways:
  A. all three in ONE remote job via three `tracer.invoke` blocks (the hour-31 path, batch > 1)
  B. three separate batch-of-one jobs (ground truth)
Prints pairwise distances within A (are the three vectors distinct?) and A-vs-B per passage
(does batching change the answer?).  Also pools the LAST token, which under left padding is
position-stable, to separate "wrong batch row" from "right row, shifted span indices".
"""
import json, time
import numpy as np
import torch
from nnsight import LanguageModel

from lsx.extract import parse_roles, tokens_in_span
from lsx.ndif import ProxyAuthBackend, retry_job

MODEL = "google/gemma-2-9b-it"
BLOCKS_USED = [19]          # residual layer 20 -> block 19 (the layer hour 31 headlines)

G = json.load(open("prompts/time_translation_v2.json"))
model = LanguageModel(MODEL, device_map="auto", dispatch=False)
tok = model.tokenizer
print("tokenizer padding_side:", tok.padding_side, flush=True)

B = model.model.layers
D = model.config.hidden_size


def _spans(marked):
    parsed = parse_roles(marked)
    enc = tok(parsed.text, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    return (parsed.text,
            tokens_in_span(offs, parsed.spans["interval"][0]),
            tokens_in_span(offs, parsed.spans["state"][0]),
            len(enc["input_ids"]))


cands = []
for k, m in list(G["prompts"].items()):
    text, iv, st, n = _spans(m)
    cands.append((k, m, n))
cands.sort(key=lambda x: x[2])
chosen = [cands[0], cands[len(cands) // 2], cands[-1]]
print("chosen:", [(k, n) for k, _, n in chosen], flush=True)


def _pooled_both(idx_iv, idx_st, block_idx, D):
    per_layer = []
    for bi in block_idx:
        o = B[bi].output[0]
        iv = o[..., idx_iv, :].mean(-2).reshape(-1, D)[-1]
        st = o[..., idx_st, :].mean(-2).reshape(-1, D)[-1]
        last = o[..., -1, :].reshape(-1, D)[-1]
        per_layer.append(torch.stack([iv, st, last]))
    return torch.stack(per_layer, dim=1)          # [3, nL, d]


specs = [_spans(m) for _, m, _ in chosen]


def batched():
    backend = ProxyAuthBackend(model.to_model_key())
    with model.trace(backend=backend) as tracer:
        with tracer.invoke(specs[0][0]):
            r0 = _pooled_both(specs[0][1], specs[0][2], BLOCKS_USED, D).cpu().save()
        with tracer.invoke(specs[1][0]):
            r1 = _pooled_both(specs[1][1], specs[1][2], BLOCKS_USED, D).cpu().save()
        with tracer.invoke(specs[2][0]):
            r2 = _pooled_both(specs[2][1], specs[2][2], BLOCKS_USED, D).cpu().save()
    res = backend.wait(tracer)
    return [res[k].float().cpu().numpy() for k in ("r0", "r1", "r2")]


def single(i):
    backend = ProxyAuthBackend(model.to_model_key())
    with model.trace(backend=backend) as tracer:
        with tracer.invoke(specs[i][0]):
            r = _pooled_both(specs[i][1], specs[i][2], BLOCKS_USED, D).cpu().save()
    res = backend.wait(tracer)
    return res["r"].float().cpu().numpy()


t0 = time.time()
A = retry_job(batched, attempts=4, wait_s=10.0)
print(f"batched job done {time.time()-t0:.0f}s", flush=True)
Bv = [retry_job(lambda i=i: single(i), attempts=4, wait_s=10.0) for i in range(3)]
print(f"single jobs done {time.time()-t0:.0f}s", flush=True)

names = ["interval", "state", "lasttok"]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


for r, nm in enumerate(names):
    print(f"\n== {nm} span, layer 20 ==")
    print("  within batched job, pairwise cosine (distinct passages should be well under 1):")
    for i in range(3):
        for j in range(i + 1, 3):
            print(f"    A{i} vs A{j}: cos={cos(A[i][r,0], A[j][r,0]):.6f}")
    print("  batched vs batch-of-one, same passage (1.000000 => batching is safe):")
    for i in range(3):
        print(f"    A{i} vs B{i}: cos={cos(A[i][r,0], Bv[i][r,0]):.6f}  "
              f"maxabs={np.abs(A[i][r,0]-Bv[i][r,0]).max():.4f}  "
              f"|A|={np.linalg.norm(A[i][r,0]):.2f} |B|={np.linalg.norm(Bv[i][r,0]):.2f}")
    print("  cross-check: batched i vs batch-of-one j (i != j) should be LOW:")
    for i in range(3):
        for j in range(3):
            if i != j:
                print(f"    A{i} vs B{j}: cos={cos(A[i][r,0], Bv[j][r,0]):.6f}")
np.savez("results/audit_h31_batch_check.npz", A=np.array(A), B=np.array(Bv))
print("\nwrote results/audit_h31_batch_check.npz", flush=True)
