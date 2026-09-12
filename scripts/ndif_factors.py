"""Factor lens / composition / cross-talk on an NDIF-hosted model (same protocol as stage6_factors.py).
Directions from a stacks npz produced by ndif_extract.py (index i = output of block i). Patches add
scale*dir to block `layer`'s output at every position. All variants of a scene are scored in ONE
padded batch job per condition (teacher-forced log-prob of the span given the lead)."""
import argparse, itertools, json, time
import numpy as np, torch
from nnsight import LanguageModel
from lsx.ndif import ProxyAuthBackend, retry_job
ap = argparse.ArgumentParser(); ap.add_argument("grid"); ap.add_argument("stacks"); ap.add_argument("--model", default="EleutherAI/gpt-j-6b")
ap.add_argument("--layer", type=int, default=14); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--out", default=None)
a = ap.parse_args()
g = json.load(open(a.grid)); z = np.load(a.stacks); X = {k: z[k] for k in z.files}
F = g["factors"]; names = list(F); S = g["scenes"]; lead = g["lead"]; spans = g["spans"]
combos = list(itertools.product(*[F[n] for n in names])); key = lambda s, c: "/".join([s, *c])
model = LanguageModel(a.model, device_map="auto", dispatch=False); tok = model.tokenizer
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "right"     # the lead mask below assumes the lead starts at position 0 of every row
def blocks(m):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = m
        try:
            for x in path.split("."): obj = getattr(obj, x)
            return obj
        except AttributeError: continue
B = blocks(model); l = a.layer
def resid(block):
    """Hidden states at a block's output. transformers >= 4.54 returns a bare Tensor [batch, seq, d]
    from Llama/Gemma/Qwen decoder layers (older versions, and GPT-J today, return a tuple). Indexing
    `block.output[0]` therefore silently means "batch row 0" on those models, so a patch written that
    way lands on the FIRST SEQUENCE OF THE BATCH ONLY -- see results/notes/random_control_diagnosis.md."""
    o = block.output
    return o if isinstance(o, torch.Tensor) else o[0]
def dirs(train):
    allv = {c: np.stack([X[key(s, c)][l] for s in train]).mean(0) for c in combos}; mu = np.mean(list(allv.values()), axis=0)
    return {n: {lvl: np.mean([allv[c] for c in combos if c[i] == lvl], axis=0) - mu for lvl in F[n]} for i, n in enumerate(names)}
CHUNK = int(__import__("os").environ.get("NDIF_CHUNK", "3"))
def batch_logprob(texts, vec=None):
    if len(texts) > CHUNK:
        return np.concatenate([batch_logprob(texts[i:i + CHUNK], vec) for i in range(0, len(texts), CHUNK)])
    return _batch_logprob(texts, vec)
def _inner__batch_logprob(texts, vec=None):
    """Sum log p(span | lead) for each text, in one remote job. vec: np [d] to add at block l output."""
    n_lead = len(tok(lead)["input_ids"])
    enc = tok(texts, return_tensors="pt", padding=True); ids, am = enc["input_ids"], enc["attention_mask"]
    tgt = ids[:, 1:]; mask = am[:, 1:].clone().float(); mask[:, : n_lead - 1] = 0     # score only span tokens
    v = None if vec is None else torch.as_tensor(vec * a.scale, dtype=torch.float32)
    backend = ProxyAuthBackend(model.to_model_key())
    with model.trace({"input_ids": ids, "attention_mask": am}, backend=backend) as tracer:
        if v is not None:
            h = resid(B[l]); h[:] = h + v.to(h.device, h.dtype)
        logits = model.lm_head.output[:, :-1, :]                                   # native dtype; no full-vocab fp32 copy
        picked = logits.gather(-1, tgt.unsqueeze(-1).to(logits.device)).squeeze(-1).float() - torch.logsumexp(logits, dim=-1).float()
        out = (picked * mask.to(picked.device)).sum(-1).save()
    res = backend.wait(tracer)
    o = res["out"] if isinstance(res, dict) and "out" in res else next(x for x in res.values() if isinstance(x, torch.Tensor))
    return o.float().cpu().numpy()
_batch_logprob = lambda *a_, **k_: retry_job(lambda: _inner__batch_logprob(*a_, **k_))
rng = np.random.default_rng(0); res = []; t0 = time.time()
ckpt = (a.out or "results/ndif_factors_ckpt.json") + ".partial"
try:
    res = json.load(open(ckpt)); print("resuming", len(res), "rows from", ckpt)
except Exception: pass
done_scenes = {x["scene"] for x in res}
def decompose(gd):
    G = np.array([gd[c] for c in combos]); gm = G.mean(); tot = ((G - gm) ** 2).sum() + 1e-12; out = {}
    for i, n in enumerate(names):
        means = {lvl: np.mean([gd[c] for c in combos if c[i] == lvl]) for lvl in F[n]}
        out[n] = float(sum((means[c[i]] - gm) ** 2 for c in combos) / tot)
    return out
for s in S:
    if s in done_scenes: continue
    rng = np.random.default_rng(S.index(s))
    D = dirs([x for x in S if x != s]); texts = [f"{lead} {spans[key(s, c)]}" for c in combos]
    base = dict(zip(combos, batch_logprob(texts)))
    def gains(vec):
        gd = dict(zip(combos, batch_logprob(texts, vec) - np.array([base[c] for c in combos])))
        if vec is not None:
            # positive control on the plumbing (hour 36): a real patch must change every candidate's
            # score, not just batch row 0. If this fires, the patch is not reaching the whole batch.
            n_changed = sum(abs(gd[c]) > 1e-6 for c in combos)
            if n_changed != len(combos):
                # One or two candidates landing on an exact bf16-precision tie (gain 0.0 while other
                # candidates in the SAME batch show real, varied O(0.1-2) gains, and it is a
                # different specific candidate each time across scenes/models/layers) is observed on
                # the 70B pair. That is NOT the hour-36 batch-row signature, which leaves only ONE
                # candidate (batch row 0) changed and every other identically, bit-for-bit, at 0 --
                # i.e. n_changed == 1, not n_changed == 7 or 8. Warn and continue on the former;
                # hard-fail only on the latter (n_changed <= 1), the actual documented failure mode.
                print(f"WARNING: patch reached {n_changed}/{len(combos)} candidates: {({str(c): float(gd[c]) for c in combos})}", flush=True)
            assert n_changed > 1, f"patch reached only {n_changed}/{len(combos)} candidates -- batch-row bug (see random_control_diagnosis.md)"
        return gd
    # mid-rank on ties: a patch that changes nothing must score at chance, not 1.0. With a strict `>`
    # every tie reads as rank 1, so a no-op (or a patch that reached only part of the batch) scores
    # as a perfect selector -- that is how hour 34's Llama numbers were manufactured.
    def rank(gd, t, cands):
        o = [c for c in cands if c != t]
        return 1 + sum(gd[c] > gd[t] for c in o) + 0.5 * sum(gd[c] == gd[t] for c in o)
    null = gains(None)   # no-patch baseline: same code path, second scoring job, no direction added
    for i, n in enumerate(names):
        for lvl in F[n]:
            r = rng.normal(size=D[n][lvl].shape); r *= np.linalg.norm(D[n][lvl]) / np.linalg.norm(r)
            for cond, vec in (("factor", D[n][lvl]), ("rand", r), ("none", None)):
                gd = null if vec is None else gains(vec); dec = decompose(gd)
                for c in combos:
                    if c[i] == lvl:
                        res.append(dict(scene=s, test="B", factor=n, cond=cond, rank=rank(gd, c, [cc for cc in combos if all(cc[j] == c[j] for j in range(len(names)) if j != i)])))
                res.append(dict(scene=s, test="X", factor=n, cond=cond, **{f"frac_{m}": dec[m] for m in names}))
    for c in combos:
        gd = gains(sum(D[n][c[i]] for i, n in enumerate(names))); res.append(dict(scene=s, test="D", rank=rank(gd, c, combos)))
    print(f"scene {s} done {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(ckpt, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(f"\n=== {a.model} layer={l} scale={a.scale} ===")
for n in names:
    k = len(F[n]); f = lambda c: np.mean([x["rank"] for x in res if x["test"] == "B" and x["factor"] == n and x["cond"] == c])
    print(f"(B) {n:6s} lens: rank/{k} factor-dir {f('factor'):.2f}  random {f('rand'):.2f}  no-patch {f('none'):.2f}   (chance {(k+1)/2:.1f})")
print(f"(D) all {len(names)} factors composed: rank/{len(combos)} {np.mean([x['rank'] for x in res if x['test']=='D']):.2f}   (chance {(len(combos)+1)/2:.1f})")
print("(X) cross-talk: rows = patched factor, cols = fraction of gain variance explained by each factor")
print("        " + "".join(f"{m:>9s}" for m in names))
for n in names + ["rand"]:
    xs = [x for x in res if x["test"] == "X" and ((x["factor"] == n and x["cond"] == "factor") if n != "rand" else x["cond"] == "rand")]
    print(f"{n:>7s} " + "".join(f"{np.mean([x[f'frac_{m}'] for x in xs]):9.2f}" for m in names))
if a.out: json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

