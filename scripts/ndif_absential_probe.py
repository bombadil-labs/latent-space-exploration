"""Controlled causal probe of the absential ring (Gemma-2-9B-it via NDIF, block 20).

For each passage: take the top-k ring features (inactive, decoder-cosine >= tau to some active
content feature) and a generality-matched control set of inactive NON-ring features. Sum each set's
decoder directions, rescale to `frac` x the passage's mean block-20 residual norm, and add that one
vector at block 20 on every position.

Measured, per passage, for {base, ring, control}:
  (a) teacher-forced log-prob of the passage's own span given the lead (sum and per-token)
  (b) the next-token distribution at the final position: KL(patched || base) and top-5 overlap
  (c) 40-token greedy continuation from the passage (qualitative)

Claim under test: ring features are the model's own "implied but absent" content, so adding them is
less disruptive / more coherent than adding norm- and generality-matched inactive features.

Writes results/absential_probe_gemma9b.json. One sequence per job (Gemma's 256k vocab).
"""
import argparse, glob, json, os, time
import numpy as np, torch
from nnsight import LanguageModel
from lsx.ndif import ProxyAuthBackend, retry_job

ap = argparse.ArgumentParser()
ap.add_argument("--tokens_npz", default="/home/user/latent-space-exploration/results/tokens_gemma9b_l20.npz")
ap.add_argument("--grid", default="/home/user/latent-space-exploration/prompts/narrative_theme_v1.json")
ap.add_argument("--sae", default=None)
ap.add_argument("--model", default="google/gemma-2-9b-it")
ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--tau", type=float, default=0.40)
ap.add_argument("--fmt_gen", type=float, default=0.90)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--frac", type=float, default=0.15, help="patch norm as a fraction of mean residual norm")
ap.add_argument("--n_passages", type=int, default=12)
ap.add_argument("--n_gen", type=int, default=6)
ap.add_argument("--gen_tokens", type=int, default=40)
ap.add_argument("--out", default="results/absential_probe_gemma9b.json")
a = ap.parse_args()

# ---------------------------------------------------------------- feature sets (local)
if a.sae is None:
    a.sae = glob.glob("/home/user/latent-space-exploration/cache/hf/hub/models--google--gemma-scope-9b-it-res/"
                      "snapshots/*/layer_20/width_16k/average_l0_47/params.npz")[0]
sae = dict(np.load(a.sae))
W_enc, b_enc, thr, W_dec = sae["W_enc"], sae["b_enc"], sae["threshold"], sae["W_dec"]
F = W_enc.shape[1]
Wn = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)

z = np.load(a.tokens_npz)
ids = [k for k in z.files if not k.endswith("__tokens")]
refs = [k for k in ids if k != "picard"]
THEMES = ("betrayal", "sacrifice", "homecoming")
theme_ids = [k for k in refs if k.split("/")[-1] in THEMES]

encode = lambda x: (lambda pre: pre * (pre > thr))(x @ W_enc + b_enc)
fires = np.zeros(F); act = {}
for k in refs:
    m = (encode(z[k][1:]) > 0).any(0); fires += m; act[k] = np.flatnonzero(m)
gen = fires / len(refs)
fmt = gen > a.fmt_gen

# 12 passages: one per (scene, theme) cell spread over eras -> balanced on theme and era
grid = json.load(open(a.grid))
scenes, eras = grid["scenes"], grid["factors"]["era"]
chosen = []
for si, s in enumerate(scenes):
    for ti, t in enumerate(THEMES):
        chosen.append(f"{s}/{eras[(si + ti) % len(eras)]}/{t}")
chosen = chosen[: a.n_passages]

rng = np.random.default_rng(0)
sets = {}
for k in chosen:
    A = act[k]; Ac = A[~fmt[A]]
    mx = (Wn @ Wn[Ac].T).max(1)
    inactive = np.ones(F, bool); inactive[A] = False
    ring = np.flatnonzero(inactive & (mx >= a.tau) & ~fmt)
    ring = ring[np.argsort(-mx[ring])][: a.k]
    # control: inactive, non-ring (max-cos < tau), matched one-for-one on generality
    pool = np.flatnonzero(inactive & (mx < a.tau) & ~fmt)
    ctrl = []
    for f in ring:
        cand = pool[np.abs(gen[pool] - gen[f]) <= 0.02]
        cand = np.setdiff1d(cand, np.array(ctrl, int))
        if len(cand) == 0:                                  # widen if the bin is empty
            cand = np.setdiff1d(pool[np.argsort(np.abs(gen[pool] - gen[f]))[:50]], np.array(ctrl, int))
        ctrl.append(int(rng.choice(cand)))
    ctrl = np.array(ctrl)

    # coherence of a set = mean pairwise cosine among its own decoder directions. The ring's members
    # are all near the same active cluster, so they are mutually similar and their sum does not
    # cancel; 8 independent random directions do. Second control: inactive non-ring features that
    # are mutually similar (coherence-matched) and generality-matched, so the patch vector is a
    # comparably "real" direction that simply is not adjacent to anything the passage activates.
    def coherence(idx):
        C = Wn[idx] @ Wn[idx].T
        return float((C.sum() - len(idx)) / (len(idx) * (len(idx) - 1)))
    coh_ring, g_ring = coherence(ring), float(gen[ring].mean())
    seeds = rng.choice(pool, size=min(400, len(pool)), replace=False)
    Cp = Wn[seeds] @ Wn[pool].T
    best, best_cost = None, 1e9
    for si, s_ in enumerate(seeds):
        nb = pool[np.argsort(-Cp[si])[: a.k]]                    # includes the seed itself
        if len(set(nb.tolist())) < a.k:
            continue
        cost = abs(coherence(nb) - coh_ring) + abs(float(gen[nb].mean()) - g_ring)
        if cost < best_cost:
            best, best_cost = nb, cost
    ctrl2 = best if best is not None else ctrl

    resid_norm = float(np.linalg.norm(z[k][1:], axis=1).mean())
    def vec(idx):
        v = W_dec[idx].sum(0)
        return v / np.linalg.norm(v) * (a.frac * resid_norm)
    sets[k] = dict(ring=ring, ctrl=ctrl, ctrl2=ctrl2,
                   v_ring=vec(ring), v_ctrl=vec(ctrl), v_ctrl2=vec(ctrl2),
                   resid_norm=resid_norm, maxcos=mx[ring],
                   gen_ring=gen[ring], gen_ctrl=gen[ctrl], gen_ctrl2=gen[ctrl2],
                   cos_ctrl=mx[ctrl], cos_ctrl2=mx[ctrl2],
                   coh=dict(ring=coh_ring, ctrl=coherence(ctrl), ctrl2=coherence(ctrl2)))

# ---------------------------------------------------------------- remote
model = LanguageModel(a.model, device_map="auto", dispatch=False)
tok = model.tokenizer
B = model.model.layers
l = a.layer
lead = grid["lead"]
texts = {k: f"{lead} {grid['spans'][k]}" for k in chosen}


def _score(text, vec=None):
    """Teacher-forced: sum log p(span|lead), per-token mean, and final-position log-softmax (fp32)."""
    n_lead = len(tok(lead)["input_ids"])
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"]
    tgt = ids[:, 1:]
    mask = torch.ones_like(tgt, dtype=torch.float32); mask[:, : n_lead - 1] = 0
    v = None if vec is None else torch.as_tensor(vec, dtype=torch.float32)
    backend = ProxyAuthBackend(model.to_model_key())
    with model.trace({"input_ids": ids, "attention_mask": enc["attention_mask"]}, backend=backend) as tracer:
        if v is not None:
            B[l].output[0][:] = B[l].output[0] + v.to(B[l].output[0])
        logits = model.lm_head.output[:, :-1, :]
        picked = logits.gather(-1, tgt.unsqueeze(-1).to(logits.device)).squeeze(-1).float() \
            - torch.logsumexp(logits, dim=-1).float()
        lp = (picked * mask.to(picked.device)).sum(-1).save()
        last = model.lm_head.output[0, -1, :].float()
        lastlp = (last - torch.logsumexp(last, dim=-1)).save()
    res = backend.wait(tracer)
    lp = res["lp"] if isinstance(res, dict) and "lp" in res else None
    if lp is None:
        vals = [x for x in res.values() if isinstance(x, torch.Tensor)]
        lp, lastlp_v = min(vals, key=lambda t: t.numel()), max(vals, key=lambda t: t.numel())
    else:
        lastlp_v = res["lastlp"]
    return float(lp.float().reshape(-1)[0]), lastlp_v.float().cpu().numpy(), int(mask.sum())


def _gen(text, vec=None):
    backend = ProxyAuthBackend(model.to_model_key())
    v = None if vec is None else torch.as_tensor(vec, dtype=torch.float32)
    with model.generate(text, max_new_tokens=a.gen_tokens, do_sample=False, backend=backend) as tracer:
        if v is not None:
            with tracer.all():
                B[l].output[0][:] = B[l].output[0] + v.to(B[l].output[0])
        out = model.generator.output.save()
    res = backend.wait(tracer)
    o = res["out"] if isinstance(res, dict) and "out" in res else next(x for x in res.values() if isinstance(x, torch.Tensor))
    o = o[0] if o.dim() == 2 else o
    return tok.decode(o[len(tok(text)["input_ids"]):], skip_special_tokens=True)


score = lambda *x: retry_job(lambda: _score(*x))
generate = lambda *x: retry_job(lambda: _gen(*x))

ckpt = a.out + ".partial"
rows = {}
try:
    rows = json.load(open(ckpt)); print("resuming", len(rows), "passages from", ckpt)
except Exception:
    pass

t0 = time.time(); n_jobs = 0
for i, k in enumerate(chosen):
    if k in rows:
        continue
    s = sets[k]
    r = dict(passage=k, resid_norm=s["resid_norm"],
             ring_features=[int(x) for x in s["ring"]], ring_maxcos=[float(x) for x in s["maxcos"]],
             ring_generality=[float(x) for x in s["gen_ring"]],
             ctrl_features=[int(x) for x in s["ctrl"]], ctrl_maxcos=[float(x) for x in s["cos_ctrl"]],
             ctrl_generality=[float(x) for x in s["gen_ctrl"]],
             ctrl2_features=[int(x) for x in s["ctrl2"]], ctrl2_maxcos=[float(x) for x in s["cos_ctrl2"]],
             ctrl2_generality=[float(x) for x in s["gen_ctrl2"]], coherence=s["coh"])
    lp0, d0, ntok = score(texts[k], None); n_jobs += 1
    r["n_span_tokens"] = ntok; r["logprob_base"] = lp0
    for cond, v in (("ring", s["v_ring"]), ("ctrl", s["v_ctrl"]), ("ctrl2", s["v_ctrl2"])):
        lp, d, _ = score(texts[k], v); n_jobs += 1
        p = np.exp(d)
        r[f"logprob_{cond}"] = lp
        r[f"dlogprob_{cond}"] = lp - lp0
        r[f"dlogprob_per_token_{cond}"] = (lp - lp0) / ntok
        r[f"kl_{cond}"] = float((p * (d - d0)).sum())
        r[f"top5_overlap_{cond}"] = len(set(np.argsort(-d)[:5]) & set(np.argsort(-d0)[:5]))
        r[f"top5_{cond}"] = [tok.decode([int(t)]) for t in np.argsort(-d)[:5]]
    r["top5_base"] = [tok.decode([int(t)]) for t in np.argsort(-d0)[:5]]
    if i < a.n_gen:
        r["gen_base"] = generate(texts[k], None); n_jobs += 1
        r["gen_ring"] = generate(texts[k], s["v_ring"]); n_jobs += 1
        r["gen_ctrl"] = generate(texts[k], s["v_ctrl"]); n_jobs += 1
        r["gen_ctrl2"] = generate(texts[k], s["v_ctrl2"]); n_jobs += 1
    rows[k] = r
    print(f"[{i+1}/{len(chosen)}] {k}: dlp/tok ring {r['dlogprob_per_token_ring']:+.4f} "
          f"ctrl {r['dlogprob_per_token_ctrl']:+.4f} ctrl2 {r['dlogprob_per_token_ctrl2']:+.4f} "
          f"| KL ring {r['kl_ring']:.3f} ctrl {r['kl_ctrl']:.3f} ctrl2 {r['kl_ctrl2']:.3f} "
          f"| {n_jobs} jobs {time.time()-t0:.0f}s", flush=True)
    json.dump(rows, open(ckpt, "w"), indent=1)

R = list(rows.values())
m = lambda f: float(np.mean([x[f] for x in R]))
summary = dict(n_passages=len(R), k=a.k, tau=a.tau, frac=a.frac, layer=a.layer, model=a.model)
for c in ("ring", "ctrl", "ctrl2"):
    summary[f"dlogprob_per_token_{c}"] = m(f"dlogprob_per_token_{c}")
    summary[f"kl_{c}"] = m(f"kl_{c}")
    summary[f"top5_overlap_{c}"] = m(f"top5_overlap_{c}")
    summary[f"mean_generality_{c}"] = float(np.mean([np.mean(x[f"{c}_generality"]) for x in R]))
    summary[f"mean_maxcos_{c}"] = float(np.mean([np.mean(x[f"{c}_maxcos"]) for x in R]))
    summary[f"mean_coherence_{c}"] = float(np.mean([x["coherence"][c] for x in R]))
for c in ("ctrl", "ctrl2"):
    summary[f"n_ring_better_logprob_than_{c}"] = int(sum(
        x["dlogprob_per_token_ring"] > x[f"dlogprob_per_token_{c}"] for x in R))
    summary[f"n_ring_lower_kl_than_{c}"] = int(sum(x["kl_ring"] < x[f"kl_{c}"] for x in R))
print("\n=== absential probe ===")
for kk, vv in summary.items():
    print(f"  {kk}: {vv}")
json.dump(dict(summary=summary, passages=rows), open(a.out, "w"), indent=1)
print("wrote", a.out)
if os.path.exists(ckpt):
    os.remove(ckpt)
