"""Continuation-defined absential test on Gemma-2-9B-it (NDIF, block 20).

Hour 18 falsified the first realization of VISION.md's `absential` operator ("absence = inactive SAE
features adjacent in *decoder space* to the live set"): those features are the dictionary's long
tail, do not track theme, and are statistically indistinguishable from the ring of an arbitrary set
of the same size. This script tests the realization that note ends by proposing: define absence from
the model's own behaviour, not from geometry.

================================  PRE-REGISTERED DESIGN  ================================
Written down in full before any NDIF job was submitted. Nothing below was revised after seeing a
number; deviations, if any, are listed in results/notes/absential_continuation.md.

An absential is an absence that does causal work: a part the story sets up and withholds, whose
representation is nevertheless active and shapes what comes next.

1. PASSAGES  `prompts/absential_v1.json` (8 items x 3 variants, written by Claude for this test,
   era-neutral three-sentence passages in the style of prompts/narrative_theme_v1.json).
     withheld  - sentence 2 sets up a betrayal (5 items) or a homecoming (3 items) that has every
                 reason and opportunity to arrive; sentence 3 ends the passage BEFORE the turn.
     delivered - identical to withheld except sentence 3 delivers the turn.
     neutral   - identical to withheld except sentence 2 is replaced by unrelated descriptive
                 detail of comparable length, so the turn is never set up.
   Sentence 1 is shared by all three. withheld vs delivered differ only in sentence 3;
   withheld vs neutral differ only in sentence 2.

2. DIRECTIONS  From the existing Gemma theme stacks
   results/stacks_gemma_2_9b_it_narrative_theme_v1.npz (36 passages, keys scene/era/theme, values
   [42, 3584]), block 20. d(level) = mean over that theme's 12 passages - grand mean over all 36,
   unit-normalised. Nothing is held out: the test passages are new text that had no part in fitting
   the directions. "Target" = the item's own withheld theme; the other two themes are controls.

3. READOUT A (representational).  Block-20 per-token residuals of each of the 24 passages.
   Project (a) the last-token residual and (b) the mean-over-span residual (BOS dropped) onto the
   target direction. Reported both raw and norm-normalised (cosine), since the variants differ in
   length and residual norm.
     A1  withheld > neutral, on the target direction, paired within item.
     A2  withheld >= delivered at the LAST TOKEN (the absence is "present" before it is stated).
     A3 (control)  the withheld - neutral gap is larger on the target direction than on the other
         two theme directions.

4. READOUT B (behavioural).  Greedy 40-token continuation of each passage, no patch. Score each
   continuation by (i) the projection of its mean block-20 residual (continuation tokens only) onto
   the target direction, and (ii) the pre-registered lexical word lists stored in
   prompts/absential_v1.json (substring stem match, hits per 100 words of continuation text only).
     B1  withheld continuations carry the target theme more than neutral continuations do, on the
         projection readout.
     B2  the same on the lexical readout.
     B3 (control)  the effect is smaller on the two non-target theme lexicons / directions.

5. READOUT C (feature-level).  Gemma Scope JumpReLU, layer 20, width 16k, average_l0_47.
   pre = x @ W_enc + b_enc ; act = pre * (pre > threshold).
   The continuation-defined absential set of a withheld passage:
       S(p) = {f : f fires on some token of p's own greedy continuation, and on NO token of p}.
   Compare the WITHHELD passage's last-token pre-activations on S(p) against the NEUTRAL passage's
   last-token pre-activations on the same S(p).
     C1  mean pre-activation on S(p) is higher (closer to threshold) for withheld than for neutral.
     C2 (control)  the same comparison on a size-matched random set of features that are inactive
         on both passages shows no such gap.

NDIF budget: 24 (passage residuals) + 24 (continuations) + 24 (passage+continuation residuals) = 72
jobs, one sequence per job (Gemma's 256k vocab). All three phases are checkpointed to <out>.partial.
=========================================================================================

POST-HOC ADDENDUM (readout B', phase 4). Added AFTER seeing phase 2, and labelled as post-hoc
everywhere it is reported. Gemma-2-9B-it is instruction-tuned, so its raw greedy continuation of a
bare passage is meta-commentary about the passage ("**Questions:** 1. What is the main conflict...")
rather than a continuation of the story. That makes the pre-registered readout B a measurement of
the model's *analysis*, not of what it writes next. B' repeats readout B with the chat-templated
instruction used in hour 13 ("Write the next three sentences of this story passage, continuing in
the same voice"), 60 tokens greedy, for the withheld and neutral variants only (16 further jobs,
88 total). The per-step block-20 residual is saved in the same job, so B' costs no extra traces.
The pre-registered readout B is reported unchanged alongside it.

Usage:
  python scripts/ndif_absential_continuation.py --out results/absential_continuation_gemma9b.json
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch
from nnsight import LanguageModel

from lsx.ndif import ProxyAuthBackend, retry_job

REPO = "/home/user/latent-space-exploration"

ap = argparse.ArgumentParser()
ap.add_argument("--passages", default=None, help="prompts/absential_v1.json")
ap.add_argument("--stacks", default=f"{REPO}/results/stacks_gemma_2_9b_it_narrative_theme_v1.npz")
ap.add_argument("--sae", default=None)
ap.add_argument("--model", default="google/gemma-2-9b-it")
ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--gen_tokens", type=int, default=40)
ap.add_argument("--out", default="results/absential_continuation_gemma9b.json")
ap.add_argument("--skip_c", action="store_true")
ap.add_argument("--chat_variants", default="withheld,neutral", help="post-hoc readout B' (phase 4); '' to skip")
ap.add_argument("--chat_tokens", type=int, default=60)
a = ap.parse_args()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if a.passages is None:
    a.passages = os.path.join(HERE, "prompts", "absential_v1.json")

doc = json.load(open(a.passages))
LEAD = doc["lead"]
ITEMS = doc["items"]
LEXICON = doc["lexicon"]
VARIANTS = doc["variants"]
THEMES = ["betrayal", "sacrifice", "homecoming"]

texts = {f"{k}/{v}": f"{LEAD} {ITEMS[k][v]}" for k in ITEMS for v in VARIANTS}

# ------------------------------------------------------------------ directions (local)
z = np.load(a.stacks)
X = {k: z[k][a.layer] for k in z.files}
mu = np.mean(list(X.values()), axis=0)
DIRS = {}
for t in THEMES:
    d = np.mean([v for k, v in X.items() if k.split("/")[2] == t], axis=0) - mu
    DIRS[t] = d / np.linalg.norm(d)

# ------------------------------------------------------------------ remote
model = LanguageModel(a.model, device_map="auto", dispatch=False)
tok = model.tokenizer
B = model.model.layers
D = model.config.hidden_size
L = a.layer


def _resid(text):
    backend = ProxyAuthBackend(model.to_model_key())
    with model.trace(text, backend=backend) as tracer:
        h = B[L].output[0].reshape(-1, D).save()
    res = backend.wait(tracer)
    v = res["h"] if isinstance(res, dict) and "h" in res else next(x for x in res.values() if isinstance(x, torch.Tensor))
    return v.float().cpu().numpy()


def _gen(text):
    backend = ProxyAuthBackend(model.to_model_key())
    with model.generate(text, max_new_tokens=a.gen_tokens, do_sample=False, backend=backend) as tracer:
        out = model.generator.output.save()
    res = backend.wait(tracer)
    o = res["out"] if isinstance(res, dict) and "out" in res else next(x for x in res.values() if isinstance(x, torch.Tensor))
    o = o[0] if o.dim() == 2 else o
    return tok.decode(o[len(tok(text)["input_ids"]):], skip_special_tokens=True)


CHAT_INSTRUCTION = ("Write the next three sentences of this story passage, continuing in the same voice. "
                    "Reply with the sentences only.\n\n")


def chat_prompt(passage):
    return tok.apply_chat_template([{"role": "user", "content": CHAT_INSTRUCTION + passage}],
                                   tokenize=False, add_generation_prompt=True)


def _gen_chat(prompt):
    """Greedy chat continuation AND the per-step block-20 residual, in one job.

    `model.generator.output` is not returned when generation stops early on EOS, so the token ids
    are reconstructed from the per-step argmax of the lm_head (identical to greedy decoding); only
    the scalar argmax crosses the wire, not the 256k-vocab logits.
    """
    backend = ProxyAuthBackend(model.to_model_key())
    with model.generate(prompt, max_new_tokens=a.chat_tokens, do_sample=False, backend=backend) as tracer:
        hs = list().save()
        ids = list().save()
        with tracer.all():
            hs.append(B[L].output[0].reshape(-1, D)[-1])
            ids.append(model.lm_head.output.reshape(-1, model.config.vocab_size)[-1].argmax())
    res = backend.wait(tracer)
    if not isinstance(res, dict) or "hs" not in res or "ids" not in res:
        raise TimeoutError(f"incomplete result keys={list(res) if isinstance(res, dict) else type(res)}")
    text = tok.decode([int(t) for t in res["ids"]], skip_special_tokens=True)
    # hs[0] is the last prefill position (it produced the first generated token); hs[1:] are the
    # residuals at the generated positions themselves.
    H = np.stack([x.float().cpu().numpy() for x in res["hs"]])[1:]
    return text, H


resid = lambda t: retry_job(lambda: _resid(t))
generate = lambda t: retry_job(lambda: _gen(t))
generate_chat = lambda t: retry_job(lambda: _gen_chat(t))

ckpt = a.out + ".partial"
state = {"gen": {}, "gen_chat": {}}
if os.path.exists(ckpt):
    state = json.load(open(ckpt))
    state.setdefault("gen", {})
    state.setdefault("gen_chat", {})
npz_path = a.out + ".resid.npz"
RES = dict(np.load(npz_path)) if os.path.exists(npz_path) else {}


def save():
    json.dump(state, open(ckpt, "w"), indent=1)
    np.savez_compressed(npz_path, **RES)


t0 = time.time()
n_jobs = 0

# phase 1: passage residuals
for i, (k, text) in enumerate(texts.items()):
    if k in RES:
        continue
    RES[k] = resid(text)
    n_jobs += 1
    print(f"[P1 {i+1}/{len(texts)}] {k} {RES[k].shape} {n_jobs}j {time.time()-t0:.0f}s", flush=True)
    save()

# phase 2: greedy continuations
for i, (k, text) in enumerate(texts.items()):
    if k in state["gen"]:
        continue
    state["gen"][k] = generate(text)
    n_jobs += 1
    print(f"[P2 {i+1}/{len(texts)}] {k} {state['gen'][k][:60]!r} {n_jobs}j {time.time()-t0:.0f}s", flush=True)
    save()

# phase 3: residuals of passage+continuation (continuation positions read off the tail)
for i, (k, text) in enumerate(texts.items()):
    kk = k + "//full"
    if kk in RES:
        continue
    RES[kk] = resid(text + state["gen"][k])
    n_jobs += 1
    print(f"[P3 {i+1}/{len(texts)}] {k} {RES[kk].shape} {n_jobs}j {time.time()-t0:.0f}s", flush=True)
    save()

# phase 4 (POST-HOC, readout B'): chat-templated narrative continuation + its per-step residuals
chat_vars = [v for v in a.chat_variants.split(",") if v]
chat_keys = [f"{k}/{v}" for k in ITEMS for v in chat_vars]
for i, k in enumerate(chat_keys):
    if k in state["gen_chat"] and (k + "//chat") in RES:
        continue
    item, var = k.split("/")
    txt, H = generate_chat(chat_prompt(ITEMS[item][var]))
    state["gen_chat"][k] = txt
    RES[k + "//chat"] = H
    n_jobs += 1
    print(f"[P4 {i+1}/{len(chat_keys)}] {k} {txt[:60]!r} {n_jobs}j {time.time()-t0:.0f}s", flush=True)
    save()

print(f"NDIF jobs this run: {n_jobs}", flush=True)

# ------------------------------------------------------------------ analysis (local)
proj = lambda x, t: float(np.dot(x, DIRS[t]))
cos = lambda x, t: float(np.dot(x, DIRS[t]) / (np.linalg.norm(x) + 1e-9))


def lex_rate(text, theme):
    low = text.lower()
    n_words = max(1, len(text.split()))
    hits = sum(len(re.findall(re.escape(s), low)) for s in LEXICON[theme])
    return 100.0 * hits / n_words


rows = {}
for k in ITEMS:
    target = ITEMS[k]["theme"]
    rows[k] = dict(theme=target)
    for v in VARIANTS:
        key = f"{k}/{v}"
        H = RES[key][1:]                       # drop BOS
        F = RES[key + "//full"]
        n_p = RES[key].shape[0]
        C = F[n_p:]                            # continuation positions
        g = state["gen"][key]
        r = dict(
            n_tokens=int(RES[key].shape[0]),
            n_cont_tokens=int(C.shape[0]),
            prefix_match=bool(tok(texts[key] + g)["input_ids"][:n_p] == tok(texts[key])["input_ids"]),
            prefix_resid_maxdiff=float(np.abs(F[:n_p] - RES[key]).max()),
            last_proj={t: proj(H[-1], t) for t in THEMES},
            last_cos={t: cos(H[-1], t) for t in THEMES},
            mean_proj={t: proj(H.mean(0), t) for t in THEMES},
            mean_cos={t: cos(H.mean(0), t) for t in THEMES},
            cont_proj={t: proj(C.mean(0), t) for t in THEMES} if len(C) else None,
            cont_cos={t: cos(C.mean(0), t) for t in THEMES} if len(C) else None,
            lex={t: lex_rate(g, t) for t in THEMES},
            continuation=g,
        )
        kc = key + "//chat"
        if kc in RES:
            Hc = RES[kc]
            gc = state["gen_chat"][key]
            r["chat_continuation"] = gc
            r["chat_cont_proj"] = {t: proj(Hc.mean(0), t) for t in THEMES}
            r["chat_cont_cos"] = {t: cos(Hc.mean(0), t) for t in THEMES}
            r["chat_lex"] = {t: lex_rate(gc, t) for t in THEMES}
        rows[k][v] = r

# ------------------------------------------------------------------ readout C
C_rows = {}
if not a.skip_c:
    if a.sae is None:
        a.sae = glob.glob(f"{REPO}/cache/hf/hub/models--google--gemma-scope-9b-it-res/"
                          "snapshots/*/layer_20/width_16k/average_l0_47/params.npz")[0]
    sae = dict(np.load(a.sae))
    W_enc, b_enc, thr = sae["W_enc"], sae["b_enc"], sae["threshold"]
    NF = W_enc.shape[1]
    enc_pre = lambda x: x @ W_enc + b_enc
    rng = np.random.default_rng(0)

    for k in ITEMS:
        kw, kn = f"{k}/withheld", f"{k}/neutral"
        Hw, Hn = RES[kw][1:], RES[kn][1:]
        n_pw = RES[kw].shape[0]
        Cw = RES[kw + "//full"][n_pw:]
        pre_pass_w = enc_pre(Hw)
        act_pass_w = (pre_pass_w > thr).any(0)
        act_cont_w = (enc_pre(Cw) > thr).any(0)
        S = np.flatnonzero(act_cont_w & ~act_pass_w)          # the absential set
        # control: features inactive on both passages and on the continuation, size-matched
        act_pass_n = (enc_pre(Hn) > thr).any(0)
        dead = np.flatnonzero(~act_pass_w & ~act_pass_n & ~act_cont_w)
        Sc = rng.choice(dead, size=min(len(S), len(dead)), replace=False)

        lw, ln = pre_pass_w[-1], enc_pre(Hn)[-1]
        C_rows[k] = dict(
            n_absential=int(len(S)),
            n_active_passage=int(act_pass_w.sum()),
            n_active_cont=int(act_cont_w.sum()),
            pre_withheld=float(lw[S].mean()), pre_neutral=float(ln[S].mean()),
            gap_withheld=float((lw[S] - thr[S]).mean()), gap_neutral=float((ln[S] - thr[S]).mean()),
            frac_within_1_withheld=float((lw[S] > thr[S] - 1.0).mean()),
            frac_within_1_neutral=float((ln[S] > thr[S] - 1.0).mean()),
            ctrl_pre_withheld=float(lw[Sc].mean()), ctrl_pre_neutral=float(ln[Sc].mean()),
            # same comparison pooled over all passage positions, not just the last token
            pre_withheld_meanpos=float(pre_pass_w[:, S].mean()),
            pre_neutral_meanpos=float(enc_pre(Hn)[:, S].mean()),
        )

# ------------------------------------------------------------------ summary
def agg(field, sub, variant):
    return np.array([rows[k][variant][field][sub if sub != "target" else rows[k]["theme"]] for k in ITEMS])


def sign_p(x):
    """two-sided sign test p-value for x > 0, n <= 20"""
    from math import comb
    n = int((x != 0).sum()); s = int((x > 0).sum())
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(max(s, n - s), n + 1))
    return float(min(1.0, 2 * tail / 2 ** n))


summary = dict(model=a.model, layer=a.layer, n_items=len(ITEMS), gen_tokens=a.gen_tokens,
               items=[dict(name=k, theme=ITEMS[k]["theme"]) for k in ITEMS])

FIELDS = ["last_proj", "last_cos", "mean_proj", "mean_cos", "cont_proj", "cont_cos", "lex"]
if any("chat_cont_proj" in rows[k][v] for k in rows for v in VARIANTS):
    FIELDS += ["chat_cont_proj", "chat_cont_cos", "chat_lex"]

for field in FIELDS:
    have = [v for v in VARIANTS if all(field in rows[k][v] for k in ITEMS)]
    blk = {}
    for sub in ("target", *THEMES):
        d = {}
        for v in have:
            x = agg(field, sub, v)
            d[v] = float(x.mean())
        wn = agg(field, sub, "withheld") - agg(field, sub, "neutral")
        d["withheld_minus_neutral"] = float(wn.mean())
        d["withheld_gt_neutral_n"] = int((wn > 0).sum())
        d["withheld_minus_neutral_p"] = sign_p(wn)
        if "delivered" in have:
            wd = agg(field, sub, "withheld") - agg(field, sub, "delivered")
            d["withheld_minus_delivered"] = float(wd.mean())
            d["withheld_gt_delivered_n"] = int((wd > 0).sum())
            d["withheld_minus_delivered_p"] = sign_p(wd)
        blk[sub] = d
    # A3/B3 control: per item, the withheld-neutral gap on the item's OWN theme direction vs the
    # mean gap on the two other theme directions.
    tg = np.array([rows[k]["withheld"][field][rows[k]["theme"]] - rows[k]["neutral"][field][rows[k]["theme"]]
                   for k in ITEMS])
    og = np.array([np.mean([rows[k]["withheld"][field][t] - rows[k]["neutral"][field][t]
                            for t in THEMES if t != rows[k]["theme"]]) for k in ITEMS])
    blk["control_target_vs_other"] = dict(
        target_gap=float(tg.mean()), other_gap=float(og.mean()), diff=float((tg - og).mean()),
        n_target_gt_other=int(((tg - og) > 0).sum()), p=sign_p(tg - og),
        per_item={k: dict(target=float(t_), other=float(o_)) for k, t_, o_ in zip(ITEMS, tg, og)})
    summary[field] = blk

summary["by_theme_group"] = {}
for grp in ("betrayal", "homecoming"):
    ks = [k for k in ITEMS if rows[k]["theme"] == grp]
    summary["by_theme_group"][grp] = {"n": len(ks)}
    for field in FIELDS:
        have = [v for v in VARIANTS if all(field in rows[k][v] for k in ks)]
        summary["by_theme_group"][grp][field] = {
            v: float(np.mean([rows[k][v][field][grp] for k in ks])) for v in have}

if C_rows:
    dc = np.array([C_rows[k]["pre_withheld"] - C_rows[k]["pre_neutral"] for k in C_rows])
    dctrl = np.array([C_rows[k]["ctrl_pre_withheld"] - C_rows[k]["ctrl_pre_neutral"] for k in C_rows])
    summary["readout_c"] = dict(
        mean_n_absential=float(np.mean([C_rows[k]["n_absential"] for k in C_rows])),
        mean_pre_withheld=float(np.mean([C_rows[k]["pre_withheld"] for k in C_rows])),
        mean_pre_neutral=float(np.mean([C_rows[k]["pre_neutral"] for k in C_rows])),
        delta=float(dc.mean()), n_positive=int((dc > 0).sum()), p=sign_p(dc),
        ctrl_delta=float(dctrl.mean()), ctrl_n_positive=int((dctrl > 0).sum()), ctrl_p=sign_p(dctrl),
        mean_gap_withheld=float(np.mean([C_rows[k]["gap_withheld"] for k in C_rows])),
        mean_gap_neutral=float(np.mean([C_rows[k]["gap_neutral"] for k in C_rows])),
    )

out = dict(design=__doc__, args=vars(a), summary=summary, per_item=rows, readout_c=C_rows)
os.makedirs(os.path.dirname(os.path.join(HERE, a.out)) or ".", exist_ok=True)
json.dump(out, open(os.path.join(HERE, a.out) if not os.path.isabs(a.out) else a.out, "w"), indent=1)
print(json.dumps(summary, indent=1)[:4000])
print("saved", a.out)
