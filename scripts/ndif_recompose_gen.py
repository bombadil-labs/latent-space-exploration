"""Recomposition at the GENERATION level: does an era shift move the address and keep the form
in text the engine actually writes?

Hour 14 showed, representationally, that adding dir_era[e2] - dir_era[e1] while the model READS a
passage moves its era readout (0.89 Qwen / 0.88 Gemma) and leaves its theme readout untouched
(0.81 / 0.94, identical to unpatched). That is a Gauge result. Law L3 of docs/ALGEBRA.md says
Gauges compose and Engines do not, so the Gauge result does not transfer for free. This script puts
the Engine in the loop.

================================ PRE-REGISTERED DESIGN ================================
Written before the run; not edited afterwards.

Models. meta-llama/Llama-3.1-70B-Instruct (80 blocks, d=8192; smoke-tested through NDIF) and
google/gemma-2-9b-it (42 blocks, d=3584) as the comparison. Patch layer ~1/3 depth, readout layer
~1/2 depth: Llama 26 / 40; Gemma 14 / 20 (the hour-14 pair, kept for comparability).

Grid. prompts/narrative_theme_v1.json: 4 scenes x 3 eras x 3 themes = 36 three-sentence passages,
lead "A passage from a story:".

Prompt format (fixed by a pilot BEFORE the measured run, recorded here for honesty). The first
attempt fed the raw "<lead> <span>" string, as hour 13's raw-continuation condition did. Both
models are instruction-tuned and both answered it as a *comprehension exercise* ("**What is the
story likely about?** This passage suggests a story about: * Betrayal and family conflict..."),
i.e. not narrative text at all, so nothing could be scored. The measured run therefore uses the
chat template with the instruction "Continue this story passage. Write the next two or three
sentences of the story itself, in the same voice. Output only the continuation, with no commentary,
heading or preamble." -- hour 13's chat-templated condition, which produced fluent prose on Gemma.
Under it both models write story text (pilot checked on 3 passages per model). The patch is added
at every position of the templated prompt and of every generated token.

Directions. Leave-one-SCENE-out. For held-out scene s, train = the other three scenes. At layer l,
mu(l) = mean over all train spans; dir_era[e](l) = mean over train spans with era e, minus mu(l);
dir_theme[t](l) likewise. Patch directions come from the patch layer, classification directions and
mu from the readout layer. Exactly as scripts/ndif_shift.py builds them.

Generation test. For each of the 36 passages (era e1, theme t), greedy 48-token continuation of
"<lead> <span>", with the patch added to the patch block's output at EVERY position including
generated ones (`with tracer.all():`, as in scripts/ndif_generate.py), under four conditions:
  (a) base   -- no patch                                                      (1 per passage)
  (b) shift  -- dir_era[e2] - dir_era[e1] at the patch layer, for each e2!=e1 (2 per passage)
  (c) rand   -- a random Gaussian direction rescaled to the mean norm of this
                passage's two shift vectors                                   (1 per passage)
36 x 4 = 144 generations per model at scale 1.0. Scale 1.5 for the shift condition if budget allows.

Scoring the continuation. The continuation ALONE is re-read, unpatched, prefixed with the grid's
lead: "<lead> <continuation>". So the score is of the new text, not of the patched context. Fetch
readout-layer residuals over the continuation tokens only (the suffix after the lead), mean-pool,
subtract the training-fold grand mean mu(readout), classify by nearest (cosine) era direction and
nearest theme direction. Re-reads are batched six to a job via tracer.invoke (verified to give
bit-identical vectors to one-per-job).

Reported per model and condition:
  era of the continuation reads as e2 (MOVED) / e1 (STAYED) / the third era (OTHER);
  theme of the continuation reads as t (KEPT).
The comparison is (b) against (a) -- which cannot move, its e2 is undefined, so for (a) we report
"reads as e1" and the theme -- and against (c), which has the same norm but no era content.

Lexical sanity check, word lists fixed in advance (case-insensitive, word-boundary, optional
plural -s):
  medieval  : sword abbey lord horse monk tithe steward castle knight priest
  1920s     : telegram automobile jazz radio motorcar gramophone cable tram cigarette typewriter
  farfuture : starship orbit drone habitat colony reactor airlock module relay cryo
For each continuation, take the era whose list has the most hits (ties and all-zero -> "none") and
report the same moved/stayed/other split over the continuations that have any hit at all.

Predictions (recorded before running). The Gauge->Engine gap of L3 should cost something: I expect
the moved rate under (b) to be clearly above (a)/(c) but below the 0.88 of hour 14, and the theme-kept
rate to be at or slightly below the base condition's. I expect the 70B to move more cleanly than the
9B, by the hour-13 steering-competence curve (theme steers generation only at 9B-instruct, and
decodability rises with scale).
=======================================================================================

Phases (both run by default; each checkpoints and resumes):
  --phase gen    generate and checkpoint continuations (1 NDIF job each)
  --phase score  re-read + classify the checkpointed continuations (6 per NDIF job)
"""
import argparse, json, os, re, time, zlib
import numpy as np, torch
from nnsight import LanguageModel
from lsx.ndif import ProxyAuthBackend, retry_job

ERA_WORDS = {
    "medieval": "sword abbey lord horse monk tithe steward castle knight priest".split(),
    "1920s": "telegram automobile jazz radio motorcar gramophone cable tram cigarette typewriter".split(),
    "farfuture": "starship orbit drone habitat colony reactor airlock module relay cryo".split(),
}

ap = argparse.ArgumentParser()
ap.add_argument("grid", nargs="?", default="prompts/narrative_theme_v1.json")
ap.add_argument("stacks")
ap.add_argument("--model", default="google/gemma-2-9b-it")
ap.add_argument("--layer", type=int, default=14, help="patch layer (~1/3 depth)")
ap.add_argument("--read", type=int, default=20, help="readout layer (~1/2 depth)")
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--tokens", type=int, default=48)
ap.add_argument("--batch", type=int, default=6, help="continuations per re-read job (must be 6)")
ap.add_argument("--phase", default="both", choices=["gen", "score", "both"])
ap.add_argument("--raw", action="store_true", help="feed the bare passage instead of the chat template")
ap.add_argument("--out", required=True)
a = ap.parse_args()

g = json.load(open(a.grid))
z = np.load(a.stacks); X = {k: z[k] for k in z.files}
E, T = g["factors"]["era"], g["factors"]["theme"]
S, lead, spans = g["scenes"], g["lead"], g["spans"]
key = lambda s, e, t: f"{s}/{e}/{t}"

model = LanguageModel(a.model, device_map="auto", dispatch=False)
tok = model.tokenizer
tok.padding_side = "left"          # batched invokes pad on the left, so suffix indices stay valid
if tok.pad_token is None: tok.pad_token = tok.eos_token


def blocks(m):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = m
        try:
            for x in path.split("."): obj = getattr(obj, x)
            return obj
        except AttributeError: continue
    raise ValueError("no blocks")


B = blocks(model); D = model.config.hidden_size; NL = len(B)
assert a.layer < NL and a.read < NL, f"{a.model} has {NL} blocks"


def dirs(l, train):
    allv = np.stack([X[key(s, e, t)][l] for s in train for e in E for t in T]); mu = allv.mean(0)
    de = {e: np.mean([X[key(s, e, t)][l] for s in train for t in T], axis=0) - mu for e in E}
    dt = {t: np.mean([X[key(s, e, t)][l] for s in train for e in E], axis=0) - mu for t in T}
    return de, dt, mu


cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
ckpt = a.out + ".partial"
rows = []
if os.path.exists(ckpt):
    rows = json.load(open(ckpt)); print("resuming", len(rows), "rows", flush=True)
rid = lambda r: (r["scene"], r["e1"], r["t"], r["cond"], r["e2"])

# ------------------------------------------------------------------ phase 1: generate
def _inner_gen(prompt, vec):
    v = None if vec is None else torch.as_tensor(vec * a.scale, dtype=torch.float32)
    backend = ProxyAuthBackend(model.to_model_key())
    with model.generate(prompt, max_new_tokens=a.tokens, do_sample=False, backend=backend) as tracer:
        if v is not None:
            with tracer.all():
                B[a.layer].output[0][:] = B[a.layer].output[0] + v.to(B[a.layer].output[0])
        out = model.generator.output.save()
    res = backend.wait(tracer)
    ts = [x for x in res.values() if isinstance(x, torch.Tensor)] if isinstance(res, dict) else []
    # NDIF occasionally returns a COMPLETED job with an empty payload; TimeoutError is what retry_job retries on
    if not ts: raise TimeoutError("empty NDIF result for generate")
    o = res["out"] if isinstance(res, dict) and "out" in res else ts[0]
    ids = o[0] if o.dim() == 2 else o
    n_in = len(tok(prompt)["input_ids"])
    return tok.decode(ids[n_in:], skip_special_tokens=True)


generate = lambda *x: retry_job(lambda: _inner_gen(*x), attempts=3, wait_s=3.0)

INSTR = ("Continue this story passage. Write the next two or three sentences of the story itself, "
         "in the same voice. Output only the continuation, with no commentary, heading or preamble.")


def prompt_for(span):
    if a.raw: return f"{lead} {span}"
    return tok.apply_chat_template([{"role": "user", "content": f"{INSTR}\n\n{span}"}],
                                   tokenize=False, add_generation_prompt=True)

if a.phase in ("gen", "both"):
    done = {rid(r) for r in rows}
    t0 = time.time(); n = 0
    for s in S:
        train = [x for x in S if x != s]
        deP, _, _ = dirs(a.layer, train)
        for e1 in E:
            for t in T:
                text = prompt_for(spans[key(s, e1, t)])
                shifts = {e2: deP[e2] - deP[e1] for e2 in E if e2 != e1}
                rng = np.random.default_rng(zlib.crc32(f"{s}/{e1}/{t}".encode()))   # stable across runs
                r = rng.normal(size=(D,))
                r *= float(np.mean([np.linalg.norm(v) for v in shifts.values()])) / np.linalg.norm(r)
                conds = [("base", "-", None), ("rand", "-", r)] + [("shift", e2, v) for e2, v in shifts.items()]
                for cond, e2, vec in conds:
                    row = dict(scene=s, e1=e1, t=t, cond=cond, e2=e2)
                    if rid(row) in done: continue
                    try:
                        row["cont"] = generate(text, vec)
                    except TimeoutError as e:   # NDIF returned a COMPLETED-but-empty payload every attempt
                        row["cont"] = ""; row["failed"] = str(e)
                    rows.append(row); n += 1
                    json.dump(rows, open(ckpt, "w"))
                    print(f"[gen {n}] {s}/{e1}/{t} {cond}->{e2} {time.time()-t0:.0f}s :: {row['cont'][:70]!r}", flush=True)

# ------------------------------------------------------------------ phase 2: re-read + classify
def _inner_read(texts):
    """Mean-pool the readout-layer residual over each text's continuation tokens. 6 texts per job."""
    sel = []
    for x in texts:
        enc = tok(x, return_offsets_mapping=True); offs = enc["offset_mapping"]; s0 = len(lead) + 1
        idx = [j for j, (u, w) in enumerate(offs) if w > u and w > s0]
        sel.append(max(1, len(idx)))          # continuation tokens are a suffix -> take the last k
    backend = ProxyAuthBackend(model.to_model_key())
    with model.trace(backend=backend) as tracer:
        with tracer.invoke(texts[0]):
            h0 = B[a.read].output[0][..., -sel[0]:, :].mean(-2).reshape(-1, D)[-1].save()
        with tracer.invoke(texts[1]):
            h1 = B[a.read].output[0][..., -sel[1]:, :].mean(-2).reshape(-1, D)[-1].save()
        with tracer.invoke(texts[2]):
            h2 = B[a.read].output[0][..., -sel[2]:, :].mean(-2).reshape(-1, D)[-1].save()
        with tracer.invoke(texts[3]):
            h3 = B[a.read].output[0][..., -sel[3]:, :].mean(-2).reshape(-1, D)[-1].save()
        with tracer.invoke(texts[4]):
            h4 = B[a.read].output[0][..., -sel[4]:, :].mean(-2).reshape(-1, D)[-1].save()
        with tracer.invoke(texts[5]):
            h5 = B[a.read].output[0][..., -sel[5]:, :].mean(-2).reshape(-1, D)[-1].save()
    res = backend.wait(tracer)
    if not isinstance(res, dict) or any(k not in res for k in ("h0", "h1", "h2", "h3", "h4", "h5")):
        raise TimeoutError("empty NDIF result for batched read")
    return [res[k].float().cpu().numpy() for k in ("h0", "h1", "h2", "h3", "h4", "h5")]


read6 = lambda texts: retry_job(lambda: _inner_read(texts), attempts=5)

if a.phase in ("score", "both"):
    todo = [r for r in rows if "era_read" not in r and r.get("cont", "").strip()]
    print(f"scoring {len(todo)} continuations, {a.batch} per job", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), 6):
        chunk = todo[i:i + 6]
        texts = [f"{lead} {r['cont'].strip()}" for r in chunk]
        pad = texts + [texts[-1]] * (6 - len(texts))
        vs = read6(pad)
        for r, v in zip(chunk, vs):
            train = [x for x in S if x != r["scene"]]
            deR, dtR, muR = dirs(a.read, train)
            u = v - muR
            r["era_read"] = max(E, key=lambda e: cos(u, deR[e]))
            r["theme_read"] = max(T, key=lambda tt: cos(u, dtR[tt]))
        json.dump(rows, open(ckpt, "w"))
        print(f"[score {i+len(chunk)}/{len(todo)}] {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------ analysis
def lex(cont):
    c = {}
    for e, ws in ERA_WORDS.items():
        c[e] = sum(len(re.findall(rf"\b{w}s?\b", cont, flags=re.I)) for w in ws)
    best = max(c.values())
    win = [e for e, v in c.items() if v == best]
    return c, (win[0] if best > 0 and len(win) == 1 else "none")


for r in rows:
    r["lex_counts"], r["lex_era"] = lex(r.get("cont", ""))

summary = {}
for cond in ("base", "shift", "rand"):
    xs = [r for r in rows if r["cond"] == cond and "era_read" in r]
    if not xs: continue
    lx = [r for r in xs if r["lex_era"] != "none"]
    summary[cond] = dict(
        n=len(xs),
        era_moved=float(np.mean([r["era_read"] == r["e2"] for r in xs])) if cond != "base" else None,
        era_stayed=float(np.mean([r["era_read"] == r["e1"] for r in xs])),
        era_other=float(np.mean([r["era_read"] not in (r["e1"], r["e2"]) for r in xs])),
        theme_kept=float(np.mean([r["theme_read"] == r["t"] for r in xs])),
        lex_n=len(lx),
        lex_moved=float(np.mean([r["lex_era"] == r["e2"] for r in lx])) if lx and cond != "base" else None,
        lex_stayed=float(np.mean([r["lex_era"] == r["e1"] for r in lx])) if lx else None,
    )

meta = dict(model=a.model, grid=a.grid, stacks=a.stacks, n_blocks=NL, d=D,
            patch_layer=a.layer, read_layer=a.read, scale=a.scale, tokens=a.tokens,
            prompt_format=("raw" if a.raw else "chat"), instruction=INSTR,
            era_words=ERA_WORDS)
json.dump(dict(meta=meta, summary=summary, rows=rows), open(a.out, "w"), indent=1)
print(f"\n=== {a.model}  patch@{a.layer} read@{a.read} scale={a.scale} tokens={a.tokens} ===")
print(f"{'cond':6s} {'n':>4s} {'moved':>7s} {'stayed':>7s} {'other':>7s} {'theme kept':>11s} | {'lex n':>6s} {'lex moved':>10s} {'lex stayed':>11s}")
for c, v in summary.items():
    f = lambda x: "  --  " if x is None else f"{x:.2f}"
    print(f"{c:6s} {v['n']:4d} {f(v['era_moved']):>7s} {f(v['era_stayed']):>7s} {f(v['era_other']):>7s} "
          f"{f(v['theme_kept']):>11s} | {v['lex_n']:6d} {f(v['lex_moved']):>10s} {f(v['lex_stayed']):>11s}")
print("wrote", a.out)
