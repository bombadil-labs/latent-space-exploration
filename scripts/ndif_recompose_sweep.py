"""Hour 29: does the era-shift patch cross the Gauge/Engine boundary at a bigger scale, or if it is
re-imposed at every decoding step, on google/gemma-2-9b-it (the model where hour 27 found the largest,
though still mostly-undirected, effect: era read as target 0.27, lexical check 0.00, scale 1.0,
patch@14/read@20)?

Hour 27 (`results/notes/recompose_gen.md`, `scripts/ndif_recompose_gen.py`) already re-imposes the
patch at every generated token (`with tracer.all():`, same pattern as `scripts/ndif_generate.py`) --
its scale-1.0 "shift" run IS a re-imposition run. So the two conditions below are not "prefix patch
vs re-imposition" as a clean binary; they are:
  (A) SCALE, same mechanism as hour 27 (re-imposed at every step) -- scales 0.5, 2.0, 3.0, reusing
      hour 27's scale-1.0 numbers as the anchor, exactly as instructed.
  (B) MECHANISM at fixed scale -- a patch applied ONLY while the model reads the prefix (once, at
      the prompt's forward pass, iteration 0 of generation -- omit `tracer.all()` so the intervention
      does not repeat on later iterations) vs re-imposed at every step, at scales 1.0 and 2.0. The
      1.0/re-imposed cell is, again, exactly hour 27's run and is reused rather than re-submitted.
This script actually runs: prefix-only@{1.0,2.0} and reimposed@{0.5,2.0,3.0}. Combined with hour 27's
reimposed@1.0, that covers both requested crossings without resubmitting a condition NDIF has already
answered. Discrepancy recorded before running, not discovered after.

================================ PRE-REGISTERED DESIGN ================================
Model. google/gemma-2-9b-it only (42 blocks, d=3584), patch@14 read@20 -- the hour-27 pair, kept for
comparability. (Hour 27 also ran Llama-3.1-70B-Instruct; that model showed almost no displacement
under reimposed scale 1.0 (era read as target 0.14) and is out of scope for this budget -- if the 9B
crosses the boundary at higher scale/reimposition and the 70B was never tested there, that is a
standing caveat, not a result.)

Grid, directions, prompt format, lexical word lists: identical to `scripts/ndif_recompose_gen.py` and
hour 27 (leave-one-scene-out era/theme directions at patch/read layers; chat-templated "continue this
story" instruction; era word lists fixed there, reused verbatim).

Conditions run here (each a separate NDIF job set, checkpointed, one JSON per condition):
  reimpose_0.5   : dir_era[e2]-dir_era[e1] x0.5, added at EVERY generation step (tracer.all())
  reimpose_2.0   : ... x2.0, every step
  reimpose_3.0   : ... x3.0, every step
  prefix_1.0     : ... x1.0, added ONLY at the prefix's forward pass (no tracer.all() -- an
                   intervention outside tracer.iter/tracer.all applies to iteration 0 only, verified
                   against nnsight's IteratorProxy/Tracer.next semantics before running)
  prefix_2.0     : ... x2.0, prefix only
Baseline reused, not re-run (identical mechanism, same NDIF answer already in hand):
  reimpose_1.0   = hour 27's "shift" row, results/recompose_gen_gemma9b.json

Budget. Only the "shift" arm (2 target eras x 36 passages = 72 generations) is run per condition --
the 72-generation fallback pre-registered in the task brief, invoked because 5 new conditions x 144
would be 720 generations plus 720/6=120 scoring jobs. "base" and "rand" controls are NOT re-run per
condition: "base" has no patch and cannot depend on scale or reimposition, so hour 27's base row
(n=36, era stayed 0.78, theme kept 0.53, lex stayed 1.00) is the reference for every condition below;
"rand" is scale-dependent (its norm tracks the shift norm) so hour 27's rand row is only an
approximate reference at other scales, flagged as such in the write-up rather than re-run.

Readouts (identical scoring pipeline to hour 27: continuation re-read unpatched with the grid's lead,
readout-layer residual mean-pooled over continuation tokens, grand mean subtracted, nearest-cosine
classification against leave-one-scene-out era/theme directions):
  era read as target (e2)  |  leaves e1 (era_read != e1)  |  theme kept (t)  |  lexical era check
    (pre-registered word-list vote, moved = matches e2 among continuations with any era-word hit)

Predictions (recorded before running, graded after):
  P1 -- scale: 2x-3x re-imposed will raise "era reads as target" above 0.5, at a cost to prose
        (visible in the verbatim examples as broken syntax, repetition, or loss of narrative coherence).
  P2 -- mechanism: re-imposition (already true of hour 27's scale-1.0 run) is what will be shown, by
        comparison with the new prefix-only@1.0 condition, to be necessary for any lexical crossing --
        i.e. I expect prefix-only@1.0 to replicate hour 27's lexical null (0.00) while reimpose@1.0
        (hour 27 itself) is the one candidate for lex > 0, so P2 restates hour 27's own number as the
        thing to beat and asks whether prefix-only is even weaker.
=======================================================================================

Phases (both by default; each checkpoints and resumes):
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
ap.add_argument("--reimpose", action="store_true", help="patch at every generated token (tracer.all()); default: prefix only")
ap.add_argument("--tokens", type=int, default=48)
ap.add_argument("--phase", default="both", choices=["gen", "score", "both"])
ap.add_argument("--out", required=True)
a = ap.parse_args()

g = json.load(open(a.grid))
z = np.load(a.stacks); X = {k: z[k] for k in z.files}
E, T = g["factors"]["era"], g["factors"]["theme"]
S, lead, spans = g["scenes"], g["lead"], g["spans"]
key = lambda s, e, t: f"{s}/{e}/{t}"

model = LanguageModel(a.model, device_map="auto", dispatch=False)
tok = model.tokenizer
tok.padding_side = "left"
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

INSTR = ("Continue this story passage. Write the next two or three sentences of the story itself, "
         "in the same voice. Output only the continuation, with no commentary, heading or preamble.")


def prompt_for(span):
    return tok.apply_chat_template([{"role": "user", "content": f"{INSTR}\n\n{span}"}],
                                   tokenize=False, add_generation_prompt=True)


# ------------------------------------------------------------------ phase 1: generate
def _inner_gen(prompt, vec):
    v = None if vec is None else torch.as_tensor(vec * a.scale, dtype=torch.float32)
    backend = ProxyAuthBackend(model.to_model_key())
    with model.generate(prompt, max_new_tokens=a.tokens, do_sample=False, backend=backend) as tracer:
        if v is not None:
            if a.reimpose:
                with tracer.all():
                    B[a.layer].output[0][:] = B[a.layer].output[0] + v.to(B[a.layer].output[0])
            else:
                # no tracer.all()/tracer.iter wrapper -> applies to iteration 0 (the prefix forward
                # pass) only; later generated-token iterations are untouched.
                B[a.layer].output[0][:] = B[a.layer].output[0] + v.to(B[a.layer].output[0])
        out = model.generator.output.save()
    res = backend.wait(tracer)
    ts = [x for x in res.values() if isinstance(x, torch.Tensor)] if isinstance(res, dict) else []
    if not ts: raise TimeoutError("empty NDIF result for generate")
    o = res["out"] if isinstance(res, dict) and "out" in res else ts[0]
    ids = o[0] if o.dim() == 2 else o
    n_in = len(tok(prompt)["input_ids"])
    return tok.decode(ids[n_in:], skip_special_tokens=True)


generate = lambda *x: retry_job(lambda: _inner_gen(*x), attempts=3, wait_s=3.0)

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
                for e2, vec in shifts.items():
                    row = dict(scene=s, e1=e1, t=t, cond="shift", e2=e2)
                    if rid(row) in done: continue
                    try:
                        row["cont"] = generate(text, vec)
                    except TimeoutError as e:
                        row["cont"] = ""; row["failed"] = str(e)
                    rows.append(row); n += 1
                    json.dump(rows, open(ckpt, "w"))
                    print(f"[gen {n}] {s}/{e1}/{t} shift->{e2} {time.time()-t0:.0f}s :: {row['cont'][:70]!r}", flush=True)

# ------------------------------------------------------------------ phase 2: re-read + classify
def _inner_read(texts):
    sel = []
    for x in texts:
        enc = tok(x, return_offsets_mapping=True); offs = enc["offset_mapping"]; s0 = len(lead) + 1
        idx = [j for j, (u, w) in enumerate(offs) if w > u and w > s0]
        sel.append(max(1, len(idx)))
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
    print(f"scoring {len(todo)} continuations, 6 per job", flush=True)
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

xs = [r for r in rows if r["cond"] == "shift" and "era_read" in r]
lx = [r for r in xs if r["lex_era"] != "none"]
summary = dict(
    n=len(xs),
    era_target=float(np.mean([r["era_read"] == r["e2"] for r in xs])) if xs else None,
    leaves_e1=float(np.mean([r["era_read"] != r["e1"] for r in xs])) if xs else None,
    theme_kept=float(np.mean([r["theme_read"] == r["t"] for r in xs])) if xs else None,
    lex_n=len(lx),
    lex_target=float(np.mean([r["lex_era"] == r["e2"] for r in lx])) if lx else None,
)

meta = dict(model=a.model, grid=a.grid, stacks=a.stacks, n_blocks=NL, d=D,
            patch_layer=a.layer, read_layer=a.read, scale=a.scale, reimpose=a.reimpose,
            tokens=a.tokens, prompt_format="chat", instruction=INSTR, era_words=ERA_WORDS)
json.dump(dict(meta=meta, summary=summary, rows=rows), open(a.out, "w"), indent=1)
print(f"\n=== {a.model} patch@{a.layer} read@{a.read} scale={a.scale} reimpose={a.reimpose} tokens={a.tokens} ===")
print(summary)
print("wrote", a.out)
