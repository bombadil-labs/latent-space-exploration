"""Piece 5: h29's 3x re-imposed era shift, re-run **with the arms it never had**.

Why this exists. The number in `RESULTS.md` -- era reads as target 0.84 at scale 3.0, re-imposed,
Gemma-2-9B-it -- is the evidence for writeup claim 8, and piece 4 verified directly that the battery
behind it has **one arm**: `recompose_sweep_reimpose_3.0.json` contains `shift` rows and nothing
else. The `base` and `rand` controls exist only at scale 1.0, in a different run
(`recompose_gen_gemma9b.json`). `CLAUDE.md`'s first non-negotiable -- every battery reports
treatment, random AND no-patch -- is not met at the scale the headline is quoted at, so the core
refuses to publish it (`MissingArm`). This module runs the missing arms.

What it changes from `scripts/ndif_recompose_sweep.py` (frozen, and correct as far as it goes):

  * three conditions at ONE scale, not one condition at five scales;
  * every remote forward through `lsx.core.remote`'s asserted path, so the §7 assertions -- padding
    side read back off the remote tokenizer, block output resolved by type and never by `output[0]`,
    batched-vs-single equivalence on the shortest item of every scoring batch, non-empty spans --
    guard h29's own scoring code for the first time. The frozen script's scoring reads
    `B[read].output[0][...]`, which is the h36 idiom, saved only by each `tracer.invoke` holding a
    single text;
  * the moved-candidates clause, which no generation job can assert on its own, run explicitly
    against the same patch tensors the generations use, on a real padded batch.

THE ARMS, AND WHERE EACH IS DECLARED TO SIT (pre-registered here, before any score is computed):

  * `shift`   -- treatment. 72 generations, dir_era[e2] - dir_era[e1] at 3.0, every decoding step.
  * `rand`    -- matched-norm random direction, same scale, same re-imposition, one fresh direction
                 per row. Declared null: **the no-patch base rate**, NOT 1/3. A direction carrying
                 no era content should leave the continuation's era where the unpatched model puts
                 it, and the unpatched model does not put it at chance -- it puts it on e1. h27
                 measured this arm at scale 1.0: era-as-target 0.156 (era_stayed 0.688).
  * `base`    -- no patch at all. Declared null: **0.1111**, which is h27's own measured base rate
                 (era_stayed 0.778, so era_other 0.222 split over the two non-e1 targets). That is
                 an independent prior measurement of the identical condition -- base has no scale --
                 so this arm is a real test of whether today's deployment reproduces it, not a
                 restatement of itself.

Declaring these two arms at the registry's 1/k would have been the easy move and it would have been
wrong: 1/k is where an *uninformative* readout sits, and an unpatched continuation is not
uninformative about its own era. An arm handed a null it cannot sit at is a false `ArmOffNull`,
which is piece 3's config-dependent-key bug in a different costume.

A structural identity is recorded alongside, because it is free and it catches a different failure:
for any arm BLIND to the target label e2, and with both non-e1 targets present per passage,
era-as-target is exactly leaves-e1 / 2. If a control arm exceeds that, the arm has seen e2.

Run:
    .venv312/bin/python -m lsx.core.rerun_h29 --phase gen
    .venv312/bin/python -m lsx.core.rerun_h29 --phase score
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
MAIN = pathlib.Path("/home/user/latent-space-exploration")

ERA_WORDS = {
    "medieval": "sword abbey lord horse monk tithe steward castle knight priest".split(),
    "1920s": "telegram automobile jazz radio motorcar gramophone cable tram cigarette typewriter".split(),
    "farfuture": "starship orbit drone habitat colony reactor airlock module relay cryo".split(),
}
INSTR = ("Continue this story passage. Write the next two or three sentences of the story itself, "
         "in the same voice. Output only the continuation, with no commentary, heading or preamble.")


def _find(rel: str) -> pathlib.Path:
    for root in (REPO, MAIN):
        if (root / rel).exists():
            return root / rel
    raise FileNotFoundError(rel)


def level_dirs(X, g, layer: int, train):
    """Leave-one-scene-out era/theme directions, verbatim h27/h29 arithmetic."""
    E, T = g["factors"]["era"], g["factors"]["theme"]
    key = lambda s, e, t: f"{s}/{e}/{t}"
    allv = np.stack([X[key(s, e, t)][layer] for s in train for e in E for t in T])
    mu = allv.mean(0)
    de = {e: np.mean([X[key(s, e, t)][layer] for s in train for t in T], axis=0) - mu for e in E}
    dt = {t: np.mean([X[key(s, e, t)][layer] for s in train for e in E], axis=0) - mu for t in T}
    return de, dt, mu


def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def lex(cont: str):
    c = {e: sum(len(re.findall(rf"\b{w}s?\b", cont, flags=re.I)) for w in ws)
         for e, ws in ERA_WORDS.items()}
    best = max(c.values())
    win = [e for e, v in c.items() if v == best]
    return c, (win[0] if best > 0 and len(win) == 1 else "none")


def rand_dir(shift_vec: np.ndarray, seed: int) -> np.ndarray:
    """A random direction with the SAME NORM as the shift it replaces (spec: matched-norm)."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(shift_vec.shape).astype(np.float32)
    return (g / np.linalg.norm(g)) * float(np.linalg.norm(shift_vec))


def row_seed(scene: str, e1: str, t: str, e2: str) -> int:
    import hashlib
    return int(hashlib.sha256(f"{scene}/{e1}/{t}/{e2}".encode()).hexdigest()[:8], 16)


def rid(r: dict):
    return (r["scene"], r["e1"], r["t"], r["cond"], r["e2"])


def main() -> None:
    from . import remote as rem

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="prompts/narrative_theme_v1.json")
    ap.add_argument("--stacks", default="results/stacks_gemma_2_9b_it_narrative_theme_v1.npz")
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--read", type=int, default=20)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--phase", default="both",
                    choices=["gen", "score", "both", "control", "stack"])
    ap.add_argument("--out", default="results/h29_arms_reimpose3.0.json")
    a = ap.parse_args()

    g = json.loads(_find(a.grid).read_text())
    z = np.load(_find(a.stacks))
    X = {k: z[k] for k in z.files}
    E, T = g["factors"]["era"], g["factors"]["theme"]
    S, lead, spans = g["scenes"], g["lead"], g["spans"]
    key = lambda s, e, t: f"{s}/{e}/{t}"

    out_path = (REPO / a.out)
    ckpt = out_path.with_suffix(".json.partial")
    rows = json.loads(ckpt.read_text()) if ckpt.exists() else []
    controls = {}
    ctl_path = out_path.with_name(out_path.stem + "_controls.json")
    if ctl_path.exists():
        controls = json.loads(ctl_path.read_text())
    print(f"resuming with {len(rows)} rows", flush=True)

    rlm = rem.RemoteLM(a.model)
    print(f"padding side read back from the remote tokenizer: {rlm.padding_side!r}", flush=True)

    def prompt_for(span):
        return rlm.tok.apply_chat_template(
            [{"role": "user", "content": f"{INSTR}\n\n{span}"}],
            tokenize=False, add_generation_prompt=True)

    # ---------------------------------------------------------------- the moved-candidates control
    if a.phase in ("control", "both", "gen") and "patch_reaches_batch" not in controls:
        train = [x for x in S if x != S[0]]
        deP, _, _ = level_dirs(X, g, a.layer, train)
        v = deP[E[1]] - deP[E[0]]
        texts = [f"{lead} {spans[key(s, E[0], T[0])]}" for s in S[:4]]
        rec = {}
        rec["n_texts"] = len(texts)
        rec["moved_shift"] = rem.assert_patch_reaches_batch(rlm, texts, a.layer, v, scale=a.scale)
        rec["moved_rand"] = rem.assert_patch_reaches_batch(
            rlm, texts, a.layer, rand_dir(v, 1), scale=a.scale)
        # negative control: the h36 idiom on the same patch must FAIL the same assertion
        try:
            rem.assert_patch_reaches_batch(rlm, texts, a.layer, v, scale=a.scale,
                                           batch_row_bug=True)
            rec["batch_row_bug"] = {"caught": False}
        except Exception as e:  # noqa: BLE001
            rec["batch_row_bug"] = {"caught": True, "mechanism": type(e).__name__,
                                    "message": str(e)[:200]}
        rec["padding_side"] = rlm.padding_side
        rec["lib_versions"] = rlm.lib_versions()
        controls["patch_reaches_batch"] = rec
        ctl_path.write_text(json.dumps(controls, indent=1))
        print("control:", json.dumps(rec)[:400], flush=True)

    # ---------------------------------------------------------------- phase 1: generate
    if a.phase in ("gen", "both"):
        done = {rid(r) for r in rows}
        t0 = time.time()
        n = 0
        for s in S:
            train = [x for x in S if x != s]
            deP, _, _ = level_dirs(X, g, a.layer, train)
            for e1 in E:
                for t in T:
                    text = prompt_for(spans[key(s, e1, t)])
                    todo = [dict(scene=s, e1=e1, t=t, cond="base", e2="-", vec=None)]
                    for e2 in [x for x in E if x != e1]:
                        v = deP[e2] - deP[e1]
                        todo.append(dict(scene=s, e1=e1, t=t, cond="shift", e2=e2, vec=v))
                        todo.append(dict(scene=s, e1=e1, t=t, cond="rand", e2=e2,
                                         vec=rand_dir(v, row_seed(s, e1, t, e2))))
                    for spec in todo:
                        v = spec.pop("vec")
                        if rid(spec) in done:
                            continue
                        try:
                            spec["cont"] = rem.asserted_remote_generate(
                                rlm, text, max_new_tokens=a.tokens,
                                patch_layer=None if v is None else a.layer,
                                patch_vec=v, scale=a.scale, reimpose=True)
                        except Exception as e:  # noqa: BLE001
                            spec["cont"] = ""
                            spec["failed"] = f"{type(e).__name__}: {e}"[:200]
                        if v is not None:
                            spec["patch_norm"] = float(np.linalg.norm(v) * a.scale)
                        rows.append(spec)
                        n += 1
                        ckpt.write_text(json.dumps(rows))
                        print(f"[gen {n}] {s}/{e1}/{t} {spec['cond']}->{spec['e2']} "
                              f"{time.time()-t0:.0f}s :: {spec['cont'][:60]!r}", flush=True)

    # ------------------------------------------------- phase 1b: the SAME grid through build_stack
    #
    # The patch vectors above come from the cached `.npz` that `scripts/ndif_extract.py` wrote, and
    # a frozen script's output is not a `Stack`: the ledger refuses a row whose provenance carries
    # no `stack_signature` (`ProvenanceNotFromStack`), which is the second of the two refusals
    # piece 4 recorded against h29. So the same grid is extracted again at the same two layers
    # through `remote.build_remote_stack`, with every §7 assertion, and the two are compared. The
    # claim then carries the asserted stack's signed provenance, and the comparison -- not a
    # promise -- is what says the directions behind the generations are that stack's.
    if a.phase in ("stack", "both"):
        from .types import Grid, Item
        items = [Item(text=f"{lead} {spans[key(s, e, t)]}",
                      factors={"scene": s, "era": e, "theme": t},
                      spans={"span": (len(lead) + 1, len(f"{lead} {spans[key(s, e, t)]}"))})
                 for s in S for e in E for t in T]
        grid = Grid(items=items, name="narrative_theme_v1", leak_check=True)
        stacks = {}
        for layer in (a.layer, a.read):
            st = rem.build_remote_stack(rlm, grid, layer, batch_size=4)
            vecs = st.vectors("span", layer)
            X2 = {f"{it.factors['scene']}/{it.factors['era']}/{it.factors['theme']}": v
                  for it, v in zip(items, vecs)}
            # cosine of each cached direction against the asserted stack's, leave-one-scene-out
            cs = []
            for s in S:
                train = [x for x in S if x != s]
                deA, dtA, muA = level_dirs(X, g, layer, train)
                deB = {e: np.mean([X2[key(s2, e, t)] for s2 in train for t in T], axis=0)
                       for e in E}
                muB = np.stack([X2[key(s2, e, t)] for s2 in train for e in E
                                for t in T]).mean(0)
                for e1 in E:
                    for e2 in E:
                        if e1 == e2:
                            continue
                        cs.append(cos(deA[e2] - deA[e1], (deB[e2] - muB) - (deB[e1] - muB)))
            stacks[str(layer)] = {
                "provenance": st.provenance,
                "min_cos_shift_vector_vs_cached_npz": float(np.min(cs)),
                "n_compared": len(cs),
                "batched_vs_single_min_cos": st.provenance["equivalence_min_cos"]}
            np.savez(out_path.with_name(f"h29_core_stack_L{layer}.npz"),
                     **{k: v for k, v in X2.items()})
            print(f"stack L{layer}: min cos vs cached npz "
                  f"{stacks[str(layer)]['min_cos_shift_vector_vs_cached_npz']:.6f}, "
                  f"batched-vs-single {st.provenance['equivalence_min_cos']:.7f}", flush=True)
        controls["asserted_stack"] = stacks
        controls["grid_leak"] = grid.leak.summary()
        ctl_path.write_text(json.dumps(controls, indent=1, default=str))

    # ---------------------------------------------------------------- phase 2: re-read + classify
    if a.phase in ("score", "both"):
        # Classify against the ASSERTED stack's directions when phase `stack` has produced them,
        # so the published number is that stack's and its signature is not merely attached to it.
        core_npz = out_path.with_name(f"h29_core_stack_L{a.read}.npz")
        if core_npz.exists():
            zz = np.load(core_npz)
            Xread = {k: zz[k] for k in zz.files}
            read_source = "asserted build_remote_stack"
        else:
            Xread, read_source = None, "cached npz (scripts/ndif_extract.py)"
        print(f"read-layer directions from: {read_source}", flush=True)
        controls["read_direction_source"] = read_source

        def read_dirs(train):
            if Xread is None:
                de, dt, mu = level_dirs(X, g, a.read, train)
                return de, dt, mu
            allv = np.stack([Xread[key(s2, e, t)] for s2 in train for e in E for t in T])
            mu = allv.mean(0)
            de = {e: np.mean([Xread[key(s2, e, t)] for s2 in train for t in T], axis=0) - mu
                  for e in E}
            dt = {t: np.mean([Xread[key(s2, e, t)] for s2 in train for e in E], axis=0) - mu
                  for t in T}
            return de, dt, mu

        pooled_store = {}
        pooled_path = out_path.with_name("h29_arms_pooled.npz")
        if pooled_path.exists():
            zp = np.load(pooled_path)
            pooled_store = {k: zp[k] for k in zp.files}
        todo = [r for r in rows if "era_read" not in r and r.get("cont", "").strip()]
        print(f"scoring {len(todo)} continuations, {a.batch} per job", flush=True)
        t0 = time.time()
        eq = []
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            texts = [f"{lead} {r['cont'].strip()}" for r in chunk]
            n_tail = []
            for x in texts:
                offs = rlm.tok(x, return_offsets_mapping=True)["offset_mapping"]
                s0 = len(lead) + 1
                n_tail.append(max(1, len([1 for (u, w) in offs if w > u and w > s0])))
            try:
                vs, info = rem.asserted_remote_tail_pool(rlm, texts, a.read, n_tail)
            except Exception as e:  # noqa: BLE001
                print(f"[score {i}] FAILED {type(e).__name__}: {e}"[:300], flush=True)
                continue
            eq.append(info)
            for r, v in zip(chunk, vs):
                train = [x for x in S if x != r["scene"]]
                deR, dtR, muR = read_dirs(train)
                u = v - muR
                r["era_read"] = max(E, key=lambda e: cos(u, deR[e]))
                r["theme_read"] = max(T, key=lambda tt: cos(u, dtR[tt]))
                pooled_store["|".join(rid(r))] = np.asarray(v, dtype=np.float32)
            np.savez(pooled_path, **pooled_store)
            ckpt.write_text(json.dumps(rows))
            print(f"[score {i+len(chunk)}/{len(todo)}] {time.time()-t0:.0f}s "
                  f"eqcos={info['equivalence_cos']}", flush=True)
        controls["scoring_equivalence"] = {
            "n_batches": len(eq),
            "min_cos": min([e["equivalence_cos"] for e in eq if e["equivalence_cos"]], default=None),
            "rule": "shortest item in each batch"}
        ctl_path.write_text(json.dumps(controls, indent=1))

    for r in rows:
        r["lex_counts"], r["lex_era"] = lex(r.get("cont", ""))

    meta = dict(model=a.model, grid=a.grid, stacks=a.stacks,
                patch_layer=a.layer, read_layer=a.read, scale=a.scale, reimpose=True,
                tokens=a.tokens, prompt_format="chat", instruction=INSTR, era_words=ERA_WORDS,
                padding_side=rlm.padding_side, lib_versions=rlm.lib_versions(),
                code_version=rem._code_version(),
                conditions=["base", "rand", "shift"],
                run="piece 5 -- the three arms at one scale")
    out_path.write_text(json.dumps(dict(meta=meta, controls=controls, rows=rows), indent=1))
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
