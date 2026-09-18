"""Piece 5: h39's Gemma clock, re-extracted so it can be reported the way §1A restates it.

§1A's row for h39 is not "reproduce 0.501 / 0.767 / 2.50". It is: *the logged numbers are raw
scores on a grid whose leak check flags 221 of 240 state spans, and the core must not print them
bare -- report gain over the measured stimulus floor.* Piece 3 refused the row for two reasons and
piece 4 closed one of them (`discrimination` is built and calibrated). The other was data: the
Gemma stacks are not cached, and §11.3 said defer rather than re-extract. This module re-extracts.

THE THREE ARMS, and what each one's text is:

  * `exp`   -- treatment. The v2 grid's experimental prompt: interval phrase + the state as it is
               at that timepoint.
  * `ctrl`  -- **the measured stimulus floor**. The grid's own control prompt: the same interval
               phrase with the **t0 state**, unchanged, at every timepoint. So it is exactly "what
               the interval phrase alone gives away", which is the leak h32/h35 measured and the
               thing §6 says the gain must be over. h39's own m7 used these prompts; it reported
               their ratio and then published the raw number anyway.
  * `shuf`  -- `shuffled_stimulus`, which `discrimination` requires and h38 never ran: the
               experimental state span with its words shuffled inside the span, seeded per item.
               Same tokens, same length, no order. h38 extracted this arm and never scored it,
               which is why piece 4's demonstration on h38's cache was refused too.

The statistic is `scripts/time_translation_discrimination.py`'s **shared** predictor, per subject,
which is the per-item form `discrimination` needs and which the logged JSON does not contain (it
carries nine per-Δt aggregates and no per-subject breakdown). For each subject's held-out
(timepoint, paraphrase) cell, the timepoint is predicted by nearest centroid built from **every
other subject's** paraphrases -- the subject is held out entirely -- and the statistic is
Spearman(predicted index, true index). One number per subject, eight subjects, which is the m=8
the arm band is computed at.

Every extraction goes through `remote.build_remote_stack`, so the padding convention, the
batched-vs-single equivalence on the shortest item of every batch, the non-empty spans and the
`resid()` resolution all run -- and the provenance is signed, which is what lets the row reach the
ledger at all.

Run:  .venv312/bin/python -m lsx.core.rerun_h39 --layer 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
MAIN = pathlib.Path("/home/user/latent-space-exploration")


def _find(rel: str) -> pathlib.Path:
    for root in (REPO, MAIN):
        if (root / rel).exists():
            return root / rel
    raise FileNotFoundError(rel)


def shuffle_words(text: str, seed: int) -> str:
    """Same words, same count, no order. Punctuation rides with its token, which is deliberate: the
    arm is 'the bag without the order', not 'the bag without the punctuation'."""
    rng = np.random.default_rng(seed)
    parts = text.split(" ")
    idx = rng.permutation(len(parts))
    return " ".join(parts[i] for i in idx)


def build_items(g: dict, tag: str):
    """(key, text, state span) for one arm of the grid."""
    from ..extract import parse_roles
    S, DT, NP = g["subjects"], g["deltas"], g["n_paraphrases"]
    order = ["t0"] + DT
    src = g["prompts"] if tag in ("exp", "shuf") else g["control_prompts"]
    out = []
    for s in S:
        for t in order:
            for p in range(NP):
                key = f"{s}/{t}/p{p}"
                parsed = parse_roles(src[key])
                text, (a, b) = parsed.text, parsed.spans["state"][0]
                if tag == "shuf":
                    # hashlib, not `hash()`: PYTHONHASHSEED randomises str hashing per process and
                    # a shuffled-stimulus arm that differs between runs is not reproducible.
                    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                    state = shuffle_words(text[a:b], seed=seed)
                    text = text[:a] + state + text[b:]
                    b = a + len(state)
                out.append((f"{tag}/{key}", text, (a, b)))
    return out


def spearman_int(x, y) -> float:
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float(rx @ ry / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def shared_spearman_per_subject(vec: dict, g: dict, tag: str) -> dict:
    """`scripts/time_translation_discrimination.py`'s SHARED predictor, verbatim arithmetic."""
    S, DT, NP = g["subjects"], g["deltas"], g["n_paraphrases"]
    order = ["t0"] + DT
    get = lambda s, t, p: vec[f"{tag}/{s}/{t}/p{p}"]
    per = {}
    for s in S:
        cent = [np.mean([get(o, tj, q) for o in S if o != s for q in range(NP)], axis=0)
                for tj in order]
        pred, true = [], []
        for ti in order:
            for p in range(NP):
                h = get(s, ti, p)
                pred.append(int(np.argmin([np.linalg.norm(h - c) for c in cent])))
                true.append(order.index(ti))
        per[s] = spearman_int(pred, true)
    return per


def main() -> None:
    from . import remote as rem
    from .types import Grid, Item

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="prompts/time_translation_v2.json")
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--tags", default="exp,ctrl,shuf")
    ap.add_argument("--out", default="results/h39_gemma_clock_arms.json")
    a = ap.parse_args()

    g = json.loads(_find(a.grid).read_text())
    out_path = REPO / a.out
    store = out_path.with_name("h39_gemma_clock_arms.npz")
    have = {k: v for k, v in np.load(store).items()} if store.exists() else {}
    meta = json.loads(out_path.read_text())["meta"] if out_path.exists() else {}

    rlm = rem.RemoteLM(a.model)
    print(f"padding side read back: {rlm.padding_side!r}", flush=True)
    t0 = time.time()

    for tag in [x for x in a.tags.split(",") if x]:
        rows = build_items(g, tag)
        todo = [r for r in rows if r[0] not in have]
        print(f"[{tag}] {len(todo)} of {len(rows)} to extract", flush=True)
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            items = [Item(text=t, factors={"key": k}, spans={"state": sp}) for k, t, sp in chunk]
            grid = Grid(items=items, name=f"time_translation_v2/{tag}", leak_check=False)
            try:
                st = rem.build_remote_stack(rlm, grid, a.layer, batch_size=a.batch)
            except Exception as e:  # noqa: BLE001
                print(f"[{tag} {i}] FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            v = st.vectors("state", a.layer)
            for (k, _, _), row in zip(chunk, v):
                have[k] = np.asarray(row, dtype=np.float32)
            meta.setdefault("provenance", {})[tag] = st.provenance
            np.savez(store, **have)
            print(f"[{tag} {i + len(chunk)}/{len(todo)}] {time.time() - t0:.0f}s "
                  f"eq {st.provenance['equivalence_min_cos']:.7f}", flush=True)
        out_path.write_text(json.dumps({"meta": meta}, indent=1, default=str))

    # ---- the statistic, per subject, for every arm that came back whole -----------------------
    measures = {}
    for tag in ("exp", "ctrl", "shuf"):
        keys = [k for k, _, _ in build_items(g, tag)]
        if not all(k in have for k in keys):
            measures[tag] = {"incomplete": sum(1 for k in keys if k not in have), "n": len(keys)}
            continue
        per = shared_spearman_per_subject(have, g, tag)
        measures[tag] = {"per_subject": per, "mean": float(np.mean(list(per.values())))}
        print(f"{tag}: shared spearman mean {measures[tag]['mean']:.4f}  {per}", flush=True)
    meta.update({"model": a.model, "grid": a.grid, "layer": a.layer,
                 "padding_side": rlm.padding_side, "lib_versions": rlm.lib_versions(),
                 "code_version": rem._code_version(),
                 "statistic": "scripts/time_translation_discrimination.py shared predictor, "
                              "per subject, subject held out entirely"})
    out_path.write_text(json.dumps({"meta": meta, "measures": measures}, indent=1, default=str))
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
