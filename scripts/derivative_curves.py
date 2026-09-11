"""Derivative curves: how a passage's projection onto factor directions evolves along the passage.

The "differentiate" operator of VISION.md's narrative calculus, at the representational level.

For every passage in a factor grid we run one forward pass, keep the residual stream at a read
layer for every token of the marked span, and project each token vector onto the factor directions
(theme / era / mood), estimated leave-one-situation-out from the mean-pooled span vectors of the
other situations.  From the per-token projection sequence we derive:

  proj_own(t)   cosine of the (grand-mean-centred) token vector with the passage's own level dir
  margin(t)     proj_own(t) - mean over the two other levels of that factor  (chance 0)
  hit(t)        1 if the own level is the argmax over that factor's levels   (chance 1/3)
  integral(t)   running mean of margin up to t  ("the integral so far")
  sentence means and their beat-to-beat differences  (the discrete derivative)

Baselines: a random direction triple of matched norms, run through the identical pipeline.

Usage (CPU, ~1 forward pass per passage):
    python scripts/derivative_curves.py --out results/derivative_curves.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

from lsx import LM
from lsx.extract import tokens_in_span

N_BINS = 12
ERAS = ["medieval", "1920s", "farfuture"]
THEMES = ["betrayal", "sacrifice", "homecoming"]
MOODS = ["dread", "tender", "comic"]


# ---------------------------------------------------------------- extraction

def passage_tokens(lm: LM, lead: str, span: str, layers: list[int]):
    """Residuals for every token of `span` in the prompt '<lead> <span>'.

    Returns (acts [n_layers, k, d], token char offsets of the span tokens, span char start).
    """
    text = f"{lead} {span}"
    s0 = len(lead) + 1
    hs, offsets = lm.residuals(text)                      # [L+1, seq, d]
    idx = tokens_in_span(offsets, (s0, s0 + len(span)))
    acts = hs[layers][:, idx].numpy()                     # [n_layers, k, d]
    return acts, [offsets[i] for i in idx], s0


def sentence_bounds(span: str) -> list[tuple[int, int]]:
    """Character ranges of the sentences of a passage (split after '. ')."""
    cuts, out, prev = [m.end() for m in re.finditer(r"\.\s+", span)], [], 0
    for c in cuts + [len(span)]:
        if c > prev:
            out.append((prev, c))
        prev = c
    return out


# ---------------------------------------------------------------- directions

def level_dirs(pooled: dict, keyf, levels, train_scenes, other_levels):
    """dir[level] = mean over training scenes and all other-factor levels, minus the grand mean."""
    allv = np.stack([pooled[keyf(s, a, b)] for s in train_scenes for a in levels for b in other_levels])
    mu = allv.mean(0)
    return {a: np.mean([pooled[keyf(s, a, b)] for s in train_scenes for b in other_levels], axis=0) - mu
            for a in levels}, mu


def cos_rows(V: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Cosine of every row of V with d."""
    return (V @ d) / (np.linalg.norm(V, axis=1) * np.linalg.norm(d) + 1e-9)


def curve_stats(V: np.ndarray, dirs: dict, own: str, levels: list[str]):
    """Per-token proj/margin/hit for a [k, d] block of centred token vectors."""
    P = np.stack([cos_rows(V, dirs[a]) for a in levels])          # [n_levels, k]
    i = levels.index(own)
    others = [j for j in range(len(levels)) if j != i]
    return dict(proj=P[i], margin=P[i] - P[others].mean(0), hit=(P.argmax(0) == i).astype(float))


def resample(y: np.ndarray, n: int = N_BINS) -> np.ndarray:
    """Mean of y within n equal bins of normalised position (token centres)."""
    k = len(y)
    pos = (np.arange(k) + 0.5) / k
    b = np.minimum((pos * n).astype(int), n - 1)
    return np.array([y[b == j].mean() if (b == j).any() else np.nan for j in range(n)])


# ---------------------------------------------------------------- grid runner

def run_grid(lm, grid_path, factor_a, factor_b, layers, rng, n_rand=8, sentences=True):
    """Per-token curves for every passage of a grid, for both factors, leave-one-scene-out."""
    g = json.load(open(grid_path))
    A, B = g["factors"][factor_a], g["factors"][factor_b]
    scenes, lead, spans = g["scenes"], g["lead"], g["spans"]
    key = lambda s, a, b: f"{s}/{a}/{b}"

    # one forward pass per passage; keep per-token acts at the requested layers
    acts, offs = {}, {}
    for k, txt in spans.items():
        acts[k], offs[k], _ = passage_tokens(lm, lead, txt, layers)
        print("  extracted", k, acts[k].shape[1], "tokens", flush=True)

    rows = []
    for li, layer in enumerate(layers):
        pooled = {k: acts[k][li].mean(0) for k in acts}
        for s in scenes:
            train = [x for x in scenes if x != s]
            dA, _ = level_dirs(pooled, key, A, train, B)
            dB, _ = level_dirs(pooled, lambda sc, b, a: key(sc, a, b), B, train, A)
            # token-level grand mean over the training passages (centring for per-token cosines)
            tok_mu = np.concatenate([acts[key(t, a, b)][li] for t in train for a in A for b in B]).mean(0)
            nrmA = np.mean([np.linalg.norm(v) for v in dA.values()])
            nrmB = np.mean([np.linalg.norm(v) for v in dB.values()])
            rA = [{a: unit(rng, len(tok_mu)) * nrmA for a in A} for _ in range(n_rand)]
            rB = [{b: unit(rng, len(tok_mu)) * nrmB for b in B} for _ in range(n_rand)]
            for a in A:
                for b in B:
                    k = key(s, a, b)
                    V = acts[k][li] - tok_mu                       # [ntok, d]
                    row = dict(scene=s, layer=layer, key=k, n_tok=int(V.shape[0]),
                               **{factor_a: a, factor_b: b})
                    for nm, dirs, lv, own in ((factor_a, dA, A, a), (factor_b, dB, B, b)):
                        c = curve_stats(V, dirs, own, lv)
                        row[nm + "_curve"] = {q: c[q].tolist() for q in c}
                    for nm, rds, lv, own in ((factor_a, rA, A, a), (factor_b, rB, B, b)):
                        cs = [curve_stats(V, rd, own, lv) for rd in rds]
                        row[nm + "_rand"] = {q: np.mean([c[q] for c in cs], 0).tolist() for q in cs[0]}
                    if sentences:
                        row["sent_tok"] = sentence_token_idx(offs[k], sentence_bounds(spans[k]), lead)
                    rows.append(row)
        print(f"layer {layer} done", flush=True)
    return rows


def unit(rng, d):
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


def sentence_token_idx(offs, sb, lead):
    """Indices (into the span's token list) belonging to each sentence."""
    s0 = len(lead) + 1
    out = []
    for b0, e in sb:
        out.append([i for i, (x, y) in enumerate(offs) if y > x and x < s0 + e and y > s0 + b0])
    return out


# ---------------------------------------------------------------- aggregation

def agg(rows, factor, field="margin", group=None, layer=None, rand=False):
    """Mean binned curve over rows, optionally restricted to one level of `group` / one layer."""
    sel = [r for r in rows if (layer is None or r["layer"] == layer)]
    if group:
        gf, gv = group
        sel = [r for r in sel if r[gf] == gv]
    key = factor + ("_rand" if rand else "_curve")
    C = np.stack([resample(np.array(r[key][field])) for r in sel])
    return C.mean(0), C.std(0) / np.sqrt(len(C)), len(C)


def sent_means(r, factor, field="margin", rand=False):
    y = np.array(r[factor + ("_rand" if rand else "_curve")][field])
    return [float(y[idx].mean()) for idx in r["sent_tok"] if idx]



def summarize(theme_rows, mood_rows, theme_layers, mood_layers):
    """Aggregate curves and sentence-level derivatives into a compact block."""
    out = {}
    nl = lambda x: [None if np.isnan(v) else round(float(v), 5) for v in x]

    def block(rows, factor, levels, layer):
        b = {}
        for lev in list(levels) + ["ALL"]:
            grp = None if lev == "ALL" else (factor, lev)
            for field in ("margin", "hit", "proj"):
                m, se, n = agg(rows, factor, field, grp, layer)
                r, _, _ = agg(rows, factor, field, grp, layer, rand=True)
                b.setdefault(lev, {})[field] = dict(mean=nl(m), se=nl(se), rand=nl(r), n=n)
        return b

    for layer in theme_layers:
        for factor, levels in (("theme", THEMES), ("era", ERAS)):
            out[f"theme_grid/{factor}/L{layer}"] = block(theme_rows, factor, levels, layer)
    for layer in mood_layers:
        for factor, levels in (("mood", MOODS), ("era", ERAS)):
            out[f"mood_grid/{factor}/L{layer}"] = block(mood_rows, factor, levels, layer)
    # sentence means and beat-to-beat derivatives (theme grid, 3-sentence passages only)
    sent = {}
    for layer in theme_layers:
        rows = [r for r in theme_rows if r["layer"] == layer and len([i for i in r["sent_tok"] if i]) == 3]
        for factor, levels in (("theme", THEMES), ("era", ERAS)):
            for lev in list(levels) + ["ALL"]:
                sel = [r for r in rows if lev == "ALL" or r[factor] == lev]
                S = np.array([sent_means(r, factor) for r in sel])
                H = np.array([sent_means(r, factor, "hit") for r in sel])
                R = np.array([sent_means(r, factor, rand=True) for r in sel])
                d = np.diff(S, axis=1)
                pos = np.clip(S, 0, None)
                sent[f"L{layer}/{factor}/{lev}"] = dict(
                    n=len(S), sent_margin=nl(S.mean(0)), sent_margin_se=nl(S.std(0) / np.sqrt(len(S))),
                    sent_hit=nl(H.mean(0)), sent_margin_rand=nl(R.mean(0)),
                    dmargin=nl(d.mean(0)), dmargin_se=nl(d.std(0) / np.sqrt(len(S))),
                    accum_share=nl((pos / (pos.sum(1, keepdims=True) + 1e-9)).mean(0)))
    # last-token vs mean-over-token readout (the hour-9 question)
    lt = {}
    for tag, rows, layers, factors in (("theme_grid", theme_rows, theme_layers, ("theme", "era")),
                                       ("mood_grid", mood_rows, mood_layers, ("mood", "era"))):
        for layer in layers:
            sel = [r for r in rows if r["layer"] == layer]
            for f in factors:
                cv = [r[f + "_curve"] for r in sel]
                lt[f"{tag}/{f}/L{layer}"] = dict(
                    last_token_hit=round(float(np.mean([c["hit"][-1] for c in cv])), 4),
                    mean_token_hit=round(float(np.mean([np.mean(c["hit"]) for c in cv])), 4),
                    first3_hit=round(float(np.mean([np.mean(c["hit"][:3]) for c in cv])), 4),
                    last3_hit=round(float(np.mean([np.mean(c["hit"][-3:]) for c in cv])), 4),
                    last_token_margin=round(float(np.mean([c["margin"][-1] for c in cv])), 4),
                    mean_token_margin=round(float(np.mean([np.mean(c["margin"]) for c in cv])), 4))
    return dict(curves=out, sentences=sent, last_vs_mean=lt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--theme-grid", default="prompts/narrative_theme_v1.json")
    ap.add_argument("--mood-grid", default="prompts/narrative_mood_v1.json")
    ap.add_argument("--theme-layers", type=int, nargs="+", default=[20])
    ap.add_argument("--mood-layers", type=int, nargs="+", default=[16, 18, 20])
    ap.add_argument("--out", default="results/derivative_curves.json")
    a = ap.parse_args()

    lm = LM.from_pretrained(a.model)
    rng = np.random.default_rng(0)
    print("== theme grid ==", flush=True)
    theme_rows = run_grid(lm, a.theme_grid, "era", "theme", a.theme_layers, rng)
    print("== mood grid ==", flush=True)
    mood_rows = run_grid(lm, a.mood_grid, "era", "mood", a.mood_layers, rng, sentences=False)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    summary = summarize(theme_rows, mood_rows, a.theme_layers, a.mood_layers)
    json.dump(dict(model=a.model, n_bins=N_BINS, theme_layers=a.theme_layers,
                   mood_layers=a.mood_layers, summary=summary, theme=theme_rows, mood=mood_rows),
              open(a.out, "w"))
    print("saved", a.out, len(theme_rows), len(mood_rows))


if __name__ == "__main__":
    main()
