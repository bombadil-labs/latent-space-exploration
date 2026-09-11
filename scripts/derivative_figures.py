"""Figures for results/derivative_curves.json -> results/figures/*.png."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derivative_curves import ERAS, THEMES, MOODS, agg, sent_means  # noqa: E402

C = {"betrayal": "#c0392b", "sacrifice": "#2980b9", "homecoming": "#27ae60",
     "medieval": "#8e44ad", "1920s": "#d68910", "farfuture": "#16a085",
     "dread": "#2c3e50", "tender": "#c2408a", "comic": "#f1c40f"}


def band(ax, x, m, se, color, label):
    ax.plot(x, m, color=color, lw=2, label=label, marker="o", ms=3)
    ax.fill_between(x, m - se, m + se, color=color, alpha=0.15, lw=0)


def curve_fig(rows, factor, levels, layer, title, ylab, path, nb):
    x = (np.arange(nb) + 0.5) / nb
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, field, yl in ((axes[0], "margin", ylab), (axes[1], "hit", "P(own level = argmax)")):
        for lev in levels:
            m, se, n = agg(rows, factor, field, (factor, lev), layer)
            band(ax, x, m, se, C[lev], f"{lev} (n={n})")
        m, se, _ = agg(rows, factor, field, None, layer)
        ax.plot(x, m, color="k", lw=2.2, ls="--", label="all passages")
        r, rse, _ = agg(rows, factor, field, None, layer, rand=True)
        band(ax, x, r, rse, "#888888", "random direction")
        if field == "hit":
            ax.axhline(1 / 3, color="#888888", ls=":", lw=1)
        else:
            ax.axhline(0, color="#888888", ls=":", lw=1)
        ax.set_xlabel("normalised position in passage")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def sentence_fig(rows, layer, path):
    rows = [r for r in rows if r["layer"] == layer and len([i for i in r["sent_tok"] if i]) == 3]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, (factor, levels) in enumerate((("theme", THEMES), ("era", ERAS))):
        top, bot = axes[0][col], axes[1][col]
        w = 0.26
        for k, lev in enumerate(levels):
            sel = [r for r in rows if r[factor] == lev]
            S = np.array([sent_means(r, factor) for r in sel])
            d = np.diff(S, axis=1)
            xs = np.arange(3) + (k - 1) * w
            top.bar(xs, S.mean(0), w, yerr=S.std(0) / np.sqrt(len(S)), color=C[lev],
                    label=f"{lev} (n={len(S)})", capsize=2)
            bot.bar(np.arange(2) + (k - 1) * w, d.mean(0), w,
                    yerr=d.std(0) / np.sqrt(len(S)), color=C[lev], capsize=2)
        R = np.array([sent_means(r, factor, rand=True) for r in rows])
        top.plot(np.arange(3), R.mean(0), "k_", ms=28, label="random direction")
        for ax in (top, bot):
            ax.axhline(0, color="k", lw=0.8)
            ax.grid(alpha=0.2, axis="y")
        top.set_xticks(range(3), ["sent 1", "sent 2", "sent 3"])
        bot.set_xticks(range(2), ["s2 - s1", "s3 - s2"])
        top.set_title(f"{factor}: per-sentence margin  (∫ slice)")
        bot.set_title(f"{factor}: beat-to-beat derivative  (Δ margin)")
        top.set_ylabel("own-level margin (cosine)")
        bot.set_ylabel("Δ margin")
        top.legend(fontsize=8)
    fig.suptitle(f"Per-sentence readout and its discrete derivative — Qwen2.5-1.5B, layer {layer}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def mood_fig(rows, layers, path, nb):
    x = (np.arange(nb) + 0.5) / nb
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for layer in layers:
        m, se, n = agg(rows, "mood", "hit", None, layer)
        ax.plot(x, m, lw=2, marker="o", ms=3, label=f"mood, layer {layer} (n={n})")
    e, _, _ = agg(rows, "era", "hit", None, layers[-1])
    ax.plot(x, e, color="k", ls="--", lw=2, label=f"era, layer {layers[-1]} (reference)")
    r, rse, _ = agg(rows, "mood", "hit", None, layers[-1], rand=True)
    band(ax, x, r, rse, "#888888", "random direction")
    ax.axhline(1 / 3, color="#888888", ls=":", lw=1)
    ax.set_xlabel("normalised position in sentence")
    ax.set_ylabel("P(own mood = argmax)")
    ax.set_title("mood is a late-token phenomenon")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    ax = axes[1]
    L = layers[len(layers) // 2]
    for lev in MOODS:
        m, se, n = agg(rows, "mood", "margin", ("mood", lev), L)
        band(ax, x, m, se, C[lev], f"{lev} (n={n})")
    m, _, _ = agg(rows, "mood", "margin", None, L)
    ax.plot(x, m, "k--", lw=2.2, label="all")
    ax.axhline(0, color="#888888", ls=":", lw=1)
    ax.set_xlabel("normalised position in sentence")
    ax.set_ylabel("own-mood margin (cosine)")
    ax.set_title(f"per mood, layer {L}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    fig.suptitle("Mood grid (one-sentence passages) — Qwen2.5-1.5B")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/derivative_curves.json")
    ap.add_argument("--outdir", default="results/figures")
    a = ap.parse_args()
    D = json.load(open(a.data))
    nb, TL, ML = D["n_bins"], D["theme_layers"][-1], D["mood_layers"]
    os.makedirs(a.outdir, exist_ok=True)
    p = lambda n: os.path.join(a.outdir, n)
    curve_fig(D["theme"], "theme", THEMES, TL,
              f"Theme accumulates mid-passage — Qwen2.5-1.5B, layer {TL}",
              "own-theme margin (cosine)", p("derivative_theme_position.png"), nb)
    curve_fig(D["theme"], "era", ERAS, TL,
              f"Era is locked in from the first tokens — Qwen2.5-1.5B, layer {TL}",
              "own-era margin (cosine)", p("derivative_era_position.png"), nb)
    sentence_fig(D["theme"], TL, p("derivative_sentence_bars.png"))
    mood_fig(D["mood"], ML, p("derivative_mood_position.png"), nb)


if __name__ == "__main__":
    main()
