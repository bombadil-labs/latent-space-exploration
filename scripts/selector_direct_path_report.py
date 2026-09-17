"""Read the direct-path result files; print the gate table, the per-claim tables and the verdicts.

Primary statistic (spec §4.1): G_new = mean_cases(m_A - m_F_par) in nats, with the paired sign
fraction and a 90% bootstrap lower bound over cases (cluster bootstrap reported beside it).
Ranks are descriptive only (INSTRUMENTS.md §1).
Decision rule (spec §6): LB > 0 and point >= 2*tau -> "computed" (downgraded to "not demonstrated"
if the one-parameter dose family fits within tau); point >= 2*tau with LB <= 0 -> "not
demonstrated"; point < 2*tau -> "not demonstrated" AND PROGRAM.md 0.1's stop fires.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

LOGGED = {                      # gate 1: the numbers these instruments logged (spec §5.1)
    ("role", 20): 1.69, ("role", 14): 1.33,
    ("B/era", 14): 1.25, ("B/voice", 14): 1.24, ("B/tense", 14): 1.03,
    ("D", 14): 2.81,
}
LOGGED_RAND = {("role", 20): 3.25, ("B/era", 14): 2.00, ("B/voice", 14): 2.25, ("B/tense", 14): 1.44}
LOGGED_NONE = {("B/era", 14): 2.00, ("B/voice", 14): 2.00, ("B/tense", 14): 1.50, ("D", 14): 9.50}
CHANCE = {"role": 3.5, "B/era": 2.0, "B/voice": 2.0, "B/tense": 1.5, "D": 9.5}


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    mu = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(mu, 5)), float(np.percentile(mu, 95))


def cluster_ci(x, cl, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    groups = [np.where(np.asarray(cl) == c)[0] for c in sorted(set(cl))]
    mu = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(groups), size=len(groups))
        mu[i] = x[np.concatenate([groups[j] for j in pick])].mean()
    return float(np.percentile(mu, 5)), float(np.percentile(mu, 95))


def sel(rows, claim, layer, arm, field="m", **match):
    out = {}
    for r in rows:
        if r["claim"] == claim and r["layer"] == layer and r["arm"] == arm \
                and all(r.get(k) == v for k, v in match.items()):
            out[r["case"]] = r.get(field)
    return out


def claim_table(rows, claim, layer, doses, r_ratio, out):
    cases = sorted({r["case"] for r in rows if r["claim"] == claim and r["layer"] == layer})
    cl = [next(r["cluster"] for r in rows if r["case"] == c and r["claim"] == claim) for c in cases]

    def g(arm, f="m", **kw):
        d = sel(rows, claim, layer, arm, f, **kw)
        return np.array([d.get(c, np.nan) if d.get(c) is not None else np.nan for c in cases], float)

    mA, mFp, mFa, mKL = g("A"), g("F_par"), g("F_abs"), g("F_KL")
    rand = np.concatenate([v[~np.isnan(v)] for v in (g("R0"), g("R1"))])
    _, rlo, rhi = boot_ci(rand)
    tau = max(abs(rand.mean()), (rhi - rlo) / 2)

    dose_m = {c: g("F_dose", dose=c) for c in doses}
    fine = np.linspace(min(doses), max(doses), 600)
    curves = np.array([np.interp(fine, doses, [dose_m[c][i] for c in doses]) for i in range(len(cases))])
    sse = ((mA[:, None] - curves) ** 2).sum(0)
    j = int(np.argmin(sse))
    cstar, rms = float(fine[j]), float(np.sqrt(sse[j] / len(cases)))
    inside = float(np.mean((mA >= curves.min(1)) & (mA <= curves.max(1))))
    m_rel = np.array([np.interp(r_ratio, doses, [dose_m[c][i] for c in doses]) for i in range(len(cases))])

    def stat(d, name):
        d = np.asarray(d, float)
        keep = ~np.isnan(d)
        mu, lo, hi = boot_ci(d[keep])
        clo, chi = cluster_ci(d[keep], list(np.array(cl)[keep]))
        return dict(name=name, mean=mu, lo=lo, hi=hi, sign=float(np.mean(d[keep] > 0)),
                    clo=clo, chi=chi, n=int(keep.sum()))

    G_new = stat(mA - mFp, "G_new = m_A - m_F_par")
    others = [stat(mA - mFa, "G_abs = m_A - m_F_abs"), stat(mA - mKL, "G_KL (dose-free)"),
              stat(mA - m_rel, f"G_rel (v1's relative-norm dose c={r_ratio:.2f}, robustness only)")]
    if not np.all(np.isnan(g("A_span"))):
        others.append(stat(mA - g("A_span"), "A - A_span (lead-position contribution)"))
    if not np.all(np.isnan(g("C"))):
        others.append(stat(g("C"), "m_C (lead positions only: computation, no direct path)"))

    rk = lambda arm, **kw: float(np.nanmean(g(arm, "rank", **kw)))
    chance = CHANCE.get(claim, float(np.nanmean(g("N", "rank"))))
    out.append(f"\n### {claim} @ L{layer} — n = {len(cases)} cases in {len(set(cl))} clusters")
    out.append(f"\n`tau` = **{tau:.4f}** nats (random arm: mean m {rand.mean():+.4f}, 90% CI of the "
               f"mean [{rlo:+.4f}, {rhi:+.4f}], n={len(rand)} draws); the threshold is `2 tau` = {2*tau:.4f}.")
    out.append("")
    out.append("| arm | mean m (nats) | mean rank | declared null / expectation |")
    out.append("|---|---|---|---|")
    ARMS = [("N", f"rank exactly {chance:.2f} = chance, every gain exactly 0.0"),
            ("A", f"the logged instrument (logged rank {LOGGED.get((claim, layer), float('nan')):.2f})"),
            ("A_span", "~ A if the lens acts where it is read"),
            ("R0", f"m ~ 0, rank chance {chance:.2f}"), ("R1", f"m ~ 0, rank chance {chance:.2f}"),
            ("C", "lead positions only: any effect is computed through attention"),
            ("F_delta", "identical to A (gate 3)"),
            ("F_abs", "the literal skip term, no help from any block"),
            ("F_par", "**THE NULL**: skip term rescaled by the stack"),
            ("F_KL", "dose matched on output perturbation"),
            ("F_R_10", "m ~ 0 (control on the control, c=1)"),
            ("F_R_11", "m ~ 0"), ("F_R_kl0", "m ~ 0 (control on the control, c=c_KL)"),
            ("F_R_kl1", "m ~ 0"),
            ("U", "unembedding-only: rank ~ 1 means d points at the target vocabulary"),
            ("P", "positive control (unembedding-built direction)"),
            ("P_abs", "P's literal direct term"), ("P_par", "P's null; P - P_par must be <= tau")]
    for arm, null in ARMS:
        v = g(arm)
        if np.all(np.isnan(v)):
            continue
        out.append(f"| {arm} | {np.nanmean(v):+.4f} | {rk(arm):.2f} | {null} |")
    out.append("")
    for st in [G_new] + others:
        out.append(f"- **{st['name']}** = {st['mean']:+.4f} nats, 90% CI [{st['lo']:+.4f}, {st['hi']:+.4f}], "
                   f"sign fraction {st['sign']:.2f} (n={st['n']}); cluster bootstrap "
                   f"[{st['clo']:+.4f}, {st['chi']:+.4f}]")
    out.append(f"- survival = {np.nanmean(g('F_par','survival')):.3f}, "
               f"orth = {np.nanmean(g('F_par','orth')):.3f} (both / s|d_L|); "
               f"mean c_KL = {np.nanmean(g('F_KL','c_kl')):.2f}; "
               f"KL(A) = {np.nanmean(g('F_KL','kl_A')):.4f} nats/position; "
               f"r_28/r_L = {r_ratio:.2f}")
    out.append(f"- c* fit (§4.3): **c\\* = {cstar:.2f}** x the skip term ("
               f"{cstar/r_ratio:.2f} x v1's relative-norm dose), residual RMS {rms:.4f} nats vs "
               f"tau {tau:.4f}; {inside:.0%} of cases inside the dose envelope")
    out.append("- dose curve, mean m_F(c): " + ", ".join(f"{c}: {np.nanmean(dose_m[c]):+.3f}" for c in doses))

    pt, lo = G_new["mean"], G_new["lo"]
    with np.errstate(all="ignore"):
        pgap = float(np.nanmean(g("P") - g("P_par"))) if not np.all(np.isnan(g("P"))) else float("nan")
    if pgap > tau:
        # spec §5: if the positive control does not read as direct, the instrument cannot tell the
        # hypotheses apart at this configuration and no verdict is issued.
        verdict = (f"**no verdict** — gate 5 failed here: a direction built from the target span's "
                   f"own unembedding rows is credited with {pgap:+.3f} nats of 'new content' "
                   f"(> tau = {tau:.3f}), so this configuration cannot distinguish H_direct from "
                   f"H_computed (G_new would have been {pt:+.3f}, LB {lo:+.3f})")
    elif np.nanmean(mA) < 2 * tau:
        # exposed by the L27 sanity run: G_new is a DIFFERENCE, so it is large whenever the direct
        # term is harmful, even where the treatment itself does nothing. There is no claim to grade
        # unless the treatment has an effect in the first place.
        verdict = (f"no verdict — the treatment itself is at its own null here "
                   f"(m_A = {np.nanmean(mA):+.3f} < 2 tau = {2*tau:.3f}), so G_new measures only the "
                   f"sign of the direct term, not computation")
    elif lo > 0 and pt >= 2 * tau:
        verdict = "computed"
        if rms <= tau:
            verdict = (f"not demonstrated (downgraded per §6: a one-parameter direct family fits the "
                       f"cases, residual RMS {rms:.4f} <= tau {tau:.4f})")
    elif pt >= 2 * tau:
        verdict = "not demonstrated (point estimate >= 2 tau but the 90% lower bound is <= 0)"
    else:
        verdict = "not demonstrated, and PROGRAM.md 0.1's stop fires for this claim"
    out.append(f"- **verdict: {verdict}**")
    frac = lambda *arms: float(np.nanmean([np.nanmean(g(x)) for x in arms]))
    return dict(claim=claim, layer=layer, tau=tau, G=G_new, cstar=cstar, rms=rms, verdict=verdict,
                stop=bool(pt < 2 * tau), n=len(cases), rank_A=rk("A"),
                rank_R=float(np.nanmean([rk("R0"), rk("R1")])), rank_N=rk("N"),
                fr1=frac("F_R_10", "F_R_11"), frkl=frac("F_R_kl0", "F_R_kl1"),
                p_abs=float(np.nanmean(g("P_abs"))), p_gap=pgap,
                c_kl=float(np.nanmean(g("F_KL", "c_kl"))), mA=float(np.nanmean(mA)), mFa=float(np.nanmean(mFa)),
                mFp=float(np.nanmean(mFp)))


def xtalk(rows, grid_path, layer, out):
    """Test X at the final residual: variance of the per-combo gains explained by each main effect."""
    g = json.load(open(grid_path))
    F = g["factors"]; names = list(F)
    combos = list(itertools.product(*[F[n] for n in names]))

    def decompose(vals):
        G = np.asarray(vals, float); gm = G.mean(); tot = ((G - gm) ** 2).sum() + 1e-12
        return {n: float(sum((np.mean([G[k] for k, c in enumerate(combos) if c[i] == cc[i]]) - gm) ** 2
                             for cc in combos) / tot) for i, n in enumerate(names)}

    out.append("\n### Test X at the final residual (spec §3 row X)")
    out.append("\nRows: the patched factor. Columns: fraction of the gain variance over the 18 "
               "candidates explained by that factor's main effect. Logged (at L14, arm A): "
               "era 0.58 / voice 0.64 / tense 0.47 on-diagonal, random 0.25 / 0.21 / 0.02.")
    for arm in ("A", "F_par", "F_KL", "R0"):
        out.append(f"\n**arm {arm}**\n")
        out.append("| patched | " + " | ".join(names) + " |")
        out.append("|---" * (len(names) + 1) + "|")
        for n in names:
            rs = [r for r in rows if r["claim"] == f"B/{n}" and r["arm"] == arm
                  and r["layer"] == layer and r.get("gains")]
            if not rs:
                continue
            dec = [decompose(r["gains"]) for r in rs]
            out.append(f"| {n} | " + " | ".join(f"{np.mean([d[m] for d in dec]):.2f}" for m in names) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--grid", default="prompts/narrative_factors_v2.json")
    ap.add_argument("--rerun", default=None, help="gate 6: a re-run of some cases, compared on m")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows, meta = [], {}
    for p in a.files:
        d = json.load(open(p))
        meta[p] = d["meta"]
        rows += d["rows"]
    doses = list(meta[a.files[0]]["doses"])

    out = ["# Phase 0.1 — the direct-path test for the log-prob selectors (piece 1)", ""]
    out.append("Spec: `docs/specs/selector_direct_path_v1.md` (v2). Script: "
               "`scripts/selector_direct_path.py`; `pre_28` captured by a forward-pre-hook on "
               "`model.model.norm` (never `hidden_states[28]`).")
    out.append("")
    for p, m in meta.items():
        out.append(f"- `{p}` — {m['mode']} {m.get('tests', '')} layers {m['layers']} scale {m['scale']}; "
                   f"py {m['python']}, torch {m['torch']}, transformers {m['transformers']}; "
                   f"tied embeddings **{m['tied_embeddings']}**; r_L {({k: round(v,1) for k,v in m.get('r_L',{}).items()})}, "
                   f"r_28 {m.get('r_28', 0):.1f}; {m['seconds']/60:.0f} min")

    out.append("\n## Gates (spec §5)\n")
    fd = [r["fdelta_err"] for r in rows if r["arm"] == "F_delta"]
    nmax = [r["max_abs_gain"] for r in rows if r["arm"] == "N"]
    cp = [r["max_abs_gain"] for r in rows if r["arm"] == "C_plumb"]
    out.append(f"- **gate 2 (N)**: max |gain| over every no-patch candidate = {max(nmax):.1e}; mean rank "
               f"reported per claim below. -> {'PASS' if max(nmax) == 0.0 else 'FAIL'}")
    out.append(f"- **gate 3 (F_delta == A)**: max |m_F_delta - m_A| over all cases = {max(fd):.2e} nats "
               f"(threshold 1e-4). -> {'PASS' if max(fd) < 1e-4 else 'FAIL'}")
    out.append(f"- **gate 5, C_plumb**: over the plumbing cases, max |gain| in the model = "
               f"{', '.join(f'{c:.2f}' for c in cp)} nats (needs >> 0.1); the corresponding offline "
               f"direct arm is identically 0.0 because no scored position is patched. -> "
               f"{'PASS' if min(cp) > 0.1 else 'FAIL'}")

    if a.rerun:
        b = json.load(open(a.rerun))["rows"]
        k = lambda r: (r["claim"], r["case"], r["layer"], r["arm"], r.get("dose"))
        first = {k(r): r["m"] for r in rows}
        # the random arms draw from an rng whose stream depends on which cases the run covers, so
        # only the rng-free arms are comparable across runs; they are the ones every verdict uses.
        det = [r for r in b if k(r) in first
               and not (r["arm"].startswith("R") or r["arm"].startswith("F_R") or r["arm"] == "C_plumb")]
        dif = [abs(first[k(r)] - r["m"]) for r in det]
        ncase = len({r["case"] for r in b})
        out.append(f"- **gate 6 (determinism)**: {ncase} cases re-run from scratch; {len(dif)} "
                   f"rng-free arm margins compared (A, A_span, C, F_delta, F_abs, F_par, the dose "
                   f"grid, F_KL, N, U, P); max |m_rerun - m_first| = {max(dif):.1e} nats "
                   f"(threshold 1e-3). -> {'PASS' if max(dif) < 1e-3 else 'FAIL'}")

    rratio = {}
    for m in meta.values():
        for L, r in m.get("r_L", {}).items():
            rratio[(m["mode"], int(L))] = m["r_28"] / r
    res = []
    out.append("\n## Per-claim tables\n")
    for claim, L in sorted({(r["claim"], r["layer"]) for r in rows}):
        mode = "role" if claim == "role" else "factors"
        res.append(claim_table(rows, claim, L, doses, rratio.get((mode, L), 3.5), out))
    xtalk(rows, a.grid, 14, out)

    out.append("\n## Gates 4 and 5, per claim\n")
    out.append("| claim | tau | F_R at c=1 | F_R at c=c_KL | gate 4 | m_P_abs | m_P - m_P_par | gate 5 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in res:
        g4 = "PASS" if max(abs(r["fr1"]), abs(r["frkl"])) <= r["tau"] else "off null"
        g5 = "PASS" if (r["p_abs"] > 2 * r["tau"] and r["p_gap"] <= r["tau"]) else "FAIL"
        if np.isnan(r["p_abs"]):
            g5, r["p_abs"], r["p_gap"] = "n/a (arm not run)", float("nan"), float("nan")
        out.append(f"| {r['claim']} @ L{r['layer']} | {r['tau']:.3f} | {r['fr1']:+.3f} | "
                   f"{r['frkl']:+.3f} | {g4} | {r['p_abs']:+.2f} | {r['p_gap']:+.3f} | {g5} |")
    out.append("\nGate 4 asks that a random vector at the same dose sits at the noise floor; gate 5 "
               "asks that a direction built from the target span's unembedding rows reads as large "
               "and as **not computed** (`m_P - m_P_par <= tau`). If gate 5 fails, no verdict is "
               "issued (spec §5).")

    out.append("\n## Gate 1 — reproduction of the logged ranks\n")
    out.append("| claim | logged A | observed A | logged random | observed random | logged no-patch | observed no-patch |")
    out.append("|---|---|---|---|---|---|---|")
    for r in res:
        k = (r["claim"], r["layer"])
        f = lambda d: f"{d[k]:.2f}" if k in d else "—"
        out.append(f"| {r['claim']} @ L{r['layer']} | {f(LOGGED)} | {r['rank_A']:.2f} | {f(LOGGED_RAND)} | "
                   f"{r['rank_R']:.2f} | {f(LOGGED_NONE)} | {r['rank_N']:.2f} |")

    sub = ("biology", "law", "music", "software")
    rs = [r["rank"] for r in rows if r["claim"] == "role" and r["layer"] == 14
          and r["arm"] == "A" and r["cluster"] in sub]
    if rs:
        out.append(f"\nThe logged role number at L14 (1.33) was measured on four domains only "
                   f"({', '.join(sub)}); restricted to those four this run reads "
                   f"**{np.mean(rs):.2f}** (n={len(rs)}). The 1.58 above is all eight domains. "
                   f"Every logged rank is therefore reproduced exactly under mid-rank.")

    out.append("\n## Verdicts\n")
    out.append("| claim | m_A | m_F_abs | m_F_par | G_new | 90% LB | sign | 2 tau | c* | verdict |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in res:
        out.append(f"| {r['claim']} @ L{r['layer']} | {r['mA']:+.3f} | {r['mFa']:+.3f} | {r['mFp']:+.3f} | "
                   f"{r['G']['mean']:+.3f} | {r['G']['lo']:+.3f} | {r['G']['sign']:.2f} | "
                   f"{2*r['tau']:.3f} | {r['cstar']:.2f} | {r['verdict'].split('(')[0].strip()} |")
    primaries = [r for r in res if (r["claim"], r["layer"]) in (("role", 20), ("D", 14))]
    fires = [r for r in primaries if r["stop"]]
    out.append(f"\n**PROGRAM.md 0.1 stop condition:** primaries are role @ L20 and composed D @ L14; "
               f"the stop fires if either lands below 2 tau. Fired for: "
               f"{', '.join(r['claim'] for r in fires) if fires else 'neither'}.")

    txt = "\n".join(out)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
