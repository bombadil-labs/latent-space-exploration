"""Commutator trajectories for narrative factor pairs on an NDIF-hosted model.

For a factor pair (A, B) and a combination of levels (a, b), generate greedily under two orderings

    AB : dir_A[a] added at block L1's output, dir_B[b] added at block L2's output
    BA : dir_B[b] added at block L1's output, dir_A[a] added at block L2's output

(the patch is re-applied at every decoding step, as in scripts/ndif_generate.py), plus a base
continuation and the two single patches (A@L1, B@L2). Divergence between the two orderings is
measured two ways:

  (a) token-level: first index where the two token sequences differ; running Hamming fraction.
  (b) readout-level: re-fetch per-token block-`read` residuals for both generated texts (no patch)
      and project each continuation token onto the factor directions at that block; the per-token
      difference of projections between the two orderings is the divergence curve. Units are sigma,
      the per-axis standard deviation of the same projections over the BASE continuation's tokens.

=============================== REGIME CLASSIFICATION RULE ===============================
Stated before any curve was looked at (groovy-commutator's five regimes). Per (pair, a, b, prompt),
with T = 60 generated tokens, curve[i] = || (p_AB(i) - p_BA(i)) / sigma ||_2 over the two on-axis
projections (A-level axis and B-level axis), and

    ham            = fraction of positions where the two orderings' token ids differ
    base_ov_X      = fraction of positions where ordering X's token id equals the base continuation's
    early          = mean(curve[0:15]),  late = mean(curve[45:60]),  growth = late - early
    consistency    = max over the two axes of the fraction of i in [5, T) at which the SIGNED
                     per-axis difference takes its modal sign (0.5 = coin flip, 1.0 = one ordering
                     is uniformly higher on that factor)

first rule that matches wins:

  1. COMMUTE      ham == 0                  and mean(curve) < 0.5
  2. DRAIN        |base_ov_AB - base_ov_BA| >= 0.30 and max(base_ov_AB, base_ov_BA) >= 0.60
                  (one ordering collapses back onto the unpatched prompt-prior continuation)
  3. CRYSTALLINE  growth < 0.5 and mean(curve) < 3.0      (bounded, roughly constant offset)
  4. STRUCTURED   growth >= 0.5 and consistency >= 0.70   (grows with a persistent signed pattern)
  5. NOISE        otherwise                               (grows without a persistent pattern)

A pair x level-combination is assigned the modal regime over prompts (ties -> the later regime in
the order above, i.e. the more divergent reading).
==========================================================================================

Controls added after the first run (hour 19 lacked all three; the rule above is unchanged):

  --prompts p1,p2,...   any subset of PROMPTS below (four neutral prompts are now defined), so an
                        aggregate regime label rests on 4 votes rather than a 1-1 tie.
  --layers L1,L2        patch blocks (default 14,20). Results for a non-default layer pair are
                        stored under the pair key "<pair>@<L1>-<L2>", so one output file can hold
                        several layer pairs. The readout block (--read) is independent of them.
  --null N              RANDOM-DIRECTION NULL. Instead of the factor directions, draw N pairs of
                        isotropic random unit vectors (u_A, u_B) per prompt and rescale each, at
                        each layer where it is applied, to the MEAN NORM of the factor direction it
                        replaces at that layer (u_A -> |dir_A|@L1 / |dir_A|@L2, likewise u_B). The
                        AB/BA protocol, the readout (projection onto u_A, u_B at --read, sigma from
                        the base continuation's own per-token spread on those axes) and the regime
                        rule are otherwise identical. Stored under the key "null_<pair>...".
                        This is the control the first run named as missing: it says how much of the
                        observed divergence is "two different patches" rather than "two factors".
  --no-single           skip the single-patch (A-only, B-only) generations, which are only used for
                        the dominance table.

Usage:
  python scripts/ndif_commutator.py --pair era_theme --grid prompts/narrative_theme_v1.json \
      --stacks results/stacks_gemma_2_9b_it_narrative_theme_v1.npz --out results/commutator_gemma9b.json
  python scripts/ndif_commutator.py --pair era_voice --grid prompts/narrative_factors_v1.json \
      --stacks results/stacks_gemma_2_9b_it_narrative_factors_v1.npz --out results/commutator_gemma9b.json
  python scripts/ndif_commutator.py --pair era_theme --grid ... --stacks ... --layers 16,24 \
      --prompts news,road --out results/commutator_gemma9b_v2.json
  python scripts/ndif_commutator.py --pair era_theme --grid ... --stacks ... --null 4 \
      --prompts news,road --out results/commutator_gemma9b_v2.json
  python scripts/ndif_commutator.py --analyze --out results/commutator_gemma9b.json
"""
import argparse, json, os, time
import numpy as np, torch

PROMPTS = {
    "news": "A passage from a story: It was late when the news reached her, and",
    "road": "A passage from a story: They had been walking the road since morning, and",
    "door": "A passage from a story: The door was already open when he got there, and",
    "fire": "A passage from a story: The fire had burned down to embers before anyone spoke, and",
}

# ----------------------------------------------------------------------------- directions
def grid_levels(grid, pair):
    """Return (nameA, levelsA, nameB, levelsB, key_index_of_A, key_index_of_B) for a grid file."""
    g = json.load(open(grid))
    if "factors" in g:                                   # narrative_theme_v1: factors + key_order
        order = [k for k in g["key_order"] if k != "scene"]
        levels = {k: g["factors"][k] for k in order}
    else:                                                # narrative_factors_v1: eras / voices
        order, levels = [], {}
        for plural, sing in (("eras", "era"), ("voices", "voice"), ("themes", "theme")):
            if plural in g:
                order.append(sing); levels[sing] = g[plural]
    a, b = pair.split("_")
    assert a in levels and b in levels, f"{pair} not in grid factors {order}"
    return g, a, levels[a], b, levels[b], order.index(a) + 1, order.index(b) + 1


def factor_dirs(stacks, layer, levels, idx):
    z = np.load(stacks); X = {k: z[k][layer] for k in z.files}
    mu = np.mean(list(X.values()), axis=0)
    return {lv: np.mean([v for k, v in X.items() if k.split("/")[idx] == lv], axis=0) - mu for lv in levels}, mu


# ----------------------------------------------------------------------------- remote jobs
def make_model(name):
    from nnsight import LanguageModel
    m = LanguageModel(name, device_map="auto", dispatch=False)
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = m
        try:
            for x in path.split("."): obj = getattr(obj, x)
            return m, obj, m.tokenizer, m.config.hidden_size
        except AttributeError: continue
    raise RuntimeError("no blocks")


def resid(block):
    """Hidden states at a block's output. transformers >= 4.54 returns a bare Tensor [batch, seq, d]
    from Llama/Gemma/Qwen decoder layers (older versions, and GPT-J today, return a tuple), so
    `block.output[0]` silently means "batch row 0" there and a patch written that way lands on the
    FIRST SEQUENCE OF THE BATCH ONLY. Harmless while this script traces one prompt per job, fatal the
    moment anyone batches it -- see results/notes/random_control_diagnosis.md (hour 36) and
    results/notes/instrument_audit.md (hour 39)."""
    o = block.output
    return o if isinstance(o, torch.Tensor) else o[0]


def run(model, B, tok, prompt, patches, n_tokens):
    """patches: list of (layer, np.ndarray). Returns list of generated token ids."""
    from lsx.ndif import ProxyAuthBackend
    backend = ProxyAuthBackend(model.to_model_key())
    vs = [(l, torch.as_tensor(v, dtype=torch.float32)) for l, v in patches]
    with model.generate(prompt, max_new_tokens=n_tokens, do_sample=False, backend=backend) as tracer:
        if vs:
            with tracer.all():
                for l, v in vs:
                    h = resid(B[l]); h[:] = h + v.to(h.device, h.dtype)
        out = model.generator.output.save()
    res = backend.wait(tracer)
    o = res["out"] if isinstance(res, dict) and "out" in res else next(x for x in res.values() if isinstance(x, torch.Tensor))
    ids = o[0] if o.dim() == 2 else o
    n_in = len(tok(prompt)["input_ids"])
    return [int(x) for x in ids[n_in:]]


def read_tokens(model, B, tok, D, text, layer):
    from lsx.ndif import ProxyAuthBackend
    backend = ProxyAuthBackend(model.to_model_key())
    with model.trace(text, backend=backend) as tracer:
        h = B[layer].output[0].reshape(-1, D).save()
    res = backend.wait(tracer)
    v = res["h"] if isinstance(res, dict) and "h" in res else next(x for x in res.values() if isinstance(x, torch.Tensor))
    return v.float().cpu().numpy()


# ----------------------------------------------------------------------------- analysis
def curves(rec, sig):
    """rec: dict with projAB/projBA [T, k]; sig: [k]. Returns curve, signed [T, k]."""
    a = np.asarray(rec["projAB"], float); b = np.asarray(rec["projBA"], float)
    T = min(len(a), len(b)); d = (a[:T] - b[:T]) / sig
    return np.linalg.norm(d, axis=1), d


def classify(ham, base_ov_ab, base_ov_ba, curve, signed):
    T = len(curve)
    early = float(curve[:15].mean()); late = float(curve[max(0, T - 15):].mean())
    growth = late - early; mean_c = float(curve.mean())
    cons = 0.0
    for k in range(signed.shape[1]):
        s = np.sign(signed[5:, k]); s = s[s != 0]
        if len(s): cons = max(cons, float(max((s > 0).mean(), (s < 0).mean())))
    stats = dict(ham=ham, base_ov_AB=base_ov_ab, base_ov_BA=base_ov_ba, mean_curve=mean_c,
                 early=early, late=late, growth=growth, consistency=cons)
    if ham == 0 and mean_c < 0.5: return "commute", stats
    if abs(base_ov_ab - base_ov_ba) >= 0.30 and max(base_ov_ab, base_ov_ba) >= 0.60: return "drain", stats
    if growth < 0.5 and mean_c < 3.0: return "crystalline", stats
    if growth >= 0.5 and cons >= 0.70: return "structured", stats
    return "noise", stats


REG_ORDER = ["commute", "crystalline", "drain", "structured", "noise"]


def summarize(rows):
    """Aggregate divergence statistics for one pair (or the null)."""
    import statistics
    fd = [r["first_div"] if r["first_div"] is not None else 60 for r in rows]
    regs = {}
    for r in rows: regs[r["regime"]] = regs.get(r["regime"], 0) + 1
    return dict(n=len(rows), regimes=regs,
                frac_identical=float(np.mean([r["ham"] == 0 for r in rows])),
                mean_ham=float(np.mean([r["ham"] for r in rows])),
                median_first_div=float(statistics.median(fd)),
                mean_curve=float(np.mean([r["mean_curve"] for r in rows])),
                mean_early=float(np.mean([r["early"] for r in rows])),
                mean_late=float(np.mean([r["late"] for r in rows])),
                mean_consistency=float(np.mean([r["consistency"] for r in rows])),
                mean_base_ov=float(np.mean([0.5 * (r["base_ov_AB"] + r["base_ov_BA"]) for r in rows])))


def regime_structure(rows, n_perm=20000, seed=0):
    """Does the regime label depend on the level combination at all, over and above the prompt?
    Statistic: sum over level combinations of the modal count across prompts. Null: shuffle the
    regime labels within each prompt (preserving that prompt's own regime marginal)."""
    cols = {}
    for r in rows: cols.setdefault(r["prompt"], {})[(r["a"], r["b"])] = r["regime"]
    if len(cols) < 3: return None
    combos = sorted({ab for c in cols.values() for ab in c})
    if any(len(c) != len(combos) for c in cols.values()): return None
    M = np.array([[cols[p][ab] for ab in combos] for p in cols])
    stat = lambda X: sum(max(list(X[:, j]).count(v) for v in set(X[:, j])) for j in range(X.shape[1]))
    obs = stat(M); rng = np.random.default_rng(seed); null = []
    for _ in range(n_perm):
        Y = M.copy()
        for i in range(Y.shape[0]): rng.shuffle(Y[i])
        null.append(stat(Y))
    null = np.array(null)
    return dict(modal_sum=int(obs), max_possible=int(M.shape[0] * M.shape[1]),
                null_mean=float(null.mean()), p=float((null >= obs).mean()),
                unanimous=int(sum(len(set(M[:, j])) == 1 for j in range(M.shape[1]))))


def dominance(P):
    """Mean token overlap of each ordering with the two single-patch continuations, over the cases
    whose single-patch generations exist (--no-single runs contribute nothing). None if there are
    no such cases."""
    g = P["gens"]; acc = {k: [] for k in ("AB|A", "AB|B", "BA|A", "BA|B")}; pids = set()
    for key in P["cases"]:
        pid, la, lb = key.split("|")
        ka, kb = f"{pid}|A|{la}", f"{pid}|B|{lb}"
        if ka not in g or kb not in g: continue
        pids.add(pid)
        A, Bo = g[ka]["ids"], g[kb]["ids"]
        for o in ("AB", "BA"):
            ids = g[f"{pid}|{o}|{la}|{lb}"]["ids"]
            T = min(len(ids), len(A), len(Bo))
            acc[f"{o}|A"].append(float(np.mean([ids[i] == A[i] for i in range(T)])))
            acc[f"{o}|B"].append(float(np.mean([ids[i] == Bo[i] for i in range(T)])))
    if not acc["AB|A"]: return None
    out = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
    out["n"] = len(acc["AB|A"]); out["prompts"] = sorted(pids)
    return out


def analyze(path):
    d = json.load(open(path)); out = {}
    for pair, P in d["pairs"].items():
        rows = []
        for key, rec in P["cases"].items():
            pid, la, lb = key.split("|")
            sig = np.asarray(rec.get("sigma") or P["sigma"][pid], float)
            sig = np.where(sig < 1e-6, 1.0, sig)
            curve, signed = curves(rec, sig)
            reg, st = classify(rec["ham"], rec["base_ov_AB"], rec["base_ov_BA"], curve, signed)
            rows.append(dict(prompt=pid, a=la, b=lb, regime=reg, first_div=rec["first_div"], **st,
                             curve=[round(float(x), 3) for x in curve]))
        out[pair] = rows
    d["analysis"] = {p: [{k: v for k, v in r.items() if k != "curve"} for r in rows] for p, rows in out.items()}
    d["summary"] = {p: summarize(rows) for p, rows in out.items()}
    d["dominance"] = {p: dominance(d["pairs"][p]) for p in out}
    d["regime_structure"] = {p: regime_structure(rows) for p, rows in out.items()}
    for p, rows in out.items():
        for r, row in zip(rows, d["pairs"][p]["cases"].values()):
            row["curve"] = r["curve"]; row["regime"] = r["regime"]
    json.dump(d, open(path, "w"), indent=1)
    # print tables
    for pair, rows in out.items():
        print(f"\n=== {pair} ===")
        print(f"{'a':<10} {'b':<10} {'prompt':<6} {'regime':<12} {'firstdiv':>8} {'ham':>5} {'ovAB':>5} {'ovBA':>5} {'mean':>6} {'early':>6} {'late':>6} {'cons':>5}")
        combos = {}
        for r in rows:
            print(f"{r['a']:<10} {r['b']:<10} {r['prompt']:<6} {r['regime']:<12} {str(r['first_div']):>8} "
                  f"{r['ham']:5.2f} {r['base_ov_AB']:5.2f} {r['base_ov_BA']:5.2f} {r['mean_curve']:6.2f} "
                  f"{r['early']:6.2f} {r['late']:6.2f} {r['consistency']:5.2f}")
            combos.setdefault((r["a"], r["b"]), []).append(r["regime"])
        print(f"-- modal regime per level combination ({pair}) --")
        for (a, b), regs in combos.items():
            best = max(set(regs), key=lambda x: (regs.count(x), REG_ORDER.index(x)))
            print(f"  {a:<10} x {b:<10} {best:<12} {regs}")
        print(f"-- summary ({pair}) -- {json.dumps(d['summary'][pair])}")
        print(f"-- dominance ({pair}) -- {json.dumps(d['dominance'][pair])}")
        print(f"-- regime structure ({pair}) -- {json.dumps(d['regime_structure'][pair])}")
    return out


def pair_key(pair, l1, l2, null):
    k = pair if (l1, l2) == (14, 20) else f"{pair}@{l1}-{l2}"
    return ("null_" + k) if null else k


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=None, help="e.g. era_theme, era_voice")
    ap.add_argument("--grid"); ap.add_argument("--stacks")
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--l1", type=int, default=14); ap.add_argument("--l2", type=int, default=20)
    ap.add_argument("--layers", default=None, help="L1,L2 patch blocks (overrides --l1/--l2)")
    ap.add_argument("--read", type=int, default=20); ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--prompts", default="news,road")
    ap.add_argument("--null", type=int, default=0, help="N random matched-norm direction pairs per prompt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-single", action="store_true", help="skip A-only / B-only generations")
    ap.add_argument("--out", default="results/commutator_gemma9b.json")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.out); return
    if a.layers:
        a.l1, a.l2 = [int(x) for x in a.layers.split(",")]

    g, fa, LA, fb, LB, ia, ib = grid_levels(a.grid, a.pair)
    dA1, _ = factor_dirs(a.stacks, a.l1, LA, ia); dA2, _ = factor_dirs(a.stacks, a.l2, LA, ia)
    dB1, _ = factor_dirs(a.stacks, a.l1, LB, ib); dB2, _ = factor_dirs(a.stacks, a.l2, LB, ib)
    dAR, muR = factor_dirs(a.stacks, a.read, LA, ia); dBR, _ = factor_dirs(a.stacks, a.read, LB, ib)
    pids = a.prompts.split(",")
    key = pair_key(a.pair, a.l1, a.l2, a.null)

    from lsx.ndif import retry_job
    model, B, tok, D = make_model(a.model)
    doc = {"model": a.model, "read": a.read, "scale": a.scale,
           "tokens": a.tokens, "prompts": {p: PROMPTS[p] for p in pids}, "pairs": {}}
    if os.path.exists(a.out):
        doc = json.load(open(a.out)); doc.setdefault("pairs", {})
        doc.setdefault("prompts", {}).update({p: PROMPTS[p] for p in pids})
    P = doc["pairs"].setdefault(key, {"factors": [fa, fb], "levels": {fa: LA, fb: LB},
                                      "grid": a.grid, "gens": {}, "sigma": {}, "cases": {}})
    P.setdefault("gens", {}); P.setdefault("sigma", {}); P.setdefault("cases", {})
    P["l1"], P["l2"], P["read"], P["scale"], P["null"] = a.l1, a.l2, a.read, a.scale, a.null
    if a.null:  # unpatched base continuations are protocol-identical; reuse if already generated
        src = doc["pairs"].get(pair_key(a.pair, a.l1, a.l2, 0), {}).get("gens", {})
        for pid in pids:
            if f"{pid}|base" in src: P["gens"].setdefault(f"{pid}|base", src[f"{pid}|base"])
    save = lambda: json.dump(doc, open(a.out, "w"), indent=1)
    t0 = time.time(); n = [0]

    def gen(key_, prompt, patches):
        if key_ in P["gens"]: return P["gens"][key_]["ids"]
        ids = retry_job(lambda: run(model, B, tok, prompt, patches, a.tokens))
        P["gens"][key_] = {"ids": ids, "text": tok.decode(ids, skip_special_tokens=True)}
        n[0] += 1; print(f"  gen[{n[0]}] {key_} {time.time()-t0:.0f}s", flush=True); save()
        return ids

    unit = lambda v: v / np.linalg.norm(v)

    def resid(key_, prompt, ids):
        """per-token block-`read` residuals of the continuation, grand mean removed."""
        text = prompt + tok.decode(ids, skip_special_tokens=True)
        H = retry_job(lambda: read_tokens(model, B, tok, D, text, a.read))
        n_in = len(tok(prompt)["input_ids"])
        H = H[n_in:] - muR
        n[0] += 1; print(f"  read[{n[0]}] {key_} {H.shape[0]}tok {time.time()-t0:.0f}s", flush=True)
        return H

    def proj(key_, prompt, ids, vecs):
        return (resid(key_, prompt, ids) @ np.stack(vecs).T).tolist()

    if a.null:
        # random directions at the mean norm of the factor direction they replace, per layer
        nrm = lambda dd: float(np.mean([np.linalg.norm(dd[l]) for l in dd]))
        nA1, nA2, nB1, nB2 = nrm(dA1), nrm(dA2), nrm(dB1), nrm(dB2)
        P["null_norms"] = {"A@l1": nA1, "A@l2": nA2, "B@l1": nB1, "B@l2": nB2}
        for pi, pid in enumerate(pids):
            prompt = PROMPTS[pid]
            base_ids = gen(f"{pid}|base", prompt, [])
            Hb = resid(f"{pid}|base", prompt, base_ids)
            for j in range(a.null):
                rng = np.random.default_rng([a.seed, pi, j])
                uA = unit(rng.normal(size=D)); uB = unit(rng.normal(size=D))
                la = lb = f"r{j}"
                kab, kba = f"{pid}|AB|{la}|{lb}", f"{pid}|BA|{la}|{lb}"
                iab = gen(kab, prompt, [(a.l1, uA * nA1 * a.scale), (a.l2, uB * nB2 * a.scale)])
                iba = gen(kba, prompt, [(a.l1, uB * nB1 * a.scale), (a.l2, uA * nA2 * a.scale)])
                ck = f"{pid}|{la}|{lb}"
                if ck in P["cases"]: continue
                axes = [uA, uB]
                sig = (Hb @ np.stack(axes).T).std(0)
                T = min(len(iab), len(iba), len(base_ids), a.tokens)
                diff = [i for i in range(T) if iab[i] != iba[i]]
                P["cases"][ck] = dict(first_div=(diff[0] if diff else None), ham=len(diff) / T,
                                      base_ov_AB=float(np.mean([iab[i] == base_ids[i] for i in range(T)])),
                                      base_ov_BA=float(np.mean([iba[i] == base_ids[i] for i in range(T)])),
                                      sigma=[float(max(x, 1e-6)) for x in sig],
                                      projAB=proj(kab, prompt, iab, axes),
                                      projBA=proj(kba, prompt, iba, axes))
                save()
        save(); print(f"done {key}: {n[0]} jobs, {time.time()-t0:.0f}s -> {a.out}"); return

    for pid in pids:
        prompt = PROMPTS[pid]
        base_ids = gen(f"{pid}|base", prompt, [])
        if pid not in P["sigma"]:
            # sigma per factor axis: spread of the BASE continuation's per-token projections,
            # averaged over the levels of that factor
            pr = np.asarray(proj(f"{pid}|base", prompt, base_ids,
                                 [unit(dAR[l]) for l in LA] + [unit(dBR[l]) for l in LB]), float)
            s = pr.std(0)
            P["sigma"][pid] = [float(max(s[:len(LA)].mean(), 1e-6)), float(max(s[len(LA):].mean(), 1e-6))]
            P["base_proj"] = P.get("base_proj", {}); P["base_proj"][pid] = pr.round(4).tolist()
            save()
        if not a.no_single:
            for la in LA:
                gen(f"{pid}|A|{la}", prompt, [(a.l1, dA1[la] * a.scale)])
            for lb in LB:
                gen(f"{pid}|B|{lb}", prompt, [(a.l2, dB2[lb] * a.scale)])
        for la in LA:
            for lb in LB:
                kab = f"{pid}|AB|{la}|{lb}"; kba = f"{pid}|BA|{la}|{lb}"
                iab = gen(kab, prompt, [(a.l1, dA1[la] * a.scale), (a.l2, dB2[lb] * a.scale)])
                iba = gen(kba, prompt, [(a.l1, dB1[lb] * a.scale), (a.l2, dA2[la] * a.scale)])
                ck = f"{pid}|{la}|{lb}"
                if ck in P["cases"]: continue
                T = min(len(iab), len(iba), len(base_ids), a.tokens)
                diff = [i for i in range(T) if iab[i] != iba[i]]
                rec = dict(first_div=(diff[0] if diff else None), ham=len(diff) / T,
                           base_ov_AB=float(np.mean([iab[i] == base_ids[i] for i in range(T)])),
                           base_ov_BA=float(np.mean([iba[i] == base_ids[i] for i in range(T)])),
                           projAB=proj(kab, prompt, iab, [unit(dAR[la]), unit(dBR[lb])]),
                           projBA=proj(kba, prompt, iba, [unit(dAR[la]), unit(dBR[lb])]))
                P["cases"][ck] = rec; save()
    save(); print(f"done {key}: {n[0]} jobs, {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
