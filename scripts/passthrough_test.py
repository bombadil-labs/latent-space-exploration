"""Pass-through arm for the stage-14 / stage-23 era shift (spec core_v1 §2a).

The claim: an era shift patched at layer 14 moves the layer-20 era readout to the target
(0.89 Qwen / 0.88 Gemma / 0.94 GPT-grid) while the theme readout stays put.

The suspicion: the readout layer is AFTER the patch layer and the residual stream is additive,
so  resid_read = resid_patch + shift + sum(block outputs).  If the era direction is stable
across layers, the readout moves by arithmetic alone.

The arm (no forward pass): passthrough = readout(base_span_vector_at_read_layer + shift).
Because the patch is added at EVERY position and the readout is a mean over span tokens,
mean_span(resid + shift) = mean_span(resid) + shift exactly, so the cached stacks are all
that is needed.  The readout code below is copied verbatim from scripts/stage7_shift.py.

Modes:
  passthrough  -- arithmetic arm at every read layer, from cached stacks (no model)
  model        -- local patched forwards, capturing ALL read layers in one pass (Qwen only)
  report       -- assemble model vs pass-through table + sanity checks -> json
"""
import argparse, json, os
import numpy as np

# ---------------------------------------------------------------- readout (verbatim from stage7_shift.py)
cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def make_dirs(X, key, E, T, l, train):
    allv = np.stack([X[key(s, e, t)][l] for s in train for e in E for t in T])
    mu = allv.mean(0)
    de = {e: np.mean([X[key(s, e, t)][l] for s in train for t in T], axis=0) - mu for e in E}
    dt = {t: np.mean([X[key(s, e, t)][l] for s in train for e in E], axis=0) - mu for t in T}
    return de, dt, mu


def classify(v, deR, dtR, muR, E, T):
    v = v - muR
    return max(E, key=lambda e: cos(v, deR[e])), max(T, key=lambda tt: cos(v, dtR[tt]))


# ---------------------------------------------------------------- shared case enumeration
def load(grid_path, stacks_path):
    g = json.load(open(grid_path))
    z = np.load(stacks_path)
    X = {k: z[k] for k in z.files if k != "roles"}
    E, T, S = g["factors"]["era"], g["factors"]["theme"], g["scenes"]
    key = lambda s, e, t: f"{s}/{e}/{t}"
    n_layers = X[key(S[0], E[0], T[0])].shape[0]
    return g, X, E, T, S, key, n_layers


def vectors(X, key, E, T, S, patch_layer, seed=0):
    """Yield (scene, e1, t, e2, cond, vec) in the SAME order/rng draw sequence as stage7_shift.py,
    so the random control is the same random control."""
    rng = np.random.default_rng(seed)
    for s in S:
        train = [x for x in S if x != s]
        deP, _, _ = make_dirs(X, key, E, T, patch_layer, train)
        for e1 in E:
            for t in T:
                yield (s, e1, t, e1, "base", None, train)
                for e2 in E:
                    if e2 == e1:
                        continue
                    shift = deP[e2] - deP[e1]
                    r = rng.normal(size=shift.shape)
                    r *= np.linalg.norm(shift) / np.linalg.norm(r)
                    yield (s, e1, t, e2, "shift", shift, train)
                    yield (s, e1, t, e2, "rand", r, train)


def score(rows):
    def acc(cond, f):
        xs = [x for x in rows if x["cond"] == cond]
        return (float(np.mean([f(x) for x in xs])), len(xs)) if xs else (float("nan"), 0)
    out = {
        "base_era_e1": acc("base", lambda x: x["era_read"] == x["e1"])[0],
        "base_theme_t": acc("base", lambda x: x["theme_read"] == x["t"])[0],
        "n_base": acc("base", lambda x: 1)[1],
    }
    for c in ("shift", "rand"):
        out[f"{c}_moved"] = acc(c, lambda x: x["era_read"] == x["e2"])[0]
        out[f"{c}_stayed"] = acc(c, lambda x: x["era_read"] == x["e1"])[0]
        out[f"{c}_kept"] = acc(c, lambda x: x["theme_read"] == x["t"])[0]
        out[f"n_{c}"] = acc(c, lambda x: 1)[1]
    return out


# ---------------------------------------------------------------- arms
def passthrough_rows(X, key, E, T, S, patch_layer, read_layer, scale=1.0, zero_shift=False,
                     norm_match=False):
    """norm_match: rescale the layer-`patch_layer` shift so that ||shift|| / ||resid|| is the same at
    the read layer as it was at the patch layer.  Without this the arm is unfairly weak at deep read
    layers, where residual norms are much larger and a fixed-norm layer-14 vector is negligible."""
    rows = []
    cache = {}
    for (s, e1, t, e2, cond, vec, train) in vectors(X, key, E, T, S, patch_layer):
        if s not in cache:
            cache[s] = make_dirs(X, key, E, T, read_layer, train)
        deR, dtR, muR = cache[s]
        base = X[key(s, e1, t)][read_layer]
        sc = 0.0 if zero_shift else scale
        if norm_match and vec is not None:
            b14 = X[key(s, e1, t)][patch_layer]
            sc *= float(np.linalg.norm(base) / (np.linalg.norm(b14) + 1e-9))
        v = base if vec is None else base + sc * vec
        er, tr = classify(v, deR, dtR, muR, E, T)
        rows.append(dict(scene=s, e1=e1, t=t, e2=e2, cond=cond, era_read=er, theme_read=tr))
    return rows


def model_rows(lm, g, X, key, E, T, S, patch_layer, read_layers, scale=1.0):
    """One patched forward per case; the returned span vector is [n_layers, d] so every read layer
    is scored from the same forward.  Returns {read_layer: rows}."""
    from lsx import extract
    from lsx.model import Patch
    from lsx.steer import add_vector
    lead, spans = g["lead"], g["spans"]
    out = {l: [] for l in read_layers}
    cache = {}
    for i, (s, e1, t, e2, cond, vec, train) in enumerate(vectors(X, key, E, T, S, patch_layer)):
        if s not in cache:
            cache[s] = {l: make_dirs(X, key, E, T, l, train) for l in read_layers}
        marked = f"{lead} [[span: {spans[key(s, e1, t)]}]]"
        if vec is None:
            full = extract(lm, marked, keep_resid=False).roles["span"]
        else:
            with lm.patched([Patch(patch_layer, add_vector(vec, scale))]):
                full = extract(lm, marked, keep_resid=False).roles["span"]
        for l in read_layers:
            deR, dtR, muR = cache[s][l]
            er, tr = classify(full[l], deR, dtR, muR, E, T)
            out[l].append(dict(scene=s, e1=e1, t=t, e2=e2, cond=cond, era_read=er, theme_read=tr))
        if i % 20 == 0:
            print(f"  case {i} {s}/{e1}/{t} {cond}", flush=True)
    return out


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["passthrough", "model", "report"])
    ap.add_argument("--grid")
    ap.add_argument("--stacks")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--reads", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.mode in ("passthrough", "model"):
        g, X, E, T, S, key, n_layers = load(a.grid, a.stacks)
        reads = [int(x) for x in a.reads.split(",")] if a.reads else [a.layer, a.layer + 2, a.layer + 6, n_layers - 1]
        res = {"grid": a.grid, "stacks": a.stacks, "patch_layer": a.layer, "n_layers": n_layers,
               "read_layers": reads, "scale": a.scale, "arms": {}}
        if a.mode == "passthrough":
            for l in reads:
                res["arms"][str(l)] = score(passthrough_rows(X, key, E, T, S, a.layer, l, a.scale))
            # sanity check 1: zero shift must reproduce the unpatched readout exactly
            res["zero_shift"] = {str(l): score(passthrough_rows(X, key, E, T, S, a.layer, l, a.scale,
                                                               zero_shift=True)) for l in reads}
            # norm-matched variant: fair pass-through at deep read layers
            res["arms_norm_matched"] = {str(l): score(passthrough_rows(X, key, E, T, S, a.layer, l,
                                                                      a.scale, norm_match=True))
                                        for l in reads}
        else:
            from lsx import LM
            lm = LM.from_pretrained(a.model)
            rowsets = model_rows(lm, g, X, key, E, T, S, a.layer, reads, a.scale)
            for l in reads:
                res["arms"][str(l)] = score(rowsets[l])
            res["raw"] = {str(l): rowsets[l] for l in reads}
        json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "raw"}, indent=1))

    else:  # ---------------------------------------------------------------- report
        # spec json: [{name, model, grid, logged, passthrough, model_arm|null, logged_read}]
        spec = json.load(open(a.grid))
        rep = {"patch_layer": a.layer, "targets": {}}
        for s in spec:
            logged = score(json.load(open(s["logged"])))
            pt = json.load(open(s["passthrough"]))
            ma = json.load(open(s["model_arm"])) if s.get("model_arm") else None
            lr = str(s["logged_read"])
            gain = None
            if lr in pt["arms"]:
                gain = {k: round(logged[k] - pt["arms"][lr][k], 4)
                        for k in ("shift_moved", "shift_kept", "shift_stayed", "rand_moved")}
            curve = None
            if ma:
                curve = {l: {"model_moved": ma["arms"][l]["shift_moved"],
                             "passthrough_moved": pt["arms"].get(l, {}).get("shift_moved"),
                             "gain_moved": (round(ma["arms"][l]["shift_moved"] - pt["arms"][l]["shift_moved"], 4)
                                            if l in pt["arms"] else None),
                             "passthrough_nm_moved": pt.get("arms_norm_matched", {}).get(l, {}).get("shift_moved"),
                             "gain_moved_vs_nm": (round(ma["arms"][l]["shift_moved"] - pt["arms_norm_matched"][l]["shift_moved"], 4)
                                                  if l in pt.get("arms_norm_matched", {}) else None),
                             "model_kept": ma["arms"][l]["shift_kept"],
                             "passthrough_kept": pt["arms"].get(l, {}).get("shift_kept"),
                             "gain_kept": (round(ma["arms"][l]["shift_kept"] - pt["arms"][l]["shift_kept"], 4)
                                           if l in pt["arms"] else None),
                             "base_era_e1": ma["arms"][l]["base_era_e1"],
                             "base_theme_t": ma["arms"][l]["base_theme_t"]}
                         for l in ma["arms"]}
            rep["targets"][s["name"]] = {
                "model": s["model"], "grid": s["grid"], "logged_read_layer": s["logged_read"],
                "logged_model_numbers": logged,
                "passthrough_by_read_layer": pt["arms"],
                "passthrough_norm_matched_by_read_layer": pt.get("arms_norm_matched"),
                "zero_shift_sanity": pt["zero_shift"],
                "model_arm_by_read_layer": (ma["arms"] if ma else None),
                "gain_at_logged_read_layer": gain,
                "layer_curve": curve,
            }
        json.dump(rep, open(a.out, "w"), indent=1)
        print("wrote", a.out)
