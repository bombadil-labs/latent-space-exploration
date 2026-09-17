"""Phase 0.1: does the log-prob selector effect travel the residual SKIP path?

docs/specs/selector_direct_path_v1.md (v2).  A direction `d` added at the input of block L reaches
the unembedding whether or not any block uses it:

    pre_28[t] = resid_L[t] + d + sum_{l=L..27} block_l(.)     logit[t] = W_U . norm(pre_28[t])

so "the stack computes the relation" needs the treatment to beat the direct path *plus whatever the
stack did along d*.  Arms, per case (spec §3):

  N        no-patch                       -> rank exactly chance, all gains 0.0
  A        s.d_L at input of block L      -> the logged instrument
  A_span   same, scored positions only
  R        equal-norm Gaussian at L, x2   -> m ~ 0; sets the noise floor tau
  F_delta  pre_28^A - pre_28^base offline -> must equal A to 1e-4 nats (gate)
  F_abs    s.d_L at pre_28, offline       -> the literal skip term
  F_par    ((Delta.dhat)dhat) at pre_28   -> THE NULL: skip term rescaled by the stack
  F_dose   c.s.d_L at pre_28              -> H_direct as a one-parameter family
  F_KL     the dose whose mean KL(base||.) equals the treatment's
  F_R      Gaussian at pre_28, c=1, c_KL  -> control on the control
  U        d_L . W_U[span tokens]         -> which direct path (no forward)
  P        unembedding-built direction    -> POSITIVE CONTROL: must read as direct (gate 5)
  C        s.d_L at LEAD positions only   -> direct path removed by construction
  C_plumb  big Gaussian at lead positions -> C's plumbing: must move the model

Primary statistic: G_new = mean_cases(m_A - m_F_par) in nats, with paired sign fraction and a 90%
bootstrap lower bound.  Ranks are descriptive (INSTRUMENTS.md §1).

Directions, prompt assembly and candidate sets are copied from scripts/stage4.py (role) and
scripts/stage6_factors.py (factors) so the reproduction gate compares like with like.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time

import numpy as np
import torch

from lsx import LM
from lsx.model import Patch
from lsx.steer import add_vector

DOSES = [0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0, 12.0, 16.0]


# ---------------------------------------------------------------- scored text (offline machinery)

class Scored:
    """One candidate continuation: its pre-norm residual, its scored positions, its offline readout."""

    def __init__(self, lm: LM, prefix: str, cont: str, patches=None):
        self.lm = lm
        enc_p, _ = lm.encode(prefix)
        enc_f, _ = lm.encode(prefix + cont)
        self.n_p = enc_p["input_ids"].shape[1]
        self.ids = enc_f["input_ids"][0].cpu()
        pre, r, logits = lm.pre_norm_residual(prefix + cont, patches)
        self.r = r
        # scored positions: logit index t = n_p-1 .. len(ids)-2, predicting ids[t+1]
        self.pre = pre[self.n_p - 1:-1]                      # [npos, d]
        self.tgt = self.ids[self.n_p:]                       # [npos]
        self.npos = len(self.tgt)
        # keep the log-softmax rows, not the raw logits: they are what KL and the gains need, and
        # one [npos, V] copy per candidate text is already the memory ceiling of this script.
        self.lsm = torch.log_softmax(logits[self.n_p - 1:-1].float(), -1)
        self.lp = float(self.lsm.gather(1, self.tgt[:, None]).sum())
        self.span_tokens = self.tgt.tolist()

    def final(self, v=None, want_logprobs: bool = False):
        """Offline readout: add `v` ([d] or [npos, d]) to pre_28 at the scored positions."""
        norm, head = self.lm.final_norm(), self.lm.head()
        with torch.no_grad():
            x = self.pre if v is None else self.pre + torch.as_tensor(v, dtype=torch.float32)
            lsm = torch.log_softmax(head(norm(x)).float(), -1)
            lp = float(lsm.gather(1, self.tgt[:, None]).sum())
        return (lp, lsm) if want_logprobs else (lp, None)


def kl_rows(base_lsm: torch.Tensor, other_lsm: torch.Tensor) -> float:
    """mean over scored positions of KL(p_base || p_other)."""
    p = base_lsm.exp()
    return float((p * (base_lsm - other_lsm)).sum(-1).mean())


# ---------------------------------------------------------------- statistics

def midrank(scores: list[float], t: int) -> float:
    return (1.0 + sum(s > scores[t] for i, s in enumerate(scores) if i != t)
            + 0.5 * sum(s == scores[t] for i, s in enumerate(scores) if i != t))


def margin_and_rank(gains: dict, subcases: list[tuple]) -> tuple[float, float]:
    """subcases: [(target_key, [sibling_keys...]), ...]; margin in nats and mid-rank, averaged."""
    ms, rs = [], []
    for tk, sibs in subcases:
        ms.append(gains[tk] - float(np.mean([gains[s] for s in sibs])))
        field = [gains[tk]] + [gains[s] for s in sibs]
        rs.append(midrank(field, 0))
    return float(np.mean(ms)), float(np.mean(rs))


# ---------------------------------------------------------------- one case

def run_case(lm, prefix, cands, subcases, d, s, rng, layer, case_meta, rows,
             n_rand=2, do_plumb=False, doses=DOSES, shared=None,
             arms=("A_span", "C", "P")):
    """cands: {key: continuation}.  d: the treatment direction (np [d_model]).

    `shared` caches the two things that depend only on the candidate set and not on the direction:
    the base pass and the no-patch gains (both deterministic, so recomputing them per case would
    return the same numbers).  Every case still RECORDS its N arm.
    """
    keys = list(cands)
    shared = shared if shared is not None else {}
    L = layer
    dn = float(np.linalg.norm(d))
    dhat = torch.as_tensor(d / dn, dtype=torch.float32)
    dvec = torch.as_tensor(d, dtype=torch.float32) * s

    if "base" not in shared:
        shared["base"] = {k: Scored(lm, prefix, cands[k]) for k in keys}
    base_scored = shared["base"]
    base_lp = {k: base_scored[k].lp for k in keys}
    n_p = base_scored[keys[0]].n_p

    KEEP_GAINS = {"A", "F_par", "F_KL", "R0", "N"}   # needed for the X variance decomposition

    def record(arm, gains, **extra):
        m, rk = margin_and_rank(gains, subcases)
        if arm in KEEP_GAINS:
            extra["gains"] = [float(gains[k]) for k in keys]
        rows.append(dict(**case_meta, layer=L, arm=arm, m=m, rank=rk,
                         gain_target=float(np.mean([gains[t] for t, _ in subcases])),
                         gain_others=float(np.mean([np.mean([gains[x] for x in sb])
                                                    for _, sb in subcases])), **extra))
        return m

    out = {}

    if arms == ("P_ONLY",):
        # gate-5 pass on its own: the positive control and its decomposition, nothing else.
        # Used to give test D the gate the first pass skipped.
        W = lm.head().weight
        tgt_keys = [t for t, _ in subcases]
        oth_keys = sorted({x for _, sb in subcases for x in sb}, key=lambda k: keys.index(k))
        with torch.no_grad():
            mt = torch.stack([W[torch.as_tensor(base_scored[k].span_tokens)].float().mean(0)
                              for k in tgt_keys]).mean(0)
            mo = torch.stack([W[torch.as_tensor(base_scored[k].span_tokens)].float().mean(0)
                              for k in oth_keys]).mean(0)
            dP = (mt - mo)
            dP = (dP / dP.norm() * dn).numpy()
        dPhat = torch.as_tensor(dP / np.linalg.norm(dP), dtype=torch.float32)
        dPvec = torch.as_tensor(dP, dtype=torch.float32) * s
        gP, gPa, gPp = {}, {}, {}
        for k in keys:
            b = base_scored[k]
            sc = Scored(lm, prefix, cands[k], [Patch(L, add_vector(dP, s))])
            gP[k] = sc.lp - base_lp[k]
            delta = sc.pre - b.pre
            par = (delta @ dPhat)[:, None] * dPhat[None, :]
            gPa[k] = b.final(dPvec)[0] - base_lp[k]
            gPp[k] = b.final(par)[0] - base_lp[k]
        record("P", gP)
        record("P_abs", gPa)
        record("P_par", gPp)
        return out, shared

    # ---- N: no patch, identical code path
    if "N" not in shared:
        shared["N"] = {k: lm.logprob(prefix, cands[k], []) - base_lp[k] for k in keys}
    gN = shared["N"]
    out["N"] = record("N", gN, max_abs_gain=float(max(abs(g) for g in gN.values())))

    # ---- A: treatment, all positions.  pre_norm_residual so the offline arms are free.
    A = {k: Scored(lm, prefix, cands[k], [Patch(L, add_vector(d, s))]) for k in keys}
    gA = {k: A[k].lp - base_lp[k] for k in keys}
    out["A"] = record("A", gA)

    # ---- A_span: same direction, scored positions only
    if "A_span" in arms:
        gAs = {k: lm.logprob(prefix, cands[k],
                             [Patch(L, add_vector(d, s, positions=slice(n_p - 1, None)))]) - base_lp[k]
               for k in keys}
        out["A_span"] = record("A_span", gAs)

    # ---- R: equal-norm Gaussian at L
    for j in range(n_rand):
        v = rng.normal(size=d.shape)
        v = v / np.linalg.norm(v) * dn
        gR = {k: lm.logprob(prefix, cands[k], [Patch(L, add_vector(v, s))]) - base_lp[k] for k in keys}
        out[f"R{j}"] = record(f"R{j}", gR)

    # ---- C: lead positions only (no scored position is patched)
    if "C" in arms:
        gC = {k: lm.logprob(prefix, cands[k],
                            [Patch(L, add_vector(d, s, positions=slice(0, n_p - 1)))]) - base_lp[k]
              for k in keys}
        out["C"] = record("C", gC)

    if do_plumb:
        v = rng.normal(size=d.shape)
        v = v / np.linalg.norm(v) * dn * 4.0
        gCp = {k: lm.logprob(prefix, cands[k],
                             [Patch(L, add_vector(v, s, positions=slice(0, n_p - 1)))]) - base_lp[k]
               for k in keys}
        # no scored position is patched, so the direct (pre_28) arm of C is identically zero
        record("C_plumb", gCp, max_abs_gain=float(max(abs(g) for g in gCp.values())), offline_gain=0.0)

    # ---- offline arms off A's pre_28
    surv, orth = [], []
    gFd, gFa, gFp = {}, {}, {}
    gdose = {c: {} for c in doses}
    kl_A, base_lsm = {}, {}
    for k in keys:
        b, a = base_scored[k], A[k]
        delta = a.pre - b.pre                                   # [npos, d]
        par = (delta @ dhat)[:, None] * dhat[None, :]
        surv.append(float((delta @ dhat).mean() / (s * dn)))
        orth.append(float((delta - par).norm(dim=-1).mean() / (s * dn)))
        base_lsm[k] = b.lsm
        gFd[k] = b.final(delta)[0] - base_lp[k]
        gFa[k] = b.final(dvec)[0] - base_lp[k]
        gFp[k] = b.final(par)[0] - base_lp[k]
        for c in doses:
            gdose[c][k] = b.final(dvec * c)[0] - base_lp[k]
        kl_A[k] = kl_rows(base_lsm[k], a.lsm)
        a.lsm = None                                  # free [npos, V]; only Delta is needed now

    record("F_delta", gFd, fdelta_err=max(abs(gFd[k] - gA[k]) for k in keys))
    out["F_abs"] = record("F_abs", gFa)
    out["F_par"] = record("F_par", gFp, survival=float(np.mean(surv)), orth=float(np.mean(orth)))
    for c in doses:
        record("F_dose", gdose[c], dose=c)

    # ---- F_KL: one dose per case matching the treatment's mean KL
    target_kl = float(np.mean([kl_A[k] for k in keys]))

    def mean_kl(c):
        return float(np.mean([kl_rows(base_lsm[k],
                                      base_scored[k].final(dvec * c, want_logprobs=True)[1])
                              for k in keys]))

    lo, hi = 1e-3, 40.0
    kl_lo, kl_hi = mean_kl(lo), mean_kl(hi)
    monotone = kl_hi > kl_lo
    if kl_hi < target_kl:
        c_kl, kl_hit = hi, kl_hi
    elif kl_lo > target_kl:
        c_kl, kl_hit = lo, kl_lo
    else:
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            if mean_kl(mid) < target_kl:
                lo = mid
            else:
                hi = mid
        c_kl = 0.5 * (lo + hi)
        kl_hit = mean_kl(c_kl)
    gKL = {k: base_scored[k].final(dvec * c_kl)[0] - base_lp[k] for k in keys}
    out["F_KL"] = record("F_KL", gKL, c_kl=c_kl, kl_A=target_kl, kl_hit=kl_hit, kl_monotone=monotone)

    # ---- F_R: Gaussian at pre_28 at c = 1 and c = c_kl
    for c, tag in ((1.0, "1"), (c_kl, "kl")):
        for j in range(2):
            v = rng.normal(size=d.shape)
            v = v / np.linalg.norm(v) * dn * s * c
            g = {k: base_scored[k].final(torch.as_tensor(v, dtype=torch.float32))[0] - base_lp[k]
                 for k in keys}
            record(f"F_R_{tag}{j}", g, dose=c)

    # ---- U: unembedding-only, no forward
    W = lm.head().weight
    with torch.no_grad():
        scoreU = {k: float((W[torch.as_tensor(base_scored[k].span_tokens)].float() @ dhat).sum())
                     / max(1, base_scored[k].npos) for k in keys}
    mU, rU = margin_and_rank(scoreU, subcases)
    rows.append(dict(**case_meta, layer=L, arm="U", m=mU, rank=rU,
                     gain_target=float(np.mean([scoreU[t] for t, _ in subcases])),
                     gain_others=float(np.mean([np.mean([scoreU[x] for x in sb])
                                                for _, sb in subcases]))))

    # ---- P: pure-direct positive control, built from the unembedding rows
    if "P" not in arms:
        return out, shared
    tgt_keys = [t for t, _ in subcases]
    oth_keys = sorted({x for _, sb in subcases for x in sb}, key=lambda k: keys.index(k))
    with torch.no_grad():
        mt = torch.stack([W[torch.as_tensor(base_scored[k].span_tokens)].float().mean(0)
                          for k in tgt_keys]).mean(0)
        mo = torch.stack([W[torch.as_tensor(base_scored[k].span_tokens)].float().mean(0)
                          for k in oth_keys]).mean(0)
        dP = (mt - mo)
        dP = (dP / dP.norm() * dn).numpy()
    Pp = {k: Scored(lm, prefix, cands[k], [Patch(L, add_vector(dP, s))]) for k in keys}
    gP = {k: Pp[k].lp - base_lp[k] for k in keys}
    dPhat = torch.as_tensor(dP / np.linalg.norm(dP), dtype=torch.float32)
    dPvec = torch.as_tensor(dP, dtype=torch.float32) * s
    gPa, gPp = {}, {}
    for k in keys:
        b = base_scored[k]
        delta = Pp[k].pre - b.pre
        par = (delta @ dPhat)[:, None] * dPhat[None, :]
        gPa[k] = b.final(dPvec)[0] - base_lp[k]
        gPp[k] = b.final(par)[0] - base_lp[k]
        Pp[k].lsm = None
    out["P"] = record("P", gP)
    out["P_abs"] = record("P_abs", gPa)
    out["P_par"] = record("P_par", gPp)

    return out, shared


# ---------------------------------------------------------------- role lens (stage4)

def run_role(a, lm, rows, meta):
    g = json.load(open(a.grid))
    roles = g["roles"]
    pat = re.compile(r"\[\[(\w+):\s*(.*?)\]\]", re.S)
    z = np.load(a.stacks)
    stacks = {k: z[k] for k in z.files if k != "roles"}
    doms = sorted({k.split("/")[0] for k in stacks})
    R = len(roles)
    avg = {d: np.mean([stacks[f"{d}/rot{r}"] for r in range(R)], axis=0) for d in doms}
    spans, lead = {}, {}
    for d in doms:
        p = g["prompts"][f"{d}/rot0"]
        lead[d] = p.split(" First,")[0]
        spans[d] = {r: t for r, t in pat.findall(p)}
    layers = [int(x) for x in a.layers.split(",")]
    rng = np.random.default_rng(0)
    meta.setdefault("r_L", {})
    for L in layers:
        for d in (doms if not a.domains else a.domains.split(",")):
            train = [x for x in doms if x != d]
            M = np.mean([avg[x] for x in train], axis=0)                  # [29, 6, 1536]
            dirs = M[L] - M[L].mean(0, keepdims=True)
            prefix = f"{lead[d]} First,"
            cands = {r: f" {spans[d][r]}." for r in roles}
            shared = {}
            for Ri, Rname in enumerate(roles):
                subcases = [(Rname, [r for r in roles if r != Rname])]
                cm = dict(claim="role", case=f"{d}/{Rname}", cluster=d,
                          dnorm=float(np.linalg.norm(dirs[Ri])))
                t0 = time.time()
                _, shared = run_case(lm, prefix, cands, subcases, dirs[Ri], a.scale, rng, L, cm,
                                     rows, n_rand=2, do_plumb=(Ri == 0), shared=shared)
                print(f"role L{L} {d}/{Rname} {time.time() - t0:.1f}s", flush=True)
            meta["r_L"][str(L)] = float(shared["base"][roles[0]].r[L])
            meta["r_28"] = float(shared["base"][roles[0]].r[lm.n_layers])


# ---------------------------------------------------------------- factor lenses (stage6 B, D)

def run_factors(a, lm, rows, meta):
    g = json.load(open(a.grid))
    z = np.load(a.stacks)
    X = {k: z[k] for k in z.files}
    F = g["factors"]
    names = list(F)
    S = g["scenes"]
    lead = g["lead"]
    spans = g["spans"]
    combos = list(itertools.product(*[F[n] for n in names]))

    def key(s, c):
        return "/".join([s, *c])

    def dirs(l, train):
        allv = {c: np.stack([X[key(s, c)][l] for s in train]).mean(0) for c in combos}
        mu = np.mean(list(allv.values()), axis=0)
        return {n: {lvl: np.mean([allv[c] for c in combos if c[i] == lvl], axis=0) - mu
                    for lvl in F[n]} for i, n in enumerate(names)}, mu

    layers = [int(x) for x in a.layers.split(",")]
    rng = np.random.default_rng(0)
    meta.setdefault("r_L", {})
    for L in layers:
        for s in (S if not a.scenes else a.scenes.split(",")):
            D, _ = dirs(L, [x for x in S if x != s])
            cands = {c: f" {spans[key(s, c)]}" for c in combos}
            shared = {}
            # ---- test B: each factor as a lens, other factors fixed
            for i, n in (list(enumerate(names)) if "B" in a.tests else []):
                for lvl in F[n]:
                    subcases = []
                    for c in combos:
                        if c[i] == lvl:
                            sibs = [cc for cc in combos if cc != c
                                    and all(cc[j] == c[j] for j in range(len(names)) if j != i)]
                            subcases.append((c, sibs))
                    cm = dict(claim=f"B/{n}", case=f"{s}/{n}/{lvl}", cluster=s,
                              dnorm=float(np.linalg.norm(D[n][lvl])))
                    t0 = time.time()
                    _, shared = run_case(lm, lead, cands, subcases, D[n][lvl], a.scale, rng, L,
                                         cm, rows, n_rand=2,
                                         do_plumb=(i == 0 and lvl == F[n][0]), shared=shared)
                    print(f"B L{L} {s}/{n}/{lvl} {time.time() - t0:.1f}s", flush=True)
            # ---- test D: all three composed, ranked among all 18
            if "D" in a.tests or "P" in a.tests:
                for c in combos:
                    dsum = sum(D[n][c[i]] for i, n in enumerate(names))
                    subcases = [(c, [cc for cc in combos if cc != c])]
                    cm = dict(claim="D", case=f"{s}/{'/'.join(c)}", cluster=s,
                              dnorm=float(np.linalg.norm(dsum)))
                    t0 = time.time()
                    _, shared = run_case(lm, lead, cands, subcases, dsum, a.scale, rng, L, cm,
                                         rows, n_rand=1, do_plumb=False, shared=shared,
                                         # D: A + R + N + the offline family (spec §3 row D);
                                         # "P" alone re-runs only the gate-5 positive control.
                                         arms=("P_ONLY",) if "P" in a.tests else ())
                    print(f"D L{L} {s}/{'/'.join(c)} {time.time() - t0:.1f}s", flush=True)
            meta["r_L"][str(L)] = float(shared["base"][combos[0]].r[L])
            meta["r_28"] = float(shared["base"][combos[0]].r[lm.n_layers])


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["role", "factors"])
    ap.add_argument("grid")
    ap.add_argument("stacks")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--layers", default="20")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--tests", default="BD", help="factors mode: which of B, D to run")
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--domains", default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    import transformers
    t0 = time.time()
    lm = LM.from_pretrained(a.model)
    rows = []
    meta = dict(mode=a.mode, grid=a.grid, tests=a.tests, scenes=a.scenes, domains=a.domains, model=a.model, layers=a.layers, scale=a.scale,
                python=sys.version.split()[0], torch=torch.__version__,
                transformers=transformers.__version__, doses=DOSES,
                tied_embeddings=bool(lm.model.lm_head.weight.data_ptr()
                                     == lm.model.model.embed_tokens.weight.data_ptr()))
    (run_role if a.mode == "role" else run_factors)(a, lm, rows, meta)
    meta["seconds"] = time.time() - t0
    json.dump(dict(meta=meta, rows=rows), open(a.out, "w"))
    print("saved", a.out, len(rows), "rows in", round(meta["seconds"], 1), "s")


if __name__ == "__main__":
    main()
