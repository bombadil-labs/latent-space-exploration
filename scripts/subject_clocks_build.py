"""Build prompts/subject_clocks_v1.json from prompts/time_translation_v2.json.

Spec: docs/specs/subject_clocks_v1.md section 1. Two-timepoint constructions:

  C1 (author-supplied change)  [[interval0: At first,]] [[state0: S(s,t0,p)]]
                               [[interval: PH(dt)]] [[state: S(s,dt,p)]]
     + t0 row: null phrase, state = S(s,t0,(p+1)%3)              8*9*3 + 8*3   = 240
  C3 (model-supplied, token-aligned) same but state = S(s,t0,p) verbatim
     + null rows, both null phrases                              8*9*3 + 8*3*2 = 264

Total 504 passages.  Also emits the decoding prefixes/candidates for piece 2 (sec 3.6).
"""
import json
import os

SRC = "prompts/time_translation_v2.json"
OUT = "prompts/subject_clocks_v1.json"
NULLS = {"null1": "At that same moment,", "null2": "Just then,"}
OPEN = "At first,"

G = json.load(open(SRC))
S, DT, PH, ST = G["subjects"], G["deltas"], G["phrases"], G["states"]
NP = G["n_paraphrases"]


def passage(s0, ph, st):
    return f"[[interval0: {OPEN}]] [[state0: {s0}]] [[interval: {ph}]] [[state: {st}]]"


C1, C3 = {}, {}
for s in S:
    for p in range(NP):
        t0 = ST[s]["t0"][p]
        for t in DT:
            C1[f"{s}/{t}/p{p}"] = passage(t0, PH[t], ST[s][t][p])
            C3[f"{s}/{t}/p{p}"] = passage(t0, PH[t], t0)
        nk = ["null1", "null2"][p % 2]
        C1[f"{s}/t0/p{p}"] = passage(t0, NULLS[nk], ST[s]["t0"][(p + 1) % NP])
        for nk2 in NULLS:
            C3[f"{s}/{nk2}/p{p}"] = passage(t0, NULLS[nk2], t0)

dec_pref = {f"{s}/{i}/p{p}": f"At first, {ST[s]['t0'][p]} " + (NULLS[i] if i in NULLS else PH[i])
            for s in S for p in range(NP) for i in list(NULLS) + DT}
dec_cand = {f"{s}/{j}/p{p}": " " + ST[s][j][p] for s in S for p in range(NP) for j in ["t0"] + DT}

D = dict(_note="subject_clocks v1, built from " + SRC + " per docs/specs/subject_clocks_v1.md sec 1",
         source=SRC, subjects=S, deltas=DT, timepoints=G["timepoints"], phrases=PH,
         null_phrases=NULLS, n_paraphrases=NP, log10_dt_days=G["log10_dt_days"],
         c1_null_phrase_by_p={str(p): ["null1", "null2"][p % 2] for p in range(NP)},
         C1=C1, C3=C3, decode_prefixes=dec_pref, decode_candidates=dec_cand)
os.makedirs("prompts", exist_ok=True)
json.dump(D, open(OUT, "w"), indent=1)
print(f"C1 {len(C1)}  C3 {len(C3)}  total {len(C1) + len(C3)}  "
      f"decode {len(dec_pref)} prefixes x {len(dec_cand)} candidates")
