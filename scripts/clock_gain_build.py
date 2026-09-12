"""Build prompts/clock_gain_v1.json: four arms derived mechanically from
prompts/time_translation_v3.json (docs/specs/clock_depth_gain_v1.md sec 2).

  A intact,  state-only : [[state: S(s,t,p)]]
  B shuffled state-only : [[state: shuffle_w(S(s,t,p))]]
  C phrase + t0 state   : = v3 control_prompts
  D phrase + state      : = v3 prompts

No new text. shuffle_w is a whitespace word shuffle with random.Random(hash((s,t,p))),
one fixed permutation per cell; terminal punctuation rides along with its word and the
first word is not re-capitalised. PYTHONHASHSEED is forced to 0 so hash() is stable.
"""
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":                 # make hash() deterministic
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import json
import random

V3 = json.load(open("prompts/time_translation_v3.json"))
S, TP, DT, NP_ = V3["subjects"], V3["timepoints"], V3["deltas"], V3["n_paraphrases"]
ST, PH = V3["states"], V3["phrases"]


def shuffle_w(text, key):
    w = text.split()
    random.Random(hash(key)).shuffle(w)
    return " ".join(w)


arms = {"A": {}, "B": {}, "C": {}, "D": {}}
for s in S:
    for t in TP:
        for p in range(NP_):
            k = f"{s}/{t}/p{p}"
            state = ST[s][t][p]
            arms["A"][k] = f"[[state: {state}]]"
            arms["B"][k] = f"[[state: {shuffle_w(state, (s, t, p))}]]"
            arms["C"][k] = V3["control_prompts"][k]
            arms["D"][k] = V3["prompts"][k]

# sanity: C/D must match the v3 files exactly, and B must preserve A's word multiset
for s in S:
    for t in TP:
        for p in range(NP_):
            k = f"{s}/{t}/p{p}"
            assert arms["C"][k] == V3["control_prompts"][k]
            assert arms["D"][k] == V3["prompts"][k]
            wa = arms["A"][k][len("[[state: "):-2].split()
            wb = arms["B"][k][len("[[state: "):-2].split()
            assert sorted(wa) == sorted(wb) and wa != wb

out = dict(
    _note="clock_depth_gain_v1.md sec2: four arms over the v3 grid, no new text. "
          "B = word-shuffled A (PYTHONHASHSEED=0, random.Random(hash((s,t,p)))). "
          "C = v3 control_prompts, D = v3 prompts.",
    _layers=list(range(29)),
    roles=V3["roles"], subjects=S, timepoints=TP, deltas=DT, phrases=PH,
    log10_dt_days=V3["log10_dt_days"], n_paraphrases=NP_, states=ST,
    arms=arms,
    prompts=arms["D"], control_prompts=arms["C"],   # so time_translation*.py can read this file
)
json.dump(out, open("prompts/clock_gain_v1.json", "w"), indent=1)
print("wrote prompts/clock_gain_v1.json", {k: len(v) for k, v in arms.items()})
print("A:", arms["A"]["street/10years/p0"][:130])
print("B:", arms["B"]["street/10years/p0"][:130])
