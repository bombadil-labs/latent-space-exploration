"""Shuffled-holonic control: keep each holonic prompt's connective template and spans, but permute
which span sits in which slot, with a different derangement per domain. Roles are labeled by SLOT.
Writes prompts/<grid>_shuffled.json containing holonic, flat, and shuffled framings."""
import json, re, sys
import numpy as np
src = sys.argv[1]; g = json.load(open(src)); roles = g["roles"]
rng = np.random.default_rng(7)
pat = re.compile(r"\[\[(\w+):\s*(.*?)\]\]", re.S)
def derangement(n):
    while True:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)): return p
out = dict(g["prompts"]); perms = {}
for k, p in g["prompts"].items():
    if not k.endswith("/holonic"): continue
    spans = pat.findall(p); assert [r for r, _ in spans] == roles
    perm = derangement(len(roles)); perms[k.split('/')[0]] = perm.tolist()
    it = iter(perm)
    new = pat.sub(lambda m: f"[[{m.group(1)}: {spans[next(it)][1]}]]", p)
    out[k.replace("/holonic", "/shuffled")] = new
g["prompts"] = out; g["shuffle_perms"] = perms  # slot i holds original role perms[d][i]
g["_note"] += " | shuffled: holonic spans permuted across slots (different derangement per domain), roles labeled by slot."
dst = src.replace(".json", "_shuffled.json"); json.dump(g, open(dst, "w"), indent=1); print("wrote", dst, len(out), "prompts")
