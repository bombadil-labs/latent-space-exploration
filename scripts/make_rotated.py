"""Position-balanced grid: the six holonic spans of each domain, presented with role-NEUTRAL
connectives in six cyclic rotations (a Latin square), so every content role occupies every slot
exactly once per domain. Roles are labeled by CONTENT. Writes prompts/<grid>_rotated.json."""
import json, re, sys
src = sys.argv[1]; g = json.load(open(src)); roles = g["roles"]
pat = re.compile(r"\[\[(\w+):\s*(.*?)\]\]", re.S)
LEAD = {"physics": "Consider a change in physics.", "psychology": "Consider a change in child development.",
        "music": "Consider a change in a tonal piece.", "law": "Consider a change in contract law.",
        "software": "Consider a change in a growing codebase.", "biology": "Consider a change in the history of life.",
        "mathematics": "Consider a change in arithmetic.", "narrative": "Consider a change in a story."}
ORD = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]
out = {}
for k, p in g["prompts"].items():
    if not k.endswith("/holonic"): continue
    d = k.split("/")[0]; spans = pat.findall(p); assert [r for r, _ in spans] == roles
    for rot in range(len(roles)):
        order = [(i + rot) % len(roles) for i in range(len(roles))]
        body = " ".join(f"{ORD[s]}, [[{spans[c][0]}: {spans[c][1]}]]." for s, c in enumerate(order))
        out[f"{d}/rot{rot}"] = f"{LEAD[d]} {body}"
dst = src.replace(".json", "_rotated.json")
json.dump({"_note": "Latin-square rotations of holonic spans with neutral connectives; roles labeled by content. slot(role) = (content_index - rot) mod 6.",
           "roles": roles, "prompts": out}, open(dst, "w"), indent=1)
print("wrote", dst, len(out), "prompts"); print(out["music/rot2"][:300])
