import json
d = json.load(open("prompts/time_translation_v2.json"))
FAR = {"100years", "1000years", "10000years", "1000000years"}
for key in list(d["prompts"].keys()):
    s, t, p = key.split("/")
    if t in FAR:
        pi = int(p[1:])
        interval = d["phrases"][t]
        state = d["states"][s][t][pi]
        d["prompts"][key] = f"[[interval: {interval}]] [[state: {state}]]"
json.dump(d, open("prompts/time_translation_v2.json", "w"), indent=1)
print("resynced prompts")
