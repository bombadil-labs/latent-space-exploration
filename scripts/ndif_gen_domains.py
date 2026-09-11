"""Generate holonic-schema candidates in many domains with an NDIF-hosted instruct model, for human
checking. Writes prompts/holonic_candidates.jsonl (one JSON object per domain, six role spans)."""
import json, re, sys, time
import torch
from nnsight import LanguageModel
from lsx.ndif import ProxyAuthBackend
MODEL = "google/gemma-2-9b-it"
DOMAINS = ["economics", "chess", "cooking", "parenting", "architecture", "linguistics", "geology", "jazz improvisation",
           "immunology", "urban planning", "painting", "friendship", "religion", "cartography", "grief", "programming languages",
           "ecology", "translation", "medicine", "theatre", "climbing", "philosophy of science", "farming", "diplomacy",
           "typography", "astronomy", "carpentry", "romantic love", "sports coaching", "poetry", "memory", "democracy"]
SPEC = """You are helping build a dataset of a specific structure, a subject/object transition (after Robert Kegan), rendered in many domains.

The structure has six roles. Fill each with ONE clause of 12-25 words, in plain prose, present or past tense, no names of people:
- embedded: the thing one simply IS at first; a subject one looks THROUGH and cannot see as a thing.
- disturbance: what makes that embeddedness fail; a demand the embedded stance cannot meet.
- objectified: the SAME thing once it can be held, seen, and chosen; now an object rather than a subject.
- new_subject: the larger frame one now IS, which holds the old thing as one part among others.
- from_below: how the transition looks from the earlier stage (typically like loss, betrayal, confusion).
- from_above: how it looks from the later stage (typically: the old stance was a special case, never wrong, only partial).

Example, domain = physics:
embedded: a particle simply had a position and a momentum, and the physicist was inside that picture without seeing it as a picture
disturbance: measurements at small scales refused to give both quantities together
objectified: the classical trajectory became one description among others, something to be chosen and examined
new_subject: a state vector from which trajectories are derived only as limits
from_below: this looks like the loss of reality, as if nature has become vague
from_above: the earlier picture was a special case that was never wrong, only partial

Now do domain = {domain}. Output exactly six lines in the format `role: clause`, nothing else."""
model = LanguageModel(MODEL, device_map="auto", dispatch=False); tok = model.tokenizer
out = open("prompts/holonic_candidates.jsonl", "a"); done = set()
try:
    for line in open("prompts/holonic_candidates.jsonl"): done.add(json.loads(line)["domain"])
except Exception: pass
for d in DOMAINS:
    if d in done: continue
    prompt = tok.apply_chat_template([{"role": "user", "content": SPEC.replace("{domain}", d)}], tokenize=False, add_generation_prompt=True)
    backend = ProxyAuthBackend(model.to_model_key())
    with model.generate(prompt, max_new_tokens=260, do_sample=False, backend=backend) as tracer:
        o = model.generator.output.save()
    res = backend.wait(tracer); ids = next(x for x in res.values() if isinstance(x, torch.Tensor)); ids = ids[0] if ids.dim() == 2 else ids
    text = tok.decode(ids[len(tok(prompt)["input_ids"]):], skip_special_tokens=True)
    roles = {}
    for m in re.finditer(r"^\s*[-*]?\s*\**(embedded|disturbance|objectified|new_subject|from_below|from_above)\**\s*:\s*(.+?)\s*$", text, re.M | re.I):
        roles[m.group(1).lower()] = m.group(2).strip().rstrip(".")
    rec = dict(domain=d, model=MODEL, roles=roles, raw=text if len(roles) < 6 else None, ok=len(roles) == 6)
    out.write(json.dumps(rec) + "\n"); out.flush(); print(d, "ok" if rec["ok"] else f"PARSE FAIL ({len(roles)})", flush=True)
