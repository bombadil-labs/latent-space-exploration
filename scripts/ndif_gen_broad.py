"""Generate a broad reference corpus (~120 short passages across 12 genres) with Gemma-2-9B-it via NDIF.

One job per genre: the model is asked for N numbered paragraphs of ~55 words in that genre; the reply
is parsed into passages. Writes results/broad_corpus.json = {id: text}, ids '<genre>/<i>'.
"""
import argparse, json, os, re
import torch
from nnsight import LanguageModel
from lsx.ndif import ProxyAuthBackend, retry_job

GENRES = {
    "news": "a wire-service news report about a local event",
    "dialogue": "a snippet of dialogue between two people, with speech tags",
    "technical": "technical prose from a software engineering document",
    "recipe": "a cooking recipe step section with ingredients named",
    "legal": "a clause from a contract or statute, in legal register",
    "poetry": "a short free-verse poem",
    "sports": "a sports match report",
    "science": "a plain-language explanation of a scientific idea",
    "email": "a personal email to a friend",
    "product": "an e-commerce product description",
    "history": "a paragraph from a popular history book",
    "howto": "a how-to instruction paragraph for a household task",
}
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="google/gemma-2-9b-it")
ap.add_argument("--per_genre", type=int, default=10)
ap.add_argument("--out", default="results/broad_corpus.json")
a = ap.parse_args()

model = LanguageModel(a.model, device_map="auto", dispatch=False); tok = model.tokenizer
out = json.load(open(a.out)) if os.path.exists(a.out) else {}

TMPL = ("Write {n} different short passages, each about 55 words, each an example of {desc}. "
        "Vary the subject matter completely between passages. "
        "Number them 1. to {n}. and write nothing else - no titles, no commentary.")

for g, desc in GENRES.items():
    have = [k for k in out if k.startswith(g + "/")]
    if len(have) >= a.per_genre:
        print(g, "have", len(have), flush=True); continue
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": TMPL.format(n=a.per_genre + 2, desc=desc)}],
        tokenize=False, add_generation_prompt=True)

    def job():
        backend = ProxyAuthBackend(model.to_model_key())
        with model.generate(prompt, max_new_tokens=1100, do_sample=False, backend=backend) as tracer:
            o = model.generator.output.save()
        res = backend.wait(tracer, timeout=900)
        return next(x for x in res.values() if isinstance(x, torch.Tensor))

    ids = retry_job(job)
    ids = ids[0] if ids.dim() == 2 else ids
    text = tok.decode(ids[len(tok(prompt)["input_ids"]):], skip_special_tokens=True)
    parts = re.split(r"(?m)^\s*\d+\.\s*", text)
    parts = [re.sub(r"\s+", " ", p).strip().strip("*").strip() for p in parts[1:]]
    parts = [p for p in parts if len(p.split()) >= 25]
    for i, p in enumerate(parts[: a.per_genre]):
        out[f"{g}/{i}"] = p
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{g}: {len(parts)} parsed, kept {min(len(parts), a.per_genre)} (total {len(out)})", flush=True)
print("saved", a.out, len(out))
