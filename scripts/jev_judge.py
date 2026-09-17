"""Score generated continuations with TypeSafe's Jev (a System One decision model).

Deliberately narrow so it can be granted a single Bash permission rule:
  - reads ONE local results JSON, sends ONLY the `cont` (continuation) field of each row,
  - POSTs to exactly https://api.typesafe.ai/v1/systemone and nothing else,
  - never prints, logs or writes the API key,
  - checkpoints after every row so a killed run resumes.

Usage:
  python3 scripts/jev_judge.py --in results/recompose_sweep_70b_3.0.json \
                               --out results/jev_recompose_sweep_70b_3.0.json [--limit N]

Env: TYPESAFE_API_KEY must be set.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

QUESTIONS = {
    "fluent": {
        "type": "noul",
        "instructions": "The text is fluent, grammatical English prose.",
        "criteria": {
            "true": "Reads as natural prose a competent writer produced.",
            "false": "Broken syntax, degenerate repetition, truncated mid-word, or nonsense.",
        },
    },
    "quality": {
        "type": "score",
        "instructions": "Rate the prose quality of this story continuation.",
        "criteria": [
            "Incoherent or degenerate.",
            "Grammatical but flat or generic.",
            "Competent prose.",
            "Vivid, controlled prose.",
        ],
    },
    "repetition": {
        "type": "noul",
        "instructions": "The text contains degenerate repetition (a word or phrase looping).",
        "criteria": {"true": "A word or phrase repeats in a way a writer would not.",
                     "false": "No abnormal repetition."},
    },
}


def ask(key: str, text: str, timeout: float = 60.0) -> dict:
    body = json.dumps({"model": MODEL, "state": text, "questions": QUESTIONS}).encode()
    req = urllib.request.Request(
        URL, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        print("TYPESAFE_API_KEY not set", file=sys.stderr)
        return 2

    rows = json.load(open(a.inp))["rows"]
    if a.limit:
        rows = rows[: a.limit]

    done = {}
    if os.path.exists(a.out):
        done = {r["id"]: r for r in json.load(open(a.out))["rows"]}
        print(f"resuming with {len(done)} scored")

    out, lat, fails = [], [], 0
    for i, r in enumerate(rows):
        rid = f"{r.get('scene')}|{r.get('e1')}|{r.get('t')}|{r.get('cond')}|{i}"
        if rid in done:
            out.append(done[rid])
            continue
        t = time.time()
        try:
            d = ask(key, r["cont"])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            fails += 1
            print(f"[{i}] FAILED {type(e).__name__}", file=sys.stderr)
            continue
        dt = time.time() - t
        lat.append(dt)
        ans = d["answers"]
        out.append({
            "id": rid, "scene": r.get("scene"), "e1": r.get("e1"), "e2": r.get("e2"),
            "t": r.get("t"), "cond": r.get("cond"),
            "fluent": ans["fluent"]["noul"],
            "quality": ans["quality"]["score"],
            "quality_conf": ans["quality"]["confidence"],
            "quality_probs": ans["quality"]["probabilities"],
            "repetition": ans["repetition"]["noul"],
            "served_model": d["model"], "usage": d["usage"], "latency_s": round(dt, 3),
        })
        json.dump({"meta": {"source": a.inp, "model": MODEL, "questions": QUESTIONS},
                   "rows": out}, open(a.out, "w"), indent=1)
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(rows)} mean {sum(lat)/len(lat):.2f}s")

    print(f"wrote {a.out}: {len(out)} scored, {fails} failed, "
          f"mean {sum(lat)/len(lat):.2f}s" if lat else f"wrote {a.out}: {len(out)} scored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
