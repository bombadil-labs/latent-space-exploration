"""Flags state spans of a time-translation grid that leak the interval lexically.

A state span "leaks" if it contains:
  (a) an explicit duration word (day, week, year, century, ... "later", "since", "ago"), or
  (b) an explicit "<number> <duration-unit>" construction (regex, catches "24 orbits",
      "a hundred years", "three hundred million years", etc.), or
  (c) any non-stopword token also present in that row's own interval phrase (e.g. the
      word "million" inside the state of a Delta t = 1,000,000-years row).

Usage: python scripts/time_translation_leak_check.py [grid.json ...]
Default: checks both v2 and v3 grids and prints a summary + a few example hits each.
"""
import json, re, sys

DURATION_WORDS = {
    "day", "days", "week", "weeks", "fortnight", "fortnights",
    "month", "months", "year", "years", "decade", "decades",
    "century", "centuries", "millennium", "millennia",
    "hour", "hours", "minute", "minutes", "second", "seconds",
    "generation", "generations", "orbit", "orbits", "orbited",
    "later", "since", "ago", "hence", "anniversary",
}
NUM_WORDS = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "dozen", "hundred", "thousand", "million", "billion",
    "few", "several", "many", "couple", "single", "a",
]
_UNIT_ALT = "|".join(sorted({w for w in DURATION_WORDS if w not in
                              {"later", "since", "ago", "hence", "anniversary"}}))
DURATION_RE = re.compile(
    r"\b(\d[\d,]*|" + "|".join(NUM_WORDS) + r")\s+(" + _UNIT_ALT + r")\b", re.I)

STOP = {"a", "an", "the", "at", "of", "on", "in", "with", "to", "and"}


def toks(s):
    return re.findall(r"[A-Za-z']+", s.lower())


def phrase_tokens(phrase):
    return set(toks(phrase)) - STOP


def check_state(text, interval_phrase, is_t0=False):
    tks = toks(text)
    tk_set = set(tks)
    hits = []
    for w in phrase_tokens(interval_phrase):
        if w in tk_set:
            hits.append(("interval_token", w))
    if is_t0:
        # t0 (Delta t = 0) has no elapsed interval to restate; its own text may
        # legitimately use temporal vocabulary for the subject's INTRINSIC properties
        # (a mayfly's hours-long adult life, an orchard's fifteen-year-old trees). Only
        # phrase-token overlap (checked above) applies at t0.
        return hits
    for w in tk_set:
        if w in DURATION_WORDS:
            hits.append(("duration_word", w))
    for m in DURATION_RE.finditer(text):
        hits.append(("duration_construction", m.group(0)))
    return hits


def check_grid(path):
    G = json.load(open(path))
    S, TP, PHRASES = G["subjects"], G["timepoints"], G["phrases"]
    flagged, total, detail = 0, 0, []
    for s in S:
        for t in TP:
            phrase = PHRASES[t]
            for pi, para in enumerate(G["states"][s][t]):
                total += 1
                hits = check_state(para, phrase, is_t0=(t == "t0"))
                if hits:
                    flagged += 1
                    detail.append((s, t, pi, hits))
    return flagged, total, detail


def main():
    paths = sys.argv[1:] or ["prompts/time_translation_v2.json", "prompts/time_translation_v3.json"]
    for path in paths:
        print(f"\n########## {path} ##########")
        flagged, total, detail = check_grid(path)
        print(f"  flagged {flagged}/{total} state spans")
        for s, t, pi, hits in detail:
            print(f"    {s}/{t}/p{pi}: {hits[:4]}")


if __name__ == "__main__":
    main()
