"""Cross-subject shared-content-word checker for the time-translation grids.

For a grid file and a set of Dt values, tokenizes each subject's state spans
(all paraphrases pooled), strips stopwords and each subject's own name-derived
tokens, and reports:
  - for every content word used by >=2 subjects at that Dt: the word and which
    subjects use it
  - n_shared_words: count of distinct content words used by >=2 subjects
  - max_subjects_per_word: the largest subject-count any single word reaches
    (the v2 spec constraint is that this must be <=2 for Dt >= 100 years)

Usage: python scripts/time_translation_vocab_check.py [grid.json ...]
"""
import json, re, sys

STOPWORDS = set("""
a an the of to in on at by for with from into onto over under after before
and or but nor so yet not no is are was were be been being has have had
having do does did will would shall should can could may might must
it its it's this that these those there here as than then thus if when
while because since although though whether unless until
one two three four five six seven eight nine ten hundred thousand million
year years day days week weeks month months later after all any some
which who whom whose what where how why every each other another
its own more most less least very still just only also even
much many few several between within across through per about
almost nearly roughly perhaps likely probably possibly certainly
own new old first last same different comparable
i you he she we they them him her his their our your
is's has's -
century centuries millennium millennia decade decades
itself whatever nothing something anything everything anyone someone
exists exist existed persists persist persisted survive survives survived
remain remains remained remaining unchanged changed continuity continuous
entire chain claim order question history times described relation
unrelated measurably outlasts superseded occupies once long people
institution institutional original figure
""".split())

def norm(tok: str) -> str:
    t = tok.lower().strip("'’.,;:!?\"()")
    return t

def content_words(text: str, extra_stop: set) -> set:
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    out = set()
    for tok in toks:
        n = norm(tok)
        if not n or n in STOPWORDS or n in extra_stop:
            continue
        if len(n) <= 2:
            continue
        out.add(n)
    return out

SUBJECT_NAME_TOKENS = {
    "street": {"alder", "street"},
    "mountain": {"caldreth", "mountain"},
    "orchard": {"orchard"},
    "mayfly": {"mayfly", "mayflies", "ephemeroptera"},
    "asteroid": {"asteroid"},
    "river": {"dorn", "river"},
    "real_population": {"london", "thames"},
    "fictional_population": {"veyle", "ossel"},
}

def check_grid(path: str):
    G = json.load(open(path))
    S, DT = G["subjects"], G["deltas"]
    report = {}
    for t in ["t0"] + DT:
        words_by_subject = {}
        for s in S:
            paras = G["states"][s][t]
            extra = SUBJECT_NAME_TOKENS.get(s, set())
            w = set()
            for p in paras:
                w |= content_words(p, extra)
            words_by_subject[s] = w
        word_to_subjects = {}
        for s, w in words_by_subject.items():
            for word in w:
                word_to_subjects.setdefault(word, set()).add(s)
        shared = {w: sorted(subs) for w, subs in word_to_subjects.items() if len(subs) >= 2}
        max_n = max((len(subs) for subs in word_to_subjects.values()), default=0)
        report[t] = dict(n_shared_words=len(shared), max_subjects_per_word=max_n,
                          shared=dict(sorted(shared.items(), key=lambda kv: -len(kv[1]))))
    return report

def main():
    paths = sys.argv[1:] or ["prompts/time_translation_v1.json", "prompts/time_translation_v2.json"]
    for path in paths:
        print(f"\n########## {path} ##########")
        rep = check_grid(path)
        for t, r in rep.items():
            print(f"  {t:14s}  n_shared_words={r['n_shared_words']:3d}  max_subjects_per_word={r['max_subjects_per_word']}")
        far = [t for t in rep if t not in ("t0",) and t not in ("1day", "1week", "6months", "1year", "10years")]
        print("  -- far-Dt words shared by >2 subjects (violations of the v2 constraint) --")
        any_violation = False
        for t in far:
            viol = {w: subs for w, subs in rep[t]["shared"].items() if len(subs) > 2}
            if viol:
                any_violation = True
                print(f"    {t}: " + ", ".join(f"{w}{subs}" for w, subs in viol.items()))
        if not any_violation:
            print("    none")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
