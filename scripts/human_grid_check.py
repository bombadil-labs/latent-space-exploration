"""Check a hand-written factor grid before it is used. Four structural checks, then the core's
leak report.

    .venv/bin/python scripts/human_grid_check.py prompts/human/human_factors_v1.json

A leak FLOOR is expected and is not a failure: real prose about the far future genuinely uses
different words than prose about millers. It is measured so that results on this grid are reported
as gain over it (spec §6), never as a raw score.
"""
import json, re, sys
from collections import defaultdict
from itertools import combinations

STOP = set("""a an the and or but if of in on at to for with from by as is are was were be been being
it its this that these those he she they them his her their you your i we our not no so then than
there here when while into over under out up down about after before again very just only own same
s t don now d ll m o re ve y""".split())
NAMES = {"era": ["medieval", "1920s", "twenties", "far future", "farfuture", "middle ages", "long ago",
                 "centuries ago", "in the future"],
         "voice": ["terse", "ornate", "child", "childlike", "laconic", "florid"],
         "tense": ["past tense", "present tense"]}


def words(s):
    return [w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOP and len(w) > 2]


def main(path):
    g = json.load(open(path))
    spans, fac = g["spans"], g["factors"]
    order = g["key_order"]
    bad = 0

    print("=== completeness ===")
    empty = [k for k, v in spans.items() if not v.strip()]
    print(f"  {len(spans) - len(empty)}/{len(spans)} cells filled" + (f"; EMPTY: {empty[:6]}" if empty else ""))
    bad += len(empty)
    if empty:
        print("\n(stopping here; fill the cells and re-run)")
        return 1

    print("=== length (15-35 words) ===")
    for k, v in spans.items():
        n = len(v.split())
        if not 15 <= n <= 35:
            print(f"  {k}: {n} words"); bad += 1
    print("  ok" if all(15 <= len(v.split()) <= 35 for v in spans.values()) else "")

    print("=== era vocabulary overlap (rule 3) ===")
    by_era = defaultdict(set)
    for k, v in spans.items():
        by_era[dict(zip(order, k.split("|")))["era"]].update(words(v))
    hits = 0
    for a, b in combinations(sorted(by_era), 2):
        shared = by_era[a] & by_era[b]
        if shared:
            print(f"  {a} & {b}: {sorted(shared)[:10]}"); hits += len(shared)
    print("  ok, no distinctive word shared across eras" if not hits else f"  {hits} shared (rule 3)")
    bad += hits

    print("=== factor-name leakage (rule 1) ===")
    leaks = 0
    for k, v in spans.items():
        low = v.lower()
        for names in NAMES.values():
            for n in names:
                if n in low:
                    print(f"  {k}: names {n!r}"); leaks += 1
    print("  ok, no cell names its own factor" if not leaks else "")
    bad += leaks

    print("=== leak floor (measured, not a pass/fail) ===")
    try:
        sys.path.insert(0, "src")
        from lsx.core.types import Grid, Item
        items = [Item(text=v, factors=dict(zip(order, k.split("|"))), spans={"passage": (0, len(v))})
                 for k, v in spans.items()]
        rep = Grid(name="human_factors_v1", items=items).leak
        print(" ", rep.summary())
        print("  -> claims on this grid report gain over this floor, never a raw score")
    except Exception as e:
        print(f"  could not run the core's leak report: {type(e).__name__}: {e}")

    print(f"\n{'CLEAN' if bad == 0 else str(bad) + ' issue(s) to fix'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "prompts/human/human_factors_v1.json"))
