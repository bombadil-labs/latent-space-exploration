"""Check the conscription grid (`prompts/human/conscription_v1.json`) before any model sees it.

    .venv/bin/python scripts/conscription_check.py prompts/human/conscription_v1.json

Leads with length, because length is the most likely confound in this design: `exit` is `enact`
plus a permission clause and every other arm carries a matched-length neutral closer as the control
for that clause (`CONSCRIPTION_INSTRUCTIONS.md` rule 5) -- if the balance fails, the closer isn't
doing its job and any `exit < enact` reading could just be "shorter prompt, different processing
depth" instead of "an exit available".

Four sections, run in this order on purpose:

  1. schema & completeness       -- can anything downstream even run?
  2. per-arm token length        -- the headline number (see above), Qwen2.5-1.5B-Instruct tokenizer,
                                     local, offline.
  3. enact/exit prefix identity  -- rule 4: exit = enact verbatim + a permission clause. Checked, the
                                     diff printed, not assumed.
  4. arm-label leakage           -- can a bag-of-words classifier recover the arm, LOO, against ITS
                                     OWN permutation null (h32's lesson: LOO on balanced labels is
                                     anti-predictive by construction, so 0.00 is not "clean" and 0.50
                                     is not "chance"). `enact` vs `true` is the number that matters:
                                     they differ only in which proposition is asserted, on the same
                                     content words, so this should sit near its own null.
  5. lexical floor                -- what the words give away, via the SAME leave-one-out machinery
                                     as (4). `reproduce.h8_lexical_floor` does not fit this design's
                                     shape (it ranks a composed direction among factorial-combination
                                     candidates; conscription has five parallel discrete arms per
                                     item, no combinatorial candidate set to rank against) -- said
                                     here rather than forced, and the minimal honest analogue is (4)
                                     itself: predicting the arm from the bag of words IS the floor a
                                     later activation-level "which arm" readout should be measured
                                     against, so the two sections share one number rather than
                                     computing it twice under different names.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LENGTH_FLAG_PCT = 0.15


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _render(tok, item, arm):
    from lsx.core.conscription import render_prompt
    return render_prompt(tok, item["prefix"], item["arms"][arm])


# --------------------------------------------------------------------------------------------
# 1. schema & completeness
# --------------------------------------------------------------------------------------------
def check_schema(g: dict) -> tuple[int, int]:
    """Returns (blocking, non_blocking). A grid still short of its full 24-item design (the current
    state of `conscription_v1.json`: worked examples, not the finished study) should still have its
    length/leakage/floor measured on what exists -- that is what 'exercise the code' means -- so
    only structural breakage (a missing arm, a non-assistant-ending prefix, a duplicate id) blocks
    the rest of this script. An incomplete domain count is reported, loudly, but does not block.
    """
    print("=== schema & completeness ===")
    blocking = 0
    warn = 0
    meta = g.get("_meta", {})
    arms = meta.get("arms", [])
    domains = meta.get("domains", [])
    items = g.get("items", [])
    print(f"  {len(items)} items; arms={arms}; domains={domains}")

    ids = [it.get("id") for it in items]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        print(f"  DUPLICATE ids: {dupes}"); blocking += len(dupes)
    else:
        print("  ids unique")

    by_domain = defaultdict(int)
    for it in items:
        by_domain[it.get("domain")] += 1
    print(f"  domain counts: {dict(by_domain)}")
    for d in domains:
        if by_domain.get(d, 0) != meta.get("items_per_domain"):
            print(f"    {d}: {by_domain.get(d, 0)} items, expected {meta.get('items_per_domain')} "
                 "(NOT BLOCKING -- design is in progress; the full 24-item grid needs this filled "
                 "before a real run, but the checks below still run on what's here)")
            warn += 1
    extra = set(by_domain) - set(domains)
    if extra:
        print(f"  items with undeclared domain(s): {sorted(extra)}"); blocking += len(extra)

    for it in items:
        iid = it.get("id", "?")
        prefix = it.get("prefix")
        if not prefix or not isinstance(prefix, list):
            print(f"  {iid}: missing/empty prefix"); blocking += 1
        elif prefix[-1].get("role") != "assistant":
            print(f"  {iid}: prefix does not end in an assistant turn (ends {prefix[-1].get('role')!r})")
            blocking += 1
        item_arms = it.get("arms", {})
        missing = [a for a in arms if a not in item_arms]
        if missing:
            print(f"  {iid}: missing arm(s) {missing}"); blocking += 1
        empty = [a for a, t in item_arms.items() if not str(t).strip()]
        if empty:
            print(f"  {iid}: empty arm text {empty}"); blocking += 1
    print("  ok" if blocking == 0 and warn == 0 else
         f"  {blocking} blocking issue(s), {warn} incompleteness warning(s)")
    return blocking, warn


# --------------------------------------------------------------------------------------------
# 2. per-arm token length (the headline)
# --------------------------------------------------------------------------------------------
def check_lengths(g: dict, tok) -> tuple[int, dict]:
    print("\n=== per-arm token-length distribution (Qwen2.5-1.5B-Instruct, whole rendered turn) ===")
    meta, items = g["_meta"], g["items"]
    arms = meta["arms"]
    lengths: dict[str, list[int]] = {a: [] for a in arms}
    for it in items:
        for a in arms:
            text = _render(tok, it, a)
            n = len(tok(text, add_special_tokens=True)["input_ids"])
            lengths[a].append(n)

    means = {a: float(np.mean(lengths[a])) for a in arms}
    sds = {a: float(np.std(lengths[a])) for a in arms}
    grand = float(np.mean([n for a in arms for n in lengths[a]]))
    for a in arms:
        print(f"  {a:<10} mean={means[a]:7.2f}  sd={sds[a]:6.2f}  n={len(lengths[a])}  "
             f"{'  '.join(str(n) for n in lengths[a])}")
    print(f"  grand mean = {grand:.2f}")

    max_gap, max_pair = 0.0, None
    bad = 0
    for a in arms:
        pct = abs(means[a] - grand) / grand
        flag = pct > LENGTH_FLAG_PCT
        print(f"  {a:<10} {pct * 100:5.1f}% from grand mean" + ("  FLAGGED (>15%)" if flag else ""))
        if flag:
            bad += 1
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            gap = abs(means[a] - means[b])
            if gap > max_gap:
                max_gap, max_pair = gap, (a, b)
    print(f"  max pairwise arm-mean gap: {max_gap:.2f} tokens ({max_pair[0]!r} vs {max_pair[1]!r})"
         if max_pair else "  (fewer than two arms)")
    return bad, {"means": means, "sds": sds, "grand_mean": grand, "max_gap": max_gap,
                "max_pair": max_pair, "raw": lengths}


# --------------------------------------------------------------------------------------------
# 3. enact/exit prefix identity
# --------------------------------------------------------------------------------------------
def check_enact_exit(g: dict) -> int:
    print("\n=== enact/exit identity (rule 4: exit = enact verbatim + permission clause) ===")
    bad = 0
    for it in g["items"]:
        iid = it["id"]
        enact, exit_ = it["arms"].get("enact", ""), it["arms"].get("exit", "")
        if exit_.startswith(enact):
            suffix = exit_[len(enact):]
            print(f"  {iid}: ok  (exit suffix = {suffix!r})")
        else:
            # report the first point of divergence rather than just failing
            k = 0
            while k < min(len(enact), len(exit_)) and enact[k] == exit_[k]:
                k += 1
            print(f"  {iid}: MISMATCH -- diverges at char {k}")
            print(f"    enact: {enact!r}")
            print(f"    exit:  {exit_!r}")
            bad += 1
    print("  ok, every exit is enact verbatim + a suffix" if bad == 0 else f"  {bad} mismatch(es)")
    return bad


# --------------------------------------------------------------------------------------------
# 4/5. arm-label leakage + lexical floor (same LOO-ridge machinery, shared)
# --------------------------------------------------------------------------------------------
def check_leakage_and_floor(g: dict) -> None:
    print("\n=== arm-label leakage (bag-of-words, leave-one-item-out) ===")
    from lsx.core.types import bag_of_tokens_recoverability

    meta, items = g["_meta"], g["items"]
    arms = meta["arms"]
    texts_all = [it["arms"][a] for it in items for a in arms]
    labels_all = [a for it in items for a in arms]
    acc, null = bag_of_tokens_recoverability(texts_all, labels_all)
    chance = 1.0 / len(arms)
    print(f"  all {len(arms)} arms: LOO accuracy {acc:.3f}  (permutation null {null:.3f}, "
         f"nominal chance {chance:.3f})")
    print("  high accuracy is expected for `neutral` (it reads differently by construction); the "
         "number that matters is enact vs true, below.")

    print("\n  --- enact vs true (the number that matters) ---")
    if "enact" in arms and "true" in arms:
        texts = [it["arms"][a] for it in items for a in ("enact", "true")]
        labels = [a for it in items for a in ("enact", "true")]
        acc2, null2 = bag_of_tokens_recoverability(texts, labels)
        gap = acc2 - null2
        print(f"  enact vs true: LOO accuracy {acc2:.3f}  (permutation null {null2:.3f}, gap "
             f"{gap:+.3f}, nominal chance 0.500)")
        verdict = ("near its own null -- consistent with the design (same content words, different "
                  "asserted proposition)" if abs(gap) < 0.2 else
                  "OFF the permutation null by more than 0.2 -- enact and true may be separable on "
                  "content words alone, which would undercut the wrongness-vs-badness contrast")
        print(f"  -> {verdict}")
    else:
        print("  enact and/or true not both present in this grid's arms; skipped")

    print("\n  --- per-arm-pair recoverability (diagnostic) ---")
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            texts = [it["arms"][x] for it in items for x in (a, b)]
            labels = [x for it in items for x in (a, b)]
            acc_p, null_p = bag_of_tokens_recoverability(texts, labels)
            print(f"  {a:<10} vs {b:<10} acc={acc_p:.3f}  null={null_p:.3f}  gap={acc_p - null_p:+.3f}")

    print("\n=== lexical floor ===")
    print("  Attempted reproduce.h8_lexical_floor(gridspec=...): it requires "
         "{factors, scenes, spans} -- factorial levels composed into a candidate SET the true "
         "variant is ranked against (midrank over combinatorial distractors). Conscription has five "
         "PARALLEL discrete arms per item and no combinatorial candidate set: there is nothing here "
         "shaped like h8's ranking problem, so forcing gridspec through it would not measure this "
         "design's floor -- it would measure a fabricated one.")
    print("  Minimal honest analogue: the leave-one-out bag-of-words classifier above IS the floor a "
         "later activation-level 'which arm' readout must be measured against -- same LOO-ridge "
         "machinery reproduce.h8_lexical_floor also uses (kernel ridge over bag-of-tokens, judged "
         "against its own permutation null, never nominal chance), applied directly to the arm label "
         "rather than recomputed under a second name. See the 'all arms' and 'enact vs true' numbers "
         "above; they ARE this design's measured lexical floor.")


def main(path: str) -> int:
    g = json.loads(pathlib.Path(path).read_text())
    blocking, warn = check_schema(g)
    if blocking:
        print("\n(stopping before length/leakage checks; fix the blocking issues and re-run)")
        return 1
    if not g.get("items"):
        print("\n(no items to check)")
        return 1
    tok = _tokenizer()
    len_bad, _ = check_lengths(g, tok)
    bad = warn + len_bad
    bad += check_enact_exit(g)
    check_leakage_and_floor(g)
    print(f"\n{'CLEAN' if bad == 0 else str(bad) + ' issue(s) to look at'}"
         f"{' (design incomplete: see domain-count warnings above)' if warn else ''}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "prompts/human/conscription_v1.json"))
