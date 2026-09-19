"""Check the conscription grid (`prompts/human/conscription_v1.json` or
`prompts/claude/conscription_claude_v1.json`) before any model sees it.

    .venv/bin/python scripts/conscription_check.py prompts/human/conscription_v1.json
    .venv/bin/python scripts/conscription_check.py prompts/claude/conscription_claude_v1.json

Four sections, run in this order on purpose:

  1. schema & completeness       -- can anything downstream even run? (GATE)
  2. per-arm token length        -- the headline confound number, Qwen2.5-1.5B-Instruct tokenizer,
                                     local, offline. (GATE: >15% from the grand mean)
  3. enact/exit prefix identity  -- rule 4: exit = enact verbatim + a permission clause. Checked, the
                                     diff printed, not assumed. (GATE)
  4. floor estimation + confound diagnostics -- NOT A GATE. See below.

WHY SECTION 4 STOPPED BEING A GATE (results/notes/conscription_prereg.md, "the leak threshold is
mis-specified for this pair"): every arm's manipulation is realised in language, by construction --
`report`'s third-party frame, `exit`'s permission clause, `true`'s contrary proposition are all
words. A bag-of-words classifier can therefore recover the arm from EVERY pair, including the pairs
where that is exactly the point (`enact` vs `true` differ only in which proposition is asserted, on
otherwise similar content -- that difference IS words). A zero-leak standard is unattainable by this
design's own logic, and a threshold on it (this script's old `verdict`/`FLAGGED`/`OFF the null`
language) was arbitrary: it graded the grid against a bar the design cannot clear and never needed
to clear, because the thing that matters is not whether words distinguish the arms but whether the
ACTIVATION contrast does more than words already do. That is `gain over a measured floor`, this
project's rule everywhere else (`docs/specs/core_v1.md`, `reproduce.h8_lexical_floor`), and it was
not applied to this checker until now.

So section 4 now REPORTS, per arm pair, leave-one-item-out bag-of-words accuracy beside its own
permutation null (h32's lesson, still load-bearing: LOO on balanced labels is anti-predictive by
construction, so 0.00 is not "clean" and 0.50 is not "chance" -- only the gap to the permutation null
means anything). That number is not a verdict. It is an input to the later activation analysis: the
floor that analysis's own gain must clear, pair by pair. It is written to a JSON sidecar next to the
grid (`<grid>.floors.json`) with a content hash of the grid, so the activation analysis can check
that the floor it is comparing against was measured on the exact text it read and not a stale one.

Also reported (diagnostics, not gates, each beside its own permutation null): negation density,
second-/third-person referent counts, type-token ratio, and mean sentence length, per arm and per
domain -- confounds named in the pre-registration (`enact`/`true` may partition on negation alone;
`report`'s third party is quantified rather than left as prose; the machine grid's stylistic
regularity is given a number; and everything is broken out per domain because a confound can live in
one domain and vanish in the pooled figure, h47's tense lens).

KNOWN MISMATCH WITH THE DOWNSTREAM ACTIVATION ANALYSIS (read this before trusting the floor number):
this floor is computed on the RAW arm-turn text only (`item["arms"][arm]`), tokenised by the same
plain regex word-splitter `lsx.core.types._bag` uses for every other floor in this project. It does
NOT go through `conscription.render_prompt` (prefix + arm + the model's own chat template) and does
NOT use the Qwen2.5-1.5B-Instruct BPE tokenizer that the activation extraction reads. That is the
same shape of error `h8_lexical_floor`'s docstring warns about under a different name ("a floor
computed with a different tokenisation ... is not that analysis's floor") and it is stated here
rather than fixed quietly: the grid CONTENT is pinned (content hash in the sidecar), but the
FEATURIZATION is not the same pipeline the activation read will use. Treat every number in this
section as "what bag-of-words gives away", not "what the model's tokenizer would give away" -- they
are almost certainly close (the vocabulary differs by BPE merges, not by which words exist) but this
was not verified byte-for-byte and the difference is exactly the kind of thing that has hidden an
h8-shaped bug before.
"""
from __future__ import annotations

import json
import hashlib
import os
import pathlib
import re
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
NULL_DRAWS = 20

# Same tokenization as lsx.core.types._bag (bag_of_tokens_recoverability's own featurizer). Kept as
# a literal copy rather than importing the private name, but it MUST stay byte-identical to that
# regex or the floor stops meaning "what bag_of_tokens_recoverability sees" -- see the module
# docstring's mismatch warning.
_WORD = re.compile(r"[a-z0-9']+")

NEGATION_BARE = {"not", "no", "never"}
REFUSAL_INCAPACITY_VERBS = {
    "cannot", "refuse", "refuses", "refused", "decline", "declines", "declined", "unable",
}
SECOND_PERSON = {"you", "your", "yours", "you're", "youre"}
THIRD_PERSON = {
    "he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs", "someone",
    "somebody", "colleague", "colleagues", "people", "person",
}


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _render(tok, item, arm):
    from lsx.core.conscription import render_prompt
    return render_prompt(tok, item["prefix"], item["arms"][arm])


def _grid_content_hash(g: dict) -> str:
    """Content hash over items only (not `_meta`), so a metadata edit doesn't stale-out a floor
    that was measured on unchanged text, but any change to an id, domain, prefix or arm text does.
    """
    payload = json.dumps(g.get("items", []), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------------
# 1. schema & completeness
# --------------------------------------------------------------------------------------------
def check_schema(g: dict) -> tuple[int, int]:
    """Returns (blocking, non_blocking). A grid still short of its full 24-item design (the current
    state of `conscription_v1.json`: worked examples, not the finished study) should still have its
    length/floor/diagnostics measured on what exists -- that is what 'exercise the code' means -- so
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
    """Rule 4 (corrected): `enact` and `exit` share an IDENTICAL ASSERTION HALF and differ only in
    the closer slot -- `exit`'s closer is the permission, every other arm's is inert. The permission
    is not appended on top of a closer; appending would make `exit` longer than every other arm and
    reintroduce the length confound rule 6 exists to prevent.

    So the check is: the shared assertion half must be a real prefix of both (not a coincidence of
    the first few words), and what differs must be only the trailing clause. Reported either way --
    the divergence point is printed so a reader can see WHERE the closer starts rather than trust
    that it starts anywhere sensible.
    """
    print("\n=== enact/exit assertion-half identity (rule 4) ===")
    bad = 0
    for it in g["items"]:
        iid = it["id"]
        enact, exit_ = it["arms"].get("enact", ""), it["arms"].get("exit", "")
        k = 0
        while k < min(len(enact), len(exit_)) and enact[k] == exit_[k]:
            k += 1
        shared = enact[:k]
        frac = k / max(len(enact), len(exit_), 1)
        # the shared half has to be most of the turn, and it has to end at a clause boundary
        clean = frac >= 0.5 and (k == len(enact) or shared.rstrip()[-1:] in ".!?," or shared.endswith(" "))
        if clean:
            print(f"  {iid}: ok  shared assertion {k} chars ({frac:.0%}); "
                  f"enact closer {enact[k:].strip()!r} | exit closer {exit_[k:].strip()!r}")
        else:
            print(f"  {iid}: SHARED HALF TOO SHORT -- only {k} chars ({frac:.0%}) before divergence")
            print(f"    enact: {enact!r}")
            print(f"    exit:  {exit_!r}")
            bad += 1
    print("  ok, every exit shares its assertion half with enact and differs only in the closer"
          if bad == 0 else f"  {bad} item(s) where the assertion halves genuinely differ")
    return bad


# --------------------------------------------------------------------------------------------
# 4. floor estimation (report, not a gate) + confound diagnostics (report, not a gate)
# --------------------------------------------------------------------------------------------
def compute_pair_floors(items: list[dict], arms: list[str], null_draws: int = NULL_DRAWS,
                        seed: int = 0) -> dict:
    """For every arm pair: LOO bag-of-words accuracy beside its own permutation null. This is the
    floor the activation contrast for that pair must beat -- reported, not judged.
    """
    from lsx.core.types import bag_of_tokens_recoverability
    floors = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            texts = [it["arms"][x] for it in items for x in (a, b)]
            labels = [x for it in items for x in (a, b)]
            acc, null = bag_of_tokens_recoverability(texts, labels, null_draws=null_draws, seed=seed)
            floors[f"{a}_vs_{b}"] = {
                "loo_accuracy": acc, "permutation_null_mean": null, "gap": acc - null,
                "n_items": len(items),
            }
    return floors


def compute_all_arms_floor(items: list[dict], arms: list[str], null_draws: int = NULL_DRAWS,
                           seed: int = 0) -> dict:
    from lsx.core.types import bag_of_tokens_recoverability
    texts = [it["arms"][a] for it in items for a in arms]
    labels = [a for it in items for a in arms]
    acc, null = bag_of_tokens_recoverability(texts, labels, null_draws=null_draws, seed=seed)
    return {"loo_accuracy": acc, "permutation_null_mean": null, "gap": acc - null,
           "nominal_chance": 1.0 / len(arms), "n_items": len(items)}


def print_floors(pair_floors: dict, all_arms: dict, arms: list[str]) -> None:
    print("\n=== floor estimation (bag-of-words, leave-one-item-out; NOT a gate) ===")
    print("  Each number below is the floor the activation contrast for that pair must beat -- not "
         "a pass/fail. A zero-leak grid is not attainable by this design's own logic (every "
         "manipulation is realised in words), so there is no threshold here to clear.")
    print(f"\n  all {len(arms)} arms (diagnostic, multiclass): LOO accuracy "
         f"{all_arms['loo_accuracy']:.3f}  (permutation null {all_arms['permutation_null_mean']:.3f}, "
         f"nominal chance {all_arms['nominal_chance']:.3f})")
    print("\n  --- per-pair floors ---")
    for pair, d in pair_floors.items():
        print(f"  {pair:<20} LOO acc={d['loo_accuracy']:.3f}  permutation null={d['permutation_null_mean']:.3f}"
             f"  gap={d['gap']:+.3f}  n_items={d['n_items']}")


def write_floor_sidecar(grid_path: pathlib.Path, g: dict, pair_floors: dict, all_arms: dict,
                        null_draws: int = NULL_DRAWS) -> pathlib.Path:
    sidecar = {
        "grid_path": str(grid_path),
        "grid_content_hash": _grid_content_hash(g),
        "n_items": len(g["items"]),
        "arms": g["_meta"]["arms"],
        "permutation_null_draws": null_draws,
        "measured_on": "raw arm-turn text only (item['arms'][arm]); excludes the shared item "
                       "prefix and the chat-template rendering the activation extraction reads",
        "tokenization": "regex word-level split ([a-z0-9']+ on lowercased text), the same "
                       "featurizer lsx.core.types._bag / bag_of_tokens_recoverability uses "
                       "elsewhere in this project -- NOT the Qwen2.5-1.5B-Instruct BPE tokenizer "
                       "or lsx.core.conscription.render_prompt's chat-template text that the "
                       "activation extraction will actually read. Guaranteed to match the "
                       "downstream analysis: the grid content (pinned by grid_content_hash). NOT "
                       "guaranteed to match: tokenization, and whether prefix text is included.",
        "meaning": "For each pair, loo_accuracy beside its own permutation_null_mean is the floor "
                  "the activation contrast for that pair must beat. Not a threshold; nothing here "
                  "passes or fails.",
        "pairs": pair_floors,
        "all_arms_diagnostic": all_arms,
    }
    path = pathlib.Path(str(grid_path) + ".floors.json")
    path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    return path


# ---- diagnostics: negation, referents, TTR, sentence length, each with its own permutation null --
def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def negation_density(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    hits = 0
    for t in toks:
        if t in NEGATION_BARE or "n't" in t or t in REFUSAL_INCAPACITY_VERBS:
            hits += 1
    return hits / len(toks)


def second_person_density(text: str) -> float:
    toks = _tokens(text)
    return (sum(1 for t in toks if t in SECOND_PERSON) / len(toks)) if toks else 0.0


def third_person_density(text: str) -> float:
    toks = _tokens(text)
    return (sum(1 for t in toks if t in THIRD_PERSON) / len(toks)) if toks else 0.0


def type_token_ratio(text: str) -> float:
    toks = _tokens(text)
    return (len(set(toks)) / len(toks)) if toks else 0.0


def mean_sentence_length(text: str) -> float:
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0
    lens = [len(_tokens(s)) for s in sentences]
    return float(np.mean(lens))


DIAGNOSTICS = {
    "negation_density": negation_density,
    "second_person_density": second_person_density,
    "third_person_density": third_person_density,
    "type_token_ratio": type_token_ratio,
    "mean_sentence_length": mean_sentence_length,
}


def diagnostic_by_arm(items: list[dict], arms: list[str], metric_fn, null_draws: int = NULL_DRAWS,
                      seed: int = 0) -> dict:
    """Per-arm mean of a scalar text metric, beside a permutation null of the same statistic: shuffle
    which arm label is attached to which text (within the pooled set) `null_draws` times and report
    the null distribution of the observed-range statistic (max arm-mean - min arm-mean). This is the
    same logic as the LOO permutation null above, applied to a simple per-arm mean instead of a
    classifier: it answers "would arms this different in <metric> arise if arm labels carried no
    information about the text at all?"
    """
    rows = [(it["arms"][a], a) for it in items for a in arms]
    texts = [t for t, _ in rows]
    labels = [a for _, a in rows]
    vals = [metric_fn(t) for t in texts]

    def arm_means(labs) -> dict:
        buckets = defaultdict(list)
        for v, l in zip(vals, labs):
            buckets[l].append(v)
        return {a: (float(np.mean(buckets[a])) if buckets[a] else float("nan")) for a in arms}

    observed = arm_means(labels)
    obs_vals = [v for v in observed.values() if not np.isnan(v)]
    obs_range = (max(obs_vals) - min(obs_vals)) if obs_vals else 0.0

    rng = np.random.default_rng(seed)
    null_ranges = []
    for _ in range(null_draws):
        perm = rng.permutation(labels)
        m = arm_means(list(perm))
        mv = [v for v in m.values() if not np.isnan(v)]
        null_ranges.append((max(mv) - min(mv)) if mv else 0.0)
    return {
        "arm_means": observed,
        "observed_range": obs_range,
        "permutation_null_range_mean": float(np.mean(null_ranges)),
        "permutation_null_range_sd": float(np.std(null_ranges)),
        "n_items": len(items),
    }


def print_diagnostics(items: list[dict], arms: list[str], domains: list[str],
                      null_draws: int = NULL_DRAWS) -> dict:
    print("\n=== confound diagnostics (reported with their own permutation nulls; NOT gates) ===")
    out = {"pooled": {}, "by_domain": {}}
    for name, fn in DIAGNOSTICS.items():
        print(f"\n  --- {name} ---")
        d = diagnostic_by_arm(items, arms, fn, null_draws=null_draws)
        out["pooled"][name] = d
        means_str = "  ".join(f"{a}={d['arm_means'][a]:.4f}" for a in arms)
        print(f"    pooled ({d['n_items']} items): {means_str}")
        print(f"    observed arm-mean range={d['observed_range']:.4f}  "
             f"permutation null range={d['permutation_null_range_mean']:.4f} "
             f"(sd={d['permutation_null_range_sd']:.4f})")
        out["by_domain"][name] = {}
        for dom in domains:
            dom_items = [it for it in items if it.get("domain") == dom]
            if not dom_items:
                print(f"    [{dom}] no items")
                continue
            dd = diagnostic_by_arm(dom_items, arms, fn, null_draws=null_draws)
            out["by_domain"][name][dom] = dd
            dmeans_str = "  ".join(f"{a}={dd['arm_means'][a]:.4f}" for a in arms)
            print(f"    [{dom:<8} n={dd['n_items']:<2}] {dmeans_str}  range={dd['observed_range']:.4f} "
                 f"(null {dd['permutation_null_range_mean']:.4f})")
    return out


# --------------------------------------------------------------------------------------------
def main(path: str) -> int:
    grid_path = pathlib.Path(path)
    g = json.loads(grid_path.read_text())
    blocking, warn = check_schema(g)
    if blocking:
        print("\n(stopping before length/floor checks; fix the blocking issues and re-run)")
        return 1
    if not g.get("items"):
        print("\n(no items to check)")
        return 1
    tok = _tokenizer()
    len_bad, _ = check_lengths(g, tok)
    bad = warn + len_bad
    bad += check_enact_exit(g)

    meta, items = g["_meta"], g["items"]
    arms, domains = meta["arms"], meta["domains"]
    pair_floors = compute_pair_floors(items, arms)
    all_arms_floor = compute_all_arms_floor(items, arms)
    print_floors(pair_floors, all_arms_floor, arms)
    sidecar_path = write_floor_sidecar(grid_path, g, pair_floors, all_arms_floor)
    print(f"\n  floor sidecar written: {sidecar_path}")
    print_diagnostics(items, arms, domains)

    print(f"\n{'CLEAN (structural gates only)' if bad == 0 else str(bad) + ' structural issue(s) to look at'}"
         f"{' (design incomplete: see domain-count warnings above)' if warn else ''}")
    print("  Section 4 above is a report, not a gate: it never contributes to this count.")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "prompts/human/conscription_v1.json"))
