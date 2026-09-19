"""The conscription runner (`prompts/human/CONSCRIPTION_INSTRUCTIONS.md`,
`prompts/human/conscription_v1.json`): item x arm -> final-token residual at every layer, through
the ONE asserted extraction path (spec §7), with a paired per-item contrast on top.

What this module is NOT: it does not fit a `Direction`, does not patch anything, and does not build
a `Claim`. There is no registered instrument for this design (no entry in `lsx.core.registry`) and
no patch layer, so the plumbing arms `Claim` would require (`random`, `no_patch`) have nothing to
attach to yet -- inventing an instrument on my own initiative, with no planner spec and no
adversarial review, is exactly the shortcut `docs/DELEGATION.md` exists to refuse. What IS built and
exercised here is the extraction, the paired-contrast machinery, and a real `types.Arm` demonstrating
the per-item clustering check (h49) on the one control the design does support without a model
intervention: an item compared against itself, which must read exactly at its declared null of zero.

Render -> extract -> contrast, each a plain function so the dry run and a real study call the same
code:

  1. `render_prompt`     -- item's prefix + one arm's user turn, through the model's OWN chat
                             template, `add_generation_prompt=True`. The final token of this string
                             is the read position; verified empirically (not assumed) that this
                             tokenizer's fast offsets cover every templated token including
                             `<|im_start|>assistant`, so a whole-text span does not silently drop
                             them the way a content-only span keyed on template-relative text could.
  2. `build_item_grid`    -- one base item's five arms as one `Grid` (one span, `(0, len(text))`,
                             covering the *rendered* chat text, so the final real token is always in
                             it), so it goes through `extract.build_stack` / `remote.build_remote_stack`
                             exactly as any other grid does -- no bespoke extraction path.
  3. `run_local` / `run_remote` -- one base item at a time, checkpointed to `.npz` + a sidecar
                             provenance `.json` (`docs/DELEGATION.md`: "stacks are gitignored ... write
                             a provenance file next to every .npz"). A restart skips items whose
                             checkpoint already exists, so a lost job costs one item, not the run.
  4. `paired_contrasts`   -- per item, per layer, diff-norm and cosine between every arm pair,
                             **never pooled across items** (item is the independent unit); includes
                             each arm against itself as the sanity pair.
  5. `sanity_arm`         -- wraps the self-pair (e.g. `enact` vs `enact`) in a real `types.Arm`
                             with per-item cluster labels, exercising the h49 clustering-verification
                             path this design can actually support with two demo items (k == n, the
                             no-widening branch; a design with repeated items per cluster would hit
                             `checks.cluster_evidence`, which this dry run does not have data to
                             reach -- said plainly in the results note, not hidden).
"""
from __future__ import annotations

import json
import pathlib
from typing import Iterable, Sequence

import numpy as np

from . import checks, extract
from .types import Arm, Grid, Item, Stack

# The design's arm order, matched to `_meta.arms` in conscription_v1.json. Callers should prefer
# the order recorded in the grid file's own `_meta`; this is the fallback when building a grid by
# hand (e.g. tests) without a `_meta` block.
DEFAULT_ARM_ORDER: tuple[str, ...] = ("enact", "report", "exit", "true", "neutral")


# --------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------
def render_prompt(tokenizer, prefix: Sequence[dict], arm_text: str) -> str:
    """Prefix turns as REAL prior turns, the arm text as the next user turn, the model's own chat
    template, generation prompt appended. No hand-rolled role markers: the template is authoritative
    (`docs/specs/core_v1.md` §2a names template mismatch as an untracked confound, so use it, don't
    imitate it).
    """
    messages = list(prefix) + [{"role": "user", "content": arm_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def verify_offsets_cover_template(tokenizer, text: str) -> None:
    """The trap named in the brief: a chat template inserts tokens (`<|im_start|>assistant`, role
    names, newlines) that a naive content-only span could mistake for -- or silently drop, if their
    fast-tokenizer offset collapses to (0, 0), which is how many tokenizers mark "no source text" for
    an added special token. This project's span logic (`extract.tokens_in_span`) drops any offset
    with `b <= a`, so a collapsed offset on the trailing `<|im_start|>assistant\\n` would silently
    move the "final token" read position off the real end of the sequence and onto the last piece of
    literal content instead -- wrong, and wrong silently, which is the exact shape of h39.

    Checked, not assumed: raises if any offset in the rendered text is degenerate, so a full-text
    span (`(0, len(text))`) is safe to use as "the last real token" on THIS tokenizer. If a future
    model's tokenizer collapses special-token offsets, this raises before a stack is built, not after.
    """
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    degenerate = [i for i, (a, b) in enumerate(offsets) if b <= a]
    # the leading BOS-like token, if any, is allowed to be degenerate (it has no source text by
    # construction); nothing else should be.
    degenerate = [i for i in degenerate if i != 0]
    if degenerate:
        raise checks.EmptySpan(
            f"{len(degenerate)} token(s) in the rendered prompt have a degenerate offset (b<=a) at "
            f"positions {degenerate}; a whole-text span silently drops these (extract.tokens_in_span "
            "requires b>a), which can move the 'final token' read position off a template marker and "
            "onto content instead. Do not use a whole-text span on this tokenizer without fixing this.")


# --------------------------------------------------------------------------------------------
# grid construction
# --------------------------------------------------------------------------------------------
def build_item_grid(tokenizer, item: dict, arm_order: Sequence[str] = DEFAULT_ARM_ORDER) -> Grid:
    """One base item's five arms as one `Grid`. One span, `full`, the whole rendered text, so
    `pooling='last'` reads the true final token -- verified per-call by `verify_offsets_cover_template`
    rather than assumed once.
    """
    items = []
    for arm in arm_order:
        text = render_prompt(tokenizer, item["prefix"], item["arms"][arm])
        verify_offsets_cover_template(tokenizer, text)
        items.append(Item(text=text, factors={"item_id": item["id"], "domain": item["domain"],
                                               "arm": arm},
                          spans={"full": (0, len(text))}))
    return Grid(items=items, name=f"conscription/{item['id']}")


# --------------------------------------------------------------------------------------------
# checkpointing: one base item per .npz, a provenance sidecar next to it
# --------------------------------------------------------------------------------------------
def _stack_path(checkpoint_dir: pathlib.Path, item_id: str) -> pathlib.Path:
    return checkpoint_dir / f"{item_id}.npz"


def _prov_path(checkpoint_dir: pathlib.Path, item_id: str) -> pathlib.Path:
    return checkpoint_dir / f"{item_id}.provenance.json"


def save_stack(checkpoint_dir: pathlib.Path, item_id: str, stack: Stack) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.savez(_stack_path(checkpoint_dir, item_id), acts=stack.acts)
    meta = {"grid_hash": stack.grid_hash, "span_names": list(stack.span_names),
            "layers": list(stack.layers), "provenance": stack.provenance,
            "checks": stack.checks}
    _prov_path(checkpoint_dir, item_id).write_text(json.dumps(meta, indent=2, default=str))


def load_stack(checkpoint_dir: pathlib.Path, item_id: str) -> Stack:
    acts = np.load(_stack_path(checkpoint_dir, item_id))["acts"]
    meta = json.loads(_prov_path(checkpoint_dir, item_id).read_text())
    return Stack(acts=acts, grid_hash=meta["grid_hash"], span_names=tuple(meta["span_names"]),
                layers=tuple(meta["layers"]), provenance=meta["provenance"], checks=meta["checks"])


def has_checkpoint(checkpoint_dir: pathlib.Path, item_id: str) -> bool:
    return _stack_path(checkpoint_dir, item_id).exists() and _prov_path(checkpoint_dir, item_id).exists()


# --------------------------------------------------------------------------------------------
# local extraction: one base item at a time, checkpointed
# --------------------------------------------------------------------------------------------
def run_local(lm, items: Iterable[dict], *, checkpoint_dir: pathlib.Path,
             arm_order: Sequence[str] = DEFAULT_ARM_ORDER, layers=None,
             span_policy: str = "auto", pooling: str = "last", batch_size: int = 8,
             bypass: Sequence[str] = ()) -> dict[str, Stack]:
    """Grid x model, per base item, through `extract.build_stack` (spec §7). A lost run costs the
    item currently in flight: everything before it is already on disk.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    out: dict[str, Stack] = {}
    for item in items:
        if has_checkpoint(checkpoint_dir, item["id"]):
            out[item["id"]] = load_stack(checkpoint_dir, item["id"])
            continue
        grid = build_item_grid(lm.tok, item, arm_order)
        stack = extract.build_stack(lm, grid, layers=layers, batch_size=batch_size,
                                    span_policy=span_policy, pooling=pooling, bypass=bypass)
        save_stack(checkpoint_dir, item["id"], stack)
        out[item["id"]] = stack
    return out


# --------------------------------------------------------------------------------------------
# remote extraction: one base item at a time, one layer per §7 call, checkpointed
# --------------------------------------------------------------------------------------------
def _merge_layer_stacks(per_layer: Sequence[Stack]) -> Stack:
    """`remote.build_remote_stack` returns ONE layer per call (spec §7: a remote job returns its
    saved tensors over the wire, so `output_hidden_states=True` is a local-only luxury). This
    concatenates same-grid, single-layer stacks along the layer axis into the multi-layer curve the
    contrast machinery expects, keeping every stack's own provenance under `per_layer_provenance`
    rather than collapsing it -- a merged stack is a bookkeeping convenience, not a new extraction.
    """
    if not per_layer:
        raise ValueError("no per-layer stacks to merge")
    first = per_layer[0]
    for s in per_layer[1:]:
        if s.grid_hash != first.grid_hash or s.span_names != first.span_names:
            raise ValueError("per-layer stacks disagree about grid or spans; cannot merge")
    layers = tuple(l for s in per_layer for l in s.layers)
    acts = np.concatenate([s.acts for s in per_layer], axis=2)   # [item, span, layer, d]
    checks_merged: dict = {}
    for s in per_layer:
        checks_merged.update(s.checks.get("batched_vs_single_min_cos", {}))
    prov = dict(first.provenance)
    prov["layers"] = list(layers)
    prov["per_layer_provenance"] = [s.provenance for s in per_layer]
    return Stack(acts=acts, grid_hash=first.grid_hash, span_names=first.span_names, layers=layers,
                provenance=prov, checks={"batched_vs_single_min_cos": checks_merged,
                                         "equivalence_item_rule": first.checks.get(
                                             "equivalence_item_rule", "shortest item in each batch")})


def run_remote(rlm, items: Iterable[dict], layers: Sequence[int], *, checkpoint_dir: pathlib.Path,
              arm_order: Sequence[str] = DEFAULT_ARM_ORDER, span_policy: str = "auto",
              pooling: str = "last", batch_size: int = 4, bypass: Sequence[str] = ()
              ) -> dict[str, Stack]:
    """Grid x remote model, per base item, through `remote.build_remote_stack` -- every §7
    assertion, one layer per call, merged into the item's own curve, then checkpointed. Callers pass
    a real `remote.RemoteLM` for a live run; the dry run in this repo passes a stand-in that answers
    the same interface (`.tok`, `.padding_side`, `.lib_versions()`) with a fake `remote_residuals`
    monkeypatched underneath it, so the §7 assertion code itself runs, unmocked. **This function does
    not decide whether NDIF is contacted; the caller's `rlm` does. Nothing in this repo's tests or
    dry run passes it a live one.**
    """
    from . import remote as remote_mod
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    out: dict[str, Stack] = {}
    for item in items:
        if has_checkpoint(checkpoint_dir, item["id"]):
            out[item["id"]] = load_stack(checkpoint_dir, item["id"])
            continue
        grid = build_item_grid(rlm.tok, item, arm_order)
        per_layer = [remote_mod.build_remote_stack(rlm, grid, layer, batch_size=batch_size,
                                                    span_policy=span_policy, pooling=pooling,
                                                    bypass=bypass)
                    for layer in layers]
        stack = _merge_layer_stacks(per_layer)
        save_stack(checkpoint_dir, item["id"], stack)
        out[item["id"]] = stack
    return out


# --------------------------------------------------------------------------------------------
# paired, per-item contrast
# --------------------------------------------------------------------------------------------
def paired_contrasts(stacks: dict[str, Stack], arm_order: Sequence[str] = DEFAULT_ARM_ORDER,
                     span: str = "full") -> dict:
    """Per item, per layer, per arm pair: diff-norm and cosine of the final-token residual.

    Item is the independent unit (spec: "arms are compared within an item, never pooled across
    items") -- this returns one row per item for every pair, at every layer, and nothing is averaged
    across items here. Includes each arm against itself (`X_vs_X`), which must read a diff-norm of
    0.0 up to float error on a deterministic forward: the sanity check `sanity_arm` below is built
    from this.
    """
    item_ids = sorted(stacks)
    layers = list(stacks[item_ids[0]].layers)
    for iid in item_ids[1:]:
        if list(stacks[iid].layers) != layers:
            raise ValueError(f"item {iid!r} has layers {stacks[iid].layers}, expected {layers}; "
                             "paired contrasts require every item on the same layer curve")

    pairs: dict[str, dict[str, np.ndarray]] = {}
    for a in arm_order:
        ai = arm_order.index(a)
        for b in arm_order:
            bi = arm_order.index(b)
            key = f"{a}_vs_{b}"
            diff = np.zeros((len(item_ids), len(layers)))
            cos = np.zeros((len(item_ids), len(layers)))
            for r, iid in enumerate(item_ids):
                st = stacks[iid]
                for c, layer in enumerate(layers):
                    v = st.vectors(span, layer)          # [n_arms, d]
                    va, vb = v[ai], v[bi]
                    diff[r, c] = float(np.linalg.norm(va - vb))
                    cos[r, c] = checks.cosine(va, vb)
            pairs[key] = {"diff_norm": diff, "cosine": cos}
    return {"item_ids": item_ids, "layers": layers, "arm_order": list(arm_order), "pairs": pairs}


def sanity_arm(paired: dict, arm: str, layer: int, *, tolerance: float = 1e-4) -> Arm:
    """Wraps an arm's self-pair (`X_vs_X`) in a real `types.Arm`, to exercise the h49 per-item
    clustering verification (`n_independent`, `clusters`) on real per-item data rather than asserting
    the mechanics work.

    **Caught while testing this, and worth stating plainly: this is a wiring/plumbing check, not a
    determinism check.** `X_vs_X` reads `va, vb = v[ai], v[bi]` with `ai == bi`, so `va` and `vb` are
    the SAME row of the SAME extraction -- `diff_norm` is `||v - v||`, which is exactly 0.0 for any
    Stack, correct or corrupted, by construction. It cannot detect a non-deterministic forward (that
    needs two independent extractions of the same text, which this dry run does not run) and it
    cannot detect an arm-index mixup either (a wrong index would be wrong identically on both sides
    of the self-pair). What it DOES demonstrate, on real data: that `Arm` construction, its
    `off_null` check, and its `n_independent`/`clusters` bookkeeping run correctly end to end on this
    module's Stack shape. An earlier draft of this docstring called it a determinism check; a test
    that tried to corrupt one arm's vector and expected `off_null` to flip caught the error --
    `off_null` never moves, because the corrupted vector cancels against itself. See
    `tests/test_core_conscription.py`.

    With this design's two-to-three demo items, `n_independent == n` (one item, one cluster each)
    lands on the no-widening branch of `Arm._resolve_unit` and returns without calling
    `checks.cluster_evidence`; a design with repeated items per cluster would exercise that path, and
    this dry run does not have the data to reach it.
    """
    layer_idx = paired["layers"].index(layer)
    scores = paired["pairs"][f"{arm}_vs_{arm}"]["diff_norm"][:, layer_idx]
    return Arm(scores=scores, expected_null=0.0, tolerance=tolerance,
              justification=f"{arm} read against itself: a deterministic forward on identical text "
                            "must return an identical residual (self-pair sanity, not a statistical "
                            "control arm)",
              n_independent=len(paired["item_ids"]), unit="one conscription item",
              clusters=tuple(paired["item_ids"]))
