"""Exercise `lsx.core.conscription`'s local path on the demo items in
`prompts/human/conscription_v1.json`. Not a study run: writes checkpoints under
`results/conscription_dry_run/` (gitignored, like every `.npz`) and prints what it did.

    .venv/bin/python scripts/conscription_dry_run.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

CKPT_DIR = ROOT / "results" / "conscription_dry_run"


def main():
    import torch
    from lsx.core import conscription
    from lsx.model import LM

    spec = json.loads((ROOT / "prompts" / "human" / "conscription_v1.json").read_text())
    items = spec["items"]
    arm_order = tuple(spec["_meta"]["arms"])
    print(f"{len(items)} items, arms={arm_order}")

    t0 = time.time()
    lm = LM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device="cpu", dtype=torch.float32)
    print(f"model loaded in {time.time() - t0:.1f}s, {lm.n_layers} layers, d={lm.d_model}, "
         f"tokenizer padding_side={lm.tok.padding_side!r}")

    # 1. first pass -- nothing checkpointed yet
    if CKPT_DIR.exists():
        import shutil
        shutil.rmtree(CKPT_DIR)
    t0 = time.time()
    stacks = conscription.run_local(lm, items, checkpoint_dir=CKPT_DIR, arm_order=arm_order)
    print(f"extracted {len(stacks)} item-stacks in {time.time() - t0:.1f}s")
    for iid, st in stacks.items():
        print(f"  {iid}: acts {st.acts.shape}, layers {st.layers[0]}..{st.layers[-1]}, "
             f"equivalence min cos {min(st.checks['batched_vs_single_min_cos'].values()):.6f}")

    # 2. checkpoint re-use: second call must not re-extract (touch mtimes, compare)
    mtimes_before = {p: p.stat().st_mtime for p in CKPT_DIR.glob("*.npz")}
    t0 = time.time()
    stacks2 = conscription.run_local(lm, items, checkpoint_dir=CKPT_DIR, arm_order=arm_order)
    dt = time.time() - t0
    mtimes_after = {p: p.stat().st_mtime for p in CKPT_DIR.glob("*.npz")}
    print(f"second call (checkpoint re-use) took {dt:.2f}s, files unchanged: "
         f"{mtimes_before == mtimes_after}")

    # 3. paired per-item contrast
    paired = conscription.paired_contrasts(stacks2, arm_order=arm_order)
    print(f"paired contrasts over {len(paired['item_ids'])} items, {len(paired['layers'])} layers")

    # sanity: every X_vs_X pair must read ~0 at every layer
    max_self = 0.0
    for a in arm_order:
        d = paired["pairs"][f"{a}_vs_{a}"]["diff_norm"]
        max_self = max(max_self, float(d.max()))
    print(f"max self-pair (X vs X) diff-norm across all arms/items/layers: {max_self:.2e}")

    # the sanity Arm, at the last layer
    last_layer = paired["layers"][-1]
    arm = conscription.sanity_arm(paired, "enact", last_layer)
    print(f"sanity_arm('enact','enact')@L{last_layer}: value={arm.value:.2e} off_null={arm.off_null} "
         f"n_independent={arm.n_independent} unit={arm.unit!r}")

    # 4. a real cross-arm reading, for a look (NOT a claim -- see the module docstring)
    for pair in ("enact_vs_true", "enact_vs_report", "enact_vs_exit", "enact_vs_neutral"):
        d = paired["pairs"][pair]["diff_norm"]
        c = paired["pairs"][pair]["cosine"]
        # report at a few layers, whole curve is what a real study would keep
        for li in (0, len(paired["layers"]) // 2, -1):
            L = paired["layers"][li]
            print(f"  {pair} @L{L}: diff_norm per item {d[:, li].round(3).tolist()}  "
                 f"cosine per item {c[:, li].round(4).tolist()}")

    print("\nOK: local path exercised end to end.")


if __name__ == "__main__":
    main()
