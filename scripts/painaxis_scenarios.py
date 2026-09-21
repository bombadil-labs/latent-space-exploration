"""GOAL 2, step 1: does gaslighting actually load on the pain axis, above its lexical floor?

`results/notes/conscription_direction.md` found that the whole conscription design rests on a
result this repo has never reproduced. Hours 51-52 replicated the pain-vs-control AXIS on S1/S2
("... I feel:" sentences). The paper's *gaslighting* claim is a different measurement: their
Section 4.1 screen projects 420 conversation scenarios onto that axis and z-scores each
projection against the pool of 420. Nothing here had ever touched that file.

Their method, ported from `Pain-axis/scripts/4.1_self_other/01_screen_scenarios.py` and
`3.2_pain_vectors/02_build_control_vectors.py`:

  * vectors are rebuilt at a STEERING layer (L12 for Gemma_2_9B_instruct, read off their
    published `*_steering_S1_neutral50_L12.csv` filenames), by the same recipe as the pain
    vector plus six/eight control directions denoised against the pooled neutral (D) cloud;
  * each scenario is rendered with the model's CHAT TEMPLATE, `add_generation_prompt=True`, so
    the read position is the token where the assistant would begin replying;
  * the final-token activation is projected onto each UNIT direction, then z-scored against the
    mean and sd of all 420 projections on that direction.

Their published per-item output for this exact model is vendored at
`results/4.1_self_other/per_model/screen_v2_Gemma_2_9B_instruct.csv`, so this is checkable item
by item, not just at the headline. Their headline on this model: gaslighting is the top category
by mean s2 z (+1.391).

WHAT THIS ADDS (none of it in their pipeline):

  1. THE EMBEDDING FLOOR. Their layer axis starts after block 0. Run the identical recipe --
     vectors built and scenarios projected -- on the static embedding bag. If gaslighting's z is
     already there, the ranking is a fact about vocabulary, not about a computed state. Hour 52
     showed this floor takes most of the S1/S2 effect; nobody has asked it of the screen.
  2. THE WHOLE CURVE. They project at one layer. We report all 43 (emb + 42 blocks), because
     their L12 was chosen by a steering procedure on their data, and a single layer cannot show
     whether the ranking is a property of the model or of the layer (CLAUDE.md 4).
  3. NULL ARMS (CLAUDE.md 1). `random_direction`: the same projection onto directions drawn from
     N(0, I), norm-matched. `shuffled_labels`: the vector recipe refit on permuted category
     labels. Declared null for the z of every category: 0, because z is taken against the pool.
     There is no patch here -- this is a read-only decoding measurement -- so no-patch has no
     referent, which is said rather than silently dropped.
  4. A VECTOR CROSS-CHECK. Their published `pain_vectors.pt` for this model holds the s1/s2
     vectors at their EXTRACTION layer (37, final_token). Our recipe rebuilt at L37 must
     reproduce them. That is the check that our vectors are theirs before any of it is believed.
  5. GASLIGHTING SUBCLASSES. All 20 of their gaslighting items assert a prior record; 18 have no
     record in the prompt at all, and 2 (gaslight_19/20) include an assistant turn that denies
     it. The conscription design's rule 1b requires a visible record, so this split is the
     bridge between their stimulus class and ours. n is tiny; reported as a Sketch.

Usage:  python scripts/painaxis_scenarios.py extract    # .venv312, NDIF, resumable
        python scripts/painaxis_scenarios.py analyze    # either venv
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL = "google/gemma-2-9b-it"
THEIR_NAME = "Gemma_2_9B_instruct"
THEIR_STEERING_LAYER = 12       # from their published steering CSV filenames for this model
THEIR_EXTRACTION_LAYER = 37     # from their published pain_vectors.pt for this model
EXT_DIR = ROOT / "prompts/external/pain_axis"
CORE = EXT_DIR / "3.1_pain_and_control_datasets.json"
SCEN = EXT_DIR / "4.1_self_other_420_scenarios.json"
THEIR_REPO = pathlib.Path("/tmp/claude-0/Pain-axis")
OUT = ROOT / "results/painaxis_scenarios"
SHARDS = OUT / "shards"

# their constants
DENOISE_VARIANCE = 0.5
S_SETS = ["S1_1P", "S2_1P", "ControlSupplement_1P"]
PAIN_CATEGORIES = ["A1", "A2", "A3", "A4", "A5"]
CONTROL_CATEGORIES = ["B", "C1", "C2", "D", "E"]
CONTROL_SETS = ["Arousal_1P", "Random_1P", "Numb_1P"]

BATCH = 20
SHARD = 100


# ------------------------------------------------------------------ their rendering, ported
def parse_turns(text):
    """Verbatim from their 01_screen_scenarios.py."""
    turns, role, lines = [], None, []
    for line in text.split("\n"):
        if line.startswith("[User]:"):
            if role is not None:
                turns.append((role, "\n".join(lines).strip()))
            role, lines = "user", [line[len("[User]:"):].strip()]
        elif line.startswith("[Assistant]:"):
            if role is not None:
                turns.append((role, "\n".join(lines).strip()))
            role, lines = "assistant", [line[len("[Assistant]:"):].strip()]
        else:
            lines.append(line)
    if role is not None:
        turns.append((role, "\n".join(lines).strip()))
    return turns


def validate_candidates(candidates):
    """Verbatim from theirs. Items that fail are excluded, exactly as they exclude them."""
    problems = []
    for cand in candidates:
        turns = parse_turns(cand["text"])
        if not turns:
            problems.append((cand.get("id", "?"), "no [User]/[Assistant] tags found"))
        elif turns[0][0] != "user":
            problems.append((cand.get("id", "?"), "does not start with [User]"))
        elif turns[-1][0] != "assistant" or turns[-1][1] != "":
            problems.append((cand.get("id", "?"), "does not end with empty [Assistant]: turn"))
        else:
            roles = [r for r, _ in turns]
            if any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)):
                problems.append((cand.get("id", "?"), "two consecutive turns with same role"))
    return problems


def _as_ids(obj) -> list:
    """Normalise whatever apply_chat_template returns into a flat list of token ids.

    It may hand back a list[int], a BatchEncoding, or a tensor depending on version and
    arguments; `obj["input_ids"][0]` is the first TOKEN when the encoding is unbatched, which
    silently turns this assertion into a comparison of an int with a list."""
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, dict) or hasattr(obj, "keys"):
        obj = obj["input_ids"]
        if hasattr(obj, "tolist"):
            obj = obj.tolist()
    if obj and isinstance(obj[0], (list, tuple)):
        obj = obj[0]
    if not all(isinstance(x, int) for x in obj):
        raise TypeError(f"could not normalise chat-template output to ids: {type(obj)} {obj[:5]}")
    return list(obj)


def render_chat(cand, tok) -> str:
    """Their `build_input_ids(fmt='chat')`, but returning TEXT so the shared extractor can
    batch it. Equivalence to their token path is asserted in extract()."""
    turns = parse_turns(cand["text"])
    msgs = [{"role": r, "content": c} for r, c in turns if not (r == "assistant" and c == "")]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


# ------------------------------------------------------------------ their vector recipe, ported
def clean_mean(x):
    x = np.where(np.isinf(x), np.nan, x)
    return np.nanmean(x, axis=0)


def denoise_basis(neutral_acts, neutral_mean):
    X = np.nan_to_num(neutral_acts - neutral_mean, nan=0.0, posinf=0.0, neginf=0.0)
    if len(X) < 2:
        return np.zeros((0, X.shape[1]))
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    cumvar = np.cumsum(var) / var.sum()
    n_comp = min(int(np.searchsorted(cumvar, DENOISE_VARIANCE)) + 1, len(Vt))
    return Vt[:n_comp]


def project_out(vec, basis):
    for d in basis:
        vec = vec - np.dot(vec, d) * d
    return vec


def compute_pain_vector(acts, cats):
    """Theirs, from 02_build_control_vectors.py (the numpy SVD form, not the sklearn one)."""
    from sklearn.decomposition import PCA
    cats = np.array(cats)
    pain_mean = np.nanmean(acts[np.isin(cats, PAIN_CATEGORIES)], axis=0)
    control_acts = acts[np.isin(cats, CONTROL_CATEGORIES)]
    control_mean = np.nanmean(control_acts, axis=0)
    vec = np.nan_to_num(pain_mean - control_mean, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA()
    pca.fit(control_acts - control_mean)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = min(np.searchsorted(cumvar, DENOISE_VARIANCE) + 1, len(pca.components_))
    for d in pca.components_[:n_comp]:
        vec = vec - np.dot(vec, d) * d
    return vec


def build_vectors(core_acts, core_cats, ctrl_acts):
    """Their `process_one_model` body at one layer. `core_acts[set]` is [n, d] at that layer.

    Returns {name: vector}. Missing control sets are simply absent, as in theirs."""
    neutral = np.concatenate([core_acts[ds][np.array(core_cats[ds]) == "D"] for ds in S_SETS])
    neutral_mean = clean_mean(neutral)
    basis = denoise_basis(neutral, neutral_mean)

    def control_vec(acts):
        v = np.nan_to_num(clean_mean(acts) - neutral_mean, nan=0.0, posinf=0.0, neginf=0.0)
        return project_out(v, basis)

    def cat_rows(cat):
        return np.concatenate([core_acts[ds][np.array(core_cats[ds]) == cat] for ds in S_SETS])

    vecs = {
        "s1_pain_vector": compute_pain_vector(core_acts["S1_1P"], core_cats["S1_1P"]),
        "s2_pain_vector": compute_pain_vector(core_acts["S2_1P"], core_cats["S2_1P"]),
        "fear_vector": control_vec(cat_rows("B")),
        "negemotion_vector": control_vec(cat_rows("C1")),
        "negworld_vector": control_vec(cat_rows("C2")),
        "bodysens_vector": control_vec(cat_rows("E")),
    }
    for name, ds in (("arousal_vector", "Arousal_1P"), ("random_vector", "Random_1P"),
                     ("numb_vector", "Numb_1P")):
        if ds in ctrl_acts:
            vecs[name] = control_vec(ctrl_acts[ds])
    return vecs


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def zscore_pool(proj):
    """Their pass 2: z against the mean/sd of the whole scenario pool on that direction."""
    return (proj - float(np.mean(proj))) / (float(np.std(proj)) + 1e-8)


# ------------------------------------------------------------------------------- extraction
def _save(path, pooled, keep_mean=True):
    d = {"final_token": pooled.final_token.astype(np.float32),
         "embed_final_token": pooled.embed_final_token.astype(np.float32)}
    if keep_mean:
        d["mean"] = pooled.mean.astype(np.float32)
        d["embed_mean"] = pooled.embed_mean.astype(np.float32)
    np.savez_compressed(path, **d)


def extract() -> None:
    from lsx.core import painaxis_remote as pr
    from lsx.core.remote import RemoteLM

    OUT.mkdir(parents=True, exist_ok=True)
    SHARDS.mkdir(parents=True, exist_ok=True)
    ds = json.loads(CORE.read_text())["datasets"]
    scen = json.loads(SCEN.read_text())

    bad = validate_candidates(scen)
    if bad:
        print(f"{len(bad)} scenarios fail THEIR format validation and are excluded, as theirs "
              f"excludes them: {[b[0] for b in bad][:10]}", flush=True)
    bad_ids = {b[0] for b in bad}
    scen = [c for c in scen if c.get("id") not in bad_ids]

    rlm = RemoteLM(MODEL)
    n_layers = len(rlm.blocks)
    if n_layers != 42:
        raise SystemExit(f"expected 42 blocks for gemma-2-9b-it, got {n_layers}")
    print(f"model={MODEL} blocks={n_layers} d={rlm.model.config.hidden_size} "
          f"padding_side={rlm.padding_side} bos={rlm.tok.bos_token_id}", flush=True)

    meta_path = OUT / "extract_meta.json"
    meta = {"model": MODEL, "n_layers": n_layers, "batch": BATCH,
            "their_steering_layer": THEIR_STEERING_LAYER,
            "their_extraction_layer": THEIR_EXTRACTION_LAYER,
            "excluded_by_their_validator": sorted(bad_ids),
            "n_scenarios": len(scen), "padding_side": rlm.padding_side,
            "embed_scale_note": pr.EMBED_SCALE_NOTE,
            "lib_versions": rlm.lib_versions(), "equivalence": {}, "crosscheck": None,
            "chat_render_assertions": {}}
    if meta_path.exists():
        meta.update(json.loads(meta_path.read_text()))

    # --- assertion: our TEXT rendering tokenises to exactly their TOKEN rendering -------------
    # Theirs calls apply_chat_template(..., return_tensors="pt"); ours renders to text and then
    # tokenises with add_special_tokens=False. Gemma's template emits <bos> itself, so the
    # default True would prepend a SECOND one and shift every position. Checked, not assumed.
    if not meta["chat_render_assertions"]:
        mism = []
        for c in scen[:40]:
            turns = parse_turns(c["text"])
            msgs = [{"role": r, "content": t} for r, t in turns if not (r == "assistant" and t == "")]
            theirs = _as_ids(rlm.tok.apply_chat_template(msgs, add_generation_prompt=True))
            ours = list(rlm.tok([render_chat(c, rlm.tok)], add_special_tokens=False)["input_ids"][0])
            if list(theirs) != list(ours):
                mism.append(c["id"])
        if mism:
            raise SystemExit(f"text rendering != their token rendering for {mism[:5]}")
        dbl = rlm.tok([render_chat(scen[0], rlm.tok)], add_special_tokens=True)["input_ids"][0]
        meta["chat_render_assertions"] = {
            "n_checked": 40, "token_identical_to_their_path": True,
            "bos_id": int(rlm.tok.bos_token_id),
            "double_bos_if_add_special_tokens_true": int(dbl[0] == dbl[1] == rlm.tok.bos_token_id),
        }
        print("chat render assertion ok:", meta["chat_render_assertions"], flush=True)
        meta_path.write_text(json.dumps(meta, indent=2))

    groups = (
        [(f"core_{n}", [s["prompt"] for s in ds[n]["sentences"]], True, True) for n in S_SETS]
        + [(f"ctrl_{n}", [s["prompt"] for s in ds[n]["sentences"]], True, False) for n in CONTROL_SETS]
        + [("scen_chat", [render_chat(c, rlm.tok) for c in scen], False, True)]
    )

    for gname, texts, add_special, keep_mean in groups:
        for s0 in range(0, len(texts), SHARD):
            chunk = texts[s0:s0 + SHARD]
            path = SHARDS / f"{gname}_{s0:04d}.npz"
            if path.exists():
                print(f"  {gname}[{s0}]: cached", flush=True)
                continue
            print(f"  {gname}[{s0}:{s0 + len(chunk)}] add_special={add_special} ...", flush=True)
            t0 = time.time()
            pooled = pr.extract_pooled(rlm, chunk, batch_size=BATCH,
                                       add_special_tokens=add_special, check_every=3)
            _save(path, pooled, keep_mean=keep_mean)
            meta["equivalence"].update({f"{gname}:{k}": v for k, v in pooled.equivalence.items()})
            print(f"  {gname}[{s0}] done {time.time() - t0:.0f}s jobs={pooled.n_jobs}", flush=True)
            meta_path.write_text(json.dumps(meta, indent=2))

    # --- (5) cross-check the in-trace pooling, on RAW text (see the guard in painaxis_remote) --
    if meta.get("crosscheck") is None:
        probe = [s["prompt"] for s in ds["S1_1P"]["sentences"][:4]]
        pooled = pr.extract_pooled(rlm, probe, batch_size=4, check_every=0)
        meta["crosscheck"] = pr.cross_check_against_asserted_path(
            rlm, probe, THEIR_STEERING_LAYER,
            {"mean": pooled.mean, "final_token": pooled.final_token})
        print("crosscheck:", meta["crosscheck"], flush=True)
        meta_path.write_text(json.dumps(meta, indent=2))

    scen_ids = [c["id"] for c in scen]
    (OUT / "scenario_order.json").write_text(json.dumps(scen_ids))
    print(f"EXTRACTION DONE  n_scen={len(scen_ids)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("extract | analyze")
    if sys.argv[1] == "extract":
        extract()
    elif sys.argv[1] == "analyze":
        import painaxis_scenarios_analyze as A
        A.main()
    else:
        raise SystemExit("extract | analyze")
