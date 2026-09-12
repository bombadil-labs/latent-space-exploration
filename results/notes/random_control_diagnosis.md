# Why the random-direction control collapsed on every Llama (hour 34)

**Cause, in one line: the patch never reached 8 of the 9 texts in the batch.** In
`scripts/ndif_factors.py` the patch was written `B[l].output[0][:] = B[l].output[0] + v`. Under
transformers >= 4.54 a Llama (also Gemma-2, Qwen) decoder layer returns a **bare Tensor
`[batch, seq, d]`**, not a `(hidden_states, ...)` tuple, so `output[0]` means *batch row 0*, not
*hidden states*. Hour 34 ran with `NDIF_CHUNK=9`, i.e. all nine variants of a scene in one padded
batch, so **every patch — factor and random alike — was added to the first sequence of the batch
only**. The other eight scored identically to base, gain exactly 0. Combined with a strict `>` in
the rank formula (all ties score rank 1), this manufactures ranks near 1.2 for *any* direction.

It is a code bug, not a property of Llama. It is fully reproduced locally on Qwen2.5-1.5B, and the
corrected battery on Llama-3.1-8B gives a healthy control (random 2.11, no-patch 2.00).

## 1. The fingerprint that identifies it without running anything

Every single (X) cross-talk row in all five hour-34 JSONs reads `frac_era = frac_theme = 0.25` to
seven decimals (48 rows per file, min 0.24999991, max 0.25000009). Exactly 0.25/0.25 is the
algebraic signature of a gain vector of the form `const + β·(one combo)`: one combo moved, eight
identical. Hour 13's Gemma file, by contrast, ranges 0.007–0.846. Nine texts, one moved — the batch
row.

The rank arithmetic follows: with only `combos[0] = (medieval, betrayal)` moved (gain < 0) and a
strict `>`, every target scores rank 1 except when the target *is* `combos[0]`, which scores 3. That
is 1 row in 9, so the expected rank is **11/9 = 1.22 for the factor direction and for the random
direction alike** — the hour-34 band (1.00–1.28) is noise around that constant, in both columns.
The (D) composed test predicts (8·1 + 9)/9 = 1.89; observed 1.81 on 8B.

## 2. Direct confirmation on NDIF (3 tiny jobs, `meta-llama/Llama-3.1-8B`, layer 10)

Three texts in one padded batch; a fixed random vector of norm 10 added at block 10; teacher-forced
log-prob gain per batch row:

| patch expression | gain row 0 | row 1 | row 2 |
|---|---|---|---|
| `B[l].output[0][:] = B[l].output[0] + v` (as shipped) | −7.4688 | **0.0** | **0.0** |
| add to the whole tensor (fix) | −7.4688 | −2.2285 | −6.7876 |

`B[l].output` on Llama-3.1-8B came back with shape `[3, 18, 4096]` — a bare Tensor. Same probe on
the other two remote models used in this project:

| model (on NDIF, today) | block output | as-shipped patch reaches | verdict |
|---|---|---|---|
| meta-llama/Llama-3.1-8B | Tensor `[3,18,4096]` | row 0 only | broken for batch > 1 |
| google/gemma-2-9b-it | Tensor `[3,16,3584]` | row 0 only | broken for batch > 1 *today* |
| EleutherAI/gpt-j-6b | tuple | whole batch | immune |

Gemma is immune *in the hour-13 results* only because that run scored one passage per job
(batch 1, forced by its 256k vocabulary): at batch 1, `output[0]` is the only row and the patch is
correct by accident. Its logged control (random 2.08 / 2.11) and non-degenerate cross-talk confirm
it empirically. But the same script re-run on Gemma with `NDIF_CHUNK>1` today would break.

## 3. Local reproduction (`Qwen/Qwen2.5-1.5B`, fp32 CPU, layer 20, theme grid, 4 scenes)

Same directions, same 9-text padded batch, same scoring expression; the nnsight patch emulated with
a forward hook in two modes (`row0` = the bug, `all` = the fix), plus a no-patch arm.
Rank/3, chance 2.0; "strict" is the shipped rank (ties → 1), "tie-avg" is mid-rank on ties.

| condition | patch reaches | era strict | era tie-avg | theme strict | theme tie-avg |
|---|---|---|---|---|---|
| factor direction | whole batch (fix) | **1.22** | 1.22 | **1.33** | 1.33 |
| random direction | whole batch (fix) | **2.00** | 2.00 | **2.11** | 2.11 |
| factor direction | row 0 only (bug) | 1.11 | 1.89 | 1.17 | 1.94 |
| random direction | row 0 only (bug) | **1.31** | 2.08 | **1.14** | 1.92 |
| **no patch at all** | — | **1.00** | 2.00 | **1.00** | 2.00 |

The collapse reproduces exactly, on a non-Llama model: under the bug the random control lands at
1.14–1.31, indistinguishable from the treatment, precisely as in hour 34. With the fix, Qwen's
numbers reproduce hour 13's (1.36/1.25 factor, 2.00/1.97 random). So the answer to the three
candidate causes is: **(a) a bug in how the control was applied — confirmed**; (b) a degenerate rank
metric plus a position/length prior — *not* the driver (see §4, the no-patch baseline is at chance
once ties are handled, and the same prior would have to move with the level, which it cannot); (c) a
genuine property of Llama — ruled out (§5).

## 4. No-patch baselines (this was missing from the battery and is now in it)

The gain metric is `logprob(patched) − logprob(base)`, so a no-patch arm is a second scoring job
with no direction added. Measured:

| model / layer | no-patch rank/3, strict `>` (as shipped) | no-patch rank/3, mid-rank (fixed) |
|---|---|---|
| Qwen2.5-1.5B @20 (local) | **1.00** (36/36 rows rank 1) | **2.00** |
| Llama-3.1-8B @10 (NDIF, fixed script) | 1.00 (implied: all gains 0) | **2.00** (72/72 rows exactly 2.0) |

Re-scoring is bit-exact, so the no-patch gains are all identically zero. This is the **second bug**:
with `rank = 1 + #{gd[c] > gd[t]}`, a total tie scores 1.0 — *doing nothing looks like a perfect
selector*. That is what turned a silently no-op patch into "rank 1.22, far under chance". Under
mid-rank the same runs would have screamed: the bugged arms score 1.89–2.08, i.e. chance.

A **third, smaller bug** was found on the way: `tok.padding_side` is `left` for Llama, Gemma and
GPT-J, but the scoring mask zeroes the first `n_lead-1` positions to drop the lead — which is only
correct for right padding. With left padding, part of the lead is scored on every row except the
longest, and by a different amount per row. Fixed by forcing `padding_side = "right"`.

## 5. The corrected Llama-3.1-8B battery (what hour 34 should have measured)

Re-extracted stacks (36/36 spans, decodability replicates hour 34 exactly: era 1.00, theme 0.89 at
layer 16) and re-ran the fixed script, `NDIF_CHUNK=9`, layer 10, scale 1.0, 279 s.
`results/selector_8b_fixed.json`, `results/selector_8b_fixed.log`.

| Llama-3.1-8B @10 | factor dir | random | no patch | chance |
|---|---|---|---|---|
| era lens rank/3 | **1.22** | 2.11 | 2.00 | 2.0 |
| theme lens rank/3 | **1.33** | 2.11 | 2.00 | 2.0 |
| composed rank/9 | **2.03** | — | — | 5.0 |

Cross-talk (fraction of gain variance): era row 0.69 / 0.11, theme row 0.20 / 0.52, rand row
0.25 / 0.21 — i.e. a normal, diagonal-dominant matrix, not the flat 0.25 of hour 34.

**So the Llama-8B selector is real and specific**, and looks like Gemma's and Qwen's rather than
weaker. The hour-34 numbers were coincidentally in the same ballpark as the true ones (1.22 is the
artifact's fixed point *and* the true era rank), which is exactly why the treatment column did not
look alarming — only the control did.

## 6. What this invalidates

| hour | claim / numbers | status |
|---|---|---|
| 34 | 8B@10 era 1.19 (rand 1.22), theme 1.22 (rand 1.28), composed 1.81 | **void** — artifact |
| 34 | 70B@26 era 1.11 (1.22), theme 1.11 (1.14), composed 1.50 | **void** |
| 34 | 70B-Instruct@26 era 1.00 (1.14), theme 1.06 (1.14), composed 1.33 | **void** |
| 34 | 70B@14 era 1.11 (1.25), theme 1.25 (1.19), composed 1.72; 70B-Instruct@14 era 1.08 (1.25), theme 1.22 (1.14), composed 1.47 | **void** |
| 34 | all three cross-talk matrices ("every row reads era 0.25 / theme 0.25") | **void** — the 0.25s are the artifact itself, not a finding about cross-talk |
| 34 | "the random control collapses on every Llama"; P1's "random ≥ 1.9 refuted"; "the lens is not shown to be specific on any Llama" | **withdrawn** — a measurement bug; on the corrected 8B run the control is at 2.11 and the lens is specific |
| 34 | layer-sweep conclusion "selector numbers do not materially differ between layers 14 and 26" | **void** — both layers were measuring the same artifact constant; the hour-33 depth confound is *not* narrowed |
| 34 | P1 graded "partially refuted" | **regrade needed**; the 8B leg now reads era 1.22 / theme 1.33, inside P1's 1.1–1.4 band |
| 34 | theme decodability 0.89 / 0.89 / 0.92 (8B / 70B / 70B-Instruct) | **stands** — local numpy over the stacks, no patching. 8B re-verified from freshly re-extracted stacks (1.00 era / 0.89 theme) |
| 34 | 405B no-go (RemoteException, not pinned for this key) | **stands** — unrelated |
| 13 | Gemma-2-9B-it theme 1.25 (rand 2.11), era 1.17 (2.08); GPT-J-6B theme 1.11 (2.06), era 1.22 (2.19); Qwen-1.5B theme 1.25 (1.97), era 1.36 (2.00) | **stand** — GPT-J returns a tuple (immune); Gemma ran at batch 1; Qwen ran locally one text at a time. All three controls sit at chance and all cross-talk fractions are non-degenerate, which is only possible if the patch landed on every candidate |
| 6, 7, 8, 9, 10, 11, 23 | all local selector batteries (`stage5_*`, `stage6_*`): factor 1.03–1.39 with random 1.83–2.25 | **stand** — local per-text scoring, one text per forward pass, no batch dimension to mis-index |
| 12, 13, 14, 19, 22, 27, 29, 31, 33 | every generation-level and shift result on NDIF (`ndif_generate`, `ndif_shift`, `ndif_commutator`, `ndif_recompose_gen`, `ndif_recompose_sweep`, `ndif_absential_*`) | **stand** — all of these trace a *single* prompt, where `output[0]` is the only batch row and the patch is correct by accident |
| 28, 30, 31, 32 | time-translation selector (`time_translation_selector.py`) | **stands** — local `lm.logprob`, per-text |

Two caveats, conservatively:

- **No pre-hour-34 result has a no-patch baseline.** For the ones with a random control at chance
  this does not matter: a control at 2.0 already proves the patch landed and that the metric was not
  reading a fixed prior. It matters only where a control is missing or low.
- One such case: hour 8's three-factor run (`stage6_qwen1.5b_three_l14.json`) has **tense factor
  1.03 with random 1.44** — the random control is well below chance there, unlike every other local
  run. That is not this bug (local, unbatched), but it is the one logged selector number whose
  control does not clear chance, and the hour-8 tense claim should carry that caveat until re-run
  with the no-patch arm.

## 7. The fix (applied, `scripts/ndif_factors.py`)

1. `resid(block)` helper resolves tuple-vs-Tensor block output; the patch is applied to the whole
   hidden-state tensor (`h[:] = h + v`). Verified against both shapes on NDIF (Llama: Tensor,
   GPT-J: tuple).
2. Mid-rank on ties (`1 + #{>} + ½·#{=}`), so a patch that changes nothing scores at chance.
3. A `none` (no-patch) condition is now run once per scene and reported next to factor and random.
4. `tok.padding_side = "right"` so the lead mask is correct.

Not refactored: the other NDIF scripts share the `output[0][:]` idiom but all trace a single prompt,
so they are correct today. They are a live trap — anyone who batches them will silently re-create
this bug. If one of them is ever batched, port `resid()` first.

## 8. What piece 2 of the scale-vs-tuning spec should do differently

- Re-run the Piece-1 selector battery with the fixed script before quoting any Llama selector
  number: the 70B and 70B-Instruct legs have no valid numbers at all right now (only 8B has been
  redone), and the tuning/scale comparison that Piece 2 is supposed to build on does not yet exist.
  Cost is small: extraction is already scripted (~230 s at 8B, ~13 min at 70B) and the battery is
  ~5 min per model.
- Require three arms in every selector battery — factor, random, **no-patch** — and treat a no-patch
  rank that is not at chance as a stop-the-line failure before reading any treatment number.
- Require a positive control on the plumbing: assert that the number of candidates whose score
  actually moved equals the batch size. One line, and it would have caught this in hour 34.
- Do not use the hour-34 layer sweep to choose a generation layer; it measured nothing. The
  hour-33 depth question is exactly as open as it was.
- Keep reading `frac` columns as a diagnostic: a cross-talk matrix that is suspiciously uniform
  (here, exactly 0.25) means degenerate gains, not a flat factor structure.

## Files

- `scripts/ndif_factors.py` (fixed: patch application, tie handling, no-patch arm, padding side)
- `results/selector_8b_fixed.json`, `results/selector_8b_fixed.log` (corrected Llama-3.1-8B battery)
- `results/stacks_llama_3.1_8b_narrative_theme_v1.npz`, `results/extract_8b_redo.log` (re-extracted;
  the hour-34 stacks were not kept)
- Affected, to be regarded as void: `results/scale_vs_tuning_selector_{8b,70b,70b_instruct}.json`,
  `results/scale_vs_tuning_selector_{70b,70b_instruct}_l14.json`, and §3 and the "P1 graded" /
  "layer sweep" sections of `results/notes/scale_vs_tuning_p1.md`.
