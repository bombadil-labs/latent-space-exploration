# Scale vs. tuning, Piece 1: reachability, extraction, selector level

Executes `docs/specs/scale_vs_tuning_v1.md`, Piece 1 only. No generation arms run (Piece 2).

## 1. Smoke tests (`scripts/ndif_smoke.py`)

| model | config+tokenizer | submit | end-to-end | result |
|---|---|---|---|---|
| Llama-3.1-8B | 4.2s | 1.3s | 4.1s | OK, non-empty (mid-layer resid norm 13.72, top-5 next-token sane) |
| Llama-3.1-70B | 3.5s | 1.0s | 3.7s | OK, non-empty (resid norm 18.41) |
| Llama-3.1-70B-Instruct | 2.1s | 1.1s | 3.8s | OK, non-empty (resid norm 17.09) |
| Llama-3.1-405B | 3.9s (run 1) / 2.6s (run 2) | 0.9-1.0s | **errored at 0.2s, 3/3 attempts** | fail |

### 405B go/no-go: **NO-GO**

Step 1 (`ndif_smoke.py meta-llama/Llama-3.1-405B`) was run three times (two back-to-back, one after
a 15s wait). All three attempts errored deterministically after 0.2s of polling with:

```
RemoteException: Model is not pinned and hotswapping is not supported for this API key.
See https://nnsight.net/status/ for a list of scheduled models.
```

This is a `RemoteException`, not a transport/timeout error, so `retry_job`'s retry classes
(`TimeoutError`, `httpx.TransportError`, `ConnectionError`, `OSError`) never catch it — it is not a
queue stall, it is the server refusing the model outright for this API key. This contradicts
`results/ndif_pinned.txt`, which lists `meta-llama/Llama-3.1-405B` as `PINNED RUNNING`; the pin list
is evidently aspirational/stale relative to what this key is actually served. **Step 1 did not
complete end-to-end** (the go criterion), so this is an unambiguous no-go and steps 2-4 (generation
latency probe, scoring-job latency, cost projection) were not run — nothing about them would change
a no-go already established at step 1. Per the spec, 405B is dropped from Piece 2 without further
debate, and Piece 2 does not need to re-plan around any 405B latency numbers because none exist.

This also means **Outcome B's clause "or sufficient scale (405B)" cannot be tested at all** in this
project — not because the pair experiment ruled it out, but because the model is unreachable at this
API key. Any writeup rewrite that lands on Outcome B must say "the tuning statement stands at the
largest base model available (70B, no larger base model reachable)" rather than leaving 405B as a
live possibility.

## 2. Extraction (`scripts/ndif_extract.py prompts/narrative_theme_v1.json`)

36/36 spans each, zero losses, zero retries needed:

| model | wall time | output |
|---|---|---|
| Llama-3.1-8B | 228s | `results/stacks_llama_3.1_8b_narrative_theme_v1.npz` |
| Llama-3.1-70B | 782s | `results/stacks_llama_3.1_70b_narrative_theme_v1.npz` |
| Llama-3.1-70B-Instruct | 720s | `results/stacks_llama_3.1_70b_instruct_narrative_theme_v1.npz` |

(70B-Instruct stacks were absent from `results/` at planning time as the spec predicted; re-extracted
here, no replica eviction this run, unlike hour 31's re-extraction of the same file.)

## 3. Selector battery (`scripts/ndif_factors.py`, `NDIF_CHUNK=9`, scale 1.0, leave-one-scene-out)

Required layers (patch = spec's 1/3-depth convention: 8B@10, 70B pair@26):

| model / layer | era lens rank/3 (rand) | theme lens rank/3 (rand) | composed rank/9 | wall |
|---|---|---|---|---|
| Llama-3.1-8B @10 | 1.19 (1.22) | 1.22 (1.28) | 1.81 | 264s |
| Llama-3.1-70B @26 | 1.11 (1.22) | 1.11 (1.14) | 1.50 | 485s |
| Llama-3.1-70B-Instruct @26 | 1.00 (1.14) | 1.06 (1.14) | 1.33 | 310s |

Cross-talk matrices (all three models): every row reads `era 0.25 / theme 0.25` (and `rand 0.25/0.25`
on 8B and 70B, `theme 0.23/0.23` for 70B-Instruct's rand row) — i.e. no cross-talk asymmetry
resolved at this scale; all three land at the same flat 0.25 split.

**Theme decodability** (leave-one-scene-out nearest-mean-direction accuracy, local numpy over the
saved stacks, readout layer = 16 for 8B / 40 for the 70B pair, mirroring h13's protocol):

| model | era decodability | theme decodability |
|---|---|---|
| Llama-3.1-8B | 1.00 | 0.89 |
| Llama-3.1-70B | 1.00 | 0.89 |
| Llama-3.1-70B-Instruct | 1.00 | 0.92 |
| (h13 reference) Gemma-2-9B-it | — | 0.94 |
| (h13 reference) GPT-J-6B | — | 0.83 |

**Cheap layer sweep on the 70B pair** (extra, addressing the hour-33 confound at the selector
level — reuses the already-extracted stacks, no re-extraction, ~2 more selector jobs per model):
patch@14/read-equivalent (Gemma's *absolute* depth) vs patch@26 (the spec's *fraction*-matched
depth), both scored on the same stacks:

| model / layer | era lens rank/3 (rand) | theme lens rank/3 (rand) | composed rank/9 | wall |
|---|---|---|---|---|
| Llama-3.1-70B @14 (absolute) | 1.11 (1.25) | 1.25 (1.19) | 1.72 | 484s |
| Llama-3.1-70B @26 (fraction) | 1.11 (1.22) | 1.11 (1.14) | 1.50 | 485s |
| Llama-3.1-70B-Instruct @14 (absolute) | 1.08 (1.25) | 1.22 (1.14) | 1.47 | 318s |
| Llama-3.1-70B-Instruct @26 (fraction) | 1.00 (1.14) | 1.06 (1.14) | 1.33 | 310s |

**What this sweep says about the hour-33 confound.** At the selector level the absolute-depth layer
(14) and the fraction-matched layer (26) give numbers within ~0.15 rank of each other on both models
— the selector picture (weak-but-present factor lens, near-collapsed random control, see below) does
not change qualitatively between the two candidate layers. This does not resolve the hour-33 question
(whether patch@26/read@40 is the right generation-level intervention point) but it does narrow it:
whatever makes the 70B's generation-level era shift plateau at 0.43 instead of crossing 0.5, it is
not that the selector-level signal is materially stronger at one of these two depths than the other
— both are comparably weak selectors relative to Gemma. If a further layer sweep is wanted, it
should be done at the *generation* level directly (Piece 2's job, not cheaply available at the
selector level from stacks already in hand) rather than by trying more selector layers, since the
selector doesn't discriminate between them.

## P1 graded

**P1 as literally stated is partially refuted.** The "lens rank 1.1-1.4" clause holds on 8B (1.22)
and 70B (1.11) but 70B-Instruct (1.06) sits just under the stated floor; all three are within 0.15
of each other (1.06-1.22), so the tuning/size-invariance claim holds in the relative sense the spec
cared about. Theme decodability clears the ≥0.85 bar on all three (0.89, 0.89, 0.92), below Gemma's
0.94 and above GPT-J's 0.83, consistent with "sharpens with scale and tuning" only weakly (70B is
not sharper than 8B; 70B-Instruct is marginally sharper than the 70B base, by 0.03).

**The clause "random ≥ 1.9" is refuted on every model, and this is the note's most important
finding.** In h13 the random-direction control landed near chance (era/theme random ranks 1.97-2.19
across Qwen-1.5B/GPT-J/Gemma). Here, on all three Llamas and at both swept layers, the random
control's rank sits in the same 1.1-1.3 band as the factor direction itself — not near the chance
value of 2.0 for a rank-of-3 test. Concretely: 8B theme random 1.28 vs factor 1.22; 70B era random
1.22 vs factor 1.11; 70B-Instruct theme random 1.14 vs factor 1.06. The two conditions are barely
separated on any model, at any of the two layers tested. Two readings, not adjudicated here: (a) a
matched-norm additive vector at these layers on Llama models perturbs the teacher-forced log-prob in
a way that is not specific to the semantic direction — i.e. the lens result on Llama is weaker
evidence of a real, isolable factor direction than the same numbers would be on Gemma/GPT-J, because
the control that should isolate "any perturbation of this size" from "this particular perturbation"
does not separate; or (b) something architecture- or scale-specific about Llama's residual stream at
these hidden sizes (4096/8192) makes any large-norm push at this depth non-neutral in a way GPT-J
and Gemma's smaller hidden sizes did not exhibit. Piece 2 should not treat the selector numbers above
as confirming that the era/theme directions are cleanly isolated on Llama the way they were on
Gemma — the generation-level results (already in hand from hours 27/29/33) are the more trustworthy
signal for whether these directions do anything, precisely because they are read from real
downstream text rather than from a log-prob difference that a random control cannot distinguish from
the real thing here.

## P2-P6 (Piece 2)

Not run — no generation arms in this piece, per the task's Piece-1-only scope. **P6 is moot**: the
405B no-go removes any way to test the scale-only substitution clause, so whatever outcome the 70B
pair's generation arms land on, "or sufficient scale (405B)" cannot be appended in Piece 2 — the
tuning statement (if Outcome B obtains) stands at 70B as the largest reachable base model, full stop.

## What is still confounded

- **The random-control anomaly above** is new and unexplained; it was not visible in h13 because
  h13 never ran a Llama model through this selector script. It should be resolved (or at least
  understood) before anyone reads a Llama selector number as equivalent evidence to a Gemma or
  GPT-J one.
- **Depth-fraction vs absolute-depth** (the hour-33 flag) is narrowed but not closed: the selector
  doesn't discriminate the two candidate layers, so the generation-level plateau at 0.43 is not
  explained by "wrong selector layer" — but neither has a generation-level layer sweep been run to
  positively rule in some other layer that would cross 0.5. That sweep is a Piece 2 (or later)
  question, not answered here.
- **405B is untested**, not ruled out by data — it is unreachable at this API key, a credentialing
  fact rather than a scientific one. If key access changes, the go/no-go should be re-run before
  assuming the no-go still holds.
- The theme decodability figures use a from-scratch local nearest-mean-direction script
  (`leave-one-scene-out`, mirroring but not reusing h13's exact code path, which is not in this
  repo's current scripts/) — numbers should replicate h13's method but were not cross-checked
  against any existing script output for Gemma/GPT-J in this session.
- Selector-level results say nothing about generation; hours 27/29/33 already show the 70B's
  generation-level era shift plateaus at 0.43 (not crossing 0.5) while Gemma reaches 0.84 at the
  same relative scale — this note does not re-litigate that, it only adds selector-level context.

## Files

- `results/stacks_llama_3.1_{8b,70b,70b_instruct}_narrative_theme_v1.npz` (new, this run)
- `results/scale_vs_tuning_selector_{8b,70b,70b_instruct}.json` (required layers)
- `results/scale_vs_tuning_selector_{70b,70b_instruct}_l14.json` (extra depth-sweep layer)
- `results/notes/scale_vs_tuning_p1.md` (this note)
- Logs (`results/extract_*.log`, `results/selector_*.log`) kept small; no large JSON or raw logs
  copied into this note.

## Headline numbers

- **405B: NO-GO** (RemoteException, not pinned for this API key, 3/3 attempts, ~0.2s each).
- Theme decodability: 8B 0.89, 70B 0.89, 70B-Instruct 0.92 (vs Gemma 0.94, GPT-J 0.83) — tuning/size
  roughly invariant, weak upward trend with tuning at fixed size.
- Theme lens rank/3: 8B 1.22, 70B 1.11, 70B-Instruct 1.06 (all well under chance 2.0) — but the
  random-direction control is equally low on every model/layer tested (1.14-1.28, not the ~2.0-2.2
  h13 saw on Gemma/GPT-J), so the lens's specificity to the semantic direction is not established
  on any Llama model in this run.
- Cheap 70B-pair layer sweep (14 vs 26): selector numbers do not materially differ by layer,
  narrowing but not resolving the hour-33 depth-fraction confound.
