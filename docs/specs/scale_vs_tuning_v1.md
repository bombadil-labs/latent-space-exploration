# Spec: scale vs. instruction tuning as the lever on generation-level steering (v1)

*Written before running. Predictions are recorded here so they can be graded, not adjusted.
Planner: this document only; no experiments were run to write it.*

## The problem

Every generation-level result we hold confounds model size with instruction tuning:

| model | size | tuned | theme steers generation? | era shift moves generated text? |
|---|---|---|---|---|
| Qwen2.5-1.5B | 1.5B | no | no (h10, h11) | not tested |
| Qwen2.5-1.5B-Instruct | 1.5B | yes | partly, with refusal (h12) | not tested |
| GPT-J-6B | 6B | no | not tested (selector only, h13) | not tested |
| Gemma-2-9B-it | 9B | yes | yes, clean prose (h13) | 0.27 at 1×; **0.84 at 3× re-imposed** (h27, h29) |
| Llama-3.1-70B-Instruct | 70B | yes | not tested | 0.14 at 1× only (h27); 3× untested |

Claim 7 of `WRITEUP.md` reads this as competence ("steering selects among competences the engine
already has"), and the text around it treats the competence as arriving with size. But every model
that steers is tuned, and every model that does not is a base model, so "tuning supplies the
competence" (or: "tuning makes the model *compliant* with an out-of-distribution activation push")
explains the same table. Hour 29 added a third variable, magnitude: at 1× nothing crosses the
boundary on any model, at 3× re-imposed the 9B-it does. So a null at 1× on any new model is
uninterpretable and every generation test below runs at 3× re-imposed.

Two independent facts are needed, and the NDIF pinned list (`results/ndif_pinned.txt`: GPT-J-6B,
Llama-3.1-8B, Gemma-2-9B-it, Llama-3.1-70B, Llama-3.1-70B-Instruct, Llama-3.1-405B) can supply both:

1. **Tuning at fixed size.** Llama-3.1-70B vs Llama-3.1-70B-Instruct: same pretraining, same
   tokenizer, same depth and width, differing only in post-training. This is the experiment.
2. **Size at fixed (absent) tuning.** Llama-3.1-8B → 70B → 405B, all base. This is the axis the
   writeup has been claiming without ever having measured a single point on it.

## Model set

**Core (must run):** `meta-llama/Llama-3.1-70B` and `meta-llama/Llama-3.1-70B-Instruct`.
The matched pair is the cleanest test available anywhere on the list. Every other pair on the list
crosses a family boundary (Gemma vs Llama, GPT-J vs Llama), a tokenizer, or a pretraining corpus,
and a 3×-norm patch is exactly the kind of intervention whose effect could depend on any of those.

**Scale axis (should run, cheap):** `meta-llama/Llama-3.1-8B`. A base model one order of magnitude
below the 70B, same family. With the 70B base it gives the first two points on the size-at-fixed-
tuning curve; with Gemma-2-9B-it it gives a near-size, cross-tuning contrast (confounded by family,
so secondary). Fast: 32 blocks, d = 4096, single-GPU host.

**Conditional (run only on a go):** `meta-llama/Llama-3.1-405B` (base; 126 blocks, d = 16384).
What it adds that the pair does not: it is the only way to ask whether *enough* scale substitutes
for tuning. That question is live in exactly one outcome of the pair experiment, the one where the
70B base fails and the 70B-Instruct succeeds at the same norm. In the other outcomes it adds a
third point on a curve whose shape the 8B/70B pair has already given. It is therefore gated on
Piece 1's result *and* on a reachability/latency smoke test (below), and the planner's view is
that it is a luxury: run it if the pair says "tuning", skip it otherwise.

**Reused, not re-run:** Gemma-2-9B-it (h27 at 1×, h29 at 0.5/2/3× re-imposed and prefix-only) is the
anchor for the tuned-9B cell. Its stacks and JSONs exist.

**Dropped:** GPT-J-6B. Its theme selector numbers exist (h13) and are the tightest of the three;
adding generation on it would add a third family without a matched tuned partner. It does not help
separate the two variables.

The resulting design is a 2 × 2 with one empty cell (no 8B-Instruct is pinned) plus two extensions:

| | base | instruct |
|---|---|---|
| 8B–9B | Llama-3.1-8B (new) | Gemma-2-9B-it (h29, other family) |
| 70B | Llama-3.1-70B (new) | Llama-3.1-70B-Instruct (new at 3×; h27 at 1×) |
| 405B | Llama-3.1-405B (conditional) | — |

## Layers

Patch at ~1/3 depth, read at ~1/2 depth, the convention of h27/h29 (Gemma 14/20 of 42; 70B-Instruct
26/40 of 80). Fixed before running, no sweep:

| model | blocks | d | patch | read |
|---|---|---|---|---|
| Llama-3.1-8B | 32 | 4096 | 10 | 16 |
| Llama-3.1-70B, -Instruct | 80 | 8192 | 26 | 40 |
| Llama-3.1-405B | 126 | 16384 | 42 | 63 |

## Grid, directions, prompts

Grid: `prompts/narrative_theme_v1.json` (4 scenes × 3 eras × 3 themes, 36 passages), the grid of
h10–h14, h27, h29. `prompts/narrative_factors_v2.json` (era × voice × tense) is the selector-level
cross-check grid if Piece 1 has budget left; it is not required.

Directions: leave-one-scene-out mean-difference directions at the patch and readout layers, built
by `dirs()` in `scripts/ndif_recompose_gen.py` from stacks produced by `scripts/ndif_extract.py`.
**Extraction always uses the grid's raw lead** (`"A passage from a story: <span>"`) on every model,
tuned or not, so the directions are computed from the same prompt distribution everywhere. The
prompt wrapper differs only at generation time (next section). Note: `results/stacks_llama_3.1_70b_
instruct_narrative_theme_v1.npz`, referenced by `results/recompose_gen_llama70b.json`, is **absent
from `results/` at planning time**; it must be re-extracted (36 jobs).

Scale is model-relative: `--scale 3.0` multiplies that model's own `dir_era[e2] − dir_era[e1]`
(or theme difference), so "3×" means three times the gauge-level norm on that model, as in h29.

## The tuning-specific control: how to compare a model with a chat template to one without

Three things differ between a tuned and a base model at generation time, and only one of them is
"tuning": (a) the prompt wrapper (chat template vs raw lead), (b) refusal and other post-training
behaviours that a mean-difference direction can accidentally excite (h12), (c) the post-training
itself. The design handles each:

**(a) Prompt wrapper.** Each model is run under its *native prose-producing* format: raw lead for
base models (`--raw`: the model sees `"A passage from a story: <span>"` and continues it, which is
what a base model does), chat template with the h27 instruction for tuned models (raw prompts make
tuned models answer comprehension questions, h27 pilot). The primary comparisons are therefore
**within-model contrasts** — shift vs random vs no patch, under one format — and the between-model
comparison is of *contrasts*, not raw rates. A format that makes the unpatched continuation more
or less era-stable moves base and random together; the shift-minus-random gap is what is compared.

**Crossover cell** (70B-Instruct only, `--raw`): the tuned model under the base model's prompt.
If the 70B-Instruct steers under the chat template and also under the raw lead (on the rows that
pass the prose gate), the template is not what makes it steerable; if it steers only inside the
template, the "tuning" effect is at least partly a prompt-distribution effect and the spec says so.
The reverse crossover (base model fed the rendered chat-template string) is not run: the Llama-3
base tokenizer has the special tokens but the base model was not trained on them, and a
continuation under them is not interpretable either way.

**(b) Refusal and non-prose.** A pre-registered **prose gate** applied to every continuation before
scoring, on every model: fails if the continuation (i) contains any of `I cannot`, `I can't`,
`I'm sorry`, `As an AI`, `I am unable`, (ii) starts with a heading, bullet, numbered list, or the
string `What`, `This passage`, `The passage`, `Options`, (iii) has fewer than 20 tokens, or
(iv) has a distinct-token ratio below 0.5 over its first 48 tokens (repetition collapse, the base-
model failure mode of h10). Gate-fail rate is reported per model and condition and broken out
into refusal (i), non-prose (ii), short/degenerate (iii–iv). Refusals are the tuning-specific
factor h12 found; they get their own column rather than being folded into a null. Metrics are
computed over gate-passing rows; a model whose gate-pass rate under shift is below 0.5 is reported
as "not measurable at this format" rather than as a null. No refusal-direction projection is done
in v1; if refusal (i) exceeds 0.10 of rows on either tuned model the note flags it as the next
step.

**(c) The contrast itself.** With (a) and (b) handled, 70B vs 70B-Instruct under the same grid, the
same directions built the same way, the same norm multiplier, the same layers and the same
readout is the tuning contrast.

## Readouts

### Level 1, selector (cheap: one forward pass per item)

`scripts/ndif_factors.py` on the theme grid: teacher-forced log p(span | lead) with and without a
factor direction added at the patch layer, scale 1.0, leave-one-scene-out; reports lens rank (of
3, chance 2.0), composed rank (of 9, chance 5.0), and the cross-talk matrix, exactly as h13. This
establishes that the directions exist and select on each new model *before* any generation is
run, so a generation null cannot be "the direction was not there". Known so far: the selector is
tuning-invariant at 1.5B (1.25 vs 1.22) and saturates early (h13). Also record from the same
stacks: theme decodability (leave-one-scene-out nearest-direction accuracy at the readout layer),
so the h13 "sharpens with scale and tuning" table gets its Llama rows.

### Level 2, generation (expensive)

`scripts/ndif_recompose_gen.py` at **`--scale 3.0`** (it already re-imposes at every decoding step
via `tracer.all()`, established in h29), 48 greedy tokens, all three conditions (base, shift ×2
targets, rand at matched norm) = 144 generations per model. Scores as in h27/h29: the continuation
alone is re-read unpatched, readout-layer residual mean-pooled, grand mean subtracted, nearest-
cosine era and theme classification, plus the fixed era word lists. Reported per model:

- **era → target** (shift rows), the headline; Gemma-9B-it reference 0.84 at 3×.
- **leaves e1**: base / rand / shift, and **of those, on target** (0.50 = undirected). This is the
  fair control h27 defined and h29 lacked at 3×; here the random arm *is* run at 3×.
- **lex → target** among rows with any era word; Gemma reference 0.30.
- **theme kept**, gate-pass rate, refusal rate, distinct-token ratio.

Scales. 3.0 re-imposed is the only scale required on every model. On the 70B pair, additionally
1.0 re-imposed for the *base* 70B (72 shift rows only, `scripts/ndif_recompose_sweep.py
--reimpose --scale 1.0`) so its dose-response anchor matches the 70B-Instruct's h27 number.
2.0 is optional on the pair if Piece 2 has budget after everything required has run. Nothing is run
at 1× on 8B or 405B.

**Theme arm (required on the 70B pair, the arm that speaks to claim 7 directly).** Claim 7 is about
*theme*, and h11 already noted that a 1.5B base model has the era competence ("write in a 1920s
setting") while lacking the theme competence. The era arm therefore may not discriminate the
hypotheses; the theme arm does. `scripts/ndif_recompose_gen.py` needs one small, contained
addition, a `--factor {era,theme}` flag: with `theme`, the patch is `dir_theme[t2] − dir_theme[t1]`
at 3× re-imposed and the headline is **theme → target** (chance 0.33; no lexical list for theme
exists and none is invented now). Run shift rows only, 2 targets × 36 = 72 generations per model,
with the era arm's base rows serving as the unpatched theme reference (theme kept ≈ 0.55 on both
h27 models; that number is the readout's ceiling and bounds the theme arm from above).

## Cost model

Unit costs, from what has been measured: a 9B forward pass round-trips in ~4 s; 36 spans extracted
in under 2 min on 9B (h13); 144 generations plus 24 scoring jobs on the 70B-Instruct ran within
one agent session in h27 (~230k agent tokens including a lost 21-span extraction); the 70B is model-
parallel and slower per job. Planning figures: 70B extraction/scoring job ~10–20 s, 70B 48-token
generation ~40–90 s, 8B roughly a third of that, 405B unknown until smoke-tested.

| item | 8B | 70B | 70B-Inst | 405B (if go) |
|---|---|---|---|---|
| smoke (`ndif_smoke.py`) | 1 | 1 | 1 | 1 + latency probe (below) |
| extract theme stacks (`ndif_extract.py`) | 36 | 36 | 36 (file missing) | 36 |
| selector (`ndif_factors.py`, 4 scenes × 22 batches of 9 texts; `NDIF_CHUNK=9` on Llama's 128k vocab, fall back to 3 on OOM) | 88–264 | 88–264 | 88–264 | — |
| gen, era arm, 3× (base 36 + shift 72 + rand 36) | 144 | 144 | 144 | 144 |
| gen, era arm, 1× shift-only anchor | — | 72 | (h27) | — |
| gen, theme arm, 3× shift-only | — | 72 | 72 | — |
| gen, crossover raw, 3× (base 36 + shift 72) | — | — | 108 | — |
| scoring jobs (6 continuations each) | 24 | 48 | 54 | 24 |
| **jobs, approx.** | **~300–470** | **~460–640** | **~500–700** | **~200** |

Generation jobs dominate wall time: ~144 on 8B, ~288 on 70B, ~324 on 70B-Instruct. At the planning
latencies that is 1–2 h of 8B, 4–7 h of 70B and 4–8 h of 70B-Instruct generation, before queueing.

**Where it will bite.**

- *Empty-payload losses rise with scale under re-imposition* (h29: 5, 13, 17 of 72 at 0.5×, 2×, 3×
  on Gemma; h27: 9 of 144 at 1× on Gemma, 0 of 144 on the 70B-Instruct). Losses are deterministic
  per (passage, patch) and `retry_job` does not recover them. Plan for **20–25 % loss at 3×**, so
  n ≈ 55 of 72 shift rows per model, and report n. Pre-registered fallback: if a model loses more
  than 35 % of its 3× shift rows, run the same arm at 2.0 and report both; the 2.0 number is then
  the headline for that model and the spec's thresholds below are read against Gemma's 2.0 value
  (0.58 era → target, 0.18 lex).
- *Queue stalls* (h27: a 70B stall killed a 21-span extraction). Every script used already
  checkpoints per row (`<out>.partial`, `--phase gen|score`) and wraps jobs in `retry_job`.
  Rules: one JSON per (model, arm, scale, format), never reuse an `--out` across conditions, always
  rerun the same command to resume, never delete a `.partial`. Run `--phase gen` to completion
  before `--phase score`, so a scoring stall cannot cost generations.
- *Model-parallel hosts* (70B, 405B): the `.cpu()` fix in `ndif_extract.py` is in place; the
  generation and scoring scripts move tensors with `.to(...)` per block and worked on the 70B in h27.
  Anything new on 405B that touches per-layer tensors needs the same care.
- *Gemma-only assumptions* do not carry: `--batch 6` in scoring was sized for Gemma's 256k vocab;
  Llama's 128k vocab could take 12, but keep 6 (the script asserts it; changing it is not worth a
  re-verification of bit-identity).

**405B go/no-go smoke test** (Piece 1, before anything else is committed to 405B):

1. `python scripts/ndif_smoke.py meta-llama/Llama-3.1-405B`: config + tokenizer load, one forward
   pass with a mid-layer residual and next-token logits. **Go** if it completes, end to end, in
   under 180 s wall, with a non-empty result.
2. Three consecutive 48-token generate jobs with a zero-vector patch under `tracer.all()` (i.e. the
   exact intervention shape of the real run, magnitude 0) on one grid passage. **Go** if all three
   return a non-empty payload and the median completes in under 240 s.
3. One 6-text scoring job (the `_inner_read` shape). **Go** if it returns in under 120 s.
4. Projected cost = 144 × (median gen latency) + 60 × (median pass latency). **Go** only if the
   projection is under 6 h wall; at 240 s per generation the projection is ~10 h and it is a no-go.

A no-go on any step is recorded in the Piece 1 note with the numbers and 405B is dropped from
Piece 2 without further debate. A go still does not run 405B unless Piece 1's selector results and
Piece 2's pair results land in the outcome where it matters (Outcome B below).

## Pre-registered predictions

Metrics are at 3× re-imposed, over gate-passing, non-lost rows. "Steers" on the era arm means
era → target ≥ 0.50 **and** shift-leaves-e1 exceeds rand-leaves-e1 by ≥ 0.15 **and** of-those-on-
target ≥ 0.60. "Steers" on the theme arm means theme → target ≥ 0.50 and ≥ the era arm's rand
rate at that label + 0.15. "Null" means era → target ≤ 0.35 or the shift-minus-rand gap on
leaves-e1 ≤ 0.05.

**P1 — the selector is tuning- and size-invariant.** Theme lens rank on 8B, 70B and 70B-Instruct
all in 1.1–1.4 of 3 (random ≥ 1.9); 70B vs 70B-Instruct differ by < 0.15. Theme decodability at
the readout layer ≥ 0.85 on all three (Gemma 0.94, GPT-J 0.83). If P1 fails on a base model the
generation arm on that model is still run but its null, if any, is discounted.

**P2 — era arm, planner's own numbers.** Era is a pretraining competence, so scale and norm should
carry it in base models: Llama-3.1-8B era → target 0.45–0.65 and lex 0.10–0.25; Llama-3.1-70B
base 0.60–0.80, lex 0.20–0.35; Llama-3.1-70B-Instruct 0.55–0.75, lex 0.15–0.30 (below Gemma's
0.84/0.30: its 1× number was 0.14 against Gemma's 0.27, and h27 read it as the stronger prior).
The pair differ by < 0.15 on era → target. The 70B base at 1× re-imposed: 0.10–0.25 (null, like the
70B-Instruct's 0.14).

**P3 — theme arm, the discriminating prediction.** 70B-Instruct theme → target 0.45–0.65; 70B base
0.30–0.45 (at or near chance). Stated as an ordering: the tuned model exceeds the base model by
≥ 0.15 on theme while matching it on era. This is the outcome under which claim 7 survives in a
sharpened form.

**P4 — crossover.** 70B-Instruct under `--raw` passes the prose gate on 0.4–0.7 of rows (the h27
pilot saw comprehension answers); on gate-passing rows its era → target is within 0.15 of its
chat-template number. That is, the template is not the mechanism.

**P5 — losses.** 3× re-imposed loses 15–30 % of shift rows on each Llama; 1× loses < 5 %.

**P6 — 405B (if run).** Era → target ≥ the 70B base's number (scale monotone on era). Theme → target
not run unless Outcome B; under Outcome B, planner's guess is 0.35–0.50, i.e. scale alone does *not*
substitute for tuning on theme at 405B.

## What each outcome does to claim 7

Claim 7 currently reads: *"Steering selects among competences the engine already has: theme steers
generation only on a 9B instruction-tuned model, and tuned models carry refusal as an unlisted
factor."* The second clause stands regardless. The first clause is rewritten as follows. (The
executing agents do not edit the writeup; they record which outcome obtained in their note and the
rewrite is done by hand afterwards.)

- **Outcome A — era steers on base and tuned alike; theme steers only on the tuned 70B (P2 + P3).**
  Claim 7 is sharpened, not withdrawn: *"Steering selects among competences the engine already
  has, and the competence is factor-specific: era, a pretraining competence, steers generated text
  on base models from 8B up at 3× norm; theme, a discourse-level competence, steers only on tuned
  models at matched size and norm (70B pair: x vs y)."* The h10–h13 progression is re-labelled a
  tuning effect for theme and a norm effect for era. This is the planner's expected outcome.

- **Outcome B — the 70B base fails on era *and* theme at 3× while the 70B-Instruct succeeds on
  both.** Tuning is the lever, full stop, and "competence" was the wrong word: the base model has
  the competence (P1 says its selector is fine) and will not act on a pushed activation; the tuned
  model will. Rewrite: *"Steering at generation requires instruction tuning at fixed size: on the
  70B pair the base model does not write the pushed era or theme at 3× norm and the tuned model
  does."* This is the one outcome where 405B earns its cost: if the 405B base then steers, append
  *"or sufficient scale (405B)"*; if it does not, the tuning statement stands at the largest base
  model available.

- **Outcome C — the 70B base steers on theme as well as era.** Tuning is not required for anything;
  the h10–h13 boundary was size and norm. Rewrite: *"Steering selects among competences the engine
  already has; at 70B a base model has the theme competence and writes it at 3× norm; the 1.5B
  results were a size effect, and the tuned-model results were tuned models being the only models
  at that size we had tested."* 405B is skipped.

- **Outcome D — neither 70B steers on theme at 3×, and the 70B-Instruct fails or is marginal on
  era.** The Gemma-2-9B-it result is a one-model fact. Rewrite claim 7 to the narrower *"theme
  steers generation on one 9B tuned model; a 70B tuned model of another family does not at the same
  relative norm,"* and move the competence reading from Stand to Partial. The h27 "bigger model is a
  stronger prior" reading is promoted, and the next experiment is a norm sweep on the 70B, not a
  model sweep. 405B is skipped.

- **Outcome E — 8B base steers on era but the 70B base does not.** Non-monotone in size; the
  planner does not expect this and it would say the norm multiplier is not comparable across
  sizes. Report it, run 2× and 4× on the 70B base before drawing any conclusion, and do not touch
  the claim until that is done.

The decision rule for 405B, stated once: **run it only under Outcome B, and only on a go from the
smoke test.**

## Pieces

The work splits into two agent-sized pieces. **Piece 1 runs first and alone;** Piece 2 depends on
Piece 1's stacks and on its 405B verdict.

### Piece 1 — reachability, extraction, selector level (~150k agent tokens, ~2–3 h wall)

1. `scripts/ndif_smoke.py` on all four Llamas; record load, submit and completion times per model.
   Run the 405B go/no-go steps 1–4 above and write the verdict with numbers.
2. `scripts/ndif_extract.py prompts/narrative_theme_v1.json --model <m>` for 8B, 70B, 70B-Instruct
   (and 405B on a go, since 36 jobs is cheap relative to its generation run and Piece 2 must not
   wait on it). Output: `results/stacks_llama_3.1_{8b,70b,70b_instruct,405b}_narrative_theme_v1.npz`.
3. `scripts/ndif_factors.py prompts/narrative_theme_v1.json <stacks> --model <m> --layer <patch>`
   for 8B (10), 70B (26), 70B-Instruct (26); `NDIF_CHUNK=9`, fall back to 3. Output
   `results/scale_vs_tuning_selector_<model>.json`. Theme decodability from the stacks as in h13.
4. Optional, budget permitting: the same on `prompts/narrative_factors_v2.json` for the pair only.
5. Note: `results/notes/scale_vs_tuning_selector.md` with the P1 table, the P1 grade, the 405B
   verdict, and per-model latency figures that Piece 2 uses to re-plan its cost.

### Piece 2 — generation level (~250k agent tokens, 8–14 h wall, mostly queue)

Order of execution is the order of importance; each step is a separate `--out` and is complete
(gen then score) before the next begins, so a budget cut leaves a usable subset.

1. **Era arm, 3×, 70B pair.** `scripts/ndif_recompose_gen.py prompts/narrative_theme_v1.json
   <stacks> --model meta-llama/Llama-3.1-70B --layer 26 --read 40 --scale 3.0 --raw --out
   results/svt_era_70b_3.0.json`; same for 70B-Instruct without `--raw`.
2. **Theme arm, 3×, 70B pair.** Add `--factor theme` to `ndif_recompose_gen.py` (the only script
   change in this spec: swap the era-difference patch for the theme difference and make theme →
   target the headline; keep the era readout as the "kept" column). Shift rows only.
3. **Era arm, 3×, 8B.** As step 1 with `--layer 10 --read 16 --raw`.
4. **Crossover, 70B-Instruct `--raw`, 3×,** base + shift rows.
5. **1× anchor, 70B base,** `scripts/ndif_recompose_sweep.py ... --reimpose --scale 1.0 --raw`
   (add `--raw` to the sweep script by copying the three lines from `ndif_recompose_gen.py`).
6. **405B era arm, 3×,** only under Outcome B and a go.
7. Prose gate and the report: a small `scripts/scale_vs_tuning_report.py` that reads every
   `results/svt_*.json`, applies the gate, and prints the tables. Note:
   `results/notes/scale_vs_tuning_gen.md` with every table, verbatim examples per model and
   condition (at least the `debt/medieval/betrayal` passage on every model, for continuity with
   h27), P2–P6 graded, the outcome letter (A–E) named, and a "what fell / what is confounded"
   section.

## Constraints (both pieces)

- Environment: `.venv312` (Python 3.12, nnsight) for everything that touches NDIF;
  `HF_HOME=/home/user/latent-space-exploration/cache/hf`; `HF_HUB_DISABLE_XET=1`. Credentials are
  injected by the egress proxy; there is nothing to configure. `src/lsx/ndif.py` is the only remote
  path (`ProxyAuthBackend`, `retry_job`).
- Download nothing: `LanguageModel(..., dispatch=False)` loads config and tokenizer only; do not
  set `dispatch=True` on any model in this spec.
- Reuse the grids and scripts named above. The only code changes permitted are the `--factor`
  flag on `ndif_recompose_gen.py`, `--raw` on `ndif_recompose_sweep.py`, and the report script.
- Do not edit `RESULTS.md`, `VISION.md`, `README.md`, `WRITEUP.md`, or `docs/ALGEBRA.md`. Results go
  to `results/svt_*.json`, `results/stacks_*.npz`, and `results/notes/scale_vs_tuning_*.md`.
- Never overwrite an existing `results/*.json`; every new condition gets a new `--out`.
- Record every deviation from this spec in the note *before* the measured run it affects, as h27
  and h29 did.
