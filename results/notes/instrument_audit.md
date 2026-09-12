# Instrument audit (hour 39): five checks that had never been run

`docs/INSTRUMENTS.md` closes with a list of things this project measures with instruments nobody has
independently checked. This note checks the five that were named, hardest-hitting first, fixes what
was cheaply fixable, and lists what has to be withdrawn or qualified.

Headline: **hour 31 does not carry the hour-36 batch-row bug — but it carries a different one, found
here, that corrupts 363 of its 480 extracted vectors. Hour 31 falls.**

---

## (a) Hour 31's Gemma extraction: the right passage, the wrong positions

**The question.** `scripts/ndif_time_translation_extract.py` batches six passages per NDIF job via
six `tracer.invoke` blocks and reads `B[bi].output[0]`. If `output[0]` meant "batch row 0" there, as
it did in `ndif_factors.py` at hour 34, five of every six extracted vectors would be passage 0's and
hour 31 would be void.

**Verdict on that question: NO. `tracer.invoke` scoping makes the idiom safe, and hour 31 is not an
instance of the hour-36 bug.** Decided from the code and then confirmed on NDIF.

From the code (nnsight 0.7.0, `intervention/batching.py`): each `tracer.invoke` gets a
`batch_group = [start, size]`, and every envoy value read inside that invoke passes through
`Batcher.narrow`, which does `acts.narrow(0, batch_start, batch_size)` on the batch dimension. Inside
invoke *i* the block output is therefore `[1, seq, d]` — that invoke's row and no other — so
`output[0]` is that passage whether the layer returns a tuple or a bare tensor. This is exactly the
"per-passage scoping" the audit brief asked about, and it is real.

**But the same test found a second, unrecorded bug in the same read, with the same blast radius.**
nnsight's `LanguageModel` loads its tokenizer with `padding_side="left"` (Gemma's own default is
left too) and pads all six invokes to a common length; `narrow` slices the **batch** dimension only,
so the sequence dimension keeps the padded length. The script computes its interval/state token
indices on the *unpadded* text and indexes them **absolutely**. Every passage that is not the longest
of its six therefore had its spans read `n_pad` positions too early — out of the padding block, or
out of earlier tokens of the right passage.

**Decisive test on NDIF** (`scripts/audit_h31_batch_check.py`,
`results/audit_h31_batch_check.log`, google/gemma-2-9b-it, layer 20, 18 s): three real v2-grid
passages of lengths 44 / 54 / 74, extracted (A) together in one batched job by the hour-31 code path
and (B) one per job. Cosine A vs B for the same passage:

| pooled span | passage 0 (30 pad) | passage 1 (20 pad) | passage 2 (0 pad) |
|---|---|---|---|
| interval span (absolute idx, the hour-31 idiom) | **0.287** | **0.297** | 0.999997 |
| state span (absolute idx, the hour-31 idiom) | **0.794** | **0.916** | 0.999991 |
| last token (position-stable under left padding) | 0.999965 | 0.999965 | 0.999954 |

Two readings in one table. The last-token row is ~1.0 for all three and its cross-passage cosines
are 0.73–0.76, so **each invoke read its own passage**: no batch-row bug. The span rows are ~1.0
only for the longest passage — the one with no padding — and the corrupted vectors match no other
passage either (all cross-cosines low), so this is misalignment inside the right passage, not the
wrong passage. Norms make it concrete: the batched interval vector has norm 179 against 418 for the
same passage alone; it is pooling padding.

**Blast radius on the v2 grid** (computed exactly with the Gemma tokenizer over the 80 six-passage
jobs in extraction order): of 480 passages, **117 were the longest of their job and are clean; 363
(76%) are corrupted.** Mean padding 5.4 tokens, max 24; **232 passages had their entire interval
span inside the padding block**, and 203 had their state span start inside it. The padding is not
random with respect to the design: jobs are consecutive grid items, i.e. one subject's successive
Δt levels, so the shift is correlated with Δt — the very axis the measurement reads.

**So hour 31 falls.** Its standing claim — "the shared clock replicates on Gemma-9B", variance share
0.478, Spearman(‖shared‖, log Δt) 0.683, adjacent/distant cosine 0.87/0.61, phrase-only ratio 1.67 —
was computed from 76%-corrupted vectors and must be withdrawn as untested. (Hour 31's other half,
"subject clocks absent, 0 of 8", was already withdrawn at hour 32 as a noise-floor artifact; it is
now doubly withdrawn.) `WRITEUP.md`'s model-invariance claim for the clock loses its Gemma leg.

**Fix and re-run.** `_spans()` now returns **negative** token indices (counted from the end), which
is correct under left padding and identical at batch 1, plus an assertion that the tokenizer is
left-padding (verified offline: under a simulated 7-token left pad the negative indices select
exactly the unpadded span's token ids, while the absolute indices return pad tokens). The whole grid
was then re-extracted on NDIF with the fixed script (480 passages, 80 jobs, 854 s,
`results/stacks_gemma_2_9b_it_time_translation_v2_auditfix.npz`) and measurements 1–7 recomputed
(`results/time_translation_gemma_auditfix_measures.json`, `results/h31_measure_fixed.log`):

| quantity (Gemma-2-9B-it) | hour 31 as logged | corrected | Qwen-1.5B L14 for reference |
|---|---|---|---|
| shared variance fraction @20 | 0.478 | **0.501** | 0.545 |
| Spearman(‖shared‖, log Δt) @20 | 0.683 | **0.767** | 0.68 |
| adjacent / distant cosine @20 | 0.87 / 0.61 | **0.89 / 0.57** | 0.88 / 0.59 |
| ‖shared_exp‖/‖shared_ctrl‖ @20 | 1.67 | **2.50** | ~3.1 |
| τ(s) @20 | 1 day, all 8 | 1 day, all 8 | 1 day, all 8 |
| shared fraction @9 / @31 | 0.498 / 0.459 | 0.480 / 0.488 | — |
| Spearman @9 / @31 | 0.750 / 0.667 | 0.583 / 0.717 | — |
| phrase ratio @9 / @31 | 1.93 / 1.72 | 2.97 / 3.31 | — |

**Every logged hour-31 number moves, so every one of them is withdrawn and replaced.** The largest
move is the phrase-only ratio — 1.67 → 2.50 at layer 20, 1.72 → 3.31 at layer 31 — which is the
measure that reads the *interval* span, i.e. exactly the span that was being pooled out of the
padding for 232 passages; hour 31's "the phrase-only ratio is weaker at 9B (1.67)" was an artifact of
the bug, and at 2.5–3.3 the ratio is in Qwen's range instead. The Spearman at layer 9 moves the other
way (0.750 → 0.583), so the corruption was not uniformly deflationary — it was noise correlated with
the design.

**The qualitative conclusion survives the correction**: on clean vectors the shared clock still
carries ~0.50 of the variance, still orders monotonically with log Δt, and still shows the same
adjacent/distant cosine geometry as Qwen, so "the clock replicates on Gemma-9B" is re-established —
but on the new numbers, not the logged ones, and still subject to the v2 grid's lexical confound
(hours 32, 35: the v2 state spans restate the interval, so any v2 shared-clock number, on either
model, is contaminated). τ is unchanged at 1 day for all 8 subjects, which is expected: that
estimator was already shown at hour 32 to return "1 day" on pure noise.

## (b) The six NDIF scripts that still used `output[0][:]`

All six were found **safe as run** — every one traces a single prompt per job, where row 0 is the
only row — and all six are now ported to the `resid()` tuple-or-tensor helper anyway, because the
"safe because batch 1" property is invisible at the call site and one `NDIF_CHUNK`-style change
re-creates hour 34 silently. No logged number changes.

| script | patch sites | why it was safe | logged results it produced |
|---|---|---|---|
| `ndif_generate.py` | 1 (`tracer.all()` generate) | one prompt per `model.generate` job | hour 13 Gemma-2-9B-it steered generations |
| `ndif_shift.py` | 1 (trace) | one text per trace; read span also batch-1 | hour 14 era-shift readout, `results/stage7_gemma9b_shift.json` |
| `ndif_commutator.py` | 1 (inside the patch loop) | one prompt per generate; `read_tokens` one text | hours 19 and 22, `results/commutator_gemma9b{,_v2}.json` |
| `ndif_recompose_gen.py` | 1 | generate is one prompt; the 6-invoke re-read is invoke-scoped **and** pools `[..., -k:, :]`, which is right-aligned and therefore immune to the left-padding shift of (a) | hour 27, `results/recompose_gen_gemma9b.json` |
| `ndif_recompose_sweep.py` | 2 | same as above | hour 29 dose-response, `results/recompose_sweep_*.json` |
| `ndif_absential_probe.py` | 2 (score + generate) | one text per job | hour 18, `results/absential_probe_gemma9b.json` |

Two notes for the record. First, `recompose_gen`/`recompose_sweep` use the same six-invoke batching
as hour 31 and are safe from (a) **only** because they pool the last *k* tokens rather than absolute
indices — that is luck, not design, and the invariant is now stated in (a)'s docstring. Second, the
remaining `output[0]` occurrences in `scripts/` (`ndif_extract.py`, `ndif_tokens.py`,
`ndif_tokens_resume.py`, `ndif_absential_continuation.py`, `ndif_smoke.py`) are **reads** at batch 1
whose `.reshape(-1, D)[-1]` idiom is shape-tolerant; they were left alone, and the same caution
applies if anyone batches them.

## (c) Local selectors: 1-on-ties and no no-patch arm

`stage5_factors.py`, `stage6_factors.py` and `time_translation_selector.py` all ranked with
`1 + #{gain > gain[target]}`, which scores a total tie as rank 1 — under that rule *doing nothing is
a perfect selector*, the second bug of hour 36 — and none of them ran the treatment-removed arm.
All three now use mid-rank ties (`1 + #{>} + ½·#{=}`) and carry a no-patch arm (a scoring pass with
a zero direction, so the plumbing is exercised, not assumed). `time_translation_selector.py` already
had a no-patch *raw* arm; it now also has the no-patch *gain* arm, which is the one the tie rule
made degenerate (all gains identically zero → rank 1.00 under the old rule, 5.00 under mid-rank).

**Re-run.** The cheapest local battery behind a headline claim is hour 8's three-factor
composition (`prompts/narrative_factors_v2.json`, Qwen2.5-1.5B, layer 14, scale 1, leave-one-scene-
out, 4 scenes × 18 spans). Re-run with the fixed script — mid-rank ties *and* the new no-patch arm —
in 46 min on CPU (`results/stage6_qwen1.5b_three_l14_auditfix.json`,
`results/stage6_three_auditfix.log`):

| hour 8 test | as logged (strict ties, no no-patch arm) | fixed (mid-rank + no-patch) | chance |
|---|---|---|---|
| (B) era lens, rank/3 | 1.25 (random 2.00) | **1.25** (random 2.00, **no-patch 2.00**) | 2.0 |
| (B) voice lens, rank/3 | 1.24 (random 2.25) | **1.24** (random 2.25, **no-patch 2.00**) | 2.0 |
| (B) tense lens, rank/2 | 1.03 (random 1.44) | **1.03** (random 1.44, **no-patch 1.50**) | 1.5 |
| (D) three composed, rank/18 | 2.81 | **2.81** (**no-patch 9.50**) | 9.5 |
| (X) cross-talk matrix | 0.58/0.14/0.00, 0.16/0.64/0.01, 0.11/0.11/0.47, rand 0.25/0.21/0.02 | identical to two decimals | — |

**Nothing moves, and that is the informative outcome.** Every treatment and random number reproduces
to two decimals, so **no logged local selector claim changes**: the tie rule never fired, because
local scoring is one text per forward pass and its gains are distinct floats — ties only appeared in
the remote battery, where a mis-targeted patch left eight candidates *identically* unmoved. The
no-patch arm is the part that was genuinely missing, and it reads **exactly chance on every test**
(2.00 / 2.00 / 1.50 / 9.50, 0.00 deviation), which is the positive control hours 4–11, 23 and 28–32
never had: their patches demonstrably land and their rank metric has no fixed prior.

One flagged item resolves. `docs/INSTRUMENTS.md` singles out hour 8's tense control — factor 1.03 of
2 with **random 1.44** against chance 1.5, the one logged selector control that does not clear
chance. With the no-patch arm beside it the reading is now clean: no-patch is 1.500 exactly, so 1.44
is not a plumbing artifact or a metric prior but a small deflection of the random arm on a 2-level
factor (72 rows). The hour-8 tense claim can keep its caveat about a slightly-low random control,
but it no longer needs the "may be the hour-36 failure mode" caveat added at hour 36.

The remaining two scripts (`stage5_factors.py`, `time_translation_selector.py`) carry the same two
fixes but were not re-run; on this evidence their logged numbers are expected to be unchanged too,
and their no-patch arms will run the next time either script is used.

## (d) The abstraction ladder (hours 15, 26) has no null — what it needs and what it would cost

No new experiment was run. For the record, precisely what is missing:

1. **The merge test has no null and needs one badly.** Hour 15 reports "the nearest 16k feature is
   more general in 12 of 15 cases" (13 of 15 on the broad corpus) with no null distribution. The
   correct null is *not* 50%: the two dictionaries have different marginal generality distributions
   by construction (16k mean 0.19 vs 131k 0.10 on the broad corpus), so pairing each 131k feature
   with a **random** 16k feature already yields "more general" well above half the time. The null to
   quote is P(≥12 of 15 | random 16k partner), estimated by 10⁴ resamples, ideally in a
   decoder-cosine-matched form (draw the random partner from 16k features at a similar cosine to
   control for "nearest-by-cosine" also meaning "more typical"). **Cost: zero model runs, pure numpy
   over `results/sae_ladder_picard.json` and `results/sae_ladder_v2.json`; minutes.**
2. **The width effect has no random-feature control.** "Narrow keeps more general features" (0.19 vs
   0.10) compares *features selected by firing on the description*. The control is the same
   generality statistic over a size-matched **random** sample of features from each dictionary: if
   the gap survives selection-free sampling, the finding is a property of the dictionaries, not of
   abstraction. **Cost: no NDIF — the layer-20 token residuals for the 111-passage broad corpus are
   cached (`results/tokens_gemma9b_broad_l20.npz`, 36 MB) and both Gemma Scope dictionaries are in
   `cache/hf`; it is one CPU encode pass, tens of minutes at 131k width.**
3. **The flow has no matched-count control and no permutation null.** Theme purity falling 0.63 →
   0.30 as g_k rises, and "era's structure dies before theme's", are both read off a curve with no
   null. Two arms: (i) at each g_k keep a **random** subset of features of the same surviving size —
   if purity decays the same way, the decay is a dimensionality effect, not a generality ordering;
   (ii) permute theme/era labels across passages for a null purity band at each g_k, and bootstrap
   over passages for the era-before-theme ordering. **Cost: zero model runs, re-analysis of the same
   cached feature matrix; under an hour including the write-up.**

Total: one CPU-only session, no NDIF, no downloads. Until it is run, hours 15 and 26 should be read
as descriptive — the effects are large and monotone, but none of the three headline statements has a
distribution to be surprised against, and this is the same missing arm as every other failure in
`docs/INSTRUMENTS.md`.

## (e) `results/ndif_pinned.txt` corrected

The file listed Llama-3.1-405B as `PINNED RUNNING`. It is not. Re-fetched
`https://api.ndif.us/status` today: the 405B **base** model has no running deployment at all
(`deployment_level: WARM`, not pinned), which corroborates hour 34's deterministic "Model is not
pinned and hotswapping is not supported for this API key". Pinned and RUNNING for this key today:
`EleutherAI/gpt-j-6b`, `google/gemma-2-9b-it`, `meta-llama/Llama-3.1-8B`,
`meta-llama/Llama-3.1-70B`, `meta-llama/Llama-3.1-70B-Instruct` — the five models the project
actually used. The file now carries its provenance, the date, and the unpinned-but-hot list
(including `meta-llama/Llama-3.1-405B-Instruct`, which *is* pinned — only the base 405B this project
needed is absent).

---

## What must now be withdrawn or qualified

| item | status after this audit |
|---|---|
| **Hour 31**: "the shared clock replicates on Gemma-9B" — shared variance 0.478, Spearman 0.683, adjacent/distant cos 0.87/0.61, phrase-only ratio 1.67 | **WITHDRAWN** — 363 of 480 extracted vectors had their pooled spans read from padding or from shifted positions (§a). Superseded by the corrected re-run below. |
| **Hour 31**: "subject clocks absent at 9B (0 of 8, all layers)" | already withdrawn at hour 32 (noise floor); now also rests on corrupted vectors |
| `WRITEUP.md`'s "the clock is model-invariant" (Qwen + Gemma) | **qualified**: its Gemma leg is the hour-31 number |
| Hours 13, 14, 18, 19, 22, 27, 29 (the six `output[0][:]` scripts) | **stand** — batch 1 throughout (§b); scripts hardened, no numbers change |
| All local selector batteries (hours 4–11, 23, 28–32) | **stand** — the representative re-run reproduces every logged number to two decimals and the new no-patch arm reads exactly chance (§c) |
| Hour 8's tense control caveat (random 1.44) | **qualified, not withdrawn** — no-patch is exactly 1.50, so the low random arm is not an instrument failure (§c) |

## Files

- `scripts/audit_h31_batch_check.py`, `results/audit_h31_batch_check.log`, `results/audit_h31_batch_check.npz` — the decisive (a) test
- `scripts/ndif_time_translation_extract.py` — negative (right-aligned) span indices + padding-side assertion
- `scripts/ndif_{generate,shift,commutator,recompose_gen,recompose_sweep,absential_probe}.py` — `resid()` helper at every patch site
- `scripts/{stage5_factors,stage6_factors,time_translation_selector}.py` — mid-rank ties + no-patch arm
- `results/ndif_pinned.txt` — corrected
