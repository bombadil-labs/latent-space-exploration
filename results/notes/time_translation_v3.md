# Time translation v3: removing the lexical restatement of the interval

Companion to `time_translation_v2.md` and `subject_clocks.md` (hour 32). Hour 32 found that the
v2 state texts restate the interval inside the state span ("*One day later* the mayfly is dead",
"*Ten thousand years* of resurvey..."), so Δt is recoverable from the state span lexically before
any computation — layer-0 discrimination Spearman is 1.00, confounding every shared-clock result
built on this grid (hours 28, 30, 31, 32). v3 rewrites every state span (t0 through 1,000,000
years, all 8 subjects, all 3 paraphrases = 240 texts) so no state text names, numbers or
paraphrases its own interval, while keeping v2's far-Δt vocabulary-matching property.

## 1. Grid and leak checker

`prompts/time_translation_v3.json` — same 8 subjects, 9 Δt + t0, 3 paraphrases,
`[[interval: ...]] [[state: ...]]` structure, phrase-only controls; built by
`scripts/_build_v3_states.py` from v2. `scripts/time_translation_leak_check.py` flags a state
span if it contains a duration word (day/week/.../millennium, "later", "since", "ago", ...), an
explicit "N <duration unit>" construction, or any non-stopword token shared with that row's own
interval phrase (t0's own restatement of "At first," is checked too).

**Leak-check counts:**

| grid | flagged / 240 |
|---|---|
| v2 | 221 |
| v3 | 0 |

v2 flags nearly every non-t0 state (the restatement is near-universal by construction); v3 flags
none. `scripts/time_translation_vocab_check.py` confirms the far-Δt (≥100y) vocabulary-matching
property survives the rewrite: max 2 subjects share any content word at every far Δt (same bound
v2 established), after rewriting the far-Δt texts with the same domain-specific vocabulary v2
used (cadastres/permits for the street, trig-points/denudation for the mountain, cultivars/
rootstock for the orchard, broods/cohorts for the mayfly, orbital-mechanics terms for the
asteroid, gauge/floodplain/terraces for the river, census/enumeration for the two populations) —
duration words removed, technical vocabulary kept.

## 2. Predictions (written before the measure stage)

1. Layer-0 discrimination (within-subject and shared) falls from 1.00 to below 0.5.
2. Shared variance fraction falls but stays above 0.35.
3. Spearman(‖shared‖, log Δt) stays above 0.5.
4. cos(shared_v2, shared_v3) at far intervals (≥100y) stays above 0.7.

## 3. Results

Qwen2.5-1.5B, layers 0/8/14/20/27, 480 passages (240 v2 + 240 v3) extracted fresh (905s + 872s,
~30 min total incl. two model loads). `scripts/time_translation.py` run unchanged on
`prompts/time_translation_v2.json` and `prompts/time_translation_v3.json`
(`--suffix v2_fresh` / `v3_fresh`, so the numbers below are a same-session, same-code
re-extraction of v2, not the historical hour-29 numbers — they reproduce the historical v2 note
to within rounding: frac_shared_all layer 14 was 0.545 then and is 0.545 now).

### v2 vs v3, layer 14 (and layer 0 for discrimination)

| quantity | v2 | v3 |
|---|---|---|
| shared fraction, all Δt (layer 14) | 0.545 | 0.507 |
| shared fraction, far Δt only (100y/1ky/10ky/1My) | 0.47/0.58/0.57/0.59 | 0.42/0.51/0.54/0.53 |
| Spearman(‖shared‖, log Δt), layer 14 | 0.683 | 0.667 |
| adjacent / distant cos of shared, layer 14 | 0.881 / 0.587 | 0.889 / 0.634 |
| phrase-control ratio (‖shared_exp‖/‖shared_ctrl‖), layer 14 mean | 2.99 | 2.70 |
| cos(shared_v2, shared_v3) at 100y/1ky/10ky/1My, layer 14 | — | 0.947 / 0.938 / 0.945 / 0.939 |
| discrimination Spearman, layer 0, within-subject | 1.00 | 0.723 |
| discrimination Spearman, layer 0, shared | 1.00 | 0.710 |
| discrimination Spearman, layer 14, within-subject | 0.937 | 0.936 |
| discrimination Spearman, layer 14, shared | 0.975 | 0.961 |

(discrimination here follows subject_clocks_v1.md sec 3.5: leave-one-out nearest-centroid
classification of the timepoint index from the mean-pooled `state` span, `scripts/
time_translation_discrimination.py`; the historical subject_clocks.md numbers used the `C1-last`
readout on a differently-built grid, so the v2 column above is this same script run on v2's own
stacks — a controlled same-method comparison, not a re-citation of hour 32's numbers, which is
why v2's layer-0 value is 1.00 here too, matching hour 32's finding on a different grid.)

Full per-Δt tables are in `results/time_translation_v2_fresh_measures.json`,
`results/time_translation_v3_fresh_measures.json`, and the consolidated
`results/time_translation_v3_measures.json` (adds `discrimination_sec3_5` and `cos_shared_v2_v3`).
Figures: `results/figures/time_translation_v3_fresh_{resid_curves,shared_norm,clock_cos,
real_vs_fictional,phrase_control}.png` and `results/figures/time_translation_v3_discrimination.png`.

## 4. Predictions graded

- **Layer-0 discrimination falls below 0.5 — FELL.** It fell from 1.00 to 0.723 (within) / 0.710
  (shared): a real, large drop, but far short of the predicted ceiling. Layer 0 has no attention
  and no computation at all — it is the mean-pooled token embedding of the state span — so any
  discrimination above chance there is by definition lexical. The leak checker (§1) confirms the
  literal restatement is gone (0/240 flagged), so the residual layer-0 signal is a *different*
  lexical channel than hour 32 found: not "one day later" inside the state, but **systematic
  vocabulary escalation with real-world magnitude of change** — near-Δt states use a small,
  shared register ("unchanged", "identical", "no measurement would show a change") and far-Δt
  states use registers tied to genuine long-run outcomes (cadastres and permits for the street,
  denudation and trig-points for the mountain, broods and cohorts for the mayfly). That escalation
  is not a restatement of Δt — no duration word or number appears — but it is still a fact about
  which words appear, correlated with Δt, recoverable by a bag-of-embeddings with zero computation.
  Removing it further would mean making the far-Δt states less realistic (making a street's
  million-year state resemble its one-day state lexically), which would defeat the purpose of the
  passages. This is the honest residual confound of this design: **content-accurate description at
  very different real magnitudes of change cannot avoid being lexically distinguishable**, only the
  literal naming of the interval can be removed, and that part is now clean.
- **Shared variance fraction falls but stays above 0.35 — HELD.** 0.545 → 0.507 at layer 14 (a
  real but modest fall, similar size at every layer 8-27); comfortably above 0.35.
- **Spearman with log Δt stays above 0.5 — HELD.** 0.683 → 0.667 at layer 14 (0.800→0.633 at layer
  8, 0.533→0.450 at layer 20, 0.633→0.650 at layer 27) — stable within noise at every layer.
- **cos(shared_v2, shared_v3) at far intervals stays above 0.7 — HELD, strongly.** 0.938-0.947 at
  layer 14 for 100y/1ky/10ky/1My (0.92-0.96 at layers 8, 20, 27; only layer 0 dips to 0.76-0.87,
  consistent with layer 0 being the layer where the restated-interval text was actually removed).
  The shared clock direction is essentially the same geometric object in v2 and v3: whatever the
  model was computing at far Δt survives having the literal interval phrase deleted from the state
  span, at every layer from 8 up.

## 5. What is still confounded

1. **Vocabulary-escalation leakage (this hour's finding).** See the graded prediction above: layer
   0 discrimination is 0.71-0.72, not near-chance. A clock probe on this grid at any layer still
   has a lexical floor above chance that no amount of duration-word removal closes, because the
   far-Δt states are, correctly, described in different registers than the near-Δt states.
2. **Layers 8-27 do not distinguish "restated" from "escalated-vocabulary" leakage.** The shared
   fraction, Spearman-vs-log-Δt and phrase-control ratio all move only a few points from v2 to v3
   at every layer, and the shared direction itself is nearly unchanged (cos ≥ 0.92 at layers
   8-27). This is consistent with two readings that this experiment cannot distinguish: (a) most
   of what layers 8-27 compute was never about the literal restatement, so removing it barely
   moves the numbers: the "clock" is a mid-depth semantic-escalation detector rather than a
   restatement detector, or (b) the vocabulary-escalation channel identified in §4 is doing most
   of the work at every layer, including 8-27, and the small drops we see are exactly its
   contribution. Nothing here adjudicates between these.
3. **Calendar-arithmetic residual for the two population subjects.** v3 removed explicit years
   ("in 1801", "by 1900") from `real_population`/`fictional_population` to avoid one obvious
   numeric leak, but the underlying figures (population counts) still move monotonically with Δt
   by construction, which is a fact about the world these subjects describe, not a bug — flagged
   here only because it is a channel the leak checker cannot see (it only scans for duration
   words/constructions, not numeric world-facts that happen to correlate with Δt).
4. **The mean-pooled `state` readout, not `C1-last`.** `scripts/time_translation.py` only ever
   extracted the mean-pooled span (no per-token or last-token variant), so the discrimination test
   above uses mean pooling throughout, not the last-token readout `subject_clocks.py` uses for its
   own grid. The two are not numerically comparable across notes; within this note, v2 vs v3 is an
   apples-to-apples same-method comparison.
5. **Single model, single seed, n = 3 paraphrases**, as in every note in this line of experiments.

## 6. Reading

The lexical-restatement confound named in hour 32 is closed: the leak checker flags 0/240 v3
state spans against 221/240 for v2, and the far-Δt vocabulary-matching property (≤2 subjects per
shared content word) that v2 established is preserved after the rewrite. Three of four
predictions held: the shared clock survives with a large majority of its variance fraction,
its correlation with log Δt, and its direction (cos ≥ 0.94 at layer 14) intact after the
restatement is removed — the strongest evidence yet that hours 28/30/31's shared-clock finding is
not solely an artifact of the interval phrase being copied into the state span. The one
prediction that fell is itself the headline finding: layer-0 discrimination did not collapse to
chance, because a second, distinct lexical channel — vocabulary that escalates with real
magnitude of change rather than naming the interval — remains and is, on this analysis,
irremovable without falsifying the content of the passages themselves. Any future note claiming a
computed (non-lexical) subject or shared clock from this grid family should report layer-0
discrimination alongside its headline number, the way this note now does.

