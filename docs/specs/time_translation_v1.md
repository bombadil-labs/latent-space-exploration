# Spec: parameterized time translation with subject-relative clocks (v1)

*Written before running. Predictions are recorded here so they can be graded, not adjusted.*

## Question

`transform` in the algebra has so far been a fixed direction (era shift, e1→e2). This spec asks
whether a **parameterized** translation exists: an operator T(Δt) that advances a described
subject by a known interval, whose latent displacement is a function of Δt, and whether that
displacement decomposes into a **shared clock** (the same for every subject) plus a
**subject-specific** part whose timescale τ is the subject's own (a mayfly's day, a mountain's
million years).

## Grid (`prompts/time_translation_v1.json`)

- **Subjects (8):** `street` (a named city street), `mountain`, `orchard`, `mayfly`, `asteroid`,
  `river`, `real_population` (London, starting 1800, with true approximate population and true
  events), `fictional_population` (an invented city, "Veyle", starting at a stated population in a
  stated year of an invented calendar, with invented events of comparable scale).
- **Timepoints (10):** t0, then Δt ∈ {1 day, 1 week, 1 month, 6 months, 1 year, 10 years,
  100 years, 1,000 years, 10,000 years, 1,000,000 years}. (Nine Δt plus t0.)
- **Paraphrases (3)** per subject × timepoint, written by the agent, 2–3 sentences, 35–60 words.
- **Passage form:** `[[interval: <phrase>]] [[state: <description>]]`. The interval span is
  exactly the phrase ("At first," for t0; "One week later," "A million years later," etc.). The
  state span describes the subject's condition at that time, written to be true to the subject's
  actual timescale: the mayfly is dead by day 2 and gone by the week; the orchard at 6 months is
  in the opposite season and at 1 year is back; the mountain is unchanged until 10,000 years and
  eroded at a million; the asteroid is unchanged at every Δt except position; the river meanders
  at 100–1,000 years; the street turns over on 10-year scale; the populations grow, and the real
  one crosses known events (1800→1900 industrial growth, 20th-century plateau) while the fictional
  one has invented events of similar magnitude and timing.
- **Phrase-only control:** the same 10 interval phrases attached to the *t0* state description of
  every subject (30 extra passages per subject, or reuse: 8 × 10 × 3 = 240). This gives the
  displacement due to the interval phrase alone.

Total: 240 experimental + 240 control passages.

## Model and layers

Primary: Qwen2.5-1.5B locally (CPU, `.venv`, `HF_HOME=$PWD/cache/hf`, `HF_HUB_DISABLE_XET=1`);
layers 0, 8, 14, 20, 27. Secondary if time allows: Gemma-2-9B-it through NDIF
(`scripts/ndif_extract.py` pattern, checkpoint per text, `retry_job`), layers 9, 20, 31.

## Measurements

Pool the **state** span (mean) at each layer: h(s, Δt, p).

1. **Displacement** d(s, Δt) = mean_p h(s, Δt, p) − mean_p h(s, t0, p).
2. **Decomposition.** shared(Δt) = mean_s d(s, Δt); resid(s, Δt) = d(s, Δt) − shared(Δt).
   Report ‖shared(Δt)‖, ‖resid(s, Δt)‖, and the fraction of ‖d‖² explained by shared, per layer.
   Compute shared leave-one-subject-out when it is used as a direction.
3. **Clock geometry.** Cosine matrix between shared(Δt_i) and shared(Δt_j). Correlation of
   ‖shared(Δt)‖ with log Δt.
4. **Subject timescale τ(s).** The smallest Δt at which ‖resid(s, Δt)‖ reaches half its maximum
   over Δt. Also the full curve, plotted per subject (matplotlib, `results/figures/`).
5. **Cyclic return (orchard).** Compare ‖d(orchard, 6 months)‖ and ‖d(orchard, 1 year)‖, and
   cos(d(6mo), d(1y)); same numbers for the mountain as a non-cyclic control.
6. **Real vs fictional population.** Overlay the two residual curves; cosine between
   resid(real, Δt) and resid(fictional, Δt) at each Δt; identify the Δt where they separate most.
7. **Phrase-only control.** Same decomposition on the control passages. The interesting quantity
   is ‖shared_exp(Δt)‖ / ‖shared_ctrl(Δt)‖ per layer, and cos(shared_exp, shared_ctrl).
8. **Selector test (one paraphrase, layer 14, and layer 8 as a check).** For each subject and
   each Δt ≥ 1 year: prefix = the t0 passage plus the interval phrase for Δt; candidates = the
   nine state spans of that subject at the nine Δt; patch = leave-one-subject-out shared(Δt)
   added at every position (`LM.logprob` with `Patch`). Rank of the matching state among nine;
   chance 5.0; control = random direction of equal norm (seed fixed). Report the mean rank with
   and without the patch and under the control. If the full run exceeds ~90 minutes on CPU,
   restrict to Δt ∈ {1 year, 100 years, 1,000,000 years} and say so.

## Predictions (graded after the run)

- **P1 (shared clock exists):** shared(Δt) explains ≥ 40% of ‖d‖² at layer 14; ‖shared‖ is
  monotone in log Δt (Spearman ≥ 0.8); adjacent-Δt cosines exceed distant ones.
- **P2 (subject-relative knees):** τ ordering mayfly < street ≈ real_pop ≈ fictional_pop <
  river < mountain ≤ asteroid. At least 5 of the 7 pairwise orderings implied hold.
- **P3 (cyclic return):** for the orchard, ‖d(1y)‖ < ‖d(6mo)‖ and cos(d(6mo), d(1y)) < 0.5; for
  the mountain, ‖d(1y)‖ ≈ ‖d(6mo)‖ and cos > 0.8.
- **P4 (history):** real and fictional population residuals are similar (cos > 0.6) through
  10 years and separate most at 100 years. Confidence 55%.
- **P5 (clock as selector):** patched rank beats random at layer 14 for Δt ≥ 1 year (mean rank
  ≤ 4.0 vs ≈ 5.0 random); no effect at layer 0 (not tested) and small at layer 8.
- **P6 (more than the phrase):** ‖shared_exp‖ / ‖shared_ctrl‖ ≥ 1.5 at layer 14 and ≈ 1.0 at
  layer 0; cos(shared_exp, shared_ctrl) declines with depth.

## Deliverables (agent branch only)

`prompts/time_translation_v1.json`, `scripts/time_translation.py` (extraction + measurements
1–7), `scripts/time_translation_selector.py` (measurement 8), `results/time_translation_*.json`,
`results/figures/time_translation_*.png`, `results/notes/time_translation.md` with every number,
each prediction graded, and a "what fell / what is confounded" section. Do not edit RESULTS.md,
VISION.md, README.md, WRITEUP.md, or docs/ALGEBRA.md.
