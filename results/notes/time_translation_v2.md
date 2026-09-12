# Time translation v2: vocabulary-matched far-interval grid

Companion to `time_translation.md` (v1). Same eight subjects, intervals, paraphrases, phrase-only
controls, model (Qwen2.5-1.5B), layers, scripts, seed. Only the state spans for Δt ≥ 100 years
were rewritten (`scripts/_build_v2_states.py`) so that no content word appears in the far-Δt
state of more than two subjects. `scripts/time_translation_vocab_check.py` confirms: in v1 the
far-Δt states shared words such as *sea, ice, gone, sediment, species, geological* across up to
five subjects; in v2 the maximum is two at every Δt ≥ 100 y, with no violations.

**Prediction (written before the run).** Shared variance fraction drops from 0.53 to below 0.4;
selector rank at 1 My worsens from 1.62 to above 3.0 but stays below random; near-Δt numbers
unchanged.

## v1 vs v2, layer 14

| quantity | v1 | v2 |
|---|---|---|
| share of Σ‖d‖² explained by shared(Δt), all Δt | 0.533 | 0.545 |
| same, far Δt only (1 ky / 10 ky / 1 My) | 0.52 / 0.55 / 0.57 | 0.58 / 0.57 / 0.59 |
| Spearman(‖shared‖, log Δt) | 0.47 | 0.68 |
| adjacent / distant cos of shared | 0.89 / 0.64 | 0.88 / 0.59 |
| ‖shared_exp‖ / ‖shared_ctrl‖ at 1 ky / 10 ky / 1 My | 2.57 / 2.61 / 2.93 | 3.19 / 3.08 / 3.12 |
| cos(shared_v1, shared_v2) at 100 y / 1 ky / 10 ky / 1 My | | 0.90 / 0.86 / 0.88 / 0.89 |
| τ(s), half-max | 1 day, all subjects | 1 day, all subjects |
| orchard ‖d(1 y)‖/‖d(6 mo)‖, cos | 0.86, 0.56 | unchanged (near Δt untouched) |
| real vs fictional residual cos, 100 y → 1 My | 0.42–0.49 | 0.49–0.59 |
| selector, gain rank of 9, Δt ≥ 1 y: clock / random | 3.88 / 5.65 | 3.62 / 4.83 |
| per Δt, clock / random: 1 y | 4.38 / 3.75 | 4.75 / 4.00 |
| 10 y | 5.38 / 6.25 | 5.38 / 5.62 |
| 100 y | 5.50 / 5.75 | 4.25 / 6.38 |
| 1 ky | 3.12 / 6.00 | 1.38 / 2.88 |
| 10 ky | 3.25 / 7.00 | 2.75 / 5.75 |
| 1 My | 1.62 / 5.12 | 3.25 / 4.38 |
| raw rank, Δt ≥ 1 y: clock / none / random | 4.65 / 4.98 / 5.06 | 5.21 / 5.35 / 5.38 |

Near-Δt numbers are identical by construction (same passages, same stacks).

## Grades

- Shared fraction below 0.4: **fell**. It rose slightly (0.53 → 0.55), and at the rewritten
  intervals it rose the most. Removing the shared vocabulary did not remove the shared direction.
- 1 My selector worse than 3.0 but better than random: **held** (3.25 vs 4.38). But the effect
  moved rather than shrank: 1 ky is now 1.38 and 100 y 4.25 vs 6.38, and the overall gain rank
  improved (3.62 vs 4.83 random).
- Near-Δt unchanged: **held** trivially.

## Reading

The shared clock is not the erasure vocabulary. With no content word shared by more than two
subjects at far Δt, the shared direction explains the same variance, points the same way as
before (cos 0.86–0.90 to the v1 direction), tracks log Δt more monotonically (Spearman 0.68), and
is three times the phrase-only displacement. The confound named in hour 28 is closed. What v2
does not change: subject-relative timescales are still absent (τ degenerate, residual curves
flat), and the random-direction control is still better than "no patch" is worse, so part of the
selector gap is avoided disruption rather than selection. Single model, single author.
