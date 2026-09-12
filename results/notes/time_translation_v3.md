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

(filled in after the run)
