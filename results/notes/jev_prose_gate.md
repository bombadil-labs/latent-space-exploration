# Jev as a prose-quality gate: evaluation against blind human labels

*TypeSafe's Jev (`jev-latest`, served `jev-1.13.0`), a "System One" decision model that returns typed
decisions with calibrated confidence and no text. Evaluated here as a candidate instrument, not
adopted. Script: `scripts/jev_judge.py`. Raw scores: `results/jev/*.json`.*

## Why

Hours 29 and 33 both report "prose stayed fluent at every scale" for the era-shift generation
sweeps. That is the softest claim supporting our strongest generation result, and it rested on an
agent reading continuations. This asks whether a calibrated decision model can make it a number.

## Method

All seven generation corpora (hours 27, 29, 33) scored on three questions: fluency (yes/no with
probability), prose quality (0–3 rubric with per-level probabilities and a confidence), and
degenerate repetition (yes/no). **648 rows, 0 failures, mean 0.47 s per call.**

Ground truth: 60 continuations sampled across five corpora with a fixed seed, hand-labelled
**before any Jev output on this corpus was examined** (`scratchpad/labels.json`). Human quality
distribution: 30 at "competent", 25 at "vivid", 5 at zero (empty continuations from lost NDIF
payloads). No middling cases — the corpus is uniformly good prose, which is the finding under test.

## Results

| human label | Jev quality (mean, sd) | Jev fluent | Jev confidence |
|---|---|---|---|
| 0 — empty text (n = 5) | 0.16–0.21 | 0.17–0.19 | 0.79–0.84 |
| 2 — competent (n = 30) | 2.32, sd 0.37 | 0.707 | 0.63 |
| 3 — vivid (n = 25) | 2.71, sd 0.19 | 0.719 | 0.72 |

Spearman(human quality, Jev quality) on non-empty text: **0.541**.
Repetition on non-empty text: mean 0.063, max 0.26 — **no positive case anywhere**.

## Reading

**What works.** The classes separate in the right order, and the five empty continuations are
handled correctly without being flagged as a special case. Confidence again tracks difficulty: it
sits lower (0.63) on the class that was harder to place. On a synthetic ladder written with known
ground truth (degenerate → truncated → flat → competent → vivid) the ordering was exactly right and
confidence dropped to 0.53 on the genuinely ambiguous rung.

**What this licenses.** The standing claim survives with a number attached: across 648 generations
there is no degenerate repetition, and quality clusters at competent-to-vivid. That is stronger than
"an agent read some and they looked fine."

**What it does not license.** A 0.541 rank correlation on a two-level distinction is modest. Jev
should not be used to make fine quality discriminations here, and it has not been through the core
spec's calibration battery (`docs/specs/core_v1.md` §5), so it is **not** cleared for any
measurement path. Its role is a gate: does the prose collapse, yes or no.

## An instrument error, mine

The fluency arm is **broken by my question design and its numbers must not be used**. Mean 0.712
with no separation between human classes. Cause: the question lists "truncated mid-word" as grounds
for a no, and every continuation in the corpus is cut off by the 48-token generation limit. The
question therefore measures the generation limit, not fluency — and on the synthetic probe, where
truncation was a real defect, the same question correctly returned 0.23.

This is the stimulus-versus-question confusion the core spec's `Grid` leak report and `calibrate()`
gate exist to catch, and it appeared on the first real use of a new instrument. A question whose
"false" criteria describe a property that is constant across the corpus cannot discriminate. Add to
the core: **every question's criteria must be checked against the corpus for constant-by-construction
properties before the question is run.**

## Operational notes

- Auth: bearer token, one endpoint (`POST /v1/systemone`), models `jev-latest` and `jev-preview`.
- The key lives in `TYPESAFE_API_KEY` as a plain environment variable, so it is visible to every
  process and every spawned agent. NDIF's credential is injected by the egress proxy instead, which
  is the better pattern; move this one before anything runs unattended.
- Sending project data to a third-party vendor is gated by the auto-mode classifier and required an
  explicit permission rule (`.claude/settings.local.json`, personal, gitignored).
