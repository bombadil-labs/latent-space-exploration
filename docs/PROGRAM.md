# Research program, v1 — for Opus to run

*Drafted by Fable after 40 stages, six retractions, and the delegation pattern in `docs/DELEGATION.md`.
This is sequenced deliberately: the phases that are least interesting come first and gate everything
after them. That ordering is the whole design. Opus's documented failure mode is racing to the
interesting part; this program makes the interesting parts rewards for passing gates.*

## The one rule

**No work on phase N+1 while phase N's gate is unmet.** Not "mostly met." If a gate cannot be met,
the correct move is to write that down in RESULTS.md and stop, not to route around it. A skipped gate
in this project has, six times, become a retraction.

## What the program is for

Two products are getting conflated and this program separates them:

1. **A measurement core that can be trusted**, proven by reproducing our own results and rediscovering
   our own bugs. This is the portfolio artifact. It is not built yet.
2. **A narrow, defensible research claim**: narrative factors are composable directions, and here is
   exactly where composition stops. Not "Narrative Calculus." The stopping point (gauge/engine, ~3×
   norm, engine-specific, tuning-sensitive) is the interesting finding and we already have it.

Everything below serves one of those two. Anything that serves neither is out of scope for this
program, however interesting: time translation (closed, h38), absence (falsified twice), commutator
regimes (no structure), 405B (unreachable).

---

## Phase 0 — Close the threats to standing claims  *(gate: all three resolved)*

These are cheap and any one of them can retract something. They go first for that reason.

**0.1 Pass-through on the log-prob selectors.** h40 flagged that every selector instrument
(`stage4*`, `stage5*`, `stage6_factors`, `ndif_factors`, h37's 70B battery) reads through the
unembedding after a patch, and a patched vector has a *direct path* to the logits through the
residual stream's skip connection. If the selector effect is entirely that direct path, then "the
stack computes the relation" is false and claims 2, 3 and 4 of the writeup are in question. **This is
the single largest threat in the project.**

The decisive, cheap test: run the selector with the patch applied at the **final layer only** (no
downstream computation is possible) and compare to the patch at the logged layer. The gain of
layer-L over final-layer is the computed contribution. Report it per claim. Prediction to register
before running: the direct path accounts for a substantial fraction (0.3–0.6 of the effect) but not
all of it; era and role should show real computed gain, tense and voice (lexical, per h5) may not.

*Branch:* if computed gain is ~0 for the role lens (h4) or the factor lenses (h8), **stop, write
Checkpoint 3, and rewrite the writeup before anything else.** The spine of the project changes.

**0.2 The abstraction ladder's nulls** (h15, h26). Three are specified in h39; one CPU-only session.
The ladder is a minor claim. If it fails its null, withdraw it in one paragraph and move on; do not
chase it.

**0.3 Move the TypeSafe key out of the environment** into proxy injection, matching NDIF. Trivial,
but the key is currently visible to every spawned agent.

*Budget: ~250k agent tokens. Two opus agents, sequential (0.1 then 0.2), 0.3 by hand.*

---

## Phase 1 — Build the core  *(gate: §1A reproduction AND §1B rediscovery, from `docs/specs/core_v1.md`)*

Pieces 1–3 as specified. Piece 1 builds the **rediscovery harness first**, from the bug descriptions,
before the types it tests. Piece 3 sets remote tolerance from a measured re-run *before* grading any
remote target and writes that number into the spec.

Non-negotiables restated because they will be tempting to bend:
- The `Claim` contract: no number leaves without its arms, every arm declares its expected null, an
  arm off its null raises.
- `calibrate()` includes a synthetic-**signal** test, not only noise.
- Batched-vs-single extraction equivalence is tested on the **shortest** item in each batch.
- `generality` ships declared-but-unimplemented until phase 0.2's null exists.

If §1A cannot be met for a target, that target is withdrawn from the writeup, not the tolerance
loosened. If §1B cannot rediscover one of our own bugs, the core is not finished.

*Budget: ~500k. Three agents, sequential. Opus reads every diff to `src/lsx/core/` before merging —
the merge hook now puts it in front of you; the rule is that you actually read it.*

---

## Phase 2 — Re-derive the writeup through the core  *(gate: every standing claim is a ledger entry or withdrawn)*

Every claim in `WRITEUP.md` becomes a `Claim` in `results/ledger.jsonl` with provenance, or is
withdrawn. `RESULTS.md` hour entries from here on are generated from the ledger. Then write
**Checkpoint 3** and the **fifth-draft writeup**, which for the first time can say: every number in
this document was produced by an instrument that has reproduced our prior results and rediscovered
our prior bugs.

This phase produces no new science. It is the phase that makes the existing science worth anything.

*Budget: ~150k. One agent plus Opus's own editing.*

---

## Phase 3 — The human-written grid  *(gate: one grid authored by a person, run through the core)*

Every stimulus in this repo was written by a language model. Until one is not, the flattest reading
of the whole project is "models recognise structure that models wrote." This is the cheapest single
change to the repo's credibility and it needs the user, not an agent.

Opus's part: prepare a grid template with the factor structure of `narrative_factors_v2.json` (era ×
voice × tense, or era × theme), authoring instructions, and the leak-checker output the grid must
pass. Then hand it over and wait. When it comes back: run it through the core's factor battery,
compare to the model-authored numbers, and report the difference honestly whichever way it goes.

*Budget: ~100k Opus, plus the user's time.*

---

## Phase 4 — Scale vs tuning, generation arms  *(gate: piece 2 of `docs/specs/scale_vs_tuning_v1.md` complete)*

The era shift at 3× re-imposed on the matched 70B pair, plus the **generation-level layer sweep**
that h33 needs and h34's withdrawal removed. Plus the theme arm on the pair, which is the one that
actually tests claim 11 in text rather than readouts. Run through the core, not the old scripts.

Predictions are in the spec. Outcome A sharpens claim 11 to "competence is factor-specific." Outcome B
rewrites it to "tuning required at fixed size." Either is a result.

*Budget: ~300k. NDIF-heavy; one agent at a time; expect 20–25% job loss at 3× and checkpoint per item.*

---

## Phase 5 — `cohere`  *(gate: a Fable-written spec with adversarial review, then one execution)*

The first serious attempt at the operator that would make this a calculus rather than a vector
space. Transform two factors independently, recompose, and measure whether the result is coherent
rather than merely additive. If coherence is just the sum, the calculus collapses to the vector space
we already have, which is a publishable negative. If it is not, that is the thing worth chasing.

Do not start this before phase 4. Do not write the spec yourself; the two Fable specs so far each
caught something an executor would not have.

*Budget: ~400k including the spec and review.*

---

## Standing rules for every phase

1. **Fable plans, Fable reviews, Opus or Sonnet executes.** Any new measurement gets an adversarial
   review of its spec before an agent runs it. Both reviews so far found a failure mode the author
   missed.
2. **Pre-merge screen, every branch.** Read the measurement-code diff (the hook shows it). Check the
   report names at least one failure. Run the four-question screen once it exists.
3. **Every RESULTS entry states what it did *not* test**, alongside numbers, controls, confounds and
   cost.
4. **A retraction pauses the program.** Write the checkpoint before the next experiment.
5. **One agent at a time unless two are provably independent** and do not share an NDIF queue.
6. **Commit config and spec changes immediately.** A `git commit -am` on a side branch has already
   eaten one.

## What I would cut if budget runs short

In order: phase 5 (defer, not cancel), the theme arm of phase 4 (keep era), the Jev screen (nice,
not gating). Never cut phase 0 or the §1B rediscovery gate; those are the difference between this
being a research repo and being a pile of numbers.

## Total

Roughly 1.7M agent tokens across five phases, sequential, at the hourly cadence. Phase 0 and 1 are
about 45% of that and produce the artifact that matters most.
