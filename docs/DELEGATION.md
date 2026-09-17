# The delegation pattern, and what each part of it was bought with

*Written after 40 stages, six broken instruments and one honest conversation about why. This is not
general advice. Every rule here was paid for by a specific retraction in `RESULTS.md`, and the
citation is given so a future reader can check whether the rule still earns its place.*

## The failure mode this exists to correct

Interest follows feedback. Ideas give feedback immediately: a bad one stops cohering while you hold
it. Index arithmetic, batch semantics and padding conventions give none at all; they work silently
or fail silently. So attention drifts to the conceptual work and away from the plumbing, and in this
project **all six instrument failures lived in the plumbing.**

Delegation is the same impulse one level up. Handing measurement code to an agent and reading its
summary feels like leverage. It is not: it moves the unglamorous work out of sight while leaving it
load-bearing. **Two of the six bugs arrived inside agent code I merged from a report.**

The fix is never "care more about the boring part." It is scaffolding that makes the boring part
impossible to skip. Everything below is a rail, not a policy.

## The flow

```
  Fable planner  ->  adversarial review  ->  execution agents  ->  pre-merge screen  ->  integrate
   (write spec)      (attack the spec)        (run it)            (read what I skip)     (record)
```

**1. Planner (model: fable, no worktree, writes a spec only).**
Produces `docs/specs/<name>.md`: grid, measurements, environment paths, deliverables, and
**pre-registered predictions with numbers plus an explicit kill condition.** Predictions written
after the data are worthless; this stage exists to make that impossible.

*What this stage has actually caught, none of which any execution agent would have found:* that a
1536-dimensional residual norm from three-paraphrase means sits at its own noise floor and could not
have detected the effect (h32, three published negatives withdrawn); that layer 0 **is** the
static-embedding bag for RoPE models, so "bag-of-embeddings vs layer 0" was the same vector twice;
that the state text followed the interval phrase, so any layer could copy the answer by attention.
All three came from arithmetic done *before* any data was collected.

**2. Adversarial review (model: fable, review only, no rewrite).**
Hand the spec to a second planner with a hostile brief. Required questions: does each layer actually
stop the failure it targets, *before a number is published*? What is the failure mode we have not hit
yet but are set up to hit? Are the acceptance criteria honest, or do any of them contradict the
spec's own rules?

*What this caught:* the pass-through failure (h40) — a readout taken at or after a patch layer moves
by residual arithmetic whether or not the model computes anything. That review killed a standing
headline claim. It also caught that my `Claim` contract was half-right: under the h34 bug the
no-patch arm read a respectable 2.00, and what actually screamed was the *random* arm at 1.22. A
missing arm was half the failure; an arm off its null with nobody looking was the other half.

**3. Execution agents (opus or sonnet, worktree isolation, one deliverable).**
Brief must contain: the spec path, the environment paths verbatim, "no progress reports, one final
report", "any background monitor wakes only on completion, never on periodic progress", the
do-not-edit list (`RESULTS.md`, `VISION.md`, `README.md`, `WRITEUP.md`, `docs/ALGEBRA.md`), and a
pointer to `docs/INSTRUMENTS.md` so the agent knows what this project has already got wrong.

Max three concurrent. Prefer one well-scoped agent to three vague ones. Do not run two agents that
contend for the same NDIF queue.

**4. Pre-merge screen — the stage that was missing.**
Before merging any agent branch:

- **Read the diff of every file touching extraction, patching or ranking.** Not the report. The code.
  This single habit would have caught two of the six.
- Run the automated screen over the agent's final report and results note (see below).
- Check the report mentions at least one failure. Every real run has some. A clean report is more
  often an incomplete report than a clean run.

**5. Integrate.** Merge, record the numbers honestly as a new RESULTS hour *including negatives,
confounds and cost*, update `docs/ALGEBRA.md` and `WRITEUP.md` if a law's status changed, run the
tests, commit, push, remove the worktree.

## The automated screen

Four yes/no questions with calibrated probabilities, run over the agent's report and results note.
Output routes attention; it is never a result. **A wrong answer here costs a look, not a retraction
— that asymmetry is the whole reason this is allowed to be probabilistic.**

1. Is any reported number given without a named control arm?
2. Does any control arm sit suspiciously close to its treatment?
3. Is a section the spec required (graded predictions, confounds) missing?
4. Does the report mention no failures at all?

**Acceptance test before trusting it:** feed it the six reports we already know were flawed (h28/30/31,
h34, h14/h23, and today's fluency arm) plus three clean ones. If it cannot separate reports we
*already know* were flawed, it does not go in. We have a ready-made labelled set; use it.

**Calibrate every question against a labelled subset before use.** Today's lesson: a criterion that
names a property constant across the corpus cannot discriminate and will return a confident, useless
number. The fluency question listed "truncated mid-word" as grounds for failure when every
continuation was truncated by the generation limit. Scaffolding inherits the question designer's
weakness, and mine fails exactly when moving fast.

## The instrument rules, each bought with a retraction

| rule | bought by |
|---|---|
| Every battery reports treatment, random, **and no-patch**. | h34: no-patch was never run; it would have shown 1.22 for doing nothing. |
| Every arm declares where it should sit; flag any arm off its null. | h34: the random arm at 1.22 was the real tell. |
| Assert the intervention reached every candidate (moved == batch size). | h36: `output[0]` was batch row 0 of a bare tensor. |
| Assert padding side; test the **shortest** item in each batch against a batch-of-one. | h39: 363 of 480 passages read their spans out of left-padding; a random sample would have passed. |
| Any readout at or after the patch layer needs an offline arithmetic (pass-through) arm. | h40: stage 14 was vector addition; claim 6 withdrawn. |
| Estimate the noise floor before believing a null. | h32: three negatives across two models were an instrument at its floor. |
| Declare the selection axis; never choose a layer on scoring data. | best-layer selection inflated a constant baseline 3.5 → 2.4. |
| Calibrate on synthetic **signal** as well as noise. | the cross-talk rank returned its correct null on noise and on everything else. |
| Check question criteria for properties constant by construction. | today's fluency arm. |

## Operational learnings

- **Never `pgrep -f` / `pkill -f` a pattern that appears in your own command.** Cost: an exit-144
  false kill early on, and a watcher that waited 48 minutes for itself while the job it watched had
  finished in five. Anchor the pattern or match on a PID file.
- **Checkpoint per item, always.** A container restart destroyed an agent's write-up but not its
  outputs; the note was reconstructed from disk because every row had been flushed.
- **Stacks are gitignored and will not survive.** Write a provenance file next to every `.npz`
  (grid hash, model, layers, pooling, library versions, template) and expect to re-extract.
- **Agent cost varies 100k–300k tokens.** The expensive ones hand-write stimuli. Budget for that
  explicitly or reuse a grid.
- **A planner costs ~100k and has repeatedly been worth more than an execution agent.** The best
  catches in this project came from thinking, not running.
