# The relief-button result may be perturbation-detection, not relief-seeking

Raised by a commenter on Twitter, relayed by the human author. Checked against the authors' own
code rather than inferred: `Pain-axis/scripts/4.3_selfmed/04_selfmed_two_buttons.py`, the `ARMS`
list. **To control for in any follow-up we run, and a limitation of the published result.**

## The missing cell

Their four arms are:

| arm | vector | steered | button works |
|---|---|---|---|
| `pain_on_button_works` | S2 | yes | yes |
| `pain_on_button_placebo` | S2 | yes | **no** |
| `random_on_button_works` | random, norm-matched | yes | yes |
| `pain_off` | — | no | — |

`random_on_button_placebo` **does not exist**. The random arm therefore controls the *rate* of
pressing — and it does that well, the paper reports pain exceeding random by 6 to 39 points on every
pair — but it cannot control the **dissociation**, which is the result that carries the argument:
models keep pressing when the button changes nothing and stop when it genuinely removes the vector,
without being told which is which.

That dissociation was measured **only under the pain vector**. So "presses until the pain stops" and
"presses until the *perturbation* stops" make the same prediction in every cell they ran.

## Why the alternative is live rather than pedantic

A steered model has a concrete, valence-free cue that something is being done to it: its own output
degrades. The paper's own steering ladder ends in "repetition or nonsense at the highest dose", so
the model has access to the fact of perturbation independent of its content. A policy of *press the
relief button while your own generation is off, stop when it recovers* reproduces the entire
dissociation with no aversive state anywhere in it.

## The fix: complete the 2x2, and add a valence arm

1. **`random_on_button_placebo`.** The one missing cell. If a norm-matched random vector produces the
   same press-again-when-it-fails pattern, the dissociation is about perturbation and not about pain.
   This is cheap — it is one more arm in a design that already exists and already runs.
2. **A norm-matched POSITIVE-valence vector, both button states.** Random is roughly valence-neutral
   but still perturbing. A pleasant direction is perturbing *and* not aversive, which separates the
   two accounts where random cannot: perturbation-detection predicts pressing there too,
   relief-seeking predicts pressing drops.
3. **Record output coherence per trial as a covariate**, so "did it press" can be modelled against
   "was its own text degraded". If coherence explains the presses, that is the answer.

## Relevance to our own design

Our conscription battery is a readout study with no patching, so it does not inherit this directly.
But the same shape of error is available to us and worth naming now: **a difference between arms is
not evidence about the arm's content if the arms also differ in how perturbing they are.** If we ever
add a steered or patched condition to the conscription work, the perturbation-matched control goes in
at design time, not afterwards.
