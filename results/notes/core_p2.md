# `lsx.core` piece 2: the instrument registry and the calibration battery

Spec: `docs/specs/core_v1.md` v1.3, §4 (the real arm registry), §5 (calibration), §8 (instrument
registry), §11 rescoped to **three instruments**: `selector`, `composition`, `readout_shift`.
`crosstalk`, `depth_gain` and `generality` are declared and unbuilt.

Files added: `src/lsx/core/{planted,registry,instruments}.py`, `tests/test_core_registry.py`,
`results/calibration/*.json` (the cached reports). Modified: `checks.py` (battery extended, not
replaced), `types.py`, `rediscovery.py` (bug 8 strengthened), `__init__.py`.
`pytest -q tests/` — **82 passed, 1.5 s**. Harness: **9/9**. `scripts/` untouched.

`planted.py` was committed **before** the statistics it tests (`1ad3c8c` then `aecc36a`), on piece
1's instruction. It paid the way it was supposed to: the generator for `readout_shift` is what
showed the first version of that statistic was biased on its own null (below).

## The three instruments

| instrument | null | how the null is fixed | claimed invariances | required arms |
|---|---|---|---|---|
| `selector` | `(k+1)/2`, the candidate midpoint (3.50 at k=6, 2.00 at k=3) | from `config["n_candidates"]`, never from the caller's `expected_null` | scale, rotation | random, no_patch, permutation |
| `composition` | `(V+1)/2` over the **joint** variant set (9.50 at h8's V=18) | from `config["n_variants"]` | scale, rotation | random, no_patch |
| `readout_shift` | **0.0 gain** over the pass-through — not chance | constant; the quantity is a difference | scale, rotation | random, no_patch, **passthrough (computed)** |

`selector` and `composition` share a mid-rank core deliberately (mid-rank on ties is the h34
guard). What separates them is the null — a per-factor 2.0 against a joint 9.50 — and the arms.

### Tolerances: measured, per instrument, per arm size

Piece 1's flat 0.15 is gone. `registry.InstrumentSpec.arm_tolerance(n, config)` returns
`3 * sd_item / sqrt(n)`, where `sd_item` is the per-item spread of that statistic's own null:
`sqrt((k²−1)/12)` for a rank on k candidates, `sqrt(2/d)` for `readout_shift`. The closed forms are
**checked against a Monte-Carlo draw from the real statistic** in
`test_declared_null_item_sd_matches_the_statistic` (selector k=3 measured 0.821 vs declared 0.816;
k=6, 1.712 vs 1.708; composition V=18, 5.252 vs 5.188 -- all within 1 %).

| shape | measured tolerance | piece 1's 0.15 |
|---|---|---|
| selector, k=6, n=40 (h4, h16) | **0.810** | 5× too tight — fires on ~50% of clean arms (measured, `test_a_clean_arm_at_h4_size_would_have_fired_piece_1s_flat_tolerance`: 0.553 of clean arms fire, against 0.001 for the measured band) |
| selector, k=3, n=30 (h34) | **0.447** | too tight; h34's tells (1.22, 1.00) still fire at 0.78 and 1.00 off |
| selector, k=3, n=400 | **0.122** | too loose — 0.14 of real drift passes in silence |
| composition, V=18, n=40 | **2.461** | 16× too tight |
| readout_shift, d=1536, n=40 | **0.084** | roughly right by accident |

`readout_shift`'s is the one I am least happy with and it is declared as such in the registry: the
instrument's own null spread is **0 by construction**, so I use `sqrt(2/d)` — two independent
unit-scaled readouts — as a conservative stand-in. The spread §2a actually asks for is the
**paraphrase-noise interval on the difference**, which needs real paraphrases. That is piece 3's
measurement; `config={"readout_sd": ...}` is where it lands.

## Calibration: every result

Six tests, not five: noise, sensitivity, **degeneracy**, self-floor, invariance, known-zero.
Cached under `results/calibration/<instrument>.<key>.json`, key = hash(statistic source, declared
null, declared invariances). `Claim` refuses a stale measured report (`CalibrationStale`) and a
missing one (`MissingCalibration`). Total run time for all three: **0.2 s**.

```
selector      [6568ca646faa7ed7] 6 tests PASS  noise 3.5362±0.0957 vs 3.5 | sensitivity
              [3.018, 2.498, 1.655, 1.060] over planted 0.05/0.1/0.2/0.4 | self-floor 3.3850
              | scale Δ=0, rotation Δ=0 | known-zero 3.5 exactly
composition   [6efcd3ebf6b96542] 6 tests PASS  noise 9.4787±0.3244 vs 9.5 | sensitivity
              [7.403, 6.090, 3.975, 1.810] | self-floor 9.8225 | scale Δ=0, rotation Δ=0
              | known-zero 9.5 exactly
readout_shift [96db0d19006b9011] 6 tests PASS  noise −0.0000±0.0000 vs 0.0 | sensitivity
              [0.2519, 0.5039, 1.0077, 2.0154] over planted 0.25/0.5/1.0/2.0 | self-floor 0.0000
              | scale Δ=0, rotation Δ=0 | known-zero 0 exactly
```

The known-zero points are analytic and exact: every candidate direction identical (the field is
wholly tied, so mid-rank must return the midpoint — this is the h34 check from the other side), and
zero shift with no block contribution.

`readout_shift`'s sensitivity values **are** the planted amplitudes to under 1 % — it is calibrated
in the literal sense, and a reported gain of 0.2 means two tenths of a residual norm along the
readout direction.

### Two calibration failures, and what they changed

**1. The failure piece 1 predicted was the sensitivity test. It was not.** Both rank instruments
failed **self-floor** on the first run: selector 3.3850 against a null of 3.5, composition 9.8225
against 9.5. Neither is a real miss — they are 1.3 and 0.9 standard errors out — they failed
because `run_calibration`'s `null_tol` was a flat **0.1**, which is piece 1's unmeasured-tolerance
mistake reappearing one level up in the same file that was supposed to fix it. `null_tol` is now
the same measured 3σ band as an arm's, at the calibration fixture's own n (selector 0.256,
composition 0.778, readout_shift 0.027). *A flat tolerance anywhere in this core is a bug.*

**2. `readout_shift` failed its noise test at −0.129 ± 0.002, and the instrument changed.** The
first version scored the readout as a **cosine** against the target direction. Its null fixture has
the blocks doing a full residual-norm of work *orthogonal* to the readout direction — the model is
busy and the readout must still register nothing — and the cosine version read −0.129: a cosine is
the *fraction* of the vector lying along the direction, so orthogonal work dilutes it and a merely
busy model registers as negative gain. The shipped statistic is the **projection** on the unit
direction divided by the item's own base norm, which cancels everything the pass-through already
carries and reads exactly 0 however much the blocks do elsewhere. The rejected version is kept as
`instruments.cosine_readout_gain` and a test asserts the battery still refuses it.

**This is a finding for piece 3's graders, not a detail.** Any *relative* readout — a rank among
candidate directions, a margin between two of them, which is what h14's "nearest era direction"
actually was — is unbiased in the same way. A bare scalar cosine difference is the shape to avoid.
Any §1A target whose readout is a scalar cosine should be re-derived with a projection or a margin
before its number is compared to the logged one.

## The pass-through arm is now computed (piece 1's third thin catch)

`instruments.PassthroughArm.compute(base=, shift=, readout=, ...)` builds
`readout(base_resid_at_read_layer + shift)` offline, no forward pass, and asserts **inside the
constructor** that at zero shift it reproduces the unpatched readout **exactly** — `array_equal`,
not `allclose`, because the zero-shift path is bitwise `base` and any difference is a real
difference in the readout code. Failure raises `PassthroughNotReproduced`: the arm is wrong, not
the claim. `Claim` raises `PassthroughNotComputed` for an arm that is merely a number someone
typed, which is precisely what piece 1 accepted. Gain is a difference everywhere; there is no ratio
in the module.

Harness bug 8 now has five stages and reproduces h40's number:

```
absent: MissingArm; hand-declared: PassthroughNotComputed; computed from a readout that does not
reproduce the unpatched value at zero shift: PassthroughNotReproduced; pass-through arm declared at
the stimulus floor: ArmOffNull; with the arm computed and declared honestly the claim CONSTRUCTS
and reports gain −0.1112 against a pass-through of 0.8300  (h40 logged −0.111)
```

The fourth stage is worth reading twice. A stage-14-shaped claim that declares its pass-through arm
where the finding *needs* it — at the stimulus floor — is refused by `ArmOffNull`, because the
arithmetic is at 0.83 and not at 0.5. To publish it at all you must declare that the arithmetic
reproduces the movement, and then the claim's own treatment reads −0.11. The contract makes the
negative result the only sayable one.

## Piece 1's unprotected list: closed, improved, left

**Closed.**

- *Arm tolerance is an unmeasured 0.15.* Measured, per instrument and per arm size, with the closed
  forms checked against the statistics (above). 0.15 survives only as `types.LEGACY_TOLERANCE`, the
  fallback for an arm on an instrument the registry does not know, and it is named as such.
- *The pass-through catch only checks the arm is present.* Closed, above.
- *`REQUIRED_ARMS` is a placeholder.* It is now a view onto `registry.REGISTRY`, which also carries
  each instrument's null, tolerance, invariances and §1A reproduction target. `registry.table()`
  prints spec §8's table from the declarations rather than from a doc kept by hand.
- *No invariance tests, no self-floor, no known-zero.* All three run, on all three instruments.
- *The cross-talk catch rests on a two-test calibration.* Improved rather than closed: the battery
  is six tests and the degenerate statistic now trips an explicit `degenerate` flag as well as the
  sensitivity test (`DEGENERATE=True` in the harness output). I own three of the other tests now,
  and the degenerate rank passes all three of them — self-floor, invariance and known-zero are
  clean for a statistic pinned at 2.0. **Only sensitivity and degeneracy separate it.** That is
  worth stating plainly: four of six calibration tests cannot see h6's bug.

**Partly closed.**

- *`Direction.held_out` is declared but never verified.* `Direction` now takes an optional
  `fit_witness` recorded by the fitter and refuses (`HeldOutViolated`) when the declared unseen set
  intersects what the fit used, or when the axes disagree. `types.fit_leave_one_out` is the
  sanctioned fitter and always records one. **A `Direction` built by hand with no witness is still
  unverified** — `Direction.verified` says which, and piece 3's ledger should refuse an unverified
  one.
- *`EffectSize` is caller-supplied with no assertion.* `EffectSize.against(values, null)` computes
  and stamps `computed=True`; `Instrument.claim()` always uses it. A hand-supplied effect on a
  built instrument is now **recomputed from the treatment scores against the registry null**, and
  the `Claim` is refused (`EffectSizeUnverified`) if the two disagree about the *verdict* — sign,
  or significance claimed where there is none. Not about the value: a caller's standard error may
  legitimately come from paraphrases rather than items. A fabricated z of +4.1 on h14's scores is
  refused; a merely imprecise z is not.

**Left, with reasons.**

- *The selection catch is a regex over the caller's prose.* I did not remove it, because removing
  it would leave hand-built claims unguarded. What I added is the honest path beside it:
  `Instrument.sweep(axis, values, score_fn)` runs the sweep **in the core** and returns a
  `Selection` carrying the curve it computed, `executed=True`, and `is_a_choice` permanently False
  because there was no choice. **What I would do next:** make `executed` a *requirement* rather
  than an option — any `Claim` whose `selection.axis` is non-None must carry a core-computed curve,
  and the prose path is retired to unswept claims only. I did not do it here because it would
  refuse every §1A target until piece 3 re-runs the sweeps through the core, and piece 3 should
  make that call with the targets in front of it. The regex is still fooled by
  `rule="we looked at the curve and quoted layer 16"`, and a test asserts that it is, so nobody
  mistakes it for a guard.
- *`Probe` is a shell; nothing NDIF-side goes through the asserted path.* Extraction, piece 3.
- *Library versions are recorded, never compared.* Ledger, piece 3.

## What is still unprotected (for piece 3)

1. **A hand-declared calibration report still passes.** `CalibrationReport.hand_declared` is
   stamped and carried into the ledger row as `calibration_hand_declared: true`, but `Claim`
   accepts it for any instrument, including the three that now have real batteries. **The ledger
   should refuse to grade a §1A target on a hand-declared report.** Piece 1's harness cases 3, 4
   and 7 use them, which is why they are still accepted.
2. **The staleness check needs `lsx.core.instruments` to have been imported.** `types` cannot
   import `instruments` (cycle), so `registry.CALIBRATION_KEYS` is populated on import of the
   package. `from lsx.core.types import Claim` alone leaves a stale measured report acceptable.
3. **`readout_shift`'s arm tolerance is an estimator stand-in, not the paraphrase interval** §2a
   asks for. First thing to measure alongside the remote re-run tolerance.
4. **Nothing ties a `Claim`'s provenance to a real `Stack`** — piece 1 said this and it is still
   true; the instruments take arrays.
5. **Two of the three shipped instruments share a statistic.** `selector` and `composition` differ
   in their null and their arms, not in their ranking code. An error in `midrank` or
   `cosine_scores` breaks both at once, and their calibrations would not be independent evidence.
6. **Sensitivity is tested with planted signal in the *readout's own geometry*.** A statistic that
   finds a planted direction may still miss a real model effect that is not a direction. The
   battery bounds the instrument, not the hypothesis.
7. **`crosstalk`, `depth_gain` and `generality` are declared with unmeasured null spreads** (0.15,
   inherited). A `Claim` naming one is constructible and carries a note saying the instrument is
   unbuilt; `generality`, which has no null at all, is refused outright.

## What piece 3 should do differently

- **Grade nothing on a hand-declared calibration.** Make that the ledger's first refusal; it is the
  one remaining way to get a number out of this core without a battery behind it.
- **Measure two tolerances, not one.** The remote re-run spread §1A asks for, *and*
  `readout_shift`'s paraphrase-noise interval. They are different quantities and the second is the
  one §2a's gain is reported with.
- **Re-derive any cosine-readout target as a projection or a margin before comparing to its logged
  value.** The −0.129 bias above is systematic and in the direction that makes a busy model look
  worse than nothing.
- **Decide whether `Selection.executed` becomes mandatory.** With the §1A targets in hand you can
  see what it would cost; from here I could only see what it would break.
- **Reproduce h8 through `composition`, not through three `selector` calls.** Its no-patch arm at
  9.50 is the single strongest null in the repo, and the registry now makes it the declared one.
