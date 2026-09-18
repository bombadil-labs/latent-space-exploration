# Phase 2 follow-up: band on measured `n_eff`, not on the raw cluster count `k`

Open problem 4j (hour-48 entry, `RESULTS.md`). Fixes `cluster_evidence`/`Arm._resolve_unit`/
`InstrumentSpec.arm_tolerance` as landed one hour before this note, per `results/notes/
phase2_derivable.md` §1. Local CPU only, `.venv`, no NDIF, nothing downloaded.

## The defect, restated

`cluster_evidence` already computed `deff = 1 + (mbar - 1) * icc` and nothing used it.
`Arm._resolve_unit` set `n_independent = k` (the number of distinct cluster labels) the moment the
permutation gate passed (`p <= 0.05`), so the mechanism decided *whether* an arm could widen its
band but never *how much*. When clustering is real but partial — a small ICC that still clears the
p-gate — the honest effective sample size `n_eff = n / deff` can sit well **above** `k`, and banding
flatly on `k` gives a band wider than the design earns. Over-wide is the permissive direction: it
hides an arm that is genuinely off its null.

## The fix

`Arm._resolve_unit`, after the permutation gate passes, now computes `n_eff = n / deff` from the
arm's own `cluster_evidence` and sets `n_independent = clamp(round(n_eff), k, n)` instead of leaving
it at `k`. `k` still decides whether the arm may widen its band at all (the p-gate is unchanged);
`deff` now decides how much.

## Before / after band

**h8's permutation arm (unchanged case).** `n = 72` (four scenes x eighteen re-rankings), `k = 4`.
Synthetic reproduction: `tests/test_core_phase2.py::
test_total_clustering_reproduces_the_h8_permutation_case_unchanged` builds a 72-item arm on 4
labels with a huge between-scene spread and near-zero within-scene noise (ICC measured `> 0.999`,
"near 1" as the ticket specifies) and asserts `arm.effective_n == k == 4`. This holds by
construction, not by a special case: at `icc = 1`, `deff = 1 + (mbar - 1) * 1 = mbar`, and `mbar =
n / k` **always** (the mean cluster size is the arithmetic mean regardless of balance), so
`n_eff = n / deff = k` exactly for any cluster sizing once clustering is total. Both bands (`k`-flat,
old; `n_eff`-measured, new) select `n_independent = 4`, tolerance `arm_tolerance(72,
{"n_candidates": 3}, n_independent=4) = 1.2247` — **bit-identical**, before and after.

**A partial-clustering case (rediscovery case 13, `src/lsx/core/rediscovery.py::
case_13_partial_clustering_k_band_admits_a_dirty_arm`).** `n = 72`, `k = 6`, a real but partial
clustering (measured ICC `+0.194`, permutation `p = 0.007` over 999 shuffles — clearly inside the
gate, nothing like h8's near-1 case). An arm sitting `0.70` off its declared null of `2.00`:

| band | `n_independent` used | tolerance | verdict on the 0.70-off arm |
|---|---|---|---|
| old (`k`-flat) | 6 | **1.0000** | **admitted** (0.70 < 1.00) |
| new (`n_eff`-measured) | 23 | **0.5108** | **flagged** — `ArmOffNull` |

The same numbers, same clustering, same declaration — only the arithmetic changed. This is exactly
the bug this fix closes: partial clustering earning a full `k`-width band it had not measured.
`tests/test_core_phase2.py::test_rediscovery_catches_the_k_flat_band_this_piece_fixed` runs this
case through the real `Instrument.claim` pipeline (not a bare tolerance comparison) and checks the
positive control (the same clustering, arm on its null) still publishes clean.

`test_a_declared_unit_bands_on_units_and_widens_nothing_by_itself` (pre-existing, phase 2's own
fixture, ICC `~0.51`) moved from `effective_n == 6` to `effective_n == 11` for the same reason —
that fixture's clustering is also partial, and the old `== 6` assertion was pinning the exact
over-wide behavior this ticket is about. Updated in place rather than left to rot.

## The ICC-degenerate branch

`cluster_evidence` clamps `icc` to `max(icc, 0)` before building `deff`, so `deff >= 1` and
`n_eff <= n` always by construction — the fix's own upper clamp to `n` is therefore a no-op in
practice, kept so the invariant is enforced locally rather than trusted from upstream.

The sharper case: the permutation p-gate can find clustering real (`p <= 0.05`) while the
method-of-moments `icc` estimate for that same draw lands at or below zero (these are two different
statistics — a rank-based permutation test and an ANOVA-style moment estimator — and they can
disagree on a noisy draw). When that happens, `deff = 1` and `n_eff = n`: the fix silently declines
to widen at all, reverting to the tight per-item band.

That is a **deliberate** choice, not an oversight of the trap constraint 5 warns about. Two ways
this arithmetic can be wrong: too tight (refuses an arm that is actually clean) or too wide (admits
an arm that is actually off its null, hidden inside a band it never earned). The project's own
stated priority is unambiguous — "over-wide is the permissive direction... which is worse." A
degenerate `icc` estimate is a case where the core cannot *size* the clustering it can still detect;
treating that as "no measured widening" fails toward the safe side (an occasional false refusal of a
clean arm, caught and re-run with more draws, exactly as spec §7 already prescribes) rather than the
dangerous one (silently admitting a dirty arm because the sizing estimate happened to be noisy in
the permissive direction). The floor is still `k`, never below it, because the design itself
guarantees at least `k` independent draws regardless of what the ICC estimator says.

## What I got wrong on the way

`Instrument.claim` (`instruments.py`) attaches the measured tolerance to an already-built `Arm` via
`dataclasses.replace(arm, tolerance=...)`. `replace` constructs a *new* `Arm` instance carrying every
current field value — including a `n_independent` this fix may have already rewritten from `k` to a
measured `n_eff != k` — and re-runs `__post_init__` → `_resolve_unit`. The pre-existing consistency
check (`clusters` present + explicit `n_independent` must equal the derived `k`, "the unit is read
off the design; it cannot be two numbers") then fired on an arm that never lied about anything: it
compared the *already-resolved* `n_eff` against the raw cluster count and raised
`ArmUnitNotInDesign`. This surfaced as rediscovery case 12's "honest" positive control (a real
per-cluster offset, arm on its null) failing to publish — caught by running the harness, not by
reasoning about the diff in advance. Fixed by making `_resolve_unit` a no-op on re-entry once
`unit_evidence` is already populated (the signal that this `Arm` already earned its band), so
`replace()`-driven reconstruction does not re-derive or re-check a resolution that already happened.
This is the kind of bug constraint 5 warned about, just not the exact shape it named (I did check
the ICC-degenerate trap directly and it behaves as intended; this second-order interaction with
`dataclasses.replace` was the one I actually hit).

## What I did not do

- Did not change `cluster_evidence` itself (its `deff`/`icc` arithmetic was already correct and
  already tested against a Monte-Carlo draw elsewhere) — only what `Arm._resolve_unit` does with the
  numbers it already returns.
- Did not touch the p-gate (`ev["p"] > 0.05` still refuses the declaration outright); this fix
  changes only the size of the widening once the gate has already let it through.
- Did not add a `.npz`-backed empirical replication of h8's actual permutation arm (no cached stack
  for it in this checkout, and the constraint only asked for the case to be unchanged, which the
  `mbar = n/k` identity above proves without one).
- Did not touch `RESULTS.md`, `WRITEUP.md`, `docs/specs/*`, or anything outside `src/lsx/core/*` and
  `tests/*`.

## Test count

`pytest -q tests/`: **178 passed** (175 before this piece, 3 added:
`test_n_eff_is_clamped_to_k_and_n_and_never_widens_past_k`,
`test_total_clustering_reproduces_the_h8_permutation_case_unchanged`,
`test_rediscovery_catches_the_k_flat_band_this_piece_fixed`). No test was deleted;
`test_a_declared_unit_bands_on_units_and_widens_nothing_by_itself` and
`test_every_rediscovery_case_still_fires` were updated to the fix's honest numbers
(`effective_n == 11`, not `6`; case count `11`, not `10`). The h48 seven-decimal pin on
`Instrument.resolved_null_tol`'s default path (`test_arm_tolerance_defaults_to_the_item_count_
exactly_as_before`, `test_the_band_fix_did_not_invalidate_any_cached_calibration`) passes unchanged,
and every cached calibration key it checks still names a report on disk.
