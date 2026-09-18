"""The rediscovery harness (spec §1B): reconstruct every bug this project has shipped, and check
that the core refuses or flags it *without being told what to look for*.

Written BEFORE the types and the extraction path it tests, from the bug descriptions in
`docs/specs/core_v1.md` §1B and `docs/INSTRUMENTS.md`. That ordering is the point: a harness written
after the implementation tests the implementation's idea of the bug, not the bug.

Each case does three things:

  1. rebuilds the known-bad configuration as closely as the tiny fixture allows;
  2. calls the ordinary core entry point -- never a private hook, never a "check for X" flag;
  3. records which mechanism fired, or that nothing did.

A case whose mechanism is the wrong one is as much a failure as a case that is not caught: the
report names the mechanism so a reader can check it is the mechanism the spec claimed.

Where possible a case also carries its POSITIVE control -- the corrected configuration must pass --
so the harness cannot be satisfied by a core that refuses everything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

CAUGHT = "caught"
NOT_CAUGHT = "not_caught"
UNWIRED = "unwired"


@dataclass
class Verdict:
    bug: float
    title: str
    status: str
    mechanism: str = ""
    detail: str = ""
    positive_control: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == CAUGHT

    def render(self) -> str:
        mark = {CAUGHT: "CAUGHT", NOT_CAUGHT: "MISSED", UNWIRED: "unwired"}[self.status]
        s = f"[{mark}] bug {self.bug}: {self.title}\n    mechanism: {self.mechanism}\n    {self.detail}"
        if self.positive_control:
            s += f"\n    positive control: {self.positive_control}"
        return s


def _refusal(fn: Callable[[], Any]) -> BaseException | None:
    """Run `fn`; return the exception it raised, or None if it returned."""
    try:
        fn()
    except BaseException as e:  # noqa: BLE001 -- the harness wants whatever came out
        return e
    return None


def _named(e: BaseException | None) -> str:
    return type(e).__name__ if e is not None else "(nothing raised)"


# --------------------------------------------------------------------------------------------
# bug 1: patching `output[0]` with batch > 1 -> the patch reaches batch row 0, not the hidden
# states (h34/h36).  Expected mechanism: moved-candidates assertion on every patched forward.
# --------------------------------------------------------------------------------------------
def case_1_batch_row_patch(lm) -> Verdict:
    from . import checks
    from .extract import asserted_patched_forward

    texts = ["a particle has a definite position",
             "measurement disturbs what it measures",
             "position and momentum are complementary descriptions"]
    v = 0.5

    def row_zero_patch(hidden):
        """`B[l].output[0][:] = B[l].output[0] + v` under transformers >= 4.54, verbatim: with a
        bare tensor `[batch, seq, d]`, `output[0]` is batch row 0."""
        h = hidden.clone()
        h[0] = h[0] + v
        return h

    def whole_tensor_patch(hidden):
        return hidden + v

    def readout(acts):           # any per-sequence scalar; the assertion counts what moved
        return acts[:, 2].mean(axis=-1)

    e = _refusal(lambda: asserted_patched_forward(lm, texts, patch_layer=1, patch_fn=row_zero_patch,
                                                  readout=readout))
    ok_ctrl = _refusal(lambda: asserted_patched_forward(lm, texts, patch_layer=1,
                                                        patch_fn=whole_tensor_patch, readout=readout))
    if isinstance(e, checks.MovedCandidates):
        return Verdict(1, "patched `output[0]` with batch > 1", CAUGHT,
                       "extract.asserted_patched_forward -> MovedCandidates",
                       str(e),
                       "whole-tensor patch passes" if ok_ctrl is None else f"FAILED: {_named(ok_ctrl)}")
    return Verdict(1, "patched `output[0]` with batch > 1", NOT_CAUGHT,
                   _named(e), "row-0 patch was not refused by the moved-candidates assertion")


# --------------------------------------------------------------------------------------------
# bug 2: absolute span indices under left padding (h39).  Expected mechanism: batched-vs-single
# equivalence on the SHORTEST item of each batch.  The padding-convention assertion catches the
# same configuration earlier, so the case is run twice: once whole, once with that assertion
# bypassed, to show the equivalence check standing on its own.
# --------------------------------------------------------------------------------------------
def _padding_grid():
    from .types import Grid, Item
    # Spans late in the text and a length spread SMALLER than the span's start index, so absolute
    # indexing under left padding lands on REAL but WRONG tokens rather than in the pad block --
    # the h39 situation for the 117 partially shifted passages, which is the hard case: nothing is
    # obviously empty, the vectors are merely the wrong ones.
    filler = "x y z w q r s t u v x y z w q r s t u v x y z w q r s t u v"   # 30 tokens
    bodies = [f"{filler}{' q r' * k} the muppets take manhattan" for k in (3, 0, 1, 2)]
    levels = ["long", "short", "mid", "midlong"]
    items = []
    for i, b in enumerate(bodies):
        j = b.index("the muppets")
        items.append(Item(text=b, factors={"len": levels[i]}, spans={"tail": (j, len(b))}))
    return Grid(items, name="padding_probe", leak_check=False)


def case_2_absolute_spans_left_padding(lm) -> Verdict:
    from . import checks
    from .extract import build_stack, per_item_equivalence

    grid = _padding_grid()
    old = lm.tok.padding_side
    lm.tok.padding_side = "left"          # what nnsight's LanguageModel does, unasked
    try:
        whole = _refusal(lambda: build_stack(lm, grid, batch_size=4, span_policy="absolute",
                                             declared_padding_side="right"))
        bypassed = _refusal(lambda: build_stack(lm, grid, batch_size=4, span_policy="absolute",
                                                declared_padding_side="right", bypass=("padding",)))
        good = _refusal(lambda: build_stack(lm, grid, batch_size=4, span_policy="end_relative"))
        # the spec's emphatic point: a RANDOM item would usually have passed.
        cos = per_item_equivalence(lm, grid, span_policy="absolute", batch_size=4)
    finally:
        lm.tok.padding_side = old

    longest = int(np.argmax([len(it.text) for it in grid.items]))
    detail = (f"padding-convention assertion: {_named(whole)}; with it bypassed: {_named(bypassed)}; "
              f"per-item batched-vs-single cosine under the bug {np.round(cos, 4).tolist()} "
              f"(longest item {cos[longest]:.5f} -- a random sample would have passed)")
    if isinstance(whole, checks.PaddingConvention) and isinstance(bypassed, checks.BatchEquivalence):
        return Verdict(2, "absolute span indices under left padding", CAUGHT,
                       "extract.build_stack -> PaddingConvention, and independently "
                       "BatchEquivalence on the shortest item",
                       detail,
                       "end-relative indexing passes" if good is None else f"FAILED: {_named(good)}",
                       extra={"cosines": cos.tolist()})
    return Verdict(2, "absolute span indices under left padding", NOT_CAUGHT,
                   f"{_named(whole)} / {_named(bypassed)}", detail)


# --------------------------------------------------------------------------------------------
# bug 3: rank-1-on-ties with no no-patch arm (h34).  Two halves, and the core must catch both:
# the missing arm, and -- once the arm is present -- the arm sitting off its declared null.
# --------------------------------------------------------------------------------------------
def case_3_rank1_ties_no_no_patch() -> Verdict:
    from . import checks
    from .types import Arm, Claim, EffectSize, Floor, Measured, Selection

    def _claim(arms):
        return Claim(instrument="selector",
                     treatment=Measured(np.full(30, 1.22), label="theme lens"),
                     arms=arms,
                     floor=Floor(stimulus=2.0, estimator=2.0),
                     selection=Selection(axis="layer", rule="full curve reported", held_out=True),
                     effect=EffectSize(size=-0.78, n=30, z=-3.1),
                     calibration=checks.CalibrationReport.hand_declared("selector", passed=True),
                     provenance={"model": "llama-3.1-8b", "layers": [10]},
                     # h34 ranked 3 candidates: the registry turns that into the null (2.0) and
                     # into the tolerance around it (3 * sqrt((9-1)/12) / sqrt(30) = 0.447), so
                     # neither is the caller's to choose.
                     config={"n_candidates": 3})

    missing = _refusal(lambda: _claim({"random": Arm(np.full(30, 1.22), expected_null=2.0),
                                       "permutation": Arm(np.full(30, 2.0), expected_null=2.0)}))
    # h34's actual numbers: no_patch reads 1.00 under rank-1-on-ties; random reads 1.22.
    off = _refusal(lambda: _claim({"random": Arm(np.full(30, 1.22), expected_null=2.0),
                                   "permutation": Arm(np.full(30, 2.0), expected_null=2.0),
                                   "no_patch": Arm(np.full(30, 1.00), expected_null=2.0)}))
    ok = _refusal(lambda: _claim({"random": Arm(np.full(30, 2.03), expected_null=2.0),
                                  "permutation": Arm(np.full(30, 1.98), expected_null=2.0),
                                  "no_patch": Arm(np.full(30, 2.00), expected_null=2.0)}))
    if isinstance(missing, checks.MissingArm) and isinstance(off, checks.ArmOffNull):
        from .registry import spec as _spec
        tol = _spec("selector").arm_tolerance(30, {"n_candidates": 3})
        return Verdict(3, "rank-1-on-ties with no no-patch arm", CAUGHT,
                       "Claim -> MissingArm('no_patch'); with the arm present, Claim -> ArmOffNull",
                       f"{missing}  ||  {off}  ||  tolerance {tol:.3f}, measured from selector's "
                       f"own null at k=3, n=30 (piece 1's flat 0.15 would have fired on a clean arm "
                       f"at this n)",
                       "arms at their nulls construct" if ok is None else f"FAILED: {_named(ok)}")
    return Verdict(3, "rank-1-on-ties with no no-patch arm", NOT_CAUGHT,
                   f"{_named(missing)} / {_named(off)}", "one half of the h34 failure went through")


# --------------------------------------------------------------------------------------------
# bug 4: best layer chosen on the evaluation split (h1-h3, h16, h38).
# --------------------------------------------------------------------------------------------
def case_4_best_layer_on_scoring_data() -> Verdict:
    from . import checks
    from .types import Arm, Claim, EffectSize, Floor, Measured, Selection

    def _claim(selection):
        return Claim(instrument="selector",
                     treatment=Measured(np.full(40, 2.01), label="role transfer @ best layer"),
                     arms={"random": Arm(np.full(40, 3.5), expected_null=3.5),
                           "no_patch": Arm(np.full(40, 3.5), expected_null=3.5),
                           "permutation": Arm(np.full(40, 3.48), expected_null=3.5)},
                     floor=Floor(stimulus=3.5, estimator=3.5),
                     selection=selection,
                     effect=EffectSize(size=-1.49, n=40, z=-4.0),
                     calibration=checks.CalibrationReport.hand_declared("selector", passed=True),
                     provenance={"model": "qwen2.5-1.5b"}, config={"n_candidates": 6})

    e = _refusal(lambda: _claim(Selection(axis="layer", rule="argmax over 29 layers", held_out=False)))
    ok = _refusal(lambda: _claim(Selection(axis="layer", rule="full curve reported", held_out=True)))
    if isinstance(e, checks.SelectionOnScoringData):
        return Verdict(4, "best layer chosen on the scoring data", CAUGHT,
                       "Claim -> SelectionOnScoringData (Selection.rule is a choice, held_out is False)",
                       str(e),
                       "full curve constructs" if ok is None else f"FAILED: {_named(ok)}")
    return Verdict(4, "best layer chosen on the scoring data", NOT_CAUGHT, _named(e),
                   "an argmax-on-scoring-data Claim was accepted")


# --------------------------------------------------------------------------------------------
# bug 5: a 1536-d residual norm with no floor (h28-h32).
# --------------------------------------------------------------------------------------------
def case_5_residual_norm_no_floor() -> Verdict:
    from . import checks
    from .types import Arm, Claim, EffectSize, Floor, Measured, Selection

    def _claim(floor):
        return Claim(instrument="discrimination",
                     treatment=Measured(np.full(8, 9.1), label="||resid|| tau"),
                     arms={"shuffled_stimulus": Arm(np.full(8, 8.9), expected_null=9.0),
                           "floor": Arm(np.full(8, 9.0), expected_null=9.0)},
                     floor=floor,
                     selection=Selection(axis=None, rule="single pre-registered layer", held_out=True),
                     effect=EffectSize(size=0.1, n=8, z=0.2),
                     calibration=checks.CalibrationReport.hand_declared("discrimination", passed=True),
                     provenance={"model": "qwen2.5-1.5b", "d": 1536},
                     # piece 4: `discrimination` is now BUILT, so its null is no longer a
                     # placeholder and this case has to declare the floor it is measured against --
                     # which is the h28 lesson the case is about, arriving one level earlier. With
                     # the floor declared, the Claim's recomputed effect (0.1 at z 0.80) agrees with
                     # the declared one about the verdict; without it the instrument's default null
                     # of 0 made a 9.1 residual norm look like a nine-sigma result, and
                     # `EffectSizeUnverified` said so.
                     config={"floor": 9.0, "m": 9})

    none_at_all = _refusal(lambda: _claim(None))
    stim_only = _refusal(lambda: _claim(Floor(stimulus=0.0, estimator=None)))
    ok = _refusal(lambda: _claim(Floor(stimulus=0.0, estimator=7.2)))
    if isinstance(none_at_all, checks.MissingFloor) and isinstance(stim_only, checks.MissingFloor):
        return Verdict(5, "high-dimensional residual norm with no floor", CAUGHT,
                       "Claim -> MissingFloor; Floor requires BOTH stimulus and estimator",
                       f"{none_at_all}  ||  {stim_only}",
                       "both floors construct" if ok is None else f"FAILED: {_named(ok)}")
    return Verdict(5, "high-dimensional residual norm with no floor", NOT_CAUGHT,
                   f"{_named(none_at_all)} / {_named(stim_only)}",
                   "a floorless (or stimulus-only) Claim was accepted -- h28's exact shape")


# --------------------------------------------------------------------------------------------
# bug 6: a cross-talk rank that averages to chance by construction (h6).  The statistic returns
# its correct null on noise AND on everything else, so only a synthetic-SIGNAL sensitivity test
# separates it from a working one.
# --------------------------------------------------------------------------------------------
def case_6_degenerate_crosstalk_rank() -> Verdict:
    from . import checks
    from .types import Arm, Claim, EffectSize, Floor, Measured, Selection

    def broken_rank(x: np.ndarray) -> float:
        """Rank each of three candidates among the same three: a permutation of {1,2,3}, mean 2,
        whatever the patch did. `x` is [n, 3] of candidate scores."""
        order = np.argsort(-x, axis=1)
        ranks = np.empty_like(order)
        seq = np.arange(1, x.shape[1] + 1)[None, :].repeat(len(x), 0)
        np.put_along_axis(ranks, order, seq, axis=1)
        return float(ranks.mean())

    def working_rank(x: np.ndarray) -> float:
        """Rank of the TARGET candidate (column 0) among three -- moves when column 0 is boosted."""
        return float((1 + (x[:, 1:] > x[:, :1]).sum(axis=1)).mean())

    def plant(x, a):
        out = x.copy()
        out[:, 0] += a
        return out

    bad = checks.run_calibration(broken_rank, shape=(512, 3), declared_null=2.0, plant=plant,
                                 name="crosstalk")
    good = checks.run_calibration(working_rank, shape=(512, 3), declared_null=2.0, plant=plant,
                                  name="crosstalk")

    def _claim(report):
        return Claim(instrument="crosstalk",
                     treatment=Measured(np.full(108, 2.0), label="C_voice_under_era"),
                     arms={"permutation": Arm(np.full(108, 2.0), expected_null=2.0)},
                     floor=Floor(stimulus=2.0, estimator=2.0),
                     selection=Selection(axis=None, rule="single layer, pre-registered", held_out=True),
                     effect=EffectSize(size=0.0, n=108, z=0.0),
                     calibration=report,
                     provenance={"model": "qwen2.5-1.5b", "layers": [14]})

    e = _refusal(lambda: _claim(bad))
    ok = _refusal(lambda: _claim(good))
    detail = (f"broken rank: noise {bad.noise_value:.3f} (null 2.0, passes), sensitivity "
              f"{np.round(bad.sensitivity_values, 3).tolist()} -> flat, DEGENERATE={bad.degenerate}; "
              f"working rank sensitivity {np.round(good.sensitivity_values, 3).tolist()}, "
              f"degenerate={good.degenerate}")
    if (not bad.noise_failed and bad.sensitivity_failed and bad.degenerate
            and isinstance(e, checks.CalibrationFailed)):
        return Verdict(6, "cross-talk rank pinned at chance by construction", CAUGHT,
                       "checks.run_calibration -> sensitivity failure AND the degenerate flag; "
                       "Claim -> CalibrationFailed",
                       detail,
                       "working rank calibrates and constructs" if ok is None else f"FAILED: {_named(ok)}")
    return Verdict(6, "cross-talk rank pinned at chance by construction", NOT_CAUGHT, _named(e), detail)


# --------------------------------------------------------------------------------------------
# bug 7: a grid whose labels are recoverable from a bag of tokens (h32/h35/h38).  ALLOWED -- but
# the Claim must report gain over the measured floor, never the raw score.
# --------------------------------------------------------------------------------------------
def _leaky_grid():
    from .types import Grid, Item
    items = []
    for i in range(12):
        near = i % 2 == 0
        text = ("one day later the mayfly is dead" if near else
                "one million years later the sediment has become stone")
        items.append(Item(text=f"{text} number {i}", factors={"interval": "near" if near else "far"},
                          spans={"state": (0, len(text))}))
    return Grid(items, name="v2_state_spans")


def _clean_grid():
    """The h35 v3 repair: the same state prose under both levels, so the bag of tokens carries no
    information about the label and only the model's computation can."""
    from .types import Grid, Item
    sentences = ["the mayfly is dead", "the sediment has become stone",
                 "the lamp is still warm", "the ridge has worn flat",
                 "the kettle has gone quiet", "the path is overgrown"]
    items = []
    for k, s in enumerate(sentences):
        for j, level in enumerate(("near", "far")):
            items.append(Item(text=f"{s} number {2 * k + j}", factors={"interval": level},
                              spans={"state": (0, len(s))}))
    return Grid(items, name="v3_state_spans")


def case_7_leaky_grid_raw_score() -> Verdict:
    from . import checks
    from .types import Arm, Claim, EffectSize, Floor, Measured, Selection

    leaky, clean = _leaky_grid(), _clean_grid()

    def _claim(grid, report_as):
        return Claim(instrument="discrimination",
                     treatment=Measured(np.full(24, 0.961), label="delta-t discrimination"),
                     arms={"shuffled_stimulus": Arm(np.full(24, 0.5), expected_null=0.5),
                           "floor": Arm(np.full(24, 0.728), expected_null=0.728)},
                     floor=Floor(stimulus=0.728, estimator=0.5),
                     selection=Selection(axis="layer", rule="full curve reported", held_out=True),
                     effect=EffectSize(size=0.233, n=24, z=3.0),
                     calibration=checks.CalibrationReport.hand_declared("discrimination", passed=True),
                     provenance={"model": "gemma-2-9b-it"},
                     # the measured lexical floor, declared to the instrument rather than only to
                     # the Floor: with it the null IS 0.728 and the declared effect of 0.233 is the
                     # gain the case is about (piece 4).
                     config={"floor": 0.728, "m": 9},
                     grid=grid, report_as=report_as)

    raw = _refusal(lambda: _claim(leaky, "raw"))
    gain = _refusal(lambda: _claim(leaky, "gain_over_floor"))
    clean_raw = _refusal(lambda: _claim(clean, "raw"))
    detail = (f"leak report on the leaky grid: {leaky.leak.summary()}; on the clean grid: "
              f"{clean.leak.summary()}")
    if leaky.leak.leaky and isinstance(raw, checks.RawScoreOnLeakyGrid) and gain is None:
        ctl = ("clean grid accepts a raw score" if clean_raw is None
               else f"FAILED: clean grid refused ({_named(clean_raw)})")
        return Verdict(7, "labels recoverable from a bag of tokens", CAUGHT,
                       "Grid.leak flags it at construction; Claim -> RawScoreOnLeakyGrid unless "
                       "report_as='gain_over_floor'",
                       detail, ctl)
    return Verdict(7, "labels recoverable from a bag of tokens", NOT_CAUGHT, _named(raw), detail)


# --------------------------------------------------------------------------------------------
# bug 7.5 (the seventh instrument, found at h40 and not in INSTRUMENTS.md's numbered list):
# `residuals()[-1]` is the final-norm OUTPUT, not the last block's residual.
# --------------------------------------------------------------------------------------------
def case_7p5_post_norm_last_hidden_state(lm) -> Verdict:
    import torch

    from . import checks
    from .extract import capture_residual

    got = capture_residual(lm, "the muppets take manhattan", layer=lm.n_layers)
    hs, _ = lm.residuals("the muppets take manhattan")
    same_as_hf = bool(torch.allclose(torch.as_tensor(got), hs[-1], atol=1e-4))
    e = _refusal(lambda: checks.assert_pre_norm(lm, np.asarray(hs[-1]), layer=lm.n_layers))
    if not same_as_hf and isinstance(e, checks.PostNormResidual):
        return Verdict(7.5, "`residuals()[-1]` is the final-norm output, not a residual", CAUGHT,
                       "extract.capture_residual routes layer == n_layers through "
                       "LM.pre_norm_residual; checks.assert_pre_norm -> PostNormResidual",
                       f"captured residual differs from hidden_states[-1] (mean position norm "
                       f"{np.linalg.norm(got, axis=-1).mean():.1f} vs "
                       f"{hs[-1].norm(dim=-1).mean():.1f})")
    return Verdict(7.5, "`residuals()[-1]` is the final-norm output, not a residual", NOT_CAUGHT,
                   _named(e), "the post-norm hidden state was accepted as a residual")


# --------------------------------------------------------------------------------------------
# bug 8: a readout at or after the patch layer, whose movement is residual arithmetic (h14/h40).
# --------------------------------------------------------------------------------------------
def case_8_readout_after_patch_no_passthrough() -> Verdict:
    """Four stages now, where piece 1 had one.

    Piece 1 could only ask whether an arm called `passthrough` was PRESENT, which a caller satisfies
    by typing a number. Piece 2's arm is computed from the residual arithmetic, asserts in its own
    constructor that it reproduces the unpatched readout exactly at zero shift, and its gain is a
    difference. So the stage-14 configuration is now refused three times over, and the fourth stage
    is the one that matters: with a correctly computed arm the claim CONSTRUCTS, and what it prints
    is h40's negative result -- no gain over the pass-through.
    """
    from . import checks
    from .instruments import PassthroughArm, build
    from .planted import unit
    from .types import Arm, Claim, EffectSize, Floor, Measured, Selection

    rng = np.random.default_rng(14)
    n, d = 18, 256
    direction = unit(rng.normal(size=d))
    base = rng.normal(size=(n, d))
    scale = float(np.linalg.norm(base, axis=-1).mean())
    shift = 0.89 * scale * direction                 # the era shift, decodable on its own
    blocks = -0.111 * scale * direction              # what layers 14-19 actually add, h40's number

    def readout(resid):                              # the SAME readout code for both arms
        return (resid @ direction) / np.linalg.norm(base, axis=-1)

    treatment_scores = readout(base + shift + blocks) - readout(base + shift)   # a DIFFERENCE
    rs = build("readout_shift", d=d)

    def _claim(arms, instrument="readout_shift", patch_layer=14, readout_layer=20,
               effect=None, treatment=None):
        return Claim(instrument=instrument,
                     treatment=treatment or Measured(treatment_scores, label="era gain over "
                                                                             "pass-through"),
                     arms=arms,
                     floor=Floor(stimulus=0.5, estimator=0.0),
                     selection=Selection(axis=None, rule="pre-registered layer pair", held_out=True),
                     effect=effect or EffectSize(size=0.39, n=n, z=4.1),
                     calibration=rs.calibration,
                     provenance={"model": "qwen2.5-1.5b"}, config={"d": d},
                     patch_layer=patch_layer, readout_layer=readout_layer)

    plumbing = {"random": Arm(np.zeros(n), expected_null=0.0),
                "no_patch": Arm(np.zeros(n), expected_null=0.0)}

    # (a) the arm is absent
    missing = _refusal(lambda: _claim(dict(plumbing)))

    # (b) the arm is a number someone typed -- piece 1 accepted exactly this
    declared = dict(plumbing)
    declared["passthrough"] = Arm(np.full(n, 0.89), expected_null=0.89,
                                  justification="norm-matched pass-through")
    hand = _refusal(lambda: _claim(declared))

    # (c) the arm is computed, but from a readout that does not reproduce the unpatched value at
    #     zero shift -- a different pooling, say. The ARM is wrong, not the claim.
    def wrong_readout(resid):
        return (resid @ direction) / np.linalg.norm(base, axis=-1) + 1e-9

    wrong = _refusal(lambda: PassthroughArm.compute(
        base=base, shift=shift, readout=wrong_readout, expected_null=0.5,
        justification="norm-matched pass-through", unpatched=readout(base)))

    # (d) the arm computed properly. The claim constructs and reports h40's negative gain --
    #     but only once the arm's declared null stops pretending the arithmetic is at the floor.
    arm = PassthroughArm.compute(base=base, shift=shift, readout=readout, expected_null=0.5,
                                 justification="if the movement were the model's work, the offline "
                                               "arithmetic would sit at the stimulus floor",
                                 match_norms=np.linalg.norm(base + shift + blocks, axis=-1),
                                 unpatched=readout(base))
    at_floor = _refusal(lambda: _claim(dict(plumbing, passthrough=arm)))

    honest = PassthroughArm.compute(
        base=base, shift=shift, readout=readout, expected_null=float(np.mean(arm.scores)),
        justification="the arithmetic alone reproduces the movement; that IS the h40 finding",
        match_norms=np.linalg.norm(base + shift + blocks, axis=-1), unpatched=readout(base))
    good = dict(plumbing, passthrough=honest)
    built, ok = None, None
    try:
        built = _claim(good, effect=EffectSize.against(treatment_scores, 0.0))
    except BaseException as exc:  # noqa: BLE001
        ok = exc

    # (e) and the fabricated positive: same scores, a claimed z of +4.1
    fabricated = _refusal(lambda: _claim(good))

    # a selector whose readout precedes its patch layer must NOT be forced to carry the arm
    before = _refusal(lambda: Claim(
        instrument="selector", treatment=Measured(np.full(40, 2.01), label="before"),
        arms={"random": Arm(np.full(40, 3.5), expected_null=3.5),
              "no_patch": Arm(np.full(40, 3.5), expected_null=3.5),
              "permutation": Arm(np.full(40, 3.5), expected_null=3.5)},
        floor=Floor(stimulus=3.5, estimator=3.5),
        selection=Selection(axis=None, rule="pre-registered layer", held_out=True),
        effect=EffectSize(size=-1.49, n=40, z=-4.0),
        calibration=checks.CalibrationReport.hand_declared("selector", passed=True),
        provenance={"model": "qwen2.5-1.5b"}, config={"n_candidates": 6},
        patch_layer=20, readout_layer=14))

    detail = (f"absent: {_named(missing)}; hand-declared: {_named(hand)}; computed from a readout "
              f"that does not reproduce the unpatched value at zero shift: {_named(wrong)}; "
              f"pass-through arm declared at the stimulus floor: {_named(at_floor)}; with the arm "
              f"computed and declared honestly the claim CONSTRUCTS and reports gain "
              f"{built.treatment.value:+.4f} against a pass-through of "
              f"{honest.value:.4f} (h40 logged -0.111)"
              if built is not None else f"FAILED to construct: {_named(ok)}")
    caught = (isinstance(missing, checks.MissingArm) and "passthrough" in str(missing)
              and isinstance(hand, checks.PassthroughNotComputed)
              and isinstance(wrong, checks.PassthroughNotReproduced)
              and isinstance(at_floor, checks.ArmOffNull)
              and isinstance(fabricated, checks.EffectSizeUnverified)
              and built is not None)
    if caught:
        ctl = ["computed arm constructs and reads a NEGATIVE gain",
               "fabricated +4.1 z on the same scores: EffectSizeUnverified",
               "readout before patch layer: arm not required" if before is None
               else f"FAILED: {_named(before)}"]
        return Verdict(8, "readout at or after the patch layer with no pass-through arm", CAUGHT,
                       "Claim -> MissingArm('passthrough'); PassthroughNotComputed for a declared "
                       "one; PassthroughNotReproduced inside the arm's own constructor; ArmOffNull "
                       "when the arithmetic does not sit where the claim needs it",
                       detail, "; ".join(ctl),
                       extra={"gain": None if built is None else built.treatment.value,
                              "passthrough": honest.value})
    return Verdict(8, "readout at or after the patch layer with no pass-through arm", NOT_CAUGHT,
                   f"{_named(missing)} / {_named(hand)} / {_named(wrong)} / {_named(at_floor)} / "
                   f"{_named(fabricated)}", detail)


# --------------------------------------------------------------------------------------------
# bugs 9, 10 and 11: the three PUBLICATION refusals piece 3 added.
#
# Piece 3 tested these directly -- "call `append` on a claim with a hand-declared report, expect
# `HandDeclaredCalibration`" -- and said in its own handover that it should not have: §1B's test is
# "fed a known-bad configuration, the core refuses WITHOUT being told what to look for", and a test
# that names the exception it wants is being told. These three cases feed the ledger a
# configuration that is bad in the way the project has actually been bad, and record what fired.
#
# Each carries its positive control, because a ledger that refuses everything is no ledger.
# --------------------------------------------------------------------------------------------
def _publishable_claim(**over):
    """A claim that is correct in every way the ledger checks, so a case can break exactly one."""
    from . import checks, instruments
    from .types import Arm, Floor, Measured, Selection, stack_signature

    inst = instruments.build("selector", n=200, d=32, n_candidates=6)
    rng = np.random.default_rng(3)
    n = 60
    treat = np.clip(np.round(rng.normal(2.1, 1.0, size=n)), 1, 6)
    arms = {name: np.clip(np.round(rng.normal(3.5, 1.7, size=n)), 1, 6)
            for name in ("random", "no_patch", "permutation")}
    prov = {"model": "Qwen/Qwen2.5-1.5B", "layers": [20], "pooling": "mean",
            "grid_hash": "harness_v1", "code_version": "harness", "tokenizer_padding": "left",
            "template": None, "lib_versions": {"torch": "x", "transformers": "y"},
            "acts_digest": "deadbeefdeadbeef", "span_policy": "end_relative"}
    prov = dict(prov, **over.pop("provenance_extra", {}))
    if over.pop("sign_provenance", True):
        prov["stack_signature"] = stack_signature(prov)
    if "tamper" in over:
        prov[over["tamper"]] = "TAMPERED"
        over.pop("tamper")
    kw = dict(treatment=Measured(treat, label="harness selector"), arms=arms,
              floor=Floor(stimulus=3.5, estimator=3.5),
              selection=Selection(axis=None, rule="pre-registered layer 20; no sweep"),
              provenance=prov, stage="harness")
    kw.update(over)
    claim = inst.claim(**kw)
    if "calibration_override" in over:
        claim.calibration = over["calibration_override"]
    return claim


def case_9_hand_declared_calibration_published() -> Verdict:
    """A claim carrying a PROMISE that the battery would pass, offered to the ledger."""
    from . import checks, ledger
    from .types import Arm, Claim, EffectSize, Floor, Measured, Selection, stack_signature

    led = ledger.Ledger(path=_tmp_ledger())

    good = _publishable_claim()
    ok = _refusal(lambda: led.append(good, note="harness positive control"))

    # the same claim in every respect, except that its calibration is a hand-declared report --
    # which is what every §11.4 instrument and every hand-built claim in this repo carries.
    prov = dict(good.provenance)
    prov.pop("calibration_key", None)
    prov.pop("stack_signature", None)
    prov["stack_signature"] = stack_signature(prov)
    promised = Claim(instrument="selector", treatment=good.treatment, arms=good.arms,
                     floor=good.floor, selection=good.selection, effect=good.effect,
                     calibration=checks.CalibrationReport.hand_declared(
                         "selector", passed=True, note="battery would pass"),
                     provenance=prov, config=dict(good.config), stage="harness")
    hand = _refusal(lambda: led.append(promised))

    caught = isinstance(hand, checks.HandDeclaredCalibration)
    ctl = ("a measured report publishes" if ok is None else f"FAILED: {_named(ok)}")
    if caught:
        return Verdict(9, "a hand-declared calibration report offered to the ledger", CAUGHT,
                       "Ledger.append -> HandDeclaredCalibration", str(hand).splitlines()[0], ctl)
    return Verdict(9, "a hand-declared calibration report offered to the ledger", NOT_CAUGHT,
                   _named(hand), "the ledger graded a promise", ctl)


def case_10_provenance_not_from_a_stack() -> Verdict:
    """Two shapes: provenance that `build_stack` never wrote, and provenance edited afterwards."""
    from . import checks, ledger

    led = ledger.Ledger(path=_tmp_ledger())
    ok = _refusal(lambda: led.append(_publishable_claim(), note="harness positive control"))

    unsigned = _refusal(lambda: led.append(_publishable_claim(sign_provenance=False)))
    # signed, then a recorded field changed -- the padding side, which is h39's field
    tampered = _publishable_claim()
    tampered.provenance["tokenizer_padding"] = "right"
    edited = _refusal(lambda: led.append(tampered))

    caught = (isinstance(unsigned, checks.ProvenanceNotFromStack)
              and isinstance(edited, checks.ProvenanceNotFromStack))
    detail = f"unsigned: {_named(unsigned)}; padding side edited after signing: {_named(edited)}"
    ctl = ("a signed, unedited stack publishes" if ok is None else f"FAILED: {_named(ok)}")
    if caught:
        return Verdict(10, "a hand-rolled extraction wrapped in a well-formed Claim", CAUGHT,
                       "Ledger.append -> ProvenanceNotFromStack (absent signature, and a "
                       "signature that no longer matches the fields it is attached to)",
                       detail, ctl)
    return Verdict(10, "a hand-rolled extraction wrapped in a well-formed Claim", NOT_CAUGHT,
                   _named(unsigned), detail, ctl)


def case_11_swept_axis_with_no_curve() -> Verdict:
    """h16's shape: an axis was swept, and what reaches the ledger is prose about the sweep.

    The prose passes the `Selection` regex -- deliberately, since piece 2 left a test asserting
    that it does -- so this case is exactly the gap between "the caller says they reported the
    curve" and "the core has the curve".
    """
    from . import checks, instruments, ledger
    from .types import Selection

    led = ledger.Ledger(path=_tmp_ledger())

    story = Selection(axis="layer", rule="we looked at the curve and quoted layer 16",
                      held_out=True)
    told = _refusal(lambda: led.append(_publishable_claim(selection=story)))

    inst = instruments.build("selector", n=200, d=32, n_candidates=6)
    curve = {0: 2.73, 4: 2.20, 10: 1.98, 16: 1.73, 20: 1.86, 24: 2.39, 28: 2.73}
    executed = inst.sweep("layer", list(curve), lambda l: curve[int(l)])
    ok = _refusal(lambda: led.append(_publishable_claim(selection=executed)))
    unswept = _refusal(lambda: led.append(_publishable_claim()))

    caught = isinstance(told, checks.SweepNotExecuted)
    ctl = (f"a core-computed curve publishes ({len(curve)} points)"
           if ok is None and unswept is None
           else f"FAILED: {_named(ok)} / {_named(unswept)}")
    if caught:
        return Verdict(11, "a swept axis carried as the caller's prose, not as a curve", CAUGHT,
                       "Ledger.append -> SweepNotExecuted; the same claim with a curve from "
                       "Instrument.sweep publishes",
                       f"prose that passes the Selection regex: {story.rule!r} -> {_named(told)}",
                       ctl)
    return Verdict(11, "a swept axis carried as the caller's prose, not as a curve", NOT_CAUGHT,
                   _named(told), f"rule={story.rule!r}", ctl)


def _tmp_ledger():
    """A throwaway ledger file. The harness must never touch `results/ledger.jsonl`: these are
    demonstrations that a mechanism fires, not results (the same line piece 3 drew for its
    hand-declared reports)."""
    import pathlib
    import tempfile
    return pathlib.Path(tempfile.mkdtemp(prefix="lsx_harness_")) / "ledger.jsonl"


# --------------------------------------------------------------------------------------------

MODEL_CASES = {1: case_1_batch_row_patch, 2: case_2_absolute_spans_left_padding,
               7.5: case_7p5_post_norm_last_hidden_state}
PURE_CASES = {3: case_3_rank1_ties_no_no_patch, 4: case_4_best_layer_on_scoring_data,
              5: case_5_residual_norm_no_floor, 6: case_6_degenerate_crosstalk_rank,
              7: case_7_leaky_grid_raw_score, 8: case_8_readout_after_patch_no_passthrough,
              9: case_9_hand_declared_calibration_published,
              10: case_10_provenance_not_from_a_stack,
              11: case_11_swept_axis_with_no_curve}


def run_all(lm=None) -> list[Verdict]:
    """Run every case. Cases needing a forward pass are skipped (reported `unwired`) without `lm`."""
    out = []
    for bug, fn in sorted({**MODEL_CASES, **PURE_CASES}.items()):
        if bug in MODEL_CASES and lm is None:
            out.append(Verdict(bug, fn.__name__, UNWIRED, "", "needs a model fixture"))
            continue
        out.append(fn(lm) if bug in MODEL_CASES else fn())
    return out


def report(lm=None) -> str:
    vs = run_all(lm)
    n = sum(v.ok for v in vs)
    return "\n".join([v.render() for v in vs] + [f"\n{n}/{len(vs)} known bugs rediscovered"])


if __name__ == "__main__":  # pragma: no cover
    print(report())
