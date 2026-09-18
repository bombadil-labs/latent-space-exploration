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
                     provenance={"model": "llama-3.1-8b", "layers": [10]})

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
        return Verdict(3, "rank-1-on-ties with no no-patch arm", CAUGHT,
                       "Claim -> MissingArm('no_patch'); with the arm present, Claim -> ArmOffNull",
                       f"{missing}  ||  {off}",
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
                     provenance={"model": "qwen2.5-1.5b"})

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
                     provenance={"model": "qwen2.5-1.5b", "d": 1536})

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
                                 name="crosstalk_rank")
    good = checks.run_calibration(working_rank, shape=(512, 3), declared_null=2.0, plant=plant,
                                  name="target_rank")

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
              f"{np.round(bad.sensitivity_values, 3).tolist()} -> flat; working rank sensitivity "
              f"{np.round(good.sensitivity_values, 3).tolist()}")
    if not bad.noise_failed and bad.sensitivity_failed and isinstance(e, checks.CalibrationFailed):
        return Verdict(6, "cross-talk rank pinned at chance by construction", CAUGHT,
                       "checks.run_calibration -> sensitivity failure; Claim -> CalibrationFailed",
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
    from . import checks
    from .types import Arm, Claim, EffectSize, Floor, Measured, Selection

    def _claim(arms, instrument="readout_shift", patch_layer=14, readout_layer=20):
        return Claim(instrument=instrument,
                     treatment=Measured(np.full(18, 0.89), label="era address moved"),
                     arms=arms,
                     floor=Floor(stimulus=0.5, estimator=0.5),
                     selection=Selection(axis=None, rule="pre-registered layer pair", held_out=True),
                     effect=EffectSize(size=0.39, n=18, z=4.1),
                     calibration=checks.CalibrationReport.hand_declared(instrument, passed=True),
                     provenance={"model": "qwen2.5-1.5b"},
                     patch_layer=patch_layer, readout_layer=readout_layer)

    plumbing = {"random": Arm(np.full(18, 0.02), expected_null=0.0),
                "no_patch": Arm(np.full(18, 0.0), expected_null=0.0)}
    missing = _refusal(lambda: _claim(dict(plumbing)))
    # the arm present, and reading where stage 40 found it: pass-through >= treatment.
    with_arm = dict(plumbing)
    with_arm["passthrough"] = Arm(np.full(18, 1.00), expected_null=0.89, tolerance=0.25,
                                  justification="norm-matched pass-through")
    present = _refusal(lambda: _claim(with_arm))
    # a selector whose readout precedes its patch layer must NOT be forced to carry the arm
    before = _refusal(lambda: _claim({"random": Arm(np.full(18, 2.0), expected_null=2.0),
                                      "no_patch": Arm(np.full(18, 2.0), expected_null=2.0),
                                      "permutation": Arm(np.full(18, 2.0), expected_null=2.0)},
                                     instrument="selector", patch_layer=20, readout_layer=14))
    if isinstance(missing, checks.MissingArm) and "passthrough" in str(missing):
        ctl = ["with the arm: constructs" if present is None else f"with the arm: {_named(present)}",
               "readout before patch layer: not required" if before is None
               else f"FAILED: {_named(before)}"]
        return Verdict(8, "readout at or after the patch layer with no pass-through arm", CAUGHT,
                       "Claim -> MissingArm('passthrough'), required because readout_layer >= patch_layer",
                       str(missing), "; ".join(ctl))
    return Verdict(8, "readout at or after the patch layer with no pass-through arm", NOT_CAUGHT,
                   _named(missing), "a stage-14-shaped Claim was accepted without a pass-through arm")


# --------------------------------------------------------------------------------------------

MODEL_CASES = {1: case_1_batch_row_patch, 2: case_2_absolute_spans_left_padding,
               7.5: case_7p5_post_norm_last_hidden_state}
PURE_CASES = {3: case_3_rank1_ties_no_no_patch, 4: case_4_best_layer_on_scoring_data,
              5: case_5_residual_norm_no_floor, 6: case_6_degenerate_crosstalk_rank,
              7: case_7_leaky_grid_raw_score, 8: case_8_readout_after_patch_no_passthrough}


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
