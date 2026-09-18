"""§9: the ledger and retraction.

A `Claim` is the only exportable type; `results/ledger.jsonl` is the only place a claim becomes a
*result*. This module is therefore the last gate, and it holds the three refusals pieces 1 and 2
each identified and each left open, because each of them is about publication rather than about
whether the arithmetic is right:

1. **A hand-declared calibration report cannot be graded.** Piece 2 stamped
   `CalibrationReport.hand_declared(...)` so it could not pass for a measured one; nothing acted on
   the stamp. Here it is refused. (Piece 1's harness cases 3, 4 and 7 use hand-declared reports on
   purpose -- they are demonstrations that a mechanism fires, not results, and they never reach
   this file.)

2. **A row whose provenance did not come from a real `Stack` cannot be published.** Every §7
   assertion guards `extract.build_stack`; until now nothing tied a `Claim` to it, so a hand-rolled
   extraction could be wrapped in a well-formed `Claim`. `build_stack` now stamps a signature over
   its own provenance fields and the activation digest, and `append` recomputes it. This is not a
   security boundary and is not meant as one: it means an unasserted extraction cannot reach the
   ledger by accident, and cannot reach it at all without someone writing the forgery on purpose.

3. **A swept axis must carry the curve the core computed, not the caller's prose about it.** Piece
   2 built `Instrument.sweep` and explicitly handed this decision to piece 3. It is taken: at the
   ledger, `selection.executed` is MANDATORY whenever `selection.axis` is not None. See
   `results/notes/core_p3.md` for what it cost.

Retraction is a first-class operation (§9). `withdraw(id, reason, superseded_by)` appends a new
line marking the claim withdrawn; the file is append-only, so the history of a retraction is itself
in the record, and `rows()` folds it to the current state. In a project with five retractions this
is the main feature.
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .checks import (HandDeclaredCalibration, LedgerConflict, ProvenanceIncomplete,
                     ProvenanceNotFromStack, SweepNotExecuted)
from .types import Claim, stack_signature

DEFAULT_PATH = pathlib.Path(__file__).resolve().parents[3] / "results" / "ledger.jsonl"

# §9's row shape names these in `provenance`. `template` may legitimately be None (a base model has
# no chat template) and `lib_versions.nnsight` may be None locally; the KEY must still be there.
REQUIRED_PROVENANCE = ("grid_hash", "model", "layers", "pooling", "template", "tokenizer_padding",
                       "code_version", "lib_versions", "calibration_key")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------------------------
# the three publication refusals
# --------------------------------------------------------------------------------------------
def check_calibration_measured(claim: Claim) -> None:
    if claim.calibration.hand_declared_report:
        raise HandDeclaredCalibration(
            f"{claim.instrument}'s calibration report is HAND-DECLARED "
            f"({'; '.join(claim.calibration.notes) or 'no note'}). A hand-declared report is a "
            "promise that the "
            "battery would pass, and the ledger grades numbers, not promises. Build the instrument "
            "through `instruments.build(...)` so `calibrate()` runs the §5 battery, or do not "
            "publish the row.")


def check_provenance_from_stack(claim: Claim) -> None:
    prov = claim.provenance
    missing = [k for k in REQUIRED_PROVENANCE if k not in prov]
    if missing:
        raise ProvenanceIncomplete(
            f"ledger row for {claim.instrument} is missing provenance {missing}; §9 names these so "
            "a changed grid or library version produces a NEW id rather than silently overwriting")
    sig = prov.get("stack_signature")
    if not sig:
        raise ProvenanceNotFromStack(
            f"the {claim.instrument} claim carries no `stack_signature`: its provenance was not "
            "written by `extract.build_stack`, so none of the §7 assertions -- padding convention, "
            "batched-vs-single equivalence on the shortest item, non-empty spans, the `resid()` "
            "helper -- is known to have run on the vectors behind it. Pieces 1 and 2 both named "
            "this as the last structural hole; it is closed here.")
    recomputed = stack_signature(prov)
    if sig != recomputed:
        raise ProvenanceNotFromStack(
            f"the stack signature does not match the provenance it is attached to "
            f"({sig} recorded, {recomputed} recomputed). A field the extraction wrote was changed "
            "afterwards -- the model, the padding side, the library versions or the activation "
            "digest. That is the case the signature exists for.")


def check_sweep_executed(claim: Claim) -> None:
    sel = claim.selection
    if sel.axis is not None and not sel.executed:
        raise SweepNotExecuted(
            f"the {claim.instrument} claim declares a selection axis {sel.axis!r} with no curve: "
            f"rule={sel.rule!r}. That rule is prose, and the prose check is a regex a caller can "
            "walk past ('we looked at the curve and quoted layer 16' passes it). Run the sweep "
            "through `Instrument.sweep(axis, values, score)` so the curve is a fact the core "
            "recorded, or set axis=None for a claim that swept nothing.")


REFUSALS = (check_calibration_measured, check_provenance_from_stack, check_sweep_executed)


def screen(claim: Claim) -> None:
    """Run every publication refusal. Raises the first that fires."""
    for fn in REFUSALS:
        fn(claim)


def would_refuse(claim: Claim) -> str | None:
    """Non-raising form, for the reproduction suite's report: the refusal's name, or None."""
    try:
        screen(claim)
    except Exception as e:  # noqa: BLE001 -- the caller wants whatever came out
        return f"{type(e).__name__}: {str(e).splitlines()[0]}"
    return None


# --------------------------------------------------------------------------------------------
# the ledger file
# --------------------------------------------------------------------------------------------
@dataclass
class Ledger:
    path: pathlib.Path = field(default_factory=lambda: DEFAULT_PATH)

    def __post_init__(self):
        self.path = pathlib.Path(self.path)

    # ---- reading ---------------------------------------------------------------------
    def lines(self) -> Iterator[dict]:
        if not self.path.exists():
            return iter(())
        return (json.loads(ln) for ln in self.path.read_text().splitlines() if ln.strip())

    def rows(self) -> dict[str, dict]:
        """Current state of every claim, folding the append-only history.

        A `withdraw` line carries only the id and the retraction; it is folded onto the row it
        retracts so that `status` and `withdrawn` are always read from one place.
        """
        out: dict[str, dict] = {}
        for ln in self.lines():
            i = ln["id"]
            if ln.get("_op") == "withdraw":
                if i not in out:
                    raise LedgerConflict(f"withdrawal of {i}, which is not in the ledger")
                out[i] = dict(out[i], status="withdrawn", withdrawn=ln["withdrawn"])
            else:
                keep = out.get(i, {}).get("withdrawn")
                out[i] = dict(ln)
                if keep and ln.get("status") != "standing":
                    out[i]["withdrawn"] = keep
        return out

    def get(self, claim_id: str) -> dict | None:
        return self.rows().get(claim_id)

    def standing(self) -> list[dict]:
        return [r for r in self.rows().values() if r.get("status") == "standing"]

    def withdrawn(self) -> list[dict]:
        return [r for r in self.rows().values() if r.get("status") == "withdrawn"]

    # ---- writing ---------------------------------------------------------------------
    def _append_line(self, obj: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")

    def append(self, claim: Claim, *, note: str = "", reproduces: str = "",
               logged: dict | None = None, screen_claim: bool = True) -> dict:
        """Publish a claim. Every refusal above runs first.

        `id` is the provenance hash, so the same experiment re-run is RECOGNISED rather than
        duplicated -- and a changed grid, code version or library version produces a different id
        rather than silently overwriting the old row (§9).
        """
        if screen_claim:
            screen(claim)
        row = claim.to_row()
        row["at"] = _now()
        if note:
            row["note"] = note
        if reproduces:
            row["reproduces"] = reproduces
        if logged is not None:
            row["logged"] = logged
        existing = self.rows().get(row["id"])
        if existing is not None:
            differs = {k for k in ("treatment", "reported", "arms", "floor", "effect")
                       if json.dumps(existing.get(k), sort_keys=True, default=str)
                       != json.dumps(row.get(k), sort_keys=True, default=str)}
            if differs:
                raise LedgerConflict(
                    f"id {row['id']} is already in the ledger with different {sorted(differs)}. The "
                    "id is the provenance hash: identical provenance and a different number means "
                    "the run is not reproducible, not that the row should be overwritten.")
            return existing
        self._append_line(row)
        return row

    def withdraw(self, claim_id: str, reason: str, superseded_by: str | None = None) -> dict:
        """Retraction, first-class (§9). Appends rather than edits: the record keeps the fact that
        the claim once stood, which is the whole point in a project with five retractions."""
        rows = self.rows()
        if claim_id not in rows:
            raise LedgerConflict(f"cannot withdraw {claim_id}: not in {self.path}")
        if not reason:
            raise LedgerConflict("a withdrawal needs a reason; every generated document prints it "
                                 "inline beside the claim")
        w = {"reason": reason, "superseded_by": superseded_by, "at": _now()}
        self._append_line({"id": claim_id, "_op": "withdraw", "withdrawn": w})
        return dict(rows[claim_id], status="withdrawn", withdrawn=w)

    # ---- generated document ----------------------------------------------------------
    def render(self, ids: Iterable[str] | None = None) -> str:
        """`RESULTS.md` hour entries are GENERATED from the ledger (§9). This is that renderer:
        every withdrawn claim regenerates showing its reason inline, so five retractions cannot
        drift out of step with five prose paragraphs in four files."""
        rows = self.rows()
        keys = list(ids) if ids is not None else list(rows)
        out = []
        for k in keys:
            r = rows.get(k)
            if r is None:
                out.append(f"- [{k}] MISSING from the ledger")
                continue
            head = (f"- **{r.get('stage') or r['instrument']}** [{k}] {r['instrument']}: "
                    f"{r['report_as']} {r['reported']:.4f} (raw {r['treatment']:.4f})")
            arms = "; ".join(f"{n} {a['value']:.3f}/{a['expected_null']:.3f}"
                             + ("!" if a["off_null"] else "")
                             for n, a in sorted(r["arms"].items()))
            line = (f"{head}\n    arms: {arms}\n    floor: stimulus "
                    f"{r['floor']['stimulus']} / estimator {r['floor']['estimator']}; "
                    f"effect {r['effect']['size']:+.3f} (n={r['effect']['n']}, "
                    f"z={r['effect']['z']:+.2f})")
            if r.get("logged"):
                line += f"\n    logged: {json.dumps(r['logged'], sort_keys=True)}"
            if r.get("status") == "withdrawn":
                w = r["withdrawn"]
                line += (f"\n    **WITHDRAWN** ({w['at']}): {w['reason']}"
                         + (f" superseded by {w['superseded_by']}" if w.get("superseded_by") else ""))
            out.append(line)
        return "\n".join(out)
