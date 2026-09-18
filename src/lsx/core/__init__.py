"""`lsx.core`: the verified measurement core (docs/specs/core_v1.md).

Piece 1: the rediscovery harness (§1B), the types (§3) and the one asserted extraction path (§7).
Pieces 2 (instrument registry + calibration) and 3 (ledger, retraction, reproduction) are not here.

Existing `scripts/` are frozen and do not use this package (spec §10).
"""
from . import checks, extract, rediscovery, types
from .checks import CoreError
from .types import (Arm, Claim, Direction, EffectSize, Floor, Grid, Item, Measured, Probe, Readout,
                    Selection, Sketch, Stack)

__all__ = ["checks", "extract", "rediscovery", "types", "CoreError", "Arm", "Claim", "Direction",
           "EffectSize", "Floor", "Grid", "Item", "Measured", "Probe", "Readout", "Selection",
           "Sketch", "Stack"]
