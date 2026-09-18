"""`lsx.core`: the verified measurement core (docs/specs/core_v1.md).

Piece 1: the rediscovery harness (§1B), the types (§3) and the one asserted extraction path (§7).
Piece 2: the instrument registry (§8), the calibration battery (§5) and the three instruments
`selector`, `composition` and `readout_shift`. Piece 3 (ledger, retraction, reproduction) is not
here. Importing this package imports `instruments`, which publishes each built instrument's
calibration key so that `Claim` can refuse a stale measured report.

Existing `scripts/` are frozen and do not use this package (spec §10).
"""
from . import checks, extract, instruments, planted, rediscovery, registry, types
from .checks import CalibrationReport, CoreError
from .instruments import Instrument, PassthroughArm
from .registry import REGISTRY, InstrumentSpec
from .types import (Arm, Claim, Direction, EffectSize, Floor, Grid, Item, Measured, Probe, Readout,
                    Selection, Sketch, Stack, fit_leave_one_out)

__all__ = ["checks", "extract", "instruments", "planted", "rediscovery", "registry", "types",
           "CalibrationReport", "CoreError", "Instrument", "InstrumentSpec", "PassthroughArm",
           "REGISTRY", "Arm", "Claim", "Direction", "EffectSize", "Floor", "Grid", "Item",
           "Measured", "Probe", "Readout", "Selection", "Sketch", "Stack", "fit_leave_one_out"]
