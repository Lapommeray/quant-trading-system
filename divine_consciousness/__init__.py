"""
Divine Consciousness package for the Quant Trading System.

Re-exports DivineConsciousness from the implementation in scripts/.
"""

import importlib.util
from pathlib import Path

_impl_path = Path(__file__).parent.parent / "scripts" / "divine_consciousness.py"
_spec = importlib.util.spec_from_file_location(
    "divine_consciousness._impl",
    str(_impl_path),
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

DivineConsciousness = _impl.DivineConsciousness

__all__ = ["DivineConsciousness"]
