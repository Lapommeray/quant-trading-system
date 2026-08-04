"""
Autonomy package - self-wiring trading organism for OKX live execution.

This is the canonical implementation for post-PR#117 autonomous trading.
It replaces legacy core/qmp_engine_v3 QuantConnect-dependent path.

Modules:
- organism: central self-wiring organism with auto-discovery and self-improvement
- consensus: weighted voting consensus engine
- execution: event-driven execution routing to okx_live
"""

from .organism import Organism, OrganismConfig, ModuleAutoDiscovery
from .consensus import ConsensusEngine, ConsensusResult
from .execution import AutonomousExecutor, ExecutorConfig

__all__ = [
    "Organism",
    "OrganismConfig",
    "ModuleAutoDiscovery",
    "ConsensusEngine",
    "ConsensusResult",
    "AutonomousExecutor",
    "ExecutorConfig",
]
