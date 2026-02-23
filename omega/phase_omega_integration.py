"""Phase Omega integration module."""

from dataclasses import dataclass
from typing import Sequence


@dataclass
class PhaseOmegaIntegrator:
    """Weighted signal combiner for Phase Omega."""

    quantum_weight: float = 0.55
    temporal_weight: float = 0.25
    defense_weight: float = 0.15
    biological_weight: float = 0.05
    mind_weight: float = 0.01

    def compute_master_signal(self, components: Sequence[float]) -> float:
        """Compute weighted score from 5 component values."""
        if len(components) != 5:
            raise ValueError("Expected exactly 5 components: quantum, temporal, defense, biological, mind")
        q, t, d, b, m = components
        return (
            q * self.quantum_weight
            + t * self.temporal_weight
            + d * self.defense_weight
            + b * self.biological_weight
            + m * self.mind_weight
        )
