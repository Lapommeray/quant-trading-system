"""
Bayesian Market State - Commitment Accounting with Information Gain

This module implements a Bayesian belief tracking system for market commitment
with KL divergence-based information gain computation. It enables active
epistemic decision-making by quantifying how much each potential trade or
probe would reduce uncertainty about market acceptance probability.

Key Features:
- Beta distribution for belief tracking over acceptance probability
- KL divergence computation for expected information gain
- Monte Carlo sampling for IG estimation
- Evidence decay for regime adaptation
- Integration hooks for regime detection and loss prevention

Mathematical Foundation:
- Prior: θ ~ Beta(α_t, β_t)
- Update: α_{t+1} = α_t + P, β_{t+1} = β_t + (E - P)
- IG = E_{x~q}[D_KL(p(θ|x) || q(θ))]
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from scipy.stats import beta as beta_dist
from scipy.special import digamma, betaln
import logging

logger = logging.getLogger("BayesianMarketState")

try:
    from advanced_modules.microstructure_detector import (
        MicrostructureDetectors, MicrostructureFlags, OrderBookSnapshot
    )
    MICROSTRUCTURE_AVAILABLE = True
except ImportError:
    MICROSTRUCTURE_AVAILABLE = False
    logger.warning("Microstructure detectors not available")


@dataclass
class CommitmentEvent:
    """Represents a market commitment event with energy and persistence"""
    energy: float
    persistence: float
    timestamp: float = 0.0
    data_series: Optional[np.ndarray] = None
    
    @property
    def commitment_ratio(self) -> float:
        """Compute commitment ratio (persistence / energy)"""
        if self.energy <= 0:
            return 0.0
        return np.clip(self.persistence / self.energy, 0.0, 1.0)


@dataclass
class BeliefState:
    """Structured belief state for external consumption"""
    p_accept: float
    confidence: float
    alpha: float
    beta: float
    variance: float
    recent_cr: Optional[float] = None
    expected_ig_bits: float = 0.0
    regime: str = "UNKNOWN"
    mm_flags: Optional[Dict[str, Any]] = None


class BayesianMarketState:
    """
    Bayesian belief tracking for market commitment with information gain.
    
    Maintains a Beta distribution over the acceptance probability θ,
    updated via Bayesian inference from observed energy/persistence events.
    Computes expected information gain for potential probes using KL divergence.
    
    Attributes:
        alpha, beta: Beta distribution parameters
        posterior: scipy.stats.beta distribution object
        p_accept: Mean acceptance probability
        confidence: 1 - variance (higher = more certain)
        commitment_history: Recent commitment ratios
    """
    
    def __init__(self, 
                 history_window: int = 200,
                 energy_threshold: float = 1.5,
                 decay_half_life: float = 100.0,
                 initial_alpha: float = 1.0,
                 initial_beta: float = 1.0):
        """
        Initialize Bayesian market state.
        
        Args:
            history_window: Max length of commitment history
            energy_threshold: Minimum energy for significant events
            decay_half_life: Half-life for evidence decay (in events)
            initial_alpha: Initial Beta alpha (prior successes + 1)
            initial_beta: Initial Beta beta (prior failures + 1)
        """
        self.history_window = history_window
        self.energy_threshold = energy_threshold
        self.decay_half_life = decay_half_life
        
        self.alpha = initial_alpha
        self.beta = initial_beta
        self.posterior = beta_dist(self.alpha, self.beta)
        
        self.p_accept = self.posterior.mean()
        self.confidence = 1.0 - self.posterior.var()
        self.variance = self.posterior.var()
        
        self.commitment_history: deque = deque(maxlen=history_window)
        self.event_count = 0
        
        self.volatility_regime = "UNKNOWN"
        self.expected_ig_bits = 0.0
        
        self._regime_detector = None
        
        self._microstructure_detectors = None
        if MICROSTRUCTURE_AVAILABLE:
            self._microstructure_detectors = MicrostructureDetectors()
            
        self._last_mm_flags: Optional[Dict[str, Any]] = None
        
        logger.info(f"BayesianMarketState initialized: α={self.alpha}, β={self.beta}")
        
    def set_regime_detector(self, detector):
        """Set external regime detector for integration"""
        self._regime_detector = detector
        
    def update_from_event(self, event_data: Dict[str, Any]) -> BeliefState:
        """
        Update belief state from a market event.
        
        Args:
            event_data: Dict with 'energy', 'persistence', optionally 'data_series'
                       For microstructure: 'bid_depth', 'ask_depth', 'adds', 'cancels',
                       'trades', 'volume', 'price_delta'
            
        Returns:
            Updated BeliefState
        """
        E = event_data.get("energy", 1.0)
        P = event_data.get("persistence", 0.5)
        
        P = max(min(P, E), 0)
        
        mm_penalty = 0.0
        self._last_mm_flags = None
        
        if self._microstructure_detectors:
            mm_flags = self._microstructure_detectors.update(event_data)
            self._last_mm_flags = mm_flags.to_dict()
            mm_penalty = mm_flags.get_bayesian_penalty()
            
            if mm_flags.spoof_detected:
                logger.debug(f"SPOOF detected: penalty={mm_penalty:.3f}")
            if mm_flags.absorption_detected:
                logger.debug(f"ABSORPTION detected: penalty={mm_penalty:.3f}")
        
        self.alpha += P - mm_penalty
        self.beta += (E - P) + mm_penalty
        
        self.alpha = max(self.alpha, 0.1)
        self.beta = max(self.beta, 0.1)
        
        self.posterior = beta_dist(self.alpha, self.beta)
        self.p_accept = self.posterior.mean()
        self.variance = self.posterior.var()
        self.confidence = 1.0 - self.variance
        
        CR = P / max(E, 1e-6)
        CR = np.clip(CR, 0, 1)
        self.commitment_history.append(CR)
        self.event_count += 1
        
        self.apply_decay()
        
        if self._regime_detector and "data_series" in event_data:
            self.volatility_regime = self._regime_detector.detect(
                event_data["data_series"],
                p_accept=self.p_accept,
                confidence=self.confidence
            )
            
        return self.get_state()
        
    def apply_decay(self):
        """Apply evidence decay to favor recent market behavior"""
        if self.decay_half_life <= 0:
            return
            
        decay_factor = np.exp(-np.log(2) / self.decay_half_life)
        
        min_param = 1.0
        self.alpha = max(self.alpha * decay_factor, min_param)
        self.beta = max(self.beta * decay_factor, min_param)
        
        self.posterior = beta_dist(self.alpha, self.beta)
        
    def sample_acceptance(self, n_samples: int = 1) -> np.ndarray:
        """Sample from the posterior acceptance distribution"""
        return self.posterior.rvs(size=n_samples)
        
    def compute_kl_divergence(self, 
                               alpha_p: float, beta_p: float,
                               alpha_q: float, beta_q: float) -> float:
        """
        Compute KL divergence between two Beta distributions.
        
        D_KL(Beta(α_p, β_p) || Beta(α_q, β_q))
        
        Uses closed-form formula for Beta distributions.
        
        Args:
            alpha_p, beta_p: Parameters of distribution P (posterior)
            alpha_q, beta_q: Parameters of distribution Q (prior)
            
        Returns:
            KL divergence in nats (>= 0)
        """
        kl = (betaln(alpha_q, beta_q) - betaln(alpha_p, beta_p) +
              (alpha_p - alpha_q) * digamma(alpha_p) +
              (beta_p - beta_q) * digamma(beta_p) +
              (alpha_q + beta_q - alpha_p - beta_p) * digamma(alpha_p + beta_p))
        
        return max(kl, 0.0)
        
    def expected_info_gain(self, 
                           hypothetical_E: float = 1.0,
                           n_samples: int = 100) -> float:
        """
        Estimate expected information gain for a potential probe.
        
        Computes E_{x~q}[D_KL(p(θ|x) || q(θ))] using Monte Carlo sampling.
        
        Args:
            hypothetical_E: Expected energy of the probe
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Expected information gain in nats
        """
        p_samples = self.sample_acceptance(n_samples)
        p_outcomes = p_samples * hypothetical_E
        
        ig = 0.0
        for p_outcome in p_outcomes:
            new_alpha = self.alpha + p_outcome
            new_beta = self.beta + (hypothetical_E - p_outcome)
            
            kl = self.compute_kl_divergence(
                new_alpha, new_beta,
                self.alpha, self.beta
            )
            ig += kl
            
        ig /= n_samples
        
        self.expected_ig_bits = ig / np.log(2)
        
        return ig
        
    def expected_info_gain_bits(self, 
                                 hypothetical_E: float = 1.0,
                                 n_samples: int = 100) -> float:
        """Compute expected information gain in bits (more intuitive)"""
        ig_nats = self.expected_info_gain(hypothetical_E, n_samples)
        return ig_nats / np.log(2)
        
    def get_state(self) -> BeliefState:
        """Return structured belief state"""
        recent_cr = self.commitment_history[-1] if self.commitment_history else None
        
        return BeliefState(
            p_accept=self.p_accept,
            confidence=self.confidence,
            alpha=self.alpha,
            beta=self.beta,
            variance=self.variance,
            recent_cr=recent_cr,
            expected_ig_bits=self.expected_ig_bits,
            regime=self.volatility_regime,
            mm_flags=self._last_mm_flags
        )
        
    def get_state_dict(self) -> Dict[str, Any]:
        """Return belief state as dictionary"""
        state = self.get_state()
        return {
            "p_accept": state.p_accept,
            "confidence": state.confidence,
            "alpha": state.alpha,
            "beta": state.beta,
            "variance": state.variance,
            "recent_cr": state.recent_cr,
            "expected_ig_bits": state.expected_ig_bits,
            "regime": state.regime,
            "mm_flags": state.mm_flags
        }
        
    def reset(self, alpha: float = 1.0, beta: float = 1.0):
        """Reset belief state to prior"""
        self.alpha = alpha
        self.beta = beta
        self.posterior = beta_dist(self.alpha, self.beta)
        self.p_accept = self.posterior.mean()
        self.variance = self.posterior.var()
        self.confidence = 1.0 - self.variance
        self.commitment_history.clear()
        self.event_count = 0
        self.expected_ig_bits = 0.0
        
    def get_exploration_recommendation(self, 
                                        ig_threshold: float = 0.1,
                                        confidence_threshold: float = 0.5) -> str:
        """
        Get exploration/exploitation recommendation based on current state.
        
        Args:
            ig_threshold: IG threshold for exploration (in nats)
            confidence_threshold: Confidence threshold for exploitation
            
        Returns:
            "EXPLORE", "EXPLOIT", or "HOLD"
        """
        ig = self.expected_info_gain(hypothetical_E=1.0, n_samples=50)
        
        if ig > ig_threshold and self.confidence < confidence_threshold:
            return "EXPLORE"
        elif self.p_accept > 0.5 and self.confidence > confidence_threshold:
            return "EXPLOIT"
        elif self.p_accept < 0.5 and self.confidence > confidence_threshold:
            return "HOLD"
        else:
            return "EXPLORE"
            
    def compute_posterior_predictive(self, n_samples: int = 1000) -> Tuple[float, float]:
        """
        Compute posterior predictive distribution statistics.
        
        Returns:
            Tuple of (mean, std) for next observation
        """
        samples = self.sample_acceptance(n_samples)
        return float(np.mean(samples)), float(np.std(samples))


class CommitmentAccountingSystem:
    """
    Commitment accounting system that tracks market commitment
    and determines trade eligibility based on Bayesian beliefs.
    """
    
    def __init__(self, 
                 max_commitment_ratio: float = 0.8,
                 min_confidence: float = 0.3):
        """
        Initialize commitment accounting.
        
        Args:
            max_commitment_ratio: Maximum allowed commitment ratio
            min_confidence: Minimum confidence for trade approval
        """
        self.max_commitment_ratio = max_commitment_ratio
        self.min_confidence = min_confidence
        self.total_committed = 0.0
        self.total_capacity = 1.0
        
    def can_commit(self, 
                   signal: Dict[str, Any],
                   p_accept: float) -> bool:
        """
        Check if a signal can be committed based on current state.
        
        Args:
            signal: Signal dict with 'size' and optionally 'risk'
            p_accept: Current acceptance probability
            
        Returns:
            True if commitment is allowed
        """
        size = signal.get("size", 0.1)
        risk = signal.get("risk", size)
        
        new_commitment = self.total_committed + risk
        commitment_ratio = new_commitment / max(self.total_capacity, 1e-6)
        
        if commitment_ratio > self.max_commitment_ratio:
            return False
            
        if p_accept < 0.3:
            return False
            
        return True
        
    def commit(self, signal: Dict[str, Any]) -> bool:
        """Record a commitment"""
        risk = signal.get("risk", signal.get("size", 0.1))
        self.total_committed += risk
        return True
        
    def release(self, amount: float):
        """Release committed capacity"""
        self.total_committed = max(0, self.total_committed - amount)
        
    def reset(self):
        """Reset commitment tracking"""
        self.total_committed = 0.0


def demo():
    """Demonstrate BayesianMarketState functionality"""
    print("=" * 60)
    print("BAYESIAN MARKET STATE DEMO")
    print("=" * 60)
    
    state = BayesianMarketState()
    
    print("\nInitial State:")
    print(f"  α={state.alpha:.2f}, β={state.beta:.2f}")
    print(f"  p_accept={state.p_accept:.3f}, confidence={state.confidence:.3f}")
    
    print("\n--- Simulating market events ---")
    
    events = [
        {"energy": 1.0, "persistence": 0.8},
        {"energy": 1.0, "persistence": 0.7},
        {"energy": 1.0, "persistence": 0.6},
        {"energy": 1.0, "persistence": 0.9},
        {"energy": 1.0, "persistence": 0.5},
    ]
    
    for i, event in enumerate(events):
        belief = state.update_from_event(event)
        ig = state.expected_info_gain(hypothetical_E=1.0)
        print(f"\nEvent {i+1}: E={event['energy']}, P={event['persistence']}")
        print(f"  α={belief.alpha:.2f}, β={belief.beta:.2f}")
        print(f"  p_accept={belief.p_accept:.3f}, confidence={belief.confidence:.3f}")
        print(f"  Expected IG: {ig:.4f} nats ({ig/np.log(2):.4f} bits)")
        print(f"  Recommendation: {state.get_exploration_recommendation()}")
        
    print("\n--- Information Gain Analysis ---")
    
    test_states = [
        (1.0, 1.0, "Very uncertain"),
        (5.0, 2.0, "Moderately confident"),
        (50.0, 10.0, "Highly confident"),
    ]
    
    print("\n| State | α | β | Mean | Var | IG (bits) |")
    print("|-------|---|---|------|-----|-----------|")
    
    for alpha, beta_val, desc in test_states:
        test_state = BayesianMarketState(initial_alpha=alpha, initial_beta=beta_val)
        ig_bits = test_state.expected_info_gain_bits(hypothetical_E=1.0)
        print(f"| {desc:20s} | {alpha:3.0f} | {beta_val:3.0f} | "
              f"{test_state.p_accept:.2f} | {test_state.variance:.4f} | {ig_bits:.4f} |")
              
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo()
