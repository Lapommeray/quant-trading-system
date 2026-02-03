"""
Trade Scheduler - Utility-Information Frontier with UCB Regret Minimization

This module implements a sophisticated trade scheduling system that ranks
candidate signals based on a combined utility score balancing:
- Expected profit
- Information gain (exploration value)
- Risk (drawdown/variance penalty)
- UCB bonus (regret minimization)

The scheduler enables active planning by ranking probes/trades on a
Utility-Information frontier, ensuring optimal balance between
exploitation (profitable trades) and exploration (uncertainty-resolving probes).

Mathematical Foundation:
U_total(s) = E[Profit_s] + λ_IG * IG(E_s) - λ_risk * R_s + λ_UCB * UCB_s

Where:
- E[Profit_s]: Expected profit from signal
- IG(E_s): Expected information gain
- R_s: Risk measure (VaR, variance, etc.)
- UCB_s: Upper confidence bound bonus for exploration
"""

import heapq
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import time

logger = logging.getLogger("TradeScheduler")

try:
    from .bayesian_market_state import BayesianMarketState, BeliefState
except ImportError:
    from bayesian_market_state import BayesianMarketState, BeliefState


@dataclass
class Signal:
    """Represents a trading signal/probe candidate"""
    id: str
    expected_profit: float
    expected_energy: float
    risk: float
    size: float = 0.1
    mode: str = "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def __hash__(self):
        return hash(self.id)
        
    def __eq__(self, other):
        if isinstance(other, Signal):
            return self.id == other.id
        return False


@dataclass
class SignalStats:
    """Statistics for a signal type (for UCB computation)"""
    n: int = 0
    mean_profit: float = 0.0
    variance: float = 0.0
    last_update: float = 0.0
    
    def update(self, realized_profit: float):
        """Update running statistics with new observation"""
        self.n += 1
        delta = realized_profit - self.mean_profit
        self.mean_profit += delta / self.n
        if self.n > 1:
            self.variance += delta * (realized_profit - self.mean_profit)
        self.last_update = time.time()
        
    def get_std(self) -> float:
        """Get standard deviation"""
        if self.n < 2:
            return float('inf')
        return math.sqrt(self.variance / (self.n - 1))


@dataclass
class ScheduledSignal:
    """A signal with computed utility score"""
    signal: Signal
    u_total: float
    ig: float
    ucb_bonus: float
    components: Dict[str, float] = field(default_factory=dict)


class UtilityFrontierScheduler:
    """
    Trade scheduler using Utility-Information frontier with UCB.
    
    Ranks candidate signals by combined utility score that balances
    expected profit, information gain, risk, and exploration bonus.
    
    Attributes:
        lambda_ig: Weight for information gain term
        lambda_risk: Weight for risk penalty term
        lambda_ucb: Weight for UCB exploration bonus
        market_state: BayesianMarketState for IG computation
        stats: Per-signal statistics for UCB
        global_eval_count: Total evaluations for UCB scaling
    """
    
    def __init__(self,
                 lambda_ig: float = 0.2,
                 lambda_risk: float = 1.0,
                 lambda_ucb: float = 0.1,
                 max_queue: int = 50,
                 exploration_budget: float = 0.05,
                 market_state: Optional[BayesianMarketState] = None):
        """
        Initialize the scheduler.
        
        Args:
            lambda_ig: Weight for information gain (default 0.2)
            lambda_risk: Weight for risk penalty (default 1.0)
            lambda_ucb: Weight for UCB bonus (default 0.1)
            max_queue: Maximum queue size
            exploration_budget: Max fraction of equity for exploration trades
            market_state: Optional shared BayesianMarketState
        """
        self.lambda_ig = lambda_ig
        self.lambda_risk = lambda_risk
        self.lambda_ucb = lambda_ucb
        self.max_queue = max_queue
        self.exploration_budget = exploration_budget
        
        self.market_state = market_state or BayesianMarketState()
        
        self.stats: Dict[str, SignalStats] = defaultdict(SignalStats)
        self.global_eval_count = 0
        
        self.candidate_queue: List[Tuple[float, ScheduledSignal]] = []
        
        self.exploration_used = 0.0
        
        self.metrics = {
            "total_scheduled": 0,
            "total_executed": 0,
            "avg_u_total": 0.0,
            "avg_ig": 0.0,
            "exploration_trades": 0,
            "exploitation_trades": 0,
        }
        
        logger.info(f"UtilityFrontierScheduler initialized: "
                   f"λ_IG={lambda_ig}, λ_risk={lambda_risk}, λ_UCB={lambda_ucb}")
                   
    def ucb_bonus(self, signal_id: str) -> float:
        """
        Compute UCB exploration bonus for a signal.
        
        UCB_s = λ_UCB * sqrt(ln(T) / (n_s + 1))
        
        Args:
            signal_id: Signal identifier
            
        Returns:
            UCB bonus value
        """
        if signal_id not in self.stats:
            return self.lambda_ucb
            
        n = self.stats[signal_id].n
        if self.global_eval_count == 0:
            return self.lambda_ucb
            
        return self.lambda_ucb * math.sqrt(
            math.log(self.global_eval_count + 1) / (n + 1)
        )
        
    def compute_utility(self, signal: Signal) -> ScheduledSignal:
        """
        Compute total utility score for a signal.
        
        U_total = E[Profit] + λ_IG * IG - λ_risk * Risk + UCB
        
        Args:
            signal: Signal to evaluate
            
        Returns:
            ScheduledSignal with computed utility
        """
        ig = self.market_state.expected_info_gain(signal.expected_energy)
        
        ucb = self.ucb_bonus(signal.id)
        
        u_total = (
            signal.expected_profit +
            self.lambda_ig * ig -
            self.lambda_risk * signal.risk +
            ucb
        )
        
        components = {
            "profit": signal.expected_profit,
            "ig_term": self.lambda_ig * ig,
            "risk_term": -self.lambda_risk * signal.risk,
            "ucb_term": ucb,
        }
        
        return ScheduledSignal(
            signal=signal,
            u_total=u_total,
            ig=ig,
            ucb_bonus=ucb,
            components=components
        )
        
    def add_candidate(self, signal: Signal) -> float:
        """
        Add a candidate signal to the queue.
        
        Args:
            signal: Signal to add
            
        Returns:
            Computed utility score
        """
        scheduled = self.compute_utility(signal)
        
        heapq.heappush(
            self.candidate_queue,
            (-scheduled.u_total, scheduled)
        )
        
        if len(self.candidate_queue) > self.max_queue:
            heapq.heappop(self.candidate_queue)
            
        self.metrics["total_scheduled"] += 1
        
        return scheduled.u_total
        
    def add_candidate_dict(self, signal_dict: Dict[str, Any]) -> float:
        """
        Add a candidate from a dictionary.
        
        Args:
            signal_dict: Dict with 'id', 'expected_profit', 'expected_energy', 'risk'
            
        Returns:
            Computed utility score
        """
        signal = Signal(
            id=signal_dict.get("id", f"signal_{time.time()}"),
            expected_profit=signal_dict.get("expected_profit", 0.0),
            expected_energy=signal_dict.get("expected_energy", 1.0),
            risk=signal_dict.get("risk", 0.1),
            size=signal_dict.get("size", 0.1),
            metadata=signal_dict.get("metadata", {})
        )
        return self.add_candidate(signal)
        
    def schedule_next(self, n: int = 1) -> List[ScheduledSignal]:
        """
        Pop top n candidates for execution.
        
        Args:
            n: Number of signals to schedule
            
        Returns:
            List of ScheduledSignal objects
        """
        scheduled = []
        
        for _ in range(min(n, len(self.candidate_queue))):
            _, sched_signal = heapq.heappop(self.candidate_queue)
            
            belief = self.market_state.get_state()
            if (sched_signal.ig > 0.1 and belief.confidence < 0.6):
                if self.exploration_used < self.exploration_budget:
                    sched_signal.signal.mode = "exploration"
                    sched_signal.signal.size *= 0.1
                    self.exploration_used += sched_signal.signal.size
                    self.metrics["exploration_trades"] += 1
                else:
                    continue
            else:
                self.metrics["exploitation_trades"] += 1
                
            scheduled.append(sched_signal)
            self.metrics["total_executed"] += 1
            
        return scheduled
        
    def record_outcome(self, signal_id: str, realized_profit: float):
        """
        Record the outcome of an executed signal.
        
        Args:
            signal_id: Signal identifier
            realized_profit: Actual profit realized
        """
        self.stats[signal_id].update(realized_profit)
        self.global_eval_count += 1
        
        n = self.metrics["total_executed"]
        if n > 0:
            self.metrics["avg_u_total"] = (
                (self.metrics["avg_u_total"] * (n - 1) + realized_profit) / n
            )
            
    def update_lambdas(self, 
                       lambda_ig: Optional[float] = None,
                       lambda_risk: Optional[float] = None,
                       lambda_ucb: Optional[float] = None):
        """
        Update weighting parameters (for evolution agent tuning).
        
        Args:
            lambda_ig: New IG weight
            lambda_risk: New risk weight
            lambda_ucb: New UCB weight
        """
        if lambda_ig is not None:
            self.lambda_ig = lambda_ig
        if lambda_risk is not None:
            self.lambda_risk = lambda_risk
        if lambda_ucb is not None:
            self.lambda_ucb = lambda_ucb
            
        logger.info(f"Updated lambdas: IG={self.lambda_ig}, "
                   f"risk={self.lambda_risk}, UCB={self.lambda_ucb}")
                   
    def decay_stats(self, decay_factor: float = 0.9):
        """
        Decay old statistics to favor recent observations.
        
        Args:
            decay_factor: Multiplicative decay (0.9 = 10% decay)
        """
        for signal_id in self.stats:
            self.stats[signal_id].n = int(self.stats[signal_id].n * decay_factor)
            
    def reset_exploration_budget(self):
        """Reset exploration budget for new cycle"""
        self.exploration_used = 0.0
        
    def clear_queue(self):
        """Clear the candidate queue"""
        self.candidate_queue = []
        
    def get_queue_summary(self) -> Dict[str, Any]:
        """Get summary of current queue state"""
        if not self.candidate_queue:
            return {"size": 0, "top_utility": None, "avg_utility": None}
            
        utilities = [-u for u, _ in self.candidate_queue]
        return {
            "size": len(self.candidate_queue),
            "top_utility": max(utilities),
            "avg_utility": np.mean(utilities),
            "min_utility": min(utilities),
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics"""
        return {
            **self.metrics,
            "queue_size": len(self.candidate_queue),
            "unique_signals": len(self.stats),
            "global_evals": self.global_eval_count,
            "exploration_used": self.exploration_used,
        }
        
    def compute_regret(self, 
                       best_possible_utility: float,
                       chosen_utility: float) -> float:
        """
        Compute instantaneous regret.
        
        Args:
            best_possible_utility: Best achievable utility
            chosen_utility: Actually chosen utility
            
        Returns:
            Regret value (>= 0)
        """
        return max(0, best_possible_utility - chosen_utility)
        
    def rank_candidates(self, 
                        candidates: List[Signal],
                        top_k: int = 10) -> List[ScheduledSignal]:
        """
        Rank a list of candidates without adding to queue.
        
        Args:
            candidates: List of signals to rank
            top_k: Number of top candidates to return
            
        Returns:
            Top k ScheduledSignal objects sorted by utility
        """
        scheduled = [self.compute_utility(c) for c in candidates]
        scheduled.sort(key=lambda x: x.u_total, reverse=True)
        return scheduled[:top_k]


class AdaptiveScheduler(UtilityFrontierScheduler):
    """
    Adaptive scheduler that auto-tunes lambda parameters based on performance.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_history: List[float] = []
        self.lambda_history: List[Dict[str, float]] = []
        self.adaptation_interval = 100
        
    def adapt_lambdas(self):
        """Adapt lambda parameters based on recent performance"""
        if len(self.performance_history) < self.adaptation_interval:
            return
            
        recent = self.performance_history[-self.adaptation_interval:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend < 0:
            self.lambda_ig *= 1.1
            self.lambda_ucb *= 1.1
        else:
            self.lambda_ig *= 0.95
            self.lambda_ucb *= 0.95
            
        self.lambda_ig = np.clip(self.lambda_ig, 0.05, 0.5)
        self.lambda_ucb = np.clip(self.lambda_ucb, 0.01, 0.3)
        
        self.lambda_history.append({
            "lambda_ig": self.lambda_ig,
            "lambda_risk": self.lambda_risk,
            "lambda_ucb": self.lambda_ucb,
            "trend": trend,
        })
        
    def record_outcome(self, signal_id: str, realized_profit: float):
        """Record outcome and trigger adaptation"""
        super().record_outcome(signal_id, realized_profit)
        self.performance_history.append(realized_profit)
        
        if len(self.performance_history) % self.adaptation_interval == 0:
            self.adapt_lambdas()


def demo():
    """Demonstrate UtilityFrontierScheduler functionality"""
    print("=" * 60)
    print("UTILITY FRONTIER SCHEDULER DEMO")
    print("=" * 60)
    
    scheduler = UtilityFrontierScheduler(
        lambda_ig=0.2,
        lambda_risk=1.0,
        lambda_ucb=0.1
    )
    
    scheduler.market_state.alpha = 5.0
    scheduler.market_state.beta = 2.0
    scheduler.market_state.posterior = scheduler.market_state.posterior
    
    print("\nMarket State:")
    belief = scheduler.market_state.get_state()
    print(f"  p_accept={belief.p_accept:.3f}, confidence={belief.confidence:.3f}")
    
    candidates = [
        Signal("probe_A", expected_profit=0.1, expected_energy=1.0, risk=0.05),
        Signal("signal_B", expected_profit=0.5, expected_energy=1.0, risk=0.1),
        Signal("trade_C", expected_profit=0.8, expected_energy=1.0, risk=0.2),
        Signal("risky_D", expected_profit=1.0, expected_energy=1.0, risk=0.5),
        Signal("safe_E", expected_profit=0.3, expected_energy=1.0, risk=0.02),
    ]
    
    print("\n--- Adding Candidates ---")
    print("\n| Signal | E[Profit] | Risk | IG | UCB | U_total |")
    print("|--------|-----------|------|-----|-----|---------|")
    
    for signal in candidates:
        u_total = scheduler.add_candidate(signal)
        sched = scheduler.compute_utility(signal)
        print(f"| {signal.id:8s} | {signal.expected_profit:9.2f} | "
              f"{signal.risk:4.2f} | {sched.ig:.3f} | {sched.ucb_bonus:.3f} | "
              f"{u_total:7.3f} |")
              
    print("\n--- Scheduling Top 3 ---")
    scheduled = scheduler.schedule_next(3)
    
    for i, sched in enumerate(scheduled):
        print(f"\n{i+1}. {sched.signal.id}")
        print(f"   U_total: {sched.u_total:.3f}")
        print(f"   Mode: {sched.signal.mode}")
        print(f"   Components: {sched.components}")
        
    print("\n--- Recording Outcomes ---")
    scheduler.record_outcome("signal_B", 0.45)
    scheduler.record_outcome("trade_C", 0.75)
    scheduler.record_outcome("safe_E", 0.35)
    
    print("\nMetrics:")
    for k, v in scheduler.get_metrics().items():
        print(f"  {k}: {v}")
        
    print("\n--- UCB After Updates ---")
    for signal in candidates:
        ucb = scheduler.ucb_bonus(signal.id)
        n = scheduler.stats[signal.id].n if signal.id in scheduler.stats else 0
        print(f"  {signal.id}: UCB={ucb:.4f}, n={n}")
        
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo()
