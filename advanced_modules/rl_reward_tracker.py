"""
RL Reward Tracker - Centralized Reward Computation for Policy Learning

This module provides a centralized place to compute and log the composite reward
for the RL evolver agent. The reward balances profit-seeking with information gain
and penalizes drawdowns.

Reward Formula:
    reward = pnl_delta + λ_ig * realized_ig - λ_risk * drawdown

The tracker maintains rolling windows of each component for stable reward estimation
and provides summary statistics for monitoring learning progress.
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger("RLRewardTracker")


@dataclass
class RewardStep:
    """Single reward step record"""
    pnl_delta: float
    realized_ig: float
    drawdown: float
    timestamp: float = 0.0
    raw_reward: float = 0.0
    normalized_reward: float = 0.0


class RLRewardTracker:
    """
    Centralized reward computation and tracking for RL agent.
    
    Maintains rolling windows of PnL, information gain, and drawdown
    to compute stable composite rewards for policy learning.
    
    Attributes:
        lambda_ig: Weight for information gain component (default 0.2)
        lambda_risk: Weight for drawdown penalty (default 1.0)
        reward_scale: Scaling factor for normalization (default 10.0)
    """
    
    def __init__(self, 
                 window: int = 200,
                 lambda_ig: float = 0.2,
                 lambda_risk: float = 1.0,
                 reward_scale: float = 10.0):
        """
        Initialize reward tracker.
        
        Args:
            window: Rolling window size for statistics
            lambda_ig: Weight for information gain term
            lambda_risk: Weight for drawdown penalty
            reward_scale: Divisor for reward normalization (tanh(reward/scale))
        """
        self.window = window
        self.lambda_ig = lambda_ig
        self.lambda_risk = lambda_risk
        self.reward_scale = reward_scale
        
        self.pnls: deque = deque(maxlen=window)
        self.igs: deque = deque(maxlen=window)
        self.drawdowns: deque = deque(maxlen=window)
        self.rewards: deque = deque(maxlen=window)
        
        self.step_history: deque = deque(maxlen=window)
        
        self.total_steps = 0
        self.cumulative_pnl = 0.0
        self.cumulative_ig = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        
        logger.info(f"RLRewardTracker initialized: λ_ig={lambda_ig}, λ_risk={lambda_risk}")
        
    def record_step(self, 
                    pnl_delta: float, 
                    realized_ig: float, 
                    drawdown: float,
                    timestamp: float = 0.0) -> float:
        """
        Record a single step and compute reward.
        
        Args:
            pnl_delta: Change in equity since last step
            realized_ig: Actual KL divergence shift after outcome
            drawdown: Current drawdown from peak (positive value)
            timestamp: Optional timestamp for logging
            
        Returns:
            Normalized reward value in approximately [-1, 1]
        """
        self.pnls.append(pnl_delta)
        self.igs.append(realized_ig)
        self.drawdowns.append(drawdown)
        
        raw_reward = pnl_delta + self.lambda_ig * realized_ig - self.lambda_risk * drawdown
        
        normalized_reward = np.tanh(raw_reward / self.reward_scale)
        
        self.rewards.append(normalized_reward)
        
        step = RewardStep(
            pnl_delta=pnl_delta,
            realized_ig=realized_ig,
            drawdown=drawdown,
            timestamp=timestamp,
            raw_reward=raw_reward,
            normalized_reward=normalized_reward
        )
        self.step_history.append(step)
        
        self.total_steps += 1
        self.cumulative_pnl += pnl_delta
        self.cumulative_ig += realized_ig
        
        self.current_equity += pnl_delta
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            
        return normalized_reward
        
    def compute_reward(self) -> float:
        """
        Compute current composite reward from recent history.
        
        Returns:
            Normalized composite reward
        """
        if not self.pnls:
            return 0.0
            
        avg_pnl = np.mean(self.pnls)
        avg_ig = np.mean(self.igs)
        max_dd = max(self.drawdowns) if self.drawdowns else 0.0
        
        raw_reward = avg_pnl + self.lambda_ig * avg_ig - self.lambda_risk * max_dd
        
        return np.tanh(raw_reward / self.reward_scale)
        
    def compute_reward_components(self) -> Dict[str, float]:
        """
        Compute individual reward components for analysis.
        
        Returns:
            Dict with pnl_component, ig_component, risk_component, total
        """
        if not self.pnls:
            return {
                "pnl_component": 0.0,
                "ig_component": 0.0,
                "risk_component": 0.0,
                "total_raw": 0.0,
                "total_normalized": 0.0
            }
            
        avg_pnl = np.mean(self.pnls)
        avg_ig = np.mean(self.igs)
        max_dd = max(self.drawdowns) if self.drawdowns else 0.0
        
        pnl_comp = avg_pnl
        ig_comp = self.lambda_ig * avg_ig
        risk_comp = self.lambda_risk * max_dd
        
        total_raw = pnl_comp + ig_comp - risk_comp
        
        return {
            "pnl_component": pnl_comp,
            "ig_component": ig_comp,
            "risk_component": risk_comp,
            "total_raw": total_raw,
            "total_normalized": np.tanh(total_raw / self.reward_scale)
        }
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics.
        
        Returns:
            Dict with all tracking metrics
        """
        return {
            "avg_pnl": float(np.mean(self.pnls)) if self.pnls else 0.0,
            "avg_ig": float(np.mean(self.igs)) if self.igs else 0.0,
            "max_drawdown": float(max(self.drawdowns)) if self.drawdowns else 0.0,
            "avg_reward": float(np.mean(self.rewards)) if self.rewards else 0.0,
            "reward_std": float(np.std(self.rewards)) if len(self.rewards) > 1 else 0.0,
            "composite_reward": self.compute_reward(),
            "total_steps": self.total_steps,
            "cumulative_pnl": self.cumulative_pnl,
            "cumulative_ig": self.cumulative_ig,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "lambda_ig": self.lambda_ig,
            "lambda_risk": self.lambda_risk
        }
        
    def get_recent_rewards(self, n: int = 10) -> List[float]:
        """Get n most recent normalized rewards"""
        return list(self.rewards)[-n:]
        
    def get_reward_trend(self, window: int = 50) -> float:
        """
        Compute reward trend (slope) over recent window.
        
        Returns:
            Positive = improving, negative = declining
        """
        if len(self.rewards) < window:
            return 0.0
            
        recent = list(self.rewards)[-window:]
        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent, 1)
        
        return float(slope)
        
    def is_improving(self, threshold: float = 0.001) -> bool:
        """Check if reward trend is positive"""
        return self.get_reward_trend() > threshold
        
    def reset(self):
        """Reset all tracking state"""
        self.pnls.clear()
        self.igs.clear()
        self.drawdowns.clear()
        self.rewards.clear()
        self.step_history.clear()
        self.total_steps = 0
        self.cumulative_pnl = 0.0
        self.cumulative_ig = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        
    def update_lambdas(self, lambda_ig: Optional[float] = None, lambda_risk: Optional[float] = None):
        """Update reward weighting parameters"""
        if lambda_ig is not None:
            self.lambda_ig = max(0.0, min(1.0, lambda_ig))
        if lambda_risk is not None:
            self.lambda_risk = max(0.0, min(2.0, lambda_risk))
            
    def set_initial_equity(self, equity: float):
        """Set initial equity for drawdown tracking"""
        self.current_equity = equity
        self.peak_equity = equity


class RewardNormalizer:
    """
    Running normalization for reward values.
    
    Maintains running mean and std for reward normalization,
    useful for stabilizing PPO training.
    """
    
    def __init__(self, clip_range: float = 10.0, epsilon: float = 1e-8):
        """
        Initialize normalizer.
        
        Args:
            clip_range: Clip normalized values to [-clip_range, clip_range]
            epsilon: Small constant for numerical stability
        """
        self.clip_range = clip_range
        self.epsilon = epsilon
        
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        
    def update(self, reward: float):
        """Update running statistics with new reward"""
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += (delta * delta2 - self.var) / self.count
        
    def normalize(self, reward: float) -> float:
        """Normalize reward using running statistics"""
        std = np.sqrt(self.var + self.epsilon)
        normalized = (reward - self.mean) / std
        return np.clip(normalized, -self.clip_range, self.clip_range)
        
    def reset(self):
        """Reset running statistics"""
        self.mean = 0.0
        self.var = 1.0
        self.count = 0


def demo():
    """Demonstrate RLRewardTracker functionality"""
    print("=" * 60)
    print("RL REWARD TRACKER DEMO")
    print("=" * 60)
    
    tracker = RLRewardTracker(lambda_ig=0.2, lambda_risk=1.0)
    tracker.set_initial_equity(10000.0)
    
    print("\n--- Simulating trading steps ---")
    
    np.random.seed(42)
    for i in range(100):
        pnl = np.random.randn() * 50 + 5
        ig = np.random.exponential(0.1)
        dd = max(0, -pnl / 100) if pnl < 0 else 0
        
        reward = tracker.record_step(pnl, ig, dd)
        
        if (i + 1) % 20 == 0:
            summary = tracker.get_summary()
            print(f"\nStep {i+1}:")
            print(f"  Avg PnL: ${summary['avg_pnl']:.2f}")
            print(f"  Avg IG: {summary['avg_ig']:.4f}")
            print(f"  Max DD: {summary['max_drawdown']:.4f}")
            print(f"  Composite Reward: {summary['composite_reward']:.4f}")
            print(f"  Cumulative PnL: ${summary['cumulative_pnl']:.2f}")
            
    print("\n--- Reward Components ---")
    components = tracker.compute_reward_components()
    print(f"  PnL Component: {components['pnl_component']:.4f}")
    print(f"  IG Component: {components['ig_component']:.4f}")
    print(f"  Risk Component: {components['risk_component']:.4f}")
    print(f"  Total Raw: {components['total_raw']:.4f}")
    print(f"  Total Normalized: {components['total_normalized']:.4f}")
    
    print("\n--- Reward Trend ---")
    trend = tracker.get_reward_trend()
    print(f"  Trend (slope): {trend:.6f}")
    print(f"  Is Improving: {tracker.is_improving()}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo()
