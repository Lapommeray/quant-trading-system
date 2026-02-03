"""
RL Evolver Bridge - Autonomous Policy Learning from Microstructure & Belief Signals

This module implements a lightweight, continual RL agent (PPO) that learns directly
from the live paper trading loop. The agent observes the current belief state and
microstructure flags, then proposes actions (exposure adjustments, feature weights,
signal multipliers) that are debated/codified by the SelfEvolutionAgent.

Key Features:
- Gym-style environment with belief + microstructure observations
- PPO agent with MLP policy for continuous action space
- Integration with SelfEvolutionAgent debate queue
- Checkpoint save/load for persistence across restarts
- Safety constraints: paper mode only, action clipping, burn-in exploration

Observation Space (~12 dims):
- p_accept, confidence, expected_ig, depth_imbalance, current_spread
- mm_spoof, mm_absorption, mm_inventory_flip (binary flags)
- current_regime, recent_pnl_delta, current_exposure, volatility_proxy

Action Space (4 dims, continuous [-1, 1]):
- delta_exposure, feature_weight_multiplier, signal_confidence_multiplier, lambda_ig_adjustment
"""

import os
import sys
import time
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field, asdict
from collections import deque
from pathlib import Path

logger = logging.getLogger("RLEvolver")

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        gym = None
        
        class MockSpaces:
            @staticmethod
            def Box(low, high, shape=None, dtype=None):
                class MockBox:
                    def __init__(self, low, high, shape, dtype):
                        self.low = np.array(low) if hasattr(low, '__iter__') else np.full(shape, low)
                        self.high = np.array(high) if hasattr(high, '__iter__') else np.full(shape, high)
                        self.shape = shape if shape else self.low.shape
                        self.dtype = dtype or np.float32
                    def sample(self):
                        return np.random.uniform(self.low, self.high).astype(self.dtype)
                return MockBox(low, high, shape, dtype)
        
        spaces = MockSpaces()
        logger.warning("Gymnasium/Gym not available. RL features disabled.")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseCallback = object
    logger.warning("stable-baselines3 not available. RL features disabled.")

try:
    from advanced_modules.rl_reward_tracker import RLRewardTracker
    REWARD_TRACKER_AVAILABLE = True
except ImportError:
    REWARD_TRACKER_AVAILABLE = False
    logger.warning("RLRewardTracker not available")


@dataclass
class RLObservation:
    """Structured observation for RL agent"""
    p_accept: float = 0.5
    confidence: float = 0.5
    expected_ig: float = 0.0
    depth_imbalance: float = 0.0
    current_spread: float = 0.0
    mm_spoof: float = 0.0
    mm_absorption: float = 0.0
    mm_inventory_flip: float = 0.0
    current_regime: float = 0.0
    recent_pnl_delta: float = 0.0
    current_exposure: float = 0.0
    volatility_proxy: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for Gym observation"""
        return np.array([
            self.p_accept,
            self.confidence,
            self.expected_ig,
            self.depth_imbalance,
            self.current_spread,
            self.mm_spoof,
            self.mm_absorption,
            self.mm_inventory_flip,
            self.current_regime,
            self.recent_pnl_delta,
            self.current_exposure,
            self.volatility_proxy
        ], dtype=np.float32)
        
    @classmethod
    def from_belief_state(cls, belief: Dict[str, Any], 
                          pnl_delta: float = 0.0,
                          exposure: float = 0.0,
                          volatility: float = 1.0,
                          spread: float = 0.0) -> 'RLObservation':
        """Create observation from BayesianMarketState belief dict"""
        mm_flags = belief.get("mm_flags", {}) or {}
        
        regime_map = {"UNKNOWN": 0.0, "LOW_VOL": 0.25, "NORMAL": 0.5, "HIGH_VOL": 0.75, "CRISIS": 1.0}
        regime_val = regime_map.get(belief.get("regime", "UNKNOWN"), 0.5)
        
        return cls(
            p_accept=belief.get("p_accept", 0.5),
            confidence=belief.get("confidence", 0.5),
            expected_ig=min(belief.get("expected_ig_bits", 0.0), 2.0) / 2.0,
            depth_imbalance=np.clip(mm_flags.get("depth_imbalance", 0.0), -1, 1),
            current_spread=min(spread, 0.01) / 0.01,
            mm_spoof=1.0 if mm_flags.get("spoof_detected", False) else 0.0,
            mm_absorption=1.0 if mm_flags.get("absorption_detected", False) else 0.0,
            mm_inventory_flip=1.0 if mm_flags.get("inventory_flip_detected", False) else 0.0,
            current_regime=regime_val,
            recent_pnl_delta=np.clip(pnl_delta / 100.0, -1, 1),
            current_exposure=np.clip(exposure, -1, 1),
            volatility_proxy=min(volatility, 3.0) / 3.0
        )


@dataclass
class RLAction:
    """Structured action from RL agent"""
    delta_exposure: float = 0.0
    feature_weight_multiplier: float = 1.0
    signal_confidence_multiplier: float = 1.0
    lambda_ig_adjustment: float = 0.0
    
    @classmethod
    def from_array(cls, action: np.ndarray) -> 'RLAction':
        """Create from Gym action array"""
        action = np.clip(action, -1, 1)
        return cls(
            delta_exposure=float(action[0]) * 0.5,
            feature_weight_multiplier=1.0 + float(action[1]) * 0.5,
            signal_confidence_multiplier=1.0 + float(action[2]) * 0.5,
            lambda_ig_adjustment=float(action[3]) * 0.1
        )
        
    def to_dict(self) -> Dict[str, float]:
        """Convert to dict for task queue"""
        return asdict(self)
        
    def to_task_description(self) -> str:
        """Generate human-readable task description"""
        parts = []
        if abs(self.delta_exposure) > 0.05:
            direction = "increase" if self.delta_exposure > 0 else "decrease"
            parts.append(f"{direction} exposure by {abs(self.delta_exposure)*100:.1f}%")
        if abs(self.feature_weight_multiplier - 1.0) > 0.05:
            parts.append(f"adjust feature weights to {self.feature_weight_multiplier:.2f}x")
        if abs(self.signal_confidence_multiplier - 1.0) > 0.05:
            parts.append(f"adjust signal confidence to {self.signal_confidence_multiplier:.2f}x")
        if abs(self.lambda_ig_adjustment) > 0.01:
            direction = "increase" if self.lambda_ig_adjustment > 0 else "decrease"
            parts.append(f"{direction} λ_IG by {abs(self.lambda_ig_adjustment):.3f}")
        return "RL-proposed: " + (", ".join(parts) if parts else "no significant changes")


class TradingEnvironment:
    """
    Gym-style environment for RL policy learning.
    
    Wraps the trading system's belief state and microstructure signals
    into a standard RL interface with observations, actions, and rewards.
    """
    
    def __init__(self,
                 reward_tracker: Optional['RLRewardTracker'] = None,
                 step_interval_sec: float = 120.0,
                 max_episode_steps: int = 1000):
        """
        Initialize trading environment.
        
        Args:
            reward_tracker: RLRewardTracker instance for reward computation
            step_interval_sec: Minimum seconds between steps
            max_episode_steps: Maximum steps per episode
        """
        self.reward_tracker = reward_tracker
        if self.reward_tracker is None and REWARD_TRACKER_AVAILABLE:
            self.reward_tracker = RLRewardTracker()
            
        self.step_interval_sec = step_interval_sec
        self.max_episode_steps = max_episode_steps
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, 0, 0, 0, 0, 0, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        self._current_obs = RLObservation()
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_step_time = 0.0
        
        self._belief_callback: Optional[Callable] = None
        self._action_callback: Optional[Callable] = None
        
        logger.info("TradingEnvironment initialized")
        
    def set_belief_callback(self, callback: Callable[[], Dict[str, Any]]):
        """Set callback to get current belief state"""
        self._belief_callback = callback
        
    def set_action_callback(self, callback: Callable[[RLAction], None]):
        """Set callback to execute actions"""
        self._action_callback = callback
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        if seed is not None:
            np.random.seed(seed)
            
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_step_time = time.time()
        
        if self.reward_tracker:
            self.reward_tracker.reset()
            
        self._current_obs = RLObservation()
        
        if self._belief_callback:
            try:
                belief = self._belief_callback()
                self._current_obs = RLObservation.from_belief_state(belief)
            except Exception as e:
                logger.warning(f"Belief callback failed: {e}")
                
        return self._current_obs.to_array(), {}
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action array from agent
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1
        
        rl_action = RLAction.from_array(action)
        
        if self._action_callback:
            try:
                self._action_callback(rl_action)
            except Exception as e:
                logger.warning(f"Action callback failed: {e}")
                
        pnl_delta = 0.0
        realized_ig = 0.0
        drawdown = 0.0
        
        if self._belief_callback:
            try:
                belief = self._belief_callback()
                self._current_obs = RLObservation.from_belief_state(belief)
                realized_ig = belief.get("expected_ig_bits", 0.0)
            except Exception as e:
                logger.warning(f"Belief callback failed: {e}")
                
        reward = 0.0
        if self.reward_tracker:
            reward = self.reward_tracker.record_step(pnl_delta, realized_ig, drawdown)
        else:
            reward = np.tanh((pnl_delta + 0.2 * realized_ig - drawdown) / 10.0)
            
        self._episode_reward += reward
        
        terminated = False
        truncated = self._step_count >= self.max_episode_steps
        
        info = {
            "step": self._step_count,
            "episode_reward": self._episode_reward,
            "action": rl_action.to_dict(),
            "pnl_delta": pnl_delta,
            "realized_ig": realized_ig
        }
        
        self._last_step_time = time.time()
        
        return self._current_obs.to_array(), reward, terminated, truncated, info
        
    def update_observation(self, belief: Dict[str, Any], 
                           pnl_delta: float = 0.0,
                           exposure: float = 0.0,
                           volatility: float = 1.0,
                           spread: float = 0.0):
        """Update current observation from external source"""
        self._current_obs = RLObservation.from_belief_state(
            belief, pnl_delta, exposure, volatility, spread
        )
        
    def record_outcome(self, pnl_delta: float, realized_ig: float, drawdown: float):
        """Record trade outcome for reward computation"""
        if self.reward_tracker:
            self.reward_tracker.record_step(pnl_delta, realized_ig, drawdown)


class RewardLoggingCallback(BaseCallback):
    """Callback for logging reward statistics during training"""
    
    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.rewards = []
        self.entropies = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            if len(self.rewards) > 0:
                mean_reward = np.mean(self.rewards[-self.log_interval:])
                logger.info(f"Step {self.n_calls}: mean_reward={mean_reward:.4f}")
            self.rewards.append(self.locals.get("rewards", [0])[0] if "rewards" in self.locals else 0)
        return True


class RLEvolver:
    """
    Main RL Evolver class that manages the PPO agent and environment.
    
    Handles:
    - Environment setup and management
    - PPO agent training and inference
    - Checkpoint save/load
    - Integration with SelfEvolutionAgent
    """
    
    def __init__(self,
                 checkpoint_path: str = "models/rl_checkpoint.zip",
                 step_interval_sec: float = 120.0,
                 burn_in_steps: int = 5000,
                 lambda_ig: float = 0.2,
                 lambda_risk: float = 1.0,
                 policy_kwargs: Optional[Dict] = None):
        """
        Initialize RL Evolver.
        
        Args:
            checkpoint_path: Path for model checkpoints
            step_interval_sec: Minimum seconds between steps
            burn_in_steps: Random exploration steps before learning
            lambda_ig: Information gain weight for reward
            lambda_risk: Drawdown penalty weight for reward
            policy_kwargs: Custom policy network architecture
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.step_interval_sec = step_interval_sec
        self.burn_in_steps = burn_in_steps
        
        if REWARD_TRACKER_AVAILABLE:
            self.reward_tracker = RLRewardTracker(
                lambda_ig=lambda_ig,
                lambda_risk=lambda_risk
            )
        else:
            self.reward_tracker = None
            
        self.env = TradingEnvironment(
            reward_tracker=self.reward_tracker,
            step_interval_sec=step_interval_sec
        )
        
        self.policy_kwargs = policy_kwargs or {
            "net_arch": [256, 128, 64]
        }
        
        self.model: Optional[PPO] = None
        self.total_steps = 0
        self.is_burn_in = True
        
        self._task_queue: Optional[Any] = None
        self._last_action: Optional[RLAction] = None
        self._action_history: deque = deque(maxlen=1000)
        
        self._initialize_model()
        
        logger.info(f"RLEvolver initialized: checkpoint={checkpoint_path}, burn_in={burn_in_steps}")
        
    def _initialize_model(self):
        """Initialize or load PPO model"""
        if not SB3_AVAILABLE or not GYM_AVAILABLE:
            logger.warning("RL dependencies not available, running in mock mode")
            return
            
        if self.checkpoint_path.exists():
            try:
                self.model = PPO.load(str(self.checkpoint_path), env=self.env)
                logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
                self._create_new_model()
        else:
            self._create_new_model()
            
    def _create_new_model(self):
        """Create new PPO model"""
        if not SB3_AVAILABLE:
            return
            
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            policy_kwargs=self.policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01
        )
        logger.info("Created new PPO model")
        
    def set_task_queue(self, task_queue):
        """Set reference to SelfEvolutionAgent task queue"""
        self._task_queue = task_queue
        
    def predict(self, observation: Optional[np.ndarray] = None, 
                deterministic: bool = False) -> RLAction:
        """
        Get action prediction from agent.
        
        Args:
            observation: Optional observation array (uses current env obs if None)
            deterministic: Use deterministic policy (no exploration)
            
        Returns:
            RLAction with proposed adjustments
        """
        if observation is None:
            observation = self.env._current_obs.to_array()
            
        if self.is_burn_in and self.total_steps < self.burn_in_steps:
            action = self.env.action_space.sample()
            logger.debug(f"Burn-in random action: {action}")
        elif self.model is not None:
            action, _ = self.model.predict(observation, deterministic=deterministic)
        else:
            action = np.zeros(4, dtype=np.float32)
            
        rl_action = RLAction.from_array(action)
        self._last_action = rl_action
        self._action_history.append({
            "step": self.total_steps,
            "action": rl_action.to_dict(),
            "timestamp": time.time()
        })
        
        return rl_action
        
    def step(self, belief: Dict[str, Any],
             pnl_delta: float = 0.0,
             exposure: float = 0.0,
             volatility: float = 1.0,
             spread: float = 0.0) -> RLAction:
        """
        Execute one RL step: observe, predict, propose action.
        
        Args:
            belief: Current belief state from BayesianMarketState
            pnl_delta: Recent PnL change
            exposure: Current position exposure
            volatility: Current volatility proxy
            spread: Current bid-ask spread
            
        Returns:
            RLAction with proposed adjustments
        """
        self.env.update_observation(belief, pnl_delta, exposure, volatility, spread)
        
        action = self.predict()
        
        self.total_steps += 1
        
        if self.total_steps >= self.burn_in_steps:
            self.is_burn_in = False
            
        return action
        
    def propose_action(self, action: RLAction):
        """
        Propose action to SelfEvolutionAgent task queue.
        
        Args:
            action: RLAction to propose
        """
        if self._task_queue is None:
            logger.debug("No task queue set, action not proposed")
            return
            
        task = {
            "type": "rl_proposal",
            "description": action.to_task_description(),
            "action": action.to_dict(),
            "step": self.total_steps,
            "timestamp": time.time()
        }
        
        try:
            if hasattr(self._task_queue, 'append'):
                self._task_queue.append(task)
            elif hasattr(self._task_queue, 'add_task'):
                self._task_queue.add_task(task)
            try:
                action_desc = action.to_task_description()
            except Exception:
                action_desc = str(action)
            logger.info(f"Proposed RL action: {action_desc}")
        except Exception as e:
            logger.warning(f"Failed to propose action: {e}")
            
    def record_outcome(self, pnl_delta: float, realized_ig: float, drawdown: float):
        """
        Record trade outcome for reward computation.
        
        Args:
            pnl_delta: Realized PnL change
            realized_ig: Actual information gain
            drawdown: Current drawdown from peak
        """
        self.env.record_outcome(pnl_delta, realized_ig, drawdown)
        
    def learn(self, total_timesteps: int = 10000, reset_num_timesteps: bool = False):
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Number of timesteps to train
            reset_num_timesteps: Reset step counter (False for continual learning)
        """
        if self.model is None:
            logger.warning("Model not initialized, cannot learn")
            return
            
        callback = RewardLoggingCallback(log_interval=1000)
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=reset_num_timesteps,
                callback=callback
            )
            logger.info(f"Completed {total_timesteps} training steps")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
    def save_checkpoint(self):
        """Save model checkpoint"""
        if self.model is None:
            return
            
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.model.save(str(self.checkpoint_path))
            logger.info(f"Saved checkpoint to {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    def load_checkpoint(self) -> bool:
        """Load model checkpoint"""
        if not self.checkpoint_path.exists():
            logger.info("No checkpoint found, starting fresh")
            return False
            
        try:
            self.model = PPO.load(str(self.checkpoint_path), env=self.env)
            logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Get current evolver status"""
        status = {
            "total_steps": self.total_steps,
            "is_burn_in": self.is_burn_in,
            "burn_in_remaining": max(0, self.burn_in_steps - self.total_steps),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "model_initialized": self.model is not None,
            "last_action": self._last_action.to_dict() if self._last_action else None
        }
        
        if self.reward_tracker:
            status["reward_summary"] = self.reward_tracker.get_summary()
            
        return status
        
    def get_reward_trend(self) -> float:
        """Get current reward trend"""
        if self.reward_tracker:
            return self.reward_tracker.get_reward_trend()
        return 0.0
        
    def is_improving(self) -> bool:
        """Check if agent is improving"""
        if self.reward_tracker:
            return self.reward_tracker.is_improving()
        return False


def create_mock_env():
    """Create mock environment for testing without dependencies"""
    
    class MockEnv:
        def __init__(self):
            self.observation_space = type('Space', (), {
                'shape': (12,),
                'sample': lambda: np.random.randn(12).astype(np.float32)
            })()
            self.action_space = type('Space', (), {
                'shape': (4,),
                'sample': lambda: np.random.randn(4).astype(np.float32)
            })()
            self._step = 0
            
        def reset(self, seed=None, options=None):
            self._step = 0
            return np.random.randn(12).astype(np.float32), {}
            
        def step(self, action):
            self._step += 1
            obs = np.random.randn(12).astype(np.float32)
            reward = np.random.randn() * 0.1
            terminated = False
            truncated = self._step >= 100
            return obs, reward, terminated, truncated, {}
            
    return MockEnv()


def demo():
    """Demonstrate RLEvolver functionality"""
    print("=" * 60)
    print("RL EVOLVER DEMO")
    print("=" * 60)
    
    print(f"\nDependencies:")
    print(f"  Gymnasium: {GYM_AVAILABLE}")
    print(f"  stable-baselines3: {SB3_AVAILABLE}")
    print(f"  RLRewardTracker: {REWARD_TRACKER_AVAILABLE}")
    
    if not GYM_AVAILABLE or not SB3_AVAILABLE:
        print("\n--- Running in mock mode (dependencies not available) ---")
        
        evolver = RLEvolver(
            checkpoint_path="models/test_checkpoint.zip",
            burn_in_steps=100
        )
        
        print("\n--- Simulating steps ---")
        for i in range(10):
            belief = {
                "p_accept": 0.5 + np.random.randn() * 0.1,
                "confidence": 0.5 + np.random.randn() * 0.1,
                "expected_ig_bits": np.random.exponential(0.1),
                "regime": "NORMAL",
                "mm_flags": {
                    "spoof_detected": np.random.random() > 0.9,
                    "absorption_detected": np.random.random() > 0.9,
                    "depth_imbalance": np.random.randn() * 0.3
                }
            }
            
            action = evolver.step(belief, pnl_delta=np.random.randn() * 10)
            print(f"Step {i+1}: {action.to_task_description()}")
            
        print("\n--- Status ---")
        status = evolver.get_status()
        for k, v in status.items():
            if k != "reward_summary":
                print(f"  {k}: {v}")
                
        if "reward_summary" in status:
            print("  Reward Summary:")
            for k, v in status["reward_summary"].items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    else:
        print("\n--- Full RL mode available ---")
        
        evolver = RLEvolver(
            checkpoint_path="models/demo_checkpoint.zip",
            burn_in_steps=50
        )
        
        print("\n--- Running short training ---")
        evolver.env.reset()
        
        for i in range(100):
            belief = {
                "p_accept": 0.5 + np.random.randn() * 0.1,
                "confidence": 0.5 + np.random.randn() * 0.1,
                "expected_ig_bits": np.random.exponential(0.1),
                "regime": "NORMAL",
                "mm_flags": {}
            }
            
            action = evolver.step(belief, pnl_delta=np.random.randn() * 10)
            evolver.record_outcome(
                pnl_delta=np.random.randn() * 10,
                realized_ig=np.random.exponential(0.05),
                drawdown=max(0, np.random.randn() * 0.01)
            )
            
            if (i + 1) % 20 == 0:
                print(f"Step {i+1}: {action.to_task_description()}")
                
        print("\n--- Status ---")
        status = evolver.get_status()
        print(f"  Total steps: {status['total_steps']}")
        print(f"  Is burn-in: {status['is_burn_in']}")
        print(f"  Reward trend: {evolver.get_reward_trend():.6f}")
        print(f"  Is improving: {evolver.is_improving()}")
        
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo()
