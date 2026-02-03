"""
Tests for RL Evolver Bridge

Tests the Gym-style environment, PPO agent integration, and reward tracking
for autonomous policy learning from microstructure and belief signals.
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from advanced_modules.rl_reward_tracker import RLRewardTracker, RewardNormalizer
from advanced_modules.rlevolver import (
    RLObservation, RLAction, TradingEnvironment, RLEvolver,
    GYM_AVAILABLE, SB3_AVAILABLE, REWARD_TRACKER_AVAILABLE
)


class TestRLRewardTracker:
    """Tests for RLRewardTracker"""
    
    def test_initialization(self):
        """Test tracker initializes with correct defaults"""
        tracker = RLRewardTracker()
        assert tracker.lambda_ig == 0.2
        assert tracker.lambda_risk == 1.0
        assert tracker.total_steps == 0
        assert tracker.cumulative_pnl == 0.0
        
    def test_record_step(self):
        """Test recording a single step"""
        tracker = RLRewardTracker()
        reward = tracker.record_step(pnl_delta=10.0, realized_ig=0.1, drawdown=0.0)
        
        assert tracker.total_steps == 1
        assert tracker.cumulative_pnl == 10.0
        assert -1 <= reward <= 1
        
    def test_reward_normalization(self):
        """Test that rewards are normalized to approximately [-1, 1]"""
        tracker = RLRewardTracker(reward_scale=10.0)
        
        reward_high = tracker.record_step(pnl_delta=100.0, realized_ig=1.0, drawdown=0.0)
        assert reward_high > 0.9
        
        tracker.reset()
        reward_low = tracker.record_step(pnl_delta=-100.0, realized_ig=0.0, drawdown=1.0)
        assert reward_low < -0.9
        
    def test_compute_reward_components(self):
        """Test reward component breakdown"""
        tracker = RLRewardTracker(lambda_ig=0.2, lambda_risk=1.0)
        tracker.record_step(pnl_delta=10.0, realized_ig=0.5, drawdown=0.1)
        
        components = tracker.compute_reward_components()
        
        assert "pnl_component" in components
        assert "ig_component" in components
        assert "risk_component" in components
        assert "total_raw" in components
        assert "total_normalized" in components
        
    def test_get_summary(self):
        """Test summary statistics"""
        tracker = RLRewardTracker()
        
        for i in range(10):
            tracker.record_step(
                pnl_delta=np.random.randn() * 10,
                realized_ig=np.random.exponential(0.1),
                drawdown=max(0, np.random.randn() * 0.01)
            )
            
        summary = tracker.get_summary()
        
        assert "avg_pnl" in summary
        assert "avg_ig" in summary
        assert "max_drawdown" in summary
        assert "composite_reward" in summary
        assert "total_steps" in summary
        assert summary["total_steps"] == 10
        
    def test_reward_trend(self):
        """Test reward trend calculation"""
        tracker = RLRewardTracker()
        
        for i in range(100):
            tracker.record_step(pnl_delta=i * 0.1, realized_ig=0.1, drawdown=0.0)
            
        trend = tracker.get_reward_trend(window=50)
        assert trend > 0
        
    def test_reset(self):
        """Test tracker reset"""
        tracker = RLRewardTracker()
        tracker.record_step(pnl_delta=10.0, realized_ig=0.1, drawdown=0.0)
        
        tracker.reset()
        
        assert tracker.total_steps == 0
        assert tracker.cumulative_pnl == 0.0
        assert len(tracker.pnls) == 0
        
    def test_update_lambdas(self):
        """Test lambda parameter updates"""
        tracker = RLRewardTracker()
        
        tracker.update_lambdas(lambda_ig=0.5, lambda_risk=1.5)
        
        assert tracker.lambda_ig == 0.5
        assert tracker.lambda_risk == 1.5
        
        tracker.update_lambdas(lambda_ig=2.0)
        assert tracker.lambda_ig == 1.0
        
        tracker.update_lambdas(lambda_risk=3.0)
        assert tracker.lambda_risk == 2.0


class TestRewardNormalizer:
    """Tests for RewardNormalizer"""
    
    def test_initialization(self):
        """Test normalizer initializes correctly"""
        normalizer = RewardNormalizer()
        assert normalizer.mean == 0.0
        assert normalizer.var == 1.0
        assert normalizer.count == 0
        
    def test_update_and_normalize(self):
        """Test running normalization"""
        normalizer = RewardNormalizer()
        
        for _ in range(100):
            reward = np.random.randn() * 10 + 5
            normalizer.update(reward)
            
        normalized = normalizer.normalize(5.0)
        assert -10 <= normalized <= 10


class TestRLObservation:
    """Tests for RLObservation"""
    
    def test_default_values(self):
        """Test default observation values"""
        obs = RLObservation()
        assert obs.p_accept == 0.5
        assert obs.confidence == 0.5
        assert obs.mm_spoof == 0.0
        
    def test_to_array(self):
        """Test conversion to numpy array"""
        obs = RLObservation(p_accept=0.7, confidence=0.8)
        arr = obs.to_array()
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (12,)
        assert arr.dtype == np.float32
        assert arr[0] == 0.7
        assert arr[1] == 0.8
        
    def test_from_belief_state(self):
        """Test creation from belief state dict"""
        belief = {
            "p_accept": 0.75,
            "confidence": 0.85,
            "expected_ig_bits": 0.5,
            "regime": "HIGH_VOL",
            "mm_flags": {
                "spoof_detected": True,
                "absorption_detected": False,
                "depth_imbalance": 0.3
            }
        }
        
        obs = RLObservation.from_belief_state(belief, pnl_delta=50.0, exposure=0.5)
        
        assert obs.p_accept == 0.75
        assert obs.confidence == 0.85
        assert obs.mm_spoof == 1.0
        assert obs.mm_absorption == 0.0
        assert obs.current_regime == 0.75


class TestRLAction:
    """Tests for RLAction"""
    
    def test_from_array(self):
        """Test creation from action array"""
        action_arr = np.array([0.5, -0.3, 0.2, 0.1])
        action = RLAction.from_array(action_arr)
        
        assert action.delta_exposure == 0.25
        assert 0.5 <= action.feature_weight_multiplier <= 1.5
        assert 0.5 <= action.signal_confidence_multiplier <= 1.5
        assert -0.1 <= action.lambda_ig_adjustment <= 0.1
        
    def test_clipping(self):
        """Test action clipping"""
        action_arr = np.array([2.0, -2.0, 2.0, -2.0])
        action = RLAction.from_array(action_arr)
        
        assert action.delta_exposure == 0.5
        assert action.feature_weight_multiplier == 0.5
        
    def test_to_dict(self):
        """Test conversion to dict"""
        action = RLAction(delta_exposure=0.1, feature_weight_multiplier=1.2)
        d = action.to_dict()
        
        assert isinstance(d, dict)
        assert d["delta_exposure"] == 0.1
        assert d["feature_weight_multiplier"] == 1.2
        
    def test_to_task_description(self):
        """Test task description generation"""
        action = RLAction(delta_exposure=0.2, feature_weight_multiplier=1.3)
        desc = action.to_task_description()
        
        assert "RL-proposed" in desc
        assert "exposure" in desc.lower() or "feature" in desc.lower()


class TestTradingEnvironment:
    """Tests for TradingEnvironment"""
    
    def test_initialization(self):
        """Test environment initializes correctly"""
        env = TradingEnvironment()
        
        assert env.observation_space.shape == (12,)
        assert env.action_space.shape == (4,)
        
    def test_reset(self):
        """Test environment reset"""
        env = TradingEnvironment()
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (12,)
        assert isinstance(info, dict)
        
    def test_step(self):
        """Test environment step"""
        env = TradingEnvironment()
        env.reset()
        
        action = np.array([0.1, 0.2, -0.1, 0.05])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (12,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
    def test_observation_bounds(self):
        """Test observation space bounds"""
        env = TradingEnvironment()
        
        assert all(env.observation_space.low >= 0) or any(env.observation_space.low < 0)
        assert all(env.observation_space.high <= 1) or any(env.observation_space.high > 1)
        
    def test_action_bounds(self):
        """Test action space bounds"""
        env = TradingEnvironment()
        
        assert all(env.action_space.low == -1)
        assert all(env.action_space.high == 1)
        
    def test_update_observation(self):
        """Test external observation update"""
        env = TradingEnvironment()
        env.reset()
        
        belief = {"p_accept": 0.9, "confidence": 0.95, "regime": "NORMAL", "mm_flags": {}}
        env.update_observation(belief, pnl_delta=100.0)
        
        assert env._current_obs.p_accept == 0.9
        assert env._current_obs.confidence == 0.95
        
    def test_max_episode_steps(self):
        """Test episode truncation"""
        env = TradingEnvironment(max_episode_steps=5)
        env.reset()
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
        assert truncated


class TestRLEvolver:
    """Tests for RLEvolver"""
    
    def test_initialization(self):
        """Test evolver initializes correctly"""
        evolver = RLEvolver(checkpoint_path="models/test_checkpoint.zip", burn_in_steps=10)
        
        assert evolver.burn_in_steps == 10
        assert evolver.total_steps == 0
        assert evolver.is_burn_in
        
    def test_predict_burn_in(self):
        """Test prediction during burn-in (random actions)"""
        evolver = RLEvolver(burn_in_steps=100)
        
        action = evolver.predict()
        
        assert isinstance(action, RLAction)
        assert evolver.total_steps == 0
        
    def test_step(self):
        """Test full step with belief state"""
        evolver = RLEvolver(burn_in_steps=10)
        
        belief = {
            "p_accept": 0.6,
            "confidence": 0.7,
            "expected_ig_bits": 0.1,
            "regime": "NORMAL",
            "mm_flags": {}
        }
        
        action = evolver.step(belief, pnl_delta=5.0)
        
        assert isinstance(action, RLAction)
        assert evolver.total_steps == 1
        
    def test_record_outcome(self):
        """Test outcome recording"""
        evolver = RLEvolver()
        
        evolver.record_outcome(pnl_delta=10.0, realized_ig=0.1, drawdown=0.0)
        
        if evolver.reward_tracker:
            assert evolver.reward_tracker.total_steps == 1
            
    def test_get_status(self):
        """Test status retrieval"""
        evolver = RLEvolver()
        
        status = evolver.get_status()
        
        assert "total_steps" in status
        assert "is_burn_in" in status
        assert "model_initialized" in status
        
    def test_task_queue_integration(self):
        """Test task queue integration"""
        evolver = RLEvolver()
        task_queue = []
        evolver.set_task_queue(task_queue)
        
        action = RLAction(delta_exposure=0.2)
        evolver.propose_action(action)
        
        assert len(task_queue) == 1
        assert task_queue[0]["type"] == "rl_proposal"
        
    def test_burn_in_completion(self):
        """Test burn-in phase completion"""
        evolver = RLEvolver(burn_in_steps=5)
        
        belief = {"p_accept": 0.5, "confidence": 0.5, "regime": "NORMAL", "mm_flags": {}}
        
        for _ in range(6):
            evolver.step(belief)
            
        assert not evolver.is_burn_in
        
    def test_reward_trend(self):
        """Test reward trend retrieval"""
        evolver = RLEvolver()
        
        for _ in range(10):
            evolver.record_outcome(pnl_delta=10.0, realized_ig=0.1, drawdown=0.0)
            
        trend = evolver.get_reward_trend()
        assert isinstance(trend, float)


class TestIntegration:
    """Integration tests for RL Evolver with SelfEvolutionAgent"""
    
    def test_action_to_task_flow(self):
        """Test action -> task queue flow"""
        evolver = RLEvolver(burn_in_steps=0)
        task_queue = []
        evolver.set_task_queue(task_queue)
        
        belief = {
            "p_accept": 0.7,
            "confidence": 0.8,
            "expected_ig_bits": 0.2,
            "regime": "HIGH_VOL",
            "mm_flags": {"spoof_detected": True}
        }
        
        action = evolver.step(belief)
        evolver.propose_action(action)
        
        assert len(task_queue) == 1
        task = task_queue[0]
        assert "action" in task
        assert "description" in task
        
    def test_full_cycle(self):
        """Test full RL cycle: observe -> predict -> propose -> record"""
        evolver = RLEvolver(burn_in_steps=0)
        task_queue = []
        evolver.set_task_queue(task_queue)
        
        for i in range(5):
            belief = {
                "p_accept": 0.5 + i * 0.05,
                "confidence": 0.5 + i * 0.05,
                "expected_ig_bits": 0.1,
                "regime": "NORMAL",
                "mm_flags": {}
            }
            
            action = evolver.step(belief, pnl_delta=i * 10)
            
            evolver.propose_action(action)
            
            success = i % 2 == 0
            evolver.record_outcome(
                pnl_delta=10.0 if success else -5.0,
                realized_ig=0.1,
                drawdown=0.0 if success else 0.01
            )
            
        assert evolver.total_steps == 5
        assert len(task_queue) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
