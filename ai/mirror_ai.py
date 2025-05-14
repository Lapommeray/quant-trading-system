"""
Mirror AI Module

This module implements the Mirror AI agent for the Liquidity Thunderdome,
which learns by observing the Aggressor AI and mirrors its strategies with
adaptations. It specializes in reverse-engineering successful trading patterns.

Dependencies:
- tensorflow
- numpy
- pandas
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime
import json
import time
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mirror_ai.log')
    ]
)

logger = logging.getLogger("MirrorAI")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, BatchNormalization, Concatenate
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    logger.info("TensorFlow loaded successfully")
except ImportError:
    logger.warning("TensorFlow not available. Install with: pip install tensorflow")
    TF_AVAILABLE = False

class MirrorMemory:
    """Memory buffer for storing observed actions and outcomes"""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the memory buffer.
        
        Parameters:
        - capacity: Maximum number of observations to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state: np.ndarray, observed_action: int, outcome: float, next_state: np.ndarray) -> None:
        """
        Store an observation in the buffer.
        
        Parameters:
        - state: State when action was observed
        - observed_action: Action that was observed
        - outcome: Outcome of the observed action (reward)
        - next_state: State after the observed action
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        self.buffer[self.position] = (state, observed_action, outcome, next_state)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of observations from the buffer.
        
        Parameters:
        - batch_size: Number of observations to sample
        
        Returns:
        - Batch of observations
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, observed_action, outcome, next_state = map(np.stack, zip(*batch))
        return state, observed_action, outcome, next_state
        
    def __len__(self) -> int:
        """Return the current size of the buffer"""
        return len(self.buffer)

class MirrorAI:
    """
    Mirror AI agent for the Liquidity Thunderdome.
    Learns by observing other agents and mirroring their successful strategies.
    """
    
    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 5,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        memory_capacity: int = 10000,
        adaptation_rate: float = 0.1,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the Mirror AI agent.
        
        Parameters:
        - state_dim: Dimension of the state space
        - action_dim: Dimension of the action space
        - learning_rate: Learning rate for the optimizer
        - batch_size: Batch size for training
        - memory_capacity: Capacity of the observation buffer
        - adaptation_rate: Rate at which to adapt observed strategies
        - confidence_threshold: Threshold for confidence in mirroring
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.adaptation_rate = adaptation_rate
        self.confidence_threshold = confidence_threshold
        
        self.memory = MirrorMemory(capacity=memory_capacity)
        self.training_enabled = True
        
        self.mirror_network = None
        self.outcome_network = None
        
        if TF_AVAILABLE:
            self._build_networks()
        else:
            logger.error("TensorFlow not available. Cannot build networks.")
            
        self.loss_history = []
        self.mirror_accuracy = []
        self.outcome_accuracy = []
        
        self.total_observations = 0
        self.successful_mirrors = 0
        self.total_actions = 0
        self.total_rewards = 0
        
        logger.info("MirrorAI initialized")
        
    def _build_networks(self) -> None:
        """Build the mirror and outcome prediction networks"""
        try:
            self.mirror_network = self._create_mirror_network()
            
            self.outcome_network = self._create_outcome_network()
            
            logger.info("Networks built successfully")
        except Exception as e:
            logger.error(f"Error building networks: {str(e)}")
            
    def _create_mirror_network(self) -> Model:
        """
        Create a neural network for mirroring actions.
        
        Returns:
        - Keras model
        """
        inputs = Input(shape=(self.state_dim,))
        
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        
        outputs = Dense(self.action_dim, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def _create_outcome_network(self) -> Model:
        """
        Create a neural network for predicting action outcomes.
        
        Returns:
        - Keras model
        """
        state_input = Input(shape=(self.state_dim,))
        
        action_input = Input(shape=(self.action_dim,))
        
        x1 = Dense(64, activation='relu')(state_input)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.2)(x1)
        
        combined = Concatenate()([x1, action_input])
        
        x = Dense(64, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=[state_input, action_input], outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
        
    def observe(self, state: np.ndarray, action: int, outcome: float, next_state: np.ndarray) -> None:
        """
        Observe an action and its outcome.
        
        Parameters:
        - state: State when action was taken
        - action: Action that was taken
        - outcome: Outcome of the action (reward)
        - next_state: State after the action
        """
        self.memory.push(state, action, outcome, next_state)
        
        self.total_observations += 1
        
        if len(self.memory) >= self.batch_size and self.training_enabled:
            self._update_networks()
            
    def _update_networks(self) -> Tuple[float, float]:
        """
        Update networks based on observed actions and outcomes.
        
        Returns:
        - Mirror loss and outcome loss
        """
        if not TF_AVAILABLE or self.mirror_network is None or self.outcome_network is None:
            return 0.0, 0.0
            
        try:
            states, actions, outcomes, next_states = self.memory.sample(self.batch_size)
            
            states = np.array(states)
            actions = np.array(actions, dtype=np.int32)
            outcomes = np.array(outcomes, dtype=np.float32)
            
            actions_one_hot = tf.one_hot(actions, self.action_dim)
            
            mirror_history = self.mirror_network.fit(
                states, actions,
                epochs=1,
                batch_size=self.batch_size,
                verbose=0
            )
            
            mirror_loss = mirror_history.history['loss'][0]
            mirror_accuracy = mirror_history.history.get('accuracy', [0])[0]
            
            outcome_history = self.outcome_network.fit(
                [states, actions_one_hot], outcomes,
                epochs=1,
                batch_size=self.batch_size,
                verbose=0
            )
            
            outcome_loss = outcome_history.history['loss'][0]
            
            self.loss_history.append((mirror_loss, outcome_loss))
            self.mirror_accuracy.append(mirror_accuracy)
            
            return mirror_loss, outcome_loss
        except Exception as e:
            logger.error(f"Error updating networks: {str(e)}")
            return 0.0, 0.0
            
    def mirror_action(self, state: np.ndarray, explore: bool = True) -> Tuple[int, float]:
        """
        Mirror an action based on observed patterns.
        
        Parameters:
        - state: Current state
        - explore: Whether to explore or exploit
        
        Returns:
        - Selected action and confidence
        """
        if not TF_AVAILABLE or self.mirror_network is None:
            return random.randint(0, self.action_dim - 1), 0.0
            
        try:
            state_tensor = np.expand_dims(state, axis=0)
            action_probs = self.mirror_network.predict(state_tensor, verbose=0)[0]
            
            if explore and random.random() < self.adaptation_rate:
                action = random.randint(0, self.action_dim - 1)
                confidence = action_probs[action]
            else:
                action = np.argmax(action_probs)
                confidence = action_probs[action]
                
                if confidence < self.confidence_threshold and self.outcome_network is not None:
                    best_outcome = float('-inf')
                    best_action = action
                    
                    for a in range(self.action_dim):
                        action_one_hot = np.zeros(self.action_dim)
                        action_one_hot[a] = 1
                        
                        predicted_outcome = self.outcome_network.predict(
                            [state_tensor, np.expand_dims(action_one_hot, axis=0)],
                            verbose=0
                        )[0][0]
                        
                        if predicted_outcome > best_outcome:
                            best_outcome = predicted_outcome
                            best_action = a
                            
                    action = best_action
                    confidence = max(confidence, 0.5)  # Minimum confidence for outcome-based action
            
            return action, float(confidence)
        except Exception as e:
            logger.error(f"Error mirroring action: {str(e)}")
            return random.randint(0, self.action_dim - 1), 0.0
            
    def mirror_liquidity(self, market_data: Dict[str, Any], observed_actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Mirror liquidity hunting based on observed actions and market data.
        
        Parameters:
        - market_data: Dictionary containing market data
        - observed_actions: List of recently observed actions (optional)
        
        Returns:
        - Action and confidence
        """
        if not TF_AVAILABLE or self.mirror_network is None:
            logger.error("Networks not available. Cannot mirror liquidity.")
            return {"action": "none", "confidence": 0.0}
            
        try:
            state = self._extract_state(market_data)
            
            if observed_actions:
                for obs in observed_actions:
                    if "state" in obs and "action_idx" in obs and "outcome" in obs and "next_state" in obs:
                        self.observe(
                            obs["state"],
                            obs["action_idx"],
                            obs["outcome"],
                            obs["next_state"]
                        )
            
            action, confidence = self.mirror_action(state, explore=False)
            
            action_map = {
                0: "aggressive_buy",
                1: "passive_buy",
                2: "hold",
                3: "passive_sell",
                4: "aggressive_sell"
            }
            
            market_action = action_map.get(action, "hold")
            
            self.total_actions += 1
            
            predicted_outcome = None
            if self.outcome_network is not None:
                action_one_hot = np.zeros(self.action_dim)
                action_one_hot[action] = 1
                
                state_tensor = np.expand_dims(state, axis=0)
                action_tensor = np.expand_dims(action_one_hot, axis=0)
                
                predicted_outcome = float(self.outcome_network.predict(
                    [state_tensor, action_tensor],
                    verbose=0
                )[0][0])
            
            return {
                "action": market_action,
                "confidence": confidence,
                "action_idx": action,
                "predicted_outcome": predicted_outcome,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error mirroring liquidity: {str(e)}")
            return {"action": "none", "confidence": 0.0, "error": str(e)}
            
    def _extract_state(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract state representation from market data.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - State vector
        """
        try:
            
            state = np.zeros(self.state_dim)
            
            if "orderbook" in market_data:
                orderbook = market_data["orderbook"]
                
                if "bids" in orderbook and "asks" in orderbook and len(orderbook["bids"]) > 0 and len(orderbook["asks"]) > 0:
                    best_bid = orderbook["bids"][0][0]
                    best_ask = orderbook["asks"][0][0]
                    spread = best_ask - best_bid
                    
                    normalized_spread = spread / best_bid
                    
                    state[0] = normalized_spread
                    
                    bid_volumes = [bid[1] for bid in orderbook["bids"][:5]]
                    ask_volumes = [ask[1] for ask in orderbook["asks"][:5]]
                    
                    total_volume = sum(bid_volumes) + sum(ask_volumes)
                    if total_volume > 0:
                        normalized_bid_volumes = [vol / total_volume for vol in bid_volumes]
                        normalized_ask_volumes = [vol / total_volume for vol in ask_volumes]
                        
                        state[1:6] = normalized_bid_volumes[:5]
                        state[6:11] = normalized_ask_volumes[:5]
                        
            if "price_history" in market_data:
                price_history = market_data["price_history"]
                if len(price_history) > 1:
                    returns = np.diff(price_history) / price_history[:-1]
                    
                    recent_returns = returns[-min(5, len(returns)):]
                    state[11:11+len(recent_returns)] = recent_returns
                    
            if "volume_profile" in market_data:
                volume_profile = market_data["volume_profile"]
                normalized_profile = volume_profile / np.sum(volume_profile)
                state[16:16+len(normalized_profile)] = normalized_profile[:min(4, len(normalized_profile))]
                
            return state
        except Exception as e:
            logger.error(f"Error extracting state: {str(e)}")
            return np.zeros(self.state_dim)
            
    def save(self, filepath: str) -> bool:
        """
        Save the agent to disk.
        
        Parameters:
        - filepath: Path to save the agent
        
        Returns:
        - Success status
        """
        if not TF_AVAILABLE or self.mirror_network is None:
            logger.error("Networks not available. Cannot save.")
            return False
            
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            self.mirror_network.save(f"{filepath}_mirror")
            
            if self.outcome_network is not None:
                self.outcome_network.save(f"{filepath}_outcome")
            
            metadata = {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "adaptation_rate": self.adaptation_rate,
                "confidence_threshold": self.confidence_threshold,
                "total_observations": self.total_observations,
                "successful_mirrors": self.successful_mirrors,
                "total_actions": self.total_actions,
                "total_rewards": self.total_rewards,
                "saved_at": datetime.now().isoformat()
            }
            
            with open(f"{filepath}_metadata.json", 'w') as f:
                json.dump(metadata, f)
                
            logger.info(f"Agent saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving agent: {str(e)}")
            return False
            
    @classmethod
    def load(cls, filepath: str) -> 'MirrorAI':
        """
        Load an agent from disk.
        
        Parameters:
        - filepath: Path to the saved agent
        
        Returns:
        - Loaded MirrorAI agent
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available. Cannot load agent.")
            return None
            
        try:
            with open(f"{filepath}_metadata.json", 'r') as f:
                metadata = json.load(f)
                
            instance = cls(
                state_dim=metadata["state_dim"],
                action_dim=metadata["action_dim"],
                learning_rate=metadata["learning_rate"],
                batch_size=metadata["batch_size"],
                adaptation_rate=metadata["adaptation_rate"],
                confidence_threshold=metadata["confidence_threshold"]
            )
            
            instance.mirror_network = tf.keras.models.load_model(f"{filepath}_mirror")
            
            try:
                instance.outcome_network = tf.keras.models.load_model(f"{filepath}_outcome")
            except:
                logger.warning("Outcome network not found. Using only mirror network.")
            
            instance.total_observations = metadata["total_observations"]
            instance.successful_mirrors = metadata["successful_mirrors"]
            instance.total_actions = metadata["total_actions"]
            instance.total_rewards = metadata["total_rewards"]
            
            logger.info(f"Agent loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"Error loading agent: {str(e)}")
            return None
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the agent.
        
        Returns:
        - Dictionary with performance metrics
        """
        mirror_accuracy = np.mean(self.mirror_accuracy) if self.mirror_accuracy else 0.0
        success_rate = self.successful_mirrors / max(1, self.total_actions)
        
        return {
            "total_observations": self.total_observations,
            "total_actions": self.total_actions,
            "total_rewards": self.total_rewards,
            "mirror_accuracy": mirror_accuracy,
            "successful_mirrors": self.successful_mirrors,
            "success_rate": success_rate,
            "adaptation_rate": self.adaptation_rate
        }
        
    def battle_test(self, aggressor_agent: Any, env, episodes: int = 100, max_steps: int = 200) -> Dict[str, Any]:
        """
        Battle test against an aggressor agent in the given environment.
        
        Parameters:
        - aggressor_agent: Aggressor AI agent to battle against
        - env: Environment with step(action) and reset() methods
        - episodes: Number of episodes to test
        - max_steps: Maximum steps per episode
        
        Returns:
        - Battle test results
        """
        if not TF_AVAILABLE or self.mirror_network is None:
            logger.error("Networks not available. Cannot battle test.")
            return {"error": "Networks not available"}
            
        try:
            mirror_rewards = []
            aggressor_rewards = []
            win_count = 0
            
            for episode in range(episodes):
                state = env.reset()
                mirror_episode_reward = 0
                aggressor_episode_reward = 0
                
                for step in range(max_steps):
                    aggressor_action = aggressor_agent.select_action(state)
                    
                    self.observe(state, aggressor_action, 0, state)  # Placeholder for outcome
                    
                    mirror_action, _ = self.mirror_action(state)
                    
                    next_state, mirror_reward, aggressor_reward, done = env.step_dual(mirror_action, aggressor_action)
                    
                    self.observe(state, aggressor_action, aggressor_reward, next_state)
                    
                    mirror_episode_reward += mirror_reward
                    aggressor_episode_reward += aggressor_reward
                    
                    state = next_state
                    
                    if done:
                        break
                        
                mirror_rewards.append(mirror_episode_reward)
                aggressor_rewards.append(aggressor_episode_reward)
                
                if mirror_episode_reward > aggressor_episode_reward:
                    win_count += 1
                    self.successful_mirrors += 1
                    
                if (episode + 1) % 10 == 0:
                    logger.info(f"Episode {episode + 1}/{episodes}, Mirror: {np.mean(mirror_rewards[-10:]):.2f}, Aggressor: {np.mean(aggressor_rewards[-10:]):.2f}")
                    
            win_rate = win_count / episodes
            
            self.total_actions += episodes
            self.total_rewards += sum(mirror_rewards)
            
            return {
                "mirror_rewards": mirror_rewards,
                "aggressor_rewards": aggressor_rewards,
                "win_rate": win_rate,
                "mirror_avg_reward": np.mean(mirror_rewards),
                "aggressor_avg_reward": np.mean(aggressor_rewards)
            }
        except Exception as e:
            logger.error(f"Error during battle test: {str(e)}")
            return {"error": str(e)}

if __name__ == "__main__":
    agent = MirrorAI(state_dim=20, action_dim=5)
    
    market_data = {
        "orderbook": {
            "bids": [(100.0, 10.0), (99.0, 20.0), (98.0, 30.0), (97.0, 40.0), (96.0, 50.0)],
            "asks": [(101.0, 10.0), (102.0, 20.0), (103.0, 30.0), (104.0, 40.0), (105.0, 50.0)]
        },
        "price_history": [95.0, 96.0, 97.0, 98.0, 99.0, 100.0],
        "volume_profile": [100, 200, 300, 400]
    }
    
    observed_actions = [
        {
            "state": np.random.rand(20),
            "action_idx": 1,  # passive_buy
            "outcome": 0.5,
            "next_state": np.random.rand(20)
        },
        {
            "state": np.random.rand(20),
            "action_idx": 0,  # aggressive_buy
            "outcome": 1.0,
            "next_state": np.random.rand(20)
        }
    ]
    
    result = agent.mirror_liquidity(market_data, observed_actions)
    print(f"Action: {result['action']}, Confidence: {result['confidence']:.2f}")
