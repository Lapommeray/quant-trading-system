"""
Aggressor AI Module

This module implements the Aggressor AI agent for the Liquidity Thunderdome,
which aggressively hunts for liquidity using reinforcement learning techniques.
It is designed to identify and exploit liquidity pockets in various market conditions.

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
        logging.FileHandler('aggressor_ai.log')
    ]
)

logger = logging.getLogger("AggressorAI")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    logger.info("TensorFlow loaded successfully")
except ImportError:
    logger.warning("TensorFlow not available. Install with: pip install tensorflow")
    TF_AVAILABLE = False

class AggressorMemory:
    """Memory buffer for experience replay in reinforcement learning"""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the memory buffer.
        
        Parameters:
        - capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition in the buffer.
        
        Parameters:
        - state: Current state
        - action: Action taken
        - reward: Reward received
        - next_state: Next state
        - done: Whether the episode is done
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions from the buffer.
        
        Parameters:
        - batch_size: Number of transitions to sample
        
        Returns:
        - Batch of transitions
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
        
    def __len__(self) -> int:
        """Return the current size of the buffer"""
        return len(self.buffer)

class AggressorAI:
    """
    Aggressor AI agent for the Liquidity Thunderdome.
    Uses reinforcement learning to hunt for liquidity in the market.
    """
    
    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 5,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_capacity: int = 10000,
        batch_size: int = 64,
        update_frequency: int = 10
    ):
        """
        Initialize the Aggressor AI agent.
        
        Parameters:
        - state_dim: Dimension of the state space
        - action_dim: Dimension of the action space
        - learning_rate: Learning rate for the optimizer
        - gamma: Discount factor for future rewards
        - epsilon_start: Initial exploration rate
        - epsilon_end: Final exploration rate
        - epsilon_decay: Rate at which epsilon decays
        - memory_capacity: Capacity of the replay buffer
        - batch_size: Batch size for training
        - update_frequency: Frequency of target network updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        
        self.memory = AggressorMemory(capacity=memory_capacity)
        self.steps = 0
        self.training_enabled = True
        
        self.policy_network = None
        self.target_network = None
        
        if TF_AVAILABLE:
            self._build_networks()
        else:
            logger.error("TensorFlow not available. Cannot build networks.")
            
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []
        
        self.total_rewards = 0
        self.total_actions = 0
        self.successful_hunts = 0
        
        logger.info("AggressorAI initialized")
        
    def _build_networks(self) -> None:
        """Build the policy and target networks"""
        try:
            self.policy_network = self._create_network()
            
            self.target_network = self._create_network()
            
            self.target_network.set_weights(self.policy_network.get_weights())
            
            logger.info("Networks built successfully")
        except Exception as e:
            logger.error(f"Error building networks: {str(e)}")
            
    def _create_network(self) -> Model:
        """
        Create a neural network for Q-learning.
        
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
        
        outputs = Dense(self.action_dim, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
        
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Parameters:
        - state: Current state
        - explore: Whether to use exploration
        
        Returns:
        - Selected action
        """
        if not TF_AVAILABLE or self.policy_network is None:
            return random.randint(0, self.action_dim - 1)
            
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = np.expand_dims(state, axis=0)
            q_values = self.policy_network.predict(state_tensor, verbose=0)[0]
            return np.argmax(q_values)
            
    def update(self) -> float:
        """
        Update the policy network using experience replay.
        
        Returns:
        - Loss value
        """
        if not TF_AVAILABLE or self.policy_network is None or len(self.memory) < self.batch_size:
            return 0.0
            
        if not self.training_enabled:
            return 0.0
            
        try:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            states = np.array(states)
            actions = np.array(actions, dtype=np.int32)
            rewards = np.array(rewards, dtype=np.float32)
            next_states = np.array(next_states)
            dones = np.array(dones, dtype=np.float32)
            
            next_q_values = self.target_network.predict(next_states, verbose=0)
            max_next_q = np.max(next_q_values, axis=1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
            current_q = self.policy_network.predict(states, verbose=0)
            
            for i in range(self.batch_size):
                current_q[i, actions[i]] = target_q[i]
                
            history = self.policy_network.fit(
                states, current_q,
                epochs=1,
                batch_size=self.batch_size,
                verbose=0
            )
            
            loss = history.history['loss'][0]
            self.loss_history.append(loss)
            
            self.steps += 1
            if self.steps % self.update_frequency == 0:
                self.target_network.set_weights(self.policy_network.get_weights())
                
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.epsilon_history.append(self.epsilon)
            
            return loss
        except Exception as e:
            logger.error(f"Error updating networks: {str(e)}")
            return 0.0
            
    def train(self, env, episodes: int = 1000, max_steps: int = 200, render: bool = False) -> Dict[str, List[float]]:
        """
        Train the agent in the given environment.
        
        Parameters:
        - env: Environment with step(action) and reset() methods
        - episodes: Number of episodes to train
        - max_steps: Maximum steps per episode
        - render: Whether to render the environment
        
        Returns:
        - Training history
        """
        if not TF_AVAILABLE or self.policy_network is None:
            logger.error("Networks not available. Cannot train.")
            return {"error": "Networks not available"}
            
        episode_rewards = []
        
        try:
            for episode in range(episodes):
                state = env.reset()
                episode_reward = 0
                
                for step in range(max_steps):
                    action = self.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    
                    self.memory.push(state, action, reward, next_state, done)
                    
                    episode_reward += reward
                    self.total_rewards += reward
                    self.total_actions += 1
                    
                    state = next_state
                    
                    loss = self.update()
                    
                    if render:
                        env.render()
                        
                    if done:
                        break
                        
                episode_rewards.append(episode_reward)
                self.reward_history.append(episode_reward)
                
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
                    
                if episode_reward > 0:
                    self.successful_hunts += 1
                    
            success_rate = self.successful_hunts / episodes
            logger.info(f"Training completed. Success rate: {success_rate:.2f}")
            
            return {
                "rewards": episode_rewards,
                "losses": self.loss_history,
                "epsilons": self.epsilon_history,
                "success_rate": success_rate
            }
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return {"error": str(e)}
            
    def hunt_liquidity(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hunt for liquidity in the market using the trained agent.
        
        Parameters:
        - market_data: Dictionary containing market data
        
        Returns:
        - Action and confidence
        """
        if not TF_AVAILABLE or self.policy_network is None:
            logger.error("Networks not available. Cannot hunt liquidity.")
            return {"action": "none", "confidence": 0.0}
            
        try:
            state = self._extract_state(market_data)
            
            action = self.select_action(state, explore=False)
            
            state_tensor = np.expand_dims(state, axis=0)
            q_values = self.policy_network.predict(state_tensor, verbose=0)[0]
            confidence = float(q_values[action] / np.max(np.abs(q_values)))
            
            action_map = {
                0: "aggressive_buy",
                1: "passive_buy",
                2: "hold",
                3: "passive_sell",
                4: "aggressive_sell"
            }
            
            market_action = action_map.get(action, "hold")
            
            self.total_actions += 1
            
            return {
                "action": market_action,
                "confidence": confidence,
                "q_values": q_values.tolist(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error hunting liquidity: {str(e)}")
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
        if not TF_AVAILABLE or self.policy_network is None:
            logger.error("Networks not available. Cannot save.")
            return False
            
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            self.policy_network.save(f"{filepath}_policy")
            
            self.target_network.save(f"{filepath}_target")
            
            metadata = {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "batch_size": self.batch_size,
                "update_frequency": self.update_frequency,
                "steps": self.steps,
                "total_rewards": self.total_rewards,
                "total_actions": self.total_actions,
                "successful_hunts": self.successful_hunts,
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
    def load(cls, filepath: str) -> 'AggressorAI':
        """
        Load an agent from disk.
        
        Parameters:
        - filepath: Path to the saved agent
        
        Returns:
        - Loaded AggressorAI agent
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
                gamma=metadata["gamma"],
                epsilon_start=metadata["epsilon"],
                epsilon_end=metadata["epsilon_end"],
                epsilon_decay=metadata["epsilon_decay"],
                batch_size=metadata["batch_size"],
                update_frequency=metadata["update_frequency"]
            )
            
            instance.policy_network = tf.keras.models.load_model(f"{filepath}_policy")
            
            instance.target_network = tf.keras.models.load_model(f"{filepath}_target")
            
            instance.steps = metadata["steps"]
            instance.total_rewards = metadata["total_rewards"]
            instance.total_actions = metadata["total_actions"]
            instance.successful_hunts = metadata["successful_hunts"]
            
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
        avg_reward = self.total_rewards / max(1, self.total_actions)
        success_rate = self.successful_hunts / max(1, len(self.reward_history))
        
        return {
            "total_rewards": self.total_rewards,
            "total_actions": self.total_actions,
            "average_reward": avg_reward,
            "successful_hunts": self.successful_hunts,
            "success_rate": success_rate,
            "epsilon": self.epsilon
        }

if __name__ == "__main__":
    agent = AggressorAI(state_dim=20, action_dim=5)
    
    market_data = {
        "orderbook": {
            "bids": [(100.0, 10.0), (99.0, 20.0), (98.0, 30.0), (97.0, 40.0), (96.0, 50.0)],
            "asks": [(101.0, 10.0), (102.0, 20.0), (103.0, 30.0), (104.0, 40.0), (105.0, 50.0)]
        },
        "price_history": [95.0, 96.0, 97.0, 98.0, 99.0, 100.0],
        "volume_profile": [100, 200, 300, 400]
    }
    
    result = agent.hunt_liquidity(market_data)
    print(f"Action: {result['action']}, Confidence: {result['confidence']:.2f}")
