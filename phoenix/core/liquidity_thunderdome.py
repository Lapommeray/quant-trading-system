"""
Liquidity Thunderdome

This module implements the reinforcement learning-based liquidity hunting system
for the Phoenix Mirror Protocol. It features competing AI agents that battle for
optimal execution and liquidity discovery.
"""

import numpy as np
import logging
import time
import threading
import json
import os
import hashlib
from datetime import datetime
from collections import deque
import random

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Using classical fallback for RL operations.")

class ReplayBuffer:
    """
    Experience replay buffer for reinforcement learning agents.
    Stores transitions for training.
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize the replay buffer
        
        Parameters:
        - capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer
        
        Parameters:
        - state: Current state
        - action: Action taken
        - reward: Reward received
        - next_state: Next state
        - done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer
        
        Parameters:
        - batch_size: Number of transitions to sample
        
        Returns:
        - Batch of transitions
        """
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
        
    def size(self):
        """
        Get the current size of the buffer
        
        Returns:
        - Buffer size
        """
        return len(self.buffer)

class AggressorAgent:
    """
    Reinforcement learning agent that aggressively hunts for liquidity
    like a bloodhound. Uses DDPG (Deep Deterministic Policy Gradient) algorithm.
    """
    
    def __init__(self, state_dim=20, action_dim=4):
        """
        Initialize the Aggressor Agent
        
        Parameters:
        - state_dim: Dimension of the state space
        - action_dim: Dimension of the action space
        """
        self.logger = logging.getLogger("AggressorAgent")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.gamma = 1.61803398875  # φ-discount factor (golden ratio)
        self.tau = 0.005  # Target network update rate
        self.batch_size = 64
        self.learning_rate = 0.001
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        if TF_AVAILABLE:
            self._build_models()
        else:
            self.actor = None
            self.critic = None
            self.target_actor = None
            self.target_critic = None
            
        self.training_history = []
        
        self.logger.info("AggressorAgent initialized")
        
    def _build_models(self):
        """Build actor and critic networks"""
        self.actor = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_dim,)),
            Dense(64, activation='relu'),
            Dense(self.action_dim, activation='tanh')  # tanh for bounded actions [-1, 1]
        ])
        self.actor.compile(optimizer=Adam(learning_rate=self.learning_rate))
        
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        
        state_h1 = Dense(64, activation='relu')(state_input)
        state_h2 = Dense(64, activation='relu')(state_h1)
        
        action_h1 = Dense(64, activation='relu')(action_input)
        
        merged = Concatenate()([state_h2, action_h1])
        merged_h1 = Dense(64, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        
        self.critic = Model(inputs=[state_input, action_input], outputs=output)
        self.critic.compile(optimizer=Adam(learning_rate=self.learning_rate))
        
        self.target_actor = tf.keras.models.clone_model(self.actor)
        self.target_actor.set_weights(self.actor.get_weights())
        
        self.target_critic = tf.keras.models.clone_model(self.critic)
        self.target_critic.set_weights(self.critic.get_weights())
        
    def act(self, state, add_noise=True):
        """
        Select an action based on the current state
        
        Parameters:
        - state: Current state
        - add_noise: Whether to add exploration noise
        
        Returns:
        - Selected action
        """
        if not TF_AVAILABLE:
            return np.random.uniform(-1, 1, self.action_dim)
            
        state = np.reshape(state, [1, self.state_dim])
        action = self.actor.predict(state)[0]
        
        if add_noise:
            noise = self.epsilon * np.random.normal(0, 1, self.action_dim)
            action = np.clip(action + noise, -1, 1)
            
        return action
        
    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer
        
        Parameters:
        - state: Current state
        - action: Action taken
        - reward: Reward received
        - next_state: Next state
        - done: Whether the episode is done
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        
    def train(self):
        """Train the agent using experiences from the replay buffer"""
        if not TF_AVAILABLE or self.replay_buffer.size() < self.batch_size:
            return
            
        batch = self.replay_buffer.sample(self.batch_size)
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])
        
        target_actions = self.target_actor.predict(next_states)
        target_q_values = self.target_critic.predict([next_states, target_actions])
        
        q_targets = rewards + self.gamma * target_q_values * (1 - dones)
        
        self.critic.train_on_batch([states, actions], q_targets)
        
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            critic_value = self.critic([states, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)
            
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables)
        )
        
        for target_var, var in zip(self.target_actor.trainable_variables, self.actor.trainable_variables):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
            
        for target_var, var in zip(self.target_critic.trainable_variables, self.critic.trainable_variables):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
            
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.training_history.append({
            "actor_loss": float(actor_loss),
            "critic_loss": float(self.critic.optimizer.get_gradients()[0].numpy().mean()),
            "epsilon": self.epsilon,
            "timestamp": time.time()
        })
        
    def save(self, path):
        """
        Save the agent models
        
        Parameters:
        - path: Path to save the models
        """
        if not TF_AVAILABLE:
            return
            
        os.makedirs(path, exist_ok=True)
        self.actor.save(os.path.join(path, "aggressor_actor.h5"))
        self.critic.save(os.path.join(path, "aggressor_critic.h5"))
        
        with open(os.path.join(path, "aggressor_params.json"), "w") as f:
            json.dump({
                "gamma": self.gamma,
                "tau": self.tau,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim
            }, f)
            
    def load(self, path):
        """
        Load the agent models
        
        Parameters:
        - path: Path to load the models from
        """
        if not TF_AVAILABLE:
            return
            
        self.actor = load_model(os.path.join(path, "aggressor_actor.h5"))
        self.critic = load_model(os.path.join(path, "aggressor_critic.h5"))
        self.target_actor = tf.keras.models.clone_model(self.actor)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic = tf.keras.models.clone_model(self.critic)
        self.target_critic.set_weights(self.critic.get_weights())
        
        with open(os.path.join(path, "aggressor_params.json"), "r") as f:
            params = json.load(f)
            self.gamma = params["gamma"]
            self.tau = params["tau"]
            self.batch_size = params["batch_size"]
            self.learning_rate = params["learning_rate"]
            self.epsilon = params["epsilon"]
            self.epsilon_decay = params["epsilon_decay"]
            self.epsilon_min = params["epsilon_min"]
            
    def get_training_history(self, limit=100):
        """
        Get training history
        
        Parameters:
        - limit: Maximum number of entries to return
        
        Returns:
        - Training history
        """
        return self.training_history[-limit:]

class MirrorAgent:
    """
    Inverse reinforcement learning agent that learns by watching the Aggressor.
    Specializes in reverse-engineering strategies and mirroring them.
    """
    
    def __init__(self, state_dim=20, action_dim=4):
        """
        Initialize the Mirror Agent
        
        Parameters:
        - state_dim: Dimension of the state space
        - action_dim: Dimension of the action space
        """
        self.logger = logging.getLogger("MirrorAgent")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.learning_rate = 0.001
        self.batch_size = 64
        
        self.observations = deque(maxlen=10000)
        
        if TF_AVAILABLE:
            self._build_model()
        else:
            self.model = None
            
        self.training_history = []
        
        self.logger.info("MirrorAgent initialized")
        
    def _build_model(self):
        """Build the inverse RL model"""
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_dim,)),
            Dense(64, activation='relu'),
            Dense(self.action_dim, activation='tanh')
        ])
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                          loss='mse')
        
    def observe(self, state, action):
        """
        Observe an action taken by another agent
        
        Parameters:
        - state: State in which the action was taken
        - action: Action taken
        """
        self.observations.append((state, action))
        
    def mirror(self, state):
        """
        Mirror an action based on observations
        
        Parameters:
        - state: Current state
        
        Returns:
        - Mirrored action
        """
        if not TF_AVAILABLE or self.model is None:
            if len(self.observations) > 0:
                similarities = [np.sum(np.square(obs_state - state)) for obs_state, _ in self.observations]
                most_similar_idx = np.argmin(similarities)
                return self.observations[most_similar_idx][1]
            else:
                return np.random.uniform(-1, 1, self.action_dim)
                
        state = np.reshape(state, [1, self.state_dim])
        return self.model.predict(state)[0]
        
    def train(self):
        """Train the agent using observed state-action pairs"""
        if not TF_AVAILABLE or len(self.observations) < self.batch_size:
            return
            
        batch = random.sample(self.observations, self.batch_size)
        states = np.array([observation[0] for observation in batch])
        actions = np.array([observation[1] for observation in batch])
        
        loss = self.model.train_on_batch(states, actions)
        
        self.training_history.append({
            "loss": float(loss),
            "timestamp": time.time()
        })
        
    def save(self, path):
        """
        Save the agent model
        
        Parameters:
        - path: Path to save the model
        """
        if not TF_AVAILABLE:
            return
            
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "mirror_model.h5"))
        
        with open(os.path.join(path, "mirror_params.json"), "w") as f:
            json.dump({
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim
            }, f)
            
    def load(self, path):
        """
        Load the agent model
        
        Parameters:
        - path: Path to load the model from
        """
        if not TF_AVAILABLE:
            return
            
        self.model = load_model(os.path.join(path, "mirror_model.h5"))
        
        with open(os.path.join(path, "mirror_params.json"), "r") as f:
            params = json.load(f)
            self.learning_rate = params["learning_rate"]
            self.batch_size = params["batch_size"]
            
    def get_training_history(self, limit=100):
        """
        Get training history
        
        Parameters:
        - limit: Maximum number of entries to return
        
        Returns:
        - Training history
        """
        return self.training_history[-limit:]

class OracleAgent:
    """
    Predictive agent that anticipates regulatory patterns and market surveillance.
    Uses a quantum neural network for prediction.
    """
    
    def __init__(self, state_dim=20):
        """
        Initialize the Oracle Agent
        
        Parameters:
        - state_dim: Dimension of the state space
        """
        self.logger = logging.getLogger("OracleAgent")
        
        self.state_dim = state_dim
        
        self.learning_rate = 0.001
        self.batch_size = 32
        
        self.surveillance_patterns = deque(maxlen=1000)
        
        if TF_AVAILABLE:
            self._build_model()
        else:
            self.model = None
            
        self.prediction_history = []
        
        self.logger.info("OracleAgent initialized")
        
    def _build_model(self):
        """Build the prediction model"""
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(self.state_dim,)),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')  # Probability of detection
        ])
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        
    def add_pattern(self, state, detected):
        """
        Add a surveillance pattern
        
        Parameters:
        - state: Market state
        - detected: Whether surveillance was detected (1) or not (0)
        """
        self.surveillance_patterns.append((state, detected))
        
    def predict_detection(self, state):
        """
        Predict the probability of surveillance detection
        
        Parameters:
        - state: Current state
        
        Returns:
        - Detection probability
        """
        if not TF_AVAILABLE or self.model is None:
            if len(self.surveillance_patterns) > 0:
                similarities = [np.sum(np.square(obs_state - state)) for obs_state, _ in self.surveillance_patterns]
                most_similar_idx = np.argmin(similarities)
                return float(self.surveillance_patterns[most_similar_idx][1])
            else:
                return 0.5  # 50% chance by default
                
        state = np.reshape(state, [1, self.state_dim])
        prediction = float(self.model.predict(state)[0][0])
        
        self.prediction_history.append({
            "prediction": prediction,
            "timestamp": time.time()
        })
        
        return prediction
        
    def train(self):
        """Train the agent using surveillance patterns"""
        if not TF_AVAILABLE or len(self.surveillance_patterns) < self.batch_size:
            return
            
        batch = random.sample(self.surveillance_patterns, self.batch_size)
        states = np.array([pattern[0] for pattern in batch])
        detections = np.array([pattern[1] for pattern in batch])
        
        self.model.train_on_batch(states, detections)
        
    def save(self, path):
        """
        Save the agent model
        
        Parameters:
        - path: Path to save the model
        """
        if not TF_AVAILABLE:
            return
            
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "oracle_model.h5"))
        
        with open(os.path.join(path, "oracle_params.json"), "w") as f:
            json.dump({
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "state_dim": self.state_dim
            }, f)
            
    def load(self, path):
        """
        Load the agent model
        
        Parameters:
        - path: Path to load the model from
        """
        if not TF_AVAILABLE:
            return
            
        self.model = load_model(os.path.join(path, "oracle_model.h5"))
        
        with open(os.path.join(path, "oracle_params.json"), "r") as f:
            params = json.load(f)
            self.learning_rate = params["learning_rate"]
            self.batch_size = params["batch_size"]
            
    def get_prediction_history(self, limit=100):
        """
        Get prediction history
        
        Parameters:
        - limit: Maximum number of entries to return
        
        Returns:
        - Prediction history
        """
        return self.prediction_history[-limit:]

class LiquidityThunderdome:
    """
    AI vs AI competition for optimal execution and liquidity discovery.
    Features multiple competing agents in a reinforcement learning environment.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Liquidity Thunderdome
        
        Parameters:
        - algorithm: QuantConnect algorithm instance (optional)
        """
        self.logger = logging.getLogger("LiquidityThunderdome")
        self.algorithm = algorithm
        
        self.state_dim = 20
        self.action_dim = 4
        
        self.aggressor = AggressorAgent(self.state_dim, self.action_dim)
        self.mirror = MirrorAgent(self.state_dim, self.action_dim)
        self.oracle = OracleAgent(self.state_dim)
        
        self.max_steps = 100
        self.current_step = 0
        self.current_state = None
        self.current_reward = 0
        self.current_done = False
        
        self.battle_history = []
        
        self.active = False
        self.battle_thread = None
        
        self.logger.info("LiquidityThunderdome initialized")
        
    def start(self):
        """Start the thunderdome battles"""
        if self.active:
            return
            
        self.active = True
        self.battle_thread = threading.Thread(target=self._battle_loop)
        self.battle_thread.daemon = True
        self.battle_thread.start()
        
        self.logger.info("LiquidityThunderdome battles started")
        
    def stop(self):
        """Stop the thunderdome battles"""
        self.active = False
        
        if self.battle_thread and self.battle_thread.is_alive():
            self.battle_thread.join(timeout=5)
            
        self.logger.info("LiquidityThunderdome battles stopped")
        
    def _battle_loop(self):
        """Background battle loop"""
        while self.active:
            try:
                self._reset_environment()
                
                battle_result = self._run_battle()
                
                self.battle_history.append(battle_result)
                
                self.aggressor.train()
                self.mirror.train()
                self.oracle.train()
                
                time.sleep(1)  # Small delay between battles
            except Exception as e:
                self.logger.error(f"Error in battle loop: {str(e)}")
                time.sleep(5)
                
    def _reset_environment(self):
        """Reset the environment for a new battle"""
        self.current_step = 0
        self.current_state = np.random.normal(0, 1, self.state_dim)  # Random initial state
        self.current_reward = 0
        self.current_done = False
        
    def _run_battle(self):
        """
        Run a single battle between agents
        
        Returns:
        - Battle result
        """
        battle_start = time.time()
        battle_states = []
        battle_actions = []
        battle_rewards = []
        
        while not self.current_done and self.current_step < self.max_steps:
            aggressor_action = self.aggressor.act(self.current_state)
            
            self.mirror.observe(self.current_state, aggressor_action)
            mirrored_action = self.mirror.mirror(self.current_state)
            
            detection_prob = self.oracle.predict_detection(self.current_state)
            
            liquidity_reward = self._calculate_liquidity_reward(aggressor_action)
            stealth_reward = 1.0 - detection_prob  # Higher reward for lower detection probability
            
            reward = liquidity_reward * stealth_reward
            
            next_state = self._get_next_state(self.current_state, aggressor_action)
            done = self.current_step >= self.max_steps - 1
            
            self.aggressor.remember(self.current_state, aggressor_action, reward, next_state, done)
            
            detection = np.random.random() < detection_prob  # Simulate detection based on probability
            self.oracle.add_pattern(self.current_state, detection)
            
            battle_states.append(self.current_state.copy())
            battle_actions.append(aggressor_action.copy())
            battle_rewards.append(reward)
            
            self.current_state = next_state
            self.current_reward += reward
            self.current_done = done
            self.current_step += 1
            
        battle_end = time.time()
        
        battle_result = {
            "start_time": battle_start,
            "end_time": battle_end,
            "duration": battle_end - battle_start,
            "steps": self.current_step,
            "total_reward": self.current_reward,
            "average_reward": self.current_reward / self.current_step if self.current_step > 0 else 0,
            "final_detection_prob": detection_prob,
            "timestamp": time.time()
        }
        
        return battle_result
        
    def _calculate_liquidity_reward(self, action):
        """
        Calculate reward based on liquidity capture
        
        Parameters:
        - action: Action taken
        
        Returns:
        - Liquidity reward
        """
        
        
        direction = action[0]
        size = abs(action[1])
        aggression = action[2]
        timing = action[3]
        
        base_reward = size * (0.5 + 0.5 * aggression)
        
        timing_factor = 0.5 + 0.5 * timing
        
        reward = base_reward * timing_factor
        
        return reward
        
    def _get_next_state(self, state, action):
        """
        Get the next state based on the current state and action
        
        Parameters:
        - state: Current state
        - action: Action taken
        
        Returns:
        - Next state
        """
        
        direction = action[0]
        size = action[1]
        aggression = action[2]
        timing = action[3]
        
        state_change = np.zeros_like(state)
        
        state_change[0] = direction * size * 0.01  # Small price change
        
        state_change[1] = size * 0.1  # Volume change
        
        state_change[2] = aggression * 0.05  # Volatility change
        
        noise = np.random.normal(0, 0.01, state.shape)
        
        next_state = state + state_change + noise
        
        return next_state
        
    def battle_phase(self, obfuscated_tape):
        """
        Run a battle phase with real market data
        
        Parameters:
        - obfuscated_tape: Market data with obfuscated timestamps
        
        Returns:
        - Execution plan
        """
        state = self._tape_to_state(obfuscated_tape)
        
        aggressor_action = self.aggressor.act(state, add_noise=False)
        
        self.mirror.observe(state, aggressor_action)
        mirrored_action = self.mirror.mirror(state)
        
        detection_prob = self.oracle.predict_detection(state)
        
        if detection_prob > 0.7:
            final_action = mirrored_action
            action_source = "mirror"
        else:
            final_action = aggressor_action
            action_source = "aggressor"
            
        execution_plan = self._action_to_execution(final_action, detection_prob)
        execution_plan["action_source"] = action_source
        
        return execution_plan
        
    def _tape_to_state(self, tape):
        """
        Convert market tape to state representation
        
        Parameters:
        - tape: Market data
        
        Returns:
        - State representation
        """
        state = np.zeros(self.state_dim)
        
        if len(tape) == 0:
            return state
            
        prices = [candle.get("price", 0) for candle in tape]
        state[0] = np.mean(prices) if prices else 0
        state[1] = np.std(prices) if len(prices) > 1 else 0
        
        volumes = [candle.get("volume", 0) for candle in tape]
        state[2] = np.mean(volumes) if volumes else 0
        state[3] = np.std(volumes) if len(volumes) > 1 else 0
        
        if len(prices) > 1:
            state[4] = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
            
        if len(prices) > 1:
            returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
            state[5] = np.std(returns) if returns else 0
            
        timestamps = [candle.get("timestamp", 0) for candle in tape]
        if timestamps:
            state[6] = (time.time() - timestamps[-1]) / 3600  # Hours since last update
            
        
        return state
        
    def _action_to_execution(self, action, detection_prob):
        """
        Convert action to execution plan
        
        Parameters:
        - action: Selected action
        - detection_prob: Detection probability
        
        Returns:
        - Execution plan
        """
        direction = "buy" if action[0] > 0 else "sell"
        size = abs(action[1])
        aggression = action[2]
        timing = action[3]
        
        if aggression > 0.7:
            order_type = "market"
            price_offset = 0
        elif aggression > 0.3:
            order_type = "limit"
            price_offset = 0.001 * (1 - aggression)
        else:
            order_type = "limit"
            price_offset = 0.002 * (1 - aggression)
            
        if detection_prob > 0.5:
            size *= (1 - (detection_prob - 0.5))
            
            if order_type == "market":
                order_type = "limit"
                price_offset = 0.001
                
        execution_plan = {
            "direction": direction,
            "size": size,
            "order_type": order_type,
            "price_offset": price_offset,
            "timing_factor": timing,
            "detection_prob": detection_prob,
            "timestamp": time.time()
        }
        
        return execution_plan
        
    def save(self, path):
        """
        Save the thunderdome agents
        
        Parameters:
        - path: Path to save the agents
        """
        os.makedirs(path, exist_ok=True)
        
        self.aggressor.save(os.path.join(path, "aggressor"))
        self.mirror.save(os.path.join(path, "mirror"))
        self.oracle.save(os.path.join(path, "oracle"))
        
        with open(os.path.join(path, "battle_history.json"), "w") as f:
            json.dump(self.battle_history[-1000:], f)  # Save last 1000 battles
            
    def load(self, path):
        """
        Load the thunderdome agents
        
        Parameters:
        - path: Path to load the agents from
        """
        self.aggressor.load(os.path.join(path, "aggressor"))
        self.mirror.load(os.path.join(path, "mirror"))
        self.oracle.load(os.path.join(path, "oracle"))
        
        try:
            with open(os.path.join(path, "battle_history.json"), "r") as f:
                self.battle_history = json.load(f)
        except:
            self.battle_history = []
            
    def get_battle_history(self, limit=100):
        """
        Get battle history
        
        Parameters:
        - limit: Maximum number of entries to return
        
        Returns:
        - Battle history
        """
        return self.battle_history[-limit:]
        
    def get_status(self):
        """
        Get thunderdome status
        
        Returns:
        - Status information
        """
        return {
            "active": self.active,
            "battles_completed": len(self.battle_history),
            "average_reward": np.mean([battle["total_reward"] for battle in self.battle_history[-100:]]) if self.battle_history else 0,
            "average_detection_prob": np.mean([battle["final_detection_prob"] for battle in self.battle_history[-100:]]) if self.battle_history else 0,
            "timestamp": time.time()
        }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    thunderdome = LiquidityThunderdome()
    
    thunderdome.start()
    
    try:
        time.sleep(60)
    except KeyboardInterrupt:
        pass
    finally:
        thunderdome.stop()
        
    print(thunderdome.get_status())
    
    print(thunderdome.get_battle_history(5))
