"""
ML-Driven Circuit Breaker Tuner Module

This module implements the ML-Driven Circuit Breaker Tuner for the QMP Overrider system.
It uses Graph Attention Networks (GATv2Conv) to analyze market structure and dynamically
adjust circuit breaker thresholds based on real-time conditions.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class MarketStructureGraph:
    """Market Structure Graph for the ML-Driven Circuit Breaker Tuner."""
    
    def __init__(self, max_nodes=100):
        self.max_nodes = max_nodes
        self.node_features = np.zeros((max_nodes, 10))
        self.edge_index = []
        self.edge_attr = []
        self.node_types = {}
        self.node_mapping = {}
        self.next_node_id = 0
    
    def add_node(self, node_type, features):
        if self.next_node_id >= self.max_nodes:
            self._remove_oldest_node()
        
        node_id = self.next_node_id
        self.node_features[node_id] = features
        self.node_types[node_id] = node_type
        self.node_mapping[f"{node_type}_{len([n for n in self.node_types.values() if n == node_type])}"] = node_id
        self.next_node_id += 1
        
        return node_id
    
    def add_edge(self, source_id, target_id, attr=None):
        if attr is None:
            attr = [1.0]
        
        self.edge_index.append((source_id, target_id))
        self.edge_attr.append(attr)
    
    def _remove_oldest_node(self):
        self.node_features[:-1] = self.node_features[1:]
        self.node_features[-1] = np.zeros(10)
        
        new_node_types = {}
        new_node_mapping = {}
        
        for node_id, node_type in self.node_types.items():
            if node_id > 0:
                new_node_types[node_id - 1] = node_type
        
        for node_name, node_id in self.node_mapping.items():
            if node_id > 0:
                new_node_mapping[node_name] = node_id - 1
        
        self.node_types = new_node_types
        self.node_mapping = new_node_mapping
        
        new_edge_index = []
        new_edge_attr = []
        
        for i, (src, tgt) in enumerate(self.edge_index):
            if src > 0 and tgt > 0:
                new_edge_index.append((src - 1, tgt - 1))
                new_edge_attr.append(self.edge_attr[i])
        
        self.edge_index = new_edge_index
        self.edge_attr = new_edge_attr
        
        self.next_node_id -= 1
    
    def build_from_market_data(self, market_data):
        self.node_features = np.zeros((self.max_nodes, 10))
        self.edge_index = []
        self.edge_attr = []
        self.node_types = {}
        self.node_mapping = {}
        self.next_node_id = 0
        
        exchange_id = self.add_node('exchange', [
            market_data.get('exchange_latency', 0) / 100,
            market_data.get('exchange_volume', 0) / 10000,
            market_data.get('exchange_volatility', 0) / 0.1,
            market_data.get('exchange_liquidity', 0) / 1000000,
            market_data.get('exchange_spread', 0) / 0.01,
            1.0 if market_data.get('exchange_status', 'normal') == 'normal' else 0.0,
            market_data.get('exchange_trade_count', 0) / 1000,
            market_data.get('exchange_order_imbalance', 0) / 10,
            market_data.get('exchange_tick_size', 0) / 0.01,
            market_data.get('exchange_fee', 0) / 0.001
        ])
        
        assets = market_data.get('assets', [])
        asset_ids = []
        
        for i, asset in enumerate(assets):
            asset_id = self.add_node('asset', [
                asset.get('price', 0) / 1000,
                asset.get('volume', 0) / 1000,
                asset.get('volatility', 0) / 0.1,
                asset.get('bid_ask_spread', 0) / 0.01,
                asset.get('market_cap', 0) / 1000000000,
                asset.get('beta', 0) / 1.5,
                asset.get('correlation', 0),
                asset.get('momentum', 0) / 0.1,
                asset.get('rsi', 0) / 100,
                asset.get('liquidity', 0) / 1000000
            ])
            asset_ids.append(asset_id)
            
            self.add_edge(exchange_id, asset_id, [1.0])
            self.add_edge(asset_id, exchange_id, [1.0])
        
        regime_id = self.add_node('regime', [
            market_data.get('regime_volatility', 0) / 0.2,
            market_data.get('regime_trend', 0) / 0.1,
            market_data.get('regime_liquidity', 0) / 1000000,
            market_data.get('regime_correlation', 0),
            market_data.get('regime_sentiment', 0) / 100,
            market_data.get('regime_momentum', 0) / 0.1,
            market_data.get('regime_vix', 0) / 30,
            market_data.get('regime_yield_curve', 0) / 0.03,
            market_data.get('regime_macro_surprise', 0) / 0.1,
            market_data.get('regime_risk_premium', 0) / 0.05
        ])
        
        self.add_edge(regime_id, exchange_id, [1.0])
        self.add_edge(exchange_id, regime_id, [1.0])
        
        for asset_id in asset_ids:
            self.add_edge(regime_id, asset_id, [1.0])
            self.add_edge(asset_id, regime_id, [1.0])
        
        return self
    
    def to_torch_geometric(self):
        if not TORCH_AVAILABLE:
            return None
        
        x = torch.tensor(self.node_features[:self.next_node_id], dtype=torch.float)
        edge_index = torch.tensor(self.edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(self.edge_attr, dtype=torch.float)
        
        return {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr
        }

class GATModel(nn.Module):
    """Graph Attention Network for market structure analysis."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, dropout=0.1):
        super(GATModel, self).__init__()
        
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout)
        
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class MLCircuitBreakerTuner:
    """ML-Driven Circuit Breaker Tuner for the QMP Overrider system."""
    
    def __init__(self, model_dir=None):
        self.logger = logging.getLogger("MLCircuitBreakerTuner")
        
        if model_dir is None:
            self.model_dir = Path("models/circuit_breakers")
        else:
            self.model_dir = Path(model_dir)
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.optimizer = None
        self.device = None
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = GATModel(
                in_channels=10,
                hidden_channels=64,
                out_channels=4,  # 4 circuit breaker parameters
                num_heads=4,
                dropout=0.1
            ).to(self.device)
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            self._load_model()
        else:
            self.logger.warning("PyTorch not available, using fallback prediction")
        
        self.graph_builder = MarketStructureGraph(max_nodes=100)
        
        self.training_history = []
        self.prediction_history = []
        
        self.logger.info(f"ML Circuit Breaker Tuner initialized")
    
    def _load_model(self):
        """Load model from file if available."""
        model_file = self.model_dir / "gat_model.pt"
        optimizer_file = self.model_dir / "optimizer.pt"
        history_file = self.model_dir / "training_history.json"
        
        if model_file.exists() and optimizer_file.exists():
            try:
                self.model.load_state_dict(torch.load(model_file, map_location=self.device))
                self.optimizer.load_state_dict(torch.load(optimizer_file, map_location=self.device))
                self.logger.info(f"Loaded model from {model_file}")
                
                if history_file.exists():
                    with open(history_file, "r") as f:
                        self.training_history = json.load(f)
                    self.logger.info(f"Loaded training history with {len(self.training_history)} entries")
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
    
    def _save_model(self):
        """Save model to file."""
        if not TORCH_AVAILABLE or self.model is None:
            return
        
        model_file = self.model_dir / "gat_model.pt"
        optimizer_file = self.model_dir / "optimizer.pt"
        history_file = self.model_dir / "training_history.json"
        
        try:
            torch.save(self.model.state_dict(), model_file)
            torch.save(self.optimizer.state_dict(), optimizer_file)
            
            with open(history_file, "w") as f:
                json.dump(self.training_history, f, indent=2)
            
            self.logger.info(f"Saved model to {model_file}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def predict(self, market_data):
        """Predict optimal circuit breaker parameters."""
        self.graph_builder.build_from_market_data(market_data)
        
        if TORCH_AVAILABLE and self.model is not None:
            graph_data = self.graph_builder.to_torch_geometric()
            
            if graph_data is None:
                return self._fallback_prediction(market_data)
            
            x = graph_data['x'].to(self.device)
            edge_index = graph_data['edge_index'].to(self.device)
            
            self.model.eval()
            
            with torch.no_grad():
                prediction = self.model(x, edge_index)
                
                volatility_threshold = torch.sigmoid(prediction[0, 0]).item() * 0.2  # 0-20%
                latency_spike_ms = torch.exp(prediction[0, 1]).item() * 10  # 10-1000ms
                order_imbalance_ratio = torch.exp(prediction[0, 2]).item()  # 1-10
                cooling_period = torch.exp(prediction[0, 3]).item() * 10  # 10-1000s
        else:
            return self._fallback_prediction(market_data)
        
        params = {
            'volatility_threshold': volatility_threshold,
            'latency_spike_ms': latency_spike_ms,
            'order_imbalance_ratio': order_imbalance_ratio,
            'cooling_period': cooling_period
        }
        
        prediction_record = {
            'timestamp': time.time(),
            'params': params,
            'market_data_summary': {
                'exchange_latency': market_data.get('exchange_latency', 0),
                'exchange_volume': market_data.get('exchange_volume', 0),
                'exchange_volatility': market_data.get('exchange_volatility', 0),
                'regime_vix': market_data.get('regime_vix', 0)
            }
        }
        self.prediction_history.append(prediction_record)
        
        self.logger.info(f"Predicted circuit breaker parameters: {params}")
        
        return params
    
    def _fallback_prediction(self, market_data):
        """Fallback prediction when model is not available."""
        volatility_threshold = 0.08  # 8%
        latency_spike_ms = 50
        order_imbalance_ratio = 3.0
        cooling_period = 60  # seconds
        
        if 'exchange_volatility' in market_data:
            volatility = market_data['exchange_volatility']
            volatility_threshold = max(0.05, min(0.2, volatility * 1.5))
        
        if 'exchange_latency' in market_data:
            latency = market_data['exchange_latency']
            latency_spike_ms = max(20, min(200, latency * 2))
        
        if 'exchange_order_imbalance' in market_data:
            imbalance = market_data['exchange_order_imbalance']
            order_imbalance_ratio = max(1.5, min(5.0, imbalance * 1.2))
        
        if 'regime_vix' in market_data:
            vix = market_data['regime_vix']
            cooling_period = max(30, min(300, 30 + vix * 5))
        
        params = {
            'volatility_threshold': volatility_threshold,
            'latency_spike_ms': latency_spike_ms,
            'order_imbalance_ratio': order_imbalance_ratio,
            'cooling_period': cooling_period
        }
        
        return params
    
    def train(self, market_data, optimal_params):
        """Train the model with feedback."""
        if not TORCH_AVAILABLE or self.model is None:
            self.logger.warning("PyTorch not available, cannot train model")
            return None
        
        self.graph_builder.build_from_market_data(market_data)
        
        graph_data = self.graph_builder.to_torch_geometric()
        
        if graph_data is None:
            return None
        
        x = graph_data['x'].to(self.device)
        edge_index = graph_data['edge_index'].to(self.device)
        
        target = torch.tensor([
            [
                torch.logit(torch.tensor(optimal_params['volatility_threshold'] / 0.2)),
                torch.log(torch.tensor(optimal_params['latency_spike_ms'] / 10)),
                torch.log(torch.tensor(optimal_params['order_imbalance_ratio'])),
                torch.log(torch.tensor(optimal_params['cooling_period'] / 10))
            ]
        ], dtype=torch.float).to(self.device)
        
        self.model.train()
        
        self.optimizer.zero_grad()
        
        prediction = self.model(x, edge_index)
        
        loss = F.mse_loss(prediction, target)
        
        loss.backward()
        
        self.optimizer.step()
        
        training_record = {
            'timestamp': time.time(),
            'loss': loss.item(),
            'target_params': optimal_params,
            'market_data_summary': {
                'exchange_latency': market_data.get('exchange_latency', 0),
                'exchange_volume': market_data.get('exchange_volume', 0),
                'exchange_volatility': market_data.get('exchange_volatility', 0),
                'regime_vix': market_data.get('regime_vix', 0)
            }
        }
        self.training_history.append(training_record)
        
        if len(self.training_history) % 10 == 0:
            self._save_model()
        
        self.logger.info(f"Training loss: {loss.item():.6f}")
        
        return loss.item()
    
    def generate_mock_data(self, num_samples=100):
        """Generate mock data for testing."""
        market_data_history = []
        optimal_params_history = []
        
        for i in range(num_samples):
            volatility = np.random.uniform(0.01, 0.2)
            latency = np.random.uniform(5, 100)
            volume = np.random.uniform(1000, 10000)
            vix = np.random.uniform(10, 40)
            
            market_data = {
                'exchange_latency': latency,
                'exchange_volume': volume,
                'exchange_volatility': volatility,
                'exchange_liquidity': np.random.uniform(100000, 1000000),
                'exchange_spread': np.random.uniform(0.001, 0.01),
                'exchange_status': 'normal',
                'exchange_trade_count': np.random.uniform(100, 1000),
                'exchange_order_imbalance': np.random.uniform(0.5, 5),
                'exchange_tick_size': np.random.uniform(0.001, 0.01),
                'exchange_fee': np.random.uniform(0.0001, 0.001),
                'assets': [
                    {
                        'price': np.random.uniform(10, 1000),
                        'volume': np.random.uniform(100, 1000),
                        'volatility': volatility * np.random.uniform(0.8, 1.2),
                        'bid_ask_spread': np.random.uniform(0.001, 0.01),
                        'market_cap': np.random.uniform(1000000, 10000000000),
                        'beta': np.random.uniform(0.5, 1.5),
                        'correlation': np.random.uniform(-1, 1),
                        'momentum': np.random.uniform(-0.1, 0.1),
                        'rsi': np.random.uniform(30, 70),
                        'liquidity': np.random.uniform(10000, 1000000)
                    }
                ],
                'regime_volatility': volatility,
                'regime_trend': np.random.uniform(-0.1, 0.1),
                'regime_liquidity': np.random.uniform(100000, 1000000),
                'regime_correlation': np.random.uniform(-1, 1),
                'regime_sentiment': np.random.uniform(0, 100),
                'regime_momentum': np.random.uniform(-0.1, 0.1),
                'regime_vix': vix,
                'regime_yield_curve': np.random.uniform(-0.02, 0.03),
                'regime_macro_surprise': np.random.uniform(-0.1, 0.1),
                'regime_risk_premium': np.random.uniform(0.01, 0.05)
            }
            
            optimal_params = {
                'volatility_threshold': min(0.2, max(0.01, volatility * 1.5 + np.random.normal(0, 0.01))),
                'latency_spike_ms': min(1000, max(10, latency * 2 + np.random.normal(0, 5))),
                'order_imbalance_ratio': min(10, max(1, 2 + vix / 20 + np.random.normal(0, 0.2))),
                'cooling_period': min(1000, max(10, 30 + vix * 5 + np.random.normal(0, 10)))
            }
            
            market_data_history.append(market_data)
            optimal_params_history.append(optimal_params)
        
        return market_data_history, optimal_params_history
    
    def train_on_mock_data(self, num_samples=1000, epochs=5):
        """Train the model on mock data."""
        if not TORCH_AVAILABLE or self.model is None:
            self.logger.warning("PyTorch not available, cannot train model")
            return None
        
        self.logger.info(f"Training on {num_samples} mock samples for {epochs} epochs")
        
        market_data_history, optimal_params_history = self.generate_mock_data(num_samples)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for i in range(len(market_data_history)):
                loss = self.train(market_data_history[i], optimal_params_history[i])
                
                if loss is not None:
                    epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
        
        self._save_model()
        
        return {
            'losses': losses,
            'final_loss': losses[-1] if losses else None
        }
    
    def integrate_with_exchange_profiles(self, exchange_name, market_data):
        """Integrate with exchange profiles for optimal circuit breaker parameters."""
        try:
            from circuit_breakers.exchange_profiles import EXCHANGE_PROFILES, AdaptiveBreaker
            
            if exchange_name in EXCHANGE_PROFILES:
                profile = EXCHANGE_PROFILES[exchange_name]
            else:
                profile = EXCHANGE_PROFILES.get('default', None)
            
            predicted_params = self.predict(market_data)
            
            breaker_config = type('BreakerConfig', (), {
                'volatility_threshold': predicted_params['volatility_threshold'],
                'latency_spike_ms': predicted_params['latency_spike_ms'],
                'order_imbalance_ratio': predicted_params['order_imbalance_ratio'],
                'cooling_off_period': predicted_params['cooling_period']
            })
            
            adaptive_breaker = AdaptiveBreaker(exchange_name, breaker_config)
            
            return adaptive_breaker
            
        except ImportError:
            self.logger.warning("Exchange profiles not available")
            return None
