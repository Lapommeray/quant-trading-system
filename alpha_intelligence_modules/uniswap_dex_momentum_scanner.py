"""
Uniswap/DEX Momentum Scanner

Front-runs low cap token surges for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import re

class UniswapDEXMomentumScanner:
    """
    Front-runs low cap token surges.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the Uniswap/DEX Momentum Scanner.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("UniswapDEXMomentumScanner")
        self.logger.setLevel(logging.INFO)
        
        self.dex_platforms = ["uniswap", "sushiswap", "pancakeswap", "curve", "balancer"]
        
        self.momentum_thresholds = {
            "low": 0.05,  # 5% increase
            "medium": 0.1,  # 10% increase
            "high": 0.2,  # 20% increase
            "extreme": 0.5  # 50% increase
        }
        
        self.volume_thresholds = {
            "low": 1.5,  # 1.5x average
            "medium": 3.0,  # 3x average
            "high": 5.0,  # 5x average
            "extreme": 10.0  # 10x average
        }
        
        self.liquidity_thresholds = {
            "micro": 100000,  # $100K
            "small": 1000000,  # $1M
            "medium": 10000000,  # $10M
            "large": 100000000  # $100M
        }
        
        self.token_data = {}
        
        self.momentum_signals = {}
        
        self.active_opportunities = []
        
        self.last_update = datetime.now()
        
        self.update_frequency = timedelta(minutes=5)  # More frequent updates for DEX data
        
        self.tracked_tokens = []
        
        self.momentum_history = {}
        
    def update(self, current_time, custom_data=None):
        """
        Update the Uniswap/DEX momentum scanner with latest data.
        
        Parameters:
        - current_time: Current datetime
        - custom_data: Custom DEX data (optional)
        
        Returns:
        - Dictionary containing momentum results
        """
        if current_time - self.last_update < self.update_frequency and custom_data is None:
            return {
                "token_data": self.token_data,
                "momentum_signals": self.momentum_signals,
                "active_opportunities": self.active_opportunities
            }
            
        if custom_data is not None:
            self._update_token_data(custom_data)
        else:
            self._update_token_data_internal()
        
        self._analyze_momentum()
        
        self._generate_signals()
        
        self._update_active_opportunities()
        
        self.last_update = current_time
        
        return {
            "token_data": self.token_data,
            "momentum_signals": self.momentum_signals,
            "active_opportunities": self.active_opportunities
        }
        
    def _update_token_data(self, custom_data):
        """
        Update token data.
        
        Parameters:
        - custom_data: Custom DEX data
        """
        for token_id, data in custom_data.items():
            if token_id not in self.token_data:
                self.token_data[token_id] = {}
            
            for key, value in data.items():
                self.token_data[token_id][key] = value
            
            if token_id not in self.tracked_tokens:
                self.tracked_tokens.append(token_id)
        
    def _update_token_data_internal(self):
        """
        Update token data internally.
        """
        
        if len(self.tracked_tokens) == 0:
            self.tracked_tokens = [
                "ETH-USDC",
                "WBTC-USDC",
                "LINK-ETH",
                "UNI-ETH",
                "AAVE-ETH",
                "SUSHI-ETH",
                "YFI-ETH",
                "SNX-ETH",
                "COMP-ETH",
                "MKR-ETH",
                "NEW-ETH",  # New token with potential surge
                "MICRO-ETH"  # Micro cap token
            ]
        
        for token_id in self.tracked_tokens:
            if token_id not in self.token_data:
                self.token_data[token_id] = {}
                
            base_price = 100.0
            base_volume = 1000000.0
            base_liquidity = 5000000.0
            
            if "ETH" in token_id:
                base_price *= 1.5
                base_volume *= 2.0
                base_liquidity *= 3.0
            elif "BTC" in token_id:
                base_price *= 3.0
                base_volume *= 2.5
                base_liquidity *= 4.0
            elif "MICRO" in token_id:
                base_price *= 0.01
                base_volume *= 0.1
                base_liquidity *= 0.05
            elif "NEW" in token_id:
                base_price *= 0.1
                base_volume *= 0.5
                base_liquidity *= 0.2
                
                if np.random.random() < 0.3:  # 30% chance of surge
                    base_price *= np.random.uniform(1.2, 2.0)
                    base_volume *= np.random.uniform(3.0, 10.0)
            
            price_change = np.random.normal(0.01, 0.05)  # Mean 1% change, std 5%
            volume_change = np.random.normal(0.0, 0.2)  # Mean 0% change, std 20%
            liquidity_change = np.random.normal(0.0, 0.05)  # Mean 0% change, std 5%
            
            current_price = base_price * (1.0 + price_change)
            current_volume = base_volume * (1.0 + volume_change)
            current_liquidity = base_liquidity * (1.0 + liquidity_change)
            
            current_price = max(0.01, current_price)
            current_volume = max(100.0, current_volume)
            current_liquidity = max(1000.0, current_liquidity)
            
            self.token_data[token_id] = {
                "price": current_price,
                "price_change_1h": price_change,
                "price_change_24h": price_change * 3.0,  # Simulate longer-term trend
                "volume_24h": current_volume,
                "volume_change_24h": volume_change,
                "liquidity": current_liquidity,
                "liquidity_change_24h": liquidity_change,
                "platform": np.random.choice(self.dex_platforms),
                "market_cap": current_price * (current_liquidity / current_price * 0.5),  # Rough estimate
                "timestamp": datetime.now()
            }
            
            if token_id not in self.momentum_history:
                self.momentum_history[token_id] = []
            
            self.momentum_history[token_id].append({
                "timestamp": datetime.now(),
                "price": current_price,
                "volume": current_volume,
                "price_change": price_change,
                "volume_change": volume_change
            })
            
            if len(self.momentum_history[token_id]) > 100:
                self.momentum_history[token_id] = self.momentum_history[token_id][-100:]
        
    def _analyze_momentum(self):
        """
        Analyze momentum data.
        """
        for token_id, data in self.token_data.items():
            momentum_metrics = {
                "price_momentum_1h": data.get("price_change_1h", 0.0),
                "price_momentum_24h": data.get("price_change_24h", 0.0),
                "volume_momentum": data.get("volume_change_24h", 0.0),
                "liquidity_momentum": data.get("liquidity_change_24h", 0.0),
                "combined_momentum": 0.0,
                "momentum_score": 0.0
            }
            
            price_weight = 0.5
            volume_weight = 0.3
            liquidity_weight = 0.2
            
            combined_momentum = (
                momentum_metrics["price_momentum_1h"] * price_weight +
                momentum_metrics["volume_momentum"] * volume_weight +
                momentum_metrics["liquidity_momentum"] * liquidity_weight
            )
            
            momentum_metrics["combined_momentum"] = combined_momentum
            
            if combined_momentum >= 0:
                momentum_score = 50 + min(50, combined_momentum * 100)
            else:
                momentum_score = 50 + max(-50, combined_momentum * 100)
            
            momentum_metrics["momentum_score"] = momentum_score
            
            if token_id not in self.momentum_signals:
                self.momentum_signals[token_id] = {}
            
            self.momentum_signals[token_id]["metrics"] = momentum_metrics
        
    def _generate_signals(self):
        """
        Generate momentum signals.
        """
        for token_id, data in self.momentum_signals.items():
            if "metrics" not in data:
                continue
                
            metrics = data["metrics"]
            token_data = self.token_data.get(token_id, {})
            
            price_momentum = metrics["price_momentum_1h"]
            volume_momentum = metrics["volume_momentum"]
            combined_momentum = metrics["combined_momentum"]
            momentum_score = metrics["momentum_score"]
            
            market_cap = token_data.get("market_cap", 0.0)
            liquidity = token_data.get("liquidity", 0.0)
            
            if market_cap < self.liquidity_thresholds["micro"]:
                market_cap_category = "micro"
            elif market_cap < self.liquidity_thresholds["small"]:
                market_cap_category = "small"
            elif market_cap < self.liquidity_thresholds["medium"]:
                market_cap_category = "medium"
            else:
                market_cap_category = "large"
            
            if price_momentum >= self.momentum_thresholds["extreme"]:
                momentum_level = "extreme"
            elif price_momentum >= self.momentum_thresholds["high"]:
                momentum_level = "high"
            elif price_momentum >= self.momentum_thresholds["medium"]:
                momentum_level = "medium"
            elif price_momentum >= self.momentum_thresholds["low"]:
                momentum_level = "low"
            else:
                momentum_level = "neutral"
            
            if volume_momentum >= self.volume_thresholds["extreme"]:
                volume_level = "extreme"
            elif volume_momentum >= self.volume_thresholds["high"]:
                volume_level = "high"
            elif volume_momentum >= self.volume_thresholds["medium"]:
                volume_level = "medium"
            elif volume_momentum >= self.volume_thresholds["low"]:
                volume_level = "low"
            else:
                volume_level = "neutral"
            
            signal_type = "NEUTRAL"
            signal_strength = 0.0
            
            if momentum_level == "extreme" and volume_level in ["high", "extreme"]:
                signal_type = "STRONG_BUY"
                signal_strength = 0.9
            elif momentum_level == "high" and volume_level in ["medium", "high", "extreme"]:
                signal_type = "BUY"
                signal_strength = 0.7
            elif momentum_level == "medium" and volume_level in ["medium", "high"]:
                signal_type = "WEAK_BUY"
                signal_strength = 0.5
            elif momentum_level == "low" and volume_level in ["medium", "high"]:
                signal_type = "WEAK_BUY"
                signal_strength = 0.3
            
            if market_cap_category == "micro":
                signal_strength *= 0.8
            elif market_cap_category == "small":
                signal_strength *= 0.9
            
            if liquidity < self.liquidity_thresholds["micro"]:
                signal_strength *= 0.7
            elif liquidity < self.liquidity_thresholds["small"]:
                signal_strength *= 0.8
            
            self.momentum_signals[token_id]["signal"] = {
                "type": signal_type,
                "strength": signal_strength,
                "momentum_level": momentum_level,
                "volume_level": volume_level,
                "market_cap_category": market_cap_category,
                "momentum_score": momentum_score
            }
        
    def _update_active_opportunities(self):
        """
        Update active opportunities.
        """
        for token_id, data in self.momentum_signals.items():
            if "signal" not in data:
                continue
                
            signal = data["signal"]
            
            if signal["type"] in ["STRONG_BUY", "BUY"] and signal["strength"] >= 0.6:
                existing = False
                for opp in self.active_opportunities:
                    if opp["token_id"] == token_id:
                        existing = True
                        break
                
                if not existing:
                    self.active_opportunities.append({
                        "token_id": token_id,
                        "entry_time": datetime.now(),
                        "entry_price": self.token_data[token_id]["price"],
                        "signal_type": signal["type"],
                        "signal_strength": signal["strength"],
                        "momentum_score": signal["momentum_score"],
                        "status": "ACTIVE"
                    })
                    
                    self.logger.info(f"New opportunity: {token_id} with {signal['type']} signal (strength: {signal['strength']:.2f})")
        
        for opp in self.active_opportunities:
            if opp["status"] != "ACTIVE":
                continue
                
            token_id = opp["token_id"]
            
            if token_id not in self.token_data:
                continue
                
            current_price = self.token_data[token_id]["price"]
            entry_price = opp["entry_price"]
            
            pnl_pct = (current_price - entry_price) / entry_price
            
            if pnl_pct >= 0.5:  # 50% profit target
                opp["exit_time"] = datetime.now()
                opp["exit_price"] = current_price
                opp["pnl_pct"] = pnl_pct
                opp["exit_reason"] = "PROFIT_TARGET"
                opp["status"] = "CLOSED"
                
                self.logger.info(f"Opportunity closed: {token_id} with {pnl_pct:.2%} profit (reason: PROFIT_TARGET)")
            elif pnl_pct <= -0.2:  # 20% stop loss
                opp["exit_time"] = datetime.now()
                opp["exit_price"] = current_price
                opp["pnl_pct"] = pnl_pct
                opp["exit_reason"] = "STOP_LOSS"
                opp["status"] = "CLOSED"
                
                self.logger.info(f"Opportunity closed: {token_id} with {pnl_pct:.2%} loss (reason: STOP_LOSS)")
            elif (datetime.now() - opp["entry_time"]) > timedelta(days=3):  # 3-day time stop
                opp["exit_time"] = datetime.now()
                opp["exit_price"] = current_price
                opp["pnl_pct"] = pnl_pct
                opp["exit_reason"] = "TIME_STOP"
                opp["status"] = "CLOSED"
                
                self.logger.info(f"Opportunity closed: {token_id} with {pnl_pct:.2%} {('profit' if pnl_pct >= 0 else 'loss')} (reason: TIME_STOP)")
            elif token_id in self.momentum_signals and "signal" in self.momentum_signals[token_id]:
                current_signal = self.momentum_signals[token_id]["signal"]
                
                if current_signal["type"] in ["NEUTRAL", "WEAK_SELL", "SELL", "STRONG_SELL"]:
                    opp["exit_time"] = datetime.now()
                    opp["exit_price"] = current_price
                    opp["pnl_pct"] = pnl_pct
                    opp["exit_reason"] = "SIGNAL_REVERSAL"
                    opp["status"] = "CLOSED"
                    
                    self.logger.info(f"Opportunity closed: {token_id} with {pnl_pct:.2%} {('profit' if pnl_pct >= 0 else 'loss')} (reason: SIGNAL_REVERSAL)")
        
    def get_token_data(self, token_id=None):
        """
        Get token data.
        
        Parameters:
        - token_id: Token ID to get data for (optional)
        
        Returns:
        - Token data
        """
        if token_id is not None:
            return self.token_data.get(token_id, {})
        else:
            return self.token_data
        
    def get_momentum_signals(self, token_id=None):
        """
        Get momentum signals.
        
        Parameters:
        - token_id: Token ID to get signals for (optional)
        
        Returns:
        - Momentum signals
        """
        if token_id is not None:
            return self.momentum_signals.get(token_id, {})
        else:
            return self.momentum_signals
        
    def get_active_opportunities(self, status=None):
        """
        Get active opportunities.
        
        Parameters:
        - status: Filter by status (optional)
        
        Returns:
        - Active opportunities
        """
        if status is not None:
            return [opp for opp in self.active_opportunities if opp["status"] == status]
        else:
            return self.active_opportunities
        
    def get_momentum_history(self, token_id=None):
        """
        Get momentum history.
        
        Parameters:
        - token_id: Token ID to get history for (optional)
        
        Returns:
        - Momentum history
        """
        if token_id is not None:
            return self.momentum_history.get(token_id, [])
        else:
            return self.momentum_history
        
    def get_trading_signal(self, token_id):
        """
        Get trading signal for a token.
        
        Parameters:
        - token_id: Token ID to get signal for
        
        Returns:
        - Trading signal
        """
        if token_id not in self.momentum_signals or "signal" not in self.momentum_signals[token_id]:
            return {
                "action": "NEUTRAL",
                "confidence": 0.0
            }
        
        signal = self.momentum_signals[token_id]["signal"]
        
        if signal["type"] == "STRONG_BUY":
            action = "BUY"
            confidence = signal["strength"]
        elif signal["type"] == "BUY":
            action = "BUY"
            confidence = signal["strength"]
        elif signal["type"] == "WEAK_BUY":
            action = "BUY"
            confidence = signal["strength"]
        else:
            action = "NEUTRAL"
            confidence = 0.0
        
        return {
            "action": action,
            "confidence": confidence
        }
