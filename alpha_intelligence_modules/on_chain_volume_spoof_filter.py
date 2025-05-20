"""
On-Chain Volume Spoof Filter

Flags wash-traded assets for the QMP Overrider system.
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import re

class OnChainVolumeSpoofFilter:
    """
    Flags wash-traded assets.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the On-Chain Volume Spoof Filter.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger("OnChainVolumeSpoofFilter")
        self.logger.setLevel(logging.INFO)
        
        self.spoof_thresholds = {
            "low": 0.2,  # 20% suspicious volume
            "medium": 0.4,  # 40% suspicious volume
            "high": 0.6,  # 60% suspicious volume
            "extreme": 0.8  # 80% suspicious volume
        }
        
        self.pattern_thresholds = {
            "round_number": 0.7,  # 70% round number trades
            "repeating": 0.6,  # 60% repeating patterns
            "wash_cycle": 0.5  # 50% wash cycle patterns
        }
        
        self.asset_data = {}
        
        self.spoof_signals = {}
        
        self.flagged_assets = []
        
        self.last_update = datetime.now()
        
        self.update_frequency = timedelta(hours=1)
        
        self.tracked_assets = []
        
        self.spoof_history = {}
        
    def update(self, current_time, custom_data=None):
        """
        Update the on-chain volume spoof filter with latest data.
        
        Parameters:
        - current_time: Current datetime
        - custom_data: Custom on-chain data (optional)
        
        Returns:
        - Dictionary containing spoof filter results
        """
        if current_time - self.last_update < self.update_frequency and custom_data is None:
            return {
                "asset_data": self.asset_data,
                "spoof_signals": self.spoof_signals,
                "flagged_assets": self.flagged_assets
            }
            
        if custom_data is not None:
            self._update_asset_data(custom_data)
        else:
            self._update_asset_data_internal()
        
        self._analyze_volume_patterns()
        
        self._generate_signals()
        
        self._update_flagged_assets()
        
        self.last_update = current_time
        
        return {
            "asset_data": self.asset_data,
            "spoof_signals": self.spoof_signals,
            "flagged_assets": self.flagged_assets
        }
        
    def _update_asset_data(self, custom_data):
        """
        Update asset data.
        
        Parameters:
        - custom_data: Custom on-chain data
        """
        for asset_id, data in custom_data.items():
            if asset_id not in self.asset_data:
                self.asset_data[asset_id] = {}
            
            for key, value in data.items():
                self.asset_data[asset_id][key] = value
            
            if asset_id not in self.tracked_assets:
                self.tracked_assets.append(asset_id)
        
    def _update_asset_data_internal(self):
        """
        Update asset data internally.
        """
        
        if len(self.tracked_assets) == 0:
            self.tracked_assets = [
                "BTC",
                "ETH",
                "LINK",
                "UNI",
                "AAVE",
                "SUSHI",
                "YFI",
                "SNX",
                "COMP",
                "MKR",
                "WASH1",  # Simulated wash-traded asset
                "WASH2"   # Another simulated wash-traded asset
            ]
        
        for asset_id in self.tracked_assets:
            if asset_id not in self.asset_data:
                self.asset_data[asset_id] = {}
                
            base_volume = 1000000.0
            base_tx_count = 5000
            
            if asset_id == "BTC":
                base_volume *= 5.0
                base_tx_count *= 3.0
            elif asset_id == "ETH":
                base_volume *= 3.0
                base_tx_count *= 2.5
            elif asset_id in ["WASH1", "WASH2"]:
                base_volume *= np.random.uniform(5.0, 10.0)  # Artificially high volume
                base_tx_count *= 0.5  # Fewer transactions
            
            volume_variation = np.random.normal(0.0, 0.1)  # Mean 0% variation, std 10%
            tx_variation = np.random.normal(0.0, 0.1)  # Mean 0% variation, std 10%
            
            current_volume = base_volume * (1.0 + volume_variation)
            current_tx_count = int(base_tx_count * (1.0 + tx_variation))
            
            current_volume = max(1000.0, current_volume)
            current_tx_count = max(100, current_tx_count)
            
            tx_patterns = {
                "round_number_pct": 0.1,  # 10% round number transactions
                "repeating_pct": 0.05,    # 5% repeating patterns
                "wash_cycle_pct": 0.02    # 2% wash cycle patterns
            }
            
            if asset_id in ["WASH1", "WASH2"]:
                tx_patterns["round_number_pct"] = np.random.uniform(0.6, 0.9)
                tx_patterns["repeating_pct"] = np.random.uniform(0.5, 0.8)
                tx_patterns["wash_cycle_pct"] = np.random.uniform(0.4, 0.7)
            
            volume_distribution = {
                "top10_pct": 0.3,  # Top 10 addresses hold 30% of volume
                "top50_pct": 0.5,  # Top 50 addresses hold 50% of volume
                "top100_pct": 0.7  # Top 100 addresses hold 70% of volume
            }
            
            if asset_id in ["WASH1", "WASH2"]:
                volume_distribution["top10_pct"] = np.random.uniform(0.7, 0.9)
                volume_distribution["top50_pct"] = np.random.uniform(0.8, 0.95)
                volume_distribution["top100_pct"] = np.random.uniform(0.9, 0.98)
            
            self.asset_data[asset_id] = {
                "volume_24h": current_volume,
                "tx_count_24h": current_tx_count,
                "avg_tx_size": current_volume / current_tx_count,
                "tx_patterns": tx_patterns,
                "volume_distribution": volume_distribution,
                "timestamp": datetime.now()
            }
            
            if asset_id not in self.spoof_history:
                self.spoof_history[asset_id] = []
            
            spoof_score = (
                tx_patterns["round_number_pct"] * 0.3 +
                tx_patterns["repeating_pct"] * 0.3 +
                tx_patterns["wash_cycle_pct"] * 0.4
            )
            
            self.spoof_history[asset_id].append({
                "timestamp": datetime.now(),
                "spoof_score": spoof_score,
                "volume": current_volume,
                "tx_count": current_tx_count
            })
            
            if len(self.spoof_history[asset_id]) > 100:
                self.spoof_history[asset_id] = self.spoof_history[asset_id][-100:]
        
    def _analyze_volume_patterns(self):
        """
        Analyze volume patterns.
        """
        for asset_id, data in self.asset_data.items():
            pattern_metrics = {
                "round_number_score": 0.0,
                "repeating_score": 0.0,
                "wash_cycle_score": 0.0,
                "volume_concentration_score": 0.0,
                "combined_spoof_score": 0.0
            }
            
            tx_patterns = data.get("tx_patterns", {})
            
            if "round_number_pct" in tx_patterns:
                pattern_metrics["round_number_score"] = min(1.0, tx_patterns["round_number_pct"] / self.pattern_thresholds["round_number"])
            
            if "repeating_pct" in tx_patterns:
                pattern_metrics["repeating_score"] = min(1.0, tx_patterns["repeating_pct"] / self.pattern_thresholds["repeating"])
            
            if "wash_cycle_pct" in tx_patterns:
                pattern_metrics["wash_cycle_score"] = min(1.0, tx_patterns["wash_cycle_pct"] / self.pattern_thresholds["wash_cycle"])
            
            volume_distribution = data.get("volume_distribution", {})
            if "top10_pct" in volume_distribution:
                pattern_metrics["volume_concentration_score"] = volume_distribution["top10_pct"]
            
            round_number_weight = 0.2
            repeating_weight = 0.3
            wash_cycle_weight = 0.3
            volume_concentration_weight = 0.2
            
            combined_spoof_score = (
                pattern_metrics["round_number_score"] * round_number_weight +
                pattern_metrics["repeating_score"] * repeating_weight +
                pattern_metrics["wash_cycle_score"] * wash_cycle_weight +
                pattern_metrics["volume_concentration_score"] * volume_concentration_weight
            )
            
            pattern_metrics["combined_spoof_score"] = combined_spoof_score
            
            if asset_id not in self.spoof_signals:
                self.spoof_signals[asset_id] = {}
            
            self.spoof_signals[asset_id]["metrics"] = pattern_metrics
        
    def _generate_signals(self):
        """
        Generate spoof signals.
        """
        for asset_id, data in self.spoof_signals.items():
            if "metrics" not in data:
                continue
                
            metrics = data["metrics"]
            
            combined_spoof_score = metrics["combined_spoof_score"]
            
            if combined_spoof_score >= self.spoof_thresholds["extreme"]:
                spoof_level = "extreme"
            elif combined_spoof_score >= self.spoof_thresholds["high"]:
                spoof_level = "high"
            elif combined_spoof_score >= self.spoof_thresholds["medium"]:
                spoof_level = "medium"
            elif combined_spoof_score >= self.spoof_thresholds["low"]:
                spoof_level = "low"
            else:
                spoof_level = "normal"
            
            signal_type = "NEUTRAL"
            signal_strength = 0.0
            
            if spoof_level == "extreme":
                signal_type = "STRONG_AVOID"
                signal_strength = 0.9
            elif spoof_level == "high":
                signal_type = "AVOID"
                signal_strength = 0.7
            elif spoof_level == "medium":
                signal_type = "CAUTION"
                signal_strength = 0.5
            elif spoof_level == "low":
                signal_type = "MONITOR"
                signal_strength = 0.3
            
            self.spoof_signals[asset_id]["signal"] = {
                "type": signal_type,
                "strength": signal_strength,
                "spoof_level": spoof_level,
                "spoof_score": combined_spoof_score
            }
        
    def _update_flagged_assets(self):
        """
        Update flagged assets.
        """
        self.flagged_assets = []
        
        for asset_id, data in self.spoof_signals.items():
            if "signal" not in data:
                continue
                
            signal = data["signal"]
            
            if signal["type"] in ["STRONG_AVOID", "AVOID"] and signal["strength"] >= 0.6:
                self.flagged_assets.append({
                    "asset_id": asset_id,
                    "flag_time": datetime.now(),
                    "spoof_level": signal["spoof_level"],
                    "spoof_score": signal["spoof_score"],
                    "signal_type": signal["type"],
                    "signal_strength": signal["strength"]
                })
                
                self.logger.warning(f"Asset flagged: {asset_id} with {signal['type']} signal (score: {signal['spoof_score']:.2f})")
        
    def get_asset_data(self, asset_id=None):
        """
        Get asset data.
        
        Parameters:
        - asset_id: Asset ID to get data for (optional)
        
        Returns:
        - Asset data
        """
        if asset_id is not None:
            return self.asset_data.get(asset_id, {})
        else:
            return self.asset_data
        
    def get_spoof_signals(self, asset_id=None):
        """
        Get spoof signals.
        
        Parameters:
        - asset_id: Asset ID to get signals for (optional)
        
        Returns:
        - Spoof signals
        """
        if asset_id is not None:
            return self.spoof_signals.get(asset_id, {})
        else:
            return self.spoof_signals
        
    def get_flagged_assets(self):
        """
        Get flagged assets.
        
        Returns:
        - Flagged assets
        """
        return self.flagged_assets
        
    def get_spoof_history(self, asset_id=None):
        """
        Get spoof history.
        
        Parameters:
        - asset_id: Asset ID to get history for (optional)
        
        Returns:
        - Spoof history
        """
        if asset_id is not None:
            return self.spoof_history.get(asset_id, [])
        else:
            return self.spoof_history
        
    def get_trading_signal(self, asset_id):
        """
        Get trading signal for an asset.
        
        Parameters:
        - asset_id: Asset ID to get signal for
        
        Returns:
        - Trading signal
        """
        if asset_id not in self.spoof_signals or "signal" not in self.spoof_signals[asset_id]:
            return {
                "action": "NEUTRAL",
                "confidence": 0.0
            }
        
        signal = self.spoof_signals[asset_id]["signal"]
        
        if signal["type"] == "STRONG_AVOID":
            action = "AVOID"
            confidence = signal["strength"]
        elif signal["type"] == "AVOID":
            action = "AVOID"
            confidence = signal["strength"]
        elif signal["type"] == "CAUTION":
            action = "CAUTION"
            confidence = signal["strength"]
        elif signal["type"] == "MONITOR":
            action = "MONITOR"
            confidence = signal["strength"]
        else:
            action = "NEUTRAL"
            confidence = 0.0
        
        return {
            "action": action,
            "confidence": confidence
        }
        
    def is_asset_flagged(self, asset_id):
        """
        Check if an asset is flagged.
        
        Parameters:
        - asset_id: Asset ID to check
        
        Returns:
        - Boolean indicating if asset is flagged
        """
        for asset in self.flagged_assets:
            if asset["asset_id"] == asset_id:
                return True
        
        return False
