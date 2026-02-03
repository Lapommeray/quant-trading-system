"""
Market Microstructure Detectors - Real-Time MM Behavior Analysis

This module implements detectors for market-maker behaviors that are observable
through order book dynamics. These detectors feed directly into the Bayesian
commitment accounting system to provide causal signals about market intent.

Key Detectors:
- SpoofDetector: Identifies fake liquidity (high add/cancel ratio, low hit rate)
- AbsorptionDetector: Detects hidden buying/selling (volume up, price stable)
- InventoryFlipDetector: Tracks MM inventory direction changes
- DepthPressureAnalyzer: Computes bid/ask pressure from order book

Mathematical Foundation:
- Spoof detection: add/cancel ratio > threshold AND low trade hit rate
- Absorption: volume spike > baseline AND |price_delta| < threshold
- Inventory flip: depth ratio crosses from >2:1 to <1:2 (or vice versa)
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("MicrostructureDetector")


@dataclass
class OrderBookSnapshot:
    """Represents a single order book snapshot"""
    timestamp: float
    bid_prices: List[float]
    bid_volumes: List[float]
    ask_prices: List[float]
    ask_volumes: List[float]
    mid_price: float = 0.0
    
    def __post_init__(self):
        if self.bid_prices and self.ask_prices:
            self.mid_price = (self.bid_prices[0] + self.ask_prices[0]) / 2
            
    @property
    def spread(self) -> float:
        if self.bid_prices and self.ask_prices:
            return self.ask_prices[0] - self.bid_prices[0]
        return 0.0
        
    @property
    def total_bid_volume(self) -> float:
        return sum(self.bid_volumes) if self.bid_volumes else 0.0
        
    @property
    def total_ask_volume(self) -> float:
        return sum(self.ask_volumes) if self.ask_volumes else 0.0


@dataclass
class MicrostructureFlags:
    """Container for all microstructure detection flags"""
    spoof_detected: bool = False
    absorption_detected: bool = False
    inventory_flip_detected: bool = False
    spoof_confidence: float = 0.0
    absorption_confidence: float = 0.0
    flip_direction: str = "NONE"
    bid_pressure: float = 0.0
    ask_pressure: float = 0.0
    depth_imbalance: float = 0.0
    implied_inventory_direction: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spoof_detected": self.spoof_detected,
            "absorption_detected": self.absorption_detected,
            "inventory_flip_detected": self.inventory_flip_detected,
            "spoof_confidence": self.spoof_confidence,
            "absorption_confidence": self.absorption_confidence,
            "flip_direction": self.flip_direction,
            "bid_pressure": self.bid_pressure,
            "ask_pressure": self.ask_pressure,
            "depth_imbalance": self.depth_imbalance,
            "implied_inventory_direction": self.implied_inventory_direction
        }
        
    def get_bayesian_penalty(self) -> float:
        """
        Compute Bayesian update penalty based on detected behaviors.
        
        Returns:
            Penalty value to add to beta (rejection evidence)
            Negative values indicate acceptance evidence
        """
        penalty = 0.0
        
        if self.spoof_detected:
            penalty += 0.5 * self.spoof_confidence
            
        if self.absorption_detected:
            penalty -= 0.3 * self.absorption_confidence
            
        if self.inventory_flip_detected:
            penalty += 0.2
            
        return penalty


class SpoofDetector:
    """
    Detects spoofing behavior in order book dynamics.
    
    Spoofing is characterized by:
    - High add/cancel ratio (orders placed and quickly cancelled)
    - Low trade hit rate (orders rarely executed)
    - Depth imbalance that doesn't persist
    
    The detector maintains a rolling window of order book changes
    and computes statistics to identify suspicious patterns.
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 spoof_threshold: float = 5.0,
                 imbalance_std_threshold: float = 0.1,
                 min_samples: int = 10):
        """
        Initialize spoof detector.
        
        Args:
            window_size: Rolling window for statistics
            spoof_threshold: Add/cancel ratio threshold for spoof flag
            imbalance_std_threshold: Max imbalance std for spoof (low = suspicious)
            min_samples: Minimum samples before detection
        """
        self.window_size = window_size
        self.spoof_threshold = spoof_threshold
        self.imbalance_std_threshold = imbalance_std_threshold
        self.min_samples = min_samples
        
        self.order_history: deque = deque(maxlen=window_size)
        self.add_cancel_ratios: deque = deque(maxlen=window_size)
        self.imbalances: deque = deque(maxlen=window_size)
        self.trade_hit_rates: deque = deque(maxlen=window_size)
        
        self._last_detection = "NORMAL"
        self._confidence = 0.0
        
    def update(self, 
               bid_depth: float,
               ask_depth: float,
               adds: int,
               cancels: int,
               trades: int = 0,
               price_move: float = 0.0) -> Tuple[str, float]:
        """
        Update detector with new order book data.
        
        Args:
            bid_depth: Total bid volume (top N levels)
            ask_depth: Total ask volume (top N levels)
            adds: Number of order additions
            cancels: Number of order cancellations
            trades: Number of trades executed
            price_move: Price change since last update
            
        Returns:
            Tuple of (detection_result, confidence)
        """
        imbalance = bid_depth / max(ask_depth, 1e-6)
        add_cancel_ratio = cancels / max(adds, 1e-6)
        hit_rate = trades / max(adds, 1e-6)
        
        self.imbalances.append(imbalance)
        self.add_cancel_ratios.append(add_cancel_ratio)
        self.trade_hit_rates.append(hit_rate)
        
        self.order_history.append({
            "imbalance": imbalance,
            "add_cancel": add_cancel_ratio,
            "hit_rate": hit_rate,
            "price_move": price_move
        })
        
        return self.detect()
        
    def detect(self) -> Tuple[str, float]:
        """
        Run spoof detection on current window.
        
        Returns:
            Tuple of (detection_result, confidence)
        """
        if len(self.order_history) < self.min_samples:
            return "NORMAL", 0.0
            
        avg_add_cancel = np.mean(list(self.add_cancel_ratios))
        std_imbalance = np.std(list(self.imbalances))
        avg_hit_rate = np.mean(list(self.trade_hit_rates))
        
        spoof_score = 0.0
        
        if avg_add_cancel > self.spoof_threshold:
            spoof_score += 0.4
            
        if std_imbalance < self.imbalance_std_threshold:
            spoof_score += 0.3
            
        if avg_hit_rate < 0.1:
            spoof_score += 0.3
            
        self._confidence = min(spoof_score, 1.0)
        
        if spoof_score >= 0.6:
            self._last_detection = "SPOOF_DETECTED"
        else:
            self._last_detection = "NORMAL"
            
        return self._last_detection, self._confidence
        
    def reset(self):
        """Reset detector state"""
        self.order_history.clear()
        self.add_cancel_ratios.clear()
        self.imbalances.clear()
        self.trade_hit_rates.clear()
        self._last_detection = "NORMAL"
        self._confidence = 0.0


class AbsorptionDetector:
    """
    Detects absorption behavior (hidden buying/selling).
    
    Absorption is characterized by:
    - Volume significantly above baseline
    - Price movement near zero (absorbed by hidden orders)
    - Depth remains relatively stable
    
    This indicates a large player is accumulating/distributing
    without moving the price.
    """
    
    def __init__(self,
                 window_size: int = 50,
                 volume_spike_threshold: float = 1.5,
                 price_move_threshold: float = 0.001,
                 min_samples: int = 10):
        """
        Initialize absorption detector.
        
        Args:
            window_size: Rolling window for baseline
            volume_spike_threshold: Multiple of avg volume for spike
            price_move_threshold: Max price move for absorption
            min_samples: Minimum samples before detection
        """
        self.window_size = window_size
        self.volume_spike_threshold = volume_spike_threshold
        self.price_move_threshold = price_move_threshold
        self.min_samples = min_samples
        
        self.volume_history: deque = deque(maxlen=window_size)
        self.price_history: deque = deque(maxlen=window_size)
        self.depth_history: deque = deque(maxlen=window_size)
        
        self._last_detection = "NORMAL"
        self._confidence = 0.0
        
    def update(self, 
               volume: float,
               price_delta: float,
               depth_change: float = 0.0) -> Tuple[str, float]:
        """
        Update detector with new market data.
        
        Args:
            volume: Trade volume
            price_delta: Price change
            depth_change: Change in total depth
            
        Returns:
            Tuple of (detection_result, confidence)
        """
        self.volume_history.append(volume)
        self.price_history.append(price_delta)
        self.depth_history.append(depth_change)
        
        return self.detect()
        
    def detect(self) -> Tuple[str, float]:
        """
        Run absorption detection on current window.
        
        Returns:
            Tuple of (detection_result, confidence)
        """
        if len(self.volume_history) < self.min_samples:
            return "NORMAL", 0.0
            
        volumes = list(self.volume_history)
        prices = list(self.price_history)
        
        avg_volume = np.mean(volumes)
        recent_volume = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
        recent_price_move = np.abs(np.mean(prices[-5:])) if len(prices) >= 5 else abs(prices[-1])
        
        absorption_score = 0.0
        
        if recent_volume > self.volume_spike_threshold * avg_volume:
            absorption_score += 0.5
            
        if recent_price_move < self.price_move_threshold:
            absorption_score += 0.5
            
        self._confidence = min(absorption_score, 1.0)
        
        if absorption_score >= 0.7:
            self._last_detection = "ABSORPTION"
        else:
            self._last_detection = "NORMAL"
            
        return self._last_detection, self._confidence
        
    def reset(self):
        """Reset detector state"""
        self.volume_history.clear()
        self.price_history.clear()
        self.depth_history.clear()
        self._last_detection = "NORMAL"
        self._confidence = 0.0


class InventoryFlipDetector:
    """
    Detects market-maker inventory direction changes.
    
    Inventory flip is characterized by:
    - Depth imbalance shifting from one side to the other
    - Ratio crossing from >2:1 to <1:2 (or vice versa)
    
    This indicates the MM is changing their position direction.
    """
    
    def __init__(self,
                 window_size: int = 50,
                 flip_ratio_threshold: float = 2.0,
                 min_samples: int = 20):
        """
        Initialize inventory flip detector.
        
        Args:
            window_size: Rolling window for tracking
            flip_ratio_threshold: Ratio threshold for flip detection
            min_samples: Minimum samples before detection
        """
        self.window_size = window_size
        self.flip_ratio_threshold = flip_ratio_threshold
        self.min_samples = min_samples
        
        self.depth_ratio_history: deque = deque(maxlen=window_size)
        
        self._last_detection = "NORMAL"
        self._flip_direction = "NONE"
        
    def update(self, bid_depth: float, ask_depth: float) -> Tuple[str, str]:
        """
        Update detector with new depth data.
        
        Args:
            bid_depth: Total bid volume
            ask_depth: Total ask volume
            
        Returns:
            Tuple of (detection_result, flip_direction)
        """
        ratio = bid_depth / max(ask_depth, 1e-6)
        self.depth_ratio_history.append(ratio)
        
        return self.detect()
        
    def detect(self) -> Tuple[str, str]:
        """
        Run inventory flip detection on current window.
        
        Returns:
            Tuple of (detection_result, flip_direction)
        """
        if len(self.depth_ratio_history) < self.min_samples:
            return "NORMAL", "NONE"
            
        ratios = list(self.depth_ratio_history)
        recent = ratios[-10:]
        
        max_recent = max(recent)
        min_recent = min(recent)
        
        if max_recent > self.flip_ratio_threshold and min_recent < 1/self.flip_ratio_threshold:
            self._last_detection = "INVENTORY_FLIP"
            
            if ratios[-1] > 1:
                self._flip_direction = "BID_HEAVY"
            else:
                self._flip_direction = "ASK_HEAVY"
        else:
            self._last_detection = "NORMAL"
            self._flip_direction = "NONE"
            
        return self._last_detection, self._flip_direction
        
    def reset(self):
        """Reset detector state"""
        self.depth_ratio_history.clear()
        self._last_detection = "NORMAL"
        self._flip_direction = "NONE"


class DepthPressureAnalyzer:
    """
    Analyzes order book depth to compute bid/ask pressure.
    
    Pressure is computed as weighted sum of volume × distance from mid.
    Higher pressure indicates stronger support/resistance.
    """
    
    def __init__(self, levels: int = 10, decay_factor: float = 0.9):
        """
        Initialize depth pressure analyzer.
        
        Args:
            levels: Number of price levels to analyze
            decay_factor: Exponential decay for distant levels
        """
        self.levels = levels
        self.decay_factor = decay_factor
        
        self.bid_pressure = 0.0
        self.ask_pressure = 0.0
        self.depth_imbalance = 0.0
        self.implied_inventory_direction = 0.0
        
    def analyze(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """
        Analyze order book snapshot for pressure metrics.
        
        Args:
            snapshot: OrderBookSnapshot with bid/ask data
            
        Returns:
            Dict with pressure metrics
        """
        mid = snapshot.mid_price
        
        self.bid_pressure = 0.0
        for i, (price, vol) in enumerate(zip(snapshot.bid_prices[:self.levels], 
                                              snapshot.bid_volumes[:self.levels])):
            distance = mid - price
            weight = self.decay_factor ** i
            self.bid_pressure += vol * distance * weight
            
        self.ask_pressure = 0.0
        for i, (price, vol) in enumerate(zip(snapshot.ask_prices[:self.levels],
                                              snapshot.ask_volumes[:self.levels])):
            distance = price - mid
            weight = self.decay_factor ** i
            self.ask_pressure += vol * distance * weight
            
        total_pressure = self.bid_pressure + self.ask_pressure
        if total_pressure > 0:
            self.depth_imbalance = (self.bid_pressure - self.ask_pressure) / total_pressure
        else:
            self.depth_imbalance = 0.0
            
        self.implied_inventory_direction = np.tanh(self.depth_imbalance * 2)
        
        return {
            "bid_pressure": self.bid_pressure,
            "ask_pressure": self.ask_pressure,
            "depth_imbalance": self.depth_imbalance,
            "implied_inventory_direction": self.implied_inventory_direction
        }


class MicrostructureDetectors:
    """
    Unified interface for all microstructure detectors.
    
    Combines spoof, absorption, and inventory flip detection
    into a single update/detect cycle that produces MicrostructureFlags.
    """
    
    def __init__(self,
                 spoof_window: int = 100,
                 absorption_window: int = 50,
                 flip_window: int = 50,
                 depth_levels: int = 10):
        """
        Initialize all detectors.
        
        Args:
            spoof_window: Window size for spoof detection
            absorption_window: Window size for absorption detection
            flip_window: Window size for inventory flip detection
            depth_levels: Number of order book levels to analyze
        """
        self.spoof_detector = SpoofDetector(window_size=spoof_window)
        self.absorption_detector = AbsorptionDetector(window_size=absorption_window)
        self.flip_detector = InventoryFlipDetector(window_size=flip_window)
        self.depth_analyzer = DepthPressureAnalyzer(levels=depth_levels)
        
        self._last_flags = MicrostructureFlags()
        
        logger.info("MicrostructureDetectors initialized")
        
    def update(self, event_data: Dict[str, Any]) -> MicrostructureFlags:
        """
        Update all detectors with new market data.
        
        Args:
            event_data: Dict containing:
                - bid_depth: Total bid volume
                - ask_depth: Total ask volume
                - adds: Order additions
                - cancels: Order cancellations
                - trades: Executed trades
                - volume: Trade volume
                - price_delta: Price change
                - snapshot: Optional OrderBookSnapshot
                
        Returns:
            MicrostructureFlags with all detection results
        """
        bid_depth = event_data.get("bid_depth", 1000.0)
        ask_depth = event_data.get("ask_depth", 1000.0)
        adds = event_data.get("adds", 10)
        cancels = event_data.get("cancels", 5)
        trades = event_data.get("trades", 1)
        volume = event_data.get("volume", 100.0)
        price_delta = event_data.get("price_delta", 0.0)
        
        spoof_result, spoof_conf = self.spoof_detector.update(
            bid_depth, ask_depth, adds, cancels, trades, price_delta
        )
        
        absorb_result, absorb_conf = self.absorption_detector.update(
            volume, price_delta
        )
        
        flip_result, flip_dir = self.flip_detector.update(bid_depth, ask_depth)
        
        pressure_metrics = {"bid_pressure": 0.0, "ask_pressure": 0.0, 
                          "depth_imbalance": 0.0, "implied_inventory_direction": 0.0}
        if "snapshot" in event_data:
            pressure_metrics = self.depth_analyzer.analyze(event_data["snapshot"])
        else:
            total = bid_depth + ask_depth
            if total > 0:
                pressure_metrics["depth_imbalance"] = (bid_depth - ask_depth) / total
                pressure_metrics["implied_inventory_direction"] = np.tanh(
                    pressure_metrics["depth_imbalance"] * 2
                )
        
        self._last_flags = MicrostructureFlags(
            spoof_detected=(spoof_result == "SPOOF_DETECTED"),
            absorption_detected=(absorb_result == "ABSORPTION"),
            inventory_flip_detected=(flip_result == "INVENTORY_FLIP"),
            spoof_confidence=spoof_conf,
            absorption_confidence=absorb_conf,
            flip_direction=flip_dir,
            bid_pressure=pressure_metrics["bid_pressure"],
            ask_pressure=pressure_metrics["ask_pressure"],
            depth_imbalance=pressure_metrics["depth_imbalance"],
            implied_inventory_direction=pressure_metrics["implied_inventory_direction"]
        )
        
        if self._last_flags.spoof_detected or self._last_flags.absorption_detected:
            logger.debug(
                f"MM flags: spoof={self._last_flags.spoof_detected} "
                f"absorb={self._last_flags.absorption_detected} "
                f"flip={self._last_flags.inventory_flip_detected}"
            )
        
        return self._last_flags
        
    def get_last_flags(self) -> MicrostructureFlags:
        """Return most recent detection flags"""
        return self._last_flags
        
    def reset(self):
        """Reset all detectors"""
        self.spoof_detector.reset()
        self.absorption_detector.reset()
        self.flip_detector.reset()
        self._last_flags = MicrostructureFlags()


def demo():
    """Demonstrate microstructure detection"""
    print("=" * 60)
    print("MICROSTRUCTURE DETECTOR DEMO")
    print("=" * 60)
    
    detectors = MicrostructureDetectors()
    
    print("\n--- Normal Market Activity ---")
    for i in range(15):
        flags = detectors.update({
            "bid_depth": 1000 + np.random.randn() * 50,
            "ask_depth": 1000 + np.random.randn() * 50,
            "adds": 10,
            "cancels": 3,
            "trades": 5,
            "volume": 100,
            "price_delta": np.random.randn() * 0.001
        })
    print(f"Flags: {flags.to_dict()}")
    
    print("\n--- Simulating Spoof Activity ---")
    for i in range(20):
        flags = detectors.update({
            "bid_depth": 2000 + np.random.randn() * 10,
            "ask_depth": 500 + np.random.randn() * 10,
            "adds": 100,
            "cancels": 90,
            "trades": 2,
            "volume": 50,
            "price_delta": np.random.randn() * 0.0001
        })
    print(f"Spoof detected: {flags.spoof_detected}, confidence: {flags.spoof_confidence:.2f}")
    print(f"Bayesian penalty: {flags.get_bayesian_penalty():.3f}")
    
    detectors.reset()
    
    print("\n--- Simulating Absorption ---")
    for i in range(15):
        flags = detectors.update({
            "bid_depth": 1000,
            "ask_depth": 1000,
            "adds": 10,
            "cancels": 3,
            "trades": 5,
            "volume": 100 if i < 10 else 500,
            "price_delta": 0.0001 if i < 10 else 0.00001
        })
    print(f"Absorption detected: {flags.absorption_detected}, confidence: {flags.absorption_confidence:.2f}")
    print(f"Bayesian penalty: {flags.get_bayesian_penalty():.3f}")
    
    detectors.reset()
    
    print("\n--- Simulating Inventory Flip ---")
    for i in range(30):
        if i < 15:
            bid_depth = 2000
            ask_depth = 500
        else:
            bid_depth = 500
            ask_depth = 2000
        flags = detectors.update({
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "adds": 10,
            "cancels": 3,
            "trades": 5,
            "volume": 100,
            "price_delta": 0.001
        })
    print(f"Inventory flip detected: {flags.inventory_flip_detected}, direction: {flags.flip_direction}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo()
