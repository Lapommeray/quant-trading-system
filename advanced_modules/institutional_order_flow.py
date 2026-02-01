"""
Institutional Order Flow Analysis Module

High-signal feature generation from order book and tick data:
- Order flow imbalance sequence detection
- Absorption/exhaustion cluster identification
- Volume delta analysis
- Institutional footprint detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging

logger = logging.getLogger("InstitutionalOrderFlow")


class FlowDirection(Enum):
    AGGRESSIVE_BUY = "aggressive_buy"
    AGGRESSIVE_SELL = "aggressive_sell"
    PASSIVE_BUY = "passive_buy"
    PASSIVE_SELL = "passive_sell"
    NEUTRAL = "neutral"


@dataclass
class OrderFlowTick:
    """Single tick with order flow classification"""
    timestamp: float
    price: float
    volume: float
    bid: float
    ask: float
    direction: FlowDirection
    delta: float
    
    
@dataclass
class AbsorptionCluster:
    """Detected absorption cluster"""
    start_time: float
    end_time: float
    price_level: float
    total_volume: float
    absorbed_volume: float
    direction: str
    strength: float
    is_exhaustion: bool


@dataclass
class ImbalanceSignal:
    """Order flow imbalance signal"""
    timestamp: float
    imbalance_ratio: float
    cumulative_delta: float
    direction: str
    confidence: float
    sequence_length: int


class OrderFlowImbalanceDetector:
    """
    Detects order flow imbalances from tick-level data.
    
    Uses cumulative volume delta and bid/ask imbalance to identify
    institutional activity and potential price direction.
    """
    
    def __init__(self, 
                 lookback_ticks: int = 100,
                 imbalance_threshold: float = 0.6,
                 sequence_min_length: int = 5):
        self.lookback_ticks = lookback_ticks
        self.imbalance_threshold = imbalance_threshold
        self.sequence_min_length = sequence_min_length
        
        self.tick_buffer: deque = deque(maxlen=lookback_ticks)
        self.cumulative_delta = 0.0
        self.imbalance_history: deque = deque(maxlen=1000)
        
    def classify_tick(self, price: float, volume: float, 
                     bid: float, ask: float, prev_price: float) -> OrderFlowTick:
        """Classify a tick as aggressive buy/sell or passive"""
        mid_price = (bid + ask) / 2
        spread = ask - bid
        
        if price >= ask:
            direction = FlowDirection.AGGRESSIVE_BUY
            delta = volume
        elif price <= bid:
            direction = FlowDirection.AGGRESSIVE_SELL
            delta = -volume
        elif price > mid_price:
            direction = FlowDirection.PASSIVE_BUY
            delta = volume * 0.5
        elif price < mid_price:
            direction = FlowDirection.PASSIVE_SELL
            delta = -volume * 0.5
        else:
            direction = FlowDirection.NEUTRAL
            delta = 0.0
            
        return OrderFlowTick(
            timestamp=0,
            price=price,
            volume=volume,
            bid=bid,
            ask=ask,
            direction=direction,
            delta=delta
        )
        
    def process_tick(self, price: float, volume: float, 
                    bid: float, ask: float, timestamp: float = 0) -> Optional[ImbalanceSignal]:
        """Process a new tick and check for imbalance signals"""
        prev_price = self.tick_buffer[-1].price if self.tick_buffer else price
        
        tick = self.classify_tick(price, volume, bid, ask, prev_price)
        tick.timestamp = timestamp
        
        self.tick_buffer.append(tick)
        self.cumulative_delta += tick.delta
        
        if len(self.tick_buffer) < self.sequence_min_length:
            return None
            
        return self._detect_imbalance()
        
    def _detect_imbalance(self) -> Optional[ImbalanceSignal]:
        """Detect order flow imbalance from recent ticks"""
        recent_ticks = list(self.tick_buffer)[-self.lookback_ticks:]
        
        buy_volume = sum(t.volume for t in recent_ticks 
                        if t.direction in [FlowDirection.AGGRESSIVE_BUY, FlowDirection.PASSIVE_BUY])
        sell_volume = sum(t.volume for t in recent_ticks 
                         if t.direction in [FlowDirection.AGGRESSIVE_SELL, FlowDirection.PASSIVE_SELL])
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return None
            
        imbalance_ratio = (buy_volume - sell_volume) / total_volume
        
        if abs(imbalance_ratio) < self.imbalance_threshold:
            return None
            
        sequence_length = self._count_consecutive_direction(recent_ticks)
        
        if sequence_length < self.sequence_min_length:
            return None
            
        direction = "BUY" if imbalance_ratio > 0 else "SELL"
        confidence = min(1.0, abs(imbalance_ratio) * (sequence_length / 10))
        
        signal = ImbalanceSignal(
            timestamp=recent_ticks[-1].timestamp,
            imbalance_ratio=imbalance_ratio,
            cumulative_delta=self.cumulative_delta,
            direction=direction,
            confidence=confidence,
            sequence_length=sequence_length
        )
        
        self.imbalance_history.append(signal)
        return signal
        
    def _count_consecutive_direction(self, ticks: List[OrderFlowTick]) -> int:
        """Count consecutive ticks in the same direction"""
        if not ticks:
            return 0
            
        count = 1
        last_direction = ticks[-1].direction
        
        for tick in reversed(ticks[:-1]):
            if self._same_side(tick.direction, last_direction):
                count += 1
            else:
                break
                
        return count
        
    def _same_side(self, d1: FlowDirection, d2: FlowDirection) -> bool:
        """Check if two directions are on the same side"""
        buy_side = {FlowDirection.AGGRESSIVE_BUY, FlowDirection.PASSIVE_BUY}
        sell_side = {FlowDirection.AGGRESSIVE_SELL, FlowDirection.PASSIVE_SELL}
        
        return (d1 in buy_side and d2 in buy_side) or (d1 in sell_side and d2 in sell_side)
        
    def get_cumulative_delta(self) -> float:
        """Get current cumulative delta"""
        return self.cumulative_delta
        
    def reset_delta(self):
        """Reset cumulative delta"""
        self.cumulative_delta = 0.0


class AbsorptionClusterDetector:
    """
    Detects absorption and exhaustion clusters in order flow.
    
    Absorption: Large volume at a price level with minimal price movement
    Exhaustion: Volume spike followed by reversal, indicating trend end
    """
    
    def __init__(self,
                 volume_threshold_multiplier: float = 2.0,
                 price_tolerance: float = 0.001,
                 min_cluster_ticks: int = 10,
                 exhaustion_reversal_threshold: float = 0.5):
        self.volume_threshold_multiplier = volume_threshold_multiplier
        self.price_tolerance = price_tolerance
        self.min_cluster_ticks = min_cluster_ticks
        self.exhaustion_reversal_threshold = exhaustion_reversal_threshold
        
        self.volume_history: deque = deque(maxlen=500)
        self.price_history: deque = deque(maxlen=500)
        self.clusters: List[AbsorptionCluster] = []
        
    def process_bar(self, open_price: float, high: float, low: float, 
                   close: float, volume: float, timestamp: float) -> Optional[AbsorptionCluster]:
        """Process a price bar and detect absorption/exhaustion"""
        self.volume_history.append(volume)
        self.price_history.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'timestamp': timestamp
        })
        
        if len(self.volume_history) < 20:
            return None
            
        avg_volume = np.mean(list(self.volume_history)[-50:])
        
        if volume < avg_volume * self.volume_threshold_multiplier:
            return None
            
        price_range = high - low
        avg_price = (high + low) / 2
        relative_range = price_range / avg_price if avg_price > 0 else 0
        
        if relative_range < self.price_tolerance:
            return self._create_absorption_cluster(timestamp, avg_price, volume)
            
        if self._detect_exhaustion_pattern():
            return self._create_exhaustion_cluster(timestamp, close, volume)
            
        return None
        
    def _create_absorption_cluster(self, timestamp: float, price: float, 
                                   volume: float) -> AbsorptionCluster:
        """Create an absorption cluster"""
        recent_prices = [p['close'] for p in list(self.price_history)[-10:]]
        direction = "BUY" if recent_prices[-1] > recent_prices[0] else "SELL"
        
        cluster = AbsorptionCluster(
            start_time=timestamp,
            end_time=timestamp,
            price_level=price,
            total_volume=volume,
            absorbed_volume=volume * 0.8,
            direction=direction,
            strength=min(1.0, volume / (np.mean(list(self.volume_history)) * 3)),
            is_exhaustion=False
        )
        
        self.clusters.append(cluster)
        logger.info(f"Absorption cluster detected at {price:.4f}, direction: {direction}")
        return cluster
        
    def _detect_exhaustion_pattern(self) -> bool:
        """Detect exhaustion pattern (volume spike + reversal)"""
        if len(self.price_history) < 20:
            return False
            
        recent = list(self.price_history)[-20:]
        volumes = list(self.volume_history)[-20:]
        
        volume_spike_idx = np.argmax(volumes[-10:]) + 10
        
        if volume_spike_idx >= len(recent) - 2:
            return False
            
        pre_spike_trend = recent[volume_spike_idx]['close'] - recent[0]['close']
        post_spike_move = recent[-1]['close'] - recent[volume_spike_idx]['close']
        
        if pre_spike_trend != 0:
            reversal_ratio = -post_spike_move / pre_spike_trend
            return reversal_ratio > self.exhaustion_reversal_threshold
            
        return False
        
    def _create_exhaustion_cluster(self, timestamp: float, price: float,
                                   volume: float) -> AbsorptionCluster:
        """Create an exhaustion cluster"""
        recent_prices = [p['close'] for p in list(self.price_history)[-10:]]
        pre_direction = "BUY" if recent_prices[-5] > recent_prices[0] else "SELL"
        direction = "SELL" if pre_direction == "BUY" else "BUY"
        
        cluster = AbsorptionCluster(
            start_time=timestamp,
            end_time=timestamp,
            price_level=price,
            total_volume=volume,
            absorbed_volume=volume,
            direction=direction,
            strength=min(1.0, volume / (np.mean(list(self.volume_history)) * 2)),
            is_exhaustion=True
        )
        
        self.clusters.append(cluster)
        logger.info(f"Exhaustion cluster detected at {price:.4f}, reversal direction: {direction}")
        return cluster
        
    def get_recent_clusters(self, count: int = 10) -> List[AbsorptionCluster]:
        """Get most recent clusters"""
        return self.clusters[-count:]


class VolumeProfileAnalyzer:
    """
    Analyzes volume profile for institutional footprint detection.
    
    Identifies:
    - Point of Control (POC)
    - Value Area High/Low
    - High Volume Nodes (HVN)
    - Low Volume Nodes (LVN)
    """
    
    def __init__(self, num_bins: int = 50, value_area_pct: float = 0.70):
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct
        
    def calculate_profile(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Calculate volume profile from price and volume data"""
        if len(prices) == 0 or len(volumes) == 0:
            return {}
            
        price_min, price_max = np.min(prices), np.max(prices)
        
        if price_min == price_max:
            return {}
            
        bins = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_indices = np.digitize(prices, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        volume_profile = np.zeros(self.num_bins)
        for i, vol in zip(bin_indices, volumes):
            volume_profile[i] += vol
            
        poc_idx = np.argmax(volume_profile)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        vah, val = self._calculate_value_area(volume_profile, bins)
        
        hvn = self._find_high_volume_nodes(volume_profile, bins)
        lvn = self._find_low_volume_nodes(volume_profile, bins)
        
        return {
            "poc": poc_price,
            "vah": vah,
            "val": val,
            "hvn": hvn,
            "lvn": lvn,
            "volume_profile": volume_profile.tolist(),
            "price_bins": bins.tolist()
        }
        
    def _calculate_value_area(self, volume_profile: np.ndarray, 
                             bins: np.ndarray) -> Tuple[float, float]:
        """Calculate Value Area High and Low"""
        total_volume = np.sum(volume_profile)
        target_volume = total_volume * self.value_area_pct
        
        poc_idx = np.argmax(volume_profile)
        
        accumulated = volume_profile[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx
        
        while accumulated < target_volume:
            expand_low = low_idx > 0
            expand_high = high_idx < len(volume_profile) - 1
            
            if expand_low and expand_high:
                if volume_profile[low_idx - 1] > volume_profile[high_idx + 1]:
                    low_idx -= 1
                    accumulated += volume_profile[low_idx]
                else:
                    high_idx += 1
                    accumulated += volume_profile[high_idx]
            elif expand_low:
                low_idx -= 1
                accumulated += volume_profile[low_idx]
            elif expand_high:
                high_idx += 1
                accumulated += volume_profile[high_idx]
            else:
                break
                
        val = (bins[low_idx] + bins[low_idx + 1]) / 2
        vah = (bins[high_idx] + bins[high_idx + 1]) / 2
        
        return vah, val
        
    def _find_high_volume_nodes(self, volume_profile: np.ndarray, 
                                bins: np.ndarray, threshold_pct: float = 0.8) -> List[float]:
        """Find High Volume Nodes"""
        threshold = np.max(volume_profile) * threshold_pct
        hvn_indices = np.where(volume_profile >= threshold)[0]
        
        return [(bins[i] + bins[i + 1]) / 2 for i in hvn_indices]
        
    def _find_low_volume_nodes(self, volume_profile: np.ndarray,
                               bins: np.ndarray, threshold_pct: float = 0.2) -> List[float]:
        """Find Low Volume Nodes"""
        threshold = np.max(volume_profile) * threshold_pct
        lvn_indices = np.where(volume_profile <= threshold)[0]
        
        return [(bins[i] + bins[i + 1]) / 2 for i in lvn_indices]


class InstitutionalOrderFlowAnalyzer:
    """
    Main class combining all order flow analysis components.
    
    Provides unified interface for:
    - Order flow imbalance detection
    - Absorption/exhaustion cluster identification
    - Volume profile analysis
    - Institutional footprint scoring
    """
    
    def __init__(self):
        self.imbalance_detector = OrderFlowImbalanceDetector()
        self.cluster_detector = AbsorptionClusterDetector()
        self.volume_analyzer = VolumeProfileAnalyzer()
        
        self.signals: List[Dict] = []
        
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive order flow analysis.
        
        Args:
            market_data: Dictionary containing:
                - ticks: List of tick data (price, volume, bid, ask, timestamp)
                - ohlcv: DataFrame with OHLCV data
                - order_book: Current order book snapshot
                
        Returns:
            Analysis results with signals and metrics
        """
        results = {
            "timestamp": market_data.get("timestamp", 0),
            "imbalance_signal": None,
            "absorption_clusters": [],
            "volume_profile": {},
            "institutional_score": 0.0,
            "signal": None,
            "confidence": 0.0
        }
        
        if "ticks" in market_data:
            for tick in market_data["ticks"]:
                signal = self.imbalance_detector.process_tick(
                    price=tick["price"],
                    volume=tick["volume"],
                    bid=tick["bid"],
                    ask=tick["ask"],
                    timestamp=tick.get("timestamp", 0)
                )
                if signal:
                    results["imbalance_signal"] = {
                        "direction": signal.direction,
                        "confidence": signal.confidence,
                        "imbalance_ratio": signal.imbalance_ratio,
                        "cumulative_delta": signal.cumulative_delta
                    }
                    
        if "ohlcv" in market_data:
            ohlcv = market_data["ohlcv"]
            if isinstance(ohlcv, pd.DataFrame) and len(ohlcv) > 0:
                for _, row in ohlcv.iterrows():
                    cluster = self.cluster_detector.process_bar(
                        open_price=row.get("open", row.get("Open", 0)),
                        high=row.get("high", row.get("High", 0)),
                        low=row.get("low", row.get("Low", 0)),
                        close=row.get("close", row.get("Close", 0)),
                        volume=row.get("volume", row.get("Volume", 0)),
                        timestamp=row.name.timestamp() if hasattr(row.name, 'timestamp') else 0
                    )
                    if cluster:
                        results["absorption_clusters"].append({
                            "price_level": cluster.price_level,
                            "direction": cluster.direction,
                            "strength": cluster.strength,
                            "is_exhaustion": cluster.is_exhaustion
                        })
                        
                prices = ohlcv["close"].values if "close" in ohlcv.columns else ohlcv["Close"].values
                volumes = ohlcv["volume"].values if "volume" in ohlcv.columns else ohlcv["Volume"].values
                results["volume_profile"] = self.volume_analyzer.calculate_profile(prices, volumes)
                
        results["institutional_score"] = self._calculate_institutional_score(results)
        
        results["signal"], results["confidence"] = self._generate_signal(results)
        
        self.signals.append(results)
        
        return results
        
    def _calculate_institutional_score(self, results: Dict) -> float:
        """Calculate institutional activity score"""
        score = 0.0
        
        if results["imbalance_signal"]:
            score += abs(results["imbalance_signal"]["imbalance_ratio"]) * 0.4
            
        if results["absorption_clusters"]:
            cluster_strength = np.mean([c["strength"] for c in results["absorption_clusters"]])
            score += cluster_strength * 0.3
            
        if results["volume_profile"]:
            profile = results["volume_profile"].get("volume_profile", [])
            if profile:
                concentration = np.max(profile) / (np.sum(profile) + 1e-10)
                score += concentration * 0.3
                
        return min(1.0, score)
        
    def _generate_signal(self, results: Dict) -> Tuple[Optional[str], float]:
        """Generate trading signal from analysis"""
        signals = []
        confidences = []
        
        if results["imbalance_signal"]:
            signals.append(results["imbalance_signal"]["direction"])
            confidences.append(results["imbalance_signal"]["confidence"])
            
        for cluster in results["absorption_clusters"]:
            signals.append(cluster["direction"])
            confidences.append(cluster["strength"])
            
        if not signals:
            return None, 0.0
            
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        
        if buy_count > sell_count:
            direction = "BUY"
            confidence = np.mean([c for s, c in zip(signals, confidences) if s == "BUY"])
        elif sell_count > buy_count:
            direction = "SELL"
            confidence = np.mean([c for s, c in zip(signals, confidences) if s == "SELL"])
        else:
            return None, 0.0
            
        confidence *= results["institutional_score"]
        
        return direction, min(1.0, confidence)


def main():
    """Demo of institutional order flow analysis"""
    analyzer = InstitutionalOrderFlowAnalyzer()
    
    np.random.seed(42)
    
    ticks = []
    base_price = 100.0
    for i in range(100):
        price = base_price + np.random.randn() * 0.1
        volume = np.random.uniform(10, 100)
        spread = 0.02
        ticks.append({
            "price": price,
            "volume": volume,
            "bid": price - spread/2,
            "ask": price + spread/2,
            "timestamp": i
        })
        base_price = price
        
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1min')
    ohlcv = pd.DataFrame({
        'open': np.random.uniform(99, 101, 100),
        'high': np.random.uniform(100, 102, 100),
        'low': np.random.uniform(98, 100, 100),
        'close': np.random.uniform(99, 101, 100),
        'volume': np.random.uniform(100, 1000, 100)
    }, index=dates)
    
    market_data = {
        "timestamp": pd.Timestamp.now().timestamp(),
        "ticks": ticks,
        "ohlcv": ohlcv
    }
    
    results = analyzer.analyze(market_data)
    
    print("=== Institutional Order Flow Analysis ===")
    print(f"Institutional Score: {results['institutional_score']:.2f}")
    print(f"Signal: {results['signal']}")
    print(f"Confidence: {results['confidence']:.2f}")
    
    if results['imbalance_signal']:
        print(f"\nImbalance Signal:")
        print(f"  Direction: {results['imbalance_signal']['direction']}")
        print(f"  Ratio: {results['imbalance_signal']['imbalance_ratio']:.2f}")
        
    if results['absorption_clusters']:
        print(f"\nAbsorption Clusters: {len(results['absorption_clusters'])}")
        for cluster in results['absorption_clusters']:
            print(f"  Price: {cluster['price_level']:.2f}, Direction: {cluster['direction']}")
            
    if results['volume_profile']:
        print(f"\nVolume Profile:")
        print(f"  POC: {results['volume_profile'].get('poc', 'N/A')}")
        print(f"  VAH: {results['volume_profile'].get('vah', 'N/A')}")
        print(f"  VAL: {results['volume_profile'].get('val', 'N/A')}")


if __name__ == "__main__":
    main()
