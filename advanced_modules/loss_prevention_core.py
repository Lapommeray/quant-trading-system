"""
Loss Prevention Core - Ultimate Loss-Avoidance Intelligence

Advanced loss prevention system with:
- Predictive stop-loss evolution using online ML
- Auto-pause trading on projected drawdown threshold
- Counter-alpha generation for detected weaknesses
- Real-time loss forecasting and preemptive adaptation

This module enables the system to proactively avoid losses
rather than just react to them.
"""

import os
import sys
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import deque
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LossPreventionCore")

try:
    from river import linear_model, preprocessing, metrics
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logger.warning("River not available. Using fallback online learning.")


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class TradingState(Enum):
    ACTIVE = "active"
    CAUTIOUS = "cautious"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class DrawdownForecast:
    timestamp: datetime
    current_drawdown: float
    predicted_drawdown_1h: float
    predicted_drawdown_24h: float
    confidence: float
    risk_level: RiskLevel
    recommended_action: str
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "current_drawdown": self.current_drawdown,
            "predicted_drawdown_1h": self.predicted_drawdown_1h,
            "predicted_drawdown_24h": self.predicted_drawdown_24h,
            "confidence": self.confidence,
            "risk_level": self.risk_level.value,
            "recommended_action": self.recommended_action
        }


@dataclass
class CounterAlpha:
    name: str
    description: str
    trigger_condition: str
    implementation: str
    expected_improvement: float
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "trigger_condition": self.trigger_condition,
            "implementation": self.implementation,
            "expected_improvement": self.expected_improvement,
            "created_at": self.created_at.isoformat()
        }


class OnlineDrawdownPredictor:
    """
    Online machine learning model for drawdown prediction.
    
    Uses River library for incremental learning, with fallback
    to simple exponential smoothing when River unavailable.
    """
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.returns_history: deque = deque(maxlen=lookback)
        self.drawdown_history: deque = deque(maxlen=lookback)
        self.predictions: List[Dict] = []
        
        if RIVER_AVAILABLE:
            self.model = preprocessing.StandardScaler() | linear_model.LinearRegression()
            self.metric = metrics.MAE()
        else:
            self.alpha = 0.3
            self.smoothed_drawdown = 0.0
            
    def update(self, returns: float, current_drawdown: float):
        """Update model with new data point"""
        self.returns_history.append(returns)
        self.drawdown_history.append(current_drawdown)
        
        if len(self.returns_history) < 10:
            return
            
        if RIVER_AVAILABLE:
            features = self._extract_features()
            if features:
                self.model.learn_one(features, current_drawdown)
        else:
            self.smoothed_drawdown = (
                self.alpha * current_drawdown + 
                (1 - self.alpha) * self.smoothed_drawdown
            )
            
    def _extract_features(self) -> Optional[Dict[str, float]]:
        """Extract features for prediction"""
        if len(self.returns_history) < 10:
            return None
            
        returns = list(self.returns_history)
        drawdowns = list(self.drawdown_history)
        
        return {
            "returns_mean": np.mean(returns[-10:]),
            "returns_std": np.std(returns[-10:]),
            "returns_skew": self._skewness(returns[-10:]),
            "drawdown_trend": drawdowns[-1] - drawdowns[-5] if len(drawdowns) >= 5 else 0,
            "max_recent_drawdown": max(drawdowns[-10:]),
            "volatility": np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns),
            "momentum": sum(returns[-5:]) if len(returns) >= 5 else 0,
        }
        
    def _skewness(self, data: List[float]) -> float:
        """Calculate skewness"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((np.array(data) - mean) / std) ** 3)
        
    def predict(self, horizon_hours: float = 1.0) -> Tuple[float, float]:
        """
        Predict future drawdown.
        
        Returns:
            Tuple of (predicted_drawdown, confidence)
        """
        if len(self.drawdown_history) < 10:
            return 0.0, 0.0
            
        if RIVER_AVAILABLE:
            features = self._extract_features()
            if features:
                prediction = self.model.predict_one(features)
                trend_factor = 1.0 + (horizon_hours / 24.0) * 0.1
                confidence = min(0.9, len(self.drawdown_history) / self.lookback)
                return prediction * trend_factor, confidence
                
        current = self.drawdown_history[-1]
        trend = (current - self.drawdown_history[-5]) / 5 if len(self.drawdown_history) >= 5 else 0
        prediction = current + trend * horizon_hours
        confidence = 0.5
        
        return max(0, prediction), confidence


class CounterAlphaGenerator:
    """
    Generates counter-alpha strategies for detected weaknesses.
    
    Analyzes performance patterns and proposes specific
    countermeasures to address identified issues.
    """
    
    COUNTER_STRATEGIES = {
        "high_volatility": {
            "name": "Volatility Dampener",
            "description": "Reduce position sizes during high volatility regimes",
            "implementation": """
def volatility_dampener(position_size: float, current_vol: float, baseline_vol: float) -> float:
    vol_ratio = current_vol / baseline_vol if baseline_vol > 0 else 1.0
    dampening_factor = 1.0 / max(1.0, vol_ratio)
    return position_size * dampening_factor
""",
            "expected_improvement": 0.15
        },
        "trend_reversal": {
            "name": "Trend Reversal Shield",
            "description": "Tighten stops and reduce exposure on trend reversal signals",
            "implementation": """
def trend_reversal_shield(signal: float, trend_strength: float, stop_distance: float) -> float:
    if trend_strength < 0.3:
        return stop_distance * 0.5
    return stop_distance
""",
            "expected_improvement": 0.10
        },
        "correlation_spike": {
            "name": "Correlation Diversifier",
            "description": "Reduce correlated positions when correlation spikes",
            "implementation": """
def correlation_diversifier(positions: dict, correlation_matrix: np.ndarray, threshold: float = 0.7) -> dict:
    adjusted = {}
    for symbol, size in positions.items():
        high_corr_count = sum(1 for c in correlation_matrix[symbol] if abs(c) > threshold)
        reduction = 1.0 / (1.0 + high_corr_count * 0.2)
        adjusted[symbol] = size * reduction
    return adjusted
""",
            "expected_improvement": 0.12
        },
        "drawdown_acceleration": {
            "name": "Drawdown Brake",
            "description": "Progressively reduce exposure as drawdown accelerates",
            "implementation": """
def drawdown_brake(current_dd: float, dd_velocity: float, max_position: float) -> float:
    if dd_velocity > 0.01:
        brake_factor = max(0.2, 1.0 - dd_velocity * 10)
        return max_position * brake_factor
    return max_position
""",
            "expected_improvement": 0.20
        },
        "liquidity_crisis": {
            "name": "Liquidity Guardian",
            "description": "Switch to more liquid instruments during liquidity stress",
            "implementation": """
def liquidity_guardian(spread: float, avg_spread: float, position_size: float) -> float:
    spread_ratio = spread / avg_spread if avg_spread > 0 else 1.0
    if spread_ratio > 2.0:
        return position_size * 0.5
    elif spread_ratio > 1.5:
        return position_size * 0.75
    return position_size
""",
            "expected_improvement": 0.08
        }
    }
    
    def __init__(self):
        self.generated_counters: List[CounterAlpha] = []
        
    def analyze_weakness(self, 
                         performance_data: Dict,
                         market_conditions: Dict) -> List[str]:
        """Analyze performance to identify weaknesses"""
        weaknesses = []
        
        if market_conditions.get("volatility", 0) > market_conditions.get("avg_volatility", 0.01) * 1.5:
            weaknesses.append("high_volatility")
            
        if performance_data.get("recent_trend_accuracy", 1.0) < 0.4:
            weaknesses.append("trend_reversal")
            
        if market_conditions.get("avg_correlation", 0) > 0.6:
            weaknesses.append("correlation_spike")
            
        dd_history = performance_data.get("drawdown_history", [])
        if len(dd_history) >= 2 and dd_history[-1] - dd_history[-2] > 0.02:
            weaknesses.append("drawdown_acceleration")
            
        if market_conditions.get("spread", 0) > market_conditions.get("avg_spread", 0.001) * 2:
            weaknesses.append("liquidity_crisis")
            
        return weaknesses
        
    def generate_counter(self, weakness: str) -> Optional[CounterAlpha]:
        """Generate counter-alpha for identified weakness"""
        if weakness not in self.COUNTER_STRATEGIES:
            return None
            
        strategy = self.COUNTER_STRATEGIES[weakness]
        
        counter = CounterAlpha(
            name=strategy["name"],
            description=strategy["description"],
            trigger_condition=weakness,
            implementation=strategy["implementation"],
            expected_improvement=strategy["expected_improvement"]
        )
        
        self.generated_counters.append(counter)
        logger.info(f"Generated counter-alpha: {counter.name} for {weakness}")
        
        return counter
        
    def get_all_counters(self, 
                         performance_data: Dict,
                         market_conditions: Dict) -> List[CounterAlpha]:
        """Get all applicable counter-alphas"""
        weaknesses = self.analyze_weakness(performance_data, market_conditions)
        counters = []
        
        for weakness in weaknesses:
            counter = self.generate_counter(weakness)
            if counter:
                counters.append(counter)
                
        return counters


class LossPreventionCore:
    """
    Main loss prevention system integrating all components.
    
    Features:
    - Real-time drawdown prediction
    - Automatic trading state management
    - Counter-alpha generation
    - Proactive risk adjustment
    """
    
    THRESHOLDS = {
        "drawdown_warning": 0.03,
        "drawdown_caution": 0.05,
        "drawdown_pause": 0.08,
        "drawdown_emergency": 0.12,
        "predicted_drawdown_multiplier": 1.5,
    }
    
    def __init__(self, 
                 base_dir: str = None,
                 alert_callback: Optional[Callable[[str, RiskLevel], None]] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(".")
        self.alert_callback = alert_callback
        
        self.predictor = OnlineDrawdownPredictor()
        self.counter_generator = CounterAlphaGenerator()
        
        self.trading_state = TradingState.ACTIVE
        self.current_drawdown = 0.0
        self.peak_equity = 100000.0
        self.current_equity = 100000.0
        
        self.forecasts: deque = deque(maxlen=1000)
        self.state_history: List[Dict] = []
        self.active_counters: List[CounterAlpha] = []
        
        self._lock = threading.Lock()
        
        logger.info("LossPreventionCore initialized")
        
    def update(self, 
               equity: float, 
               returns: float = 0.0,
               market_conditions: Optional[Dict] = None):
        """Update with new equity value"""
        with self._lock:
            self.current_equity = equity
            
            if equity > self.peak_equity:
                self.peak_equity = equity
                
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
            
            self.predictor.update(returns, self.current_drawdown)
            
            forecast = self._generate_forecast()
            self.forecasts.append(forecast)
            
            new_state = self._determine_state(forecast)
            if new_state != self.trading_state:
                self._transition_state(new_state, forecast)
                
            if market_conditions:
                performance_data = {
                    "drawdown_history": list(self.predictor.drawdown_history),
                    "recent_trend_accuracy": 0.5,
                }
                counters = self.counter_generator.get_all_counters(
                    performance_data, market_conditions
                )
                self.active_counters = counters
                
            return forecast
            
    def _generate_forecast(self) -> DrawdownForecast:
        """Generate drawdown forecast"""
        pred_1h, conf_1h = self.predictor.predict(horizon_hours=1.0)
        pred_24h, conf_24h = self.predictor.predict(horizon_hours=24.0)
        
        confidence = (conf_1h + conf_24h) / 2
        
        max_predicted = max(self.current_drawdown, pred_1h, pred_24h)
        
        if max_predicted >= self.THRESHOLDS["drawdown_emergency"]:
            risk_level = RiskLevel.EMERGENCY
            action = "EMERGENCY STOP: Liquidate all positions immediately"
        elif max_predicted >= self.THRESHOLDS["drawdown_pause"]:
            risk_level = RiskLevel.CRITICAL
            action = "PAUSE: Stop all new trades, consider reducing positions"
        elif max_predicted >= self.THRESHOLDS["drawdown_caution"]:
            risk_level = RiskLevel.HIGH
            action = "CAUTION: Reduce position sizes by 50%"
        elif max_predicted >= self.THRESHOLDS["drawdown_warning"]:
            risk_level = RiskLevel.MEDIUM
            action = "WARNING: Tighten stops, monitor closely"
        else:
            risk_level = RiskLevel.LOW
            action = "NORMAL: Continue trading with standard parameters"
            
        return DrawdownForecast(
            timestamp=datetime.now(),
            current_drawdown=self.current_drawdown,
            predicted_drawdown_1h=pred_1h,
            predicted_drawdown_24h=pred_24h,
            confidence=confidence,
            risk_level=risk_level,
            recommended_action=action
        )
        
    def _determine_state(self, forecast: DrawdownForecast) -> TradingState:
        """Determine trading state based on forecast"""
        if forecast.risk_level == RiskLevel.EMERGENCY:
            return TradingState.EMERGENCY_STOP
        elif forecast.risk_level == RiskLevel.CRITICAL:
            return TradingState.PAUSED
        elif forecast.risk_level in [RiskLevel.HIGH, RiskLevel.MEDIUM]:
            return TradingState.CAUTIOUS
        else:
            return TradingState.ACTIVE
            
    def _transition_state(self, new_state: TradingState, forecast: DrawdownForecast):
        """Handle state transition"""
        old_state = self.trading_state
        self.trading_state = new_state
        
        transition = {
            "timestamp": datetime.now().isoformat(),
            "from_state": old_state.value,
            "to_state": new_state.value,
            "trigger_drawdown": forecast.current_drawdown,
            "predicted_drawdown": forecast.predicted_drawdown_1h
        }
        
        self.state_history.append(transition)
        
        logger.warning(f"Trading state transition: {old_state.value} -> {new_state.value}")
        
        if self.alert_callback:
            self.alert_callback(
                f"State changed to {new_state.value}: {forecast.recommended_action}",
                forecast.risk_level
            )
            
    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on current state"""
        multipliers = {
            TradingState.ACTIVE: 1.0,
            TradingState.CAUTIOUS: 0.5,
            TradingState.PAUSED: 0.0,
            TradingState.EMERGENCY_STOP: 0.0
        }
        return multipliers.get(self.trading_state, 0.0)
        
    def should_trade(self) -> bool:
        """Check if trading is allowed"""
        return self.trading_state in [TradingState.ACTIVE, TradingState.CAUTIOUS]
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        recent_forecast = self.forecasts[-1] if self.forecasts else None
        
        return {
            "trading_state": self.trading_state.value,
            "current_drawdown": self.current_drawdown,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "position_multiplier": self.get_position_multiplier(),
            "should_trade": self.should_trade(),
            "latest_forecast": recent_forecast.to_dict() if recent_forecast else None,
            "active_counters": [c.to_dict() for c in self.active_counters],
            "state_transitions": len(self.state_history)
        }
        
    def reset_peak(self):
        """Reset peak equity (use with caution)"""
        self.peak_equity = self.current_equity
        self.current_drawdown = 0.0
        logger.info("Peak equity reset")
        
    def force_state(self, state: TradingState, reason: str = "Manual override"):
        """Force a specific trading state"""
        old_state = self.trading_state
        self.trading_state = state
        
        self.state_history.append({
            "timestamp": datetime.now().isoformat(),
            "from_state": old_state.value,
            "to_state": state.value,
            "reason": reason,
            "forced": True
        })
        
        logger.warning(f"Forced state change: {old_state.value} -> {state.value} ({reason})")


def demo():
    """Demonstration of loss prevention system"""
    print("=" * 60)
    print("LOSS PREVENTION CORE DEMO")
    print("=" * 60)
    
    def alert_handler(message: str, level: RiskLevel):
        print(f"[ALERT - {level.value.upper()}] {message}")
        
    core = LossPreventionCore(alert_callback=alert_handler)
    
    print("\n--- Simulating equity curve with drawdown ---")
    
    equity = 100000
    np.random.seed(42)
    
    for i in range(50):
        returns = np.random.randn() * 0.01
        
        if i > 30:
            returns -= 0.005
            
        equity *= (1 + returns)
        
        market_conditions = {
            "volatility": abs(returns) * 10,
            "avg_volatility": 0.01,
            "spread": 0.0002,
            "avg_spread": 0.0001,
            "avg_correlation": 0.3 + (i / 100)
        }
        
        forecast = core.update(equity, returns, market_conditions)
        
        if i % 10 == 0:
            print(f"\nStep {i}:")
            print(f"  Equity: ${equity:,.2f}")
            print(f"  Drawdown: {core.current_drawdown:.2%}")
            print(f"  State: {core.trading_state.value}")
            print(f"  Predicted DD (1h): {forecast.predicted_drawdown_1h:.2%}")
            print(f"  Action: {forecast.recommended_action}")
            
    print("\n--- Final Status ---")
    status = core.get_status()
    print(json.dumps(status, indent=2, default=str))
    
    print("\n--- Active Counter-Alphas ---")
    for counter in core.active_counters:
        print(f"  - {counter.name}: {counter.description}")
        
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()
