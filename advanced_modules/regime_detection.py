"""
Multi-Timeframe Regime Detection Module

Advanced market regime detection using:
- Volatility regime classification
- Correlation regime analysis
- Hidden Markov Models for regime switching
- Multi-timeframe aggregation
- Information gain-based entropy modulation for faster regime resolution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
import warnings

logger = logging.getLogger("RegimeDetection")

try:
    from .bayesian_market_state import BayesianMarketState
    BAYESIAN_AVAILABLE = True
except ImportError:
    try:
        from bayesian_market_state import BayesianMarketState
        BAYESIAN_AVAILABLE = True
    except ImportError:
        BAYESIAN_AVAILABLE = False

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not available, using simplified regime detection")


class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING_HIGH_VOL = "ranging_high_vol"
    RANGING_LOW_VOL = "ranging_low_vol"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class VolatilityRegime(Enum):
    """Volatility regime classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class CorrelationRegime(Enum):
    """Correlation regime classifications"""
    DECORRELATED = "decorrelated"
    NORMAL = "normal"
    HIGH_CORRELATION = "high_correlation"
    CRISIS_CORRELATION = "crisis_correlation"


@dataclass
class RegimeState:
    """Current regime state"""
    market_regime: MarketRegime
    volatility_regime: VolatilityRegime
    correlation_regime: CorrelationRegime
    confidence: float
    duration: int
    transition_probability: float
    timestamp: float


@dataclass
class RegimeSignal:
    """Trading signal based on regime"""
    regime: MarketRegime
    direction: Optional[str]
    confidence: float
    position_size_multiplier: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    recommended_strategy: str


class VolatilityRegimeDetector:
    """
    Detects volatility regime using multiple measures:
    - Realized volatility
    - Parkinson volatility
    - Garman-Klass volatility
    - GARCH-style volatility clustering
    """
    
    def __init__(self, 
                 lookback: int = 20,
                 vol_percentiles: Dict[str, float] = None):
        self.lookback = lookback
        self.vol_percentiles = vol_percentiles or {
            "very_low": 10,
            "low": 25,
            "normal_low": 40,
            "normal_high": 60,
            "high": 75,
            "extreme": 90
        }
        self.vol_history: deque = deque(maxlen=252)
        
    def calculate_realized_volatility(self, returns: np.ndarray) -> float:
        """Calculate realized volatility"""
        if len(returns) < 2:
            return 0.0
        return float(np.std(returns) * np.sqrt(252))
        
    def calculate_parkinson_volatility(self, highs: np.ndarray, 
                                       lows: np.ndarray) -> float:
        """Calculate Parkinson volatility (high-low based)"""
        if len(highs) < 2:
            return 0.0
            
        log_hl = np.log(highs / lows)
        return float(np.sqrt(np.mean(log_hl ** 2) / (4 * np.log(2))) * np.sqrt(252))
        
    def calculate_garman_klass_volatility(self, opens: np.ndarray, highs: np.ndarray,
                                          lows: np.ndarray, closes: np.ndarray) -> float:
        """Calculate Garman-Klass volatility"""
        if len(opens) < 2:
            return 0.0
            
        log_hl = np.log(highs / lows)
        log_co = np.log(closes / opens)
        
        gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        return float(np.sqrt(np.mean(gk) * 252))
        
    def detect_regime(self, ohlcv: pd.DataFrame) -> Tuple[VolatilityRegime, float]:
        """Detect current volatility regime"""
        if len(ohlcv) < self.lookback:
            return VolatilityRegime.NORMAL, 0.5
            
        closes = ohlcv['close'].values if 'close' in ohlcv.columns else ohlcv['Close'].values
        highs = ohlcv['high'].values if 'high' in ohlcv.columns else ohlcv['High'].values
        lows = ohlcv['low'].values if 'low' in ohlcv.columns else ohlcv['Low'].values
        opens = ohlcv['open'].values if 'open' in ohlcv.columns else ohlcv['Open'].values
        
        returns = np.diff(closes) / closes[:-1]
        
        realized_vol = self.calculate_realized_volatility(returns[-self.lookback:])
        parkinson_vol = self.calculate_parkinson_volatility(
            highs[-self.lookback:], lows[-self.lookback:]
        )
        gk_vol = self.calculate_garman_klass_volatility(
            opens[-self.lookback:], highs[-self.lookback:],
            lows[-self.lookback:], closes[-self.lookback:]
        )
        
        current_vol = (realized_vol + parkinson_vol + gk_vol) / 3
        self.vol_history.append(current_vol)
        
        if len(self.vol_history) < 20:
            return VolatilityRegime.NORMAL, 0.5
            
        vol_array = np.array(self.vol_history)
        percentile = np.percentile(vol_array, 
                                   [self.vol_percentiles["very_low"],
                                    self.vol_percentiles["low"],
                                    self.vol_percentiles["high"],
                                    self.vol_percentiles["extreme"]])
        
        if current_vol <= percentile[0]:
            regime = VolatilityRegime.VERY_LOW
            confidence = 1.0 - current_vol / percentile[0]
        elif current_vol <= percentile[1]:
            regime = VolatilityRegime.LOW
            confidence = (percentile[1] - current_vol) / (percentile[1] - percentile[0])
        elif current_vol <= percentile[2]:
            regime = VolatilityRegime.NORMAL
            confidence = 1.0 - abs(current_vol - np.median(vol_array)) / np.std(vol_array)
        elif current_vol <= percentile[3]:
            regime = VolatilityRegime.HIGH
            confidence = (current_vol - percentile[2]) / (percentile[3] - percentile[2])
        else:
            regime = VolatilityRegime.EXTREME
            confidence = min(1.0, (current_vol - percentile[3]) / percentile[3])
            
        return regime, max(0.0, min(1.0, confidence))


class CorrelationRegimeDetector:
    """
    Detects correlation regime across multiple assets.
    
    Identifies:
    - Normal correlation structure
    - Decorrelation (diversification opportunity)
    - High correlation (risk-on/risk-off)
    - Crisis correlation (everything moves together)
    """
    
    def __init__(self, lookback: int = 60, crisis_threshold: float = 0.8):
        self.lookback = lookback
        self.crisis_threshold = crisis_threshold
        self.correlation_history: deque = deque(maxlen=252)
        
    def calculate_rolling_correlation(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix from returns"""
        if returns_matrix.shape[0] < 2 or returns_matrix.shape[1] < 2:
            return np.eye(returns_matrix.shape[1] if returns_matrix.ndim > 1 else 1)
            
        return np.corrcoef(returns_matrix.T)
        
    def detect_regime(self, returns_dict: Dict[str, np.ndarray]) -> Tuple[CorrelationRegime, float]:
        """Detect correlation regime from multiple asset returns"""
        if len(returns_dict) < 2:
            return CorrelationRegime.NORMAL, 0.5
            
        min_length = min(len(r) for r in returns_dict.values())
        if min_length < self.lookback:
            return CorrelationRegime.NORMAL, 0.5
            
        returns_matrix = np.column_stack([
            r[-self.lookback:] for r in returns_dict.values()
        ])
        
        corr_matrix = self.calculate_rolling_correlation(returns_matrix)
        
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        avg_correlation = np.mean(np.abs(upper_tri))
        
        self.correlation_history.append(avg_correlation)
        
        if len(self.correlation_history) < 20:
            return CorrelationRegime.NORMAL, 0.5
            
        corr_array = np.array(self.correlation_history)
        historical_mean = np.mean(corr_array)
        historical_std = np.std(corr_array)
        
        if avg_correlation >= self.crisis_threshold:
            regime = CorrelationRegime.CRISIS_CORRELATION
            confidence = (avg_correlation - self.crisis_threshold) / (1 - self.crisis_threshold)
        elif avg_correlation > historical_mean + historical_std:
            regime = CorrelationRegime.HIGH_CORRELATION
            confidence = (avg_correlation - historical_mean) / (2 * historical_std)
        elif avg_correlation < historical_mean - historical_std:
            regime = CorrelationRegime.DECORRELATED
            confidence = (historical_mean - avg_correlation) / (2 * historical_std)
        else:
            regime = CorrelationRegime.NORMAL
            confidence = 1.0 - abs(avg_correlation - historical_mean) / historical_std
            
        return regime, max(0.0, min(1.0, confidence))


class HMMRegimeDetector:
    """
    Hidden Markov Model based regime detection.
    
    Uses HMM to identify latent market states and
    predict regime transitions.
    """
    
    def __init__(self, n_states: int = 4, lookback: int = 252):
        self.n_states = n_states
        self.lookback = lookback
        self.model = None
        self.fitted = False
        self.state_mapping = {
            0: MarketRegime.TRENDING_BULL,
            1: MarketRegime.TRENDING_BEAR,
            2: MarketRegime.RANGING_LOW_VOL,
            3: MarketRegime.RANGING_HIGH_VOL
        }
        
    def fit(self, returns: np.ndarray, volatility: np.ndarray) -> bool:
        """Fit HMM model to historical data"""
        if not HMM_AVAILABLE:
            logger.warning("HMM not available, using fallback")
            return False
            
        if len(returns) < self.lookback:
            return False
            
        try:
            features = np.column_stack([returns, volatility])
            
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(features)
                
            self.fitted = True
            return True
            
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            return False
            
    def predict_regime(self, returns: np.ndarray, 
                      volatility: np.ndarray) -> Tuple[MarketRegime, float, float]:
        """Predict current regime and transition probability"""
        if not self.fitted or not HMM_AVAILABLE:
            return self._fallback_prediction(returns, volatility)
            
        try:
            features = np.column_stack([returns[-self.lookback:], volatility[-self.lookback:]])
            
            hidden_states = self.model.predict(features)
            current_state = hidden_states[-1]
            
            state_probs = self.model.predict_proba(features)[-1]
            confidence = state_probs[current_state]
            
            transition_prob = self.model.transmat_[current_state].max()
            
            regime = self.state_mapping.get(current_state, MarketRegime.RANGING_LOW_VOL)
            
            return regime, confidence, transition_prob
            
        except Exception as e:
            logger.error(f"HMM prediction failed: {e}")
            return self._fallback_prediction(returns, volatility)
            
    def _fallback_prediction(self, returns: np.ndarray, 
                            volatility: np.ndarray) -> Tuple[MarketRegime, float, float]:
        """Fallback prediction without HMM"""
        if len(returns) < 20:
            return MarketRegime.RANGING_LOW_VOL, 0.5, 0.5
            
        recent_return = np.mean(returns[-20:])
        recent_vol = np.mean(volatility[-20:]) if len(volatility) >= 20 else 0.02
        
        vol_threshold = np.percentile(volatility, 75) if len(volatility) > 50 else 0.02
        
        if recent_return > 0.001 and recent_vol < vol_threshold:
            regime = MarketRegime.TRENDING_BULL
        elif recent_return < -0.001 and recent_vol < vol_threshold:
            regime = MarketRegime.TRENDING_BEAR
        elif recent_vol >= vol_threshold:
            regime = MarketRegime.RANGING_HIGH_VOL
        else:
            regime = MarketRegime.RANGING_LOW_VOL
            
        confidence = min(1.0, abs(recent_return) / 0.01 + 0.3)
        
        return regime, confidence, 0.5


class MultiTimeframeRegimeFilter:
    """
    Aggregates regime detection across multiple timeframes.
    
    Combines signals from different timeframes to produce
    a robust regime classification with confidence weighting.
    """
    
    TIMEFRAME_WEIGHTS = {
        "1m": 0.05,
        "5m": 0.10,
        "15m": 0.15,
        "1h": 0.25,
        "4h": 0.25,
        "1d": 0.20
    }
    
    def __init__(self):
        self.vol_detector = VolatilityRegimeDetector()
        self.corr_detector = CorrelationRegimeDetector()
        self.hmm_detector = HMMRegimeDetector()
        
        self.regime_history: deque = deque(maxlen=100)
        self.current_regime: Optional[RegimeState] = None
        self.regime_duration = 0
        
    def analyze_timeframe(self, ohlcv: pd.DataFrame, 
                         timeframe: str) -> Dict[str, Any]:
        """Analyze regime for a single timeframe"""
        vol_regime, vol_confidence = self.vol_detector.detect_regime(ohlcv)
        
        closes = ohlcv['close'].values if 'close' in ohlcv.columns else ohlcv['Close'].values
        returns = np.diff(closes) / closes[:-1]
        
        volatility = pd.Series(returns).rolling(20).std().fillna(0.01).values
        
        market_regime, hmm_confidence, transition_prob = self.hmm_detector.predict_regime(
            returns, volatility
        )
        
        return {
            "timeframe": timeframe,
            "market_regime": market_regime,
            "volatility_regime": vol_regime,
            "vol_confidence": vol_confidence,
            "hmm_confidence": hmm_confidence,
            "transition_prob": transition_prob,
            "weight": self.TIMEFRAME_WEIGHTS.get(timeframe, 0.1)
        }
        
    def aggregate_regimes(self, timeframe_results: List[Dict]) -> RegimeState:
        """Aggregate regime signals across timeframes"""
        regime_votes = {}
        vol_regime_votes = {}
        total_weight = 0
        weighted_confidence = 0
        weighted_transition = 0
        
        for result in timeframe_results:
            weight = result["weight"]
            total_weight += weight
            
            regime = result["market_regime"]
            if regime not in regime_votes:
                regime_votes[regime] = 0
            regime_votes[regime] += weight
            
            vol_regime = result["volatility_regime"]
            if vol_regime not in vol_regime_votes:
                vol_regime_votes[vol_regime] = 0
            vol_regime_votes[vol_regime] += weight
            
            weighted_confidence += result["hmm_confidence"] * weight
            weighted_transition += result["transition_prob"] * weight
            
        if total_weight > 0:
            weighted_confidence /= total_weight
            weighted_transition /= total_weight
            
        market_regime = max(regime_votes.keys(), key=lambda k: regime_votes[k])
        vol_regime = max(vol_regime_votes.keys(), key=lambda k: vol_regime_votes[k])
        
        if self.current_regime and self.current_regime.market_regime == market_regime:
            self.regime_duration += 1
        else:
            self.regime_duration = 1
            
        state = RegimeState(
            market_regime=market_regime,
            volatility_regime=vol_regime,
            correlation_regime=CorrelationRegime.NORMAL,
            confidence=weighted_confidence,
            duration=self.regime_duration,
            transition_probability=weighted_transition,
            timestamp=pd.Timestamp.now().timestamp()
        )
        
        self.current_regime = state
        self.regime_history.append(state)
        
        return state
        
    def detect_regime(self, data_by_timeframe: Dict[str, pd.DataFrame],
                     correlation_data: Optional[Dict[str, np.ndarray]] = None) -> RegimeState:
        """
        Main regime detection method.
        
        Args:
            data_by_timeframe: Dictionary of OHLCV DataFrames by timeframe
            correlation_data: Optional dictionary of returns for correlation analysis
            
        Returns:
            Current regime state
        """
        timeframe_results = []
        
        for timeframe, ohlcv in data_by_timeframe.items():
            if len(ohlcv) >= 20:
                result = self.analyze_timeframe(ohlcv, timeframe)
                timeframe_results.append(result)
                
        if not timeframe_results:
            return RegimeState(
                market_regime=MarketRegime.RANGING_LOW_VOL,
                volatility_regime=VolatilityRegime.NORMAL,
                correlation_regime=CorrelationRegime.NORMAL,
                confidence=0.5,
                duration=1,
                transition_probability=0.5,
                timestamp=pd.Timestamp.now().timestamp()
            )
            
        state = self.aggregate_regimes(timeframe_results)
        
        if correlation_data:
            corr_regime, corr_confidence = self.corr_detector.detect_regime(correlation_data)
            state.correlation_regime = corr_regime
            state.confidence = (state.confidence + corr_confidence) / 2
            
        return state
        
    def generate_signal(self, regime_state: RegimeState) -> RegimeSignal:
        """Generate trading signal based on regime"""
        regime = regime_state.market_regime
        vol_regime = regime_state.volatility_regime
        
        strategy_map = {
            MarketRegime.TRENDING_BULL: ("trend_following", "BUY", 1.2),
            MarketRegime.TRENDING_BEAR: ("trend_following", "SELL", 1.2),
            MarketRegime.RANGING_HIGH_VOL: ("mean_reversion", None, 0.7),
            MarketRegime.RANGING_LOW_VOL: ("breakout", None, 0.8),
            MarketRegime.BREAKOUT: ("momentum", "BUY", 1.0),
            MarketRegime.BREAKDOWN: ("momentum", "SELL", 1.0),
            MarketRegime.CRISIS: ("defensive", None, 0.3),
            MarketRegime.RECOVERY: ("accumulation", "BUY", 0.8)
        }
        
        strategy, direction, size_mult = strategy_map.get(
            regime, ("neutral", None, 0.5)
        )
        
        vol_adjustments = {
            VolatilityRegime.VERY_LOW: (0.5, 1.5),
            VolatilityRegime.LOW: (0.7, 1.3),
            VolatilityRegime.NORMAL: (1.0, 1.0),
            VolatilityRegime.HIGH: (1.5, 0.8),
            VolatilityRegime.EXTREME: (2.0, 0.5)
        }
        
        sl_mult, tp_mult = vol_adjustments.get(vol_regime, (1.0, 1.0))
        
        if regime_state.correlation_regime == CorrelationRegime.CRISIS_CORRELATION:
            size_mult *= 0.5
            sl_mult *= 1.5
            
        return RegimeSignal(
            regime=regime,
            direction=direction,
            confidence=regime_state.confidence,
            position_size_multiplier=size_mult,
            stop_loss_multiplier=sl_mult,
            take_profit_multiplier=tp_mult,
            recommended_strategy=strategy
        )


class InformationGainRegimeModulator:
    """
    Modulates regime detection using information gain from Bayesian market state.
    
    High IG indicates the system can learn a lot from the current market state,
    which effectively lowers the entropy of regime classification. This enables
    faster regime resolution in uncertain periods.
    
    Integration:
    - High IG → Lower effective entropy → Quicker regime convergence
    - Low IG → Normal entropy → Standard regime detection
    """
    
    def __init__(self, 
                 ig_weight: float = 0.3,
                 entropy_floor: float = 0.1):
        """
        Initialize the IG regime modulator.
        
        Args:
            ig_weight: Weight for IG in entropy modulation (0-1)
            entropy_floor: Minimum entropy to prevent over-confidence
        """
        self.ig_weight = ig_weight
        self.entropy_floor = entropy_floor
        
        if BAYESIAN_AVAILABLE:
            self.market_state = BayesianMarketState()
        else:
            self.market_state = None
            logger.warning("BayesianMarketState not available. IG modulation disabled.")
            
        self.metrics = {
            "total_modulations": 0,
            "avg_ig": 0.0,
            "avg_entropy_reduction": 0.0,
        }
        
    def update_market_state(self, event_data: Dict[str, Any]):
        """Update the underlying market state with new event"""
        if self.market_state:
            self.market_state.update_from_event(event_data)
            
    def modulate_entropy(self, 
                         base_entropy: float,
                         expected_energy: float = 1.0) -> Tuple[float, float]:
        """
        Modulate entropy score based on information gain.
        
        Higher IG → Lower effective entropy → Faster regime resolution.
        
        Args:
            base_entropy: Original entropy score (0-1)
            expected_energy: Expected energy for IG computation
            
        Returns:
            Tuple of (modulated_entropy, ig_value)
        """
        if not self.market_state:
            return base_entropy, 0.0
            
        ig = self.market_state.expected_info_gain(hypothetical_E=expected_energy)
        ig_normalized = min(ig, 1.0)
        
        entropy_reduction = ig_normalized * self.ig_weight
        modulated_entropy = max(
            self.entropy_floor,
            base_entropy * (1.0 - entropy_reduction)
        )
        
        self.metrics["total_modulations"] += 1
        n = self.metrics["total_modulations"]
        self.metrics["avg_ig"] = ((n - 1) * self.metrics["avg_ig"] + ig) / n
        self.metrics["avg_entropy_reduction"] = (
            (n - 1) * self.metrics["avg_entropy_reduction"] + entropy_reduction
        ) / n
        
        return modulated_entropy, ig
        
    def modulate_confidence(self, 
                            base_confidence: float,
                            expected_energy: float = 1.0) -> float:
        """
        Modulate confidence score based on information gain.
        
        Higher IG in low-confidence states → Boost confidence (learning opportunity).
        Higher IG in high-confidence states → Slight reduction (regime may be shifting).
        
        Args:
            base_confidence: Original confidence score (0-1)
            expected_energy: Expected energy for IG computation
            
        Returns:
            Modulated confidence score
        """
        if not self.market_state:
            return base_confidence
            
        ig = self.market_state.expected_info_gain(hypothetical_E=expected_energy)
        belief = self.market_state.get_state()
        
        if belief.confidence < 0.5 and ig > 0.1:
            confidence_boost = ig * 0.2
            return min(1.0, base_confidence + confidence_boost)
        elif belief.confidence > 0.7 and ig > 0.15:
            confidence_reduction = ig * 0.1
            return max(0.3, base_confidence - confidence_reduction)
            
        return base_confidence
        
    def get_regime_recommendation(self) -> str:
        """Get recommendation based on current IG state"""
        if not self.market_state:
            return "NORMAL"
            
        recommendation = self.market_state.get_exploration_recommendation()
        return recommendation
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get modulator metrics"""
        belief = self.market_state.get_state_dict() if self.market_state else {}
        return {
            **self.metrics,
            "recommendation": self.get_regime_recommendation(),
            "belief_state": belief,
        }


class RegimeAwareRiskManager:
    """
    Risk management that adapts to current regime.
    """
    
    def __init__(self, base_risk: float = 0.02):
        self.base_risk = base_risk
        self.regime_filter = MultiTimeframeRegimeFilter()
        
    def calculate_position_size(self, regime_state: RegimeState,
                               account_balance: float,
                               entry_price: float,
                               stop_loss: float) -> float:
        """Calculate regime-adjusted position size"""
        signal = self.regime_filter.generate_signal(regime_state)
        
        adjusted_risk = self.base_risk * signal.position_size_multiplier
        
        if regime_state.confidence < 0.5:
            adjusted_risk *= regime_state.confidence
            
        risk_amount = account_balance * adjusted_risk
        
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return 0.0
            
        position_size = risk_amount / risk_per_unit
        
        return position_size
        
    def calculate_stop_loss(self, regime_state: RegimeState,
                           entry_price: float,
                           direction: str,
                           atr: float) -> float:
        """Calculate regime-adjusted stop loss"""
        signal = self.regime_filter.generate_signal(regime_state)
        
        sl_distance = atr * 2 * signal.stop_loss_multiplier
        
        if direction == "BUY":
            return entry_price - sl_distance
        else:
            return entry_price + sl_distance
            
    def calculate_take_profit(self, regime_state: RegimeState,
                             entry_price: float,
                             direction: str,
                             atr: float) -> float:
        """Calculate regime-adjusted take profit"""
        signal = self.regime_filter.generate_signal(regime_state)
        
        tp_distance = atr * 3 * signal.take_profit_multiplier
        
        if direction == "BUY":
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance


def main():
    """Demo of regime detection"""
    np.random.seed(42)
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='1h')
    
    trend = np.cumsum(np.random.randn(500) * 0.001)
    noise = np.random.randn(500) * 0.01
    prices = 100 * np.exp(trend + noise)
    
    ohlcv_1h = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, 500)),
        'high': prices * (1 + np.random.uniform(0, 0.01, 500)),
        'low': prices * (1 - np.random.uniform(0, 0.01, 500)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 500)
    }, index=dates)
    
    ohlcv_4h = ohlcv_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    ohlcv_1d = ohlcv_1h.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    regime_filter = MultiTimeframeRegimeFilter()
    
    data_by_timeframe = {
        "1h": ohlcv_1h,
        "4h": ohlcv_4h,
        "1d": ohlcv_1d
    }
    
    regime_state = regime_filter.detect_regime(data_by_timeframe)
    
    print("=== Multi-Timeframe Regime Detection ===")
    print(f"Market Regime: {regime_state.market_regime.value}")
    print(f"Volatility Regime: {regime_state.volatility_regime.value}")
    print(f"Correlation Regime: {regime_state.correlation_regime.value}")
    print(f"Confidence: {regime_state.confidence:.2f}")
    print(f"Duration: {regime_state.duration} periods")
    print(f"Transition Probability: {regime_state.transition_probability:.2f}")
    
    signal = regime_filter.generate_signal(regime_state)
    
    print(f"\n=== Trading Signal ===")
    print(f"Direction: {signal.direction}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Position Size Multiplier: {signal.position_size_multiplier:.2f}")
    print(f"Stop Loss Multiplier: {signal.stop_loss_multiplier:.2f}")
    print(f"Take Profit Multiplier: {signal.take_profit_multiplier:.2f}")
    print(f"Recommended Strategy: {signal.recommended_strategy}")


if __name__ == "__main__":
    main()
