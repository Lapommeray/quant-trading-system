import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime
from scipy.stats import zscore
from scipy.signal import hilbert
from enum import Enum
from dataclasses import dataclass

class MarketState(Enum):
    EXPANSION = 1
    CONTRACTION = 2
    TRANSITION = 3

@dataclass
class BreathConfig:
    base_risk: float = 0.02
    max_risk: float = 0.05
    min_risk: float = 0.005
    emotion_weights: Dict[str, float] = None

class DNABreath:
    """
    DNA Breath - Emotion to Risk Curve Transcription
    
    Converts market emotions and psychological states into quantified risk metrics
    using fractal breathing patterns and genetic sequence analysis.
    """
    
    def __init__(self):
        self.config = BreathConfig(
            emotion_weights={
                'fear': 0.3, 'greed': 1.4, 'faith': 0.7,
                'serenity': 0.5, 'bliss': 0.8, 'hope': 0.6,
                'rage': 1.7, 'void': 0.9, 'neutral': 1.0,
                'euphoria': 1.8, 'panic': 0.2, 'confidence': 1.1,
                'uncertainty': 0.6
            }
        )
        self.state = MarketState.TRANSITION
        self.emotion_history = []
        self.fractal_history = []
        self.dna_sequences = []
        
    def calculate_risk(self, emotion: str, current_volatility: float) -> float:
        """Calculate risk based on emotion and current market volatility"""
        state_modifier = {
            MarketState.EXPANSION: 1.2,
            MarketState.CONTRACTION: 0.8,
            MarketState.TRANSITION: 0.5
        }.get(self.state, 1.0)
        
        emotion_factor = self.transcribe_emotion(emotion)
        volatility_factor = current_volatility / 0.05
        breathing_factor = self._calculate_breathing_factor()
        
        raw_risk = self.config.base_risk * emotion_factor * volatility_factor * state_modifier * breathing_factor
        return max(self.config.min_risk, min(self.config.max_risk, raw_risk))
        
    def predict(self, data: Union[pd.DataFrame, Dict]) -> float:
        """Predict risk signal from market data"""
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                closes = data['Close'].values
            else:
                return 0.0
        else:
            return 0.0
            
        dna_pattern = self._extract_dna_pattern(closes)
        emotion = self._detect_market_emotion(closes)
        volatility = self._calculate_volatility(closes)
        
        risk_score = self.calculate_risk(emotion, volatility)
        
        if risk_score > 0.035:
            return -0.8
        elif risk_score < 0.015:
            return 0.8
        else:
            return 0.0
            
    def train(self, data):
        """Train the DNA breath model on historical data"""
        if hasattr(data, 'items'):
            for key, df in data.items():
                if hasattr(df, 'columns') and 'Close' in df.columns:
                    pattern = self._extract_dna_pattern(df['Close'].values)
                    self.dna_sequences.append(pattern)
                    
    def _detect_market_state(self, prices: List[float]) -> MarketState:
        """Classifies current market regime"""
        if len(prices) < 50:
            return MarketState.TRANSITION
            
        returns = np.diff(prices[-50:]) / prices[-51:-1]
        if np.mean(returns) > 0.001:
            return MarketState.EXPANSION
        elif np.mean(returns) < -0.001:
            return MarketState.CONTRACTION
        return MarketState.TRANSITION

    def transcribe_emotion(self, emotion: str) -> float:
        """Converts emotion to risk modifier"""
        return self.config.emotion_weights.get(emotion.lower(), 1.0)
        
    def _extract_dna_pattern(self, prices: np.ndarray) -> List[int]:
        """Extract binary DNA sequence from price movements"""
        if len(prices) < 2:
            return [0, 1, 0, 1]
            
        changes = np.diff(prices)
        binary_sequence = (changes > 0).astype(int)
        
        self.dna_sequences.append(binary_sequence[-10:].tolist())
        
        return binary_sequence[-10:].tolist()
        
    def _detect_market_emotion(self, prices: np.ndarray) -> str:
        """Detect dominant market emotion from price action"""
        if len(prices) < 14:
            return 'neutral'
            
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        momentum = np.mean(returns[-5:])
        
        if momentum > 0.02 and volatility > 0.03:
            return 'greed'
        elif momentum < -0.02 and volatility > 0.05:
            return 'panic'
        elif momentum > 0.01:
            return 'confidence'
        elif momentum < -0.01:
            return 'fear'
        elif volatility > 0.04:
            return 'uncertainty'
        else:
            return 'neutral'
            
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate current volatility"""
        if len(prices) < 2:
            return 0.02
            
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns[-14:]) if len(returns) >= 14 else np.std(returns)
        
    def _calculate_breathing_factor(self) -> float:
        """Calculate fractal breathing pattern factor"""
        if len(self.fractal_history) < 3:
            return 1.0
            
        recent_fractals = self.fractal_history[-3:]
        breathing_cycle = np.sin(np.mean(recent_fractals) * np.pi)
        
        return 0.8 + 0.4 * (breathing_cycle + 1) / 2
        
    def _calculate_shannon_entropy(self, sequence: np.ndarray) -> float:
        """Calculate Shannon entropy of binary sequence"""
        if len(sequence) == 0:
            return 0.0
            
        unique, counts = np.unique(sequence, return_counts=True)
        probabilities = counts / len(sequence)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy
        
    def _calculate_fractal_dimension(self, series: np.ndarray) -> float:
        """Calculate Higuchi fractal dimension"""
        n = len(series)
        if n < 4:
            return 1.0
            
        k = min(int(np.log2(n)), 10)
        l = []
        
        for i in range(1, k):
            m = n // (2**i)
            if m > 1:
                d = np.abs(np.diff(series[::m]))
                if len(d) > 0:
                    l.append(np.log(np.mean(d)))
                    
        if len(l) < 2:
            return 1.0
            
        x = np.arange(1, len(l)+1)
        fractal_dim = 1 + np.polyfit(x, l, 1)[0]
        
        self.fractal_history.append(fractal_dim)
        if len(self.fractal_history) > 100:
            self.fractal_history = self.fractal_history[-100:]
            
        return fractal_dim
        
    def analyze_emotional_state(self, market_data: Dict) -> Dict:
        """Analyze emotional state from market data"""
        if 'ohlcv' not in market_data:
            return {'emotion': 'neutral', 'confidence': 0.5}
            
        closes = np.array([x['close'] for x in market_data['ohlcv']])
        volumes = np.array([x['volume'] for x in market_data['ohlcv']])
        
        emotion = self._detect_market_emotion(closes)
        volatility = self._calculate_volatility(closes)
        fractal_dim = self._calculate_fractal_dimension(closes)
        
        volume_emotion = self._analyze_volume_emotion(volumes)
        
        confidence = min(1.0, (abs(fractal_dim - 1.5) + volatility) / 2)
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'fractal_dimension': fractal_dim,
            'volatility': volatility,
            'volume_emotion': volume_emotion,
            'dna_pattern': self._extract_dna_pattern(closes)
        }
        
    def _analyze_volume_emotion(self, volumes: np.ndarray) -> str:
        """Analyze emotion from volume patterns"""
        if len(volumes) < 5:
            return 'neutral'
            
        recent_vol = np.mean(volumes[-3:])
        historical_vol = np.mean(volumes[:-3]) if len(volumes) > 3 else recent_vol
        
        vol_ratio = recent_vol / (historical_vol + 1e-10)
        
        if vol_ratio > 2.0:
            return 'excitement'
        elif vol_ratio > 1.5:
            return 'interest'
        elif vol_ratio < 0.5:
            return 'apathy'
        elif vol_ratio < 0.7:
            return 'caution'
        else:
            return 'normal'
            
    def get_risk_curve(self, emotion: str, volatility_range: np.ndarray) -> np.ndarray:
        """Generate risk curve for given emotion across volatility range"""
        risk_curve = []
        
        for vol in volatility_range:
            risk = self.calculate_risk(emotion, vol)
            risk_curve.append(risk)
            
        return np.array(risk_curve)
        
    def transcribe_emotion_to_risk(self, emotional_state: Dict) -> Dict:
        """Transcribe emotional state to comprehensive risk assessment"""
        emotion = emotional_state.get('emotion', 'neutral')
        confidence = emotional_state.get('confidence', 0.5)
        volatility = emotional_state.get('volatility', 0.02)
        
        base_risk = self.calculate_risk(emotion, volatility)
        
        confidence_adjustment = 1.0 + (confidence - 0.5) * 0.4
        adjusted_risk = base_risk * confidence_adjustment
        
        risk_level = 'LOW'
        if adjusted_risk > 0.035:
            risk_level = 'HIGH'
        elif adjusted_risk > 0.025:
            risk_level = 'MEDIUM'
            
        return {
            'base_risk': base_risk,
            'adjusted_risk': min(self.config.max_risk, max(self.config.min_risk, adjusted_risk)),
            'risk_level': risk_level,
            'emotion': emotion,
            'confidence': confidence,
            'volatility': volatility,
            'recommendation': self._get_risk_recommendation(adjusted_risk)
        }
        
    def _get_risk_recommendation(self, risk_score: float) -> str:
        """Get trading recommendation based on risk score"""
        if risk_score > 0.04:
            return 'EXTREME_CAUTION'
        elif risk_score > 0.03:
            return 'REDUCE_POSITION'
        elif risk_score > 0.02:
            return 'NORMAL_TRADING'
        elif risk_score > 0.01:
            return 'INCREASE_POSITION'
        else:
            return 'AGGRESSIVE_TRADING'
