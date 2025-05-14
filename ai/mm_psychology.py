"""
Market Maker Psychology Module

Purpose: Predicts MM emotional bias and decision-making patterns.
"""
import numpy as np
import pandas as pd
import logging
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import os
from collections import defaultdict

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MMPsychology")

class MMPsychology:
    """
    Predicts market maker emotional bias and decision-making patterns
    using behavioral finance principles and machine learning.
    """
    
    def __init__(self, sensitivity=0.8, memory_length=50, emotion_threshold=0.7):
        """
        Initialize the MMPsychology with specified parameters.
        
        Parameters:
        - sensitivity: Detection sensitivity (0.0-1.0)
        - memory_length: Number of events to remember for pattern detection
        - emotion_threshold: Threshold for emotional state detection
        """
        self.sensitivity = sensitivity
        self.memory_length = memory_length
        self.emotion_threshold = emotion_threshold
        
        self.emotional_states = {
            "fear": 0.0,
            "greed": 0.0,
            "uncertainty": 0.0,
            "confidence": 0.0,
            "panic": 0.0
        }
        
        self.memory = []
        
        self.mm_profiles = {}
        
        logger.info(f"Initialized MMPsychology with sensitivity: {sensitivity}, "
                   f"memory: {memory_length}, threshold: {emotion_threshold}")
        
    def analyze_mm_psychology(self, market_data, order_flow=None, mm_identifier=None):
        """
        Analyze market maker psychology from market data and order flow.
        
        Parameters:
        - market_data: Dictionary with market data (prices, volumes, etc.)
        - order_flow: Optional order flow data
        - mm_identifier: Optional market maker identifier
        
        Returns:
        - Dictionary with psychological analysis
        """
        features = self._extract_features(market_data, order_flow)
        
        self._update_memory(features, mm_identifier)
        
        emotional_state = self._analyze_emotional_state(features)
        
        decision_bias = self._predict_decision_bias(emotional_state, mm_identifier)
        
        if mm_identifier:
            self._update_mm_profile(mm_identifier, emotional_state, decision_bias)
            
        self._log_emotional_shifts(emotional_state, mm_identifier)
        
        return {
            "emotional_state": emotional_state,
            "decision_bias": decision_bias,
            "confidence": self._calculate_confidence(features)
        }
        
    def _extract_features(self, market_data, order_flow=None):
        """
        Extract psychological features from market data.
        
        Parameters:
        - market_data: Dictionary with market data
        - order_flow: Optional order flow data
        
        Returns:
        - Dictionary of extracted features
        """
        features = {}
        
        if 'prices' in market_data and len(market_data['prices']) > 1:
            prices = market_data['prices']
            returns = np.diff(prices) / prices[:-1]
            
            features['volatility'] = np.std(returns)
            features['recent_trend'] = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            features['acceleration'] = np.diff(returns).mean() if len(returns) > 1 else 0
        else:
            features['volatility'] = 0.0
            features['recent_trend'] = 0.0
            features['acceleration'] = 0.0
            
        if 'volumes' in market_data and len(market_data['volumes']) > 1:
            volumes = market_data['volumes']
            
            features['volume_trend'] = np.mean(np.diff(volumes)) / np.mean(volumes[:-1])
            features['volume_volatility'] = np.std(volumes) / np.mean(volumes)
        else:
            features['volume_trend'] = 0.0
            features['volume_volatility'] = 0.0
            
        if order_flow:
            if 'cancellations' in order_flow and 'placements' in order_flow:
                cancel_rate = order_flow['cancellations'] / max(1, order_flow['placements'])
                features['cancel_rate'] = cancel_rate
            else:
                features['cancel_rate'] = 0.0
                
            if 'modifications' in order_flow:
                features['modification_rate'] = order_flow['modifications'] / max(1, order_flow['placements'])
            else:
                features['modification_rate'] = 0.0
        else:
            features['cancel_rate'] = 0.0
            features['modification_rate'] = 0.0
            
        if 'sentiment' in market_data:
            features['market_sentiment'] = market_data['sentiment']
        else:
            features['market_sentiment'] = 0.0
            
        return features
        
    def _update_memory(self, features, mm_identifier=None):
        """
        Update memory with new features.
        
        Parameters:
        - features: Dictionary of extracted features
        - mm_identifier: Optional market maker identifier
        """
        memory_entry = {
            'timestamp': datetime.now(),
            'features': features,
            'mm_identifier': mm_identifier
        }
        
        self.memory.append(memory_entry)
        
        if len(self.memory) > self.memory_length:
            self.memory = self.memory[-self.memory_length:]
            
    def _analyze_emotional_state(self, features):
        """
        Analyze emotional state from features.
        
        Parameters:
        - features: Dictionary of extracted features
        
        Returns:
        - Dictionary with emotional state scores
        """
        emotional_state = {
            "fear": 0.0,
            "greed": 0.0,
            "uncertainty": 0.0,
            "confidence": 0.0,
            "panic": 0.0
        }
        
        fear_score = features['volatility'] * max(0, -features['recent_trend'])
        emotional_state['fear'] = min(1.0, fear_score * 10)
        
        greed_score = max(0, features['recent_trend']) * max(0, features['volume_trend'])
        emotional_state['greed'] = min(1.0, greed_score * 10)
        
        uncertainty_score = features['cancel_rate'] * features['modification_rate']
        emotional_state['uncertainty'] = min(1.0, uncertainty_score * 5)
        
        confidence_score = max(0, features['recent_trend']) * (1 - min(1, features['volatility'] * 10))
        emotional_state['confidence'] = min(1.0, confidence_score * 10)
        
        panic_score = features['volatility'] * features['volume_volatility'] * max(0, -features['recent_trend'])
        emotional_state['panic'] = min(1.0, panic_score * 20)
        
        for emotion in emotional_state:
            emotional_state[emotion] *= self.sensitivity
            
        for emotion in self.emotional_states:
            self.emotional_states[emotion] = (
                0.8 * self.emotional_states[emotion] + 
                0.2 * emotional_state[emotion]
            )
            
        return emotional_state
        
    def _predict_decision_bias(self, emotional_state, mm_identifier=None):
        """
        Predict decision bias from emotional state.
        
        Parameters:
        - emotional_state: Dictionary with emotional state scores
        - mm_identifier: Optional market maker identifier
        
        Returns:
        - Dictionary with decision bias predictions
        """
        decision_bias = {
            "risk_aversion": 0.0,
            "momentum_chasing": 0.0,
            "contrarian": 0.0,
            "liquidity_hoarding": 0.0,
            "aggressive_positioning": 0.0
        }
        
        decision_bias['risk_aversion'] = (
            0.7 * emotional_state['fear'] + 
            0.3 * emotional_state['uncertainty']
        )
        
        decision_bias['momentum_chasing'] = (
            0.6 * emotional_state['greed'] + 
            0.4 * emotional_state['confidence']
        )
        
        decision_bias['contrarian'] = (
            0.8 * emotional_state['confidence'] * 
            (1 - emotional_state['fear'])
        )
        
        decision_bias['liquidity_hoarding'] = (
            0.5 * emotional_state['fear'] + 
            0.5 * emotional_state['panic']
        )
        
        decision_bias['aggressive_positioning'] = (
            0.7 * emotional_state['greed'] * 
            (1 - emotional_state['uncertainty'])
        )
        
        if mm_identifier and mm_identifier in self.mm_profiles:
            profile = self.mm_profiles[mm_identifier]
            
            for bias in decision_bias:
                if bias in profile['bias_tendency']:
                    decision_bias[bias] = (
                        0.7 * decision_bias[bias] + 
                        0.3 * profile['bias_tendency'][bias]
                    )
                    
        return decision_bias
        
    def _update_mm_profile(self, mm_identifier, emotional_state, decision_bias):
        """
        Update market maker profile with new data.
        
        Parameters:
        - mm_identifier: Market maker identifier
        - emotional_state: Dictionary with emotional state scores
        - decision_bias: Dictionary with decision bias predictions
        """
        if mm_identifier not in self.mm_profiles:
            self.mm_profiles[mm_identifier] = {
                'emotional_history': [],
                'bias_tendency': defaultdict(float),
                'volatility_response': 0.0,
                'trend_response': 0.0
            }
            
        profile = self.mm_profiles[mm_identifier]
        
        profile['emotional_history'].append({
            'timestamp': datetime.now(),
            'emotional_state': emotional_state
        })
        
        if len(profile['emotional_history']) > self.memory_length:
            profile['emotional_history'] = profile['emotional_history'][-self.memory_length:]
            
        for bias, value in decision_bias.items():
            profile['bias_tendency'][bias] = (
                0.9 * profile['bias_tendency'][bias] + 
                0.1 * value
            )
            
        if len(self.memory) >= 2:
            recent_features = [entry['features'] for entry in self.memory[-2:]]
            
            volatility_change = recent_features[1]['volatility'] - recent_features[0]['volatility']
            trend_change = recent_features[1]['recent_trend'] - recent_features[0]['recent_trend']
            
            emotional_change = {
                emotion: emotional_state[emotion] - self.emotional_states[emotion]
                for emotion in emotional_state
            }
            
            if volatility_change != 0:
                volatility_response = sum(emotional_change.values()) / volatility_change
                profile['volatility_response'] = (
                    0.9 * profile['volatility_response'] + 
                    0.1 * volatility_response
                )
                
            if trend_change != 0:
                trend_response = sum(emotional_change.values()) / trend_change
                profile['trend_response'] = (
                    0.9 * profile['trend_response'] + 
                    0.1 * trend_response
                )
                
    def _log_emotional_shifts(self, emotional_state, mm_identifier=None):
        """
        Log significant emotional shifts.
        
        Parameters:
        - emotional_state: Dictionary with emotional state scores
        - mm_identifier: Optional market maker identifier
        """
        for emotion, value in emotional_state.items():
            if value >= self.emotion_threshold and value > self.emotional_states[emotion] * 1.5:
                mm_str = f" for {mm_identifier}" if mm_identifier else ""
                logger.warning(f"Significant {emotion} increase{mm_str}: {value:.2f}")
                
    def _calculate_confidence(self, features):
        """
        Calculate confidence in psychological analysis.
        
        Parameters:
        - features: Dictionary of extracted features
        
        Returns:
        - Confidence score (0.0-1.0)
        """
        feature_count = sum(1 for value in features.values() if value != 0)
        feature_confidence = min(1.0, feature_count / 10)
        
        memory_confidence = min(1.0, len(self.memory) / self.memory_length)
        
        confidence = 0.7 * feature_confidence + 0.3 * memory_confidence
        
        return confidence
        
    def get_mm_profile(self, mm_identifier):
        """
        Get market maker profile.
        
        Parameters:
        - mm_identifier: Market maker identifier
        
        Returns:
        - Market maker profile or None if not found
        """
        return self.mm_profiles.get(mm_identifier)
        
    def simulate_mm_behavior(self, mm_type="aggressive", market_scenario="volatile"):
        """
        Simulate market maker behavior for testing.
        
        Parameters:
        - mm_type: Type of market maker to simulate
        - market_scenario: Market scenario to simulate
        
        Returns:
        - Simulated market data and analysis results
        """
        market_data = self._generate_market_data(market_scenario)
        
        order_flow = self._generate_order_flow(mm_type, market_scenario)
        
        mm_identifier = f"{mm_type}_mm"
        
        analysis = self.analyze_mm_psychology(market_data, order_flow, mm_identifier)
        
        self._plot_mm_simulation(market_data, analysis, mm_type, market_scenario)
        
        return {
            "market_data": market_data,
            "order_flow": order_flow,
            "analysis": analysis
        }
        
    def _generate_market_data(self, market_scenario):
        """
        Generate simulated market data.
        
        Parameters:
        - market_scenario: Market scenario to simulate
        
        Returns:
        - Simulated market data
        """
        market_data = {
            'prices': [],
            'volumes': [],
            'sentiment': 0.0
        }
        
        base_price = 50000.0  # Base price (e.g., BTC/USD)
        base_volume = 100.0  # Base volume
        
        if market_scenario == "volatile":
            price = base_price
            for _ in range(50):
                price += np.random.normal(0, 500)
                market_data['prices'].append(price)
                
            for _ in range(50):
                volume = base_volume * np.random.lognormal(0, 0.5)
                market_data['volumes'].append(volume)
                
            market_data['sentiment'] = -0.3
            
        elif market_scenario == "trending":
            price = base_price
            for _ in range(50):
                price += np.random.normal(100, 200)
                market_data['prices'].append(price)
                
            for i in range(50):
                volume = base_volume * (1 + i/100) * np.random.lognormal(0, 0.2)
                market_data['volumes'].append(volume)
                
            market_data['sentiment'] = 0.6
            
        elif market_scenario == "ranging":
            price = base_price
            for _ in range(50):
                price = base_price + np.random.normal(0, 200)
                market_data['prices'].append(price)
                
            for _ in range(50):
                volume = base_volume * np.random.lognormal(0, 0.1)
                market_data['volumes'].append(volume)
                
            market_data['sentiment'] = 0.0
            
        return market_data
        
    def _generate_order_flow(self, mm_type, market_scenario):
        """
        Generate simulated order flow.
        
        Parameters:
        - mm_type: Type of market maker to simulate
        - market_scenario: Market scenario to simulate
        
        Returns:
        - Simulated order flow
        """
        order_flow = {
            'placements': 100,
            'cancellations': 0,
            'modifications': 0
        }
        
        if mm_type == "aggressive":
            if market_scenario == "volatile":
                order_flow['cancellations'] = 80
                order_flow['modifications'] = 60
            elif market_scenario == "trending":
                order_flow['cancellations'] = 40
                order_flow['modifications'] = 30
            else:  # ranging
                order_flow['cancellations'] = 20
                order_flow['modifications'] = 10
                
        elif mm_type == "passive":
            if market_scenario == "volatile":
                order_flow['cancellations'] = 90
                order_flow['modifications'] = 70
            elif market_scenario == "trending":
                order_flow['cancellations'] = 30
                order_flow['modifications'] = 20
            else:  # ranging
                order_flow['cancellations'] = 10
                order_flow['modifications'] = 5
                
        elif mm_type == "neutral":
            if market_scenario == "volatile":
                order_flow['cancellations'] = 50
                order_flow['modifications'] = 40
            elif market_scenario == "trending":
                order_flow['cancellations'] = 30
                order_flow['modifications'] = 25
            else:  # ranging
                order_flow['cancellations'] = 15
                order_flow['modifications'] = 10
                
        return order_flow
        
    def _plot_mm_simulation(self, market_data, analysis, mm_type, market_scenario):
        """
        Generate plot for market maker simulation.
        
        Parameters:
        - market_data: Simulated market data
        - analysis: Analysis results
        - mm_type: Type of market maker simulated
        - market_scenario: Market scenario simulated
        """
        try:
            plt.figure(figsize=(12, 10))
            
            plt.subplot(3, 1, 1)
            plt.plot(market_data['prices'])
            plt.title(f"Market Maker Psychology Simulation: {mm_type.capitalize()} MM in {market_scenario.capitalize()} Market")
            plt.ylabel("Price")
            plt.grid(True)
            
            plt.subplot(3, 1, 2)
            plt.bar(range(len(market_data['volumes'])), market_data['volumes'], alpha=0.7)
            plt.ylabel("Volume")
            plt.grid(True)
            
            plt.subplot(3, 1, 3)
            emotional_state = analysis['emotional_state']
            plt.bar(emotional_state.keys(), emotional_state.values(), color='blue', alpha=0.7)
            plt.axhline(y=self.emotion_threshold, color='r', linestyle='--', 
                       label=f"Threshold ({self.emotion_threshold})")
            plt.title("Emotional State Analysis")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True)
            
            decision_bias = analysis['decision_bias']
            bias_text = "\n".join([f"{bias}: {value:.2f}" for bias, value in decision_bias.items()])
            
            plt.figtext(0.5, 0.01, 
                       f"Decision Bias:\n{bias_text}\n\nConfidence: {analysis['confidence']:.2f}",
                       ha="center", fontsize=10, 
                       bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})
                       
            os.makedirs("output", exist_ok=True)
            plt.savefig(f"output/mm_psychology_{mm_type}_{market_scenario}.png")
            plt.close()
            
            logger.info(f"Saved plot to output/mm_psychology_{mm_type}_{market_scenario}.png")
            
        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}")

def main():
    """
    Main function for command-line execution.
    """
    parser = argparse.ArgumentParser(description="Market Maker Psychology Module")
    
    parser.add_argument("--simulate", action="store_true",
                        help="Run market maker simulation")
    
    parser.add_argument("--mm-type", type=str, default="aggressive",
                        choices=["aggressive", "passive", "neutral"],
                        help="Type of market maker to simulate")
    
    parser.add_argument("--market-scenario", type=str, default="volatile",
                        choices=["volatile", "trending", "ranging"],
                        help="Market scenario to simulate")
    
    parser.add_argument("--sensitivity", type=float, default=0.8,
                        help="Detection sensitivity (0.0-1.0)")
    
    args = parser.parse_args()
    
    mm_psychology = MMPsychology(sensitivity=args.sensitivity)
    
    if args.simulate:
        result = mm_psychology.simulate_mm_behavior(
            mm_type=args.mm_type,
            market_scenario=args.market_scenario
        )
        
        analysis = result['analysis']
        
        print(f"\nMarket Maker Psychology Analysis: {args.mm_type.capitalize()} MM in {args.market_scenario.capitalize()} Market")
        print("\nEmotional State:")
        for emotion, value in analysis['emotional_state'].items():
            print(f"  - {emotion}: {value:.2f}")
            
        print("\nDecision Bias:")
        for bias, value in analysis['decision_bias'].items():
            print(f"  - {bias}: {value:.2f}")
            
        print(f"\nAnalysis Confidence: {analysis['confidence']:.2f}")
        
        dominant_emotion = max(analysis['emotional_state'].items(), key=lambda x: x[1])
        dominant_bias = max(analysis['decision_bias'].items(), key=lambda x: x[1])
        
        print("\nTrading Recommendation:")
        print(f"  - Dominant Emotion: {dominant_emotion[0]} ({dominant_emotion[1]:.2f})")
        print(f"  - Dominant Bias: {dominant_bias[0]} ({dominant_bias[1]:.2f})")
        
        if dominant_bias[0] == "risk_aversion" and dominant_bias[1] > 0.7:
            print("  - Consider counter-trend positions as MM likely to reduce liquidity")
        elif dominant_bias[0] == "momentum_chasing" and dominant_bias[1] > 0.7:
            print("  - Consider trend-following as MM likely to amplify moves")
        elif dominant_bias[0] == "contrarian" and dominant_bias[1] > 0.7:
            print("  - Be cautious of reversals as MM likely to fade extreme moves")
        elif dominant_bias[0] == "liquidity_hoarding" and dominant_bias[1] > 0.7:
            print("  - Expect widening spreads and reduced market depth")
        elif dominant_bias[0] == "aggressive_positioning" and dominant_bias[1] > 0.7:
            print("  - Prepare for increased volatility and potential stop hunts")

if __name__ == "__main__":
    main()
