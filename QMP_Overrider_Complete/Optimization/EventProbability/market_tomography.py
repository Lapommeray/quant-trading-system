"""
market_tomography.py

Event Probability Engine for QMP Overrider

Provides Bayesian forecasting of market events and scenario weighting.
"""

import numpy as np
from datetime import datetime, timedelta
import random

class EventProbabilityEngine:
    """
    Event Probability Engine for QMP Overrider
    
    Provides Bayesian forecasting of market events and scenario weighting.
    """
    
    def __init__(self, algorithm=None):
        """
        Initialize the Event Probability Engine
        
        Parameters:
        - algorithm: QCAlgorithm instance (optional)
        """
        self.algorithm = algorithm
        self.event_registry = {}
        self.scenario_weights = {}
        self.last_forecast = None
        self.last_check_time = None
        self.confidence_threshold = 0.7
        self.lookback_days = 90
        self.initialized = False
    
    def initialize(self):
        """
        Initialize the Event Probability Engine
        
        Returns:
        - True if successful, False otherwise
        """
        if self.initialized:
            return True
        
        self._initialize_event_registry()
        
        self._initialize_scenario_weights()
        
        self.initialized = True
        
        if self.algorithm:
            self.algorithm.Debug("Event Probability Engine: Initialized")
        
        return True
    
    def _initialize_event_registry(self):
        """Initialize event registry"""
        event_types = [
            "fed_meeting",
            "earnings_release",
            "economic_data",
            "geopolitical_event",
            "market_crash",
            "liquidity_crisis",
            "regulatory_change",
            "black_swan"
        ]
        
        self.event_registry = {}
        
        for event_type in event_types:
            self.event_registry[event_type] = {
                "events": [],
                "prior_probability": self._get_default_prior(event_type),
                "impact_distribution": self._get_default_impact(event_type)
            }
    
    def _get_default_prior(self, event_type):
        """
        Get default prior probability for an event type
        
        Parameters:
        - event_type: Event type
        
        Returns:
        - Prior probability (0.0 to 1.0)
        """
        priors = {
            "fed_meeting": 0.2,
            "earnings_release": 0.3,
            "economic_data": 0.4,
            "geopolitical_event": 0.1,
            "market_crash": 0.01,
            "liquidity_crisis": 0.05,
            "regulatory_change": 0.1,
            "black_swan": 0.001
        }
        
        return priors.get(event_type, 0.1)
    
    def _get_default_impact(self, event_type):
        """
        Get default impact distribution for an event type
        
        Parameters:
        - event_type: Event type
        
        Returns:
        - Impact distribution parameters
        """
        impacts = {
            "fed_meeting": {"mean": 0.5, "std": 0.2},
            "earnings_release": {"mean": 0.3, "std": 0.3},
            "economic_data": {"mean": 0.2, "std": 0.1},
            "geopolitical_event": {"mean": 0.7, "std": 0.3},
            "market_crash": {"mean": 0.9, "std": 0.1},
            "liquidity_crisis": {"mean": 0.8, "std": 0.2},
            "regulatory_change": {"mean": 0.6, "std": 0.2},
            "black_swan": {"mean": 0.95, "std": 0.05}
        }
        
        return impacts.get(event_type, {"mean": 0.5, "std": 0.2})
    
    def _initialize_scenario_weights(self):
        """Initialize scenario weights"""
        scenarios = [
            "baseline",
            "bullish",
            "bearish",
            "volatile",
            "crash",
            "recovery"
        ]
        
        self.scenario_weights = {}
        
        for scenario in scenarios:
            self.scenario_weights[scenario] = self._get_default_weight(scenario)
    
    def _get_default_weight(self, scenario):
        """
        Get default weight for a scenario
        
        Parameters:
        - scenario: Scenario name
        
        Returns:
        - Scenario weight (0.0 to 1.0)
        """
        weights = {
            "baseline": 0.5,
            "bullish": 0.2,
            "bearish": 0.2,
            "volatile": 0.05,
            "crash": 0.01,
            "recovery": 0.04
        }
        
        return weights.get(scenario, 0.1)
    
    def register_event(self, event_type, event_data):
        """
        Register an event
        
        Parameters:
        - event_type: Event type
        - event_data: Event data
        
        Returns:
        - True if successful, False otherwise
        """
        if not self.initialized:
            self.initialize()
        
        if event_type not in self.event_registry:
            self.event_registry[event_type] = {
                "events": [],
                "prior_probability": self._get_default_prior(event_type),
                "impact_distribution": self._get_default_impact(event_type)
            }
        
        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now()
        
        self.event_registry[event_type]["events"].append(event_data)
        
        self.event_registry[event_type]["events"].sort(key=lambda x: x["timestamp"])
        
        self._update_prior(event_type)
        
        return True
    
    def _update_prior(self, event_type):
        """
        Update prior probability for an event type
        
        Parameters:
        - event_type: Event type
        """
        if event_type not in self.event_registry:
            return
        
        events = self.event_registry[event_type]["events"]
        
        now = datetime.now()
        lookback = now - timedelta(days=self.lookback_days)
        
        count = sum(1 for e in events if e["timestamp"] >= lookback)
        
        prior = count / self.lookback_days
        
        self.event_registry[event_type]["prior_probability"] = prior
    
    def forecast_events(self, market_state=None):
        """
        Forecast events based on market state
        
        Parameters:
        - market_state: Dictionary with market state information (optional)
        
        Returns:
        - Dictionary with event forecasts
        """
        if not self.initialized:
            self.initialize()
        
        now = datetime.now()
        self.last_check_time = now
        
        forecast = {
            "events": {},
            "scenarios": {},
            "timestamp": now
        }
        
        for event_type, registry in self.event_registry.items():
            forecast["events"][event_type] = self._forecast_event_type(event_type, market_state)
        
        self._update_scenario_weights(market_state)
        
        forecast["scenarios"] = self.scenario_weights.copy()
        
        self.last_forecast = forecast
        
        if self.algorithm:
            self.algorithm.Debug(f"Event Probability Engine: Forecast updated")
            
            high_prob_events = {k: v for k, v in forecast["events"].items() if v["probability"] > self.confidence_threshold}
            if high_prob_events:
                self.algorithm.Debug(f"High probability events: {list(high_prob_events.keys())}")
            
            dominant_scenario = max(forecast["scenarios"], key=forecast["scenarios"].get)
            self.algorithm.Debug(f"Dominant scenario: {dominant_scenario} ({forecast['scenarios'][dominant_scenario]:.2f})")
        
        return forecast
    
    def _forecast_event_type(self, event_type, market_state=None):
        """
        Forecast an event type
        
        Parameters:
        - event_type: Event type
        - market_state: Dictionary with market state information (optional)
        
        Returns:
        - Dictionary with event forecast
        """
        if event_type not in self.event_registry:
            return {
                "probability": 0.0,
                "impact": 0.0,
                "expected_value": 0.0,
                "confidence": 0.0
            }
        
        registry = self.event_registry[event_type]
        
        prior = registry["prior_probability"]
        
        impact_dist = registry["impact_distribution"]
        
        likelihood = self._calculate_likelihood(event_type, market_state)
        
        posterior = (prior * likelihood) / ((prior * likelihood) + ((1 - prior) * (1 - likelihood)))
        
        impact = random.normalvariate(impact_dist["mean"], impact_dist["std"])
        impact = max(0.0, min(1.0, impact))
        
        expected_value = posterior * impact
        
        confidence = 1.0 - (impact_dist["std"] / impact_dist["mean"])
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "probability": posterior,
            "impact": impact,
            "expected_value": expected_value,
            "confidence": confidence
        }
    
    def _calculate_likelihood(self, event_type, market_state=None):
        """
        Calculate likelihood of an event type based on market state
        
        Parameters:
        - event_type: Event type
        - market_state: Dictionary with market state information (optional)
        
        Returns:
        - Likelihood (0.0 to 1.0)
        """
        likelihood = 0.5
        
        if not market_state:
            return likelihood
        
        if event_type == "fed_meeting":
            if "inflation" in market_state and "unemployment" in market_state:
                inflation = market_state["inflation"]
                unemployment = market_state["unemployment"]
                
                inflation_target = 2.0
                unemployment_target = 4.0
                
                inflation_dev = abs(inflation - inflation_target) / inflation_target
                unemployment_dev = abs(unemployment - unemployment_target) / unemployment_target
                
                likelihood = 0.5 + 0.5 * max(inflation_dev, unemployment_dev)
        
        elif event_type == "earnings_release":
            if "day_in_quarter" in market_state:
                day = market_state["day_in_quarter"]
                
                if 30 <= day <= 45:
                    likelihood = 0.8
                else:
                    likelihood = 0.2
        
        elif event_type == "economic_data":
            if "day_of_week" in market_state:
                day = market_state["day_of_week"]
                
                if day in [3, 4]:  # 0=Monday, 4=Friday
                    likelihood = 0.7
                else:
                    likelihood = 0.3
        
        elif event_type == "geopolitical_event":
            if "global_tension" in market_state:
                tension = market_state["global_tension"]
                likelihood = tension
        
        elif event_type == "market_crash":
            if "volatility" in market_state and "leverage_ratio" in market_state:
                volatility = market_state["volatility"]
                leverage = market_state["leverage_ratio"]
                
                vol_factor = max(0.0, (volatility - 20) / 30)
                lev_factor = max(0.0, (leverage - 2) / 3)
                
                likelihood = max(vol_factor, lev_factor)
        
        elif event_type == "liquidity_crisis":
            if "liquidity" in market_state:
                liquidity = market_state["liquidity"]
                likelihood = 1.0 - liquidity
        
        elif event_type == "regulatory_change":
            if "recent_disruption" in market_state:
                disruption = market_state["recent_disruption"]
                likelihood = disruption
        
        elif event_type == "black_swan":
            likelihood = 0.01
            
            if "extreme_conditions" in market_state:
                extreme = market_state["extreme_conditions"]
                likelihood = 0.01 + 0.09 * extreme
        
        likelihood = max(0.0, min(1.0, likelihood))
        
        return likelihood
    
    def _update_scenario_weights(self, market_state=None):
        """
        Update scenario weights based on market state
        
        Parameters:
        - market_state: Dictionary with market state information (optional)
        """
        weights = {
            "baseline": 0.5,
            "bullish": 0.2,
            "bearish": 0.2,
            "volatile": 0.05,
            "crash": 0.01,
            "recovery": 0.04
        }
        
        if market_state:
            if "trend" in market_state:
                trend = market_state["trend"]
                
                if trend > 0.3:
                    weights["bullish"] += 0.2
                    weights["bearish"] -= 0.1
                    weights["crash"] -= 0.01
                    weights["recovery"] += 0.05
                elif trend < -0.3:
                    weights["bullish"] -= 0.1
                    weights["bearish"] += 0.2
                    weights["crash"] += 0.05
                    weights["recovery"] -= 0.05
            
            if "volatility" in market_state:
                volatility = market_state["volatility"]
                
                if volatility > 25:
                    weights["volatile"] += 0.1
                    weights["baseline"] -= 0.1
                    weights["crash"] += 0.05
                elif volatility < 10:
                    weights["volatile"] -= 0.03
                    weights["baseline"] += 0.05
                    weights["crash"] -= 0.01
            
            if "leverage_ratio" in market_state:
                leverage = market_state["leverage_ratio"]
                
                if leverage > 2.5:
                    weights["crash"] += 0.05
                    weights["volatile"] += 0.05
                    weights["baseline"] -= 0.05
            
            if "liquidity" in market_state:
                liquidity = market_state["liquidity"]
                
                if liquidity < 0.3:
                    weights["crash"] += 0.1
                    weights["volatile"] += 0.1
                    weights["baseline"] -= 0.1
                    weights["bullish"] -= 0.1
            
            if "recent_crash" in market_state and market_state["recent_crash"]:
                weights["recovery"] += 0.2
                weights["crash"] -= 0.01
                weights["baseline"] -= 0.1
        
        total = sum(weights.values())
        for scenario in weights:
            weights[scenario] /= total
        
        self.scenario_weights = weights
    
    def get_dominant_scenario(self):
        """
        Get the dominant scenario
        
        Returns:
        - Tuple with dominant scenario name and weight
        """
        if not self.scenario_weights:
            return ("baseline", 0.5)
        
        dominant = max(self.scenario_weights.items(), key=lambda x: x[1])
        
        return dominant
    
    def get_high_probability_events(self, threshold=None):
        """
        Get high probability events
        
        Parameters:
        - threshold: Probability threshold (optional)
        
        Returns:
        - Dictionary with high probability events
        """
        if not self.last_forecast:
            return {}
        
        if threshold is None:
            threshold = self.confidence_threshold
        
        high_prob = {}
        
        for event_type, forecast in self.last_forecast["events"].items():
            if forecast["probability"] > threshold:
                high_prob[event_type] = forecast
        
        return high_prob
    
    def get_status(self):
        """
        Get Event Probability Engine status
        
        Returns:
        - Dictionary with status information
        """
        return {
            "initialized": self.initialized,
            "event_types": list(self.event_registry.keys()),
            "scenarios": self.scenario_weights,
            "last_forecast": self.last_forecast,
            "last_check_time": self.last_check_time,
            "confidence_threshold": self.confidence_threshold
        }
