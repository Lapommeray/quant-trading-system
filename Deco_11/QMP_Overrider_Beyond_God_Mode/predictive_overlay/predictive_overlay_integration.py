"""
Predictive Overlay Integration

Integrates all predictive overlay components into a unified system for the QMP Overrider,
providing advanced visualization and forecasting capabilities.
"""

from .neural_forecaster import NeuralForecaster
from .ghost_candle_projector import GhostCandleProjector
from .timeline_warp_plot import TimelineWarpPlot
from .future_zone_sensory import FutureZoneSensory

class PredictiveOverlaySystem:
    """
    Unified system that integrates all predictive overlay components,
    providing advanced visualization and forecasting capabilities.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the predictive overlay system.
        
        Parameters:
        - algorithm: The QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        
        self.neural_forecaster = NeuralForecaster(algorithm)
        self.ghost_projector = GhostCandleProjector(algorithm)
        self.timeline_warp = TimelineWarpPlot(algorithm)
        self.future_zone = FutureZoneSensory(algorithm)
        
        self.last_forecast = {}
        self.last_update_time = None
        self.forecast_accuracy = {}
        self.consciousness_level = 0.5
    
    def update(self, symbol, history_data, gate_scores, transcendent_signal=None):
        """
        Update all predictive overlay components with latest data.
        
        Parameters:
        - symbol: Trading symbol
        - history_data: Dictionary of DataFrames for different timeframes
        - gate_scores: Dictionary of gate scores from QMP engine
        - transcendent_signal: Signal from transcendent intelligence
        
        Returns:
        - Dictionary containing all predictive overlay data
        """
        symbol_str = str(symbol)
        
        if transcendent_signal and "consciousness_level" in transcendent_signal:
            self.consciousness_level = transcendent_signal["consciousness_level"]
        
        forecast = self.neural_forecaster.forecast(symbol, history_data)
        
        ghost_candles = self.ghost_projector.project_ghost_candles(
            symbol, 
            history_data, 
            transcendent_signal
        )
        
        timelines = self.timeline_warp.generate_timelines(
            symbol, 
            history_data, 
            transcendent_signal
        )
        
        future_zones = self.future_zone.generate_future_zones(
            symbol, 
            history_data, 
            gate_scores, 
            transcendent_signal
        )
        
        convergence = self.timeline_warp.get_convergence_analysis(symbol)
        
        self.last_forecast[symbol_str] = {
            "neural_forecast": forecast,
            "ghost_candles": ghost_candles,
            "timelines": timelines,
            "future_zones": future_zones,
            "convergence": convergence,
            "timestamp": self.algorithm.Time
        }
        
        self.last_update_time = self.algorithm.Time
        
        return {
            "symbol": symbol_str,
            "timestamp": self.algorithm.Time,
            "neural_forecast": forecast,
            "ghost_candles": ghost_candles,
            "timelines": timelines,
            "future_zones": future_zones,
            "convergence": convergence,
            "consciousness_level": self.consciousness_level
        }
    
    def get_forecast_data(self, symbol):
        """
        Get the latest forecast data for a symbol.
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Latest forecast data
        """
        symbol_str = str(symbol)
        
        if symbol_str in self.last_forecast:
            return self.last_forecast[symbol_str]
            
        return None
    
    def evaluate_accuracy(self, symbol, actual_price):
        """
        Evaluate the accuracy of previous forecasts.
        
        Parameters:
        - symbol: Trading symbol
        - actual_price: The actual price that materialized
        
        Returns:
        - Accuracy evaluation results
        """
        symbol_str = str(symbol)
        
        ghost_accuracy = self.ghost_projector.calculate_projection_accuracy(symbol, {"Close": actual_price})
        
        zone_accuracy = self.future_zone.get_zone_accuracy(symbol, actual_price)
        
        if symbol_str not in self.forecast_accuracy:
            self.forecast_accuracy[symbol_str] = {
                "ghost_accuracy": [],
                "zone_accuracy": []
            }
            
        self.forecast_accuracy[symbol_str]["ghost_accuracy"].append(ghost_accuracy)
        self.forecast_accuracy[symbol_str]["zone_accuracy"].append(zone_accuracy)
        
        avg_ghost_accuracy = sum(self.forecast_accuracy[symbol_str]["ghost_accuracy"]) / \
                            len(self.forecast_accuracy[symbol_str]["ghost_accuracy"])
                            
        avg_zone_accuracy = sum(self.forecast_accuracy[symbol_str]["zone_accuracy"]) / \
                           len(self.forecast_accuracy[symbol_str]["zone_accuracy"])
        
        combined_accuracy = (ghost_accuracy + zone_accuracy) / 2
        
        return {
            "symbol": symbol_str,
            "timestamp": self.algorithm.Time,
            "actual_price": actual_price,
            "ghost_accuracy": ghost_accuracy,
            "zone_accuracy": zone_accuracy,
            "combined_accuracy": combined_accuracy,
            "avg_ghost_accuracy": avg_ghost_accuracy,
            "avg_zone_accuracy": avg_zone_accuracy
        }
    
    def get_dashboard_data(self, symbol):
        """
        Get data formatted for dashboard visualization.
        
        Parameters:
        - symbol: Trading symbol
        
        Returns:
        - Dashboard-ready data
        """
        symbol_str = str(symbol)
        
        if symbol_str not in self.last_forecast:
            return None
            
        forecast = self.last_forecast[symbol_str]
        
        dashboard_data = {
            "symbol": symbol_str,
            "timestamp": forecast["timestamp"],
            "consciousness_level": self.consciousness_level,
            
            "forecast_direction": forecast["neural_forecast"].get("direction", "neutral") if forecast["neural_forecast"] else "neutral",
            "forecast_confidence": forecast["neural_forecast"].get("confidence", 0.5) if forecast["neural_forecast"] else 0.5,
            "forecast_prices": forecast["neural_forecast"].get("forecast_prices", []) if forecast["neural_forecast"] else [],
            
            "ghost_candles": [
                {
                    "time": str(candle["Time"]),
                    "open": candle["Open"],
                    "high": candle["High"],
                    "low": candle["Low"],
                    "close": candle["Close"],
                    "confidence": candle["Confidence"]
                }
                for candle in forecast["ghost_candles"]
            ] if forecast["ghost_candles"] else [],
            
            "timelines": forecast["timelines"].get("timelines", []) if forecast["timelines"] else [],
            
            "future_zones": forecast["future_zones"].get("future_zones", []) if forecast["future_zones"] else [],
            
            "convergence_zones": forecast["convergence"].get("high_convergence_zones", []) if forecast["convergence"] else []
        }
        
        if symbol_str in self.forecast_accuracy:
            dashboard_data["accuracy"] = {
                "ghost_accuracy": self.forecast_accuracy[symbol_str]["ghost_accuracy"][-1] if self.forecast_accuracy[symbol_str]["ghost_accuracy"] else 0.0,
                "zone_accuracy": self.forecast_accuracy[symbol_str]["zone_accuracy"][-1] if self.forecast_accuracy[symbol_str]["zone_accuracy"] else 0.0,
                "avg_ghost_accuracy": sum(self.forecast_accuracy[symbol_str]["ghost_accuracy"]) / len(self.forecast_accuracy[symbol_str]["ghost_accuracy"]) if self.forecast_accuracy[symbol_str]["ghost_accuracy"] else 0.0,
                "avg_zone_accuracy": sum(self.forecast_accuracy[symbol_str]["zone_accuracy"]) / len(self.forecast_accuracy[symbol_str]["zone_accuracy"]) if self.forecast_accuracy[symbol_str]["zone_accuracy"] else 0.0
            }
        
        return dashboard_data
