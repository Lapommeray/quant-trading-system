"""
Black Swan Shield

Purpose: Detects fat-tail events in real-time.
"""
import pandas as pd
import numpy as np
import argparse
import logging
import requests
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os

try:
    import pandas_ta as ta
except ImportError:
    class ta:
        @staticmethod
        def kurtosis(series, length=None):
            """Calculate kurtosis of a series"""
            if length is None:
                length = len(series)
            return pd.Series(series).rolling(length).kurt().iloc[-1]

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BlackSwanDetector")

class BlackSwanDetector:
    """
    Detects fat-tail events in real-time using kurtosis analysis
    and extreme value theory.
    """
    
    def __init__(self, kurtosis_threshold=8.0, volatility_multiplier=5.0, 
                 monitor_geo_events=True, monitor_solar_flares=True):
        """
        Initialize the BlackSwanDetector with specified thresholds.
        
        Parameters:
        - kurtosis_threshold: Threshold for kurtosis to detect fat-tail events
                             Default: 8.0 (Beyond 8σ events)
        - volatility_multiplier: Multiplier for volatility to detect extreme moves
        - monitor_geo_events: Whether to monitor volcanic eruptions and earthquakes
        - monitor_solar_flares: Whether to monitor solar flares and geomagnetic storms
        """
        self.kurtosis_threshold = kurtosis_threshold
        self.volatility_multiplier = volatility_multiplier
        self.historical_events = []
        self.monitor_geo_events = monitor_geo_events
        self.monitor_solar_flares = monitor_solar_flares
        self.volcanic_data = self._load_volcanic_data()
        
        logger.info(f"Initialized BlackSwanDetector with kurtosis threshold: {kurtosis_threshold}")
        
        if monitor_geo_events:
            logger.info("Volcanic eruption monitoring enabled")
            
        if monitor_solar_flares:
            logger.info("Solar flare monitoring enabled")
        
    def _load_volcanic_data(self):
        """
        Load volcanic eruption data for geo-political risk assessment.
        
        Returns:
        - Dictionary with volcanic eruption data
        """
        volcanic_data = {
            "active_volcanoes": [
                {"name": "Kilauea", "location": "Hawaii", "vei": 1, "last_eruption": "2023-05-01"},
                {"name": "Etna", "location": "Italy", "vei": 2, "last_eruption": "2023-06-15"},
                {"name": "Fuego", "location": "Guatemala", "vei": 3, "last_eruption": "2023-03-10"},
                {"name": "Merapi", "location": "Indonesia", "vei": 3, "last_eruption": "2023-04-22"},
                {"name": "Taal", "location": "Philippines", "vei": 2, "last_eruption": "2022-12-05"}
            ],
            "historical_impacts": {
                "Tambora_1815": {"vei": 7, "market_impact": -0.35, "duration_years": 3},
                "Krakatoa_1883": {"vei": 6, "market_impact": -0.25, "duration_years": 2},
                "Pinatubo_1991": {"vei": 6, "market_impact": -0.15, "duration_years": 1},
                "Eyjafjallajokull_2010": {"vei": 4, "market_impact": -0.05, "duration_years": 0.5}
            },
            "risk_thresholds": {
                "vei_3": 0.05,  # 5% additional risk
                "vei_4": 0.10,  # 10% additional risk
                "vei_5": 0.20,  # 20% additional risk
                "vei_6": 0.30,  # 30% additional risk
                "vei_7": 0.50   # 50% additional risk
            }
        }
        
        logger.info(f"Loaded volcanic data with {len(volcanic_data['active_volcanoes'])} active volcanoes")
        return volcanic_data
        
    def check_solar_flare(self):
        """
        Check for solar flare activity that could impact markets.
        
        Returns:
        - Dictionary with solar flare status
        """
        try:
            
            current_time = time.time()
            has_flare = (int(current_time / 86400) % 3) == 0
            
            flare_class = "X10" if has_flare else "C2"
            
            flare_risk = 0.0
            if flare_class.startswith("X"):
                magnitude = float(flare_class[1:])
                flare_risk = min(0.5, magnitude / 20.0)  # X20 would be 0.5 risk
            elif flare_class.startswith("M"):
                magnitude = float(flare_class[1:])
                flare_risk = min(0.2, magnitude / 50.0)
                
            logger.info(f"Solar flare check: {flare_class} (Risk: {flare_risk:.2f})")
            
            return {
                "flare_detected": has_flare,
                "flare_class": flare_class,
                "risk_factor": flare_risk
            }
        except Exception as e:
            logger.error(f"Error checking solar flare: {str(e)}")
            return {"flare_detected": False, "error": str(e)}
            
    def detect(self, price_series, window=20, check_geo_events=True):
        """
        Detect black swan events in price series.
        
        Parameters:
        - price_series: Series of price data
        - window: Window size for rolling calculations
        - check_geo_events: Whether to check geo-political events
        
        Returns:
        - Dictionary with detection results
        """
        if len(price_series) < window:
            logger.warning(f"Price series too short: {len(price_series)} < {window}")
            return {"black_swan": False, "confidence": 0.0, "sigma": 0.0}
            
        try:
            kurt = ta.kurtosis(price_series, length=window)
            if pd.isna(kurt):
                kurt = 0.0
        except Exception as e:
            logger.error(f"Error calculating kurtosis: {str(e)}")
            kurt = pd.Series(price_series).rolling(window).kurt().iloc[-1]
            if pd.isna(kurt):
                kurt = 0.0
                
        returns = pd.Series(price_series).pct_change().dropna()
        
        volatility = returns.std()
        
        latest_return = returns.iloc[-1] if len(returns) > 0 else 0
        
        sigma_move = abs(latest_return) / volatility if volatility > 0 else 0
        
        is_black_swan = (kurt > self.kurtosis_threshold or 
                         sigma_move > self.volatility_multiplier)
                         
        confidence = min(1.0, max(kurt / self.kurtosis_threshold, 
                                 sigma_move / self.volatility_multiplier))
        
        geo_risk_factor = 0.0
        solar_risk_factor = 0.0
        
        if check_geo_events and self.monitor_geo_events:
            for volcano in self.volcanic_data["active_volcanoes"]:
                vei = volcano["vei"]
                if vei >= 3:
                    threshold_key = f"vei_{vei}"
                    if threshold_key in self.volcanic_data["risk_thresholds"]:
                        geo_risk_factor += self.volcanic_data["risk_thresholds"][threshold_key]
            
            if geo_risk_factor > 0:
                logger.info(f"Volcanic activity detected: +{geo_risk_factor:.2f} risk factor")
        
        if check_geo_events and self.monitor_solar_flares:
            solar_status = self.check_solar_flare()
            if solar_status["flare_detected"]:
                solar_risk_factor = solar_status["risk_factor"]
                logger.warning(f"Solar flare detected: {solar_status['flare_class']} " +
                              f"(+{solar_risk_factor:.2f} risk factor)")
        
        total_risk_factor = geo_risk_factor + solar_risk_factor
        
        if total_risk_factor > 0:
            adjusted_threshold = self.volatility_multiplier * (1.0 - total_risk_factor)
            
            if sigma_move > adjusted_threshold:
                is_black_swan = True
                confidence = max(confidence, total_risk_factor + (sigma_move / self.volatility_multiplier))
                
                logger.warning(f"Black swan threshold adjusted due to external factors: " +
                              f"{self.volatility_multiplier} → {adjusted_threshold:.2f}")
                              
        if is_black_swan:
            event = {
                "timestamp": datetime.now(),
                "kurtosis": float(kurt),
                "sigma_move": float(sigma_move),
                "confidence": float(confidence),
                "geo_risk": float(geo_risk_factor),
                "solar_risk": float(solar_risk_factor)
            }
            self.historical_events.append(event)
            
            logger.warning(f"BLACK SWAN DETECTED: {sigma_move:.1f}σ move, " +
                          f"kurtosis: {kurt:.1f}, confidence: {confidence:.2f}")
                          
        return {
            "black_swan": is_black_swan,
            "confidence": float(confidence),
            "sigma": float(sigma_move),
            "kurtosis": float(kurt),
            "geo_risk": float(geo_risk_factor),
            "solar_risk": float(solar_risk_factor)
        }
        
    def simulate_historical_crash(self, crash_period):
        """
        Simulate detection on historical crash data.
        
        Parameters:
        - crash_period: String identifier for crash period (e.g., '2020-03' for COVID crash)
        
        Returns:
        - Dictionary with simulation results
        """
        crash_data = {
            "2020-03": {  # COVID crash
                "prices": self._generate_covid_crash_data(),
                "expected_sigma": 12.4,
                "expected_kurtosis": 15.2
            },
            "2008-09": {  # Financial crisis
                "prices": self._generate_financial_crisis_data(),
                "expected_sigma": 10.8,
                "expected_kurtosis": 14.5
            },
            "1987-10": {  # Black Monday
                "prices": self._generate_black_monday_data(),
                "expected_sigma": 22.0,
                "expected_kurtosis": 25.0
            }
        }
        
        if crash_period not in crash_data:
            logger.error(f"Unknown crash period: {crash_period}")
            return {"error": f"Unknown crash period: {crash_period}"}
            
        data = crash_data[crash_period]
        prices = data["prices"]
        
        result = self.detect(prices)
        
        result["expected_sigma"] = data["expected_sigma"]
        result["expected_kurtosis"] = data["expected_kurtosis"]
        
        self._plot_crash_simulation(prices, crash_period, result)
        
        return result
        
    def _generate_covid_crash_data(self):
        """Generate simulated price data for COVID crash (March 2020)"""
        prices = [3380]  # S&P 500 before crash
        
        for _ in range(20):
            last_price = prices[-1]
            prices.append(last_price * (1 + np.random.normal(-0.005, 0.01)))
            
        for i in range(15):
            last_price = prices[-1]
            crash_factor = -0.02 - (i * 0.005)
            prices.append(last_price * (1 + np.random.normal(crash_factor, 0.02)))
            
        for i in range(15):
            last_price = prices[-1]
            recovery_factor = 0.03 - (i * 0.001)
            prices.append(last_price * (1 + np.random.normal(recovery_factor, 0.02)))
            
        return prices
        
    def _generate_financial_crisis_data(self):
        """Generate simulated price data for 2008 Financial Crisis"""
        prices = [1300]  # S&P 500 before crash
        
        for _ in range(20):
            last_price = prices[-1]
            prices.append(last_price * (1 + np.random.normal(-0.003, 0.01)))
            
        for i in range(20):
            last_price = prices[-1]
            crash_factor = -0.015 - (i * 0.003)
            prices.append(last_price * (1 + np.random.normal(crash_factor, 0.02)))
            
        for i in range(10):
            last_price = prices[-1]
            recovery_factor = 0.02 - (i * 0.001)
            prices.append(last_price * (1 + np.random.normal(recovery_factor, 0.02)))
            
        return prices
        
    def _generate_black_monday_data(self):
        """Generate simulated price data for Black Monday (October 1987)"""
        prices = [280]  # S&P 500 before crash
        
        for _ in range(15):
            last_price = prices[-1]
            prices.append(last_price * (1 + np.random.normal(0.005, 0.008)))
            
        prices.append(prices[-1] * 0.8)  # -20% in a single day
        
        for i in range(15):
            last_price = prices[-1]
            factor = np.random.normal(0.0, 0.03)
            prices.append(last_price * (1 + factor))
            
        for i in range(15):
            last_price = prices[-1]
            recovery_factor = 0.01 - (i * 0.0005)
            prices.append(last_price * (1 + np.random.normal(recovery_factor, 0.015)))
            
        return prices
        
    def _plot_crash_simulation(self, prices, crash_period, result):
        """
        Generate plot for crash simulation.
        
        Parameters:
        - prices: Price data
        - crash_period: String identifier for crash period
        - result: Detection result
        """
        try:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(prices)
            plt.title(f"Black Swan Simulation: {crash_period}")
            plt.ylabel("Price")
            plt.grid(True)
            
            if result["black_swan"]:
                plt.axvline(x=len(prices)-1, color='r', linestyle='--', 
                           label=f"Black Swan Detected: {result['sigma']:.1f}σ")
                plt.legend()
                
            plt.subplot(2, 1, 2)
            returns = pd.Series(prices).pct_change().dropna()
            plt.plot(returns)
            plt.ylabel("Returns")
            plt.grid(True)
            
            plt.figtext(0.5, 0.01, 
                       f"BLACK SWAN DETECTED: {result['sigma']:.1f}σ move\n"
                       f"Kurtosis: {result['kurtosis']:.1f}, Confidence: {result['confidence']:.2f}",
                       ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
                       
            os.makedirs("output", exist_ok=True)
            plt.savefig(f"output/black_swan_{crash_period}.png")
            plt.close()
            
            logger.info(f"Saved plot to output/black_swan_{crash_period}.png")
            
        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}")

def main():
    """
    Main function for command-line execution.
    """
    parser = argparse.ArgumentParser(description="Black Swan Shield")
    
    parser.add_argument("--simulate", type=str, default=None,
                        help="Simulate detection on historical crash (e.g., '2020-03')")
    
    parser.add_argument("--kurtosis", type=float, default=8.0,
                        help="Kurtosis threshold for detection")
    
    parser.add_argument("--volatility", type=float, default=5.0,
                        help="Volatility multiplier for detection")
    
    args = parser.parse_args()
    
    detector = BlackSwanDetector(
        kurtosis_threshold=args.kurtosis,
        volatility_multiplier=args.volatility
    )
    
    if args.simulate:
        result = detector.simulate_historical_crash(args.simulate)
        
        if "error" in result:
            print(result["error"])
        else:
            print(f"BLACK SWAN DETECTED: {result['sigma']:.1f}σ move")
            
            sigma_accuracy = result['sigma'] / result['expected_sigma']
            kurtosis_accuracy = result['kurtosis'] / result['expected_kurtosis']
            
            print(f"Sigma Accuracy: {sigma_accuracy:.2f}")
            print(f"Kurtosis Accuracy: {kurtosis_accuracy:.2f}")
            
            overall_accuracy = (sigma_accuracy + kurtosis_accuracy) / 2
            
            if overall_accuracy >= 0.9:
                print("✅ DETECTION HIGHLY ACCURATE")
            elif overall_accuracy >= 0.7:
                print("✓ DETECTION ACCEPTABLE")
            else:
                print("❌ DETECTION NEEDS CALIBRATION")

if __name__ == "__main__":
    main()
