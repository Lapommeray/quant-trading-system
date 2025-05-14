"""
Dark Liquidity Trap Detector

Purpose: Detects spoofing traps and manipulative order book patterns in dark pools.
"""
import numpy as np
import pandas as pd
import logging
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import os

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrapDetector")

class TrapDetector:
    """
    Detects spoofing traps and manipulative order book patterns in dark pools.
    Uses advanced pattern recognition to identify market manipulation.
    """
    
    def __init__(self, sensitivity=0.75, window_size=50, trap_threshold=0.8):
        """
        Initialize the TrapDetector with specified parameters.
        
        Parameters:
        - sensitivity: Detection sensitivity (0.0-1.0)
        - window_size: Window size for pattern detection
        - trap_threshold: Threshold for trap confirmation
        """
        self.sensitivity = sensitivity
        self.window_size = window_size
        self.trap_threshold = trap_threshold
        self.historical_traps = []
        
        logger.info(f"Initialized TrapDetector with sensitivity: {sensitivity}, "
                   f"window: {window_size}, threshold: {trap_threshold}")
        
    def detect_trap(self, order_book, volume_profile=None):
        """
        Detect spoofing traps in order book data.
        
        Parameters:
        - order_book: Dictionary with 'bids' and 'asks' arrays
        - volume_profile: Optional volume profile data
        
        Returns:
        - Dictionary with detection results
        """
        if not self._validate_order_book(order_book):
            return {"trap_detected": False, "confidence": 0.0, "type": None}
            
        features = self._extract_features(order_book, volume_profile)
        
        spoofing_score = self._detect_spoofing(features)
        
        layering_score = self._detect_layering(features)
        
        iceberg_score = self._detect_iceberg(features)
        
        trap_scores = {
            "spoofing": spoofing_score,
            "layering": layering_score,
            "iceberg": iceberg_score
        }
        
        max_score_type = max(trap_scores, key=trap_scores.get)
        max_score = trap_scores[max_score_type]
        
        adjusted_score = max_score * self.sensitivity
        
        trap_detected = adjusted_score >= self.trap_threshold
        
        if trap_detected:
            trap_event = {
                "timestamp": datetime.now(),
                "type": max_score_type,
                "confidence": adjusted_score,
                "features": features
            }
            self.historical_traps.append(trap_event)
            
            logger.warning(f"TRAP DETECTED: {max_score_type} with "
                          f"confidence: {adjusted_score:.2f}")
                          
        return {
            "trap_detected": trap_detected,
            "confidence": float(adjusted_score),
            "type": max_score_type if trap_detected else None,
            "all_scores": trap_scores
        }
        
    def _validate_order_book(self, order_book):
        """
        Validate order book data structure.
        
        Parameters:
        - order_book: Order book data to validate
        
        Returns:
        - Boolean indicating if order book is valid
        """
        if not isinstance(order_book, dict):
            logger.error("Order book must be a dictionary")
            return False
            
        if 'bids' not in order_book or 'asks' not in order_book:
            logger.error("Order book must contain 'bids' and 'asks' keys")
            return False
            
        if not order_book['bids'] or not order_book['asks']:
            logger.warning("Order book contains empty bids or asks")
            return False
            
        return True
        
    def _extract_features(self, order_book, volume_profile=None):
        """
        Extract features from order book for trap detection.
        
        Parameters:
        - order_book: Dictionary with 'bids' and 'asks' arrays
        - volume_profile: Optional volume profile data
        
        Returns:
        - Dictionary of extracted features
        """
        features = {}
        
        best_bid = max(bid[0] for bid in order_book['bids'])
        best_ask = min(ask[0] for ask in order_book['asks'])
        spread = best_ask - best_bid
        features['spread'] = spread
        
        bid_volume = sum(bid[1] for bid in order_book['bids'])
        ask_volume = sum(ask[1] for ask in order_book['asks'])
        volume_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        features['volume_imbalance'] = volume_imbalance
        
        features['bid_depth'] = len(order_book['bids'])
        features['ask_depth'] = len(order_book['asks'])
        
        bid_prices = [bid[0] for bid in order_book['bids']]
        ask_prices = [ask[0] for ask in order_book['asks']]
        
        if len(bid_prices) > 1:
            bid_clustering = np.std(np.diff(sorted(bid_prices)))
            features['bid_clustering'] = bid_clustering
        else:
            features['bid_clustering'] = 0.0
            
        if len(ask_prices) > 1:
            ask_clustering = np.std(np.diff(sorted(ask_prices)))
            features['ask_clustering'] = ask_clustering
        else:
            features['ask_clustering'] = 0.0
            
        bid_volumes = [bid[1] for bid in order_book['bids']]
        ask_volumes = [ask[1] for ask in order_book['asks']]
        
        features['bid_volume_std'] = np.std(bid_volumes) if len(bid_volumes) > 1 else 0.0
        features['ask_volume_std'] = np.std(ask_volumes) if len(ask_volumes) > 1 else 0.0
        
        if volume_profile is not None:
            features['volume_profile_correlation'] = self._calculate_volume_correlation(
                order_book, volume_profile
            )
        else:
            features['volume_profile_correlation'] = 0.0
            
        return features
        
    def _detect_spoofing(self, features):
        """
        Detect spoofing patterns in order book features.
        
        Parameters:
        - features: Dictionary of order book features
        
        Returns:
        - Spoofing score (0.0-1.0)
        """
        
        imbalance_score = abs(features['volume_imbalance'])
        
        volume_std_score = max(
            features['bid_volume_std'],
            features['ask_volume_std']
        ) / (sum(features['bid_volume_std'] + features['ask_volume_std']) + 1e-10)
        
        spoofing_score = (0.7 * imbalance_score + 0.3 * volume_std_score)
        
        return min(1.0, spoofing_score)
        
    def _detect_layering(self, features):
        """
        Detect layering patterns in order book features.
        
        Parameters:
        - features: Dictionary of order book features
        
        Returns:
        - Layering score (0.0-1.0)
        """
        
        depth_imbalance = abs(
            features['bid_depth'] - features['ask_depth']
        ) / (features['bid_depth'] + features['ask_depth'] + 1e-10)
        
        clustering_score = min(
            features['bid_clustering'],
            features['ask_clustering']
        ) + 1e-10
        clustering_score = 1.0 / (1.0 + clustering_score)
        
        layering_score = (0.6 * depth_imbalance + 0.4 * clustering_score)
        
        return min(1.0, layering_score)
        
    def _detect_iceberg(self, features):
        """
        Detect iceberg orders in order book features.
        
        Parameters:
        - features: Dictionary of order book features
        
        Returns:
        - Iceberg score (0.0-1.0)
        """
        
        spread_score = 1.0 / (1.0 + features['spread'])
        
        profile_score = features['volume_profile_correlation']
        
        iceberg_score = (0.5 * spread_score + 0.5 * profile_score)
        
        return min(1.0, iceberg_score)
        
    def _calculate_volume_correlation(self, order_book, volume_profile):
        """
        Calculate correlation between order book and volume profile.
        
        Parameters:
        - order_book: Dictionary with 'bids' and 'asks' arrays
        - volume_profile: Volume profile data
        
        Returns:
        - Correlation score (0.0-1.0)
        """
        
        return 0.5  # Placeholder value
        
    def simulate_trap(self, trap_type="spoofing"):
        """
        Simulate a specific type of trap for testing.
        
        Parameters:
        - trap_type: Type of trap to simulate ("spoofing", "layering", "iceberg")
        
        Returns:
        - Simulated order book and detection results
        """
        simulated_order_book = {
            "bids": [],
            "asks": []
        }
        
        base_price = 50000.0  # Base price (e.g., BTC/USD)
        
        if trap_type == "spoofing":
            for i in range(10):
                simulated_order_book["bids"].append(
                    [base_price - i * 10, 10.0]  # Normal bids
                )
            
            simulated_order_book["bids"].append(
                [base_price - 5, 100.0]  # Spoofing bid
            )
            
            for i in range(10):
                simulated_order_book["asks"].append(
                    [base_price + i * 10 + 10, 5.0]
                )
                
        elif trap_type == "layering":
            for i in range(20):
                simulated_order_book["bids"].append(
                    [base_price - i * 5, 2.0]  # Layered bids
                )
            
            for i in range(5):
                simulated_order_book["asks"].append(
                    [base_price + i * 10 + 10, 5.0]
                )
                
        elif trap_type == "iceberg":
            for i in range(10):
                simulated_order_book["bids"].append(
                    [base_price - i * 10, 5.0]  # Normal bids
                )
            
            simulated_order_book["bids"].append(
                [base_price - 15, 7.0]  # Iceberg tip
            )
            
            for i in range(10):
                simulated_order_book["asks"].append(
                    [base_price + i * 10 + 10, 5.0]
                )
                
        result = self.detect_trap(simulated_order_book)
        
        self._plot_trap_simulation(simulated_order_book, trap_type, result)
        
        return {
            "order_book": simulated_order_book,
            "detection": result
        }
        
    def _plot_trap_simulation(self, order_book, trap_type, result):
        """
        Generate plot for trap simulation.
        
        Parameters:
        - order_book: Simulated order book
        - trap_type: Type of trap simulated
        - result: Detection result
        """
        try:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            
            bid_prices = [bid[0] for bid in order_book['bids']]
            bid_volumes = [bid[1] for bid in order_book['bids']]
            ask_prices = [ask[0] for ask in order_book['asks']]
            ask_volumes = [ask[1] for ask in order_book['asks']]
            
            plt.bar(bid_prices, bid_volumes, color='green', alpha=0.5, label='Bids')
            
            plt.bar(ask_prices, ask_volumes, color='red', alpha=0.5, label='Asks')
            
            plt.title(f"Dark Liquidity Trap Simulation: {trap_type.capitalize()}")
            plt.xlabel("Price")
            plt.ylabel("Volume")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            scores = result['all_scores']
            plt.bar(scores.keys(), scores.values(), color='blue', alpha=0.7)
            plt.axhline(y=self.trap_threshold, color='r', linestyle='--', 
                       label=f"Threshold ({self.trap_threshold})")
            plt.title("Trap Detection Scores")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True)
            
            detection_text = "TRAP DETECTED" if result['trap_detected'] else "NO TRAP DETECTED"
            confidence_text = f"Confidence: {result['confidence']:.2f}"
            type_text = f"Type: {result['type']}" if result['trap_detected'] else ""
            
            plt.figtext(0.5, 0.01, 
                       f"{detection_text}\n{confidence_text}\n{type_text}",
                       ha="center", fontsize=12, 
                       bbox={"facecolor":"orange" if result['trap_detected'] else "green", 
                             "alpha":0.5, "pad":5})
                             
            os.makedirs("output", exist_ok=True)
            plt.savefig(f"output/trap_{trap_type}.png")
            plt.close()
            
            logger.info(f"Saved plot to output/trap_{trap_type}.png")
            
        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}")

def main():
    """
    Main function for command-line execution.
    """
    parser = argparse.ArgumentParser(description="Dark Liquidity Trap Detector")
    
    parser.add_argument("--simulate", type=str, default=None,
                        choices=["spoofing", "layering", "iceberg"],
                        help="Simulate a specific type of trap")
    
    parser.add_argument("--sensitivity", type=float, default=0.75,
                        help="Detection sensitivity (0.0-1.0)")
    
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Trap detection threshold (0.0-1.0)")
    
    args = parser.parse_args()
    
    detector = TrapDetector(
        sensitivity=args.sensitivity,
        trap_threshold=args.threshold
    )
    
    if args.simulate:
        result = detector.simulate_trap(trap_type=args.simulate)
        
        detection = result['detection']
        
        if detection['trap_detected']:
            print(f"TRAP DETECTED: {detection['type']} with "
                 f"confidence: {detection['confidence']:.2f}")
        else:
            print(f"NO TRAP DETECTED (highest score: {max(detection['all_scores'].values()):.2f})")
            
        print("\nDetection Scores:")
        for trap_type, score in detection['all_scores'].items():
            print(f"  - {trap_type}: {score:.2f}")
            
        print(f"\nThreshold: {args.threshold}")
        print(f"Sensitivity: {args.sensitivity}")

if __name__ == "__main__":
    main()
