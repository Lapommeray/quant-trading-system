import numpy as np
from scipy.stats import zscore

class WhaleDetector:
    def __init__(self, sigma_threshold=4.0, window_size=50):
        self.threshold = sigma_threshold
        self.window = window_size
    
    def detect_whale(self, order_book):
        volumes = np.array([bid[1] for bid in order_book['bids']])
        z_scores = zscore(volumes[-self.window:])
        anomalies = np.where(z_scores > self.threshold)[0]
        return {
            'whale_present': len(anomalies) > 0,
            'confidence': float(np.max(z_scores))
        }
