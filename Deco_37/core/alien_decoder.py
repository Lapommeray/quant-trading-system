
import numpy as np

def decode_alien_numerical_patterns(price_data):
    """
    Analyzes numerical sequences in OHLC data for non-human anomalies.
    Returns a score between 0.0 and 1.0.
    """

    try:
        values = np.array([
            price_data['open'],
            price_data['high'],
            price_data['low'],
            price_data['close']
        ])

        # Calculate harmonic fingerprint
        ratios = np.diff(values) / values[:-1]
        variance = np.var(ratios)

        # If price data follows non-random, alien-like symmetry, variance is low
        alien_score = 1.0 - min(1.0, variance * 10)

        return round(alien_score, 3)

    except Exception:
        return 0.0  # Fallback if data is malformed
