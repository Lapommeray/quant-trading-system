
import numpy as np

def is_zero_point_state(price_data):
    """
    Detects if current candle is in a zero-point (stillness before expansion).
    """
    open_p = price_data['open']
    high_p = price_data['high']
    low_p = price_data['low']
    close_p = price_data['close']

    # Calculate volatility range
    candle_range = high_p - low_p
    body_size = abs(close_p - open_p)

    if candle_range == 0:
        return False

    body_ratio = body_size / candle_range

    # Zero-point conditions:
    return (
        body_ratio < 0.1 and                # very small body (indecision)
        candle_range < 2.0 and              # very quiet candle
        abs(close_p - open_p) < 1.0         # near-perfect doji
    )
