
import numpy as np

def detect_big_move_setup(price_data, threshold=0.1):
    """
    Detects compression + hidden accumulation/distribution signals.
    """
    open_p = price_data['open']
    high_p = price_data['high']
    low_p = price_data['low']
    close_p = price_data['close']

    range_ = high_p - low_p
    body = abs(close_p - open_p)

    # Compression condition: small candle body + tight range
    is_compressed = (range_ < 3.0 and body / (range_ + 1e-9) < 0.25)

    # Hidden pressure detection (based on close position)
    upper_close = close_p >= (high_p - 0.3 * range_)
    lower_close = close_p <= (low_p + 0.3 * range_)

    big_buy_signal = is_compressed and upper_close
    big_sell_signal = is_compressed and lower_close

    result = {
        "big_buy_imminent": big_buy_signal,
        "big_sell_imminent": big_sell_signal,
        "compression_detected": is_compressed,
        "body_ratio": round(body / (range_ + 1e-9), 3),
        "range": round(range_, 2),
        "close": close_p
    }

    return result
