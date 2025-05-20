
def get_latest_price(asset_symbol='XAUUSD'):
    """
    Simulated real-time OHLC data.
    Replace this later with real-time API if needed.
    """
    import random

    base_price = 2000.0 if asset_symbol == 'XAUUSD' else 30000.0
    variation = random.uniform(-5, 5)

    open_price = base_price + variation
    high_price = open_price + random.uniform(1, 10)
    low_price = open_price - random.uniform(1, 10)
    close_price = open_price + random.uniform(-5, 5)

    return {
        'open': round(open_price, 2),
        'high': round(high_price, 2),
        'low': round(low_price, 2),
        'close': round(close_price, 2)
    }
