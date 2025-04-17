
from data_fetcher import get_latest_price

def get_asset_direction(asset_symbol):
    data = get_latest_price(asset_symbol)
    if data['close'] > data['open']:
        return 'Bullish'
    elif data['close'] < data['open']:
        return 'Bearish'
    else:
        return 'Neutral'

def confirm_multiverse_alignment(primary_direction):
    """
    Confirms other asset classes are aligned in same direction.
    """
    btc_dir = get_asset_direction('BTCUSD')
    dxy_dir = get_asset_direction('DXY')
    us500_dir = get_asset_direction('US500')

    alignment_count = sum([
        btc_dir == primary_direction,
        dxy_dir == primary_direction,
        us500_dir == primary_direction
    ])

    return alignment_count >= 2  # At least 2 confirm
