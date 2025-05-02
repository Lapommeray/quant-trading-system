
import datetime

def is_master_369_number(n):
    # Number is considered sacred if divisible by 3, 6, or 9
    return any(n % x == 0 for x in [3, 6, 9])

def sacred_369_alignment(price_data):
    """
    Check if all major OHLC values end in 3, 6, or 9 (vibrational sync)
    """
    try:
        values = [
            int(str(int(price_data['open']))[-1]),
            int(str(int(price_data['high']))[-1]),
            int(str(int(price_data['low']))[-1]),
            int(str(int(price_data['close']))[-1])
        ]
        return all(is_master_369_number(v) for v in values)
    except Exception:
        return False

def check_divine_timing(current_time):
    """
    Checks if current UTC time is within a divine time window
    Divine timing window = minute is 3, 6, 9, 12, 18, 21, 24, 27, 30, 33, ...
    """
    sacred_minutes = [x for x in range(0, 60) if x % 3 == 0]
    return current_time.minute in sacred_minutes
