
import random

def evaluate_possible_timelines(price_data):
    """
    Simulates future price outcomes and chooses if a win-aligned timeline is active.
    """
    open_p = price_data['open']
    close_p = price_data['close']
    delta = close_p - open_p

    # Simulate directional outcomes (mocking 5 timelines)
    timelines = []
    for _ in range(5):
        sim_delta = delta + random.uniform(-1.5, 1.5)
        timelines.append(sim_delta)

    # Count how many are favorably aligned
    bullish_count = sum([1 for d in timelines if d > 0])
    bearish_count = sum([1 for d in timelines if d < 0])

    return {
        "bullish_timeline_dominance": bullish_count,
        "bearish_timeline_dominance": bearish_count,
        "winning_timeline": (
            bullish_count >= 4 if close_p > open_p else bearish_count >= 4
        )
    }
