
# intention_decoder.py
# This module decodes the invisible market structure to detect hidden intention

class IntentionDecoder:
    def __init__(self, algorithm):
        self.algo = algorithm

    def decode(self, symbol, history_window):
        """
        Analyze the last N candles to detect internal structure, pressure, and market intent.
        :param symbol: The trading symbol
        :param history_window: List of TradeBars (e.g., 25 1-minute bars)
        :return: "BUY", "SELL", or "WAIT"
        """

        bullish_count = 0
        bearish_count = 0
        wicks = []
        ranges = []

        for bar in history_window:
            body = abs(bar.Close - bar.Open)
            total_range = bar.High - bar.Low
            wick_size = total_range - body
            wicks.append(wick_size)
            ranges.append(total_range)

            if bar.Close > bar.Open:
                bullish_count += 1
            elif bar.Close < bar.Open:
                bearish_count += 1

        wick_avg = sum(wicks) / len(wicks) if len(wicks) > 0 else 0
        range_avg = sum(ranges) / len(ranges) if len(ranges) > 0 else 0

        # Conditions that hint at hidden intent
        if bullish_count >= len(history_window) * 0.8 and wick_avg < range_avg * 0.2:
            return "BUY"

        if bearish_count >= len(history_window) * 0.8 and wick_avg < range_avg * 0.2:
            return "SELL"

        if wick_avg > range_avg * 0.5:
            self.algo.Debug("High indecision or trap detected. Standing down.")

        return "WAIT"
