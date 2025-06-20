class EventProbabilityEngine:
    def __init__(self, branch_data, entropy_baseline):
        self.branch_data = branch_data
        self.entropy_baseline = entropy_baseline

    def analyze_market_signals(self, signals):
        enhanced_signals = {}
        for signal in signals:
            entropy = self._calculate_entropy(signal)
            score = self._score_signal(signal, entropy)
            enhanced_signals[signal["event"]] = score
        return enhanced_signals

    def get_event_probabilities(self):
        probabilities = {}
        for event in self.branch_data:
            void_level = self._scan_liquidity_void(event)
            score = entropy_score(event) * (1 + void_level)
            probabilities[event] = normalize(score)
        return probabilities

    def _scan_liquidity_void(self, event):
        """Quantify liquidity vacuum for probability scaling."""
        orderflow = self.branch_data[event].get("order_flow", [])
        if not orderflow:
            return 0
        spread = max(orderflow) - min(orderflow)
        return spread / (sum(orderflow) + 1e-9)

    def _calculate_entropy(self, signal):
        return abs(hash(signal["value"])) % 100 / 100.0

    def _score_signal(self, signal, entropy):
        return (1 - entropy) * signal.get("magnitude", 1)

    def _aggregate_entropy(self, branches):
        return sum([abs(hash(branch)) % 100 for branch in branches]) / (100.0 * len(branches))

    def _normalize_entropy(self, entropy):
        return max(0.0, min(1.0, 1.0 - abs(entropy - self.entropy_baseline)))
