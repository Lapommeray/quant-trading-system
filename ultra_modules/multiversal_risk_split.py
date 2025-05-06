from ultra_modules.event_probability_engine import EventProbabilityEngine

class MultiversalRiskSplit:
    def __init__(self, branch_data, entropy_baseline):
        self.engine = EventProbabilityEngine(branch_data, entropy_baseline)

    def calculate_split(self, capital):
        probabilities = self.engine.get_event_probabilities()
        risk_alloc = {}
        for event, prob in probabilities.items():
            weight = self._calculate_dimensional_weight(event)
            risk_alloc[event] = round(capital * prob * weight, 2)
        return risk_alloc

    def _calculate_dimensional_weight(self, event):
        return 1.0 if "critical" in event.lower() else 0.618
