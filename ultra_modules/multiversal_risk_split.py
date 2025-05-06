from ultra_modules.event_probability_engine import EventProbabilityEngine

class MultiversalRiskSplit:
    def __init__(self, branch_data, entropy_baseline):
        self.engine = EventProbabilityEngine(branch_data, entropy_baseline)
        self.capital = 0

    def calculate_split(self, capital):
        self.capital = capital
        results = []
        for t in range(9):  # 9-timeline simulation
            alt_probs = self._simulate_branch_entropy(t)
            alloc = self._simulate_allocation(alt_probs)
            results.append(alloc)
        return self._aggregate_timelines(results)

    def _simulate_branch_entropy(self, seed):
        np.random.seed(seed)
        return {
            event: np.random.dirichlet(np.ones(3))[0]
            for event in self.engine.branch_data
        }

    def _simulate_allocation(self, alt_probs):
        risk_alloc = {}
        for event, prob in alt_probs.items():
            weight = self._calculate_dimensional_weight(event)
            risk_alloc[event] = round(self.capital * prob * weight, 2)
        return risk_alloc

    def _aggregate_timelines(self, results):
        final_alloc = {}
        for alloc in results:
            for event, val in alloc.items():
                final_alloc[event] = final_alloc.get(event, 0) + val / len(results)
        return final_alloc

    def _calculate_dimensional_weight(self, event):
        return 1.0 if "critical" in event.lower() else 0.618
