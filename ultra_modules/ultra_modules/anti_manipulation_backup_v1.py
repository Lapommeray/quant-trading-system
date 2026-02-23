# ANTI-MANIPULATION DEFENSE (SHA3-512 HASHED)


class AntiManipulationSystem:
    def __init__(self):
        self.reality_anchors = {"BTC": 1.0, "ETH": 1.0, "GOLD": 1.0}

    def check_reality_anchor(self, asset, current_price):
        """Detect if price deviates beyond configured tolerance."""
        if asset not in self.reality_anchors:
            raise KeyError(f"Unknown asset: {asset}")
        expected = self.reality_anchors[asset]
        tolerance = expected * 0.9  # allow deviation up to 90% of the expected value in either direction
        return abs(current_price - expected) <= tolerance
