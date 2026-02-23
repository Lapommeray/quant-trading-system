from dataclasses import dataclass
from typing import Dict


@dataclass
class TradeDecision:
    action: str
    confidence: float
    entry: float
    stop_loss: float
    take_profit: float
    reason: str


class InstitutionalSignalOrchestrator:
    def __init__(self, confidence_threshold: float = 0.62):
        self.confidence_threshold = confidence_threshold

    def decide(self, price: float, trend_score: float, vol_score: float, liquidity_score: float) -> TradeDecision:
        raw_edge = 0.55 * trend_score + 0.25 * liquidity_score - 0.20 * vol_score
        # Normalize so that the edge is centered around 0 when inputs are 0.5
        # (midpoint of [0,1] range): max raw = 0.80, min raw = -0.20, mid = 0.30
        directional_edge = raw_edge - 0.30
        confidence = max(0.0, min(1.0, 0.5 + directional_edge))

        if confidence < self.confidence_threshold:
            return TradeDecision("HOLD", confidence, price, price, price, "Confidence below threshold")

        side = "BUY" if directional_edge >= 0 else "SELL"
        stop_distance = max(0.008, 0.012 + 0.01 * vol_score)
        take_distance = stop_distance * 2.0

        if side == "BUY":
            sl = price * (1 - stop_distance)
            tp = price * (1 + take_distance)
        else:
            sl = price * (1 + stop_distance)
            tp = price * (1 - take_distance)

        return TradeDecision(side, confidence, price, sl, tp, "Institutional multi-factor signal")

    def learn(self, metrics: Dict[str, float]) -> None:
        sharpe = metrics.get("sharpe", 0.0)
        drawdown = metrics.get("max_drawdown", 0.0)
        if sharpe > 1.5 and drawdown < 0.08:
            self.confidence_threshold = max(0.55, self.confidence_threshold - 0.01)
        elif drawdown > 0.12:
            self.confidence_threshold = min(0.75, self.confidence_threshold + 0.02)
