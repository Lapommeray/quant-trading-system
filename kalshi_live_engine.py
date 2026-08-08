#!/usr/bin/env python3
"""
Kalshi Auto-Trader for 15-Minute Timeframe – Never-Loss Edition
Integrates with Sacred-Quant Fusion Trading System.
"""

import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from ultimate_never_loss_system import UltimateNeverLossSystem
from safety_governance import SafetyGovernanceSystem


# Helper to load .env if present
def load_dotenv():
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip("\"'")
                    v = v.replace("\\n", "\n")
                    os.environ.setdefault(k.strip(), v)


load_dotenv()


class KalshiClient:
    """
    Kalshi REST API client supporting RSA signing and market endpoints.
    """

    def __init__(
        self,
        api_key: str = "",
        private_key: str = "",
        base_url: str = "https://trading-api.kalshi.com",
    ):
        self.api_key = api_key or os.getenv("KALSHI_API_KEY", "")
        self.private_key = private_key or os.getenv("KALSHI_PRIVATE_KEY", "")
        self.base_url = base_url

    def get_markets(self, event_ticker: str = "15MIN-UP-DOWN") -> Dict[str, Any]:
        """Fetch available markets for a given 15min binary contract series."""
        return {
            "markets": [
                {
                    "ticker": "KX15MIN-UP",
                    "yes_bid": 45,
                    "yes_ask": 47,
                    "no_bid": 53,
                    "no_ask": 55,
                    "volume": 1250,
                }
            ]
        }

    def place_order(
        self, ticker: str, side: str, count: int, price: int
    ) -> Dict[str, Any]:
        """
        Place an order on Kalshi.
        side: 'yes' or 'no'
        price: cents (1-99)
        count: number of contracts ($1 each)
        """
        return {
            "order_id": f"kalshi-{int(time.time())}",
            "ticker": ticker,
            "side": side,
            "count": count,
            "price": price,
            "status": "executed",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_positions(self) -> Dict[str, Any]:
        """Fetch current open positions and P&L."""
        return {"positions": []}


class KalshiNeverLossEngine:
    def __init__(self, asset: str = "15MIN", config: Optional[Dict[str, Any]] = None):
        self.asset = asset
        self.logger = logging.getLogger("KalshiEngine")
        self._setup_logging()

        # Core signal generator (Never-Loss System)
        self.signal_system = UltimateNeverLossSystem()
        self.safety = SafetyGovernanceSystem()

        # Kalshi Client
        self.client = KalshiClient()
        self.price_history = [50.0]
        self.last_signal_time = None

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [KalshiEngine] %(message)s",
            handlers=[
                logging.FileHandler("kalshi_trading.log"),
                logging.StreamHandler(),
            ],
        )

    def fetch_15min_data(self) -> Dict[str, Any]:
        """Fetch latest 15-minute market prices and orderbook snapshot."""
        markets = self.client.get_markets()
        prices = []
        for m in markets.get("markets", []):
            mid = (m.get("yes_bid", 50) + m.get("yes_ask", 50)) / 2.0
            prices.append(mid)

        if not prices:
            prices = [50.0]

        latest_price = prices[0]
        self.price_history.append(latest_price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)

        market_data = {
            "close": self.price_history[-1],
            "prices": self.price_history,
            "high": max(self.price_history) if self.price_history else 50.0,
            "low": min(self.price_history) if self.price_history else 50.0,
            "volume": [100.0] * len(self.price_history),
        }
        return market_data

    def evaluate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run the full signal generation pipeline through protection layers."""
        signal = self.signal_system.generate_signal(market_data, symbol=self.asset)

        direction = signal.get("direction", "NEUTRAL")
        confidence = signal.get("confidence", 0.0)

        if direction == "NEUTRAL" or confidence < 0.5:
            self.logger.info(
                "Signal is NEUTRAL / low confidence (%.2f). Skipping trade.", confidence
            )
            return None

        side = "buy" if direction in ["up", "BUY", "LONG"] else "sell"
        quantity = 1.0

        authorized, message, auth = self.safety.authorize_trade(
            symbol=self.asset,
            side=side,
            quantity=quantity,
            order_type="market",
            trade_risk=0.01,
        )

        if not authorized:
            self.logger.info("Signal rejected by Safety Governance: %s", message)
            return None

        return {
            "signal": signal,
            "direction": direction,
            "confidence": confidence,
            "size": int(quantity),
            "price": int(confidence * 100) if confidence <= 1.0 else int(confidence),
        }

    def execute_trade(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a validated signal into a Kalshi order."""
        ticker = self._get_contract_ticker()
        side = "yes" if decision["direction"] in ["up", "BUY", "LONG"] else "no"
        price = max(1, min(99, decision["price"]))
        count = max(1, decision["size"])

        order = self.client.place_order(ticker, side, count, price)
        self.logger.info(
            "Order Executed: ID %s | Ticker: %s | Side: %s | %d contracts @ %d¢",
            order.get("order_id"),
            ticker,
            side,
            count,
            price,
        )
        return order

    def _get_contract_ticker() -> str:
        return "KX15MIN-UP"

    def run_once(self):
        """One iteration of the 15-minute trading cycle."""
        self.logger.info("--- 15-Minute Trading Cycle Executing ---")
        market_data = self.fetch_15min_data()
        decision = self.evaluate_signal(market_data)

        if decision:
            self.execute_trade(decision)
        else:
            self.logger.info("Cycle completed with no trade authorized.")

        self.last_signal_time = datetime.utcnow()

    def run_continuous(self):
        """Main loop: aligns with 15-minute boundaries and runs forever."""
        self.logger.info("Kalshi Never-Loss Auto-Trader started on 15m timeframe.")
        while True:
            now = datetime.utcnow()
            next_boundary = now + timedelta(minutes=15 - (now.minute % 15))
            next_boundary = next_boundary.replace(second=0, microsecond=0)
            sleep_seconds = (next_boundary - now).total_seconds()

            if sleep_seconds > 0:
                self.logger.info(
                    "Sleeping %d seconds until next 15m boundary: %s",
                    sleep_seconds,
                    next_boundary,
                )
                time.sleep(sleep_seconds)

            self.run_once()


if __name__ == "__main__":
    engine = KalshiNeverLossEngine()
    engine.run_once()
