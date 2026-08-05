"""
Funding Rate Indicator - Auto Self-Coding
Detects crowded longs/shorts via perp funding zscore.
Edge: Funding +0.09% per 8h at top before -8% washout (BTC 2024-03-13).

Move WITH whale after crowd, fade crowd like market maker.

Asset: BTCUSD, ETHUSD (not SPY)
Reads cached REST (30s) - not hot path, preconscious check.
Publishes ALPHA_SIGNAL + FUNDING_CROWDED event.
"""
from core.base_module import BaseTradingModule, register_module, ModuleResult
from core.event_bus import get_event_bus, EventPriority
import time
import math

@register_module
class FundingRateDetector(BaseTradingModule):
    module_name = "funding_detector"
    category = "derivatives"
    version = "2.0.0"
    dependencies = []

    def __init__(self, config=None, event_bus=None):
        super().__init__(config, event_bus)
        self.config.setdefault("adaptive", {
            "confidence_floor": 0.70,
            "weight_multiplier": 1.0,
            "lookback": 72,  # hours for zscore
            "cooldown_seconds": 30.0,
            "zscore_threshold": 2.5,
            "funding_extreme_bps": 9  # 0.09% per 8h = extreme
        })
        self.funding_history = []
        self.last_fetch_ts = 0
        self.cached_funding = 0.0
        self.cached_z = 0.0
        self.last_regime = "unknown"

    def initialize(self):
        bus = get_event_bus()
        bus.subscribe("REGIME_CHANGE", self.on_event)
        return True

    def _fetch_funding(self, symbol="BTCUSDT"):
        now = time.time()
        if now - self.last_fetch_ts < 30:  # cache 30s
            return self.cached_funding, self.cached_z

        funding = 0.0
        try:
            import os, requests
            # Binance funding rate API - public, no key
            # GET https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=100
            # For safety, try with timeout 2s
            try:
                r = requests.get("https://fapi.binance.com/fapi/v1/fundingRate", params={"symbol": symbol, "limit": 100}, timeout=2)
                if r.status_code == 200:
                    data = r.json()
                    if data:
                        funding = float(data[-1].get("fundingRate", 0)) * 100  # in %
                        # keep history
                        rates = [float(x.get("fundingRate", 0))*100 for x in data]
                        self.funding_history = rates[-self.config["adaptive"]["lookback"]:]
            except:
                # fallback mock 0
                funding = 0.0
        except:
            funding = 0.0

        # Compute zscore
        if len(self.funding_history) > 10:
            import numpy as np
            mean = float(np.mean(self.funding_history))
            std = float(np.std(self.funding_history))
            z = (funding - mean) / (std if std>1e-6 else 1)
        else:
            z = 0.0

        self.cached_funding = funding
        self.cached_z = z
        self.last_fetch_ts = now
        return funding, z

    def analyze(self, market_data):
        start = time.time()
        symbol = market_data.get("symbol", "BTCUSDT")
        if "SPY" in symbol or "ES" in symbol:
            return ModuleResult(self.module_name, "NEUTRAL", 0.0, {"reason": "no_funding_spy"})

        funding, z = self._fetch_funding(symbol)

        signal = "NEUTRAL"
        conf = 0.0

        z_thr = self.config["adaptive"]["zscore_threshold"]
        # Crowded long: funding high positive -> fade long = SELL
        if z > z_thr and funding > 0.03:  # >0.03% per 8h
            signal = "SELL"
            conf = min(1.0, abs(z)/3.0 * 0.9)
        # Crowded short: funding negative large -> fade short = BUY (short squeeze)
        elif z < -z_thr and funding < -0.02:
            signal = "BUY"
            conf = min(1.0, abs(z)/3.0 * 0.9)

        if conf < self.config["adaptive"]["confidence_floor"]:
            signal = "NEUTRAL"
            conf = 0.0

        latency = (time.time() - start)*1000
        self.record_success(latency)

        payload = {
            "funding_pct": float(funding),
            "zscore": float(z),
            "symbol": symbol,
            "crowded": "long" if z>z_thr else "short" if z<-z_thr else "neutral"
        }

        # Publish FUNDING_CROWDED for other modules
        if signal != "NEUTRAL":
            try:
                get_event_bus().publish("FUNDING_CROWDED", payload, source=self.module_name, priority=EventPriority.OPERATIONAL)
            except:
                pass

        result = ModuleResult(
            module_name=self.module_name,
            signal=signal,
            confidence=conf * self.config["adaptive"]["weight_multiplier"],
            features=payload,
            latency_ms=latency
        )

        try:
            get_event_bus().publish("ALPHA_SIGNAL", result.to_dict(), source=self.module_name, priority=EventPriority.OPERATIONAL)
        except:
            pass

        self._last_result = result
        return result

    def on_event(self, event_type, payload=None):
        if hasattr(event_type, 'event_type'):
            ev = event_type
            event_type = ev.event_type
            payload = ev.payload
        if event_type == "REGIME_CHANGE":
            self.last_regime = payload.get("label", "unknown") if isinstance(payload, dict) else "unknown"

    def learn_from_outcome(self, outcome):
        if outcome.get("module_name") != self.module_name:
            return {"learned": False}
        pnl = outcome.get("pnl", 0)
        if pnl < 0:
            self.config["adaptive"]["zscore_threshold"] = min(3.5, self.config["adaptive"]["zscore_threshold"] + 0.1)
            self.config["adaptive"]["confidence_floor"] = min(0.85, self.config["adaptive"]["confidence_floor"] + 0.02)
            return {"learned": True}
        else:
            self.config["adaptive"]["confidence_floor"] = max(0.60, self.config["adaptive"]["confidence_floor"] - 0.005)
            return {"learned": True}
        return {"learned": False}

    def diagnose(self, context=None):
        stats = (context or {}).get("stats", {})
        if stats.get("win_rate", 0.5) < 0.48:
            return {"issue": "low_win_rate", "params": {"zscore_threshold": 2.8, "confidence_floor": 0.75}}
        return {"issue": "none"}
