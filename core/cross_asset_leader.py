"""
Cross Asset Leader - Auto Self-Coding
BTC leads ETH, ES leads SPY, DXY inverse leads risk.

Edge: BTC OFI 200-400ms before ETH moves. ES futures 5-15ms before SPY.
Move with leader market maker, not retail lagging.

Publishes CROSS_ASSET_SIGNAL.

Asset: BTCUSD, ETHUSD, SPY, ES, DXY
"""

from core.base_module import BaseTradingModule, register_module, ModuleResult
from core.event_bus import get_event_bus, EventPriority
from core.data_ring import get_data_ring
import time
import numpy as np


@register_module
class CrossAssetLeaderDetector(BaseTradingModule):
    module_name = "cross_asset_leader"
    category = "cross_asset"
    version = "2.0.0"
    dependencies = ["ofi_detector"]

    def __init__(self, config=None, event_bus=None):
        super().__init__(config, event_bus)
        self.config.setdefault(
            "adaptive",
            {
                "confidence_floor": 0.62,
                "weight_multiplier": 1.0,
                "lookback": 50,
                "cooldown_seconds": 1.0,
                "btc_eth_lag_ms": 300,
                "es_spy_lead_ms": 10,
            },
        )
        self.last_btc_signal = None
        self.last_es_signal = None
        self.last_dxy_move = 0.0
        self.last_signal_ts = 0

    def initialize(self):
        bus = get_event_bus()
        bus.subscribe("ALPHA_SIGNAL", self.on_event)
        bus.subscribe("REGIME_CHANGE", self.on_event)
        return True

    def on_event(self, event_type, payload=None):
        if hasattr(event_type, "event_type"):
            ev = event_type
            event_type = ev.event_type
            payload = ev.payload
        if event_type == "ALPHA_SIGNAL" and isinstance(payload, dict):
            mod = payload.get("module_name")
            sym = payload.get("features", {}).get("symbol") or payload.get("symbol")
            sig = payload.get("signal")
            if "btc" in str(sym).lower() or "BTC" in str(sym):
                self.last_btc_signal = {
                    "signal": sig,
                    "conf": payload.get("confidence", 0),
                    "ts": time.time(),
                }
            if "ES" in str(sym) or "es" in str(sym).lower():
                self.last_es_signal = {
                    "signal": sig,
                    "conf": payload.get("confidence", 0),
                    "ts": time.time(),
                }

    def analyze(self, market_data):
        start = time.time()
        symbol = market_data.get("symbol", "ETHUSDT")

        signal = "NEUTRAL"
        conf = 0.0
        features = {}

        now = time.time()

        # ETH follows BTC
        if "ETH" in symbol:
            if (
                self.last_btc_signal
                and now - self.last_btc_signal["ts"]
                < self.config["adaptive"]["btc_eth_lag_ms"] / 1000.0 * 2
            ):
                btc_sig = self.last_btc_signal["signal"]
                btc_conf = self.last_btc_signal["conf"]
                if btc_sig in ("BUY", "SELL") and btc_conf > 0.65:
                    signal = btc_sig
                    conf = btc_conf * 0.85  # ETH lag trade slightly lower conf
                    features["leader"] = "BTC"
                    features["leader_conf"] = btc_conf

        # SPY follows ES
        if "SPY" in symbol:
            if self.last_es_signal and now - self.last_es_signal["ts"] < 0.1:  # 100ms
                es_sig = self.last_es_signal["signal"]
                es_conf = self.last_es_signal["conf"]
                if es_sig in ("BUY", "SELL") and es_conf > 0.60:
                    signal = es_sig
                    conf = es_conf * 0.9
                    features["leader"] = "ES"
                    features["leader_conf"] = es_conf

        # DXY inverse: if DXY spikes +0.3% in 5 min, BTC/SPY down
        # TODO fetch DXY via Polygon or OANDA

        if (
            time.time() - self.last_signal_ts
            < self.config["adaptive"]["cooldown_seconds"]
        ):
            signal = "NEUTRAL"
            conf = 0.0

        if conf < self.config["adaptive"]["confidence_floor"]:
            signal = "NEUTRAL"
            conf = 0.0

        if signal != "NEUTRAL":
            self.last_signal_ts = time.time()

        latency = (time.time() - start) * 1000
        self.record_success(latency)

        result = ModuleResult(
            module_name=self.module_name,
            signal=signal,
            confidence=conf * self.config["adaptive"]["weight_multiplier"],
            features=features,
            latency_ms=latency,
        )

        try:
            get_event_bus().publish(
                "ALPHA_SIGNAL",
                result.to_dict(),
                source=self.module_name,
                priority=EventPriority.OPERATIONAL,
            )
            get_event_bus().publish(
                "CROSS_ASSET_SIGNAL",
                features,
                source=self.module_name,
                priority=EventPriority.OPERATIONAL,
            )
        except:
            pass

        self._last_result = result
        return result

    def learn_from_outcome(self, outcome):
        if outcome.get("module_name") != self.module_name:
            return {"learned": False}
        pnl = outcome.get("pnl", 0)
        if pnl < 0:
            self.config["adaptive"]["confidence_floor"] = min(
                0.85, self.config["adaptive"]["confidence_floor"] + 0.02
            )
        else:
            self.config["adaptive"]["confidence_floor"] = max(
                0.55, self.config["adaptive"]["confidence_floor"] - 0.005
            )
        return {"learned": True}

    def diagnose(self, context=None):
        stats = (context or {}).get("stats", {})
        if stats.get("win_rate", 0.5) < 0.48:
            return {"issue": "low_win_rate_cross", "params": {"confidence_floor": 0.70}}
        return {"issue": "none"}
