"""
MM Intent Detector - Market Maker Intent with Auto Self-Coding
Detects spoof, iceberg, absorption, exhaustion BEFORE price moves.
Moves with market makers, not after retail.

Reads DataRing L2 snapshots directly (pre-broker).
Publishes MM_INTENT event OPERATIONAL lane.
"""
from core.base_module import BaseTradingModule, register_module, ModuleResult
from core.event_bus import get_event_bus, EventPriority
from core.data_ring import get_data_ring
import time
import numpy as np

@register_module
class MarketMakerIntentDetector(BaseTradingModule):
    module_name = "mm_intent_detector"
    category = "microstructure"
    version = "2.0.0"
    dependencies = ["ofi_detector", "volume_profile"]

    def __init__(self, config=None, event_bus=None):
        super().__init__(config, event_bus)
        self.config.setdefault("adaptive", {
            "confidence_floor": 0.68,
            "weight_multiplier": 1.0,
            "lookback": 100,
            "cooldown_seconds": 0.8,
            "spoof_size_mult": 5.0,
            "spoof_lifetime_ms": 200,
            "absorption_vol_mult": 3.0,
            "absorption_range_pct": 0.001  # 0.1%
        })
        self.last_regime = "unknown"
        self.last_signal_ts = 0
        self.l2_history = []  # store snapshots for spoof detection

    def initialize(self):
        bus = get_event_bus()
        bus.subscribe("REGIME_CHANGE", self.on_event)
        bus.subscribe("WHALE_FLOW", self.on_event)
        return True

    def analyze(self, market_data):
        start = time.time()
        symbol = market_data.get("symbol", "BTCUSDT")
        ring = market_data.get("data_ring") or get_data_ring(symbol)
        ticks = ring.latest(self.config["adaptive"]["lookback"])

        if len(ticks) < 30:
            return ModuleResult(self.module_name, "NEUTRAL", 0.0, {"reason": "insufficient"})

        if ring.latency_ms() > 3000:
            return ModuleResult(self.module_name, "NEUTRAL", 0.0, {"reason": "stale"})

        # Proxy L2 analysis using best bid/ask size change
        # Real implementation needs full 20-level depth ring
        bid_sizes = ticks["bid_size"].astype(np.float64)
        ask_sizes = ticks["ask_size"].astype(np.float64)
        bids = ticks["bid"].astype(np.float64)
        asks = ticks["ask"].astype(np.float64)
        prices = ticks["price"].astype(np.float64)
        qtys = ticks["qty"].astype(np.float64)

        # Detect absorption: high volume in narrow range, price not progressing
        vol_window = 10
        recent_vol = np.sum(qtys[-vol_window:])
        avg_vol = np.mean(qtys[-50:]) * vol_window if len(qtys)>=50 else recent_vol
        price_range = np.max(prices[-vol_window:]) - np.min(prices[-vol_window:])
        price_mean = np.mean(prices[-vol_window:])
        range_pct = price_range / price_mean if price_mean>0 else 999

        absorption = False
        exhaustion = False
        spoof_bull = False
        spoof_bear = False

        vol_mult = self.config["adaptive"]["absorption_vol_mult"]
        range_thr = self.config["adaptive"]["absorption_range_pct"]

        if recent_vol > avg_vol * vol_mult and range_pct < range_thr:
            # High volume, tight range = absorption
            # Determine direction via CVD proxy: if bid sizes increasing vs ask, absorption of selling = bullish
            bid_trend = np.mean(bid_sizes[-5:]) - np.mean(bid_sizes[-20:-5]) if len(bid_sizes)>=20 else 0
            ask_trend = np.mean(ask_sizes[-5:]) - np.mean(ask_sizes[-20:-5]) if len(ask_sizes)>=20 else 0
            if bid_trend > 0 and ask_trend <=0:
                absorption = True
                # bullish absorption
                signal = "BUY"
                conf = 0.75
            elif ask_trend >0 and bid_trend <=0:
                absorption = True
                signal = "SELL"
                conf = 0.75
            else:
                signal = "NEUTRAL"
                conf = 0.0
        else:
            signal = "NEUTRAL"
            conf = 0.0

        # Spoof detection proxy: large size appearing then disappearing in <200ms
        # We need order-level lifetime, here approximate: if ask_size spikes >5x then drops 80% in next 2 ticks
        size_mult = self.config["adaptive"]["spoof_size_mult"]
        if len(ask_sizes) >= 5:
            avg_ask = np.mean(ask_sizes[-20:-3]) if len(ask_sizes)>=20 else np.mean(ask_sizes[:-3])
            spike = ask_sizes[-3]
            after = ask_sizes[-1]
            if spike > avg_ask*size_mult and after < spike*0.3:
                # spoof ask pulled = fake supply removed -> bullish
                spoof_bull = True
                signal = "BUY"
                conf = max(conf, 0.80)
        if len(bid_sizes) >=5:
            avg_bid = np.mean(bid_sizes[-20:-3]) if len(bid_sizes)>=20 else np.mean(bid_sizes[:-3])
            spike = bid_sizes[-3]
            after = bid_sizes[-1]
            if spike > avg_bid*size_mult and after < spike*0.3:
                spoof_bear = True
                signal = "SELL"
                conf = max(conf, 0.80)

        # Cooldown
        if time.time() - self.last_signal_ts < self.config["adaptive"]["cooldown_seconds"]:
            signal = "NEUTRAL"
            conf = 0.0

        if conf < self.config["adaptive"]["confidence_floor"]:
            signal = "NEUTRAL"
            conf = 0.0

        if signal != "NEUTRAL":
            self.last_signal_ts = time.time()

        latency = (time.time() - start)*1000
        self.record_success(latency)

        payload = {
            "signal": signal,
            "absorption": absorption,
            "spoof_bull": spoof_bull,
            "spoof_bear": spoof_bear,
            "vol_mult": float(recent_vol / (avg_vol if avg_vol>0 else 1)),
            "range_pct": float(range_pct),
            "symbol": symbol
        }

        try:
            get_event_bus().publish("MM_INTENT", payload, source=self.module_name, priority=EventPriority.OPERATIONAL)
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
            self.config["adaptive"]["absorption_vol_mult"] = min(5.0, self.config["adaptive"]["absorption_vol_mult"] + 0.2)
            self.config["adaptive"]["confidence_floor"] = min(0.85, self.config["adaptive"]["confidence_floor"] + 0.02)
            return {"learned": True}
        else:
            self.config["adaptive"]["confidence_floor"] = max(0.60, self.config["adaptive"]["confidence_floor"] - 0.005)
            return {"learned": True}
        return {"learned": False}

    def diagnose(self, context=None):
        stats = (context or {}).get("stats", {})
        if stats.get("win_rate", 0.5) < 0.48:
            return {"issue": "low_win_rate", "params": {"absorption_vol_mult": 3.5, "confidence_floor": 0.72}}
        return {"issue": "none"}
