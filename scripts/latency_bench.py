"""
Latency bench - pre-broker DataRing -> OFI -> CVD -> Consensus <50ms target
BTCUSD ETHUSD SPY
"""

import time
import numpy as np
from core.data_ring import get_data_ring


def bench_symbol(symbol="BTCUSDT", n=100):
    ring = get_data_ring(symbol)
    # warmup pushes
    for i in range(200):
        ring.push(
            time.time(),
            67000 + i * 0.01,
            67000.5 + i * 0.01,
            1.2 + (i % 5) * 0.1,
            0.8,
            67000.2 + i * 0.01,
            0.1 + (i % 3) * 0.2,
            1 if i % 2 == 0 else -1,
        )

    from core.ofi_detector import OFIDetector
    from core.cvd_indicator import CVDDetector
    from core.mm_intent_detector import MarketMakerIntentDetector
    from core.volume_profile import VolumeProfileDetector

    ofi = OFIDetector()
    cvd = CVDDetector()
    mm = MarketMakerIntentDetector()
    vp = VolumeProfileDetector()

    latencies = {"ofi": [], "cvd": [], "mm": [], "vp": [], "e2e": []}

    for _ in range(n):
        # simulate new tick
        ring.push(
            time.time(),
            67000 + np.random.randn() * 0.5,
            67000.5 + np.random.randn() * 0.5,
            1.2,
            0.8,
            67000.2,
            0.3,
            1,
        )
        md = {"symbol": symbol, "data_ring": ring}

        s = time.time()
        r1 = ofi.analyze(md)
        r2 = cvd.analyze(md)
        r3 = mm.analyze(md)
        r4 = vp.analyze(md)
        e2e = (time.time() - s) * 1000

        latencies["ofi"].append(r1.latency_ms)
        latencies["cvd"].append(r2.latency_ms)
        latencies["mm"].append(r3.latency_ms)
        latencies["vp"].append(r4.latency_ms)
        latencies["e2e"].append(e2e)

    for k, v in latencies.items():
        arr = np.array(v)
        print(
            f"{symbol} {k}: p50 {np.percentile(arr,50):.2f}ms p95 {np.percentile(arr,95):.2f}ms avg {np.mean(arr):.2f}ms max {np.max(arr):.2f}ms"
        )


if __name__ == "__main__":
    for sym in ["BTCUSDT", "ETHUSDT", "SPY"]:
        print(f"\n=== Bench {sym} ===")
        bench_symbol(sym, n=50)
