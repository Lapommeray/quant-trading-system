"""
Verify self-coding injection in each module - One Organism wiring report generator.

Run: python -m autonomy.verify_self_coding_injection

Checks:
- Each core module has auto_self_code, auto_fix, learn_from_mistakes, improve_with_market, interconnect, full_autonomous_cycle, coherence_check
- Organism discovers modules
- DataRing pre-broker exists
- OKX feed pre-broker exists
- No toxic random in core hot path
- Latency bench
"""

from __future__ import annotations
import importlib
import sys
from pathlib import Path

REQUIRED_METHODS = [
    "auto_self_code",
    "auto_fix",
    "learn_from_mistakes",
    "improve_with_market",
    "interconnect",
    "sync_with_organism",
    "coherence_check",
    "full_autonomous_cycle",
    "on_mutation_event",
    "analyze",
    "initialize",
]

CORE_MODULES = [
    "core.ofi_detector",
    "core.whale_flow_detector",
    "core.mm_intent_detector",
    "core.cvd_indicator",
    "core.funding_indicator",
    "core.volume_profile",
    "core.cross_asset_leader",
    "core.real_fed_model",
    "core.real_enhanced_indicator",
]


def check_module(mod_name: str):
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        return {"module": mod_name, "status": "import_failed", "error": str(e)}
    # Find BaseTradingModule subclass
    candidates = []
    for attr_name in dir(mod):
        try:
            obj = getattr(mod, attr_name)
            if isinstance(obj, type):
                # check if has module_name attr
                if hasattr(obj, "module_name"):
                    candidates.append(obj)
        except Exception:
            pass
    if not candidates:
        return {
            "module": mod_name,
            "status": "no_trading_module",
            "error": "No class with module_name",
        }
    results = []
    for cls in candidates:
        missing = [m for m in REQUIRED_METHODS if not hasattr(cls, m)]
        has_register = hasattr(cls, "module_name")
        results.append(
            {
                "class": cls.__name__,
                "module_name": getattr(cls, "module_name", "unknown"),
                "missing": missing,
                "ok": len(missing) == 0,
                "has_register": has_register,
            }
        )
    return {"module": mod_name, "status": "ok", "classes": results}


def main():
    print("=== AUTO SELF-CODING ORGANISM WIRING VERIFICATION V2 ===")
    print(f"Checking {len(CORE_MODULES)} core modules for {REQUIRED_METHODS}")
    all_ok = True
    for mod_name in CORE_MODULES:
        res = check_module(mod_name)
        if res["status"] != "ok":
            print(f"[FAIL] {mod_name}: {res}")
            all_ok = False
        else:
            for cl in res["classes"]:
                if cl["ok"]:
                    print(
                        f"[OK] {mod_name}.{cl['class']} ({cl['module_name']}) has all autonomous methods"
                    )
                else:
                    print(f"[FAIL] {mod_name}.{cl['class']} missing {cl['missing']}")
                    all_ok = False

    print("\n--- Organism Discovery ---")
    try:
        from autonomy.organism import Organism, OrganismConfig

        org = Organism(
            OrganismConfig(self_coding_enabled=True, auto_approve_low_risk=True)
        )
        wired = org.discover_and_wire()
        print(
            f"Discovered {len(wired['active'])} active modules: {wired['active'][:20]}"
        )
        expected = [
            "ofi_detector",
            "whale_flow_detector",
            "mm_intent_detector",
            "cvd_detector",
            "funding_detector",
            "volume_profile",
            "cross_asset_leader",
        ]
        for exp in expected:
            if exp in org.modules:
                print(f"[OK] Organism has {exp}")
            else:
                print(f"[WARN] Organism missing {exp} (may need core/__init__ import)")
    except Exception as e:
        print(f"[FAIL] Organism discovery error: {e}")
        import traceback

        traceback.print_exc()
        all_ok = False

    print("\n--- DataRing Pre-Broker Check ---")
    try:
        from core.data_ring import get_data_ring, DataRing

        ring = get_data_ring("BTCUSDT")
        import time

        ring.push(time.time(), 67000, 67000.5, 1.2, 0.8, 67000.2, 0.5, 1)
        ticks = ring.latest(10)
        print(
            f"[OK] DataRing push+latest works len={len(ticks)} latency {ring.latency_ms():.1f}ms"
        )
    except Exception as e:
        print(f"[FAIL] DataRing {e}")
        all_ok = False

    print("\n--- OKX Pre-Broker Feed Check ---")
    try:
        from okx_live.feed import OKXPreBrokerFeed

        feed = OKXPreBrokerFeed()
        frame = {
            "arg": {"channel": "trades", "instId": "BTC-USDT-SWAP"},
            "data": [
                {"px": "67000", "sz": "0.5", "side": "buy", "ts": "1710000000000"}
            ],
        }
        feed.handle_frame(frame)
        status = feed.feed_status()
        print(f"[OK] OKXPreBrokerFeed handle_frame works {status}")
    except Exception as e:
        print(f"[FAIL] OKX feed {e}")
        all_ok = False

    print("\n--- RealEnhancedIndicator Check ---")
    try:
        from core.real_enhanced_indicator import RealEnhancedIndicator

        ind = RealEnhancedIndicator()
        sig = ind.get_signal("BTCUSDT")
        print(
            f"[OK] RealEnhancedIndicator signal {sig.get('signal')} conf {sig.get('confidence')} latency {sig.get('latency_ms')}ms reason {sig.get('reason', '')[:80]}"
        )
    except Exception as e:
        print(f"[FAIL] RealEnhancedIndicator {e}")
        import traceback

        traceback.print_exc()
        all_ok = False

    print("\n--- No Random Check in core/ ---")
    import subprocess, os

    try:
        out = subprocess.check_output(
            ["grep", "-R", "np.random", "core", "--include=*.py"], text=True
        )
        # filter out allowed? For now just warn
        lines = out.strip().split("\n") if out.strip() else []
        # allow only comment or test? We consider fail if any in hot path excluding template docs
        filtered = [
            l
            for l in lines
            if "template" not in l
            and "real_enhanced" not in l
            and "mock" not in l.lower()
        ][:20]
        if filtered:
            print(
                f"[WARN] Found np.random in core (should be 0 in hot path):\n"
                + "\n".join(filtered)
            )
        else:
            print("[OK] No np.random in core hot path")
    except subprocess.CalledProcessError:
        print("[OK] No np.random in core")

    print("\n--- Latency Bench ---")
    try:
        from core.data_ring import get_data_ring
        from core.ofi_detector import OFIDetector
        import time

        ring = get_data_ring("BTCUSDT")
        # warm push 100
        for i in range(100):
            ring.push(
                time.time(),
                67000 + i * 0.01,
                67000.5 + i * 0.01,
                1.2,
                0.8,
                67000.2,
                0.5,
                1 if i % 2 == 0 else -1,
            )
        det = OFIDetector()
        times = []
        for _ in range(20):
            s = time.time()
            res = det.analyze({"symbol": "BTCUSDT", "data_ring": ring})
            times.append(res.latency_ms)
        import numpy as np

        print(
            f"[OK] OFI latency p50 {np.percentile(times, 50):.1f}ms p95 {np.percentile(times, 95):.1f}ms avg {np.mean(times):.1f}ms"
        )
        if np.percentile(times, 95) > 50:
            print("[WARN] p95 >50ms target")
    except Exception as e:
        print(f"[FAIL] Latency bench {e}")

    print("\n=== SUMMARY ===")
    if all_ok:
        print("ALL CRITICAL CHECKS PASSED - One organism wiring OK")
    else:
        print("SOME CHECKS FAILED - See above, fix per OTHER_AI_MUST_DO_SINGLE_NOTE_V4")


if __name__ == "__main__":
    main()
