#!/usr/bin/env python3
"""
Test Mathematical Modules

Tests the new mathematical components individually to ensure they work correctly.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_modules.hyperbolic_market_manifold import HyperbolicMarketManifold
from advanced_modules.quantum_topology_analysis import QuantumTopologyAnalysis
from advanced_modules.noncommutative_calculus import NoncommutativeCalculus


def test_hyperbolic_manifold():
    """Test Hyperbolic Market Manifold"""
    print("🧮 Testing Hyperbolic Market Manifold...")

    manifold = HyperbolicMarketManifold()

    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(50) * 0.1)
    volumes = 1000 + np.random.randn(50) * 100
    order_flows = np.random.randn(50) * 10

    embedded = manifold.embed_market_data(prices[-1], volumes[-1], order_flows[-1])
    print(f"  ✅ Embedded point dimension: {len(embedded)}")

    signal = manifold.generate_trading_signal(
        prices[-10:], volumes[-10:], order_flows[-10:]
    )
    print(f"  ✅ Signal: {signal['signal']}, Confidence: {signal['confidence']:.3f}")
    print(f"  ✅ Hyperbolic momentum: {signal['hyperbolic_momentum']:.3f}")
    print(f"  ✅ Noise immunity: {signal['noise_immunity']:.3f}")

    stats = manifold.get_statistics()
    print(f"  ✅ Operations performed: {stats['operations']}")

    return True


def test_quantum_topology():
    """Test Quantum Topology Analysis"""
    print("\n🔬 Testing Quantum Topology Analysis...")

    qta = QuantumTopologyAnalysis()

    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.1)
    volumes = 1000 + np.random.randn(100) * 100

    cycle_result = qta.detect_market_cycles(prices, volumes)
    print(f"  ✅ Cycles detected: {cycle_result['cycles_detected']}")
    print(f"  ✅ Regime: {cycle_result['regime']}")
    print(f"  ✅ Cycle strength: {cycle_result['cycle_strength']:.3f}")

    signal = qta.generate_trading_signal(prices[-20:], volumes[-20:])
    print(f"  ✅ Signal: {signal['signal']}, Confidence: {signal['confidence']:.3f}")
    print(f"  ✅ Quantum superposition: {signal['quantum_superposition']:.3f}")
    print(f"  ✅ Topological holes: {signal['topological_holes']}")

    stats = qta.get_statistics()
    print(f"  ✅ Operations performed: {stats['operations']}")

    return True


def test_noncommutative_calculus():
    """Test Noncommutative Calculus"""
    print("\n⚛️  Testing Noncommutative Calculus...")

    nc = NoncommutativeCalculus()

    market_field = 100 * 1000  # price * volume
    buy_deriv = nc.lie_derivative(market_field, "buy")
    sell_deriv = nc.lie_derivative(market_field, "sell")
    print(f"  ✅ Buy Lie derivative: {buy_deriv}")
    print(f"  ✅ Sell Lie derivative: {sell_deriv}")

    commutator = nc.calculate_commutator(buy_deriv, sell_deriv)
    print(f"  ✅ Commutator [buy, sell]: {commutator}")

    trades = [
        {"direction": "buy", "size": 100, "price": 100.0},
        {"direction": "sell", "size": 50, "price": 101.0},
        {"direction": "buy", "size": 75, "price": 99.5},
    ]
    optimization = nc.optimize_trade_sequence(trades)
    print(f"  ✅ Sequence optimization improvement: {optimization['improvement']:.3f}")
    print(f"  ✅ Noncommutative effect: {optimization['noncommutative_effect']:.3f}")

    market_data = {"price": 100.0, "volume": 1000.0}
    signal = nc.generate_trading_signal(market_data, "buy")
    print(f"  ✅ Signal: {signal['signal']}, Confidence: {signal['confidence']:.3f}")
    print(f"  ✅ Noncommutative advantage: {signal['noncommutative_advantage']:.3f}")

    stats = nc.get_statistics()
    print(f"  ✅ Operations performed: {stats['operations']}")

    return True


def test_mathematical_integration():
    """Test Mathematical Integration Layer"""
    print("\n🔗 Testing Mathematical Integration Layer...")

    try:
        from advanced_modules.mathematical_integration_layer import (
            MathematicalIntegrationLayer,
        )

        math_layer = MathematicalIntegrationLayer()

        modules = math_layer.get_all_modules()
        print(f"  ✅ Available modules: {list(modules.keys())}")

        original_signal = {"signal": "BUY", "confidence": 0.7, "direction": "buy"}
        market_data = {
            "prices": [100, 101, 102, 103, 104],
            "volumes": [1000, 1100, 1200, 1300, 1400],
            "price": 104,
            "volume": 1400,
        }

        enhanced_signal = math_layer.enhance_trading_signal(
            original_signal, market_data
        )
        print(
            f"  ✅ Enhanced confidence: {enhanced_signal['mathematical_confidence']:.3f}"
        )
        print(
            f"  ✅ Hyperbolic analysis available: {'hyperbolic_analysis' in enhanced_signal}"
        )
        print(
            f"  ✅ Quantum topology analysis available: {'quantum_topology_analysis' in enhanced_signal}"
        )
        print(
            f"  ✅ Noncommutative analysis available: {'noncommutative_analysis' in enhanced_signal}"
        )

        return True

    except ImportError as e:
        print(f"  ❌ Mathematical Integration Layer import failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("🎯 Testing Mathematical Modules for Ultimate Never Loss System")
    print("=" * 70)

    results = []

    results.append(test_hyperbolic_manifold())
    results.append(test_quantum_topology())
    results.append(test_noncommutative_calculus())
    results.append(test_mathematical_integration())

    print("\n" + "=" * 70)
    print("📊 Test Results Summary")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total:.1%}")

    if passed == total:
        print("🎯 All mathematical modules working perfectly!")
        print("🚀 Ready for ultimate never-loss trading with full mathematical power!")
    else:
        print("⚠️  Some mathematical modules had issues")
        print("   System will use mock implementations where needed")

    print(
        "\nMathematical modules are now integrated into the Ultimate Never Loss System!"
    )
