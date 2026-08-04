import pytest
import pandas as pd
import numpy as np
from core.event_bus import EventBus, get_event_bus, reset_event_bus
from core.organism import Organism, OrganismConfig
from core.base_module import BaseTradingModule, ModuleResult, register_module, get_registered_modules, clear_registry

def make_dummy_history():
    dates = pd.date_range("2024-01-01", periods=100, freq="1min")
    df = pd.DataFrame({
        "Open": 50000 + np.random.randn(100),
        "High": 50100 + np.random.randn(100),
        "Low": 49900 + np.random.randn(100),
        "Close": 50000 + np.random.randn(100),
        "Volume": 100 + np.random.randn(100),
    }, index=dates)
    return {
        "1m": df,
        "5m": df,
        "10m": df,
        "15m": df,
        "20m": df,
        "25m": df,
    }

def test_event_bus_pubsub():
    bus = EventBus()
    bus.start()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe("TEST_EVENT", handler)
    bus.publish("TEST_EVENT", {"x": 1}, source="test")
    # sync dispatch already called
    assert len(received) == 1
    assert received[0].payload["x"] == 1
    bus.stop()

def test_organism_discovery():
    reset_event_bus()
    org = Organism(config=OrganismConfig(auto_discover=False))
    result = org.discover_and_wire()
    assert isinstance(result, dict)
    assert "active" in result
    # should not crash even with 0 modules
    org.stop()

def test_organism_consensus_with_dummy_module():
    reset_event_bus()
    clear_registry()

    @register_module
    class DummyAlpha(BaseTradingModule):
        module_name = "dummy_alpha"
        category = "alpha"

        def initialize(self):
            return True

        def generate_signal(self, symbol, history_data):
            return ModuleResult(module_name=self.module_name, signal="BUY", confidence=0.9)

    bus = EventBus()
    bus.start()
    org = Organism(config=OrganismConfig(auto_discover=True, enable_self_improvement=False), event_bus=bus)
    wired = org.discover_and_wire()
    assert "dummy_alpha" in wired["active"] or "dummy_alpha" in org.modules

    hist = make_dummy_history()
    consensus = org.generate_consensus_signal("BTC/USDT", hist)
    assert consensus["symbol"] == "BTC/USDT"
    org.stop()
    bus.stop()
    clear_registry()

def test_organism_self_improvement():
    reset_event_bus()
    clear_registry()

    @register_module
    class Improvable(BaseTradingModule):
        module_name = "improvable"
        category = "test"

        def initialize(self):
            return True

        def generate_signal(self, symbol, history_data):
            return ModuleResult(module_name=self.module_name, signal="BUY", confidence=0.6)

        def self_improve(self, history):
            return {"module": self.module_name, "improved": True}

    bus = EventBus()
    bus.start()
    org = Organism(config=OrganismConfig(enable_self_improvement=False), event_bus=bus)
    org.discover_and_wire()
    org.feedback("improvable", 0.9)
    cycle = org.run_self_improvement_cycle()
    assert "weights" in cycle
    assert "improvable" in cycle["weights"]
    org.stop()
    bus.stop()
    clear_registry()

def test_event_driven_executor():
    from execution.okx_engine import OKXEngine
    from execution.event_driven_executor import EventDrivenExecutor, ExecutorConfig
    reset_event_bus()
    bus = get_event_bus()
    engine = OKXEngine(paper_mode=True, event_bus=bus)
    engine.connect()
    executor = EventDrivenExecutor(okx_engine=engine, event_bus=bus, config=ExecutorConfig(min_confidence=0.5))
    executor.start()

    # Publish a signal
    bus.publish("SIGNAL_GENERATED", {"symbol": "BTC/USDT", "final_signal": "BUY", "confidence": 0.9, "weighted_confidence": 0.8}, source="test")

    # Allow thread processing (sync dispatch already)
    import time
    time.sleep(0.1)
    stats = executor.get_stats()
    assert stats["signals_received"] >= 1
    executor.stop()
