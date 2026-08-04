# Organism - Autonomous Module Communication & Self-Improvement

## Overview

The Organism is a self-wiring autonomous trading organism that:

1. **Auto-discovers** all trading modules via decorators and package scans
2. **Wires** them via a central `EventBus` (pub/sub) without tight coupling
3. **Generates consensus signals** via weighted voting
4. **Self-improves** by adjusting module weights based on trade feedback
5. **Isolates** failing modules automatically

## Core Components

### EventBus (`core/event_bus.py`)

- Thread-safe, in-memory pub/sub
- Event types: `SIGNAL_GENERATED`, `ORDER_REQUEST`, `ORDER_FILLED`, `RISK_ALERT`, `MODULE_HEALTH`, `SELF_IMPROVEMENT`, `KILL_SWITCH`, `ORGANISM_WIRED`, `ORGANISM_STARTED`
- Methods:
  ```python
  from core.event_bus import get_event_bus
  bus = get_event_bus()
  bus.subscribe("SIGNAL_GENERATED", lambda e: print(e.payload))
  bus.publish("CUSTOM_EVENT", {"foo": "bar"}, source="my_module")
  bus.get_history("SIGNAL_GENERATED", limit=10)
  bus.get_stats()
  ```

### BaseTradingModule (`core/base_module.py`)

All new modules should inherit from this:

```python
from core.base_module import BaseTradingModule, register_module

@register_module
class MyAlpha(BaseTradingModule):
    module_name = "my_alpha"
    category = "alpha"
    version = "1.0.0"
    dependencies = []  # other module names

    def initialize(self) -> bool:
        # setup resources
        return True

    def analyze(self, market_data: dict):
        # produce ModuleResult
        from core.base_module import ModuleResult
        return ModuleResult(module_name=self.module_name, signal="BUY", confidence=0.75)

    def generate_signal(self, symbol, history_data):
        # or higher-level
        return self.analyze({"symbol": symbol})

    def on_event(self, event_type, payload):
        if event_type == "RISK_ALERT":
            self.enabled = False

    def self_improve(self, performance_history):
        # return {"improved": True, "details": ...}
        return {"module": self.module_name, "improved": False}
```

- Auto-registered via `@register_module` decorator, discovered by Organism.

### ModuleAutoDiscovery & AutoModuleRegistry

- `advanced_modules/module_registry.py`: scans `advanced_modules/*.py` and `core/*.py` for classes inheriting BaseTradingModule or legacy protocol (`initialize()` + `analyze/detect/decode/predict`).
- `core/organism.py::ModuleAutoDiscovery` also does package discovery.

### Organism (`core/organism.py`)

```python
from core.organism import Organism, OrganismConfig
from core.event_bus import get_event_bus

bus = get_event_bus()
org = Organism(config=OrganismConfig(auto_discover=True, enable_self_improvement=True), event_bus=bus)

result = org.discover_and_wire()  # {"active": [...], "legacy": [...], "total_active": N}
org.start()  # starts self-improvement & health threads

consensus = org.generate_consensus_signal("BTC/USDT", history_data)
# {
#   "symbol": ...,
#   "final_signal": "BUY",
#   "confidence": 0.78,
#   "weighted_confidence": 0.65,
#   "votes": {"BUY": 0.6, "SELL": 0.1, ...},
#   "module_results": {...},
#   "latency_ms": ...
# }

# Feedback loop:
org.feedback("my_alpha", reward=1.0)  # reward 0-1 from PnL

status = org.get_status()
org.stop()
```

**Self-improvement loop**:
- Every `self_improvement_interval_sec` (default 300s), organism:
  - Computes average feedback per module
  - Boosts weight by 5% if avg score >0.6, decays by 5% if <0.4
  - Normalizes weights sum to 1.0
  - Calls each module's `self_improve(history)` hook
  - Publishes `SELF_IMPROVEMENT` event
  - Persists JSON log to `data/organism_logs/`

**Health monitoring**:
- Tracks success/error counts, avg latency
- Isolates module if failures >= max_module_failures (default 5)
- Publishes `MODULE_HEALTH` on isolation

## Wiring Diagram

```
[BaseTradingModuleA] \
                     +--subscribe(*)--> EventBus <--> Organism (orchestrator)
[BaseTradingModuleB] /                   |
                                         |---> SIGNAL_GENERATED -> EventDrivenExecutor -> OKXEngine
                                         |---> SELF_IMPROVEMENT (log)
                                         |---> MODULE_HEALTH (isolation)
                                         |---> ORDER_FILLED feedback -> organism.feedback()
```

No direct module-to-module imports required. Communication via events.

## Historical Archive & QC Removal

- `Deco_*` folders are historical snapshots; NOT scanned by default (only `advanced_modules/` and `core/` limited set).
- `core/qmp_engine_v3.py` now has optional AlgorithmImports. If QC present, uses it; else fallback to mock.
- Legacy QC code to be archived into `legacy/` or external repo.

## Testing Organism

```python
from core.organism import Organism
from core.event_bus import reset_event_bus

reset_event_bus()
org = Organism()
discovered = org.discover_and_wire()
assert len(discovered["active"]) >= 0  # may be 0 if no new-style modules yet
org.start()
# simulate history
import pandas as pd
import numpy as np
...
org.stop()
```

## Adding a New Module (Checklist)

1. Create file `advanced_modules/my_new_module.py`
2. Implement class inheriting `BaseTradingModule` with `@register_module`
3. Implement `initialize()` returning True
4. Implement `analyze()` or `generate_signal()`
5. (Optional) `on_event()` and `self_improve()`
6. Add unit test `tests/test_my_new_module.py`
7. Run `python -m pytest`
8. Verify auto-discovered: `from advanced_modules.module_registry import get_auto_registry; get_auto_registry().discover()`

## Future: Self-Coding Extension

`core/self_coder.py` and `self_evolution_agent.py` can generate new modules via LLM and register them automatically. The Organism's event bus provides the hook: new module file dropped into `advanced_modules/`, next discovery cycle picks it up without restart if you call `discover_and_wire()` again.

