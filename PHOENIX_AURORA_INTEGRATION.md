# Phoenix Protocol and Aurora Gateway Integration

This document describes the integration of the Phoenix Protocol and Aurora Gateway components into the QMP Overrider system.

## Overview

The Phoenix Protocol and Aurora Gateway are advanced intelligence layers that enhance the QMP Overrider system with specialized capabilities:

- **Phoenix Protocol**: Provides market collapse detection, regime classification, and anti-failure decision making capabilities.
- **Aurora Gateway**: Provides advanced signal fusion and market intelligence by integrating multiple data sources and analysis methods.

These components are integrated with the Truth Checker, which compares signals from all three systems (QMP, Phoenix, Aurora) to ensure signal consistency and provide a higher-level decision mechanism.

## Integration Architecture

```
                  ┌─────────────────┐
                  │  Truth Checker  │
                  └────────┬────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
┌──────────▼─────────┐ ┌───▼───────────┐ ┌─▼──────────────┐
│   QMP Overrider    │ │Phoenix Protocol│ │ Aurora Gateway │
└──────────┬─────────┘ └───────────────┘ └────────────────┘
           │
┌──────────▼─────────┐
│  Oversoul Director │
└──────────┬─────────┘
           │
┌──────────▼─────────┐
│   Module Routing   │
└────────────────────┘
```

## Phoenix Protocol

The Phoenix Protocol is implemented in the `phoenix_protocol` package and provides:

1. **Collapse Memory**: Detects market collapse patterns based on historical crashes.
2. **Regime Sentience**: Classifies the current market regime (Crisis, Volatile, Bearish, etc.).
3. **Anti-Failure Engine**: Generates survival actions based on regime and collapse risk.

### Usage

```python
from phoenix_protocol import PhoenixProtocol

# Initialize
phoenix = PhoenixProtocol(algorithm)

# Get Phoenix action
metrics = {
    "leverage_ratio": 2.5,
    "volatility": 25.0,
    "vix_term_structure": 0.9,
    "etf_flow_velocity": -0.5
}
phoenix_result = phoenix.get_phoenix_action(metrics)

# Use the result
action = phoenix_result["decision"]["action"]
multiplier = phoenix_result["decision"]["position_multiplier"]
```

## Aurora Gateway

The Aurora Gateway is implemented in the `aurora_gateway` package and provides:

1. **Module Loading**: Loads specialized intelligence modules (Tartarian, Atlantean, Fed, etc.).
2. **Signal Collection**: Collects signals from all modules based on market data.
3. **Signal Fusion**: Fuses all collected signals into a single recommendation.

### Usage

```python
from aurora_gateway import AuroraGateway

# Initialize
aurora = AuroraGateway(algorithm)

# Load modules
aurora.load_modules()

# Collect signals
market_data = {
    "close": current_price,
    "ma_fast": fast_ma,
    "ma_slow": slow_ma
}
aurora.collect_signals(market_data)

# Fuse signals
aurora_result = aurora.fuse_signals()

# Use the result
signal = aurora_result["signal"]
confidence = aurora_result["confidence"]
```

## Truth Checker

The Truth Checker is implemented in the `truth_checker` package and provides:

1. **Signal Comparison**: Compares signals from QMP, Phoenix, and Aurora.
2. **Agreement Detection**: Determines the level of agreement between signals.
3. **Signal Resolution**: Resolves conflicts and provides a final decision.

### Usage

```python
from truth_checker import TruthChecker

# Initialize
truth_checker = TruthChecker(algorithm)

# Add signals
truth_checker.add_signal("qmp", "BUY", 0.85)
truth_checker.add_signal("phoenix", "BUY", 0.75)
truth_checker.add_signal("aurora", "BUY", 0.90)

# Resolve signals
result = truth_checker.resolve_signal()

# Use the result
final_signal = result["signal"]
confidence = result["confidence"]
agreement = result["agreement"]
```

## Ritual Lock

The Ritual Lock is implemented in the `ritual_lock` package and provides:

1. **Moon Phase Analysis**: Calculates the current moon phase.
2. **Geomagnetic Storm Detection**: Checks for geomagnetic storms.
3. **Mercury Retrograde Tracking**: Tracks Mercury retrograde periods.
4. **Solar Activity Monitoring**: Monitors solar activity levels.

### Usage

```python
from ritual_lock import RitualLock

# Initialize
ritual_lock = RitualLock(algorithm)

# Check alignment
is_aligned = ritual_lock.is_aligned("BUY")

# Use the result
if is_aligned:
    # Execute trade
else:
    # Skip trade due to cosmic misalignment
```

## Agent Lab

The Agent Lab is implemented in the `agent_lab` package and provides:

1. **Population Initialization**: Creates a population of QMP agents.
2. **Fitness Evaluation**: Evaluates the fitness of each agent.
3. **Darwinian Evolution**: Evolves the population through selection, crossover, and mutation.

### Usage

```python
from agent_lab import AgentLab

# Initialize
agent_lab = AgentLab(algorithm, population_size=100)

# Run evolution cycle
result = agent_lab.darwin_cycle(market_data)

# Get best parameters
best_params = agent_lab.get_best_parameters()
```

## Consciousness Layer

The Consciousness Layer is implemented in the `consciousness_layer` package and provides:

1. **Decision Explanation**: Generates human-readable explanations for system decisions.
2. **Market Memory**: Records and learns from market events.
3. **Prediction Tracking**: Tracks prediction accuracy and evolves consciousness.
4. **Intention Setting**: Sets intentions in the consciousness field.

### Usage

```python
from consciousness_layer import ConsciousnessLayer

# Initialize
consciousness = ConsciousnessLayer(algorithm)

# Generate explanation
explanation = consciousness.explain(
    decision="BUY",
    gate_scores=gate_scores,
    market_data=market_data,
    additional_context={
        "phoenix_regime": "BULLISH",
        "aurora_signal": "BUY"
    }
)

# Record market memory
consciousness.record_market_memory(
    timestamp=algorithm.Time,
    symbol=symbol,
    event_type="signal",
    event_data={"direction": "BUY", "confidence": 0.85}
)

# Record prediction
consciousness.record_prediction(
    prediction="BUY",
    actual_outcome="BUY"
)

# Get consciousness data
consciousness_data = consciousness.get_consciousness_data()
```

## Integration with Main System

These components are integrated into the main QMP Overrider system through the Oversoul Director, which manages all modules and provides meta-control over the entire system.

The integration ensures that:

1. All signals pass through the Truth Checker for validation.
2. The Phoenix Protocol provides regime awareness and collapse protection.
3. The Aurora Gateway contributes specialized intelligence from multiple sources.
4. The Ritual Lock prevents trades during cosmic misalignment.
5. The Agent Lab continuously evolves and optimizes the system.
6. The Consciousness Layer provides human-readable explanations and evolving awareness.

This integrated architecture creates a unified, intelligent trading system that combines multiple layers of market perception, decision-making, and self-awareness.
