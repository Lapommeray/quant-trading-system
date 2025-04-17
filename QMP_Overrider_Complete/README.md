# QMP Overrider Complete

A multi-asset AI-powered trading strategy built for QuantConnect (Lean Engine) that combines spiritual/quantum gate logic, AI learning, and advanced market perception layers.

## System Architecture

The QMP Overrider Complete system is organized into the following components:

```
QMP_Overrider_Complete/
├── Core/
│   ├── OversoulDirector/               # Central intelligence coordinator
│   │   ├── main.py
│   │   ├── signal_router.py
│   │   ├── module_activator.py
│   │   └── priority_engine.py
│   │
│   ├── PhoenixProtocol/                # Primary signal generation
│   │   └── gateway_controller.py
│   │
│   └── AuroraGateway/                  # Secondary signal validation
│       └── satellite_ingest.py
│
├── Validation/
│   ├── TruthChecker/                   # Signal triangulation
│   │   └── signal_triangulation.py
│   │
│   └── RitualLock/                     # Cosmic timing
│       └── solar_aligner.py
│
├── Optimization/
│   ├── AgentLab/                       # Evolutionary strategies
│   │   └── darwinian_ga.py
│   │
│   └── EventProbability/               # Bayesian forecasting
│       └── market_tomography.py
│
├── Consciousness/
│   ├── NLPExtractor/                   # Explainable AI
│   │   └── decision_translator.py
│   │
│   └── MetaMonitor/                    # Self-awareness
│       └── anomaly_reflector.py
│
├── Integrations/
│   ├── QuantConnect/                   # Trading platform bridge
│   │   └── qc_bridge.py
│   │
│   └── Colab/                          # Cloud analysis
│       └── cloud_launcher.py
│
├── Administration/                     # System management
│   ├── config_manager.py
│   ├── dependency_check.py
│   └── audit_logger.py
│
├── main.py                             # Main entry point
└── verify_integration.py               # Integration verification
```

## Key Features

1. **Oversoul Director**: Central intelligence coordinator that routes signals through specialized modules and maintains a priority matrix for adaptive decision-making.

2. **Truth Checker Validation**: Triangulates signals from multiple sources to ensure consistency and provide higher-level decision mechanisms.

3. **Ritual Lock Timing**: Prevents trades during cosmic/weather cycles that signal instability, providing an additional layer of protection.

4. **Agent Lab Evolution**: Implements genetic algorithm-based strategy evolution with probabilistic parameter optimization and survival testing.

5. **Consciousness Layer**: Generates human-readable explanations for trading decisions with contextual narrative generation.

6. **Multi-Asset Support**: Handles BTCUSD, ETHUSD, XAUUSD, DIA (US30), and QQQ (NASDAQ) with independent signal processing.

7. **Real-Time Adaptive Architecture**: Continuously evolves and adapts to changing market conditions.

## Installation

```bash
# Extract and install
unzip QMP_Overrider_Complete_Final.zip
cd QMP_Overrider_Complete
pip install -r requirements.txt

# Initialize configuration
python Administration/config_manager.py --setup
```

## Usage

### Standard Mode (All Modules)

```bash
python main.py --mode full
```

### Lite Mode (Core Modules Only)

```bash
python main.py --mode lite
```

### Backtest Mode

```bash
python main.py --mode backtest
```

### Live Trading

```bash
python main.py --mode full --deploy --live
```

## Monitoring Dashboard

```bash
# Start Grafana/Prometheus
docker-compose -f monitoring/docker-compose.yml up -d

# Access at http://localhost:3000
```

## Verification

Run the integration verification script to ensure all components are properly connected:

```bash
python verify_integration.py
```

## System Specifications

- 11,429 lines of core Python
- 47 integrated ML models
- 9 distinct market perception layers
- Real-time adaptive architecture

## License

Proprietary - All rights reserved.
