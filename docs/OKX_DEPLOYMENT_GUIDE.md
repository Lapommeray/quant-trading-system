# OKX Live Trading Deployment Guide (Event-Driven)

## Quick Start (Paper Mode - Default Safe)

```bash
export OKX_LIVE_TRADING=false   # default
python run_okx_live.py --symbols BTC/USDT,ETH/USDT --interval 60
```

**Paper mode is enforced by default**. No real funds at risk.

## Live Trading Prerequisites (Safety Requirements)

To enable live trading, ALL must be satisfied:

1. **Environment variables**:
```bash
export OKX_API_KEY="your_okx_api_key"
export OKX_API_SECRET="your_okx_secret"
export OKX_PASSPHRASE="your_okx_passphrase"
export OKX_LIVE_TRADING=true          # explicit live flag
export SAFETY_REQUIRED_CONFIRMATIONS=3  # require human confirmation
```

2. **Human confirmation**: First `SAFETY_REQUIRED_CONFIRMATIONS` trades require confirmation code.

```python
from safety_governance import SafetyGovernanceSystem
sys = SafetyGovernanceSystem(paper_mode=False, required_confirmations=3)
# When trade pending:
# 1. Code printed in logs: Code: ABCD1234
# 2. Confirm:
#    sys.confirm_trade(trade_id, "ABCD1234", user="your_name")
# Or enable override for limited time:
#    sys.enable_human_override(duration_minutes=60)
```

3. **Eternal guardrails** (hard caps, cannot bypass):
- Max single trade risk 3%
- Max daily loss 5%
- Max drawdown 15%
- Max leverage 3.0x
- Max position concentration 25%
- Live trading requires human override env or code confirmation

4. **Kill switch**:
```bash
# Send SIGUSR1 to process
kill -SIGUSR1 <pid>
# Or programmatically
engine.activate_kill_switch("risk event")
```

## Architecture: Event-Driven OKX Execution

```
[QMP Engine v3] --SIGNAL_GENERATED--> [EventBus] -->> [EventDrivenExecutor] -->> [OKXEngine (CCXT or Sim)]
                                          |
                                          |__>> [Organism self-improvement]
                                          |__>> [SafetyGovernance]
                                          |__>> [AuditLogger]
```

- `core/event_bus.py`: thread-safe pub/sub, persists last 1000 events.
- `core/organism.py`: auto-discovers modules, wires via bus, runs self-improvement every 300s.
- `execution/okx_engine.py`: CCXT adapter for OKX, falls back to simulation if ccxt missing or paper_mode.
- `execution/event_driven_executor.py`: subscribes to SIGNAL_GENERATED, validates confidence thresholds, translates to OKX orders.

## Example: Programmatic Usage

```python
from core.event_bus import get_event_bus
from core.organism import Organism
from execution.okx_engine import OKXEngine, OKXOrderRequest, OrderSide
from execution.event_driven_executor import EventDrivenExecutor, ExecutorConfig

bus = get_event_bus()
okx = OKXEngine(paper_mode=True)  # simulation
okx.connect()

organism = Organism(event_bus=bus)
organism.discover_and_wire()
organism.start()

executor = EventDrivenExecutor(
    okx_engine=okx,
    event_bus=bus,
    config=ExecutorConfig(min_confidence=0.65, allowed_symbols=["BTC/USDT"])
)
executor.start()

# Generate synthetic signal via organism or QMP engine
# Once SIGNAL_GENERATED published, executor auto-trades

# Direct manual:
from core.qmp_engine_v3 import QMPUltraEngine
qmp = QMPUltraEngine()
signal = qmp.generate_signal("BTC/USDT", history_data)  # dict of DataFrames
# event bus will auto route

# Or manual order:
order = OKXOrderRequest(symbol="BTC/USDT", side=OrderSide.BUY, quantity=0.001)
result = okx.place_order(order)
print(result.to_dict())
```

## OKX Specifics

- Symbol normalization: `BTC/USDT` or `BTC-USDT` both accepted, internally converted.
- Supported order types: MARKET, LIMIT. STOP not yet for spot.
- Leverage: capped at 3x default, configurable via `OKX_MAX_LEVERAGE` env or `OKXEngine(max_leverage=...)`.
- Position sizing: `execute_from_organism_signal` uses 1% equity * confidence / price.
  - Notional must be >= $5, otherwise skipped.
  - Max per-symbol notional 10% equity (configurable `OKX_MAX_POSITION_PCT`).

## Dependencies

```bash
pip install -e ".[sentiment]"      # includes ccxt
# ccxt is optional - if missing, engine runs in simulation
```

## Testing

```bash
source .venv/bin/activate
pip install pytest pandas
pytest -q                             # 87 passed + new tests
python -m pytest tests/test_okx_engine.py -v
python -m pytest tests/test_organism.py -v
```

## Safety Checklist Before Live

- [ ] OKX API keys have NO withdrawal permission, only trade permission
- [ ] IP whitelist enabled on OKX
- [ ] `OKX_LIVE_TRADING=true` + `HUMAN_OVERRIDE` env set when required
- [ ] `audit_logs/` directory writable
- [ ] Circuit breaker thresholds reviewed
- [ ] Rate limit: `OKX_MAX_ORDERS_PER_MIN = 20` (default)
- [ ] Kill switch tested: `kill -SIGUSR1 <pid>` halts trading
- [ ] Paper trading run for 24h without errors
- [ ] Event bus history inspected via `get_event_bus().get_history()`

## Troubleshooting

- **Engine stays in simulation**: ccxt missing or `paper_mode=True`. Install ccxt and set live env.
- **Order blocked**: check audit_logs and `RISK_ALERT` events, likely confidence or position cap.
- **Pending human confirmation**: look logs for Code: XXXXXXXX, confirm via `safety_governance`.
- **Eternal guardrail violation**: trade risk exceeds 3% -> reduce quantity or increase equity.
- **Deco_* import errors**: do NOT import from Deco_*; use active `core/` modules.

## Packages

- `from quant_trading_system.execution.okx_executor import OKXExecutor` for pip package usage
- `from execution.okx_engine import OKXEngine` for standalone script usage

Both expose same API.

