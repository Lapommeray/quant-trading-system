# Autonomous organism guide

`autonomy/organism.py` is the canonical active runtime.  It is deliberately
separate from the historical `Deco_*` snapshots and does not require
QuantConnect.

## Runtime flow

```text
+------------------+      MARKET_REGIME       +-------------------+
| Real market data | ------------------------> | Registered modules|
+------------------+                           +---------+---------+
                                                        |
                                                        v
                                             +----------------------+
                                             | Weighted consensus   |
                                             +----------+-----------+
                                                        |
                           SIGNAL_GENERATED             v
                                             +----------------------+
                                             | fail-closed executor |
                                             +----------+-----------+
                                                        |
                                      TRADE_OUTCOME / ORDER_FILLED
                                                        v
                                             +----------------------+
                                             | LearningStore         |
                                             | outcomes + mistakes   |
                                             +----------+-----------+
                                                        |
                                                        v
                                             +----------------------+
                                             | Self-coding coordinator|
                                             | diagnose/validate/     |
                                             | approve/recover       |
                                             +----------------------+
```

`core/event_bus.get_event_bus()` and `autonomy.get_event_bus()` return the same
thread-safe bus.  Modules, the organism, the executor and the OKX adapter can
therefore communicate without importing one another.

## Module contract

```python
from autonomy import BaseModule, ModuleResult, register_module

@register_module(name="example_alpha")
class ExampleAlpha(BaseModule):
    module_name = "example_alpha"
    category = "alpha"

    def generate_signal(self, symbol, history_data):
        return ModuleResult(self.module_name, "NEUTRAL", 0.0)
```

The optional hooks are:

- `on_event(event_type, payload)` for shared context;
- `learn_from_outcome(outcome)` for model/memory updates;
- `apply_adaptive_parameters(parameters)` for the narrow runtime allow-list;
- `self_improve(history)` for module-local bounded tuning;
- `repair(reason)` for reinitialization after a health failure.

No hook may mutate order execution, credentials, leverage, position limits,
kill switches, or source files.

## Discovery and interconnection

`Organism.discover_and_wire()` imports only configured active packages
(default: `autonomy`, `core`, and optional `advanced_modules`) and accepts
classes derived from `BaseTradingModule`.  Import failures from optional
research modules are isolated and reported in the return value.  Registered
modules are wired once; duplicate event subscriptions are ignored by the
canonical bus.

The repository includes two always-importable baseline modules:

- `momentum_alpha`: bounded price momentum signal;
- `regime_risk_gate`: neutral veto signal during high-volatility conditions.

They are intentionally conservative and remain subject to `okx_live` safety
checks.

## Learning and adaptation

`autonomy.learning.LearningStore` records predictions, realized outcomes,
feedback, market regimes, and negative-outcome lessons in JSONL.  A prediction
id is included in each consensus payload.  The executor emits a correlated
`TRADE_OUTCOME` event on a successful fill so reconciliation can associate a
result with all contributing modules.

`MarketRegimeDetector` describes recent data as combinations such as
`low_bull`, `low_range`, or `high_bear`.  Weight updates are damped and
regime-aware.  Regime context can tune only the safe adaptive namespace; it
cannot bypass the executor's confidence/risk filters.

## Self-coding and recovery

`autonomy.self_coding.SelfCodingEngine` performs four bounded steps:

1. diagnose module health and recent mistake statistics;
2. create a deterministic Python policy artifact containing allow-listed
   parameters and lessons;
3. AST/policy validate it without executing it;
4. auto-approve low-risk artifacts when policy allows;
5. shadow-test and auto-promote only after performance and gold-set gates.

Artifacts are written to `strategies/evolved/` by default.  Protected modules
and paths are classified `critical`/`high` and stay `pending_approval`.  The
engine never overwrites live source and never imports an artifact into the
running trader.

After repeated module errors, the organism first calls the module's `repair`
(or `initialize`) hook.  If repair fails, the module is isolated and its
weight is removed from consensus.  A `MODULE_HEALTH` event records the action.

See [`SELF_IMPROVEMENT.md`](SELF_IMPROVEMENT.md) and
[`AUTONOMOUS_SAFETY.md`](AUTONOMOUS_SAFETY.md) for configuration and
operational examples.
