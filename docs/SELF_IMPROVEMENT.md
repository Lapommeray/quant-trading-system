# Bounded self-improvement runtime

The canonical runtime is `autonomy/`.  It now has one controlled improvement
loop for every registered module:

```text
market data
    -> MARKET_REGIME event
    -> module signals
    -> consensus + SIGNAL_GENERATED
    -> execution adapter
    -> TRADE_OUTCOME / ORDER_FILLED
    -> LearningStore (prediction, outcome, mistake)
    -> weight and parameter adaptation
    -> diagnosis -> validated code artifact -> governed approval
```

## What is automatic

Each module that implements `BaseModule`/`BaseTradingModule` receives the same
lifecycle:

- event-bus wiring and health tracking;
- `learn_from_outcome()` calls after a reconciled outcome;
- market-regime context (`low_bull`, `high_range`, etc.);
- damped module-weight updates based on recent outcomes in that regime;
- an allow-listed adaptive parameter update;
- deterministic code-artifact generation from the module's diagnosis and
  mistake memory;
- AST/policy validation;
- automatic approval of **low-risk generated artifacts** when enabled;
- shadow deployment of approved candidates using the same market observations
  as the active module;
- automatic promotion only after the shadow observation/outperformance,
  drawdown, Sharpe, and gold-set gates pass;
- automatic reinitialization through a module `repair()` hook after repeated
  failures.

The learning memory is an append-only JSONL file at
`data/autonomy/learning.jsonl` by default.  Runtime artifacts and cycle logs
are ignored by Git.  Set `QTS_LEARNING_PATH` and `QTS_SELF_CODING_DIR` to place
them elsewhere.

## What is deliberately not automatic

A generated artifact is not imported into the live process.  The self-coder
never overwrites source files.  Changes touching order execution, risk limits,
leverage, credentials, kill switches, the event bus, or the organism are
classified as protected and remain `pending_approval`.  A syntax check is not
a profitability or safety proof, so promotion of a protected change must be a
reviewed deployment step with backtests and operational checks.

This is intentional: autonomous live-source mutation could turn a transient
market/data error into an uncontrolled order path.

## Configuration

```python
from autonomy import Organism, OrganismConfig

organism = Organism(
    OrganismConfig(
        enable_self_improvement=True,
        self_coding_enabled=True,
        auto_approve_low_risk=True,
        auto_apply_low_risk=True,
        max_auto_changes_per_cycle=3,
    )
)
organism.discover_and_wire()
organism.start()
```

For a fully review-driven deployment:

```python
OrganismConfig(
    auto_approve_low_risk=False,
    auto_apply_low_risk=False,
)
```

## Adding a module

```python
from autonomy import BaseModule, ModuleResult, register_module

@register_module(name="my_alpha")
class MyAlpha(BaseModule):
    module_name = "my_alpha"
    category = "alpha"

    def generate_signal(self, symbol, history_data):
        return ModuleResult(self.module_name, "NEUTRAL", 0.0)
```

The organism discovers the module in the active package, wires it to the
canonical `core.event_bus`, and gives it the same learning/self-coding hooks.
No module should write source files or change execution/risk controls from an
event callback.

## Inspection

```python
status = organism.get_status()
cycle = organism.run_self_improvement_cycle()
print(status["learning"])
print(status["self_coder"])
print(cycle["improvements"])
```

The `SelfCodingEngine` can also be used directly in a test or maintenance
job.  `SafeCodeValidator` parses and policy-checks generated Python without
executing it; a caller may inject a separate backtest/test function when a
candidate is evaluated.  The older `self_evolution_agent.py` remains a
compatibility daemon; its `--auto-apply` option now stores validated candidates
under `strategies/quarantine/` rather than mutating active modules.

For systemic containment details, see [`AUTONOMOUS_SAFETY.md`](AUTONOMOUS_SAFETY.md).
