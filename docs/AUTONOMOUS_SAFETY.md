# Hardened autonomous safety system

The active runtime now treats evolution as a **shadow deployment**, not as a
live-source write. The immune system is independent from the adaptive modules
and is allowed to veto them.

## 1. Candidate shadow deployment

`autonomy.shadow.ShadowManager` clones the active module and applies candidate
parameters only to the clone. Every subsequent market observation is sent to
both the active clone and the candidate clone. Only the active module's signal
is sent to `SIGNAL_GENERATED` and execution.

A candidate needs, by default:

- 100 observations;
- at least 5% cumulative outperformance;
- no more than 1 percentage point additional drawdown;
- no worse Sharpe ratio;
- a passing gold-set stress test.

A successful comparison invokes `Organism._promote_shadow`, swaps the module
instance, and emits `SHADOW_PROMOTED`. No generated file is imported or
executed as live code during this process. Set
`ORGANISM_AUTO_PROMOTE_SHADOWS=false` when a deployment requires a human
promotion decision after the quantitative gates.

## 2. Generated tests and regression tests

Every `CodeProposal` contains a `(code, test_suite)` pair. The generated test
suite is AST-validated and run in `autonomy.sandbox.SandboxExecutor` with a
process timeout and best-effort Unix resource limits. When
`run_baseline_tests=True`, the configured baseline command (default
`python -m pytest -q`) also runs before approval. A failed generated test or
baseline regression rejects the proposal. A rejection is written to the
learning memory as a self-coder mistake by the organism.

The baseline runner uses `QTS_BASELINE_RUN=1` to prevent recursive baseline
invocation when the test suite itself exercises the self-coder.

## 3. Panic sentinel / survival mode

`MultiTimeframeSentinel` checks `1m`, `15m`, and `1h` observations. A 3-sigma
shock in all three enters `SURVIVAL_MODE`; stabilization requires several
non-shock observations. During survival mode:

- consensus signals are forced to neutral;
- adaptive weights are ignored;
- the executor blocks new orders;
- OKX live execution applies the same hard stop;
- guardrails use the survival profile.

The sentinel also has a heartbeat. A stale or failed sentinel is treated as a
critical failure and forces survival mode; the system never interprets missing
sentinel data as permission to trade.

## 4. Priority event lanes

`core.event_bus.EventBus` exposes four lanes. Critical callbacks are
synchronous; evolutionary callbacks can be queued and are never allowed to
hold an execution lock. The bus also exposes `drain()` and lane statistics for
operations checks.

| Lane | Priority | Examples |
|---|---:|---|
| Critical | 0 | kill switch, risk alert, order request/fill, survival mode |
| Operational | 1 | market data, signals, consensus |
| Adaptive | 2 | regime, health, weight updates |
| Evolutionary | 3 | memory, self-coding, improvement events |

`publish_async` and asynchronous subscriptions are scheduled through a
`PriorityQueue`. Critical synchronous callbacks are never placed behind
learning or code-generation work.

## 5. Gold-set stress validation

`data/gold_set.jsonl` contains 10 crash traces and 10 high-volatility traces.
`GoldSetStressTester` paper-simulates active and candidate modules across the
set and rejects candidates that breach the drawdown/equity floor or materially
degrade active stress behavior. The file is a versioned calibration fixture;
production deployments should add audited venue-specific OHLCV traces.

## 6. Strict AST policy and penalty box

`SafeCodeValidator` now uses an import allow-list, forbidden call/name/dunder
checks, code-size limits, complexity/nesting limits, and basic obfuscation
checks. It rejects `os`, `subprocess`, `eval`, `exec`, environment access,
reflection, multiprocessing, threading, and unknown imports.

A policy violation triggers `PenaltyBox`, temporarily disabling the self-coder.
The penalty cannot be removed by generated code. It expires or requires an
explicit operational reset; financial guardrails require a separate manual
reset.

## 7. Immutable financial containment

`AutonomousGuardrails` uses a frozen limits object and accepts only tighter
caller limits. Defaults are:

- 2% maximum position per trade;
- 50 trades per UTC day;
- 5% daily loss stop;
- 1x maximum leverage;
- 10-second minimum interval;
- tighter survival-mode position limits.

The guardrails run immediately before the live OKX order path and are
independent of generated module configuration.  They record violations and
trigger an emergency stop when the daily loss limit is reached.

## 8. Audit locations

By default, append-only records are written under `audit_logs/`:

- `autonomous_events.jsonl` — event bus activity;
- `trade_audit.jsonl` — direct protected trade attempts;
- `violations.jsonl` — immutable guardrail violations;
- `code_generation.jsonl` — proposal/validation/approval lifecycle;
- `survival_mode.jsonl` — panic and heartbeat failures;
- `shadow_promotions.jsonl` — candidates promoted after gates.

Set `QTS_AUDIT_PATH` to provide a separate aggregate event path.
