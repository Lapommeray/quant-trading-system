# Repository Resume — Sacred-Quant Fusion Trading System

Prepared for the `evolve.py` self-evolving daemon integration. This resume tells
you exactly where to hook in, what metric to guard, what is safe to modify, how
to test, and which branch/name to use.

---

## 1. What the system actually is

The project self-describes as a **"quant trading system"** but in practice it is
a **research/backtesting + signal-generation + autonomous self-improvement
codebase** more than a single runnable bot. Concretely it bundles:

- **Live trading engines / bridges** — `mt5_live_engine.py`, `mt5_bridge.py`,
  `run_mt5_live.py`, `okx_live/`, `kalshi_live_engine.py`, `main.py`.
- **Signal generation** — `signals/`, `advanced_modules/`, `core/` (dozens of
  "sacred-geometry"/"quantum" indicator modules).
- **Backtesting** — `backtest/walk_forward_quantum_backtest.py`,
  `advanced_modules/enhanced_backtester.py`.
- **Autonomous self-improvement** — the real daemons already exist:
  - `self_evolution_agent.py` — "perpetual innovation daemon" (multi-agent
    debate: researcher → coder → critic → tester; arXiv ingestion).
  - `meta_evolve.py` — genetic strategy breeder → `strategies/evolved/`.
  - `evolve.py` — the **autonomous evolution guardian** you are tailoring.
  - `autonomy/` — the canonical, governed self-improvement runtime
    (`organism.py`, `self_coding.py`, `shadow.py`, `gold_set.py`, `learning.py`).

**Important context:** the tree is heavily polluted with duplicate snapshot
directories (`Deco_10` … `Deco_39`, `QMP_*`, `QMP_GOD_MODE_v2_5_FINAL`) and
dozens of cosmetic `*_TESTAMENT.md` files. There are **6,395 tracked files**.
Most "performance" reports hardcode a 100%-win-rate ("no-hopium") outcome, which
the README itself warns is not a real performance proof. Do not let the daemon
optimize those; they are not real metrics (see §4).

---

## 2. Key entry points

| Purpose | File(s) |
|---|---|
| Live runner (MT5 / signals) | `main.py`, `run_mt5_live.py`, `mt5_live_engine.py`, `mt5_bridge.py` |
| Perpetual self-evolution daemon | `self_evolution_agent.py` (`--demo`, `--daemon`, `--auto-apply`) |
| Genetic strategy breeder | `meta_evolve.py` |
| **Autonomous evolution guardian (target)** | `evolve.py` |
| Canonical governed self-improvement runtime | `autonomy/` (`Organism`) |
| Config | `config.json`, `.env.example` |
| Package metadata / scripts | `pyproject.toml` (`quant-trading = quant_trading_system.cli:main`) |

### Where the daemon's build/quality tests are run
- `evolve.py` defines the suite: `pytest -x --tb=short`,
  `python3 test_system_integration.py`,
  `python3 run_comprehensive_test.py`,
  `python3 run_complete_enhanced_test.py`.

---

## 3. Testing entry points

- **pytest** (config in `pytest.ini`, mirrored in `pyproject.toml`):
  - `addopts = --import-mode=importlib`
  - `testpaths = tests, testing, covid_test, Sa_son_code/quant_trading_system/tests`
  - `norecursedirs` excludes `.venv`, `QMP_GOD_MODE_v2_5_FINAL`, and
    `tests/quantum_enhancements`.
- **CI** (`.github/workflows/ci.yml`): on push/PR to `main` — `flake8 .` +
  `black --check .`, then `pytest`. CI pins Python **3.8** (note: local sandbox
  has Python 3.11 and **pandas/numpy are not installed** yet).
- **`.flake8`**: `max-line-length = 88`, excludes `.md/.rst/.txt`, `Deco_*`,
  `quarantine`, `docs`.
- Standalone runners: `run_comprehensive_test.py` (module factory must discover
  ≥19 modules), `run_complete_enhanced_test.py` (4 suites, **writes the metrics
  JSON**), `run_ultimate_test.py`, `test_system_integration.py`.

---

## 4. Success metric — where to look and what to guard ⚠️

There are two "metric worlds" in this repo. This is the single most important
thing for your daemon:

### A. The metric `evolve.py` *currently* claims to guard (broken/no-op)
- `evolve.py` sets `METRICS_FILE = complete_enhanced_test_results.json` and
  `load_backtest_metrics()` reads `win_rate`, `max_drawdown_pct`,
  `sharpe_ratio`, `profit_factor`, `total_pnl` from it.
- **But that JSON is written by `run_complete_enhanced_test.py`, which only
  emits `success_rate`, `test_suites` (booleans), `total_suites`,
  `passed_suites`, `system_features`. It does NOT contain any of the five
  metrics `evolve.py` looks for.**
- Result: `load_backtest_metrics()` returns all zeros (via `flat.get(...,0.0)`),
  so `metrics_degraded()` compares zero vs zero and **never fires**. The guard
  is effectively a no-op today.
- Also note a key-name mismatch: the real backtester emits `max_drawdown`
  (not `max_drawdown_pct`).
- `complete_enhanced_test_results.json` is gitignored and currently absent, so
  the daemon cannot even load a baseline until a test run produces it.

### B. The real, reproducible metric source (recommended guard)
- `advanced_modules/enhanced_backtester.py` → `_calculate_metrics()` returns a
  dict with exactly:
  ```python
  {'total_trades', 'winning_trades', 'losing_trades', 'win_rate',
   'total_pnl', 'avg_win', 'avg_loss', 'profit_factor', 'sharpe_ratio',
   'max_drawdown'}
  ```
  and `export_results()` writes `backtest_results_<ts>.json` under
  `output_dir`. This is a genuine, math-backed metric set.

**Recommendation for `evolve.py`:** wire the guard to
`enhanced_backtester`/`walk_forward_quantum_backtest` output (or your own
walk-forward run), normalize to a canonical key set
(`win_rate`, `sharpe_ratio`, `profit_factor`, `max_drawdown`), and add a
`total_pnl`/mean-return objective. Guard **win rate, Sharpe, profit factor, max
drawdown** as hard floors (reject if any degrades) and use **total PnL** as the
improvement objective.

---

## 5. Sensitive / protected paths (never alter automatically)

`evolve.py` already ships a `PROTECTED_PATHS` allowlist that its patch filter
(`patch_touches_protected_paths`) enforces. Keep and extend it. Categories:

- **Risk / safety / compliance:** `safety_governance.py`,
  `risk_mitigation_layers/`, `compliance_check.py`,
  `monitoring_tools/compliance_firewall.py`, `capital_controller.py`.
- **Credentials / config:** `.env`, `.env.example`, `config.json`,
  `SECURITY_RED_ALERT_OKX_KEYS.md`.
- **Execution / live bridges:** `mt5_bridge.py`, `okx_live/`,
  `kalshi_live_engine.py`, `kalshi_complementary_arb.py`,
  `cross_asset_arbiter.py`, `meta_order_router.py`, `mt5_live_engine.py`.
- **Test infra:** `conftest.py`, `pytest.ini`, `pyproject.toml`.
- **Other daemons / governance:** `meta_evolve.py`, `oracle_sentry.py`,
  `self_evolution_agent.py`, `run_7day_autonomous_cycle.py`,
  `quantum_coherence_tracker.py`, `zk_proof_verifier.py`, plus all the
  `*_engine.py` "transcendence" files and their `.log`/`.json` artifacts.

Per `docs/AUTONOMOUS_SAFETY.md` and `docs/SELF_IMPROVEMENT.md`, changes touching
**order execution, risk limits, leverage, credentials, kill switches, the event
bus, or the organism** must stay `pending_approval` for humans. Generated code
is written as artifacts under `strategies/evolved/` and `strategies/quarantine/`
— never imported into the live process.

**Safest mutation surface for evolution:** the indicator/alpha modules under
`advanced_modules/`, `signals/`, `core/` (non-execution signal logic) and the
backtest parameter files.

---

## 6. Branch name & repo layout (the misstep to avoid) ⚠️

- The **current session branch is `arena/019fdf4d-quant-trading-system`**
  (that is what must be used for all work, commits and pushes).
- **`evolve.py` currently hardcodes `MAIN_BRANCH = "arena/019fde33-quant-trading-system"`** — a *stale* session branch that is not checked out here. Its
  `evolve()` would `git checkout` to a branch that doesn't exist locally, which
  would fail or misbehave.
- **Fix required:** set `MAIN_BRANCH = "arena/019fdf4d-quant-trading-system"`.
- Do **not** point the daemon at `main`. PRs are opened against `main` (see
  `.github/workflows/push_and_open_pr.yml`), but the working/session branch is
  the `arena/*` branch. The daemon should create short-lived
  `evolve/<ts>-<hash>-<desc>` candidate branches off the session branch, test,
  and only merge if metrics don't degrade.
- Layout convention: evolution artifacts live in `strategies/evolved/`,
  `strategies/generated/`, `strategies/quarantine/` (all gitignored); lineage in
  `genetic_lineage.json` (gitignored).

---

## 7. Environment status / gotchas

- Local sandbox: **Python 3.11.2**, `pandas`/`numpy` **not installed** → any
  pytest run importing them fails until `requirements.txt` is installed.
- CI uses Python **3.8** + `flake8` + `black` + `pytest`; your evolved code must
  stay 3.8-compatible and pass `flake8` (88-char lines).
- The daemon's `repo_clean()` aborts if the tree is dirty, and each candidate is
  isolated on its own branch — keep that isolation property.
- `evolution/refinement_loop.py` and `autonomy/` already implement genetic /
  governed improvement loops; `evolve.py` should reuse their conventions
  (`strategies/evolved/`, shadow gates, gold-set stress) rather than reinvent.

---

## 8. Concrete integration checklist for `evolve.py`

1. Fix `MAIN_BRANCH` → `arena/019fdf4d-quant-trading-system`.
2. Point the metric source at a real backtest (enhanced_backtester or walk-forward).
3. Canonicalize keys → guard `win_rate`, `sharpe_ratio`, `profit_factor`,
   `max_drawdown` as floors; drive `total_pnl` up as objective.
4. Keep `PROTECTED_PATHS`; add any execution/risk/credential module.
5. Keep per-candidate branch isolation + `repo_clean()` + `git apply --check`.
6. Keep test commands pinned to maintained suites (respect `norecursedirs`).
7. Ensure evolved artifacts land in gitignored `strategies/evolved/`, never in
   live source.
