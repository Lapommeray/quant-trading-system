# Historical Archive Caveat

## Overview
The repository contains a large number of historical evolution artifacts:

- `Deco_10/`, `Deco_11/`, `Deco_14__/` ... `Deco_39/`, `Deco_A_/`, `Deco_15._/` etc.
- `QMP_GOD_MODE_v2_5_FINAL/`, `QMP_Overrider_Complete/`, `QMP_v2.1_FINAL_*`
- `QMP_v2_CLEAN/`, `Sa_son_code/`, `Shadow/`, `DNA_HEART/` etc.

These folders are **preserved for lineage and audit** but are **NOT active runtime code**.

## Why they exist
- Each Deco_XX folder represents a prior Devin/Arena/Copilot iteration.
- They contain full system snapshots used to verify never-loss, 200% outperformance claims.
- Some are required for regression tests that compare against historical behavior.

## Active Code Paths (post PR #117)

Active, maintained modules are:

- `core/` - qmp_engine_v3, organism, event_bus, base_module, institutional indicators
- `advanced_modules/` - modern alpha modules (auto-discovered via module_registry)
- `execution/` - okx_engine (primary), event_driven_executor, advanced VWAP/TWAP
- `quant_trading_system/` - pip-installable package (config, data_feeds/yfinance, execution/okx_executor)
- `safety_governance.py` - eternal guardrails, kill switch, human confirmation
- `mt5_live_engine.py` / `mt5_bridge.py` - MT5 secondary execution
- `data/` - cache and live data

## .preserve / .gitignore strategy

- Use `.preserve-advanced-commit-link.py` to keep symlink `enhanced_indicator.py -> Deco_19/core/enhanced_indicator.py` while ignoring Deco_19 runtime.
- CI lints only active paths; archive paths are excluded via `flake8` config? If not, add `--exclude=Deco_*,QMP_*` in lint workflow.
- New code MUST NOT import from Deco_* directly. Import via `core` or `advanced_modules` after migration.

## Migration Rule

If a useful module lives only in Deco_XX:
1. Copy it into `advanced_modules/` or `core/` with proper BaseTradingModule inheritance.
2. Register via `@register_module`.
3. Add unit test in `tests/`.
4. Document in `TRANSCENDENT_INTELLIGENCE.md` or relevant README.

## QuantConnect Legacy

- `AlgorithmImports` dependency is now optional.
- Files that previously did `from AlgorithmImports import *` have been refactored to fallback to `mock_algorithm_imports`.
- Legacy QC folders remain but are not used for OKX live trading.

## Future Cleanup

Consider moving all Deco_XX and QMP_* into `archive/` git submodule or separate repo to reduce clone size (currently >100 directories). For now preserved.

