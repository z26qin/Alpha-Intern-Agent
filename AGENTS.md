# AGENTS.md

Operating guide for AI coding agents (Codex, Claude, etc.) working inside
**AlphaInternAgent**.

## Project Goal

AlphaInternAgent ("Alpha Intern") is a lightweight, Hermes-style
stock-market machine-learning **research** agent. It is designed to grow
gradually, like a junior research analyst:

1. Load market data.
2. Engineer simple features.
3. Train simple ML signal models.
4. Run walk-forward backtests.
5. Record research notes into a memory store.
6. Accumulate reusable research skills in a skill registry.
7. Expose a small CLI surface so an LLM agent can later call these
   primitives as tools.

This is a **research and education** project. It is not a trading system
and it is not financial advice.

## Module Boundaries

- `src/alpha_intern/data/` — market data loading & normalization.
- `src/alpha_intern/features/` — backward-looking feature engineering.
- `src/alpha_intern/models/` — thin ML wrappers (no hyperparameter search).
- `src/alpha_intern/backtest/` — walk-forward backtesting and metrics.
- `src/alpha_intern/memory/` — JSONL research memory + skill registry.
- `src/alpha_intern/tools/` — reserved for tools an LLM agent can call.
- `src/alpha_intern/cli.py` — Typer CLI surface.

Keep modules independent. Lower layers (data, features) must not import
from higher layers (backtest, memory).

## Coding Style

- Python 3.11+.
- Type hints on public functions.
- Prefer pure functions and small classes.
- Use `pandas` for tabular data and `pydantic` for structured records.
- No global mutable state.
- Keep names explicit (`target_return_5d_forward`, not `y`).

## Safety Rules

- **Make the smallest safe change.**
- **Do not rewrite unrelated files.**
- **Add or update tests for new behavior.**
- **Avoid look-ahead bias** in features and backtests. Anything that
  uses future data must be named `target_*` or `forward_*`.
- **Do not add live trading or broker APIs.** No order routing, no
  brokerage SDKs, no portfolio management for real money.
- **Do not store secrets** (API keys, tokens) in the repo or in tests.
- **No network calls in tests.** Use synthetic data.
- This project is **research-only and is not financial advice**.

## Finance / Backtest Correctness Rules

- Features must use only data available at or before time `t`.
- Targets (`target_*`, `forward_*`) may use data from `t+1` onward, but
  must never be passed as features into a model.
- Group time-series operations by `ticker` and sort by `date` before
  rolling / shifting.
- Trading costs are modeled as simple basis-point haircuts. Do not
  pretend to model microstructure.
- Prefer transparent, deterministic backtests. No data snooping to
  inflate Sharpe.

## Testing

Install in editable mode, then run:

```bash
pip install -e .
pytest -q
```

All tests must be deterministic and offline.

## What This Repo Is *Not*

- Not a live trading bot.
- Not a portfolio optimizer for real capital.
- Not a financial advisor.
- Not a heavyweight LLM agent framework (no LangChain, LlamaIndex,
  vector DBs, web UI, or Docker yet).
