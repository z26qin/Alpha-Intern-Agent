# Alpha Intern Agent

> Your tireless (and slightly nervous) quant research intern.

**AlphaInternAgent** — "Alpha Intern" for short — is a lightweight,
Hermes-style stock-market machine-learning research agent. It loads
market data, builds features, trains simple signal models, runs
walk-forward backtests, takes notes, and gradually accumulates a library
of reusable research skills.

The long-term vision is an LLM-driven research assistant that can be
asked, *"Investigate momentum in mid-cap industrials,"* and respond with
a reproducible study. This repository is the scaffolding that gets us
there, one small PR at a time.

## Why "Alpha Intern"?

Real quant teams don't ship a hedge fund on day one. They start with
an intern who:

- Pulls clean data.
- Computes a handful of careful features.
- Runs disciplined, walk-forward experiments.
- Writes notes so they don't repeat mistakes.
- Slowly turns repeated workflows into reusable skills.

Alpha Intern is that intern, in code. It is meant to be earnest, modest,
and gradually more useful — not to pretend it has alpha on day one.

## Architecture Overview

```
src/alpha_intern/
├── data/        # load + normalize price data
├── features/    # backward-looking feature engineering
├── models/      # thin sklearn signal-model wrappers
├── backtest/    # walk-forward backtests + metrics
├── memory/      # JSONL research notes + skill registry
├── tools/       # (reserved) tools an LLM agent can call
├── config.py    # project settings
└── cli.py       # Typer CLI
```

Lower layers (`data`, `features`) know nothing about higher layers
(`backtest`, `memory`). An LLM agent will eventually orchestrate these
modules through `tools/`, but this PR keeps the foundation deliberately
boring.

## Quickstart

```bash
git clone https://github.com/z26qin/AlphaInternAgent.git
cd AlphaInternAgent
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
```

## Example CLI

```bash
alpha-intern hello
alpha-intern skills-list
alpha-intern memory-add --title "Momentum in tech" \
    --content "12-1 momentum looks weak in mega-caps since 2023." \
    --ticker AAPL --tags momentum,notes
alpha-intern memory-search "momentum"
```

## Roadmap

- **v0 (this PR)** — package scaffold, features, simple model, simple
  backtest, memory + skill registry, CLI.
- **v0.1** — real walk-forward loop with rolling refits.
- **v0.2** — `tools/` exposes each capability as a callable tool with
  a JSON schema.
- **v0.3** — minimal LLM agent loop that plans → calls tools → writes
  research notes back to memory.
- **v0.4** — richer feature library (cross-sectional, fundamentals).
- **v1.0** — self-improving research workflow: skills get refined from
  past experiments.

## Disclaimer

This project is for **research and educational purposes only**. It is
**not** financial advice. It does not place trades, does not connect to
brokers, and should not be used to make investment decisions. Past
backtest results, especially toy ones, do not predict future returns.
