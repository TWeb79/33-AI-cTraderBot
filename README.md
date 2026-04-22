# 33 AI cTrader Bot

AI-enhanced algorithmic trading system combining momentum breakout strategy with machine learning probability filtering.

## Architecture

```
src/
├── core/           # Pure strategy logic (no I/O)
│   ├── types.py           — Bar, TradeState, Signal enum
│   ├── indicators.py      — EMA, ATR, volume avg
│   ├── patterns.py        — Base 'n Break, Wedge Pop, Wedge Drop detection
│   ├── features.py        — MTF alignment, sessions, trend maturity
│   ├── regime.py          — MarketRegimeDetector, LLMRegimeDetector
│   ├── ai_filter.py       — AITradeFilter (ML classifier, confidence→risk)
│   └── risk.py            — RiskManager (position size, trailing stop)
├── execution/      # Transport layer (I/O)
│   ├── base.py            — IMarketDataHandler, IExecutionHandler interfaces
│   ├── backtest_executor.py — DataFrame-driven backtester
│   └── live_executor.py   — cTrader Open API client with auto-reconnect
└── bot.py          — TradingBot orchestrator (wires core → execution)
```

Optional `src/core/lightllm_regime.py` — LightLLM backend for fast LLM inference.

## Overview

Hybrid system integrating:
- Classical momentum breakout patterns (Base 'n Break, Wedge Pop, Wedge Drop)
- EMA trend alignment and volume confirmation
- AI-powered trade filtering and probability assessment
- Adaptive position sizing (continuous confidence scaling)
- Self-learning architecture with continuous model improvement
- Market regime filtering (avoid RANGING/VOLATILE)
- Optional LLM regime overlay (Ollama or LightLLM)

## Core Components

### Signal Layer
- EMA 10/20 trend confirmation
- ATR-based volatility measurement
- Volume breakout validation
- Pattern detection (tight bases, wedge formations, wedge drop context)

### AI Probability Filter
Each signal is evaluated before execution. The repository currently provides a compact feature set and supports both Gradient Boosting (default) and an MLP (neural) classifier via `MODEL_TYPE` in `config.ini`.

- Probability from an ML model (Gradient Boosting or MLP)
- Only trades with `P(win)` > threshold (default 0.65) execute
- Confidence-weighted position sizing (base risk scaled by confidence)

Note: The running monolithic bot (aitradingbot.py) and the modular `src/` implementation may expose different feature sets. See "AI Model Features" below for what's implemented now and planned enhancements.

### Market Regime Filter
- Real-time regime classification: TRENDING_UP / TRENDING_DOWN / RANGING / VOLATILE / UNKNOWN
- Primary detector: volatility ratio + trend strength (always on)
- Optional LLM overlay: Ollama (default) or **LightLLM** (faster, GPU recommended)
- Trades blocked in RANGING, VOLATILE, UNKNOWN

### Risk Management
- Base risk 1–2% per trade, scaled continuously by confidence
- Trailing stop: `price − (TRAIL_ATR_MULT × ATR)` (never moves down)
- Position size formula: `(account × risk_pct) / |entry − stop|`

### Parameter Optimizer (Offline)
- CLI: `python aitradingbot.py optimize`
- Random-search over EMA periods, ATR multipliers, thresholds
- Objective: Sharpe + profit factor − drawdown penalty

## AI Model Features (implemented vs planned)

Implemented (core features used by the current ML filter):

| Feature | Description |
|---------|-------------|
| `ema_slope` | EMA fast slope over ~5 bars |
| `ema_distance_ratio` | (close − ema_slow) / atr |
| `atr_expansion` | atr / mean(atr[10]) |
| `base_tightness` | (base_high − base_low) / atr |
| `breakout_velocity` | (close − base_low) / atr |
| `volume_ratio` | volume / avg_volume |

Planned / documented (not yet implemented in the compact filter):

- `mtf_alignment` (multi-timeframe EMA alignment)
- `trend_maturity` (consecutive trend bar count)
- `regime_score` (LLM or volatility-based regime indicator)
- session flags (`asia_session`, `london_session`, `ny_session`)

These additional features are outlined in tradingstrategy.md and projectdescription.md and are planned for the modular `src/` implementation.

## Entry Patterns

**Base 'n Break** (Primary)
- 3–15 candle tight consolidation (`range ≤ 0.5× ATR`)
- Price > EMA 20
- Breakout close above base high with `volume ≥ 1.5× average`

**Wedge Pop** (Continuation)
- 5-bar compression (lower highs + lower lows)
- Price > EMA 10
- Breakout above wedge high with volume confirmation

**Wedge Drop** (Context only)
- 5-bar pullback (lower highs + lower lows)
- Maintains > EMA 10
- Logs trend pause; no entry generated

## Exit Triggers

| Trigger | Condition | Action |
|---------|-----------|--------|
| Exhaustion Extension | `(close − ema20) > 3 × ATR` | Scale out or exit |
| EMA Crossback | `close < EMA10 AND close < EMA20` | Exit immediately |
| Trailing Stop | `price − (TRAIL_ATR_MULT × ATR)` | Stop rises with price; never falls |

## Performance Targets

| Metric | Target |
|--------|--------|
| Win rate | 60–70% |
| Profit factor | > 1.8 |
| Max drawdown | < 15% |
| Sharpe | improving over time |

## Technology Stack

- **Python** 3.10+
- **NumPy / pandas** — data handling
- **scikit-learn** — ML classifier (Gradient Boosting / MLP)
- **ctrader-open-api** + **Twisted** — live trading
- **lightllm** (optional) — fast local LLM inference (GPU)
- **ollama** (optional) — easy local LLM (CPU)

## Quick Start

### Installation

```bash
# Clone repo
git clone <your-repo>
cd 33-AI-cTraderBot

# Install dependencies
pip install -r requirements.txt
```

For **LightLLM** (faster regime LLM):
```bash
pip install lightllm
# Requires CUDA GPU and model downloaded separately
```

For **Ollama** (easier):
```bash
# Install from https://ollama.ai, then:
ollama pull llama3.2
```
## Configuration

Create `config.ini` from template:

```bash
cp config.ini.example config.ini
```

Edit `config.ini`:

- `[ctrader]` section — fill API credentials from https://openapi.ctrader.com/
- `[ai]` — `MODEL_TYPE = gb` (GradientBoosting) or `mlp` (MLP neural classifier). The code will load a saved model from `ai_trade_model.pkl` if present, otherwise it instantiates a default model of the chosen type.
- `[llm]` — `enabled = true/false`, `backend = ollama|lightllm`, `model = llama3.2`

You can also configure base numeric strategy parameters (e.g., `PROBABILITY_THRESHOLD`, `RISK_PERCENT`, `BASE_ATR_RATIO`) under a `[strategy]` section in `config.ini` if desired.
- `[llm]` — `enabled = true/false`, `backend = ollama|lightllm`, `model = llama3.2`

### Running the Bot

**Backtest** (demo data):
```bash
python main.py
```

**Backtest** (custom CSV/DataFrame — edit `main.py` or write custom script):
```python
from src.bot import TradingBot
import pandas as pd

bot = TradingBot()
df = pd.read_csv("my_data.csv")
trades = bot.run_backtest(df)
```

**Live trading** (requires cTrader credentials):
```bash
python main.py live
```

The new modular `TradingBot` class provides clean programmatic access for scripted backtests.

### Running Tests

```bash
# All unit + integration tests
pytest

# Specific module
pytest tests/test_patterns.py -v

# With coverage report (optional)
pip install pytest-cov
pytest --cov=src
```

All tests should pass (34+). LLM-dependent tests are skipped if `ollama` not installed.

## Optimizer / Auto-configuration

This repository includes a small optimizer tool that searches for good strategy parameter configurations using historical data.

- Location: tools/optimizer.py
- Modes:
  - brute — random/grid sampling of parameter combinations
  - ml    — ML-guided search (surrogate model with GradientBoosting)

Usage examples:

```bash
# Brute-force random search (n_samples random configs)
python tools/optimizer.py --mode brute --n_samples 100

# ML-guided search (initial random samples, then surrogate-guided candidates)
python tools/optimizer.py --mode ml --initial 50 --candidates 200 --rounds 3
```

Outputs:

- data/best_config.json — best parameter set found by the optimizer
- analysis artifacts (if any signals are found): data/analysis_confidences_<ts>.csv and data/analysis_confidence_hist_<ts>.png

Notes:

- The optimizer evaluates candidate configurations by running the modular backtester on the loaded historical dataset (data/US500_H1.json by default) and measuring total PnL / win rate / profit factor. If the dataset produces no trades under the current detection rules, many candidates may report zero PnL. Consider expanding the dataset or relaxing thresholds if optimization returns trivial results.
- The ML-guided search trains a simple surrogate model on initial evaluations to propose promising candidate configurations; it does not guarantee global optimum but can find improvements faster than blind search.

## Configuration Reference

| Config Key | Section | Default | Description |
|------------|---------|---------|-------------|
| `CLIENT_ID` | `[ctrader]` | — | cTrader Open API client ID |
| `CLIENT_SECRET` | `[ctrader]` | — | Client secret |
| `ACCESS_TOKEN` | `[ctrader]` | — | Access token |
| `ACCOUNT_ID` | `[ctrader]` | 123456 | Numeric account ID |
| `MODEL_TYPE` | `[ai]` | `gb` | `gb` (GradientBoosting) or `mlp` (MLP neural classifier) |
| `PROBABILITY_THRESHOLD` | `[strategy]` | `0.65` | AI filter threshold |
| `RISK_PERCENT` | `[strategy]` | `1.5` | Base risk per trade (%) |
| `EXHAUSTION_ATR_MULT` | `[strategy]` | `3.0` | Exhaustion distance |
| `BASE_ATR_RATIO` | `[strategy]` | `0.5` | Max base width / ATR |
| `VOLUME_SURGE_MULT` | `[strategy]` | `1.5` | Breakout volume multiplier |
| `TRAIL_ATR_MULT` | `[strategy]` | `1.5` | Trailing stop ATR multiplier |

## Project Structure

```
33-AI-cTraderBot/
├── src/
│   ├── __init__.py
│   ├── bot.py                # TradingBot orchestrator
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py                — domain types (Bar, TradeState, Signal)
│   │   ├── indicators.py           — EMA, ATR, volume window
│   │   ├── patterns.py             — pattern detector (3 patterns + exits)
│   │   ├── features.py             — context features (MTF, sessions, maturity)
│   │   ├── regime.py               — volatility + optional LLM regime detector
│   │   ├── ai_filter.py            — ML model + continuous risk scaling
│   │   ├── risk.py                 — position sizing & trailing stop
│   │   └── lightllm_regime.py      — LightLLM backend (optional)
│   └── execution/
│       ├── __init__.py
│       ├── base.py                 — abstract interfaces (IMarketDataHandler, IExecutionHandler)
│       ├── backtest_executor.py    — DataFrame backtest runner
│       └── live_executor.py        — cTrader Open API client with reconnection
├── tests/
│   ├── conftest.py           — pytest fixtures
│   ├── test_indicators.py
│   ├── test_patterns.py
│   ├── test_features.py
│   ├── test_regime.py
│   ├── test_ai_filter.py
│   ├── test_risk.py
│   └── test_backtest.py
├── fixtures/                — (optional) static test data files
├── main.py                  # CLI entry point (backtest/live)
├── aitradingbot.py          # [DEPRECATED] legacy monolithic version; use main.py
├── config.ini.example       — configuration template
├── requirements.txt         — Python dependencies
├── README.md                — this file
├── tradingstrategy.md       — detailed strategy specification
└── projectdescription.md    — system architecture & future roadmap
```

## Optional LLM Regime Detection

For LLM-based regime classification you can use either:

### Ollama (CPU-friendly)
```bash
# Install from https://ollama.ai
ollama pull llama3.2
# config.ini: [llm] enabled=true, backend=ollama, model=llama3.2
```

### LightLLM (GPU-accelerated, faster)
```bash
pip install lightllm
# Download a compatible model checkpoint (follow LightLLM docs)
# config.ini: [llm] enabled=true, backend=lightllm, lightllm_model_path=/path/to/model
```
LightLLM provides ~10–50× speedup over Ollama on GPU.

---

## Disclaimer

For research and educational purposes only. Trading involves significant financial risk. No guarantee of profitability is provided.
