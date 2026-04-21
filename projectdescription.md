# AI-Enhanced Trading Bot — Project Description

## Overview

This project is an advanced algorithmic trading system that combines:

- Rule-based technical trading (Base 'n Break, Wedge Pop, **Wedge Drop context**)
- Machine learning probability modeling (12-feature classifier)
- Hybrid parameter optimization (random-search RL-inspired tuner)
- Optional local AI (LLM) integration for market regime analysis

The system is designed to evolve into a **self-learning, adaptive trading engine** capable of improving performance over time.

---

## Core Objectives

- Increase win rate using probabilistic filtering
- Reduce drawdowns through adaptive risk management
- Automatically optimize strategy parameters
- Detect and adapt to changing market regimes
- Maintain full transparency and explainability of trades

---

## System Architecture

### Layered Design

```
 ┌─────────────────────────────────────────────┐
 │   TradingBot (orchestrator)                 │  ← bot.py
 │   — wires core → execution                  │
 ├─────────────────────────────────────────────┤
 │   Core Layer (pure logic, no I/O)           │  ← src/core/
 │   — patterns, features, regime, ai, risk    │
 ├─────────────────────────────────────────────┤
 │   Execution Layer (transport)               │  ← src/execution/
 │   — BacktestExecutor, LiveExecutor          │
 └─────────────────────────────────────────────┘
```

### 1. Signal Generation Layer (`src/core/patterns.py`)
Deterministic pattern detection:
- **Base 'n Break** — tight consolidation breakout
- **Wedge Pop** — continuation pullback breakout
- **Wedge Drop** — context-only pullback signal
- **Exhaustion Extension** — exit on vertical move
- **EMA Crossback** — exit on trend failure

### 2. AI Probability Layer (`src/core/ai_filter.py`)
- 12-feature classifier (GradientBoosting / MLP)
- Predicts `P(win)`, filters low-probability trades
- Continuous confidence→risk multiplier

### 3. Market Regime Layer (`src/core/regime.py`)
- Primary: volatility ratio + trend strength detector
- Optional LLM overlay: **Ollama** (CPU) or **LightLLM** (GPU-accelerated)
- Blocks entries in RANGING/VOLATILE/UNKNOWN

### 4. Risk Management Layer (`src/core/risk.py`)
- Position sizing: `(balance × risk_pct) / |entry − stop|`
- Trailing stop: `price − (TRAIL_ATR_MULT × ATR)`, monotonic up-only

### 5. Execution Layer (`src/execution/`)
- **BacktestExecutor** — runs `TradingBot.on_bar()` over DataFrame
- **LiveExecutor** — connects to cTrader Open API, auto-reconnect on drop

---

### 2. AI Probability Layer

A machine learning model evaluates each signal and outputs:

- Probability of success
- Expected return (R-multiple)
- Confidence score

Only high-probability trades are executed.

---

### 3. Reinforcement Learning Layer (Offline Tuner)

An **offline optimizer** performs batch hyperparameter sweeps using random search and backtest evaluation.

**Tunable parameters:**
- EMA periods, ATR multipliers, trailing stop multiplier
- Volume surge threshold, base width ratio, probability threshold

**Objective:** Maximize composite score: Sharpe + profit factor + win rate − drawdown penalty.

**Usage:** `python aitradingbot.py optimize`

Note: True online RL (bandit/actor-critic) is a future enhancement.

---

## Key Features

### Probabilistic Trade Filtering
Trades are executed only if:
P(win | context) > threshold

---

### Adaptive Position Sizing (Continuous)

Risk is a smooth function of confidence:
```
risk_mult = 0.2 + √confidence × 1.3  ∈ [0.2, 1.5]
```
Stronger signals get larger allocations; weak signals minimized but not censored.

---

### Market Regime Filtering

Trades are blocked in RANGING, VOLATILE, or UNKNOWN regimes. Prevents overtrading in low-edge conditions.

---

### Context Features

Multi-timeframe trend alignment (short/mid/long EMA stack)  
Session timing indicators (Asia/London/NY session flags)  
Trend maturity counters (consecutive trend-bar streaks)

---

### Self-Learning Loop

1. Execute trades
2. Log results
3. Update dataset
4. Retrain model
5. Deploy improved model

---

### Market Regime Detection (Optional)

Two-tier regime classifier:

**Primary:** Volatility- and trend-strength-based detector (always on)  
**Optional:** Local LLM overlay via Ollama — validates or overrides primary

Classifies into: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, UNKNOWN

---

## Technology Stack

- **Python** 3.10+
- **NumPy / pandas** — data structures
- **scikit-learn** — ML (GradientBoostingClassifier, MLPClassifier)
- **ctrader-open-api** + **Twisted** — live trading connectivity
- **lightllm** (optional) — fast local LLM inference (GPU)
- **ollama** (optional) — easy local LLM (CPU)
- **pytest** — test suite

---

## Data Pipeline

```
Market Data (DataFrame / cTrader stream)
     ↓
Indicator Engine (EMA, ATR, volume)
     ↓
Feature Extraction (12 features: structure + context)
     ↓
Model Prediction (P(win))
     ↓
Trade Validation (threshold + regime filter)
     ↓
Position Sizing (confidence-scaled risk)
     ↓
Execution (backtest or live order)
     ↓
Trade Logging + Outcome capture
     ↓
Model Retraining (after ≥10 closed trades)
```

---

## Backtesting & Evaluation

**Metrics tracked:** win rate, profit factor, Sharpe ratio, max drawdown, expectancy.

**Run backtest:**
```bash
python main.py
```

**Custom script:**
```python
from src.bot import TradingBot
import pandas as pd

bot = TradingBot()
df = pd.read_csv("my_ohlcv.csv")
trades = bot.run_backtest(df)
```

---

## Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure credentials**
   ```bash
   cp config.ini.example config.ini
   # Edit config.ini with your cTrader Open API credentials
   ```

3. **Run backtest (demo data)**
   ```bash
   python main.py
   ```

4. **Run unit tests**
   ```bash
   pytest           # all tests
   pytest -v        # verbose
   pytest tests/test_patterns.py -v  # single module
   ```

5. **Live trading** (requires funded cTrader account + API access)
   ```bash
   python main.py live
   ```

6. **Optional LLM regime detection**
   - **Ollama** (easier): `ollama pull llama3.2` and set `backend = ollama` in `[llm]`
   - **LightLLM** (faster, GPU): `pip install lightllm` and set `backend = lightllm`, `lightllm_model_path = /path/to/model`

---

## Future Enhancements

- **Online/incremental learning** — real-time model updates without full retrain
- **Expected return (R-multiple) head** — regress target R-multiple alongside win probability
- **Secondary entry on retest** — wedge-pop retest entries
- **Partial scaling exits** — tiered exits (1×, 2×, 3× risk)
- **True online RL** — bandit/actor-critic for live parameter adaptation
- **Pattern classification feature** — encode detected pattern as categorical feature
- **Volatility compression score** — explicit coiling metric
- **Trend-strength index** — ADX-like composite (EMA slope + ATR)
- **Genetic algorithm search** — optimizer alternative to random search
- **Multi-asset portfolio** — position sizing across symbols with correlation awareness
- **Distributed training** — parallelized parameter sweep
- **LightLLM integration** — GPU-accelerated regime classification (implemented)

---

## Disclaimer

This project is for research and educational purposes only.

Trading involves significant financial risk. No guarantee of profitability is provided.
