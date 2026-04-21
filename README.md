# 33 AI cTrader Bot

An AI-enhanced algorithmic trading system combining momentum breakout strategy with machine learning probability filtering.

## Overview

A hybrid trading system that integrates:
- Classical momentum breakout patterns (Base 'n Break, Wedge Pop)
- EMA trend alignment and volume confirmation
- AI-powered trade filtering and probability assessment
- Adaptive position sizing based on confidence scores
- Self-learning architecture with continuous model improvement

## Core Components

### Signal Layer
- EMA 10/20 trend confirmation
- ATR-based volatility measurement
- Volume breakout validation
- Pattern detection (tight bases, wedge formations)

### AI Probability Filter
Each signal is evaluated before execution:
- Probability of success computed from features
- Only trades with P(win) > threshold execute
- Reduces low-quality breakout entries

### Risk Management
- 1-2% risk per trade (adaptive by confidence)
- Trailing stops below swing lows/EMA10
- Position size = (Account x Risk%) / (Entry - Stop)

## Indicators

| Indicator | Purpose |
|-----------|--------|
| EMA 10 | Short-term trend support |
| EMA 20 | Primary trend confirmation |
| ATR (14) | Volatility measurement |
| Volume | Breakout confirmation |

## Entry Patterns

**Base 'n Break** (Primary)
- 3-15 candle consolidation <= 0.5x ATR
- Price above EMA 20
- Breakout with 1.5x average volume

**Wedge Pop** (Continuation)
- 5-bar compression
- Structure above EMA 10
- Volume breakout confirmation

## Exit Triggers

- Exhaustion Extension: distance from EMA20 > 3x ATR
- EMA Crossback: close < EMA10 AND close < EMA20
- Trailing Stop: price - (1.5 x ATR)

## Performance Targets

| Metric | Target |
|--------|--------|
| Win rate | 60-70% |
| Profit factor | > 1.8 |
| Max drawdown | < 15% |

## Technology Stack

- Python
- NumPy / Pandas
- LightGBM / XGBoost
- cTrader Open API

## Files

-  - Main trading bot implementation
-  - Detailed strategy specification
-  - System architecture documentation
-  - Configuration template

## Disclaimer

For research and educational purposes only. Trading involves significant financial risk.
