# AI-Enhanced Trading Bot — Project Description

## Overview

This project is an advanced algorithmic trading system that combines:

- Rule-based technical trading (Oliver Kell strategy)
- Machine learning probability modeling
- Reinforcement learning for parameter optimization
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

The system consists of three main layers:

### 1. Signal Generation Layer
Implements the deterministic strategy:
- Base 'n Break
- Wedge Pop
- EMA-based exits
- ATR-based volatility logic

This layer produces raw trade signals.

---

### 2. AI Probability Layer

A machine learning model evaluates each signal and outputs:

- Probability of success
- Expected return (R-multiple)
- Confidence score

Only high-probability trades are executed.

---

### 3. Reinforcement Learning Layer

Continuously optimizes strategy parameters such as:

- EMA periods
- ATR multipliers
- Stop-loss distances
- Volume thresholds

Objective:
- Maximize Sharpe ratio
- Improve profit factor
- Stabilize equity curve

---

## Key Features

### Probabilistic Trade Filtering
Trades are executed only if:
P(win | context) > threshold

---

### Adaptive Position Sizing

Risk per trade is dynamically adjusted:
position_size ∝ model_confidence

---

### Self-Learning Loop

1. Execute trades
2. Log results
3. Update dataset
4. Retrain model
5. Deploy improved model

---

### Market Regime Detection (Optional)

Using a local AI model:

- Detect trend vs range environments
- Identify volatility conditions
- Incorporate macro sentiment

---

## Technology Stack

- Python
- NumPy / Pandas
- LightGBM / XGBoost
- PyTorch (optional for LSTM)
- cTrader Open API
- Local LLM runtime (optional)

---

## Data Pipeline
Market Data
→ Indicator Engine
→ Feature Extraction
→ Model Prediction
→ Trade Execution
→ Trade Logging
→ Model Retraining


---

## Backtesting & Evaluation

Performance metrics include:

- Win rate
- Profit factor
- Sharpe ratio
- Maximum drawdown
- Expectancy per trade

---

## Future Enhancements

- Multi-asset portfolio optimization
- Bayesian probability updates
- Online learning (real-time updates)
- Genetic algorithm parameter search
- Distributed training pipeline

---

## Disclaimer

This project is for research and educational purposes only.

Trading involves significant financial risk. No guarantee of profitability is provided.
