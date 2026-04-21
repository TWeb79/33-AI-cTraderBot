Here is your **merged, production-ready strategy specification** as a single coherent **strategy.md** document suitable for a repo, whitepaper, or quant system documentation. I unified duplicated sections, aligned terminology, integrated AI logic into the classical structure, and upgraded it into a **hybrid deterministic + probabilistic execution model**.

---

# 📄 `strategy.md`

# AI Momentum Breakout Strategy

## Strategy Overview

The ** AI Momentum Breakout Strategy** is a hybrid trading system combining:

* classical momentum breakout structure
* EMA trend alignment
* volatility-aware exits
* volume-validated continuation entries
* probabilistic AI trade filtering
* confidence-weighted position sizing
* adaptive parameter optimization

The strategy trades **trend-aligned consolidation breakouts** and exits during **trend exhaustion or structure failure**.

It is designed as a **self-learning probability-driven execution framework**, not a purely predictive model.

---

# Strategy Philosophy

Core principle:

```
Trade strong trends.
Enter tight consolidations.
Exit strength expansions.
Protect capital aggressively.
Filter trades probabilistically.
```

Focus:

```
Quality > Quantity
```

---

# Core Indicators

| Indicator    | Purpose                    |
| ------------ | -------------------------- |
| EMA 10       | Short-term trend support   |
| EMA 20       | Primary trend confirmation |
| ATR (14)     | Volatility measurement     |
| Volume       | Breakout confirmation      |
| EMA distance | Exhaustion detection       |

Rule:

Price must remain **above EMA 10 and EMA 20** for long entries.

---

# Strategy Architecture

The system operates in three layers:

```
Pattern Engine
→ AI Probability Filter
→ Adaptive Risk Engine
```

---

# Key Price-Action Patterns

## 1. Base 'n Break (Primary Entry)

A tight consolidation forming above EMA structure followed by volume-confirmed breakout.

Conditions:

* Base length: 3–15 candles
* Base range ≤ 0.5 × ATR
* Price above EMA 20
* Breakout above base high
* Volume ≥ 1.5 × average volume

Entry:

Enter on breakout close.

Optional secondary entry:

Retest of breakout level.

---

## 2. Wedge Pop (Continuation Entry)

Continuation breakout after compressed pullback.

Conditions:

* 5-bar compression
* Lower highs + lower lows
* Structure above EMA 10
* Breakout above wedge high
* Volume confirmation

Entry:

Enter breakout candle close.

---

## 3. Wedge Drop (Context-Only Signal)

Temporary pullback forming lower highs and lower lows while maintaining structure above EMA 10.

**Conditions:**

* 5-bar compression with declining highs/lows
* Price stays above EMA 10 (no close below)
* No breakout — merely a pullback formation

**Interpretation:**

```
trend pause
≠ reversal
potential upcoming Base 'n Break or Wedge Pop continuation
```

**Action:** Context signal only. No direct entry. Logged for regime context and potential continuation setup anticipation.

---

## 4. Exhaustion Extension (Primary Exit Zone)

Occurs when price accelerates vertically away from EMA structure.

Condition:

```
distance_from_EMA20 > 3 × ATR
```

Action:

Scale out or exit position.

---

## 5. EMA Crossback (Trend Failure Exit)

Exit trigger when:

```
close < EMA10
AND
close < EMA20
```

Confirms trend weakening.

Exit immediately.

---

## 6. Reversal Extension (Trend Termination)

Sharp move opposite primary trend following exhaustion.

Signals:

```
trend cycle complete
prepare for new setup cycle
```

Exit remaining exposure.

---

# Trade Cycle Model

Typical trend progression:

```
Base 'n Break
→ Trend Expansion
→ Exhaustion Extension
→ EMA Crossback
→ Base Reset
→ Second Base 'n Break
→ Wedge Pop
→ Final Reversal Extension
```

Strategy participates mainly in:

```
Expansion phases
```

Avoids:

```
distribution phases
```

---

# Entry Rules

## Primary Entry: Base 'n Break

Requirements:

```
price > EMA10
price > EMA20
tight base forms
volume expansion
breakout close above structure
```

Stop loss:

Below base low.

---

## Secondary Entry: Wedge Pop

Requirements:

```
compression pullback
structure above EMA10
volume breakout confirmation
```

Stop loss:

Below wedge structure low.

---

# Exit Rules

## Exit Trigger 1 — Exhaustion Extension

Condition:

```
distance_from_EMA20 > 3 × ATR
```

Action:

Scale out or fully exit.

---

## Exit Trigger 2 — EMA Crossback

Condition:

```
close < EMA10
AND
close < EMA20
```

Action:

Exit immediately.

---

## Exit Trigger 3 — Trailing Stop

Trailing rule:

```
stop = price − (1.5 × ATR)
```

Constraint:

Stop never moves downward.

---

# AI Probability Filter

Before executing any trade, the model computes `P(win | features_t)` using a Gradient Boosting or MLP classifier.

**Execution gate:**

```
trade_allowed ⇔ P(win) > threshold
```

Default threshold: `0.65` (configurable).

**Rejection:** Trades with insufficient confidence are skipped entirely (no position opened).

**Model input:** 12-dimensional feature vector (market structure + context features).

**Model output:** Probability score `[0–1]`.

**Purpose:** Reduce low-quality breakout entries; filter out low-edge regime conditions.

**Retraining:** After every 10+ closed trades, the model is retrained incrementally on the latest outcome-labeled dataset (`_retrain_model`).

---

# Confidence-Weighted Position Sizing (Continuous)

Risk is a continuous function of model confidence, not discrete buckets.

**Formula:**

```
risk_multiplier = 0.2 + (confidence^0.5) × 1.3
final_risk = base_risk × risk_multiplier
```

Clamped to range `[0.2, 1.5]`.

**Scaling behavior:**

| Confidence | Risk Multiplier |
| ---------- | --------------- |
| 0.50       | ~0.82          |
| 0.60       | ~0.95          |
| 0.70       | ~1.08          |
| 0.80       | ~1.24          |
| 0.90       | ~1.44          |

**Effect:** Stronger AI confidence → larger allocation; weak signals are minimized but not eliminated. Implemented in `AITradeFilter.get_confidence_risk()`.

---

# Position Size Formula

```
Position Size =
(Account × Risk%)
÷
(Entry − Stop Loss)
```

Example:

```
Account = $10,000
Risk = 1%
Entry = 50
Stop = 48

Position size = 50 units
```

---

# Feature Set for AI Model (12 features)

Features are extracted per bar at signal time and fed to the ML classifier.

## Market Structure Features (5)

| Feature | Formula | Purpose |
| ------- | --------| ------- |
| `ema_slope` | `(ema_fast_t − ema_fast_{t−5}) / 5` | Short-term trend momentum |
| `ema_distance_ratio` | `(close − ema_slow) / atr` | Normalized distance from slow EMA |
| `atr_expansion` | `atr_t / mean(atr_{t−10:t})` | Volatility regime expansion |
| `base_tightness` | `(base_high − base_low) / atr` | Consolidation tightness (lower = tighter) |
| `breakout_velocity` | `(close − base_low) / atr` | Breakout strength normalized |

## Context Features (6)

| Feature | Type | Purpose |
| ------- | ---- | ------- |
| `mtf_alignment` | float ∈ {−1,0,1} | Multi-timeframe trend alignment: short/mid/long EMA stack |
| `trend_maturity` | int ≥ 0 | Number of consecutive trend bars reinforcing current direction |
| `asia_session` | 0/1 | Hour ∈ [0,8) |
| `london_session` | 0/1 | Hour ∈ [8,16) |
| `ny_session` | 0/1 | Hour ∈ [13,21) |
| `regime_score` | float ∈ {−1,0,1} | Regime numeric score: TRENDING_UP=1, TRENDING_DOWN=−1, else 0 |

## Implementation

Features computed in `PatternDetector.extract_features()` and `FeatureEngine.extract_context_features()`.

---

# Market Regime Detection & Filtering

Real-time regime classification determines whether the strategy participates or stays in cash.

## Regime Types

| Regime | Detection Method | Characteristics | Action |
| ------ | ---------------- | --------------- | ------ |
| TRENDING_UP | EMA slope + ATR volatility + EMA cross alignment | Positive slope, low volatility expansion, EMA bullish cross | **Trade** |
| TRENDING_DOWN | EMA slope + EMA bearish cross | Negative slope, expanding range, EMA bearish cross | Avoid (only long) |
| RANGING | Low trend strength, high compression | Price range-bound, low ATR expansion | **Avoid** |
| VOLATILE | ATR > 2× its 10-bar moving average | News spikes, erratic swings | **Avoid** |
| UNKNOWN | Insufficient data | Cold start or sparse history | **Avoid** |

## Detection Methods

**Primary:** `MarketRegimeDetector` (volatility + trend-strength based)
**Optional:** `LLMRegimeDetector` via local Ollama — provides LLM-based classification when enabled in `config.ini`.

Both are queried via `PatternDetector.get_regime()`; LLM overrides if available, otherwise falls back to volatility-based detector.

## Integration

All `BASE_BREAK` and `WEDGE_POP` entries are gated by regime:

```
if regime ∈ {RANGING, VOLATILE, UNKNOWN}:
    skip trade
```

---

# Trade Lifecycle

Execution pipeline:

```
1  SCAN          ← pattern detection (Base 'n Break, Wedge Pop, Wedge Drop)
2  EXTRACT       ← 12-feature vector (structure + context)
3  REGIME CHECK  ← skip if regime ∈ {RANGING, VOLATILE, UNKNOWN}
4  PROBABILITY   ← P(win) = model.predict_proba(features)
5  VALIDATE      ← P(win) > threshold ?
6  POSITION SIZING ← risk = base_risk × continuous_risk_multiplier(confidence)
7  EXECUTE       ← market order + initial stop
8  TRAILING      ← stop = price − (TRAIL_ATR_MULT × ATR); never moves down
9  EXIT          ← Exhaustion Extension or EMA Crossback
10 LOG + RETRAIN ← append trade outcome; retrain after ≥10 trades
```

**Context-only (`WEDGE_DROP`):** Logged but no entry generated.

---

# Risk Management Rules

Base rule:

```
risk per trade = 1% – 2%
```

Adaptive rule:

```
risk scaled by confidence score
```

Stop placement:

```
below breakout base
```

Trailing logic:

```
below swing lows
or
below EMA10
```

Never increase risk after entry.

---

# Performance Targets

| Metric        | Target              |
| ------------- | ------------------- |
| Win rate      | 60–70%              |
| Profit factor | > 1.8               |
| Max drawdown  | < 15%               |
| Expectancy    | positive            |
| Sharpe ratio  | improving over time |

---

# Strengths

```
trend-aligned entries
tight risk control
volume confirmation logic
AI removes weak setups
adaptive position sizing
self-learning architecture
```

---

# Weaknesses

```
requires trending environments
false signals in chop
dependent on data quality
model retraining required
```

---

# Parameter Optimization (RL-Inspired Hybrid Tuner)

An optional offline `ParameterOptimizer` performs random-search backtest sweeps to find robust hyperparameters.

## Tunable Parameters

- `EMA_FAST`, `EMA_SLOW` — trend baseline periods
- `EXHAUSTION_ATR_MULT` — exhaustion distance threshold (default 3.0)
- `BASE_ATR_RATIO` — maximum base consolidation width as ATR fraction (default 0.5)
- `VOLUME_SURGE_MULT` — breakout volume multiplier (default 1.5)
- `TRAIL_ATR_MULT` — trailing stop ATR multiplier (default 1.5)
- `PROBABILITY_THRESHOLD` — AI filter threshold (default 0.65)

`BASE_MIN_BARS` and `BASE_MAX_BARS` remain fixed at 3–15.

## Objective Function

```
score = 0.4 × profit_factor
      + 0.4 × sharpe_ratio
      + 0.2 × win_rate
      − 10 × max(0, max_drawdown − 0.15)
```

**Usage:** `python aitradingbot.py optimize` runs N iterations and prints the best parameter set.

**Note:** This is a **batch random search** tuner, not online RL. True online RL/bandit-based parameter adjustment is a future enhancement (see Future Enhancements).

---

# Strategy Summary

Execution loop:

```
SCAN
→ strong trend above EMA10 + EMA20

WAIT
→ tight base formation

ENTER
→ breakout with volume confirmation

FILTER
→ AI probability validation

MANAGE
→ trailing stop + scaling logic

EXIT
→ exhaustion extension
or EMA crossback

LOG
→ update learning dataset
```

---

# Final Notes

This strategy is:

```
probability-driven
trend-aligned
risk-controlled
machine-assisted
self-improving
```

Edge sources:

```
selectivity
discipline
structure recognition
volatility awareness
adaptive sizing
AI filtering
```

---

# Operational Reliability

**Automatic reconnection:** On cTrader Open API disconnect, `LiveExecutor._on_disconnected()` schedules `_reconnect()` after 10 seconds via `reactor.callLater`. `_reconnect()` creates a fresh `Client` instance and re-registers all callbacks, then restarts the service. Twisted reactor remains running.

**Graceful degradation:** Missing optional dependencies (`sklearn`, `ctrader-open-api`) trigger fallback modes (default thresholds, simulation mode) rather than hard failures.

**Modular design:** Strategy core (`src/core/`) is decoupled from execution (`src/execution/`), enabling isolated unit testing and clean separation of concerns.

---

# Implementation Reference

| Module | File | Responsibility |
|--------|------|----------------|
| Bar/Signal types | `src/core/types.py` | Domain dataclasses & enums |
| Indicators | `src/core/indicators.py` | EMA, ATR, volume average |
| Pattern detector | `src/core/patterns.py` | Base 'n Break, Wedge Pop, Wedge Drop, exhaust/cross exits |
| Context features | `src/core/features.py` | MTF alignment, sessions, trend maturity |
| Regime detector | `src/core/regime.py` | Volatility-based + LLM overlay (Ollama/LightLLM) |
| AI filter | `src/core/ai_filter.py` | ML classifier, confidence→risk mapping |
| Risk manager | `src/core/risk.py` | Position size, trailing stop |
| Backtest runner | `src/execution/backtest_executor.py` | DataFrame-driven simulation engine |
| Live executor | `src/execution/live_executor.py` | cTrader Open API client with auto-reconnect |
| Orchestrator | `src/bot.py` | TradingBot — wires all components |
| Entry point | `main.py` | CLI: `python main.py` (backtest) or `python main.py live` |
| Legacy monolith | `aitradingbot.py` | Deprecated — use `main.py` + modular architecture |

**LightLLM integration** — `src/core/lightllm_regime.py`: optional fast LLM backend (GPU-accelerated). Configure via `config.ini` (`backend = lightllm`, `lightllm_model_path = …`).
