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

## 3. Wedge Drop (Pre-Continuation Compression)

Temporary pullback pattern.

Signals:

```
trend pause
not reversal
potential upcoming Base 'n Break
```

No direct entry.

Context signal only.

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

Before executing any trade:

Model computes:

```
P(win | features)
```

Trade executes only if:

```
P(win) > threshold
```

Typical threshold:

```
0.60 – 0.70
```

Example:

```
P = 0.72 → trade allowed
P = 0.54 → trade rejected
```

Purpose:

Reduce low-quality breakout entries.

---

# Confidence-Weighted Position Sizing

Instead of fixed risk:

```
risk = base_risk × confidence_score
```

Example:

| Confidence | Risk |
| ---------- | ---- |
| 0.80       | 1.2% |
| 0.60       | 0.8% |
| 0.50       | 0.5% |

Effect:

```
strong signals = larger allocation
weak signals = smaller allocation
```

Improves equity curve stability.

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

# Feature Set for AI Model

## Market Structure Features

```
EMA slope
EMA distance ratio
ATR expansion ratio
volatility compression score
trend strength index
```

---

## Signal Features

```
base width
base tightness
breakout velocity
volume expansion ratio
pattern classification
```

---

## Context Features

```
multi-timeframe trend alignment
volatility regime classification
session timing (Asia / London / NY)
trend maturity stage
```

Optional:

local regime detection via **Ollama**

---

# Market Regime Awareness

Strategy performs best in:

```
trend environments
momentum expansion cycles
relative strength leaders
```

Avoids:

```
sideways compression markets
low-volume environments
news-driven volatility spikes
```

---

# Trade Lifecycle

Execution pipeline:

```
Detect signal
→ Extract features
→ Compute probability
→ Validate trade
→ Calculate position size
→ Execute order
→ Manage trailing stop
→ Monitor exits
→ Log trade result
→ Retrain model
```

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

# Optimization Targets (AI-Controlled)

Parameters dynamically tuned:

```
EMA_FAST
EMA_SLOW
ATR multipliers
volume thresholds
probability threshold
trailing stop multiplier
base detection length
```

Objective:

```
maximize Sharpe ratio
maximize expectancy
stabilize drawdown
```

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
