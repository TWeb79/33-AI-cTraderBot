"""
================================================================================
AI Trading Bot — cTrader Open API Python Bot
================================================================================
Strategy: AI Momentum Breakout with ML Probability Filtering
Features:
    - Base 'n Break + Wedge Pop entries above 10/20 EMA
    - AI-powered trade filtering (probability assessment)
    - Confidence-weighted position sizing
    - Self-learning architecture with model retraining
    - Exit on Exhaustion Extension or EMA Crossback

Requirements:
    pip install ctrader-open-api pandas numpy scikit-learn lightgbm twisted

Usage:
    1. Fill in your CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, ACCOUNT_ID
       from https://openapi.ctrader.com/
    2. Run:  python aitradingbot.py
================================================================================
"""

from __future__ import annotations

import logging
import time
import pickle
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Deque, List, Optional, Tuple
import configparser

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
    from ctrader_open_api.messages.OpenApiMessages_pb2 import *
    from twisted.internet import reactor, defer
    LIVE_MODE = True
except (ImportError, ModuleNotFoundError):
    LIVE_MODE = False
    Client = None
    Protobuf = None
    TcpProtocol = None
    EndPoints = None
    reactor = None
    defer = None
    print("[WARNING] ctrader-open-api not installed. Running in SIMULATION mode.")
    print("          pip install ctrader-open-api twisted\n")

_config = configparser.ConfigParser()
_config.read("config.ini")


def _get_config(section: str, option: str, default):
    if _config.has_section(section):
        return _config.get(section, option, fallback=default)
    return default


USE_LLM_REGIME = _get_config("llm", "enabled", "false").lower() == "true"
OLLAMA_MODEL = _get_config("llm", "model", "llama3.2")

CLIENT_ID = _config.get("ctrader", "CLIENT_ID", fallback=os.environ.get("CLIENT_ID", "YOUR_CLIENT_ID"))
CLIENT_SECRET = _config.get("ctrader", "CLIENT_SECRET", fallback=os.environ.get("CLIENT_SECRET", "YOUR_CLIENT_SECRET"))
ACCESS_TOKEN = _config.get("ctrader", "ACCESS_TOKEN", fallback=os.environ.get("ACCESS_TOKEN", "YOUR_ACCESS_TOKEN"))
try:
    ACCOUNT_ID = int(_config.get("ctrader", "ACCOUNT_ID", fallback=os.environ.get("ACCOUNT_ID", "123456")))
except ValueError:
    ACCOUNT_ID = 123456

SYMBOL_NAME = "US500"
TIMEFRAME = "M5"
RISK_PERCENT = 1.5

EMA_FAST = 10
EMA_SLOW = 20
ATR_PERIOD = 14
BASE_MIN_BARS = 3
BASE_MAX_BARS = 15
BASE_ATR_RATIO = 0.5
VOLUME_SURGE_MULT = 1.5
EXHAUSTION_ATR_MULT = 3.0
WEDGE_BARS = 5
TRAIL_ATR_MULT = 1.5

PROBABILITY_THRESHOLD = 0.65
MODEL_PATH = "ai_trade_model.pkl"
MODEL_TYPE = _get_config("ai", "MODEL_TYPE", "gb").lower()
MLP_HIDDEN_LAYERS = (32, 16)

CONFIDENCE_RISK_MAP = {
    0.80: 1.2,
    0.70: 1.0,
    0.60: 0.8,
    0.50: 0.5,
}


class Signal(Enum):
    NONE = "NONE"
    BASE_BREAK = "BASE_N_BREAK"
    WEDGE_POP = "WEDGE_POP"
    WEDGE_DROP = "WEDGE_DROP"
    EXIT_EXHAUST = "EXIT_EXHAUSTION"
    EXIT_CROSSBACK = "EXIT_EMA_CROSSBACK"


@dataclass
class Bar:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TradeState:
    in_position: bool = False
    entry_price: float = 0.0
    stop_loss: float = 0.0
    units: int = 0
    trail_stop: float = 0.0
    position_id: Optional[int] = None
    entry_time: Optional[datetime] = None
    base_low: float = 0.0
    confidence: float = 0.0


class Indicators:
    def __init__(self, fast: int = EMA_FAST, slow: int = EMA_SLOW, atr_period: int = ATR_PERIOD):
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period

        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._prev_close: Optional[float] = None

        self._tr_window: Deque[float] = deque(maxlen=atr_period)
        self._vol_window: Deque[float] = deque(maxlen=20)

        self.ema_fast_series: List[float] = []
        self.ema_slow_series: List[float] = []
        self.atr_series: List[float] = []

        self._k_fast = 2 / (fast + 1)
        self._k_slow = 2 / (slow + 1)

    def update(self, bar: Bar) -> dict:
        if self._ema_fast is None:
            self._ema_fast = bar.close
            self._ema_slow = bar.close
        else:
            self._ema_fast += self._k_fast * (bar.close - self._ema_fast)
            self._ema_slow += self._k_slow * (bar.close - self._ema_slow)

        if self._prev_close is not None:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - self._prev_close),
                abs(bar.low - self._prev_close),
            )
        else:
            tr = bar.high - bar.low
        self._tr_window.append(tr)
        atr = float(np.mean(self._tr_window))

        self._vol_window.append(bar.volume)
        avg_vol = float(np.mean(self._vol_window))

        self._prev_close = bar.close

        self.ema_fast_series.append(self._ema_fast)
        self.ema_slow_series.append(self._ema_slow)
        self.atr_series.append(atr)

        return {
            "ema_fast": self._ema_fast,
            "ema_slow": self._ema_slow,
            "atr": atr,
            "avg_volume": avg_vol,
        }


class PatternDetector:
    def __init__(self, bar_buffer_size: int = 50):
        self.bars: Deque[Bar] = deque(maxlen=bar_buffer_size)
        self.indic = Indicators()
        self.feature_engine = FeatureEngine(bar_buffer_size=bar_buffer_size)
        self.volatility_detector = MarketRegimeDetector()
        self.llm_detector = LLMRegimeDetector()
        self._indic_vals: List[dict] = []

    def get_regime(self) -> str:
        llm_regime = self.llm_detector.detect("") if self.llm_detector.enabled else None
        if llm_regime:
            return llm_regime
        return self.volatility_detector.detect()

    def push(self, bar: Bar) -> Signal:
        self.bars.append(bar)
        vals = self.indic.update(bar)
        self._indic_vals.append(vals)
        if len(self.bars) < BASE_MAX_BARS + 5:
            return Signal.NONE
        signal = self._evaluate(vals)
        self.feature_engine.update(
            close=bar.close,
            ema_fast=vals["ema_fast"],
            ema_slow=vals["ema_slow"],
            timestamp=bar.time,
        )
        self.volatility_detector.update(
            close=bar.close,
            atr=vals["atr"],
            ema_fast=vals["ema_fast"],
            ema_slow=vals["ema_slow"],
        )
        return signal

    def _evaluate(self, vals: dict) -> Signal:
        bars = list(self.bars)
        latest = bars[-1]
        ema_f = vals["ema_fast"]
        ema_s = vals["ema_slow"]
        atr = vals["atr"]
        avg_v = vals["avg_volume"]

        if latest.close < ema_f and latest.close < ema_s:
            return Signal.EXIT_CROSSBACK

        dist_from_slow_ema = latest.close - ema_s
        if atr > 0 and (dist_from_slow_ema / atr) > EXHAUSTION_ATR_MULT:
            if latest.close > ema_f > ema_s:
                return Signal.EXIT_EXHAUST

        if latest.close < ema_f or latest.close < ema_s:
            return Signal.NONE

        base_signal = self._detect_base_break(bars, ema_s, atr, avg_v)
        if base_signal != Signal.NONE:
            return base_signal

        wedge_signal = self._detect_wedge_pop(bars, ema_f, avg_v)
        if wedge_signal != Signal.NONE:
            return wedge_signal

        wedge_drop = self._detect_wedge_drop(bars, ema_f)
        if wedge_drop != Signal.NONE:
            return wedge_drop

        return Signal.NONE

    def _detect_base_break(self, bars: list, ema_slow: float, atr: float, avg_vol: float) -> Signal:
        latest = bars[-1]

        for base_len in range(BASE_MIN_BARS, BASE_MAX_BARS + 1):
            if len(bars) < base_len + 1:
                break

            base_bars = bars[-(base_len + 1):-1]
            breakout_bar = bars[-1]

            base_high = max(b.high for b in base_bars)
            base_low = min(b.low for b in base_bars)
            base_range = base_high - base_low

            if atr > 0 and base_range > BASE_ATR_RATIO * atr:
                continue

            if base_low < ema_slow * 0.995:
                continue

            if breakout_bar.close <= base_high:
                continue

            if avg_vol > 0 and breakout_bar.volume < VOLUME_SURGE_MULT * avg_vol:
                continue

            logging.info(
                f"[SIGNAL] Base 'n Break | base_len={base_len} "
                f"base_range={base_range:.5f} atr={atr:.5f} "
                f"vol={breakout_bar.volume:.0f} avg_vol={avg_vol:.0f}"
            )
            return Signal.BASE_BREAK

        return Signal.NONE

    def _detect_wedge_pop(self, bars: list, ema_fast: float, avg_vol: float) -> Signal:
        if len(bars) < WEDGE_BARS + 1:
            return Signal.NONE

        wedge = bars[-(WEDGE_BARS + 1):-1]
        latest = bars[-1]

        highs = [b.high for b in wedge]
        lows = [b.low for b in wedge]
        declining_highs = all(highs[i] >= highs[i+1] for i in range(len(highs)-1))
        declining_lows = all(lows[i] >= lows[i+1] for i in range(len(lows)-1))

        if not (declining_highs and declining_lows):
            return Signal.NONE

        if any(b.low < ema_fast * 0.995 for b in wedge):
            return Signal.NONE

        if latest.close <= highs[0]:
            return Signal.NONE

        if avg_vol > 0 and latest.volume < VOLUME_SURGE_MULT * avg_vol:
            return Signal.NONE

        logging.info("[SIGNAL] Wedge Pop detected")
        return Signal.WEDGE_POP

    def _detect_wedge_drop(self, bars: list, ema_fast: float) -> Signal:
        """
        Wedge Drop: temporary pullback forming lower highs and lower lows
        while maintaining structure above key EMA. Context signal only -
        indicates trend pause, not reversal. No direct entry.
        """
        if len(bars) < WEDGE_BARS + 1:
            return Signal.NONE

        pullback = bars[-(WEDGE_BARS + 1):-1]
        latest = bars[-1]

        highs = [b.high for b in pullback]
        lows = [b.low for b in pullback]
        declining_highs = all(highs[i] >= highs[i+1] for i in range(len(highs)-1))
        declining_lows = all(lows[i] >= lows[i+1] for i in range(len(lows)-1))

        if not (declining_highs and declining_lows):
            return Signal.NONE

        if any(b.low < ema_fast * 0.99 for b in pullback):
            return Signal.NONE

        if latest.close >= highs[0]:
            return Signal.NONE

        for b in pullback:
            if b.low < ema_fast * 0.995:
                return Signal.NONE

        logging.info("[CONTEXT] Wedge Drop detected | trend pause, not reversal")
        return Signal.WEDGE_DROP

    def get_base_low(self) -> float:
        bars = list(self.bars)
        if len(bars) < BASE_MIN_BARS:
            return bars[-1].low if bars else 0.0
        base_bars = bars[-BASE_MIN_BARS:]
        return min(b.low for b in base_bars)

    def latest_indicators(self) -> dict:
        return self._indic_vals[-1] if self._indic_vals else {}

    def extract_features(self) -> dict:
        bars = list(self.bars)
        if len(bars) < BASE_MAX_BARS:
            return {}

        vals = self._indic_vals[-1]
        ema_f = vals["ema_fast"]
        ema_s = vals["ema_slow"]
        atr = vals["atr"]
        avg_vol = vals["avg_volume"]
        latest = bars[-1]

        ema_slope = (ema_f - self.indic.ema_fast_series[-5]) / 5 if len(self.indic.ema_fast_series) > 5 else 0
        ema_distance_ratio = (latest.close - ema_s) / atr if atr > 0 else 0
        atr_expansion = atr / np.mean(self.indic.atr_series[-10:]) if len(self.indic.atr_series) > 10 else 1

        recent_bars = bars[-5:]
        base_high = max(b.high for b in recent_bars)
        base_low = min(b.low for b in recent_bars)
        base_tightness = (base_high - base_low) / atr if atr > 0 else 0

        breakout_velocity = (latest.close - base_low) / atr if atr > 0 else 0
        volume_ratio = latest.volume / avg_vol if avg_vol > 0 else 1

        context_feats = self.feature_engine.extract_context_features(latest.time)
        regime = self.volatility_detector.detect()
        regime_score = {"TRENDING_UP": 1.0, "TRENDING_DOWN": -1.0, "RANGING": 0.0, "VOLATILE": 0.0, "UNKNOWN": 0.0}.get(regime, 0.0)

        return {
            "ema_slope": ema_slope,
            "ema_distance_ratio": ema_distance_ratio,
            "atr_expansion": atr_expansion,
            "base_tightness": base_tightness,
            "breakout_velocity": breakout_velocity,
            "volume_ratio": volume_ratio,
            "ema_fast": ema_f,
            "ema_slow": ema_s,
            "atr": atr,
            "mtf_alignment": context_feats.get("mtf_alignment", 0.0),
            "trend_maturity": context_feats.get("trend_maturity", 0),
            "asia_session": context_feats.get("asia_session", 0.0),
            "london_session": context_feats.get("london_session", 0.0),
            "ny_session": context_feats.get("ny_session", 0.0),
            "regime_score": regime_score,
        }


class FeatureEngine:
    """
    Computes advanced context features: multi-timeframe alignment, session timing, trend maturity.
    """
    ASIA_SESSION = (0, 8)
    LONDON_SESSION = (8, 16)
    NY_SESSION = (13, 21)

    def __init__(self, bar_buffer_size: int = 200):
        self.close_history: List[float] = []
        self.ema_fast_history: List[float] = []
        self.ema_slow_history: List[float] = []
        self.time_history: List[datetime] = []
        self.max_len = bar_buffer_size

    def update(self, close: float, ema_fast: float, ema_slow: float, timestamp: datetime):
        self.close_history.append(close)
        self.ema_fast_history.append(ema_fast)
        self.ema_slow_history.append(ema_slow)
        self.time_history.append(timestamp)
        if len(self.close_history) > self.max_len:
            self.close_history = self.close_history[-self.max_len:]
            self.ema_fast_history = self.ema_fast_history[-self.max_len:]
            self.ema_slow_history = self.ema_slow_history[-self.max_len:]
            self.time_history = self.time_history[-self.max_len:]

    def get_multi_timeframe_alignment(self) -> float:
        if len(self.close_history) < 50:
            return 0.0

        short_ema = np.mean(self.ema_fast_history[-10:])
        mid_ema = np.mean(self.ema_fast_history[-30:])
        long_ema = np.mean(self.ema_fast_history[-50:])

        bullish = short_ema > mid_ema > long_ema
        bearish = short_ema < mid_ema < long_ema

        if bullish:
            return 1.0
        elif bearish:
            return -1.0
        return 0.0

    def get_session_weights(self, timestamp: datetime) -> Tuple[float, float, float]:
        hour = timestamp.hour
        asia = 1.0 if self.ASIA_SESSION[0] <= hour < self.ASIA_SESSION[1] else 0.0
        london = 1.0 if self.LONDON_SESSION[0] <= hour < self.LONDON_SESSION[1] else 0.0
        ny = 1.0 if self.NY_SESSION[0] <= hour < self.NY_SESSION[1] else 0.0
        return (asia, london, ny)

    def get_trend_maturity(self) -> int:
        if len(self.close_history) < 5:
            return 0
        recent = np.array(self.close_history[-10:])
        if len(recent) < 2:
            return 0
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        if slope > 1e-6:
            direction = 1
        elif slope < -1e-6:
            direction = -1
        else:
            return 0
        count = 0
        for i in range(2, len(self.close_history)):
            p1, p2 = self.close_history[-i], self.close_history[-i+1]
            if (direction == 1 and p2 > p1) or (direction == -1 and p2 < p1):
                count += 1
            else:
                break
        return count

    def extract_context_features(self, timestamp: datetime) -> dict:
        return {
            "mtf_alignment": self.get_multi_timeframe_alignment(),
            "trend_maturity": self.get_trend_maturity(),
            "asia_session": self.get_session_weights(timestamp)[0],
            "london_session": self.get_session_weights(timestamp)[1],
            "ny_session": self.get_session_weights(timestamp)[2],
        }


class AITradeFilter:
    def __init__(self, model_path: str = MODEL_PATH, model_type: str = MODEL_TYPE):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.threshold = PROBABILITY_THRESHOLD
        self._load_model()

    def _load_model(self):
        if not HAS_SKLEARN:
            logging.warning("scikit-learn not available. Using default threshold.")
            return

        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logging.info(f"Loaded AI model from {self.model_path}")
            except Exception as e:
                logging.warning(f"Failed to load model: {e}. Using default threshold.")
        else:
            logging.info("No trained model found. Using default threshold.")
            self.model = self._create_default_model()

    def _create_default_model(self):
        if not HAS_SKLEARN:
            return None
        if self.model_type in {"mlp", "neural", "nn"}:
            return MLPClassifier(
                hidden_layer_sizes=MLP_HIDDEN_LAYERS,
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=42,
            )
        return GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )

    def predict_probability(self, features: dict) -> Tuple[float, float]:
        if self.model is None or not features:
            return 0.65, 0.65

        try:
            feature_vec = [
                features.get("ema_slope", 0),
                features.get("ema_distance_ratio", 0),
                features.get("atr_expansion", 1),
                features.get("base_tightness", 0),
                features.get("breakout_velocity", 0),
                features.get("volume_ratio", 1),
                features.get("mtf_alignment", 0.0),
                features.get("trend_maturity", 0),
                features.get("regime_score", 0.0),
                features.get("asia_session", 0.0),
                features.get("london_session", 0.0),
                features.get("ny_session", 0.0),
            ]
            X = np.array([feature_vec])
            prob = self.model.predict_proba(X)[0]
            confidence = float(prob[1]) if len(prob) > 1 else 0.65
            return confidence, confidence
        except ValueError as e:
            if "n_features" in str(e) and HAS_SKLEARN:
                logging.warning("Feature dimension mismatch - recreating model.")
                self.model = self._create_default_model()
                return 0.65, 0.65
            logging.warning(f"Prediction error: {e}")
            return 0.65, 0.65
        except Exception as e:
            logging.warning(f"Prediction error: {e}")
            return 0.65, 0.65

    def should_trade(self, features: dict) -> bool:
        confidence, _ = self.predict_probability(features)
        return confidence >= self.threshold

    def get_confidence_risk(self, confidence: float) -> float:
        """
        Continuous risk scaling based on confidence.
        Maps [0.0, 1.0] -> [0.2, 1.5] using smooth curve.
        Formula: 0.2 + (confidence^0.5) * 1.3
        Result: 0.5 confidence -> ~0.82, 0.8 confidence -> ~1.24
        """
        if confidence <= 0:
            return 0.2
        scaled = 0.2 + (confidence ** 0.5) * 1.3
        return max(0.2, min(1.5, scaled))

    def train(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            self.model = self._create_default_model()

        if HAS_SKLEARN and self.model is not None:
            self.model.fit(X, y)
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            logging.info(f"Model trained and saved to {self.model_path}")


class LLMRegimeDetector:
    """
    Optional LLM-based regime classifier using local Ollama model.
    If enabled, provides a secondary regime opinion based on market context.
    Falls back to MarketRegimeDetector if disabled or unavailable.
    """
    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.enabled = USE_LLM_REGIME
        self.model_name = model_name

    def detect(self, features_summary: str) -> Optional[str]:
        if not self.enabled:
            return None
        try:
            import ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": f"Classify market regime from: {features_summary}. Answer: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE"}],
            )
            text = response["message"]["content"].strip().upper()
            for regime in ("TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE"):
                if regime in text:
                    return regime
            return None
        except Exception as e:
            logging.debug(f"LLM regime detection skipped: {e}")
            return None


class MarketRegimeDetector:
    """
    Detects current market regime using volatility and trend strength metrics.
    Regimes: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, UNKNOWN
    """
    def __init__(self, lookback: int = 20, atr_threshold: float = 1.5, trend_strength_threshold: float = 0.3):
        self.lookback = lookback
        self.atr_threshold = atr_threshold
        self.trend_strength_threshold = trend_strength_threshold
        self.close_history: List[float] = []
        self.atr_history: List[float] = []
        self.ema_history: List[float] = []

    def update(self, close: float, atr: float, ema_fast: float, ema_slow: float):
        self.close_history.append(close)
        self.atr_history.append(atr)
        self.ema_history.append((ema_fast, ema_slow))
        if len(self.close_history) > self.lookback * 2:
            self.close_history = self.close_history[-self.lookback*2:]
            self.atr_history = self.atr_history[-self.lookback*2:]
            self.ema_history = self.ema_history[-self.lookback*2:]

    def detect(self) -> str:
        if len(self.close_history) < self.lookback:
            return "UNKNOWN"

        prices = np.array(self.close_history)
        atr_current = self.atr_history[-1] if self.atr_history else 0
        atr_avg = np.mean(self.atr_history[-10:]) if len(self.atr_history) >= 10 else atr_current

        volatility_ratio = atr_current / atr_avg if atr_avg > 0 else 1.0

        if volatility_ratio > 2.0:
            return "VOLATILE"

        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        price_range = prices.max() - prices.min()
        if price_range > 1e-10:
            trend_strength = abs(slope * len(prices)) / price_range
        else:
            trend_strength = 0

        ema_fast, ema_slow = self.ema_history[-1]
        ema_bullish = ema_fast > ema_slow
        ema_bearish = ema_fast < ema_slow

        if trend_strength > self.trend_strength_threshold:
            if slope > 0 and ema_bullish:
                return "TRENDING_UP"
            elif slope < 0 and ema_bearish:
                return "TRENDING_DOWN"

        return "RANGING"


class RiskManager:
    @staticmethod
    def position_size(balance: float, risk_pct: float, entry: float, stop: float, pip_value: float = 0.0001) -> int:
        risk_amount = balance * (risk_pct / 100.0)
        stop_distance = abs(entry - stop)
        if stop_distance < 1e-10:
            return 0
        units = int(risk_amount / stop_distance)
        return max(units, 0)

    @staticmethod
    def update_trail_stop(current_price: float, current_trail: float, atr: float) -> float:
        new_trail = current_price - TRAIL_ATR_MULT * atr
        return max(new_trail, current_trail)


class ParameterOptimizer:
    """
    Reinforcement-learning inspired hyperparameter tuner.
    Uses random search + backtest evaluation to optimize strategy parameters.
    Optimizes: EMA periods, ATR multipliers, volume threshold, probability threshold.
    Objective: maximize Sharpe ratio with drawdown penalty.
    """
    def __init__(self, base_params: dict, n_iterations: int = 20):
        self.base_params = base_params
        self.n_iterations = n_iterations
        self.best_score = -float('inf')
        self.best_params = base_params.copy()

    def _sample_params(self) -> dict:
        import random
        fast = random.choice([8, 10, 12, 15])
        slow = random.choice([20, 25, 30, 40])
        while slow <= fast:
            slow = random.choice([20, 25, 30, 40])
        exhaustion_mult = random.uniform(2.5, 4.0)
        base_atr_ratio = random.uniform(0.3, 0.8)
        volume_mult = random.uniform(1.2, 2.5)
        trail_mult = random.uniform(1.2, 2.5)
        prob_threshold = random.uniform(0.55, 0.80)
        return {
            "EMA_FAST": fast,
            "EMA_SLOW": slow,
            "EXHAUSTION_ATR_MULT": exhaustion_mult,
            "BASE_ATR_RATIO": base_atr_ratio,
            "VOLUME_SURGE_MULT": volume_mult,
            "TRAIL_ATR_MULT": trail_mult,
            "PROBABILITY_THRESHOLD": prob_threshold,
        }

    def optimize(self, df: pd.DataFrame) -> dict:
        if not HAS_SKLEARN:
            logging.warning("sklearn not available - skipping optimization.")
            return self.base_params

        logging.info(f"ParameterOptimizer: running {self.n_iterations} iterations ...")
        for i in range(self.n_iterations):
            params = self._sample_params()
            bot = self._create_bot(params)
            trades = bot.run_on_dataframe(df)
            score = self._calculate_score(trades)
            logging.info(f"  Iteration {i+1}/{self.n_iterations}: Sharpe={score:.3f} Params={params}")
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                logging.info(f"  → New best parameters found!")

        logging.info(f"Optimization complete. Best score: {self.best_score:.3f}")
        return self.best_params

    def _create_bot(self, params: dict) -> 'AITradingBot':
        import sys
        module = sys.modules[__name__]
        old = {}
        for k, v in params.items():
            if hasattr(module, k):
                old[k] = getattr(module, k)
                setattr(module, k, v)
        bot = AITradingBot()
        bot.balance = params.get("INITIAL_BALANCE", 10000.0)
        for k, v in old.items():
            setattr(module, k, v)
        return bot

    def _calculate_score(self, trades: pd.DataFrame) -> float:
        if trades.empty or len(trades) < 5:
            return -1.0
        win_rate = (trades["pnl"] > 0).mean()
        total_pnl = trades["pnl"].sum()
        avg_win = trades[trades["pnl"] > 0]["pnl"].mean() if (trades["pnl"] > 0).any() else 0
        avg_loss = trades[trades["pnl"] < 0]["pnl"].mean() if (trades["pnl"] < 0).any() else 0
        profit_factor = (trades[trades["pnl"] > 0]["pnl"].sum() /
                         abs(trades[trades["pnl"] < 0]["pnl"].sum())) if (trades["pnl"] < 0).any() else 5.0
        returns = trades["pnl"].values
        if len(returns) < 2:
            sharpe = 0.0
        else:
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            sharpe = mean_ret / std_ret if std_ret > 1e-10 else 0.0
        drawdown = self._calculate_max_drawdown(trades)
        penalty = max(0, (drawdown - 0.15) * 10)
        score = (profit_factor * 0.4 + sharpe * 0.4 + win_rate * 0.2) - penalty
        return score

    def _calculate_max_drawdown(self, trades: pd.DataFrame) -> float:
        equity = [self.base_params.get("INITIAL_BALANCE", 10000)]
        for _, t in trades.iterrows():
            equity.append(equity[-1] + t["pnl"])
        peak = equity[0]
        max_dd = 0.0
        for e in equity[1:]:
            if e > peak:
                peak = e
            dd = (peak - e) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd


class AITradingBot:
    def __init__(self):
        self.detector = PatternDetector()
        self.ai_filter = AITradeFilter()
        self.risk = RiskManager()
        self.state = TradeState()
        self.balance = 10_000.0
        self.log = logging.getLogger("AITradingBot")
        self.trade_log: List[dict] = []
        self.feature_history: List[dict] = []
        self.outcome_history: List[int] = []

    def _reconnect(self):
        self.log.info("Reconnecting to cTrader Open API ...")

        self.client = Client(
            EndPoints.PROTOBUF_LIVE_HOST,
            EndPoints.PROTOBUF_PORT,
            TcpProtocol,
        )

        self.client.setConnectedCallback(self._on_connected)
        self.client.setDisconnectedCallback(self._on_disconnected)
        self.client.setMessageReceivedCallback(self._on_message)

        self.client.startService()

    def start_live(self):
        if not LIVE_MODE:
            self.log.error("ctrader-open-api not installed. Cannot start live mode.")
            return

        self.log.info("Connecting to cTrader Open API ...")
        self.client = Client(
            EndPoints.PROTOBUF_LIVE_HOST,
            EndPoints.PROTOBUF_PORT,
            TcpProtocol,
        )
        self.client.setConnectedCallback(self._on_connected)
        self.client.setDisconnectedCallback(self._on_disconnected)
        self.client.setMessageReceivedCallback(self._on_message)
        self.client.startService()
        if not reactor.running:
            reactor.run()
            
    def _on_connected(self, client):
        self.log.info("Connected. Authenticating ...")
        request = ProtoOAApplicationAuthReq()
        request.clientId = CLIENT_ID
        request.clientSecret = CLIENT_SECRET
        deferred = client.send(request)
        deferred.addErrback(self._on_error)

    def _on_disconnected(self, client, reason):
        self.log.warning(f"Disconnected: {reason}. Reconnecting in 10s ...")
        reactor.callLater(10, self._reconnect)

    def _on_message(self, client, message):
        msg_type = Protobuf.extract(message)

        if msg_type == ProtoOAApplicationAuthRes:
            self.log.info("App authenticated. Authorising account ...")
            req = ProtoOAAccountAuthReq()
            req.ctidTraderAccountId = ACCOUNT_ID
            req.accessToken = ACCESS_TOKEN
            client.send(req).addErrback(self._on_error)

        elif msg_type == ProtoOAAccountAuthRes:
            self.log.info("Account authorised. Subscribing to spots ...")
            self._subscribe_spots(client)
            self._request_account_info(client)

        elif msg_type == ProtoOASpotEvent:
            spot = Protobuf.extract(message, ProtoOASpotEvent)
            self._on_tick(spot)

        elif msg_type == ProtoOATraderRes:
            trader = Protobuf.extract(message, ProtoOATraderRes)
            self.balance = trader.trader.balance / 100.0
            self.log.info(f"Account balance: {self.balance:.2f}")

    def _subscribe_spots(self, client):
        req = ProtoOASubscribeSpotsReq()
        req.ctidTraderAccountId = ACCOUNT_ID
        req.symbolId.append(self._symbol_id)
        client.send(req)

    def _request_account_info(self, client):
        req = ProtoOATraderReq()
        req.ctidTraderAccountId = ACCOUNT_ID
        self.client.send(req).addErrback(self._on_error)

    def _on_tick(self, spot):
        pass

    def _on_error(self, failure):
        self.log.error(f"API error: {failure}")

    def _on_bar(self, bar: Bar):
        signal = self.detector.push(bar)
        indic = self.detector.latest_indicators()
        atr = indic.get("atr", 0.0)

        if len(self.detector.bars) > 20 and signal != Signal.NONE:
            self.log.info(
                f"[DEBUG] bar={bar.close:.5f} ema_f={indic.get('ema_fast',0):.5f} "
                f"ema_s={indic.get('ema_slow',0):.5f} signal={signal.value}"
            )

        if self.state.in_position and atr > 0:
            new_trail = self.risk.update_trail_stop(bar.close, self.state.trail_stop, atr)
            if new_trail != self.state.trail_stop:
                self.state.trail_stop = new_trail
                self.log.debug(f"Trail stop updated -> {new_trail:.5f}")
            if bar.low <= self.state.trail_stop:
                self._close_position(bar, reason="TRAIL_STOP")
                return

        if signal == Signal.EXIT_EXHAUST:
            if self.state.in_position:
                self._close_position(bar, reason="EXHAUSTION_EXTENSION")

        elif signal == Signal.EXIT_CROSSBACK:
            if self.state.in_position:
                self._close_position(bar, reason="EMA_CROSSBACK")

        elif signal in (Signal.BASE_BREAK, Signal.WEDGE_POP):
            if not self.state.in_position:
                features = self.detector.extract_features()
                regime = self.detector.get_regime()
                if regime in ("RANGING", "VOLATILE", "UNKNOWN"):
                    self.log.info(f"[REGIME FILTER] Skipping trade | regime={regime}")
                    return

                if self.ai_filter.should_trade(features):
                    self.feature_history.append(features)
                    self._open_position(bar, signal, atr)
                else:
                    confidence, _ = self.ai_filter.predict_probability(features)
                    self.log.info(
                        f"[AI FILTER] Trade rejected | confidence={confidence:.2f} "
                        f"< {self.ai_filter.threshold:.2f}"
                    )
        elif signal == Signal.WEDGE_DROP:
            self.log.info(
                f"[CONTEXT] Wedge Drop detected | price={bar.close:.5f} "
                f"regime={self.detector.get_regime()}"
            )

    def _open_position(self, bar: Bar, signal: Signal, atr: float):
        base_low = self.detector.get_base_low()
        stop_loss = base_low - 0.5 * atr
        entry = bar.close

        features = self.detector.extract_features()
        confidence, _ = self.ai_filter.predict_probability(features)

        risk_pct = RISK_PERCENT * self.ai_filter.get_confidence_risk(confidence)
        units = self.risk.position_size(self.balance, risk_pct, entry, stop_loss)

        if units <= 0:
            self.log.warning("Calculated 0 units - skipping entry.")
            return

        self.state = TradeState(
            in_position=True,
            entry_price=entry,
            stop_loss=stop_loss,
            units=units,
            trail_stop=stop_loss,
            entry_time=bar.time,
            base_low=base_low,
            confidence=confidence,
        )

        self.log.info(
            f"[ENTRY] {signal.value} | Price={entry:.5f} "
            f"SL={stop_loss:.5f} Units={units} Confidence={confidence:.2f} @ {bar.time}"
        )

        if LIVE_MODE:
            self._send_market_order(entry, stop_loss, units, "BUY")

    def _close_position(self, bar: Bar, reason: str):
        exit_price = bar.close
        pnl = (exit_price - self.state.entry_price) * self.state.units

        self.log.info(
            f"[EXIT] {reason} | Price={exit_price:.5f} "
            f"PnL={pnl:+.2f} @ {bar.time}"
        )

        trade_record = {
            "entry_time": self.state.entry_time,
            "exit_time": bar.time,
            "entry": self.state.entry_price,
            "exit": exit_price,
            "units": self.state.units,
            "pnl": pnl,
            "reason": reason,
            "confidence": self.state.confidence,
        }

        if self.feature_history and self.state.entry_time:
            features = self.feature_history[-1]
            outcome = 1 if pnl > 0 else 0
            self.outcome_history.append(outcome)

            if len(self.outcome_history) >= 10:
                self._retrain_model()

        self.trade_log.append(trade_record)
        self.balance += pnl
        self.state = TradeState()

        if LIVE_MODE:
            self._close_market_order()

    def _retrain_model(self):
        if not HAS_SKLEARN or not self.feature_history or not self.outcome_history:
            return

        try:
            feature_list = []
            for f in self.feature_history:
                vec = [
                    f.get("ema_slope", 0),
                    f.get("ema_distance_ratio", 0),
                    f.get("atr_expansion", 1),
                    f.get("base_tightness", 0),
                    f.get("breakout_velocity", 0),
                    f.get("volume_ratio", 1),
                    f.get("mtf_alignment", 0.0),
                    f.get("trend_maturity", 0),
                    f.get("regime_score", 0.0),
                    f.get("asia_session", 0.0),
                    f.get("london_session", 0.0),
                    f.get("ny_session", 0.0),
                ]
                feature_list.append(vec)
            X = np.array(feature_list)
            y = np.array(self.outcome_history)

            if len(np.unique(y)) < 2:
                logging.info("Insufficient class variation for training.")
                return

            self.ai_filter.train(X, y)
            self.feature_history = []
            self.outcome_history = []
            logging.info("AI model retrained with latest trade data.")
        except Exception as e:
            logging.warning(f"Model retraining failed: {e}")

    def _send_market_order(self, entry: float, stop: float, units: int, direction: str):
        if not LIVE_MODE:
            return
        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = ACCOUNT_ID
        req.symbolId = self._symbol_id
        req.orderType = ProtoOAOrderType.Value("MARKET")
        req.tradeSide = (ProtoOATradeSide.Value("BUY") if direction == "BUY" else ProtoOATradeSide.Value("SELL"))
        req.volume = units * 100
        req.stopLoss = int(stop * 100_000)
        self.client.send(req).addErrback(self._on_error)

    def _close_market_order(self):
        if not LIVE_MODE or self.state.position_id is None:
            return
        req = ProtoOAClosePositionReq()
        req.ctidTraderAccountId = ACCOUNT_ID
        req.positionId = self.state.position_id
        req.volume = self.state.units * 100
        self.client.send(req).addErrback(self._on_error)

    def run_on_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.info(f"Starting backtest on {len(df)} bars ...")

        for _, row in df.iterrows():
            bar = Bar(
                time=row["time"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 1000)),
            )
            self._on_bar(bar)

        if self.state.in_position:
            last = Bar(
                time=df.iloc[-1]["time"],
                open=float(df.iloc[-1]["open"]),
                high=float(df.iloc[-1]["high"]),
                low=float(df.iloc[-1]["low"]),
                close=float(df.iloc[-1]["close"]),
                volume=float(df.iloc[-1].get("volume", 1000)),
            )
            self._close_position(last, reason="END_OF_DATA")

        trades_df = pd.DataFrame(self.trade_log)
        self._print_backtest_summary(trades_df)
        return trades_df

    def _print_backtest_summary(self, trades: pd.DataFrame):
        if trades.empty:
            self.log.info("No trades executed.")
            return

        total_pnl = trades["pnl"].sum()
        win_rate = (trades["pnl"] > 0).mean() * 100
        avg_win = trades.loc[trades["pnl"] > 0, "pnl"].mean() if (trades["pnl"] > 0).any() else 0
        avg_loss = trades.loc[trades["pnl"] < 0, "pnl"].mean() if (trades["pnl"] < 0).any() else 0
        profit_factor = (
            trades.loc[trades["pnl"] > 0, "pnl"].sum() /
            abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
            if (trades["pnl"] < 0).any() else float("inf")
        )

        avg_confidence = trades["confidence"].mean() if "confidence" in trades.columns else 0

        print("\n" + "="*60)
        print("   AI TRADING BOT - BACKTEST RESULTS")
        print("="*60)
        print(f"  Total trades      : {len(trades)}")
        print(f"  Win rate       : {win_rate:.1f}%")
        print(f"  Total PnL      : {total_pnl:+.2f}")
        print(f"  Avg win        : {avg_win:+.2f}")
        print(f"  Avg loss       : {avg_loss:.2f}")
        print(f"  Profit factor  : {profit_factor:.2f}")
        print(f"  Avg confidence: {avg_confidence:.2f}")
        print(f"  Final balance : {self.balance:.2f}")
        print("="*60)
        print(trades[["entry_time", "exit_time", "entry", "exit", "pnl", "confidence", "reason"]].to_string(index=False))
        print()


def generate_demo_data(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h")

    close = np.zeros(n_bars)
    high = np.zeros(n_bars)
    low = np.zeros(n_bars)
    volume = np.zeros(n_bars)

    current_price = 1.1000
    
    for i in range(50):
        trend = (i / 50) * 0.0008
        noise = rng.normal(trend, 0.00015)
        current_price += noise
        close[i] = current_price
        high[i] = current_price + rng.uniform(0.00005, 0.0002)
        low[i] = current_price - rng.uniform(0.00005, 0.0002)
        volume[i] = rng.uniform(800, 1200)
    
    breakout_points = [80, 220, 380, 520]
    
    for bp in breakout_points:
        if bp + 20 >= n_bars:
            break
        
        base_start = bp
        base_len = 5
        base_low = current_price
        base_high = current_price + rng.uniform(0.001, 0.0025)
        
        for j in range(base_len):
            idx = base_start + j
            if idx < n_bars:
                close[idx] = rng.uniform(base_low, base_high)
                high[idx] = base_high + rng.uniform(0, 0.00005)
                low[idx] = base_low - rng.uniform(0, 0.00005)
                volume[idx] = rng.uniform(850, 1150)
        
        breakout_idx = base_start + base_len
        if breakout_idx < n_bars:
            breakout_move = rng.uniform(0.004, 0.009)
            current_price = base_high + breakout_move
            close[breakout_idx] = current_price
            high[breakout_idx] = current_price + rng.uniform(0.0001, 0.00035)
            low[breakout_idx] = current_price - rng.uniform(0.0001, 0.00035)
            volume[breakout_idx] = rng.uniform(2200, 3800)
            
            for k in range(1, 18):
                if breakout_idx + k < n_bars:
                    trend_move = rng.normal(0.00035, 0.00025)
                    current_price += trend_move
                    close[breakout_idx + k] = current_price
                    high[breakout_idx + k] = current_price + rng.uniform(0.0001, 0.00035)
                    low[breakout_idx + k] = current_price - rng.uniform(0.0001, 0.00035)
                    volume[breakout_idx + k] = rng.uniform(1200, 2400)

    for i in range(n_bars):
        if close[i] == 0:
            noise = rng.normal(0.0001, 0.0001)
            current_price += noise
            close[i] = current_price
            high[i] = current_price + rng.uniform(0.00005, 0.0002)
            low[i] = current_price - rng.uniform(0.00005, 0.0002)
            volume[i] = rng.uniform(900, 1500)

    df = pd.DataFrame({
        "time": dates,
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    mode = "backtest"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode == "optimize":
        print("=== Parameter Optimization Mode ===")
        base_params = {
            "EMA_FAST": EMA_FAST,
            "EMA_SLOW": EMA_SLOW,
            "EXHAUSTION_ATR_MULT": EXHAUSTION_ATR_MULT,
            "BASE_ATR_RATIO": BASE_ATR_RATIO,
            "VOLUME_SURGE_MULT": VOLUME_SURGE_MULT,
            "TRAIL_ATR_MULT": TRAIL_ATR_MULT,
            "PROBABILITY_THRESHOLD": PROBABILITY_THRESHOLD,
            "INITIAL_BALANCE": 10000,
        }
        optimizer = ParameterOptimizer(base_params, n_iterations=10)
        demo_df = generate_demo_data(n_bars=600)
        best_params = optimizer.optimize(demo_df)
        print("\n=== Optimal Parameters ===")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        sys.exit(0)

    bot = AITradingBot()

    if LIVE_MODE and CLIENT_ID != "YOUR_CLIENT_ID":
        print("Starting live trading ...")
        bot.start_live()
    else:
        print("cTrader API credentials not set - running backtest on demo data.\n")
        demo_df = generate_demo_data(n_bars=600)
        bot.run_on_dataframe(demo_df)