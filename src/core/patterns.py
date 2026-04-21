"""
Pattern detection engine: Base 'n Break, Wedge Pop, Wedge Drop (context).
"""
from collections import deque
from typing import Deque, List, Optional
import logging

import numpy as np

from .types import Bar, Signal
from .indicators import Indicators


class PatternDetector:
    """
    Detects price action patterns and emits trading signals.
    Wedge Drop is context-only (no direct entry).
    """

    def __init__(
        self,
        bar_buffer_size: int = 50,
        base_min_bars: int = 3,
        base_max_bars: int = 15,
        base_atr_ratio: float = 0.5,
        volume_surge_mult: float = 1.5,
        wedge_bars: int = 5,
        exhaustion_atr_mult: float = 3.0,
    ):
        self.bars: Deque[Bar] = deque(maxlen=bar_buffer_size)
        self.indic = Indicators()
        self._indic_vals: List[dict] = []

        # Configurable parameters
        self.base_min_bars = base_min_bars
        self.base_max_bars = base_max_bars
        self.base_atr_ratio = base_atr_ratio
        self.volume_surge_mult = volume_surge_mult
        self.wedge_bars = wedge_bars
        self.exhaustion_atr_mult = exhaustion_atr_mult

        # External feature engines (injected or created internally)
        self.feature_engine: Optional["FeatureEngine"] = None
        self.volatility_detector: Optional["MarketRegimeDetector"] = None

    def push(self, bar: Bar) -> Signal:
        self.bars.append(bar)
        vals = self.indic.update(bar)
        self._indic_vals.append(vals)

        if len(self.bars) < self.base_max_bars + 5:
            return Signal.NONE

        signal = self._evaluate(vals)

        # Update external feature engines if attached
        if self.feature_engine is not None:
            self.feature_engine.update(
                close=bar.close,
                ema_fast=vals["ema_fast"],
                ema_slow=vals["ema_slow"],
                timestamp=bar.time,
            )
        if self.volatility_detector is not None:
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

        # Exit signal: EMA crossback
        if latest.close < ema_f and latest.close < ema_s:
            return Signal.EXIT_CROSSBACK

        # Exit signal: Exhaustion Extension
        dist_from_slow_ema = latest.close - ema_s
        if atr > 0 and (dist_from_slow_ema / atr) > self.exhaustion_atr_mult:
            if latest.close > ema_f > ema_s:
                return Signal.EXIT_EXHAUST

        # Must be above both EMAs for entry patterns
        if latest.close < ema_f or latest.close < ema_s:
            return Signal.NONE

        # Check Base 'n Break
        base_signal = self._detect_base_break(bars, ema_s, atr, avg_v)
        if base_signal != Signal.NONE:
            return base_signal

        # Check Wedge Pop
        wedge_signal = self._detect_wedge_pop(bars, ema_f, avg_v)
        if wedge_signal != Signal.NONE:
            return wedge_signal

        # Check Wedge Drop (context only)
        wedge_drop = self._detect_wedge_drop(bars, ema_f)
        if wedge_drop != Signal.NONE:
            return wedge_drop

        return Signal.NONE

    def _detect_base_break(self, bars: list, ema_slow: float, atr: float, avg_vol: float) -> Signal:
        latest = bars[-1]

        for base_len in range(self.base_min_bars, self.base_max_bars + 1):
            if len(bars) < base_len + 1:
                break

            base_bars = bars[-(base_len + 1):-1]
            breakout_bar = bars[-1]

            base_high = max(b.high for b in base_bars)
            base_low = min(b.low for b in base_bars)
            base_range = base_high - base_low

            if atr > 0 and base_range > self.base_atr_ratio * atr:
                continue

            if base_low < ema_slow * 0.995:
                continue

            if breakout_bar.close <= base_high:
                continue

            if avg_vol > 0 and breakout_bar.volume < self.volume_surge_mult * avg_vol:
                continue

            logging.info(
                f"[SIGNAL] Base 'n Break | base_len={base_len} "
                f"base_range={base_range:.5f} atr={atr:.5f} "
                f"vol={breakout_bar.volume:.0f} avg_vol={avg_vol:.0f}"
            )
            return Signal.BASE_BREAK

        return Signal.NONE

    def _detect_wedge_pop(self, bars: list, ema_fast: float, avg_vol: float) -> Signal:
        if len(bars) < self.wedge_bars + 1:
            return Signal.NONE

        wedge = bars[-(self.wedge_bars + 1):-1]
        latest = bars[-1]

        highs = [b.high for b in wedge]
        lows = [b.low for b in wedge]
        declining_highs = all(highs[i] >= highs[i + 1] for i in range(len(highs) - 1))
        declining_lows = all(lows[i] >= lows[i + 1] for i in range(len(lows) - 1))

        if not (declining_highs and declining_lows):
            return Signal.NONE

        if any(b.low < ema_fast * 0.995 for b in wedge):
            return Signal.NONE

        if latest.close <= highs[0]:
            return Signal.NONE

        if avg_vol > 0 and latest.volume < self.volume_surge_mult * avg_vol:
            return Signal.NONE

        logging.info("[SIGNAL] Wedge Pop detected")
        return Signal.WEDGE_POP

    def _detect_wedge_drop(self, bars: list, ema_fast: float) -> Signal:
        """
        Wedge Drop: temporary pullback forming lower highs and lower lows
        while maintaining structure above EMA. Context signal only.
        """
        if len(bars) < self.wedge_bars + 1:
            return Signal.NONE

        pullback = bars[-(self.wedge_bars + 1):-1]
        latest = bars[-1]

        highs = [b.high for b in pullback]
        lows = [b.low for b in pullback]
        declining_highs = all(highs[i] >= highs[i + 1] for i in range(len(highs) - 1))
        declining_lows = all(lows[i] >= lows[i + 1] for i in range(len(lows) - 1))

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

    # Helper methods
    @property
    def ema_fast(self) -> Optional[float]:
        return self.indic._ema_fast

    @property
    def ema_slow(self) -> Optional[float]:
        return self.indic._ema_slow

    def latest_indicators(self) -> dict:
        return self._indic_vals[-1] if self._indic_vals else {}

    def extract_features(self, feature_engine: "FeatureEngine") -> dict:
        """Extract full feature set at current bar."""
        bars = list(self.bars)
        if len(bars) < self.base_max_bars:
            return {}

        vals = self._indic_vals[-1]
        ema_f = vals["ema_fast"]
        ema_s = vals["ema_slow"]
        atr = vals["atr"]
        avg_vol = vals["avg_volume"]
        latest = bars[-1]

        # Structure features
        ema_slope = (ema_f - self.indic.ema_fast_series[-5]) / 5 if len(self.indic.ema_fast_series) > 5 else 0.0
        ema_distance_ratio = (latest.close - ema_s) / atr if atr > 0 else 0.0
        atr_expansion = atr / np.mean(self.indic.atr_series[-10:]) if len(self.indic.atr_series) > 10 else 1.0

        recent_bars = bars[-5:]
        base_high = max(b.high for b in recent_bars)
        base_low = min(b.low for b in recent_bars)
        base_tightness = (base_high - base_low) / atr if atr > 0 else 0.0

        breakout_velocity = (latest.close - base_low) / atr if atr > 0 else 0.0
        volume_ratio = latest.volume / avg_vol if avg_vol > 0 else 1.0

        # Context features
        context_feats = feature_engine.extract_context_features(latest.time) if feature_engine else {}
        regime = self.volatility_detector.detect() if self.volatility_detector else "UNKNOWN"
        regime_score = {
            "TRENDING_UP": 1.0,
            "TRENDING_DOWN": -1.0,
            "RANGING": 0.0,
            "VOLATILE": 0.0,
            "UNKNOWN": 0.0,
        }.get(regime, 0.0)

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

    def get_base_low(self) -> float:
        bars = list(self.bars)
        if len(bars) < self.base_min_bars:
            return bars[-1].low if bars else 0.0
        base_bars = bars[-self.base_min_bars:]
        return min(b.low for b in base_bars)
