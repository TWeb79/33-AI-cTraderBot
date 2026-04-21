"""
Feature engineering: multi-timeframe alignment, session timing, trend maturity.
"""
from typing import Tuple
from datetime import datetime
import numpy as np

from .types import Bar


class FeatureEngine:
    """
    Computes context features that supplement the core pattern signals.
    """

    ASIA_SESSION = (0, 8)
    LONDON_SESSION = (8, 16)
    NY_SESSION = (13, 21)

    def __init__(self, bar_buffer_size: int = 200):
        self.close_history: list[float] = []
        self.ema_fast_history: list[float] = []
        self.ema_slow_history: list[float] = []
        self.time_history: list[datetime] = []
        self.max_len = bar_buffer_size

    def update(self, close: float, ema_fast: float, ema_slow: float, timestamp: datetime):
        self.close_history.append(close)
        self.ema_fast_history.append(ema_fast)
        self.ema_slow_history.append(ema_slow)
        self.time_history.append(timestamp)
        if len(self.close_history) > self.max_len:
            self.close_history = self.close_history[-self.max_len :]
            self.ema_fast_history = self.ema_fast_history[-self.max_len :]
            self.ema_slow_history = self.ema_slow_history[-self.max_len :]
            self.time_history = self.time_history[-self.max_len :]

    def get_multi_timeframe_alignment(self) -> float:
        """
        Computes EMA stack alignment: short > mid > long → bullish (1),
        short < mid < long → bearish (-1), else 0.
        """
        if len(self.ema_fast_history) < 50:
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
        """
        Counts consecutive bars reinforcing the current short-term trend direction.
        Returns streak length (0 if flat).
        """
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
            p1, p2 = self.close_history[-i], self.close_history[-i + 1]
            if (direction == 1 and p2 > p1) or (direction == -1 and p2 < p1):
                count += 1
            else:
                break
        return count

    def extract_context_features(self, timestamp: datetime) -> dict:
        asia, london, ny = self.get_session_weights(timestamp)
        return {
            "mtf_alignment": self.get_multi_timeframe_alignment(),
            "trend_maturity": self.get_trend_maturity(),
            "asia_session": asia,
            "london_session": london,
            "ny_session": ny,
        }
