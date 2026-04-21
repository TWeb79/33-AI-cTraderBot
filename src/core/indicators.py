"""
Technical indicators: EMA, ATR, volume averages.
"""
from collections import deque
from typing import Deque, List, Optional
import numpy as np

from .types import Bar


class Indicators:
    def __init__(
        self,
        fast: int = 10,
        slow: int = 20,
        atr_period: int = 14,
        volume_window: int = 20,
    ):
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period
        self.volume_window = volume_window

        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._prev_close: Optional[float] = None

        self._tr_window: Deque[float] = deque(maxlen=atr_period)
        self._vol_window: Deque[float] = deque(maxlen=volume_window)

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
