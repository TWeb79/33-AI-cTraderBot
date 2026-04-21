"""
Tests for technical indicators: EMA and ATR calculations accuracy.
"""
import pytest
from src.core.indicators import Indicators
from src.core.types import Bar
from datetime import datetime


def test_ema_initialization():
    ind = Indicators(fast=10, slow=20)
    bar = Bar(
        time=datetime(2024, 1, 1),
        open=1.0,
        high=1.001,
        low=0.999,
        close=1.0,
        volume=1000,
    )
    vals = ind.update(bar)
    assert vals["ema_fast"] == pytest.approx(1.0, abs=1e-6)
    assert vals["ema_slow"] == pytest.approx(1.0, abs=1e-6)


def test_ema_convergence():
    """EMA should gradually equal close price on steady trend."""
    ind = Indicators(fast=2, slow=2)
    price = 1.0
    for i in range(20):
        bar = Bar(
            time=datetime(2024, 1, 1, 0, i),
            open=price,
            high=price + 0.001,
            low=price - 0.001,
            close=price,
            volume=1000,
        )
        ind.update(bar)
    assert ind._ema_fast == pytest.approx(1.0, abs=0.01)


def test_atr_calculation():
    """ATR must be positive after minimum period filled."""
    ind = Indicators(atr_period=3)
    base = 1.0
    highs = [1.001, 1.002, 1.003]
    lows = [0.999, 0.998, 0.997]
    closes = [1.0005, 1.0015, 1.0025]
    prev_close = None
    for i in range(3):
        bar = Bar(
            time=datetime(2024, 1, 1, 0, i),
            open=base,
            high=highs[i],
            low=lows[i],
            close=closes[i],
            volume=1000,
        )
        vals = ind.update(bar)
        prev_close = closes[i]
    assert vals["atr"] > 0


def test_volume_average():
    ind = Indicators(volume_window=5)
    for i in range(5):
        bar = Bar(
            time=datetime(2024, 1, 1, 0, i),
            open=1.0,
            high=1.001,
            low=0.999,
            close=1.0,
            volume=1000 + i * 100,
        )
        ind.update(bar)
    vals = ind.update(
        Bar(
            time=datetime(2024, 1, 1, 0, 5),
            open=1.0,
            high=1.001,
            low=0.999,
            close=1.0,
            volume=2000,
        )
    )
    # Average of last 5 volumes: [1100,1200,1300,1400,2000] = 1400
    assert vals["avg_volume"] == pytest.approx(1400.0, abs=1e-3)


def test_indicator_state_after_many_bars():
    ind = Indicators()
    base = datetime(2024, 1, 1, 0, 0)
    for i in range(100):
        # Keep minute within 0-59 by using second as i % 60
        ts = base.replace(minute=i % 60, second=i // 60)
        bar = Bar(
            time=ts,
            open=1.0 + i * 0.0001,
            high=1.0 + i * 0.0001 + 0.001,
            low=1.0 + i * 0.0001 - 0.001,
            close=1.0 + (i + 1) * 0.0001,
            volume=1000,
        )
        ind.update(bar)
    assert len(ind.ema_fast_series) == 100
    assert len(ind.atr_series) == 100
