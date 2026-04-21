"""
Tests for pattern detector using stubbed indicators and deterministic bar sequences.
"""
import pytest
from datetime import datetime, timedelta
from src.core.patterns import PatternDetector, Signal
from src.core.types import Bar


class DummyIndic:
    """Stub indicators returning constant values; maintains series for slope lookbacks."""
    def __init__(self, ema_fast, ema_slow, atr, avg_volume):
        self.ema_fast_val = ema_fast
        self.ema_slow_val = ema_slow
        self.atr_val = atr
        self.avg_volume_val = avg_volume
        # Provide enough series history for slope calculations (needs last 5)
        self.ema_fast_series = [ema_fast] * 20
        self.ema_slow_series = [ema_slow] * 20
        self.atr_series = [atr] * 20

    def update(self, bar):
        return {
            "ema_fast": self.ema_fast_val,
            "ema_slow": self.ema_slow_val,
            "atr": self.atr_val,
            "avg_volume": self.avg_volume_val,
        }


def make_bar(close, time, volume=1000, high_offset=0.0002, low_offset=0.0002):
    return Bar(
        time=time,
        open=close - 0.0001,
        high=close + high_offset,
        low=close - low_offset,
        close=close,
        volume=volume,
    )


def push_dummy_bars(detector, n, start_time, close=100.0, volume=100):
    """Push n dummy bars to satisfy detector's minimum bar count guard."""
    for i in range(n):
        bar = make_bar(close=close, time=start_time + timedelta(minutes=i), volume=volume)
        detector.push(bar)
    return start_time + timedelta(minutes=n)


def test_base_n_break_with_dummy_indicator():
    detector = PatternDetector(
        bar_buffer_size=50,
        base_min_bars=2,
        base_max_bars=3,
        base_atr_ratio=0.9,
        volume_surge_mult=1.05,
        exhaustion_atr_mult=3.0,
    )
    detector.indic = DummyIndic(ema_fast=99.0, ema_slow=98.5, atr=2.0, avg_volume=100)

    base_time = datetime(2024, 1, 1)
    # Pad to satisfy len(self.bars) >= base_max_bars + 5 = 8
    next_time = push_dummy_bars(detector, 8, base_time)

    # Base bars (2)
    b1 = make_bar(close=100.0, time=next_time, volume=100)
    b2 = make_bar(close=100.1, time=next_time + timedelta(minutes=1), volume=100)
    detector.push(b1)
    detector.push(b2)

    # Breakout bar
    breakout = make_bar(close=101.0, time=next_time + timedelta(minutes=2), volume=200, high_offset=0.0005, low_offset=0.0005)
    signal = detector.push(breakout)
    assert signal == Signal.BASE_BREAK


def test_wedge_pop_with_dummy_indicator():
    detector = PatternDetector(
        bar_buffer_size=50,
        wedge_bars=2,
        volume_surge_mult=1.05,
        base_max_bars=3,
        exhaustion_atr_mult=3.0,
    )
    detector.indic = DummyIndic(ema_fast=99.0, ema_slow=98.5, atr=2.0, avg_volume=100)

    base_time = datetime(2024, 1, 1)
    next_time = push_dummy_bars(detector, 8, base_time)

    # Wedge pullback: descending highs and descending lows
    w1 = make_bar(close=101.0, time=next_time, volume=100, high_offset=1.0, low_offset=0.5)
    # w1: high=102.0, low=100.5, close=101.0
    w2 = make_bar(close=100.3, time=next_time + timedelta(minutes=1), volume=100, high_offset=0.3, low_offset=0.8)
    # w2: high=100.6, low=99.5, close=100.3 → lower high (100.6<102.0), lower low (99.5<100.5)
    detector.push(w1)
    detector.push(w2)

    # Breakout above wedge high (w1.high=102.0)
    breakout = make_bar(close=102.5, time=next_time + timedelta(minutes=2), volume=200, high_offset=0.5, low_offset=0.5)
    signal = detector.push(breakout)
    assert signal == Signal.WEDGE_POP


def test_wedge_drop_context_with_dummy_indicator():
    detector = PatternDetector(
        bar_buffer_size=50,
        wedge_bars=2,
        base_max_bars=3,
        exhaustion_atr_mult=3.0,
    )
    detector.indic = DummyIndic(ema_fast=99.0, ema_slow=98.5, atr=2.0, avg_volume=100)

    base_time = datetime(2024, 1, 1)
    next_time = push_dummy_bars(detector, 8, base_time)

    # Wedge pullback: descending highs and descending lows
    w1 = make_bar(close=101.0, time=next_time, volume=100, high_offset=1.0, low_offset=0.5)  # high=102, low=100.5
    w2 = make_bar(close=100.3, time=next_time + timedelta(minutes=1), volume=100, high_offset=0.3, low_offset=0.8)  # high=100.6, low=99.5
    detector.push(w1)
    detector.push(w2)

    # Continuation bar stays below wedge high (w1.high=102) → WEDGE_DROP (context)
    cont = make_bar(close=99.5, time=next_time + timedelta(minutes=2), volume=100, high_offset=0.3, low_offset=0.3)
    signal = detector.push(cont)
    assert signal == Signal.WEDGE_DROP


def test_exhaustion_exit_signal_with_dummy_indicator():
    detector = PatternDetector(
        bar_buffer_size=50,
        base_max_bars=3,
        exhaustion_atr_mult=2.0,  # lower so condition triggers
    )
    detector.indic = DummyIndic(ema_fast=100.0, ema_slow=99.0, atr=1.0, avg_volume=100)

    base_time = datetime(2024, 1, 1)
    next_time = push_dummy_bars(detector, 8, base_time)

    # Normal bar (above EMAs)
    normal = make_bar(close=101.0, time=next_time, volume=100)
    detector.push(normal)

    # Spike bar far above EMAs
    spike = make_bar(close=103.5, time=next_time + timedelta(minutes=1), volume=100)  # dist=4.5, ratio=4.5/1=4.5>2
    signal = detector.push(spike)
    assert signal == Signal.EXIT_EXHAUST


def test_ema_crossback_exit_signal_with_dummy_indicator():
    detector = PatternDetector(bar_buffer_size=50, base_max_bars=3)
    detector.indic = DummyIndic(ema_fast=100.0, ema_slow=99.5, atr=1.0, avg_volume=100)

    base_time = datetime(2024, 1, 1)
    next_time = push_dummy_bars(detector, 8, base_time)

    # Bar above EMAs
    above = make_bar(close=101.0, time=next_time, volume=100)
    detector.push(above)

    # Drop below both EMAs
    drop = make_bar(close=99.0, time=next_time + timedelta(minutes=1), volume=100)
    signal = detector.push(drop)
    assert signal == Signal.EXIT_CROSSBACK
