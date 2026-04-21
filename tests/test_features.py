"""
Tests for feature engineering: MTF alignment, sessions, trend maturity.
"""
import pytest
from datetime import datetime, timedelta
import numpy as np

from src.core.features import FeatureEngine
from src.core.types import Bar


def test_mtf_alignment_bullish():
    fe = FeatureEngine()
    base = datetime(2024, 1, 1)
    # Create short > mid > long EMA stack
    for i in range(60):
        # rising price
        close = 1.0 + i * 0.0002
        ema_f = close * (1 - 0.01)
        ema_s = close * (1 - 0.02)
        fe.update(close, ema_f, ema_s, base + timedelta(minutes=i))
    alignment = fe.get_multi_timeframe_alignment()
    assert alignment == pytest.approx(1.0, abs=0.1)


def test_mtf_alignment_bearish():
    fe = FeatureEngine()
    base = datetime(2024, 1, 1)
    for i in range(60):
        close = 1.0 - i * 0.0002
        ema_f = close * (1 + 0.01)
        ema_s = close * (1 + 0.02)
        fe.update(close, ema_f, ema_s, base + timedelta(minutes=i))
    alignment = fe.get_multi_timeframe_alignment()
    assert alignment == pytest.approx(-1.0, abs=0.1)


def test_mtf_alignment_mixed():
    fe = FeatureEngine()
    base = datetime(2024, 1, 1)
    flat = 1.0
    for i in range(60):
        fe.update(flat, flat * 0.99, flat * 1.01, base + timedelta(minutes=i))
    alignment = fe.get_multi_timeframe_alignment()
    assert alignment == 0.0


def test_session_flags():
    fe = FeatureEngine()
    # Asia: 02:00 UTC => Asia active only
    t_asia = datetime(2024, 1, 1, 2, 0, 0)
    asia, lon, ny = fe.get_session_weights(t_asia)
    assert asia == 1.0 and lon == 0.0 and ny == 0.0
    # London: 12:00 UTC => London active, NY not yet (starts 13)
    t_lon = datetime(2024, 1, 1, 12, 0, 0)
    asia, lon, ny = fe.get_session_weights(t_lon)
    assert lon == 1.0 and asia == 0.0 and ny == 0.0
    # NY: 18:00 UTC => NY active
    t_ny = datetime(2024, 1, 1, 18, 0, 0)
    asia, lon, ny = fe.get_session_weights(t_ny)
    assert ny == 1.0
    # Overlap: 13:00 UTC => London (8-16) and NY (13-21) both active
    t_overlap = datetime(2024, 1, 1, 13, 0, 0)
    asia, lon, ny = fe.get_session_weights(t_overlap)
    assert lon == 1.0 and ny == 1.0


def test_trend_maturity_measures_streak():
    fe = FeatureEngine()
    base = datetime(2024, 1, 1)
    # Build strong uptrend
    price = 1.0
    for i in range(10):
        price += 0.001
        fe.update(price, price * 0.999, price * 0.998, base + timedelta(minutes=i))
    maturity = fe.get_trend_maturity()
    assert maturity >= 8  # at least 8 consecutive higher closes


def test_trend_maturity_resets_on_reversal():
    fe = FeatureEngine()
    base = datetime(2024, 1, 1)
    price = 1.0
    # Up 5, then down 2
    for i in range(5):
        price += 0.001
        fe.update(price, price * 0.999, price * 0.998, base + timedelta(minutes=i))
    for i in range(2):
        price -= 0.001
        fe.update(price, price * 0.999, price * 0.998, base + timedelta(minutes=5 + i))
    maturity = fe.get_trend_maturity()
    assert maturity <= 2  # only most recent consecutive direction


def test_feature_extraction_integration(feature_engine):
    """Full context features extraction."""
    base = datetime(2024, 1, 1, 12, 0, 0)
    price = 1.1000
    for i in range(60):
        fe = FeatureEngine()
        # Use per-call FE instance for test isolation
        close = price + i * 0.0001
        ema_f = close * 0.999
        ema_s = close * 0.998
        fe.update(close, ema_f, ema_s, base + timedelta(minutes=i))
    feats = fe.extract_context_features(base + timedelta(minutes=59))
    assert "mtf_alignment" in feats
    assert "trend_maturity" in feats
    assert "asia_session" in feats
    assert "london_session" in feats
    assert "ny_session" in feats
