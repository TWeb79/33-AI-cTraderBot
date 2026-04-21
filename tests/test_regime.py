"""
Tests for market regime detection (volatility-based + LLM stub).
"""
import pytest
import numpy as np

from src.core.regime import MarketRegimeDetector, LLMRegimeDetector
from src.core.types import Bar
from datetime import datetime


def make_bar(close: float) -> Bar:
    return Bar(time=datetime(2024, 1, 1), open=close, high=close + 0.001, low=close - 0.001, close=close, volume=1000)


def test_regime_trending_up():
    det = MarketRegimeDetector()
    price = 1.0
    for i in range(30):
        price += 0.001
        det.update(price, atr=0.0005, ema_fast=price * 0.999, ema_slow=price * 0.998)
    regime = det.detect()
    assert regime == "TRENDING_UP"


def test_regime_trending_down():
    det = MarketRegimeDetector()
    price = 1.0
    for i in range(30):
        price -= 0.001
        det.update(price, atr=0.0005, ema_fast=price * 1.001, ema_slow=price * 1.002)
    regime = det.detect()
    assert regime == "TRENDING_DOWN"


def test_regime_volatile_spike():
    det = MarketRegimeDetector()
    price = 1.0
    base = datetime(2024, 1, 1)
    # 20 bars with low ATR (baseline)
    low_atr = 0.0005
    for i in range(20):
        price += 0.0001
        det.update(price, atr=low_atr, ema_fast=price, ema_slow=price)
    # Single bar with high ATR spikes volatility ratio
    high_atr = 0.005
    price += 0.001
    det.update(price, atr=high_atr, ema_fast=price, ema_slow=price)
    regime = det.detect()
    assert regime == "VOLATILE"


def test_regime_ranging():
    det = MarketRegimeDetector()
    price = 1.0
    for i in range(30):
        # oscillate around flat
        offset = 0.0005 * (-1 if i % 2 == 0 else 1)
        price += offset
        det.update(price, atr=0.0005, ema_fast=price, ema_slow=price)
    regime = det.detect()
    assert regime in ("RANGING", "UNKNOWN")  # may be UNKNOWN if not enough trend


def test_llm_detector_disabled():
    llm = LLMRegimeDetector(enabled=False)
    result = llm.detect("test")
    assert result is None


def test_llm_detector_parses_valid_response(monkeypatch):
    try:
        import ollama  # noqa: F401
    except ImportError:
        pytest.skip("ollama not installed")
    class DummyMsg:
        def __init__(self, content):
            self.content = content
    def fake_chat(**kwargs):
        return {"message": DummyMsg("TRENDING_UP")}
    monkeypatch.setattr("ollama.chat", fake_chat)
    llm = LLMRegimeDetector(enabled=True, model_name="test-model")
    result = llm.detect("some context")
    assert result == "TRENDING_UP"


def test_llm_detector_handles_unknown(monkeypatch):
    try:
        import ollama
    except ImportError:
        pytest.skip("ollama not installed")
    class DummyMsg:
        def __init__(self, content):
            self.content = content
    def fake_chat(**kwargs):
        return {"message": DummyMsg("UNKNOWN")}
    monkeypatch.setattr(ollama, "chat", fake_chat)
    llm = LLMRegimeDetector(enabled=True)
    result = llm.detect("context")
    assert result is None
