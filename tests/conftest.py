"""
Pytest fixtures for common test data.
"""
import pytest
from datetime import datetime, timedelta
from src.core.types import Bar
from src.core.indicators import Indicators
from src.core.patterns import PatternDetector
from src.core.features import FeatureEngine
from src.core.regime import MarketRegimeDetector
from src.core.ai_filter import AITradeFilter
from src.core.risk import RiskManager


@pytest.fixture
def sample_bars_uptrend():
    """10-bar steady uptrend."""
    bars = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    price = 1.1000
    for i in range(10):
        open_p = price
        high = price + 0.0005
        low = price - 0.0003
        close = price + 0.0004
        bars.append(
            Bar(
                time=base_time + timedelta(minutes=i * 5),
                open=open_p,
                high=high,
                low=low,
                close=close,
                volume=1000,
            )
        )
        price = close
    return bars


@pytest.fixture
def sample_bars_downtrend():
    """10-bar steady downtrend."""
    bars = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    price = 1.1000
    for i in range(10):
        open_p = price
        high = price + 0.0003
        low = price - 0.0005
        close = price - 0.0004
        bars.append(
            Bar(
                time=base_time + timedelta(minutes=i * 5),
                open=open_p,
                high=high,
                low=low,
                close=close,
                volume=1000,
            )
        )
        price = close
    return bars


@pytest.fixture
def indicators():
    return Indicators(fast=10, slow=20, atr_period=14)


@pytest.fixture
def pattern_detector():
    return PatternDetector(bar_buffer_size=50)


@pytest.fixture
def feature_engine():
    return FeatureEngine(bar_buffer_size=200)


@pytest.fixture
def regime_detector():
    return MarketRegimeDetector()


@pytest.fixture
def ai_filter():
    return AITradeFilter(model_path="test_model.pkl", model_type="gb", threshold=0.65)


@pytest.fixture
def risk_mgr():
    return RiskManager()
