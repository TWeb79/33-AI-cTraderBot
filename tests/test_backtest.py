"""
Integration test: backtest executor end-to-end on synthetic dataset.
"""
import pytest
import pandas as pd
import numpy as np
import logging

from src.core.types import Bar, TradeState, Signal
from src.core.patterns import PatternDetector
from src.core.features import FeatureEngine
from src.core.regime import MarketRegimeDetector
from src.core.ai_filter import AITradeFilter
from src.core.risk import RiskManager
from src.execution.backtest_executor import BacktestExecutor


def generate_synthetic_trending_data(n_bars: int = 200) -> pd.DataFrame:
    """Creates a simple uptrend with occasional pullbacks and breakouts."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
    close = np.zeros(n_bars)
    high = np.zeros(n_bars)
    low = np.zeros(n_bars)
    volume = np.zeros(n_bars)

    price = 1.1000
    for i in range(n_bars):
        noise = rng.normal(0.00005, 0.0002)
        price += noise
        close[i] = price
        high[i] = price + rng.uniform(0.0001, 0.0003)
        low[i] = price - rng.uniform(0.0001, 0.0003)
        volume[i] = rng.uniform(800, 1500)

    # Inject a base + breakout
    base_start = 50
    base_low_ = price
    base_high = price + 0.001
    for j in range(5):
        idx = base_start + j
        if idx < n_bars:
            close[idx] = rng.uniform(base_low_, base_high)
            high[idx] = base_high
            low[idx] = base_low_
            volume[idx] = 1000
    breakout_idx = base_start + 5
    if breakout_idx < n_bars:
        close[breakout_idx] = base_high + 0.005
        high[breakout_idx] = close[breakout_idx] + 0.001
        low[breakout_idx] = close[breakout_idx] - 0.001
        volume[breakout_idx] = 3000

    df = pd.DataFrame(
        {
            "time": dates,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return df


@pytest.fixture
def bot_components():
    detector = PatternDetector()
    feature_engine = FeatureEngine()
    regime_detector = MarketRegimeDetector()
    ai_filter = AITradeFilter(threshold=0.5)
    risk_mgr = RiskManager()

    detector.feature_engine = feature_engine
    detector.volatility_detector = regime_detector

    return {
        "detector": detector,
        "feature_engine": feature_engine,
        "regime_detector": regime_detector,
        "ai_filter": ai_filter,
        "risk_mgr": risk_mgr,
    }


def test_backtest_executor_runs_without_error(bot_components):
    """Test that executor processes DataFrame and completes without throwing."""
    df = generate_synthetic_trending_data(200)
    comps = bot_components

    executor = BacktestExecutor(
        df=df,
        detector=comps["detector"],
        ai_filter=comps["ai_filter"],
        risk_mgr=comps["risk_mgr"],
        initial_balance=10000.0,
    )
    # Create minimal trading bot-like wrapper
    class MockBot:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    bot = MockBot(**comps, state=TradeState(), balance=10000.0, trade_log=[], feature_history=[], outcome_history=[])
    # Fill in on_bar method by binding core logic from src.bot
    from src.bot import TradingBot as RealBot
    real = RealBot()
    bot.on_bar = real.on_bar.__get__(bot, MockBot)  # bind method

    executor.on_bar = bot.on_bar  # type: ignore
    trades = executor.run()

    assert isinstance(trades, pd.DataFrame)
    logging.info(f"Backtest completed. Trades: {len(trades)}")


# Re-define TradeState for this test file
from src.core.types import TradeState
