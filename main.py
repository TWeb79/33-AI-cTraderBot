"""
33 AI cTrader Bot — modular refactored version.
Entry point for backtest and live trading.
"""
import sys
import logging
import pandas as pd

from src.bot import TradingBot
from src.core.indicators import Indicators
from src.core.patterns import PatternDetector
from src.core.features import FeatureEngine
from src.core.regime import MarketRegimeDetector, LLMRegimeDetector
from src.core.ai_filter import AITradeFilter
from src.core.risk import RiskManager
from src.execution.backtest_executor import BacktestExecutor
from src.execution.live_executor import LiveExecutor


def generate_demo_data(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic data generator for testing."""
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    close = np.zeros(n_bars)
    high = np.zeros(n_bars)
    low = np.zeros(n_bars)
    volume = np.zeros(n_bars)

    current_price = 1.1000
    # Initial trend
    for i in range(50):
        trend = (i / 50) * 0.0008
        noise = rng.normal(trend, 0.00015)
        current_price += noise
        close[i] = current_price
        high[i] = current_price + rng.uniform(0.00005, 0.0002)
        low[i] = current_price - rng.uniform(0.00005, 0.0002)
        volume[i] = rng.uniform(800, 1200)

    # Breakout points with consolidation
    breakout_points = [80, 220, 380]
    for bp in breakout_points:
        if bp + 20 >= n_bars:
            break
        base_start = bp
        base_len = 5
        base_low_ = current_price
        base_high = current_price + rng.uniform(0.001, 0.0025)
        for j in range(base_len):
            idx = base_start + j
            if idx < n_bars:
                close[idx] = rng.uniform(base_low_, base_high)
                high[idx] = base_high + rng.uniform(0, 0.00005)
                low[idx] = base_low_ - rng.uniform(0, 0.00005)
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

    # Fill gaps
    for i in range(n_bars):
        if close[i] == 0:
            noise = rng.normal(0.0001, 0.0001)
            current_price += noise
            close[i] = current_price
            high[i] = current_price + rng.uniform(0.00005, 0.0002)
            low[i] = current_price - rng.uniform(0.00005, 0.0002)
            volume[i] = rng.uniform(900, 1500)

    return pd.DataFrame(
        {
            "time": dates,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def run_backtest(df: pd.DataFrame, bot: TradingBot) -> pd.DataFrame:
    executor = BacktestExecutor(
        df=df,
        detector=bot.detector,
        ai_filter=bot.ai_filter,
        risk_mgr=bot.risk_mgr,
        initial_balance=bot.balance,
    )
    # Wire strategy loop
    executor.on_bar = bot.on_bar  # type: ignore
    return executor.run()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    mode = "backtest"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode == "optimize":
        print("=== Parameter Optimization Not Implemented in Modular Version ===")
        print("Use old aitradingbot.py for optimization, or implement optimizer over new modules.")
        sys.exit(0)

    bot = TradingBot()

    if mode == "live":
        print("Starting live trading ...")
        bot.start_live()
    else:
        print("Running backtest on demo data.\n")
        df = generate_demo_data(n_bars=6000)
        trades = run_backtest(df, bot)
        print("\nBacktest completed. See summary above.")


if __name__ == "__main__":
    main()
