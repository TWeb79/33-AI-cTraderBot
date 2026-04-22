"""
33 AI cTrader Bot — modular refactored version.
Entry point for backtest and live trading.
"""
import sys
import logging
import pandas as pd
import glob
import os
import io

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
    executor.subscribe(bot.on_bar)
    return executor.run()


def load_historic_data(symbol: str, timeframe: str, data_dir: str = "data") -> pd.DataFrame:
    """Load historic data for a symbol/timeframe from the data directory.

    Supports JSON, gzipped JSON, CSV. Matches files like:
      data/US500_H1.json, data/US500_H1_2024.json.gz, data/US500_H1.csv
    Returns a DataFrame with columns: time, open, high, low, close, volume
    """
    pattern = os.path.join(data_dir, f"{symbol}_*{timeframe}*.json*")
    candidates = glob.glob(pattern)
    if not candidates:
        # try CSV
        pattern_csv = os.path.join(data_dir, f"{symbol}_*{timeframe}*.csv*")
        candidates = glob.glob(pattern_csv)

    if not candidates:
        # Attempt to fetch from Yahoo Finance as a fallback and save locally
        print(f"No local historic file found for {symbol} {timeframe}. Attempting to fetch from Yahoo Finance...")
        try:
            import yfinance as yf

            # map common symbol name to Yahoo ticker (US500 -> ^GSPC)
            ticker = "^GSPC" if symbol.upper().startswith("US500") or symbol.upper().startswith("SPX") else symbol
            interval = "60m" if timeframe.upper() in ("H1", "H") else "1d"
            # download last 5 years of hourly data
            yf_df = yf.download(tickers=ticker, period="5y", interval=interval, progress=False)
            if yf_df.empty:
                raise RuntimeError("Yahoo returned no data")

            yf_df = yf_df.reset_index()
            yf_df.rename(columns={"Datetime": "time", "Date": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
            if "time" not in yf_df.columns and "Date" in yf_df.columns:
                yf_df["time"] = pd.to_datetime(yf_df["Date"])
            # Ensure proper columns
            df = yf_df[[col for col in ["time", "open", "high", "low", "close", "volume"] if col in yf_df.columns]]

            # save to data directory for future runs
            os.makedirs(data_dir, exist_ok=True)
            out_path = os.path.join(data_dir, f"{symbol}_{timeframe}.json.gz")
            df.to_json(out_path, orient="records", date_format="iso", compression="gzip")
            print(f"Saved fetched historic data to {out_path}")
            return df
        except Exception as e:
            print(f"Failed to fetch from Yahoo Finance: {e}. Falling back to demo data.")
            # create demo data to match expected format
            demo = generate_demo_data(n_bars=6000)
            os.makedirs(data_dir, exist_ok=True)
            out_path = os.path.join(data_dir, f"{symbol}_{timeframe}_demo.json.gz")
            demo.to_json(out_path, orient="records", date_format="iso", compression="gzip")
            print(f"Saved demo historic data to {out_path}")
            return demo

    path = candidates[0]
    # Reading depending on extension: handle CSV, JSON array, JSON lines, gzipped
    if path.endswith('.csv') or '.csv.' in path:
        df = pd.read_csv(path)
    else:
        import gzip
        import json

        # Peek first non-whitespace character to detect JSON lines vs array
        first_char = None
        try:
            open_func = gzip.open if path.endswith('.gz') else open
            with open_func(path, 'rt', encoding='utf-8', errors='ignore') as fh:
                while True:
                    ch = fh.read(1)
                    if not ch:
                        break
                    if ch.isspace():
                        continue
                    first_char = ch
                    break
        except Exception:
            first_char = None

        lines_mode = True if first_char == '{' else False

        try:
            df = pd.read_json(path, lines=lines_mode, compression='infer')
        except Exception:
            # Fallback: manual line-by-line JSON parsing
            records = []
            open_func = gzip.open if path.endswith('.gz') else open
            with open_func(path, 'rt', encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        # try to skip invalid trailing content
                        continue
                    records.append(obj)
            df = pd.DataFrame.from_records(records)

    # normalize columns
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    elif 'date' in df.columns:
        df['time'] = pd.to_datetime(df['date'])
    else:
        # if time is index
        df = df.reset_index()
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])

    # ensure numeric columns exist
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            df[col] = None

    return df[['time', 'open', 'high', 'low', 'close', 'volume']]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Capture a copy of the log output in-memory so we can print the START of the log
    log_buffer = io.StringIO()
    buf_handler = logging.StreamHandler(log_buffer)
    buf_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    buf_handler.setFormatter(formatter)
    logging.getLogger().addHandler(buf_handler)

    mode = "backtest"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode == "optimize":
        print("=== Parameter Optimization Not Implemented in Modular Version ===")
        print("Use old aitradingbot.py for optimization, or implement optimizer over new modules.")
        sys.exit(0)

    bot = TradingBot()

    # Default symbol/timeframe for backtests
    SYMBOL_NAME = "US500"
    TIMEFRAME = "H1"

    if mode == "live":
        print("Starting live trading ...")
        bot.start_live()
    else:
        print("Running backtest. Trying to load historic data if available...\n")
        try:
            df = load_historic_data(SYMBOL_NAME, TIMEFRAME)
        except Exception as e:
            print(f"Failed to load historic data: {e}")
            df = None

        if df is None:
            print("Historic data not found or failed to load — using demo generator.\n")
            df = generate_demo_data(n_bars=60)

        # Analyze loaded data to show sample AI confidences / predictions
        try:
            analysis = bot.analyze_data(df, sample_examples=5)
            print(f"Data analysis: signals={analysis['n_signals']} mean_confidence={analysis['mean_confidence']:.3f} mean_pred_price={analysis['mean_pred_price']:.4f}")
            if analysis['examples']:
                print("Example signals:\n")
                for ex in analysis['examples']:
                    print(f" time={ex['time']} signal={ex['signal']} conf={ex['confidence']:.3f} pred_next={ex['pred_next_price']:.4f}")
        except Exception as e:
            print(f"Analysis failed: {e}")

        trades = run_backtest(df, bot)
        print("\nBacktest completed. See summary above.")

        # Print the start of the captured log (useful when historic data is large and scroll hides the top)
        try:
            buf_contents = log_buffer.getvalue().splitlines()
            head_lines = 60
            print("\n--- Log start (first %d lines) ---" % head_lines)
            for line in buf_contents[:head_lines]:
                print(line)
            print("--- End log start ---\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
