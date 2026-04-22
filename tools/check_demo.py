import numpy as np
import pandas as pd

def generate_demo_data(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    close = np.zeros(n_bars)
    high = np.zeros(n_bars)
    low = np.zeros(n_bars)
    volume = np.zeros(n_bars)

    current_price = 1.1000
    for i in range(50):
        trend = (i / 50) * 0.0008
        noise = rng.normal(trend, 0.00015)
        current_price += noise
        close[i] = current_price
        high[i] = current_price + rng.uniform(0.00005, 0.0002)
        low[i] = current_price - rng.uniform(0.00005, 0.0002)
        volume[i] = rng.uniform(800, 1200)

    breakout_points = [80, 220, 380]
    for bp in breakout_points:
        if bp + 20 >= n_bars:
            break
        base_start = bp
        base_len = 5
        base_low = current_price
        base_high = current_price + rng.uniform(0.001, 0.0025)
        for j in range(base_len):
            idx = base_start + j
            if idx < n_bars:
                close[idx] = rng.uniform(base_low, base_high)
                high[idx] = base_high + rng.uniform(0, 0.00005)
                low[idx] = base_low - rng.uniform(0, 0.00005)
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

    for i in range(n_bars):
        if close[i] == 0:
            noise = rng.normal(0.0001, 0.0001)
            current_price += noise
            close[i] = current_price
            high[i] = current_price + rng.uniform(0.00005, 0.0002)
            low[i] = current_price - rng.uniform(0.00005, 0.0002)
            volume[i] = rng.uniform(900, 1500)

    df = pd.DataFrame({
        'time': dates,
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    return df

if __name__ == '__main__':
    for n in (6000, 100):
        df = generate_demo_data(n_bars=n)
        c = df['close'].values
        print(f'n_bars={n} min={c.min():.6f} max={c.max():.6f}')
        in_range = ((c >= 6000) & (c <= 7500)).any()
        print('Any value in 6000-7500?', in_range)
