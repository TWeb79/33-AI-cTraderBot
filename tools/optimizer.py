"""Optimizer utilities: brute-force (random/grid) search and ML-guided search
over strategy parameters using historical data and the modular TradingBot.

Usage:
  python tools/optimizer.py --mode brute --n_samples 100
  python tools/optimizer.py --mode ml --initial 100 --candidates 200 --rounds 3

Outputs best config to data/best_config.json
"""
import json
import random
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import load_historic_data
from src.bot import TradingBot


DEFAULT_GRID = {
    "EMA_FAST": [8, 10, 12],
    "EMA_SLOW": [18, 20, 24],
    "ATR_PERIOD": [10, 14, 20],
    "EXHAUSTION_ATR_MULT": [2.5, 3.0, 3.5],
    "BASE_ATR_RATIO": [0.3, 0.5, 0.8],
    "VOLUME_SURGE_MULT": [1.2, 1.5, 2.0],
    "PROBABILITY_THRESHOLD": [0.55, 0.65, 0.75],
    "RISK_PERCENT": [0.5, 1.0, 1.5],
    "TRAIL_ATR_MULT": [1.0, 1.5, 2.0],
}


def sample_random_config(grid: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {k: random.choice(v) for k, v in grid.items()}


def evaluate_config(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    # Create bot and apply config
    bot = TradingBot()
    # apply strategy params
    bot.EMA_FAST = int(config.get("EMA_FAST", bot.EMA_FAST))
    bot.EMA_SLOW = int(config.get("EMA_SLOW", bot.EMA_SLOW))
    bot.ATR_PERIOD = int(config.get("ATR_PERIOD", bot.ATR_PERIOD))
    bot.EXHAUSTION_ATR_MULT = float(config.get("EXHAUSTION_ATR_MULT", bot.EXHAUSTION_ATR_MULT))
    bot.BASE_ATR_RATIO = float(config.get("BASE_ATR_RATIO", bot.BASE_ATR_RATIO))
    bot.VOLUME_SURGE_MULT = float(config.get("VOLUME_SURGE_MULT", bot.VOLUME_SURGE_MULT))
    bot.PROBABILITY_THRESHOLD = float(config.get("PROBABILITY_THRESHOLD", bot.PROBABILITY_THRESHOLD))
    bot.RISK_PERCENT = float(config.get("RISK_PERCENT", bot.RISK_PERCENT))
    bot.TRAIL_ATR_MULT = float(config.get("TRAIL_ATR_MULT", bot.TRAIL_ATR_MULT))

    # rebuild core components to pick up new params
    bot.detector = bot.detector.__class__(
        bar_buffer_size=50,
        base_min_bars=3,
        base_max_bars=15,
        base_atr_ratio=bot.BASE_ATR_RATIO,
        volume_surge_mult=bot.VOLUME_SURGE_MULT,
        wedge_bars=5,
        exhaustion_atr_mult=bot.EXHAUSTION_ATR_MULT,
    )
    bot.feature_engine = bot.feature_engine.__class__(bar_buffer_size=200)
    bot.ai_filter = bot.ai_filter.__class__(model_path=bot.ai_filter.model_path, model_type=bot.ai_filter.model_type)
    bot.risk_mgr = bot.risk_mgr.__class__()

    # Run backtest
    trades = bot.run_backtest(df)

    # Defensive handling when trades DataFrame is empty or missing pnl column
    if trades.empty or 'pnl' not in trades.columns:
        total_pnl = 0.0
        win_rate = 0.0
        profit_factor = 0.0
    else:
        total_pnl = trades['pnl'].sum()
        win_rate = (trades['pnl'] > 0).mean()
        total_wins = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
        total_losses = abs(trades.loc[trades['pnl'] < 0, 'pnl'].sum()) if (trades['pnl'] < 0).any() else 0.0
        if total_losses > 0:
            profit_factor = total_wins / total_losses
        else:
            profit_factor = float('inf') if total_wins > 0 else 0.0

    result = {
        "config": config,
        "total_pnl": float(total_pnl),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "n_trades": int(len(trades)),
    }
    return result


def brute_force_search(df: pd.DataFrame, grid=DEFAULT_GRID, n_samples: int = 100) -> Dict[str, Any]:
    results = []
    for i in range(n_samples):
        cfg = sample_random_config(grid)
        res = evaluate_config(df, cfg)
        results.append(res)
        print(f"[{i+1}/{n_samples}] cfg={cfg} pnl={res['total_pnl']:.2f} trades={res['n_trades']}")

    best = max(results, key=lambda r: r['total_pnl'])
    return best


def ml_guided_search(df: pd.DataFrame, grid=DEFAULT_GRID, initial=50, candidates=200, rounds=3) -> Dict[str, Any]:
    # initial random sampling
    samples = []
    X = []
    y = []
    for i in range(initial):
        cfg = sample_random_config(grid)
        res = evaluate_config(df, cfg)
        samples.append(res)
        feat = list(cfg.values())
        X.append(feat)
        y.append(res['total_pnl'])
        print(f"init[{i+1}/{initial}] pnl={res['total_pnl']:.2f}")

    for r in range(rounds):
        model = GradientBoostingRegressor()
        model.fit(np.array(X), np.array(y))

        # sample candidates, predict, evaluate top-k
        cand_cfgs = [sample_random_config(grid) for _ in range(candidates)]
        cand_X = np.array([list(c.values()) for c in cand_cfgs])
        preds = model.predict(cand_X)
        top_idx = np.argsort(preds)[-10:]
        for idx in top_idx:
            cfg = cand_cfgs[idx]
            res = evaluate_config(df, cfg)
            samples.append(res)
            X.append(list(cfg.values()))
            y.append(res['total_pnl'])
            print(f"round{r+1} eval pnl={res['total_pnl']:.2f}")

    best = max(samples, key=lambda r: r['total_pnl'])
    return best


def save_best(best: Dict[str, Any], out_path: str = 'data/best_config.json'):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(best, f, indent=2)
    print('Saved best config to', out_path)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('brute', 'ml'), default='brute')
    p.add_argument('--n_samples', type=int, default=50)
    p.add_argument('--initial', type=int, default=50)
    p.add_argument('--candidates', type=int, default=200)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--symbol', default='US500')
    p.add_argument('--timeframe', default='H1')
    args = p.parse_args()

    df = load_historic_data(args.symbol, args.timeframe)

    if args.mode == 'brute':
        best = brute_force_search(df, n_samples=args.n_samples)
    else:
        best = ml_guided_search(df, initial=args.initial, candidates=args.candidates, rounds=args.rounds)

    save_best(best)
    print('Best config:', best)
