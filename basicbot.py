"""
==============================================================================
Oliver Kell Trading Strategy — cTrader Open API Python Bot
==============================================================================
Strategy:  Base 'n Break + Wedge Pop entries above 10/20 EMA
           Exit on Exhaustion Extension or EMA Crossback
Risk Mgmt: 1–2% account risk per trade, trailing stop

Requirements:
    pip install ctrader-open-api pandas numpy twisted

Usage:
    1. Fill in your CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, ACCOUNT_ID
       from https://openapi.ctrader.com/
    2. Set SYMBOL_NAME, TIMEFRAME, and RISK_PERCENT as needed
    3. Run:  python oliver_kell_bot.py
==============================================================================
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Deque, List, Optional
import os
import configparser

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# cTrader Open API imports (install: pip install ctrader-open-api)
# ---------------------------------------------------------------------------
try:
    from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
    from ctrader_open_api.messages.OpenApiMessages_pb2 import *
    from twisted.internet import reactor, defer
    LIVE_MODE = True
except ImportError:
    LIVE_MODE = False
    print("[WARNING] ctrader-open-api not installed. Running in SIMULATION mode.")
    print("          pip install ctrader-open-api twisted\n")

# ---------------------------------------------------------------------------
# ─── CONFIGURATION ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# Credentials are read from config.ini (ignored by git) or environment variables.
# Create a config.ini in the project root with a [ctrader] section, or set
# CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, ACCOUNT_ID environment variables.
_config = configparser.ConfigParser()
_config.read("config.ini")

CLIENT_ID = _config.get("ctrader", "CLIENT_ID", fallback=os.environ.get("CLIENT_ID", "YOUR_CLIENT_ID"))
CLIENT_SECRET = _config.get("ctrader", "CLIENT_SECRET", fallback=os.environ.get("CLIENT_SECRET", "YOUR_CLIENT_SECRET"))
ACCESS_TOKEN = _config.get("ctrader", "ACCESS_TOKEN", fallback=os.environ.get("ACCESS_TOKEN", "YOUR_ACCESS_TOKEN"))
try:
    ACCOUNT_ID = int(_config.get("ctrader", "ACCOUNT_ID", fallback=os.environ.get("ACCOUNT_ID", "123456")))
except ValueError:
    ACCOUNT_ID = 123456

SYMBOL_NAME     = "EURUSD"        # instrument to trade
TIMEFRAME       = "H1"            # M1 M5 M15 M30 H1 H4 D1
RISK_PERCENT    = 1.5             # % of balance to risk per trade (1–2)

# EMA periods
EMA_FAST        = 10
EMA_SLOW        = 20

# Pattern detection parameters
BASE_MIN_BARS       = 3           # minimum bars in a base/consolidation
BASE_MAX_BARS       = 15          # maximum bars in a base
BASE_ATR_RATIO      = 0.5         # base height ≤ this × ATR  → "tight"
ATR_PERIOD          = 14
VOLUME_SURGE_MULT   = 1.5         # breakout volume ≥ this × avg volume
EXHAUSTION_ATR_MULT = 3.0         # price distance from 20 EMA > this × ATR
WEDGE_BARS          = 5           # bars to look back for wedge pattern

# Trailing stop
TRAIL_ATR_MULT      = 1.5         # trail stop distance = this × ATR below price

# ---------------------------------------------------------------------------
# ─── DATA STRUCTURES ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class Signal(Enum):
    NONE       = "NONE"
    BASE_BREAK = "BASE_N_BREAK"
    WEDGE_POP  = "WEDGE_POP"
    EXIT_EXHAUST = "EXIT_EXHAUSTION"
    EXIT_CROSSBACK = "EXIT_EMA_CROSSBACK"

@dataclass
class Bar:
    time:   datetime
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float

@dataclass
class TradeState:
    in_position:    bool  = False
    entry_price:    float = 0.0
    stop_loss:      float = 0.0
    units:          int   = 0
    trail_stop:     float = 0.0
    position_id:    Optional[int] = None
    entry_time:     Optional[datetime] = None
    base_low:       float = 0.0   # lowest point of the breakout base

# ---------------------------------------------------------------------------
# ─── INDICATOR ENGINE ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class Indicators:
    """Lightweight incremental indicator calculator."""

    def __init__(self, fast: int = EMA_FAST, slow: int = EMA_SLOW,
                 atr_period: int = ATR_PERIOD):
        self.fast = fast
        self.slow = slow
        self.atr_period = atr_period

        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._prev_close: Optional[float] = None

        self._tr_window:     Deque[float] = deque(maxlen=atr_period)
        self._vol_window:    Deque[float] = deque(maxlen=20)

        self.ema_fast_series: List[float] = []
        self.ema_slow_series: List[float] = []
        self.atr_series:      List[float] = []

        self._k_fast = 2 / (fast + 1)
        self._k_slow = 2 / (slow + 1)

    def update(self, bar: Bar) -> dict:
        """Feed a new closed bar and return current indicator values."""
        # EMA
        if self._ema_fast is None:
            self._ema_fast = bar.close
            self._ema_slow = bar.close
        else:
            self._ema_fast += self._k_fast * (bar.close - self._ema_fast)
            self._ema_slow += self._k_slow * (bar.close - self._ema_slow)

        # ATR (Wilder's True Range)
        if self._prev_close is not None:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - self._prev_close),
                abs(bar.low  - self._prev_close),
            )
        else:
            tr = bar.high - bar.low
        self._tr_window.append(tr)
        atr = float(np.mean(self._tr_window))

        # Volume average
        self._vol_window.append(bar.volume)
        avg_vol = float(np.mean(self._vol_window))

        self._prev_close = bar.close

        self.ema_fast_series.append(self._ema_fast)
        self.ema_slow_series.append(self._ema_slow)
        self.atr_series.append(atr)

        return {
            "ema_fast": self._ema_fast,
            "ema_slow": self._ema_slow,
            "atr": atr,
            "avg_volume": avg_vol,
        }

# ---------------------------------------------------------------------------
# ─── PATTERN DETECTOR ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class PatternDetector:
    """
    Detects Oliver Kell's key patterns on a rolling bar buffer.

    Patterns:
        • Base 'n Break  — tight consolidation then volume breakout
        • Wedge Pop      — compressed pullback followed by upside burst
        • Exhaustion Ext — price parabolic far above 20 EMA
        • EMA Crossback  — close below both EMAs
    """

    def __init__(self, bar_buffer_size: int = 50):
        self.bars:  Deque[Bar]   = deque(maxlen=bar_buffer_size)
        self.indic: Indicators   = Indicators()
        self._indic_vals: List[dict] = []

    def push(self, bar: Bar) -> Signal:
        self.bars.append(bar)
        vals = self.indic.update(bar)
        self._indic_vals.append(vals)
        if len(self.bars) < BASE_MAX_BARS + 5:
            return Signal.NONE
        return self._evaluate(vals)

    # ── private helpers ──────────────────────────────────────────────────────

    def _evaluate(self, vals: dict) -> Signal:
        bars = list(self.bars)
        latest = bars[-1]
        ema_f  = vals["ema_fast"]
        ema_s  = vals["ema_slow"]
        atr    = vals["atr"]
        avg_v  = vals["avg_volume"]

        # 1. EMA Crossback exit (highest priority when in position)
        if latest.close < ema_f and latest.close < ema_s:
            return Signal.EXIT_CROSSBACK

        # 2. Exhaustion Extension exit
        dist_from_slow_ema = latest.close - ema_s
        if atr > 0 and (dist_from_slow_ema / atr) > EXHAUSTION_ATR_MULT:
            # Only signal if we are significantly extended
            if latest.close > ema_f > ema_s:
                return Signal.EXIT_EXHAUST

        # Must be above both EMAs for entry signals
        if latest.close < ema_f or latest.close < ema_s:
            return Signal.NONE

        # 3. Base 'n Break
        base_signal = self._detect_base_break(bars, ema_s, atr, avg_v)
        if base_signal != Signal.NONE:
            return base_signal

        # 4. Wedge Pop
        wedge_signal = self._detect_wedge_pop(bars, ema_f, avg_v)
        if wedge_signal != Signal.NONE:
            return wedge_signal

        return Signal.NONE

    def _detect_base_break(self, bars: list, ema_slow: float,
                            atr: float, avg_vol: float) -> Signal:
        """
        Look for a tight base (BASE_MIN_BARS–BASE_MAX_BARS bars) followed by
        a breakout candle with strong volume on the latest bar.
        """
        latest = bars[-1]

        for base_len in range(BASE_MIN_BARS, BASE_MAX_BARS + 1):
            if len(bars) < base_len + 1:
                break

            base_bars   = bars[-(base_len + 1):-1]   # the potential base
            breakout_bar = bars[-1]                   # current bar = breakout

            base_high = max(b.high  for b in base_bars)
            base_low  = min(b.low   for b in base_bars)
            base_range = base_high - base_low

            # Tight base: range ≤ BASE_ATR_RATIO × ATR
            if atr > 0 and base_range > BASE_ATR_RATIO * atr:
                continue

            # Base must be above the slow EMA
            if base_low < ema_slow * 0.995:
                continue

            # Breakout: close above the base high
            if breakout_bar.close <= base_high:
                continue

            # Volume surge
            if avg_vol > 0 and breakout_bar.volume < VOLUME_SURGE_MULT * avg_vol:
                continue

            # Valid Base 'n Break
            logging.info(
                f"[SIGNAL] Base 'n Break detected | base_len={base_len} "
                f"base_range={base_range:.5f} atr={atr:.5f} "
                f"vol={breakout_bar.volume:.0f} avg_vol={avg_vol:.0f}"
            )
            return Signal.BASE_BREAK

        return Signal.NONE

    def _detect_wedge_pop(self, bars: list, ema_fast: float,
                           avg_vol: float) -> Signal:
        """
        Simplified wedge pop: look for WEDGE_BARS of lower highs + lower lows
        (compression) all above EMA, then a strong up-bar on volume.
        """
        if len(bars) < WEDGE_BARS + 1:
            return Signal.NONE

        wedge = bars[-(WEDGE_BARS + 1):-1]
        latest = bars[-1]

        # Wedge: declining highs
        highs = [b.high for b in wedge]
        lows  = [b.low  for b in wedge]
        declining_highs = all(highs[i] >= highs[i+1] for i in range(len(highs)-1))
        declining_lows  = all(lows[i]  >= lows[i+1]  for i in range(len(lows)-1))

        if not (declining_highs and declining_lows):
            return Signal.NONE

        # All wedge bars above fast EMA
        if any(b.low < ema_fast * 0.995 for b in wedge):
            return Signal.NONE

        # Pop: breakout above the first bar's high on volume
        if latest.close <= highs[0]:
            return Signal.NONE

        if avg_vol > 0 and latest.volume < VOLUME_SURGE_MULT * avg_vol:
            return Signal.NONE

        logging.info("[SIGNAL] Wedge Pop detected")
        return Signal.WEDGE_POP

    def get_base_low(self) -> float:
        """Return the low of the most recent consolidation base (for stop loss)."""
        bars = list(self.bars)
        if len(bars) < BASE_MIN_BARS:
            return bars[-1].low if bars else 0.0
        base_bars = bars[-BASE_MIN_BARS:]
        return min(b.low for b in base_bars)

    def latest_indicators(self) -> dict:
        return self._indic_vals[-1] if self._indic_vals else {}

# ---------------------------------------------------------------------------
# ─── RISK MANAGER ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class RiskManager:
    """Calculates position size and manages trailing stop."""

    @staticmethod
    def position_size(balance: float, risk_pct: float,
                      entry: float, stop: float,
                      pip_value: float = 0.0001) -> int:
        """
        Returns units to trade.
        risk_pct : e.g. 1.5  (means 1.5%)
        pip_value: value of 1 pip in account currency (default forex assumption)
        """
        risk_amount = balance * (risk_pct / 100.0)
        stop_distance = abs(entry - stop)
        if stop_distance < 1e-10:
            return 0
        units = int(risk_amount / stop_distance)
        return max(units, 0)

    @staticmethod
    def update_trail_stop(current_price: float, current_trail: float,
                          atr: float) -> float:
        """Move trail stop up if price has moved in our favour."""
        new_trail = current_price - TRAIL_ATR_MULT * atr
        return max(new_trail, current_trail)   # never move stop down

# ---------------------------------------------------------------------------
# ─── MAIN BOT ───────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class BasicBot:
    """
    Oliver Kell Strategy — cTrader Python Bot

    In live mode  → connects to cTrader Open API via Twisted reactor
    In sim mode   → call `run_on_dataframe(df)` for backtesting
    """

    def __init__(self):
        self.detector   = PatternDetector()
        self.state      = TradeState()
        self.risk       = RiskManager()
        self.balance    = 10_000.0   # updated from API in live mode
        self.log        = logging.getLogger("BasicBot")

        # Trade log for analysis
        self.trade_log: List[dict] = []

    # ── LIVE MODE ────────────────────────────────────────────────────────────

    def start_live(self):
        if not LIVE_MODE:
            self.log.error("ctrader-open-api not installed. Cannot start live mode.")
            return

        self.log.info("Connecting to cTrader Open API …")
        self.client = Client(
            EndPoints.PROTOBUF_LIVE_HOST,
            EndPoints.PROTOBUF_PORT,
            TcpProtocol,
        )
        self.client.setConnectedCallback(self._on_connected)
        self.client.setDisconnectedCallback(self._on_disconnected)
        self.client.setMessageReceivedCallback(self._on_message)
        self.client.startService()
        reactor.run()

    def _on_connected(self, client):
        self.log.info("Connected. Authenticating …")
        request = ProtoOAApplicationAuthReq()
        request.clientId     = CLIENT_ID
        request.clientSecret = CLIENT_SECRET
        deferred = client.send(request)
        deferred.addErrback(self._on_error)

    def _on_disconnected(self, client, reason):
        self.log.warning(f"Disconnected: {reason}. Reconnecting in 10s …")
        reactor.callLater(10, self.start_live)

    def _on_message(self, client, message):
        msg_type = Protobuf.extract(message)

        if msg_type == ProtoOAApplicationAuthRes:
            self.log.info("App authenticated. Authorising account …")
            req = ProtoOAAccountAuthReq()
            req.ctidTraderAccountId = ACCOUNT_ID
            req.accessToken         = ACCESS_TOKEN
            client.send(req).addErrback(self._on_error)

        elif msg_type == ProtoOAAccountAuthRes:
            self.log.info("Account authorised. Subscribing to spots …")
            self._subscribe_spots(client)
            self._request_account_info(client)

        elif msg_type == ProtoOASpotEvent:
            spot = Protobuf.extract(message, ProtoOASpotEvent)
            self._on_tick(spot)

        elif msg_type == ProtoOAGetAccountListByAccessTokenRes:
            pass  # handle if needed

        elif msg_type == ProtoOATraderRes:
            trader = Protobuf.extract(message, ProtoOATraderRes)
            self.balance = trader.trader.balance / 100.0  # convert cents
            self.log.info(f"Account balance: {self.balance:.2f}")

    def _subscribe_spots(self, client):
        req = ProtoOASubscribeSpotsReq()
        req.ctidTraderAccountId = ACCOUNT_ID
        req.symbolId.append(self._symbol_id)
        client.send(req)

    def _request_account_info(self, client):
        req = ProtoOATraderReq()
        req.ctidTraderAccountId = ACCOUNT_ID
        self.client.send(req).addErrback(self._on_error)

    def _on_tick(self, spot):
        """Process live tick — aggregate into bars externally or use trend bars."""
        # In production: use ProtoOAGetTrendbarsReq to get OHLCV bars
        # and call self._on_bar() for each new closed bar.
        pass

    def _on_error(self, failure):
        self.log.error(f"API error: {failure}")

    # ── BAR PROCESSING (shared between live and sim) ──────────────────────────

    def _on_bar(self, bar: Bar):
        """Called for every newly closed bar."""
        signal = self.detector.push(bar)
        indic  = self.detector.latest_indicators()
        atr    = indic.get("atr", 0.0)

        # ── Update trailing stop if in position ──
        if self.state.in_position and atr > 0:
            new_trail = self.risk.update_trail_stop(
                bar.close, self.state.trail_stop, atr
            )
            if new_trail != self.state.trail_stop:
                self.state.trail_stop = new_trail
                self.log.debug(f"Trail stop updated → {new_trail:.5f}")
            # Check if stop hit
            if bar.low <= self.state.trail_stop:
                self._close_position(bar, reason="TRAIL_STOP")
                return

        # ── Handle signal ──
        if signal == Signal.EXIT_EXHAUST:
            if self.state.in_position:
                self._close_position(bar, reason="EXHAUSTION_EXTENSION")

        elif signal == Signal.EXIT_CROSSBACK:
            if self.state.in_position:
                self._close_position(bar, reason="EMA_CROSSBACK")

        elif signal in (Signal.BASE_BREAK, Signal.WEDGE_POP):
            if not self.state.in_position:
                self._open_position(bar, signal, atr)
            else:
                # Scale in: add partial position on second signal
                self.log.info(f"[SCALE-IN] {signal.value} while in position — skipping (scale-in disabled)")

    def _open_position(self, bar: Bar, signal: Signal, atr: float):
        base_low  = self.detector.get_base_low()
        stop_loss = base_low - 0.5 * atr   # small buffer below base low
        entry     = bar.close
        units     = self.risk.position_size(
            self.balance, RISK_PERCENT, entry, stop_loss
        )

        if units <= 0:
            self.log.warning("Calculated 0 units — skipping entry.")
            return

        self.state = TradeState(
            in_position  = True,
            entry_price  = entry,
            stop_loss    = stop_loss,
            units        = units,
            trail_stop   = stop_loss,
            entry_time   = bar.time,
            base_low     = base_low,
        )

        self.log.info(
            f"[ENTRY] {signal.value} | Price={entry:.5f} "
            f"SL={stop_loss:.5f} Units={units} @ {bar.time}"
        )

        if LIVE_MODE:
            self._send_market_order(entry, stop_loss, units, "BUY")

    def _close_position(self, bar: Bar, reason: str):
        exit_price = bar.close
        pnl = (exit_price - self.state.entry_price) * self.state.units

        self.log.info(
            f"[EXIT] {reason} | Price={exit_price:.5f} "
            f"PnL={pnl:+.2f} @ {bar.time}"
        )

        self.trade_log.append({
            "entry_time":  self.state.entry_time,
            "exit_time":   bar.time,
            "entry":       self.state.entry_price,
            "exit":        exit_price,
            "units":       self.state.units,
            "pnl":         pnl,
            "reason":      reason,
        })

        self.balance += pnl
        self.state = TradeState()

        if LIVE_MODE:
            self._close_market_order()

    def _send_market_order(self, entry: float, stop: float,
                            units: int, direction: str):
        if not LIVE_MODE:
            return
        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = ACCOUNT_ID
        req.symbolId            = self._symbol_id
        req.orderType           = ProtoOAOrderType.Value("MARKET")
        req.tradeSide           = (ProtoOATradeSide.Value("BUY")
                                   if direction == "BUY"
                                   else ProtoOATradeSide.Value("SELL"))
        req.volume              = units * 100  # cTrader volume in lots*100
        req.stopLoss            = int(stop * 100_000)
        self.client.send(req).addErrback(self._on_error)

    def _close_market_order(self):
        if not LIVE_MODE or self.state.position_id is None:
            return
        req = ProtoOAClosePositionReq()
        req.ctidTraderAccountId = ACCOUNT_ID
        req.positionId          = self.state.position_id
        req.volume              = self.state.units * 100
        self.client.send(req).addErrback(self._on_error)

    # ── SIMULATION / BACKTEST MODE ────────────────────────────────────────────

    def run_on_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Backtest on a DataFrame with columns:
            time, open, high, low, close, volume

        Returns a DataFrame of all trades.
        """
        self.log.info(f"Starting backtest on {len(df)} bars …")

        for _, row in df.iterrows():
            bar = Bar(
                time   = row["time"],
                open   = float(row["open"]),
                high   = float(row["high"]),
                low    = float(row["low"]),
                close  = float(row["close"]),
                volume = float(row.get("volume", 1000)),
            )
            self._on_bar(bar)

        # Close any open position at end of data
        if self.state.in_position and self.trade_log is not None:
            last = Bar(
                time   = df.iloc[-1]["time"],
                open   = float(df.iloc[-1]["open"]),
                high   = float(df.iloc[-1]["high"]),
                low    = float(df.iloc[-1]["low"]),
                close  = float(df.iloc[-1]["close"]),
                volume = float(df.iloc[-1].get("volume", 1000)),
            )
            self._close_position(last, reason="END_OF_DATA")

        trades_df = pd.DataFrame(self.trade_log)
        self._print_backtest_summary(trades_df)
        return trades_df

    def _print_backtest_summary(self, trades: pd.DataFrame):
        if trades.empty:
            self.log.info("No trades executed.")
            return

        total_pnl   = trades["pnl"].sum()
        win_rate    = (trades["pnl"] > 0).mean() * 100
        avg_win     = trades.loc[trades["pnl"] > 0, "pnl"].mean() if (trades["pnl"] > 0).any() else 0
        avg_loss    = trades.loc[trades["pnl"] < 0, "pnl"].mean() if (trades["pnl"] < 0).any() else 0
        profit_factor = (
            trades.loc[trades["pnl"] > 0, "pnl"].sum() /
            abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
            if (trades["pnl"] < 0).any() else float("inf")
        )

        print("\n" + "="*55)
        print("   OLIVER KELL BOT — BACKTEST RESULTS")
        print("="*55)
        print(f"  Total trades    : {len(trades)}")
        print(f"  Win rate        : {win_rate:.1f}%")
        print(f"  Total PnL       : {total_pnl:+.2f}")
        print(f"  Avg win         : {avg_win:+.2f}")
        print(f"  Avg loss        : {avg_loss:+.2f}")
        print(f"  Profit factor   : {profit_factor:.2f}")
        print(f"  Final balance   : {self.balance:.2f}")
        print("="*55)
        print(trades[["entry_time","exit_time","entry","exit","pnl","reason"]].to_string(index=False))
        print()

# ---------------------------------------------------------------------------
# ─── DEMO: GENERATE SYNTHETIC DATA & BACKTEST ───────────────────────────────
# ---------------------------------------------------------------------------

def generate_demo_data(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic uptrending OHLCV dataset for demonstration.
    Replace this with real data from your broker or a CSV file.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h")

    # Simulate a trending series with occasional pullbacks
    returns = rng.normal(0.0002, 0.003, n_bars)
    # inject trend
    returns[50:100]   += 0.001
    returns[150:200]  += 0.001
    returns[250:350]  += 0.0015
    returns[400:450]  += 0.001

    close = 1.1000 * np.exp(np.cumsum(returns))
    vol   = rng.uniform(800, 2500, n_bars)
    # volume surge on breakout-like bars
    vol[returns > 0.004] *= 2.5

    df = pd.DataFrame({
        "time":   dates,
        "open":   close * rng.uniform(0.999, 1.001, n_bars),
        "high":   close * rng.uniform(1.001, 1.004, n_bars),
        "low":    close * rng.uniform(0.996, 0.999, n_bars),
        "close":  close,
        "volume": vol,
    })
    return df


# ---------------------------------------------------------------------------
# ─── ENTRY POINT ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(message)s",
        datefmt = "%H:%M:%S",
    )

    bot = BasicBot()

    if LIVE_MODE and CLIENT_ID != "YOUR_CLIENT_ID":
        # ── LIVE TRADING ──
        print("Starting live trading …")
        bot.start_live()
    else:
        # ── BACKTEST / SIMULATION ──
        print("cTrader API credentials not set — running backtest on demo data.\n")
        demo_df = generate_demo_data(n_bars=600)
        bot.run_on_dataframe(demo_df)