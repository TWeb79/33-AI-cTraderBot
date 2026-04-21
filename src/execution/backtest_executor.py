"""
Backtest executor: runs strategy against historical DataFrame.
"""
import pandas as pd
import logging
from typing import Optional, Callable

from ..core.types import Bar, TradeState, Signal
from ..core.patterns import PatternDetector
from ..core.features import FeatureEngine
from ..core.regime import MarketRegimeDetector
from ..core.ai_filter import AITradeFilter
from ..core.risk import RiskManager
from .base import IMarketDataHandler


class BacktestExecutor(IMarketDataHandler):
    """
    Pulls bars from a DataFrame, feeds them through the strategy pipeline,
    and records trade results.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        detector: PatternDetector,
        ai_filter: AITradeFilter,
        risk_mgr: RiskManager,
        initial_balance: float = 10_000.0,
    ):
        self.df = df.reset_index(drop=True)
        self.current_index = 0
        self.detector = detector
        self.ai_filter = ai_filter
        self.risk_mgr = risk_mgr
        self.balance = initial_balance
        self.trade_log: list[dict] = []
        self.feature_history: list[dict] = []
        self.outcome_history: list[int] = []
        self.state = TradeState()
        self._bar_callback: Optional[Callable[[Bar], None]] = None

    def subscribe(self, callback: Callable[[Bar], None]) -> None:
        self._bar_callback = callback

    def unsubscribe(self) -> None:
        self._bar_callback = None

    def get_latest_bar(self) -> Optional[Bar]:
        if self.current_index == 0:
            return None
        row = self.df.iloc[self.current_index - 1]
        return Bar(
            time=row["time"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 1000)),
        )

    def run(self) -> pd.DataFrame:
        logging.info(f"Starting backtest on {len(self.df)} bars ...")
        for idx, row in self.df.iterrows():
            bar = Bar(
                time=row["time"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 1000)),
            )
            if self._bar_callback:
                self._bar_callback(bar)
            self.current_index += 1

        # Close any open position at last bar
        if self.state.in_position:
            last_row = self.df.iloc[-1]
            last_bar = Bar(
                time=last_row["time"],
                open=float(last_row["open"]),
                high=float(last_row["high"]),
                low=float(last_row["low"]),
                close=float(last_row["close"]),
                volume=float(last_row.get("volume", 1000)),
            )
            self._close_position(last_bar, reason="END_OF_DATA")

        trades_df = pd.DataFrame(self.trade_log)
        self._print_summary(trades_df)
        return trades_df

    def on_bar(self, bar: Bar):
        """
        Called per bar by the executor. Contains strategy logic.
        Mirrors original AITradingBot._on_bar but decoupled from API.
        """
        signal = self.detector.push(bar)
        indic = self.detector.latest_indicators()
        atr = indic.get("atr", 0.0)

        # Debug logging
        if len(self.detector.bars) > 20 and signal != Signal.NONE:
            logging.info(
                f"[DEBUG] bar={bar.close:.5f} ema_f={indic.get('ema_fast',0):.5f} "
                f"ema_s={indic.get('ema_slow',0):.5f} signal={signal.value}"
            )

        # Trailing stop management
        if self.state.in_position and atr > 0:
            new_trail = self.risk_mgr.update_trail_stop(bar.close, self.state.trail_stop, atr)
            if new_trail != self.state.trail_stop:
                self.state.trail_stop = new_trail
                logging.debug(f"Trail stop updated -> {new_trail:.5f}")
            if bar.low <= self.state.trail_stop:
                self._close_position(bar, reason="TRAIL_STOP")
                return

        # Exit signals
        if signal == Signal.EXIT_EXHAUST:
            if self.state.in_position:
                self._close_position(bar, reason="EXHAUSTION_EXTENSION")
        elif signal == Signal.EXIT_CROSSBACK:
            if self.state.in_position:
                self._close_position(bar, reason="EMA_CROSSBACK")
        # Entry signals
        elif signal in (Signal.BASE_BREAK, Signal.WEDGE_POP):
            if not self.state.in_position:
                features = self.detector.extract_features(self.detector.feature_engine)
                regime = (
                    self.detector.volatility_detector.detect()
                    if self.detector.volatility_detector
                    else "UNKNOWN"
                )
                if regime in ("RANGING", "VOLATILE", "UNKNOWN"):
                    logging.info(f"[REGIME FILTER] Skipping trade | regime={regime}")
                    return

                if self.ai_filter.should_trade(features):
                    self.feature_history.append(features)
                    self._open_position(bar, signal, atr)
                else:
                    confidence, _ = self.ai_filter.predict_probability(features)
                    logging.info(
                        f"[AI FILTER] Trade rejected | confidence={confidence:.2f} "
                        f"< {self.ai_filter.threshold:.2f}"
                    )
        elif signal == Signal.WEDGE_DROP:
            logging.info(
                f"[CONTEXT] Wedge Drop detected | price={bar.close:.5f} "
                f"regime={self.detector.volatility_detector.detect() if self.detector.volatility_detector else 'UNKNOWN'}"
            )

    def _open_position(self, bar: Bar, signal: Signal, atr: float):
        base_low = self.detector.get_base_low()
        stop_loss = base_low - 0.5 * atr
        entry = bar.close

        features = self.detector.extract_features(self.detector.feature_engine)
        confidence, _ = self.ai_filter.predict_probability(features)

        risk_pct = 1.5 * self.ai_filter.get_confidence_risk(confidence)
        units = self.risk_mgr.position_size(self.balance, risk_pct, entry, stop_loss)

        if units <= 0:
            logging.warning("Calculated 0 units - skipping entry.")
            return

        self.state = TradeState(
            in_position=True,
            entry_price=entry,
            stop_loss=stop_loss,
            units=units,
            trail_stop=stop_loss,
            entry_time=bar.time,
            base_low=base_low,
            confidence=confidence,
        )

        logging.info(
            f"[ENTRY] {signal.value} | Price={entry:.5f} "
            f"SL={stop_loss:.5f} Units={units} Confidence={confidence:.2f} @ {bar.time}"
        )

        self._try_retrain()

    def _close_position(self, bar: Bar, reason: str):
        exit_price = bar.close
        pnl = (exit_price - self.state.entry_price) * self.state.units

        logging.info(
            f"[EXIT] {reason} | Price={exit_price:.5f} "
            f"PnL={pnl:+.2f} @ {bar.time}"
        )

        trade_record = {
            "entry_time": self.state.entry_time,
            "exit_time": bar.time,
            "entry": self.state.entry_price,
            "exit": exit_price,
            "units": self.state.units,
            "pnl": pnl,
            "reason": reason,
            "confidence": self.state.confidence,
        }

        if self.feature_history and self.state.entry_time:
            features = self.feature_history[-1]
            outcome = 1 if pnl > 0 else 0
            self.outcome_history.append(outcome)

        self.trade_log.append(trade_record)
        self.balance += pnl
        self.state = TradeState()

        self._try_retrain()

    def _try_retrain(self):
        if not HAS_SKLEARN or not self.feature_history or not self.outcome_history:
            return
        if len(self.outcome_history) < 10:
            return
        try:
            feature_list = [
                [
                    f.get("ema_slope", 0),
                    f.get("ema_distance_ratio", 0),
                    f.get("atr_expansion", 1),
                    f.get("base_tightness", 0),
                    f.get("breakout_velocity", 0),
                    f.get("volume_ratio", 1),
                    f.get("mtf_alignment", 0.0),
                    f.get("trend_maturity", 0),
                    f.get("regime_score", 0.0),
                    f.get("asia_session", 0.0),
                    f.get("london_session", 0.0),
                    f.get("ny_session", 0.0),
                ]
                for f in self.feature_history
            ]
            X = np.array(feature_list)
            y = np.array(self.outcome_history)
            if len(np.unique(y)) < 2:
                logging.info("Insufficient class variation for training.")
                return
            self.ai_filter.train(X, y)
            self.feature_history = []
            self.outcome_history = []
            logging.info("AI model retrained with latest trade data.")
        except Exception as e:
            logging.warning(f"Model retraining failed: {e}")

    def _print_summary(self, trades: pd.DataFrame):
        if trades.empty:
            logging.info("No trades executed.")
            return
        total_pnl = trades["pnl"].sum()
        win_rate = (trades["pnl"] > 0).mean() * 100
        avg_win = trades.loc[trades["pnl"] > 0, "pnl"].mean() if (trades["pnl"] > 0).any() else 0
        avg_loss = trades.loc[trades["pnl"] < 0, "pnl"].mean() if (trades["pnl"] < 0).any() else 0
        profit_factor = (
            trades.loc[trades["pnl"] > 0, "pnl"].sum()
            / abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
            if (trades["pnl"] < 0).any()
            else float("inf")
        )
        avg_confidence = trades["confidence"].mean() if "confidence" in trades.columns else 0

        print("\n" + "=" * 60)
        print("   AI TRADING BOT - BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Total trades      : {len(trades)}")
        print(f"  Win rate          : {win_rate:.1f}%")
        print(f"  Total PnL         : {total_pnl:+.2f}")
        print(f"  Avg win           : {avg_win:+.2f}")
        print(f"  Avg loss          : {avg_loss:.2f}")
        print(f"  Profit factor     : {profit_factor:.2f}")
        print(f"  Avg confidence    : {avg_confidence:.2f}")
        print(f"  Final balance     : {self.balance:.2f}")
        print("=" * 60)
        print(trades[["entry_time", "exit_time", "entry", "exit", "pnl", "confidence", "reason"]].to_string(index=False))
        print()
