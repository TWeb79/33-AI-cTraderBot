"""
Main orchestrator: connects core strategy to execution layer.
Can run in backtest mode (DataFrame) or live mode (cTrader API).
"""
import logging
from typing import Optional
import pandas as pd
import configparser
import os

from .core.types import Bar, TradeState, Signal
from .core.patterns import PatternDetector
from .core.features import FeatureEngine
from .core.regime import MarketRegimeDetector, LLMRegimeDetector
from .core.ai_filter import AITradeFilter
from .core.risk import RiskManager
from .execution.backtest_executor import BacktestExecutor
from .execution.live_executor import LiveExecutor
from .execution.base import IMarketDataHandler


class TradingBot:
    """
    Unified trading bot — stateless orchestrator.
    Wires together:
      PatternDetector → FeatureEngine + MarketRegimeDetector → AITradeFilter → RiskManager
    and delegates execution to an IExecutionHandler implementation.
    """

    def __init__(
        self,
        config_path: str = "config.ini",
        executor: Optional[IMarketDataHandler] = None,
    ):
        self._load_config(config_path)
        self.log = logging.getLogger("TradingBot")

        # Core components
        self.detector = PatternDetector(
            bar_buffer_size=50,
            base_min_bars=3,
            base_max_bars=15,
            base_atr_ratio=self.BASE_ATR_RATIO,
            volume_surge_mult=self.VOLUME_SURGE_MULT,
            wedge_bars=5,
            exhaustion_atr_mult=self.EXHAUSTION_ATR_MULT,
        )
        self.feature_engine = FeatureEngine(bar_buffer_size=200)
        self.volatility_detector = MarketRegimeDetector()
        self.llm_detector = LLMRegimeDetector(
            enabled=self.USE_LLM_REGIME,
            model_name=self.OLLAMA_MODEL,
            use_lightllm=(self.LLM_BACKEND == "lightllm"),
            lightllm_model_path=self.LIGHTLLM_MODEL_PATH,
        )
        self.ai_filter = AITradeFilter(
            model_path=self.MODEL_PATH,
            model_type=self.MODEL_TYPE,
            threshold=self.PROBABILITY_THRESHOLD,
        )
        self.risk_mgr = RiskManager()

        # Wire engines into detector
        self.detector.feature_engine = self.feature_engine
        self.detector.volatility_detector = self.volatility_detector

        # State
        self.state = TradeState()
        self.balance = 10_000.0
        self.trade_log: list[dict] = []
        self.feature_history: list[dict] = []
        self.outcome_history: list[int] = []

        # Executor (backtest or live)
        self.executor = executor

    def _load_config(self, path: str):
        _config = configparser.ConfigParser()
        _config.read(path)

        def _get(section, option, default):
            if _config.has_section(section):
                return _config.get(section, option, fallback=default)
            return default

        # cTrader
        self.CLIENT_ID = _get("ctrader", "CLIENT_ID", os.environ.get("CLIENT_ID", "YOUR_CLIENT_ID"))
        self.CLIENT_SECRET = _get("ctrader", "CLIENT_SECRET", os.environ.get("CLIENT_SECRET", "YOUR_CLIENT_SECRET"))
        self.ACCESS_TOKEN = _get("ctrader", "ACCESS_TOKEN", os.environ.get("ACCESS_TOKEN", "YOUR_ACCESS_TOKEN"))
        try:
            self.ACCOUNT_ID = int(_get("ctrader", "ACCOUNT_ID", os.environ.get("ACCOUNT_ID", "123456")))
        except ValueError:
            self.ACCOUNT_ID = 123456
        self.SYMBOL_ID = 1  # TODO: make configurable

        # AI
        self.MODEL_TYPE = _get("ai", "MODEL_TYPE", "gb").lower()
        self.MODEL_PATH = "ai_trade_model.pkl"

        # LLM
        self.USE_LLM_REGIME = _get("llm", "enabled", "false").lower() == "true"
        self.LLM_BACKEND = _get("llm", "backend", "ollama").lower()  # "ollama" or "lightllm"
        self.OLLAMA_MODEL = _get("llm", "model", "llama3.2")
        self.LIGHTLLM_MODEL_PATH = _get("llm", "lightllm_model_path", "")

        # Strategy parameters
        self.EMA_FAST = int(_get("strategy", "EMA_FAST", "10"))
        self.EMA_SLOW = int(_get("strategy", "EMA_SLOW", "20"))
        self.ATR_PERIOD = int(_get("strategy", "ATR_PERIOD", "14"))
        self.EXHAUSTION_ATR_MULT = float(_get("strategy", "EXHAUSTION_ATR_MULT", "3.0"))
        self.BASE_ATR_RATIO = float(_get("strategy", "BASE_ATR_RATIO", "0.5"))
        self.VOLUME_SURGE_MULT = float(_get("strategy", "VOLUME_SURGE_MULT", "1.5"))
        self.TRAIL_ATR_MULT = float(_get("strategy", "TRAIL_ATR_MULT", "1.5"))
        self.PROBABILITY_THRESHOLD = float(_get("strategy", "PROBABILITY_THRESHOLD", "0.65"))
        self.RISK_PERCENT = float(_get("strategy", "RISK_PERCENT", "1.5"))

    def on_bar(self, bar: Bar):
        """
        Entry point for each new bar (from executor).
        Handles signal detection, regime filtering, entry, exit, trailing.
        """
        signal = self.detector.push(bar)
        indic = self.detector.latest_indicators()
        atr = indic.get("atr", 0.0)

        if len(self.detector.bars) > 20 and signal != Signal.NONE:
            self.log.info(
                f"[DEBUG] bar={bar.close:.5f} ema_f={indic.get('ema_fast',0):.5f} "
                f"ema_s={indic.get('ema_slow',0):.5f} signal={signal.value}"
            )

        if self.state.in_position and atr > 0:
            new_trail = self.risk_mgr.update_trail_stop(bar.close, self.state.trail_stop, atr, self.TRAIL_ATR_MULT)
            if new_trail != self.state.trail_stop:
                self.state.trail_stop = new_trail
                self.log.debug(f"Trail stop updated -> {new_trail:.5f}")
            if bar.low <= self.state.trail_stop:
                self._close_position(bar, reason="TRAIL_STOP")
                return

        if signal == Signal.EXIT_EXHAUST:
            if self.state.in_position:
                self._close_position(bar, reason="EXHAUSTION_EXTENSION")
        elif signal == Signal.EXIT_CROSSBACK:
            if self.state.in_position:
                self._close_position(bar, reason="EMA_CROSSBACK")
        elif signal in (Signal.BASE_BREAK, Signal.WEDGE_POP):
            if not self.state.in_position:
                features = self.detector.extract_features(self.feature_engine)
                # Regime gating
                regime = self._get_regime()
                if regime in ("RANGING", "VOLATILE", "UNKNOWN"):
                    self.log.info(f"[REGIME FILTER] Skipping trade | regime={regime}")
                    return
                if self.ai_filter.should_trade(features):
                    self.feature_history.append(features)
                    self._open_position(bar, signal, atr)
                else:
                    confidence, _ = self.ai_filter.predict_probability(features)
                    self.log.info(
                        f"[AI FILTER] Trade rejected | confidence={confidence:.2f} "
                        f"< {self.ai_filter.threshold:.2f}"
                    )
        elif signal == Signal.WEDGE_DROP:
            self.log.info(
                f"[CONTEXT] Wedge Drop detected | price={bar.close:.5f} "
                f"regime={self._get_regime()}"
            )

    def _get_regime(self) -> str:
        llm = self.llm_detector.detect("")
        if llm:
            return llm
        return self.volatility_detector.detect()

    def _open_position(self, bar: Bar, signal: Signal, atr: float):
        base_low = self.detector.get_base_low()
        stop_loss = base_low - 0.5 * atr
        entry = bar.close

        features = self.detector.extract_features(self.feature_engine)
        confidence, _ = self.ai_filter.predict_probability(features)

        risk_pct = self.RISK_PERCENT * self.ai_filter.get_confidence_risk(confidence)
        units = self.risk_mgr.position_size(self.balance, risk_pct, entry, stop_loss)

        if units <= 0:
            self.log.warning("Calculated 0 units - skipping entry.")
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

        self.log.info(
            f"[ENTRY] {signal.value} | Price={entry:.5f} "
            f"SL={stop_loss:.5f} Units={units} Confidence={confidence:.2f} @ {bar.time}"
        )

        if self.executor:
            self.executor.place_market_order(entry, stop_loss, units, "BUY")

        self._try_retrain()

    def _close_position(self, bar: Bar, reason: str):
        exit_price = bar.close
        pnl = (exit_price - self.state.entry_price) * self.state.units

        self.log.info(
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

        if self.executor:
            self.executor.close_position()

        self._try_retrain()

    def _try_retrain(self):
        if not self.feature_history or not self.outcome_history or len(self.outcome_history) < 10:
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
                self.log.info("Insufficient class variation for training.")
                return
            self.ai_filter.train(X, y)
            self.feature_history = []
            self.outcome_history = []
            self.log.info("AI model retrained with latest trade data.")
        except Exception as e:
            self.log.warning(f"Model retraining failed: {e}")

    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        executor = BacktestExecutor(
            df=df,
            detector=self.detector,
            ai_filter=self.ai_filter,
            risk_mgr=self.risk_mgr,
            initial_balance=self.balance,
        )
        # Wire on_bar callback
        executor.on_bar = self.on_bar  # type: ignore
        return executor.run()

    def start_live(self):
        if not self._check_live_deps():
            return
        executor = LiveExecutor(
            client_id=self.CLIENT_ID,
            client_secret=self.CLIENT_SECRET,
            access_token=self.ACCESS_TOKEN,
            account_id=self.ACCOUNT_ID,
            symbol_id=self.SYMBOL_ID,
            detector=self.detector,
            ai_filter=self.ai_filter,
            risk_mgr=self.risk_mgr,
            initial_balance=self.balance,
        )
        executor.on_bar = self.on_bar  # type: ignore
        executor.start()

    def _check_live_deps(self) -> bool:
        try:
            from ctrader_open_api import Client  # noqa: F401
            return True
        except ImportError:
            self.log.error("ctrader-open-api not installed. Cannot start live mode.")
            return False
