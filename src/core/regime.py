"""
Market regime detection: volatility-based primary detector + optional LLM overlay.
Regimes: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, UNKNOWN
"""
from typing import Optional
import numpy as np
import logging

from .types import Bar

try:
    from src.core.lightllm_regime import LightLLMRegimeDetector
    HAS_LIGHTLLM = True
except ImportError:
    HAS_LIGHTLLM = False


class MarketRegimeDetector:
    """
    Primary regime detector based on volatility ratio and trend strength.
    No external dependencies — pure technical analysis.
    """

    def __init__(self, lookback: int = 20, atr_threshold: float = 1.5, trend_strength_threshold: float = 0.3):
        self.lookback = lookback
        self.atr_threshold = atr_threshold
        self.trend_strength_threshold = trend_strength_threshold
        self.close_history: list[float] = []
        self.atr_history: list[float] = []
        self.ema_history: list[tuple[float, float]] = []

    def update(self, close: float, atr: float, ema_fast: float, ema_slow: float):
        self.close_history.append(close)
        self.atr_history.append(atr)
        self.ema_history.append((ema_fast, ema_slow))
        if len(self.close_history) > self.lookback * 2:
            self.close_history = self.close_history[-self.lookback * 2 :]
            self.atr_history = self.atr_history[-self.lookback * 2 :]
            self.ema_history = self.ema_history[-self.lookback * 2 :]

    def detect(self) -> str:
        if len(self.close_history) < self.lookback:
            return "UNKNOWN"

        prices = np.array(self.close_history)
        atr_current = self.atr_history[-1] if self.atr_history else 0.0
        atr_avg = np.mean(self.atr_history[-10:]) if len(self.atr_history) >= 10 else atr_current

        volatility_ratio = atr_current / atr_avg if atr_avg > 0 else 1.0

        if volatility_ratio > 2.0:
            return "VOLATILE"

        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        price_range = prices.max() - prices.min()
        if price_range > 1e-10:
            trend_strength = abs(slope * len(prices)) / price_range
        else:
            trend_strength = 0

        ema_fast, ema_slow = self.ema_history[-1]
        ema_bullish = ema_fast > ema_slow
        ema_bearish = ema_fast < ema_slow

        if trend_strength > self.trend_strength_threshold:
            if slope > 0 and ema_bullish:
                return "TRENDING_UP"
            elif slope < 0 and ema_bearish:
                return "TRENDING_DOWN"

        return "RANGING"


class LLMRegimeDetector:
    """
    Optional LLM-based regime classifier.
    Uses LightLLM if available and model_path configured; otherwise falls back to Ollama.
    Returns None if disabled or if all backends fail.
    """
    def __init__(self, enabled: bool = False, model_name: str = "llama3.2", use_lightllm: bool = False, lightllm_model_path: str = ""):
        self.enabled = enabled
        self.model_name = model_name
        self.use_lightllm = use_lightllm and HAS_LIGHTLLM
        self.lightllm_model_path = lightllm_model_path
        self._lightllm_client = None
        if self.use_lightllm and HAS_LIGHTLLM:
            try:
                from src.core.lightllm_regime import LightLLMRegimeDetector
                self._lightllm_client = LightLLMRegimeDetector(model_path=lightllm_model_path)
            except Exception as e:
                logging.debug(f"LightLLM init failed: {e}")
                self._lightllm_client = None
                self.use_lightllm = False

    def detect(self, features_summary: str = "") -> Optional[str]:
        if not self.enabled:
            return None
        # Try LightLLM first if configured
        if self.use_lightllm and self._lightllm_client is not None:
            try:
                result = self._lightllm_client.detect(features_summary)
                if result:
                    return result
            except Exception as e:
                logging.debug(f"LightLLM detection failed: {e}")
                self._lightllm_client = None
                self.use_lightllm = False
        # Fallback to Ollama
        try:
            import ollama
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": f"Classify market regime from: {features_summary}. Answer one word: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE",
                    }
                ],
            )
            text = response["message"]["content"].strip().upper()
            for regime in ("TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE"):
                if regime in text:
                    return regime
            return None
        except Exception as e:
            logging.debug(f"Ollama detection skipped: {e}")
            return None
