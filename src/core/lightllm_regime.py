"""
Optional LLM regime detector using LightLLM for fast local inference.
Falls back to Ollama if LightLLM not available or model not found.
"""
from typing import Optional
import logging

try:
    from lightllm import LLM as LightLLM
    HAS_LIGHTLLM = True
except ImportError:
    HAS_LIGHTLLM = False


class LightLLMRegimeDetector:
    """
    High-performance local LLM detector using LightLLM.
    Loads model once and reuses for low-latency inference.
    Requires lightllm package and a local model checkpoint.
    """
    def __init__(self, model_path: str = "", device: str = "cuda"):
        if not HAS_LIGHTLLM:
            raise ImportError("lightllm not installed. pip install lightllm")
        self.model_path = model_path or "default_regime_classifier"
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = LightLLM(
                model_path=self.model_path,
                device=self.device,
                infer_mode="common",  # or 'pipeline' for batch
            )
            logging.info(f"LightLLM regime detector loaded model from {self.model_path}")
        except Exception as e:
            logging.warning(f"LightLLM model load failed: {e}. Will fallback to Ollama.")
            self.model = None

    def detect(self, features_summary: str = "") -> Optional[str]:
        if self.model is None:
            return None
        prompt = f"Classify market regime: {features_summary}. Answer one word: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE"
        try:
            outputs = self.model.generate([prompt], max_new_tokens=5, temperature=0.1)
            text = outputs[0].strip().upper()
            for regime in ("TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE"):
                if regime in text:
                    return regime
            return None
        except Exception as e:
            logging.debug(f"LightLLM inference error: {e}")
            return None
