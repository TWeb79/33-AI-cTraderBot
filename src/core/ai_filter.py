"""
AI Probability Filter: ML-based trade filtering and confidence-weighted risk scaling.
"""
from typing import Tuple, Optional
import os
import pickle
import numpy as np
import logging

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class AITradeFilter:
    """
    Loads/trains a binary classifier, predicts win probability, and maps confidence to risk multiplier.
    """

    def __init__(
        self,
        model_path: str = "ai_trade_model.pkl",
        model_type: str = "gb",
        threshold: float = 0.65,
        hidden_layers: tuple = (32, 16),
    ):
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.threshold = threshold
        self.hidden_layers = hidden_layers
        self.model = None
        self._load_model()

    def _load_model(self):
        if not HAS_SKLEARN:
            logging.warning("scikit-learn not available. Using default threshold.")
            return

        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logging.info(f"Loaded AI model from {self.model_path}")
            except Exception as e:
                logging.warning(f"Failed to load model: {e}. Creating new model.")
                self.model = self._create_default_model()
        else:
            logging.info("No trained model found. Creating default model.")
            self.model = self._create_default_model()

    def _create_default_model(self):
        if not HAS_SKLEARN:
            return None
        if self.model_type in {"mlp", "neural", "nn"}:
            return MLPClassifier(
                hidden_layer_sizes=self.hidden_layers,
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=42,
            )
        return GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )

    def predict_probability(self, features: dict) -> Tuple[float, float]:
        """
        Returns (confidence, expected_r_multiple).
        Currently returns same value for both; expected_r_multiple reserved for future extension.
        """
        if self.model is None or not features:
            return 0.65, 0.65

        try:
            feature_vec = self._features_to_vector(features)
            X = np.array([feature_vec])
            prob = self.model.predict_proba(X)[0]
            confidence = float(prob[1]) if len(prob) > 1 else 0.65
            return confidence, confidence
        except ValueError as e:
            if "n_features" in str(e) and HAS_SKLEARN:
                logging.warning("Feature dimension mismatch - recreating model.")
                self.model = self._create_default_model()
            return 0.65, 0.65
        except Exception as e:
            logging.warning(f"Prediction error: {e}")
            return 0.65, 0.65

    def should_trade(self, features: dict) -> bool:
        confidence, _ = self.predict_probability(features)
        return confidence >= self.threshold

    def get_confidence_risk(self, confidence: float) -> float:
        """
        Continuous risk scaling based on confidence.
        Two-segment linear mapping:
          [0.0, 0.5)  : 0.2 → 0.5 (slope 0.6)
          [0.5, 1.0]  : 0.5 → 1.5 (slope 2.0)
        Result: conf=0.5 -> 0.5, 0.6 -> 0.7, 0.7 -> 0.9, 0.8 -> 1.1, 0.9 -> 1.3, 1.0 -> 1.5.
        """
        if confidence <= 0:
            return 0.2
        if confidence < 0.5:
            return 0.2 + confidence * 0.6
        return 0.5 + (confidence - 0.5) * 2.0

    def train(self, X: np.ndarray, y: np.ndarray):
        if self.model is None and HAS_SKLEARN:
            self.model = self._create_default_model()
        if HAS_SKLEARN and self.model is not None:
            self.model.fit(X, y)
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            logging.info(f"Model trained and saved to {self.model_path}")

    # Feature order must match training
    EXPECTED_FEATURES = [
        "ema_slope",
        "ema_distance_ratio",
        "atr_expansion",
        "base_tightness",
        "breakout_velocity",
        "volume_ratio",
        "mtf_alignment",
        "trend_maturity",
        "regime_score",
        "asia_session",
        "london_session",
        "ny_session",
    ]

    def _features_to_vector(self, features: dict) -> list[float]:
        return [
            features.get("ema_slope", 0.0),
            features.get("ema_distance_ratio", 0.0),
            features.get("atr_expansion", 1.0),
            features.get("base_tightness", 0.0),
            features.get("breakout_velocity", 0.0),
            features.get("volume_ratio", 1.0),
            features.get("mtf_alignment", 0.0),
            features.get("trend_maturity", 0),
            features.get("regime_score", 0.0),
            features.get("asia_session", 0.0),
            features.get("london_session", 0.0),
            features.get("ny_session", 0.0),
        ]
