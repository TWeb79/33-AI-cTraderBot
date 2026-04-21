"""
Tests for AI filter: probability prediction, confidence-risk mapping, model persistence.
"""
import pytest
import numpy as np
import os
import tempfile

from src.core.ai_filter import AITradeFilter


def test_confidence_risk_continuous_scaling():
    filt = AITradeFilter(threshold=0.65)
    # Linear mapping: 0.5->0.5, 0.6->0.7, 0.7->0.9, 0.8->1.1, 0.9->1.3, 1.0->1.5
    assert filt.get_confidence_risk(0.5) == pytest.approx(0.5, abs=0.01)
    assert filt.get_confidence_risk(0.6) == pytest.approx(0.7, abs=0.01)
    assert filt.get_confidence_risk(0.7) == pytest.approx(0.9, abs=0.01)
    assert filt.get_confidence_risk(0.8) == pytest.approx(1.1, abs=0.01)
    assert filt.get_confidence_risk(0.9) == pytest.approx(1.3, abs=0.01)
    assert filt.get_confidence_risk(1.0) == pytest.approx(1.5, abs=0.01)
    assert filt.get_confidence_risk(0.0) == 0.2
    assert filt.get_confidence_risk(0.3) == pytest.approx(0.2 + 0.3*0.6, abs=0.01)  # below 0.5 linear


def test_should_trade_above_threshold():
    filt = AITradeFilter(threshold=0.65)
    # With no model, predict_probability returns 0.65
    assert filt.should_trade({}) is True


def test_should_trade_below_threshold():
    filt = AITradeFilter(threshold=0.75)

    class DummyModel:
        def predict_proba(self, X):
            return [[0.6, 0.4]]  # positive class 0.4

    filt.model = DummyModel()
    assert filt.should_trade({}) is False


def test_feature_vector_order():
    filt = AITradeFilter(threshold=0.65)
    feats = {
        "ema_slope": 1.0,
        "ema_distance_ratio": 2.0,
        "atr_expansion": 3.0,
        "base_tightness": 4.0,
        "breakout_velocity": 5.0,
        "volume_ratio": 6.0,
        "mtf_alignment": 7.0,
        "trend_maturity": 8,
        "regime_score": 9.0,
        "asia_session": 10.0,
        "london_session": 11.0,
        "ny_session": 12.0,
    }
    vec = filt._features_to_vector(feats)
    assert vec == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8,
        9.0,
        10.0,
        11.0,
        12.0,
    ]


def test_model_persistence():
    """Train and save model, then reload produces same predictions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pkl")
        # Train on simple dataset
        X = np.random.randn(100, 12)
        y = np.random.randint(0, 2, 100)
        filt1 = AITradeFilter(model_path=path, model_type="gb", threshold=0.5)
        filt1.train(X, y)
        assert os.path.exists(path)
        # Reload
        filt2 = AITradeFilter(model_path=path, model_type="gb", threshold=0.5)
        assert filt2.model is not None
        # Same input
        x = X[:1]
        p1 = filt1.model.predict_proba(x)[0]
        p2 = filt2.model.predict_proba(x)[0]
        assert np.allclose(p1, p2)


def test_predict_probability_on_missing_features():
    filt = AITradeFilter(threshold=0.65)
    conf, exp = filt.predict_probability({})
    assert conf == 0.65 and exp == 0.65
