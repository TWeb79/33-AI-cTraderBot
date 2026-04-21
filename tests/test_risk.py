"""
Tests for risk manager: position sizing and trailing stop logic.
"""
import pytest
from src.core.risk import RiskManager


def test_position_size_calculation():
    rm = RiskManager()
    units = rm.position_size(balance=10000.0, risk_pct=1.0, entry=100.0, stop=99.0)
    # Risk amount = 100, stop distance = 1 → 100 units
    assert units == 100


def test_position_size_zero_stop_distance():
    rm = RiskManager()
    units = rm.position_size(balance=1000.0, risk_pct=1.0, entry=100.0, stop=100.0)
    assert units == 0


def test_trailing_stop_moves_up_only():
    rm = RiskManager()
    trail = 98.0
    new = rm.update_trail_stop(current_price=101.0, current_trail=trail, atr=1.0, multiplier=1.5)
    # new_trail = price - multiplier*atr = 101 - 1.5 = 99.5; max with 98 = 99.5
    assert new == pytest.approx(99.5)


def test_trailing_stop_does_not_move_down():
    rm = RiskManager()
    trail = 99.0
    price_drop = 99.8
    new = rm.update_trail_stop(current_price=price_drop, current_trail=trail, atr=1.0, multiplier=1.5)
    # price - atr*mult = 99.8 - 1.5 = 98.3 < 99.0 → stays at 99.0
    assert new == 99.0


def test_trailing_stop_with_custom_multiplier():
    rm = RiskManager()
    new_trail = rm.update_trail_stop(current_price=110.0, current_trail=108.0, atr=2.0, multiplier=2.0)
    # new_trail = 110 - 4 = 106; max(106,108) = 108
    assert new_trail == 108.0
