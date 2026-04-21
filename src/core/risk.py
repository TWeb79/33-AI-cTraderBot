"""
Risk management: position sizing and trailing stop calculations.
"""
from typing import Tuple


class RiskManager:
    @staticmethod
    def position_size(
        balance: float,
        risk_pct: float,
        entry: float,
        stop: float,
        pip_value: float = 0.0001,
    ) -> int:
        """
        Calculate position size in units.
        Formula: units = (balance * risk_pct/100) / abs(entry - stop)
        """
        risk_amount = balance * (risk_pct / 100.0)
        stop_distance = abs(entry - stop)
        if stop_distance < 1e-10:
            return 0
        units = int(risk_amount / stop_distance)
        return max(units, 0)

    @staticmethod
    def update_trail_stop(current_price: float, current_trail: float, atr: float, multiplier: float = 1.5) -> float:
        """
        Update trailing stop: new_stop = current_price - (multiplier * ATR)
        Stop never moves downward (returns max of new and current).
        """
        new_trail = current_price - multiplier * atr
        return max(new_trail, current_trail)
