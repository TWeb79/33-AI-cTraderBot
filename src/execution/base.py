"""
Abstract interfaces for execution and market data handlers.
"""
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime

from ..core.types import Bar, TradeState


class IMarketDataHandler(ABC):
    """Abstract market data source (live or backtest)."""

    @abstractmethod
    def subscribe(self, callback: Callable[[Bar], None]) -> None:
        """Start streaming bars to callback."""
        pass

    @abstractmethod
    def unsubscribe(self) -> None:
        """Stop streaming."""
        pass

    @abstractmethod
    def get_latest_bar(self) -> Optional[Bar]:
        """Return the most recent completed bar."""
        pass


class IExecutionHandler(ABC):
    """Abstract order execution engine."""

    @abstractmethod
    def place_market_order(self, entry: float, stop: float, units: int, direction: str = "BUY") -> Optional[Any]:
        """Send market order; returns order/position ID or None."""
        pass

    @abstractmethod
    def close_position(self) -> bool:
        """Close current open position."""
        pass

    @abstractmethod
    def get_balance(self) -> float:
        """Return current account balance."""
        pass


class EventBus:
    """
    Simple pub/sub bus for internal events (bar, signal, trade, error).
    Used to decouple core strategy from execution infrastructure.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable):
        self._subscribers.setdefault(event_type, []).append(callback)

    def publish(self, event_type: str, data: Any = None):
        for cb in self._subscribers.get(event_type, []):
            try:
                cb(data)
            except Exception as e:
                logging.getLogger("EventBus").error(f"Event callback error: {e}")
