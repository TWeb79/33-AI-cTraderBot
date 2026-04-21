"""
Core domain types and data structures.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Deque, List, Optional


class Signal(Enum):
    NONE = "NONE"
    BASE_BREAK = "BASE_N_BREAK"
    WEDGE_POP = "WEDGE_POP"
    WEDGE_DROP = "WEDGE_DROP"
    EXIT_EXHAUST = "EXIT_EXHAUSTION"
    EXIT_CROSSBACK = "EXIT_EMA_CROSSBACK"


@dataclass
class Bar:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TradeState:
    in_position: bool = False
    entry_price: float = 0.0
    stop_loss: float = 0.0
    units: int = 0
    trail_stop: float = 0.0
    position_id: Optional[int] = None
    entry_time: Optional[datetime] = None
    base_low: float = 0.0
    confidence: float = 0.0
