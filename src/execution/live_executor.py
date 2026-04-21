"""
Live executor: connects to cTrader Open API and runs strategy in real-time.
Import guarded: module loads even if ctrader-open-api is not installed.
Instantiation of LiveExecutor will fail if used without ctrader.
"""
import logging
from typing import Optional, Callable

from twisted.internet import reactor, defer

from ..core.types import Bar
from ..core.patterns import PatternDetector
from ..core.features import FeatureEngine
from ..core.regime import MarketRegimeDetector, LLMRegimeDetector
from ..core.ai_filter import AITradeFilter
from ..core.risk import RiskManager
from .base import IMarketDataHandler, IExecutionHandler

# Attempt to import ctrader symbols; if not available, set to None
try:
    from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (
        ProtoOAApplicationAuthRes,
        ProtoOAAccountAuthRes,
        ProtoOASpotEvent,
        ProtoOATraderRes,
        ProtoOANewOrderReq,
        ProtoOAClosePositionReq,
        ProtoOAOrderType,
        ProtoOATradeSide,
    )
    HAS_CTRADER = True
except ImportError:
    HAS_CTRADER = False
    Client = Protobuf = TcpProtocol = EndPoints = None
    ProtoOAApplicationAuthRes = ProtoOAAccountAuthRes = ProtoOASpotEvent = ProtoOATraderRes = None
    ProtoOANewOrderReq = ProtoOAClosePositionReq = ProtoOAOrderType = ProtoOATradeSide = None


class LiveExecutor(IMarketDataHandler, IExecutionHandler):
    """
    Connects to cTrader Open API, subscribes to spot events, and executes live trades.
    Handles reconnection automatically.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        access_token: str,
        account_id: int,
        symbol_id: int,
        detector: PatternDetector,
        ai_filter: AITradeFilter,
        risk_mgr: RiskManager,
        initial_balance: float = 10_000.0,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.account_id = account_id
        self.symbol_id = symbol_id
        self.detector = detector
        self.ai_filter = ai_filter
        self.risk_mgr = risk_mgr
        self.balance = initial_balance
        self._bar_callback: Optional[Callable[[Bar], None]] = None

        self.client: Optional[Client] = None
        self.position_id: Optional[int] = None

    def subscribe(self, callback: Callable[[Bar], None]) -> None:
        self._bar_callback = callback

    def unsubscribe(self) -> None:
        self._bar_callback = None

    def get_latest_bar(self):
        # Live streaming feed doesn't buffer; returns None, bars pushed via callback
        return None

    def place_market_order(self, entry: float, stop: float, units: int, direction: str = "BUY") -> Optional[int]:
        if not self.client:
            logging.error("Cannot place order — not connected.")
            return None
        try:
            req = ProtoOANewOrderReq()
            req.ctidTraderAccountId = self.account_id
            req.symbolId = self.symbol_id
            req.orderType = ProtoOAOrderType.Value("MARKET")
            req.tradeSide = (
                ProtoOATradeSide.Value("BUY") if direction == "BUY" else ProtoOATradeSide.Value("SELL")
            )
            req.volume = units * 100
            req.stopLoss = int(stop * 100_000)
            deferred = self.client.send(req)
            deferred.addCallback(self._on_order_submitted)
            deferred.addErrback(self._on_order_error)
            return self.position_id
        except Exception as e:
            logging.error(f"Order placement failed: {e}")
            return None

    def close_position(self) -> bool:
        if not self.client or self.position_id is None:
            return False
        try:
            req = ProtoOAClosePositionReq()
            req.ctidTraderAccountId = self.account_id
            req.positionId = self.position_id
            req.volume = 1
            self.client.send(req)
            return True
        except Exception as e:
            logging.error(f"Close order failed: {e}")
            return False

    def get_balance(self) -> float:
        return self.balance

    def start(self):
        """Connect and start the Twisted reactor."""
        if not HAS_CTRADER:
            logging.error("ctrader-open-api not installed. Cannot start live mode.")
            return
        logging.info("Connecting to cTrader Open API ...")
        self._connect()

    def _connect(self):
        assert Client is not None and Protobuf is not None and TcpProtocol is not None and EndPoints is not None
        self.client = Client(
            EndPoints.PROTOBUF_LIVE_HOST,
            EndPoints.PROTOBUF_PORT,
            TcpProtocol,
        )
        self.client.setConnectedCallback(self._on_connected)
        self.client.setDisconnectedCallback(self._on_disconnected)
        self.client.setMessageReceivedCallback(self._on_message)
        self.client.startService()

    def _reconnect(self):
        """Scheduled on disconnect — fresh client instance + callback re-registration."""
        logging.info("Reconnecting to cTrader Open API ...")
        self._connect()

    def _on_connected(self, client):
        logging.info("Connected. Authenticating ...")
        request = ProtoOAApplicationAuthReq()
        request.clientId = self.client_id
        request.clientSecret = self.client_secret
        deferred = client.send(request)
        deferred.addErrback(self._on_error)

    def _on_disconnected(self, client, reason):
        logging.warning(f"Disconnected: {reason}. Reconnecting in 10s ...")
        reactor.callLater(10, self._reconnect)

    def _on_message(self, client, message):
        msg_type = Protobuf.extract(message)

        if msg_type == ProtoOAApplicationAuthRes:
            logging.info("App authenticated. Authorising account ...")
            req = ProtoOAAccountAuthReq()
            req.ctidTraderAccountId = self.account_id
            req.accessToken = self.access_token
            client.send(req).addErrback(self._on_error)

        elif msg_type == ProtoOAAccountAuthRes:
            logging.info("Account authorised. Subscribing to spots ...")
            self._subscribe_spots(client)
            self._request_account_info(client)

        elif msg_type == ProtoOASpotEvent:
            spot = Protobuf.extract(message, ProtoOASpotEvent)
            bar = Bar(
                time=spot.timestamp,
                open=float(spot.bidOpen),
                high=float(spot.bidHigh),
                low=float(spot.bidLow),
                close=float(spot.bidClose),
                volume=float(getattr(spot, "volume", 1000)),
            )
            if self._bar_callback:
                self._bar_callback(bar)

        elif msg_type == ProtoOATraderRes:
            trader = Protobuf.extract(message, ProtoOATraderRes)
            self.balance = trader.trader.balance / 100.0
            logging.info(f"Account balance: {self.balance:.2f}")

    def _subscribe_spots(self, client):
        req = ProtoOASubscribeSpotsReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId.append(self.symbol_id)
        client.send(req)

    def _request_account_info(self, client):
        req = ProtoOATraderReq()
        req.ctidTraderAccountId = self.account_id
        client.send(req).addErrback(self._on_error)

    def _on_order_submitted(self, response):
        # May contain positionId; extract if available
        logging.info(f"Order submitted: {response}")
        # In practice, position_id may come in a separate confirmation message.
        # For now, we simply acknowledge submission.
        pass

    def _on_order_error(self, failure):
        logging.error(f"Order failed: {failure}")

    def _on_error(self, failure):
        logging.error(f"API error: {failure}")
