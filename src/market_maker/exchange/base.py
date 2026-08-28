from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional
import time

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    LIMIT_MAKER = "LIMIT_MAKER"

class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class TimeInForce(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"

@dataclass
class Order:
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: Optional[str] = None
    order_id: Optional[int] = None
    status: OrderStatus = OrderStatus.NEW
    executed_qty: Decimal = field(default_factory=lambda: Decimal("0"))
    cummulative_quote_qty: Decimal = field(default_factory=lambda: Decimal("0"))
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    fills: list = field(default_factory=list)

    @property
    def remaining_qty(self) -> Decimal:
        return self.quantity - self.executed_qty

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

@dataclass
class Fill:
    order_id: int
    symbol: str
    side: OrderSide
    price: Decimal
    quantity: Decimal
    commission: Decimal
    commission_asset: str
    timestamp: float
    is_maker: bool
    trade_id: int

@dataclass
class Balance:
    asset: str
    free: Decimal
    locked: Decimal

    @property
    def total(self) -> Decimal:
        return self.free + self.locked

@dataclass
class SymbolInfo:
    symbol: str
    base_asset: str
    quote_asset: str
    price_tick: Decimal
    qty_step: Decimal
    min_qty: Decimal
    max_qty: Decimal
    min_notional: Decimal
    status: str = "TRADING"

class ExchangeBase(ABC):
    """Abstract exchange interface for Simulated and Binance Testnet."""
    @abstractmethod
    async def connect(self) -> None: ...
    @abstractmethod
    async def disconnect(self) -> None: ...
    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> SymbolInfo: ...
    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 20) -> dict: ...
    @abstractmethod
    async def place_order(self, order: Order) -> Order: ...
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: int) -> Order: ...
    @abstractmethod
    async def cancel_all_orders(self, symbol: str) -> list[Order]: ...
    @abstractmethod
    async def get_open_orders(self, symbol: str) -> list[Order]: ...
    @abstractmethod
    async def get_order(self, symbol: str, order_id: int) -> Optional[Order]: ...
    @abstractmethod
    async def get_balances(self) -> dict[str, Balance]: ...
    @abstractmethod
    async def get_mid_price(self, symbol: str) -> Optional[Decimal]: ...
