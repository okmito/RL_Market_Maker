from __future__ import annotations
import time
import uuid
import random
import math
from decimal import Decimal, ROUND_DOWN
from typing import Optional
import numpy as np
from loguru import logger

from market_maker.exchange.base import ExchangeBase, Order, OrderSide, OrderType, OrderStatus, Fill, Balance, SymbolInfo, TimeInForce
from market_maker.market_data.order_book import OrderBook

class SimulatedExchange(ExchangeBase):
    """Deterministic local exchange simulator with realistic frictions."""
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        initial_price: float = 50000.0,
        initial_spread_bps: float = 10.0,
        tick_size: float = 0.01,
        lot_size: float = 0.0001,
        maker_fee_bps: float = 1.0,
        taker_fee_bps: float = 5.0,
        latency_ms: int = 10,
        fill_model: str = "queue_aware",
        volatility: float = 0.02,
        drift: float = 0.0,
        mean_reversion: float = 0.1,
        initial_cash: float = 100000.0,
        initial_inventory: float = 0.0,
        seed: int = 42,
    ):
        self.symbol = symbol
        self.tick_size = Decimal(str(tick_size))
        self.lot_size = Decimal(str(lot_size))
        self.maker_fee_rate = Decimal(str(maker_fee_bps / 10000))
        self.taker_fee_rate = Decimal(str(taker_fee_bps / 10000))
        self.latency_ms = latency_ms
        self.fill_model = fill_model
        self.volatility = volatility
        self.drift = drift
        self.mean_reversion = mean_reversion
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # State
        self.mid_price = Decimal(str(initial_price))
        self.cash = Decimal(str(initial_cash))
        self.inventory = Decimal(str(initial_inventory))
        self._initial_cash = Decimal(str(initial_cash))
        # Order book
        spread = self.mid_price * Decimal(str(initial_spread_bps / 10000))
        best_bid = self.mid_price - spread/2
        best_ask = self.mid_price + spread/2
        self.order_book = OrderBook(symbol=symbol, max_levels=100)
        self._init_book(best_bid, best_ask)

        self._orders: dict[int, Order] = {}
        self._next_order_id = 100000
        self._trade_id = 1
        self._balances = {
            "USDT": Balance(asset="USDT", free=Decimal(str(initial_cash)), locked=Decimal("0")),
            "BTC": Balance(asset="BTC", free=Decimal(str(initial_inventory)), locked=Decimal("0")),
        }
        # Queue position tracking for queue-aware model
        self._queue_position: dict[int, float] = {}  # order_id -> queue fraction (0 front, 1 back)
        self._market_trades: list[dict] = []
        self._price_history: list[float] = [float(self.mid_price)]

    def _init_book(self, best_bid: Decimal, best_ask: Decimal):
        bids = []
        asks = []
        for i in range(20):
            px = best_bid - Decimal(str(i * 0.5))
            qty = Decimal(str(self._rng.uniform(0.5, 5.0)))
            bids.append((px, qty))
            px2 = best_ask + Decimal(str(i * 0.5))
            qty2 = Decimal(str(self._rng.uniform(0.5, 5.0)))
            asks.append((px2, qty2))
        self.order_book.initialize_from_snapshot(bids=bids, asks=asks, timestamp=time.time(), sequence=1, last_update_id=1)

    def _round_price(self, price: Decimal) -> Decimal:
        return (price / self.tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * self.tick_size

    def _round_qty(self, qty: Decimal) -> Decimal:
        return (qty / self.lot_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * self.lot_size

    async def connect(self) -> None:
        logger.info(f"SimulatedExchange connected: {self.symbol} mid={self.mid_price}")

    async def disconnect(self) -> None:
        logger.info("SimulatedExchange disconnected")

    async def get_symbol_info(self, symbol: str) -> SymbolInfo:
        return SymbolInfo(symbol=symbol, base_asset=symbol.replace("USDT",""), quote_asset="USDT",
                          price_tick=self.tick_size, qty_step=self.lot_size,
                          min_qty=Decimal("0.0001"), max_qty=Decimal("100"), min_notional=Decimal("10"))

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        snap = self.order_book.to_snapshot()
        return {"bids": snap.bids[:limit], "asks": snap.asks[:limit], "timestamp": snap.timestamp}

    async def get_mid_price(self, symbol: str) -> Optional[Decimal]:
        return self.order_book.mid_price()

    async def get_balances(self) -> dict[str, Balance]:
        return self._balances

    async def place_order(self, order: Order) -> Order:
        # Validate
        order.price = self._round_price(order.price) if order.price else None
        order.quantity = self._round_qty(order.quantity)
        info = await self.get_symbol_info(order.symbol)
        if order.quantity < info.min_qty:
            order.status = OrderStatus.REJECTED
            return order
        order.order_id = self._next_order_id
        self._next_order_id += 1
        order.created_at = time.time()
        order.updated_at = time.time()
        if not order.client_order_id:
            order.client_order_id = f"sim_{uuid.uuid4().hex[:12]}"
        self._orders[order.order_id] = order
        # Queue position: random 0.3-0.8 behind queue
        self._queue_position[order.order_id] = self._rng.uniform(0.3, 0.9)
        logger.debug(f"ORDER_SUBMITTED id={order.order_id} {order.side} {order.quantity}@{order.price}")
        return order

    async def cancel_order(self, symbol: str, order_id: int) -> Order:
        o = self._orders.get(order_id)
        if not o:
            raise ValueError(f"Order {order_id} not found")
        if o.is_active:
            o.status = OrderStatus.CANCELED
            o.updated_at = time.time()
            logger.debug(f"ORDER_CANCELLED id={order_id}")
        return o

    async def cancel_all_orders(self, symbol: str) -> list[Order]:
        result = []
        for oid, o in list(self._orders.items()):
            if o.symbol == symbol and o.is_active:
                o.status = OrderStatus.CANCELED
                o.updated_at = time.time()
                result.append(o)
        return result

    async def get_open_orders(self, symbol: str) -> list[Order]:
        return [o for o in self._orders.values() if o.symbol==symbol and o.is_active]

    async def get_order(self, symbol: str, order_id: int) -> Optional[Order]:
        return self._orders.get(order_id)

    def step_market(self, steps: int = 1) -> None:
        """Advance market price using GBM with mean reversion and jumps."""
        for _ in range(steps):
            dt = 1.0
            # Geometric Brownian with mean reversion
            prev = float(self.mid_price)
            drift = self.drift - self.mean_reversion * (prev - self._price_history[0])/self._price_history[0] if self._price_history else 0
            shock = self._np_rng.normal(0, self.volatility*math.sqrt(dt))
            # Jump
            if self._rng.random() < 0.001:
                shock += self._rng.choice([-1,1]) * 0.05
            new_price = prev * math.exp(drift*dt + shock)
            new_price = max(new_price, 1000)
            self.mid_price = Decimal(str(round(new_price,2)))
            self._price_history.append(new_price)
            # Update order book around new mid
            spread_bps = self._rng.uniform(5, 20)
            spread = self.mid_price * Decimal(str(spread_bps/10000))
            best_bid = self.mid_price - spread/2
            best_ask = self.mid_price + spread/2
            # Apply drift to book
            self._update_book(best_bid, best_ask)
            # Attempt fills
            self._process_fills()
            # Age queue
            for oid in list(self._queue_position.keys()):
                self._queue_position[oid] = max(0, self._queue_position[oid] - 0.05)

    def _update_book(self, best_bid: Decimal, best_ask: Decimal):
        # Rebuild incremental update around new mid
        bids = [(best_bid - Decimal(str(i*0.5)), Decimal(str(self._rng.uniform(0.3,4)))) for i in range(5)]
        asks = [(best_ask + Decimal(str(i*0.5)), Decimal(str(self._rng.uniform(0.3,4)))) for i in range(5)]
        # Use apply_update with negative qty removal simulation is simplified: reinit incrementally
        # Instead reinitialize if drift large else incremental
        # For stability, just apply updates for best 5 levels
        self.order_book.apply_update(bids=bids, asks=asks, timestamp=time.time(), sequence=self.order_book.sequence+1)

    def _process_fills(self):
        mid = self.order_book.mid_price()
        if not mid:
            return
        # Simple fill logic: if market mid crosses our limit price, chance to fill depends on model
        for oid, order in list(self._orders.items()):
            if not order.is_active or order.price is None:
                continue
            # Use fills.py helper for DRY — logic verified in tests/test_no_lookahead
            from market_maker.environment.fills import FillConfig, should_fill
            best_bid = self.order_book.best_bid()
            best_ask = self.order_book.best_ask()
            bid_px = float(best_bid[0]) if best_bid else float(mid)-1
            ask_px = float(best_ask[0]) if best_ask else float(mid)+1
            qp = self._queue_position.get(oid, 0.5)
            fill_cfg = FillConfig(model=self.fill_model)  # type: ignore
            fill, is_maker = should_fill(float(order.price), bid_px, ask_px, order.side.value, qp, fill_cfg, self._rng)
            if fill:
                exec_qty = order.remaining_qty
                # Partial fills occasionally
                if self._rng.random() < 0.1 and exec_qty > self.lot_size*2:
                    exec_qty = self._round_qty(exec_qty * Decimal(str(self._rng.uniform(0.3,0.8))))
                price = order.price
                commission = exec_qty * price * (self.maker_fee_rate if is_maker else self.taker_fee_rate)
                f = Fill(order_id=oid, symbol=order.symbol, side=order.side, price=price,
                         quantity=exec_qty, commission=commission, commission_asset="USDT",
                         timestamp=time.time(), is_maker=is_maker, trade_id=self._trade_id)
                self._trade_id+=1
                order.fills.append(f)
                order.executed_qty += exec_qty
                order.cummulative_quote_qty += exec_qty*price
                order.updated_at=time.time()
                if order.executed_qty >= order.quantity:
                    order.status = OrderStatus.FILLED
                    self._queue_position.pop(oid, None)
                else:
                    order.status = OrderStatus.PARTIALLY_FILLED
                # Update inventory/cash
                if order.side == OrderSide.BUY:
                    self.inventory += exec_qty
                    self.cash -= exec_qty*price + commission
                else:
                    self.inventory -= exec_qty
                    self.cash += exec_qty*price - commission
                logger.debug(f"ORDER_FILLED id={oid} qty={exec_qty} price={price} is_maker={is_maker}")
                self._market_trades.append({"price": float(price), "qty": float(exec_qty), "side": order.side, "time": time.time()})

    @property
    def portfolio_value(self) -> Decimal:
        mid = self.order_book.mid_price() or self.mid_price
        return self.cash + self.inventory * mid

    @property
    def unrealized_pnl(self) -> Decimal:
        mid = self.order_book.mid_price() or self.mid_price
        return self.inventory * mid

    def get_inventory(self) -> Decimal:
        return self.inventory
    def get_cash(self) -> Decimal:
        return self.cash
