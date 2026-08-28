from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from typing import Optional
import numpy as np


@dataclass(slots=True)
class PriceLevel:
    """Single price level in the order book."""
    price: Decimal
    quantity: Decimal
    orders: int = 0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.quantity < 0:
            raise ValueError("Quantity cannot be negative")

    def add_quantity(self, qty: Decimal) -> None:
        self.quantity += qty
        self.orders += 1
        self.timestamp = time.time()

    def remove_quantity(self, qty: Decimal) -> None:
        self.quantity -= qty
        self.orders = max(0, self.orders - 1)
        self.timestamp = time.time()
        if self.quantity < 0:
            self.quantity = Decimal("0")

    def is_empty(self) -> bool:
        return self.quantity <= Decimal("0")

    def to_tuple(self) -> tuple[Decimal, Decimal]:
        return (self.price, self.quantity)


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    bids: list[tuple[Decimal, Decimal]]
    asks: list[tuple[Decimal, Decimal]]
    timestamp: float
    sequence: int
    symbol: str

    def best_bid(self) -> Optional[tuple[Decimal, Decimal]]:
        return self.bids[0] if self.bids else None

    def best_ask(self) -> Optional[tuple[Decimal, Decimal]]:
        return self.asks[0] if self.asks else None

    def mid_price(self) -> Optional[Decimal]:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            return (bid[0] + ask[0]) / Decimal("2")
        return None

    def spread(self) -> Optional[Decimal]:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            return ask[0] - bid[0]
        return None

    def spread_bps(self) -> Optional[float]:
        mid = self.mid_price()
        spr = self.spread()
        if mid and spr and mid > 0:
            return float(spr / mid * 10000)
        return None


class OrderBook:
    """
    Limit Order Book engine maintaining bid/ask price levels.

    Supports:
    - Incremental updates (Binance style)
    - Snapshot initialization
    - Consistency validation
    - Crossed book detection
    - Stale data detection
    """

    def __init__(
        self,
        symbol: str,
        max_levels: int = 100,
        price_precision: int = 8,
        qty_precision: int = 8,
        validate_sequence: bool = True,
        allow_crossed: bool = False,
        stale_threshold_ms: int = 5000,
    ):
        self.symbol = symbol
        self.max_levels = max_levels
        self.price_precision = price_precision
        self.qty_precision = qty_precision
        self.validate_sequence = validate_sequence
        self.allow_crossed = allow_crossed
        self.stale_threshold_ms = stale_threshold_ms

        self._bids: dict[Decimal, PriceLevel] = {}
        self._asks: dict[Decimal, PriceLevel] = {}
        self._bid_prices: list[Decimal] = []  # Sorted descending
        self._ask_prices: list[Decimal] = []  # Sorted ascending

        self._sequence: int = 0
        self._last_update_id: int = 0
        self._last_timestamp: float = 0.0
        self._initialized: bool = False
        self._update_count: int = 0
        self._gap_count: int = 0
        self._crossed_count: int = 0

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def last_update_id(self) -> int:
        return self._last_update_id

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def last_timestamp(self) -> float:
        return self._last_timestamp

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def gap_count(self) -> int:
        return self._gap_count

    @property
    def crossed_count(self) -> int:
        return self._crossed_count

    def _round_price(self, price: Decimal) -> Decimal:
        quantize_str = "0." + "0" * self.price_precision
        return price.quantize(Decimal(quantize_str), rounding=ROUND_DOWN)

    def _round_qty(self, qty: Decimal) -> Decimal:
        quantize_str = "0." + "0" * self.qty_precision
        return qty.quantize(Decimal(quantize_str), rounding=ROUND_DOWN)

    def _insert_bid_price(self, price: Decimal) -> None:
        """Insert price into sorted bid prices (descending)."""
        import bisect
        idx = bisect.bisect_left([-p for p in self._bid_prices], -price)
        if idx == len(self._bid_prices) or self._bid_prices[idx] != price:
            self._bid_prices.insert(idx, price)

    def _insert_ask_price(self, price: Decimal) -> None:
        """Insert price into sorted ask prices (ascending)."""
        import bisect
        idx = bisect.bisect_left(self._ask_prices, price)
        if idx == len(self._ask_prices) or self._ask_prices[idx] != price:
            self._ask_prices.insert(idx, price)

    def _remove_bid_price(self, price: Decimal) -> None:
        """Remove price from bid prices."""
        try:
            self._bid_prices.remove(price)
        except ValueError:
            pass

    def _remove_ask_price(self, price: Decimal) -> None:
        """Remove price from ask prices."""
        try:
            self._ask_prices.remove(price)
        except ValueError:
            pass

    def initialize_from_snapshot(
        self,
        bids: list[tuple[Decimal, Decimal]],
        asks: list[tuple[Decimal, Decimal]],
        timestamp: float,
        sequence: int = 0,
        last_update_id: int = 0,
    ) -> None:
        """Initialize order book from full snapshot."""
        self._bids.clear()
        self._asks.clear()
        self._bid_prices.clear()
        self._ask_prices.clear()

        for price, qty in bids[:self.max_levels]:
            price = self._round_price(price)
            qty = self._round_qty(qty)
            if qty > 0:
                self._bids[price] = PriceLevel(price=price, quantity=qty)
                self._insert_bid_price(price)

        for price, qty in asks[:self.max_levels]:
            price = self._round_price(price)
            qty = self._round_qty(qty)
            if qty > 0:
                self._asks[price] = PriceLevel(price=price, quantity=qty)
                self._insert_ask_price(price)

        self._sequence = sequence
        self._last_update_id = last_update_id
        self._last_timestamp = timestamp
        self._initialized = True
        self._update_count = 0
        self._gap_count = 0
        self._crossed_count = 0

        self._validate()

    def apply_update(
        self,
        bids: list[tuple[Decimal, Decimal]],
        asks: list[tuple[Decimal, Decimal]],
        timestamp: float,
        sequence: int | None = None,
        first_update_id: int | None = None,
        final_update_id: int | None = None,
    ) -> bool:
        """
        Apply incremental update to order book.

        Returns True if update was applied successfully, False if gap detected.
        """
        if not self._initialized:
            return False

        # Sequence validation
        if self.validate_sequence and sequence is not None:
            expected = self._sequence + 1
            if sequence != expected:
                self._gap_count += 1
                # If gap is small, we might still process (depending on policy)
                if sequence > expected + 100:
                    return False
            self._sequence = sequence

        # Update ID validation (Binance style)
        if first_update_id is not None and final_update_id is not None:
            if first_update_id <= self._last_update_id:
                # This update is stale or duplicate
                if final_update_id < self._last_update_id:
                    return True  # Already processed
            self._last_update_id = final_update_id

        # Apply bid updates
        for price, qty in bids:
            price = self._round_price(price)
            qty = self._round_qty(qty)

            if qty <= 0:
                # Remove price level
                if price in self._bids:
                    del self._bids[price]
                    self._remove_bid_price(price)
            else:
                # Update or add price level
                if price in self._bids:
                    self._bids[price].quantity = qty
                    self._bids[price].timestamp = timestamp
                else:
                    self._bids[price] = PriceLevel(price=price, quantity=qty, timestamp=timestamp)
                    self._insert_bid_price(price)

        # Apply ask updates
        for price, qty in asks:
            price = self._round_price(price)
            qty = self._round_qty(qty)

            if qty <= 0:
                if price in self._asks:
                    del self._asks[price]
                    self._remove_ask_price(price)
            else:
                if price in self._asks:
                    self._asks[price].quantity = qty
                    self._asks[price].timestamp = timestamp
                else:
                    self._asks[price] = PriceLevel(price=price, quantity=qty, timestamp=timestamp)
                    self._insert_ask_price(price)

        self._last_timestamp = timestamp
        self._update_count += 1

        # Trim to max levels
        self._trim_levels()

        # Validate
        return self._validate()

    def _trim_levels(self) -> None:
        """Trim price levels to max_levels."""
        # Trim bids (keep highest prices)
        while len(self._bid_prices) > self.max_levels:
            price = self._bid_prices.pop()  # Lowest bid
            if price in self._bids:
                del self._bids[price]

        # Trim asks (keep lowest prices)
        while len(self._ask_prices) > self.max_levels:
            price = self._ask_prices.pop()  # Highest ask
            if price in self._asks:
                del self._asks[price]

    def _validate(self) -> bool:
        """Validate order book consistency."""
        # Check for crossed book
        best_bid = self.best_bid()
        best_ask = self.best_ask()

        if best_bid and best_ask and best_bid[0] >= best_ask[0]:
            self._crossed_count += 1
            if not self.allow_crossed:
                # Try to fix by removing crossed levels
                self._fix_crossed_book()
                if not self.allow_crossed:
                    return False

        # Check for negative quantities
        for price, level in self._bids.items():
            if level.quantity < 0:
                level.quantity = Decimal("0")
        for price, level in self._asks.items():
            if level.quantity < 0:
                level.quantity = Decimal("0")

        # Check stale data
        if self._last_timestamp > 0:
            age_ms = (time.time() - self._last_timestamp) * 1000
            if age_ms > self.stale_threshold_ms:
                pass  # Log warning but don't fail

        return True

    def _fix_crossed_book(self) -> None:
        """Attempt to fix crossed book by removing crossed levels."""
        best_bid = self.best_bid()
        best_ask = self.best_ask()

        while best_bid and best_ask and best_bid[0] >= best_ask[0]:
            # Remove the level with smaller quantity
            if best_bid[1] <= best_ask[1]:
                del self._bids[best_bid[0]]
                self._remove_bid_price(best_bid[0])
            else:
                del self._asks[best_ask[0]]
                self._remove_ask_price(best_ask[0])

            best_bid = self.best_bid()
            best_ask = self.best_ask()

    def best_bid(self) -> Optional[tuple[Decimal, Decimal]]:
        if self._bid_prices:
            price = self._bid_prices[0]
            level = self._bids[price]
            return (level.price, level.quantity)
        return None

    def best_ask(self) -> Optional[tuple[Decimal, Decimal]]:
        if self._ask_prices:
            price = self._ask_prices[0]
            level = self._asks[price]
            return (level.price, level.quantity)
        return None

    def mid_price(self) -> Optional[Decimal]:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            return (bid[0] + ask[0]) / Decimal("2")
        return None

    def microprice(self) -> Optional[Decimal]:
        """Microprice weighted by volume at best bid/ask."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            bid_vol = bid[1]
            ask_vol = ask[1]
            total_vol = bid_vol + ask_vol
            if total_vol > 0:
                return (bid[0] * ask_vol + ask[0] * bid_vol) / total_vol
        return self.mid_price()

    def spread(self) -> Optional[Decimal]:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid and ask:
            return ask[0] - bid[0]
        return None

    def spread_bps(self) -> Optional[float]:
        mid = self.mid_price()
        spr = self.spread()
        if mid and spr and mid > 0:
            return float(spr / mid * 10000)
        return None

    def get_bids(self, levels: int = 10) -> list[tuple[Decimal, Decimal]]:
        result = []
        for price in self._bid_prices[:levels]:
            level = self._bids[price]
            result.append((level.price, level.quantity))
        return result

    def get_asks(self, levels: int = 10) -> list[tuple[Decimal, Decimal]]:
        result = []
        for price in self._ask_prices[:levels]:
            level = self._asks[price]
            result.append((level.price, level.quantity))
        return result

    def get_depth(self, levels: int = 10) -> tuple[list, list]:
        return self.get_bids(levels), self.get_asks(levels)

    def cumulative_depth(self, side: str, levels: int) -> Decimal:
        """Cumulative quantity up to N levels."""
        total = Decimal("0")
        if side == "bid":
            for price in self._bid_prices[:levels]:
                total += self._bids[price].quantity
        elif side == "ask":
            for price in self._ask_prices[:levels]:
                total += self._asks[price].quantity
        return total

    def depth_imbalance(self, levels: int = 10) -> float:
        """Order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)."""
        bid_vol = self.cumulative_depth("bid", levels)
        ask_vol = self.cumulative_depth("ask", levels)
        total = bid_vol + ask_vol
        if total > 0:
            return float((bid_vol - ask_vol) / total)
        return 0.0

    def weighted_mid_price(self, levels: int = 5) -> Optional[Decimal]:
        """Volume-weighted mid price over N levels."""
        bid_vol = Decimal("0")
        bid_px_vol = Decimal("0")
        ask_vol = Decimal("0")
        ask_px_vol = Decimal("0")

        for price in self._bid_prices[:levels]:
            level = self._bids[price]
            bid_vol += level.quantity
            bid_px_vol += level.price * level.quantity

        for price in self._ask_prices[:levels]:
            level = self._asks[price]
            ask_vol += level.quantity
            ask_px_vol += level.price * level.quantity

        total_vol = bid_vol + ask_vol
        if total_vol > 0:
            return (bid_px_vol + ask_px_vol) / total_vol
        return self.mid_price()

    def to_snapshot(self) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            bids=self.get_bids(self.max_levels),
            asks=self.get_asks(self.max_levels),
            timestamp=self._last_timestamp,
            sequence=self._sequence,
            symbol=self.symbol,
        )

    def is_stale(self, threshold_ms: int | None = None) -> bool:
        threshold = threshold_ms or self.stale_threshold_ms
        age_ms = (time.time() - self._last_timestamp) * 1000
        return age_ms > threshold

    def stats(self) -> dict:
        return {
            "symbol": self.symbol,
            "initialized": self._initialized,
            "sequence": self._sequence,
            "last_update_id": self._last_update_id,
            "bid_levels": len(self._bid_prices),
            "ask_levels": len(self._ask_prices),
            "update_count": self._update_count,
            "gap_count": self._gap_count,
            "crossed_count": self._crossed_count,
            "last_timestamp": self._last_timestamp,
            "best_bid": str(self.best_bid()[0]) if self.best_bid() else None,
            "best_ask": str(self.best_ask()[0]) if self.best_ask() else None,
            "mid_price": str(self.mid_price()) if self.mid_price() else None,
            "spread_bps": self.spread_bps(),
            "depth_imbalance_10": self.depth_imbalance(10),
        }

    def __repr__(self) -> str:
        bid = self.best_bid()
        ask = self.best_ask()
        mid = self.mid_price()
        return f"OrderBook({self.symbol}, bid={bid}, ask={ask}, mid={mid}, levels={len(self._bid_prices)}/{len(self._ask_prices)})"