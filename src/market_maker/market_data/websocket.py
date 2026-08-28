from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Optional
from urllib.parse import urljoin

import aiohttp
from loguru import logger

from market_maker.config import get_config
from market_maker.market_data.order_book import OrderBook


@dataclass
class WebSocketMessage:
    stream: str
    data: dict
    timestamp: float


class BinanceWebSocketClient:
    """
    Binance WebSocket client for market data streams.

    Handles:
    - Depth stream (@depth@100ms or @depth@1000ms)
    - Trade stream (@trade)
    - Kline stream (@kline_1m)
    - Ticker stream (@ticker)
    - Automatic reconnection with exponential backoff
    - Heartbeat/ping-pong
    """

    def __init__(
        self,
        symbol: str,
        streams: list[str],
        ws_url: str,
        on_message: Callable[[WebSocketMessage], None],
        on_error: Callable[[Exception], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ):
        self.symbol = symbol.lower()
        self.streams = streams
        self.ws_url = ws_url
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._running = False
        self._reconnect_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._last_pong: float = 0
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._base_reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._heartbeat_interval = 30

    async def start(self) -> None:
        """Start WebSocket connection."""
        self._running = True
        self._session = aiohttp.ClientSession()
        await self._connect()

    async def stop(self) -> None:
        """Stop WebSocket connection."""
        self._running = False

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()

        if self._session:
            await self._session.close()

    async def _connect(self) -> None:
        """Establish WebSocket connection."""
        stream_path = "/".join(self.streams)
        url = urljoin(self.ws_url, f"stream?streams={stream_path}")

        try:
            logger.info(f"Connecting to {url}")
            self._ws = await self._session.ws_connect(
                url,
                heartbeat=self._heartbeat_interval,
                autoping=True,
            )
            self._reconnect_attempts = 0
            self._last_pong = time.time()

            # Start message loop
            asyncio.create_task(self._message_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            await self._schedule_reconnect()

    async def _message_loop(self) -> None:
        """Process incoming messages."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        stream = data.get("stream", "")
                        payload = data.get("data", {})
                        message = WebSocketMessage(
                            stream=stream,
                            data=payload,
                            timestamp=time.time(),
                        )
                        self.on_message(message)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse message: {e}")
                elif msg.type == aiohttp.WSMsgType.PONG:
                    self._last_pong = time.time()
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                    logger.info("WebSocket connection closed")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Message loop error: {e}")
            if self.on_error:
                self.on_error(e)
        finally:
            if self._running:
                await self._schedule_reconnect()

    async def _heartbeat_loop(self) -> None:
        """Monitor connection health."""
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            if time.time() - self._last_pong > self._heartbeat_interval * 2:
                logger.warning("Heartbeat timeout, reconnecting...")
                if self._ws:
                    await self._ws.close()
                break

    async def _schedule_reconnect(self) -> None:
        """Schedule reconnection with exponential backoff."""
        if not self._running:
            return

        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error("Max reconnect attempts reached")
            if self.on_close:
                self.on_close()
            return

        delay = min(
            self._base_reconnect_delay * (2 ** self._reconnect_attempts),
            self._max_reconnect_delay,
        )
        self._reconnect_attempts += 1

        logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)

        if self._running:
            await self._connect()


class OrderBookManager:
    """
    Manages multiple order books from WebSocket streams.

    Handles:
    - Snapshot + incremental update synchronization
    - Multiple symbols
    - Data validation and consistency checks
    """

    def __init__(
        self,
        symbols: list[str],
        ws_client: BinanceWebSocketClient,
        max_levels: int = 100,
        price_precision: int = 8,
        qty_precision: int = 8,
    ):
        self.symbols = [s.upper() for s in symbols]
        self.ws_client = ws_client
        self.books: dict[str, OrderBook] = {}

        for symbol in self.symbols:
            self.books[symbol] = OrderBook(
                symbol=symbol,
                max_levels=max_levels,
                price_precision=price_precision,
                qty_precision=qty_precision,
            )

        self._snapshot_buffer: dict[str, list] = defaultdict(list)
        self._waiting_for_snapshot: set[str] = set(self.symbols)
        self._last_update_ids: dict[str, int] = {}

    def handle_message(self, message: WebSocketMessage) -> None:
        """Route message to appropriate handler."""
        stream = message.stream
        data = message.data

        if "@depth" in stream:
            self._handle_depth_update(stream, data, message.timestamp)
        elif "@trade" in stream:
            self._handle_trade(stream, data, message.timestamp)
        elif "@kline" in stream:
            self._handle_kline(stream, data, message.timestamp)

    def _handle_depth_update(self, stream: str, data: dict, timestamp: float) -> None:
        """Handle depth update (snapshot or incremental)."""
        # Extract symbol from stream name (e.g., "btcusdt@depth@100ms")
        symbol = stream.split("@")[0].upper()

        if symbol not in self.books:
            return

        book = self.books[symbol]

        # Check if this is a snapshot (first message or full depth)
        if "lastUpdateId" in data and "bids" in data and "asks" in data:
            # Full snapshot
            self._apply_snapshot(book, data, timestamp)
        elif "u" in data and "b" in data and "a" in data:
            # Incremental update (Binance format)
            self._apply_incremental(book, data, timestamp)
        else:
            logger.warning(f"Unknown depth format: {data.keys()}")

    def _apply_snapshot(self, book: OrderBook, data: dict, timestamp: float) -> None:
        """Apply full order book snapshot."""
        bids = [(Decimal(str(p)), Decimal(str(q))) for p, q in data["bids"]]
        asks = [(Decimal(str(p)), Decimal(str(q))) for p, q in data["asks"]]
        last_update_id = data.get("lastUpdateId", 0)

        book.initialize_from_snapshot(
            bids=bids,
            asks=asks,
            timestamp=timestamp,
            last_update_id=last_update_id,
        )
        self._last_update_ids[book.symbol] = last_update_id
        self._waiting_for_snapshot.discard(book.symbol)
        logger.info(f"Initialized {book.symbol} from snapshot (update_id={last_update_id})")

    def _apply_incremental(self, book: OrderBook, data: dict, timestamp: float) -> None:
        """Apply incremental depth update."""
        symbol = book.symbol

        # If we haven't received snapshot yet, buffer updates
        if symbol in self._waiting_for_snapshot:
            self._snapshot_buffer[symbol].append((data, timestamp))
            return

        # Validate update sequence (Binance: u = firstUpdateId, U = lastUpdateId)
        first_update_id = data.get("U", 0)  # First update ID in this message
        final_update_id = data.get("u", 0)  # Last update ID in this message
        prev_update_id = self._last_update_ids.get(symbol, 0)

        # Check for gaps
        if first_update_id > prev_update_id + 1:
            logger.warning(
                f"Gap detected for {symbol}: expected {prev_update_id + 1}, got {first_update_id}"
            )
            # Request new snapshot
            self._waiting_for_snapshot.add(symbol)
            self._snapshot_buffer[symbol] = []
            return

        bids = [(Decimal(str(p)), Decimal(str(q))) for p, q in data["b"]]
        asks = [(Decimal(str(p)), Decimal(str(q))) for p, q in data["a"]]

        success = book.apply_update(
            bids=bids,
            asks=asks,
            timestamp=timestamp,
            first_update_id=first_update_id,
            final_update_id=final_update_id,
        )

        if success:
            self._last_update_ids[symbol] = final_update_id
        else:
            logger.error(f"Failed to apply update for {symbol}, requesting snapshot")
            self._waiting_for_snapshot.add(symbol)
            self._snapshot_buffer[symbol] = []

    def _handle_trade(self, stream: str, data: dict, timestamp: float) -> None:
        """Handle trade stream."""
        # Trade data available for microstructure features
        pass

    def _handle_kline(self, stream: str, data: dict, timestamp: float) -> None:
        """Handle kline/candlestick stream."""
        pass

    def get_book(self, symbol: str) -> OrderBook | None:
        return self.books.get(symbol.upper())

    def get_all_books(self) -> dict[str, OrderBook]:
        return self.books.copy()

    def all_initialized(self) -> bool:
        return len(self._waiting_for_snapshot) == 0