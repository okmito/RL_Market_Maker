from __future__ import annotations

import asyncio
import msgpack
import os
import time
from dataclasses import dataclass, asdict
from decimal import Decimal
from pathlib import Path
from typing import Optional

import aiofiles

from market_maker.market_data.order_book import OrderBook, OrderBookSnapshot


@dataclass
class RecordedSnapshot:
    """Recorded order book snapshot for replay."""
    symbol: str
    timestamp: float
    sequence: int
    bids: list[tuple[str, str]]  # Stored as strings for precision
    asks: list[tuple[str, str]]
    trades: list[dict] = None

    def to_order_book_snapshot(self) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            bids=[(Decimal(p), Decimal(q)) for p, q in self.bids],
            asks=[(Decimal(p), Decimal(q)) for p, q in self.asks],
            timestamp=self.timestamp,
            sequence=self.sequence,
            symbol=self.symbol,
        )


class MarketDataRecorder:
    """
    Records market data to disk for replay and training.

    Format: msgpack (efficient binary serialization)
    """

    def __init__(
        self,
        output_dir: str | Path,
        symbol: str,
        buffer_size: int = 1000,
        flush_interval: float = 5.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

        self._buffer: list[RecordedSnapshot] = []
        self._file: Optional[aiofiles.threadpool.AsyncFileIO] = None
        self._writer_task: Optional[asyncio.Task] = None
        self._running = False
        self._record_count = 0
        self._start_time = time.time()

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.filepath = self.output_dir / f"{symbol}_{timestamp}.msgpack"

    async def start(self) -> None:
        """Start recording."""
        self._running = True
        self._file = await aiofiles.open(self.filepath, "wb")
        self._writer_task = asyncio.create_task(self._flush_loop())
        print(f"Recording to {self.filepath}")

    async def stop(self) -> None:
        """Stop recording and flush buffer."""
        self._running = False
        if self._writer_task:
            await self._writer_task
        await self._flush()
        if self._file:
            await self._file.close()
        elapsed = time.time() - self._start_time
        print(f"Recorded {self._record_count} snapshots in {elapsed:.1f}s to {self.filepath}")

    async def record(self, snapshot: OrderBookSnapshot) -> None:
        """Record a snapshot."""
        if not self._running:
            return

        recorded = RecordedSnapshot(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            sequence=snapshot.sequence,
            bids=[(str(p), str(q)) for p, q in snapshot.bids],
            asks=[(str(p), str(q)) for p, q in snapshot.asks],
        )
        self._buffer.append(recorded)
        self._record_count += 1

        if len(self._buffer) >= self.buffer_size:
            await self._flush()

    async def _flush_loop(self) -> None:
        """Periodic flush task."""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self._flush()

    async def _flush(self) -> None:
        """Flush buffer to disk."""
        if not self._buffer or not self._file:
            return

        # Serialize all buffered snapshots
        data = msgpack.packb(
            [asdict(s) for s in self._buffer],
            use_bin_type=True,
        )
        await self._file.write(data)
        await self._file.flush()
        self._buffer.clear()


class MarketDataReplayer:
    """
    Replays recorded market data for backtesting and training.
    """

    def __init__(self, filepath: str | Path, speed: float = 1.0):
        self.filepath = Path(filepath)
        self.speed = speed
        self._snapshots: list[RecordedSnapshot] = []
        self._loaded = False

    async def load(self) -> None:
        """Load all snapshots from file."""
        if self._loaded:
            return

        async with aiofiles.open(self.filepath, "rb") as f:
            data = await f.read()

        # msgpack can contain multiple packed arrays
        unpacked = msgpack.unpackb(data, raw=False)
        self._snapshots = [RecordedSnapshot(**s) for s in unpacked]
        self._loaded = True
        print(f"Loaded {len(self._snapshots)} snapshots from {self.filepath}")

    def get_snapshots(self) -> list[OrderBookSnapshot]:
        """Get all snapshots as OrderBookSnapshot objects."""
        if not self._loaded:
            raise RuntimeError("Call load() first")
        return [s.to_order_book_snapshot() for s in self._snapshots]

    async def replay(
        self,
        on_snapshot: callable,
        realtime: bool = False,
    ) -> None:
        """
        Replay snapshots.

        Args:
            on_snapshot: Callback function(snapshot, timestamp)
            realtime: If True, respect original timing (scaled by speed)
        """
        if not self._loaded:
            await self.load()

        if not self._snapshots:
            return

        start_time = self._snapshots[0].timestamp
        replay_start = time.time()

        for snapshot in self._snapshots:
            if realtime:
                # Calculate target time
                elapsed = (snapshot.timestamp - start_time) / self.speed
                target_time = replay_start + elapsed
                now = time.time()
                if target_time > now:
                    await asyncio.sleep(target_time - now)

            on_snapshot(snapshot.to_order_book_snapshot(), snapshot.timestamp)

    def get_time_range(self) -> tuple[float, float] | None:
        """Get (start_time, end_time) of recorded data."""
        if not self._loaded:
            return None
        if not self._snapshots:
            return None
        return (self._snapshots[0].timestamp, self._snapshots[-1].timestamp)

    def stats(self) -> dict:
        if not self._loaded:
            return {}
        return {
            "num_snapshots": len(self._snapshots),
            "time_range": self.get_time_range(),
            "file_size_mb": self.filepath.stat().st_size / (1024 * 1024),
        }


def find_recordings(data_dir: str | Path, symbol: str | None = None) -> list[Path]:
    """Find all recording files."""
    data_dir = Path(data_dir)
    pattern = f"{symbol}_*.msgpack" if symbol else "*.msgpack"
    return sorted(data_dir.glob(pattern))