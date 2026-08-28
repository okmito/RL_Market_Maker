from __future__ import annotations

from decimal import Decimal
from typing import Optional

import numpy as np

from market_maker.market_data.order_book import OrderBook


class LOBFeatures:
    """
    Extract features from Limit Order Book for RL agent.

    Features include:
    - Price features (mid, microprice, spread)
    - Depth features (multiple levels)
    - Imbalance features
    - Microstructure features
    """

    def __init__(
        self,
        depth_levels: int = 10,
        include_microstructure: bool = True,
    ):
        self.depth_levels = depth_levels
        self.include_microstructure = include_microstructure

    def extract(self, book: OrderBook) -> dict[str, float]:
        """Extract all features from order book."""
        features = {}

        # Price features
        features.update(self._extract_price_features(book))

        # Depth features
        features.update(self._extract_depth_features(book))

        # Imbalance features
        features.update(self._extract_imbalance_features(book))

        # Microstructure features
        if self.include_microstructure:
            features.update(self._extract_microstructure_features(book))

        return features

    def _extract_price_features(self, book: OrderBook) -> dict[str, float]:
        """Extract price-based features."""
        mid = book.mid_price()
        microprice = book.microprice()
        spread = book.spread()
        spread_bps = book.spread_bps()

        return {
            "mid_price": float(mid) if mid else 0.0,
            "microprice": float(microprice) if microprice else 0.0,
            "spread": float(spread) if spread else 0.0,
            "spread_bps": spread_bps if spread_bps else 0.0,
        }

    def _extract_depth_features(self, book: OrderBook) -> dict[str, float]:
        """Extract depth-based features at multiple levels."""
        features = {}
        bids = book.get_bids(self.depth_levels)
        asks = book.get_asks(self.depth_levels)

        for i in range(self.depth_levels):
            # Bid features
            if i < len(bids):
                bid_px, bid_qty = bids[i]
                features[f"bid_price_{i}"] = float(bid_px)
                features[f"bid_qty_{i}"] = float(bid_qty)
            else:
                features[f"bid_price_{i}"] = 0.0
                features[f"bid_qty_{i}"] = 0.0

            # Ask features
            if i < len(asks):
                ask_px, ask_qty = asks[i]
                features[f"ask_price_{i}"] = float(ask_px)
                features[f"ask_qty_{i}"] = float(ask_qty)
            else:
                features[f"ask_price_{i}"] = 0.0
                features[f"ask_qty_{i}"] = 0.0

        # Cumulative depth
        for side, depths in [("bid", bids), ("ask", asks)]:
            cum_qty = Decimal("0")
            for px, qty in depths:
                cum_qty += qty
                features[f"cum_qty_{side}_{px}"] = float(cum_qty)

        # Depth at different levels
        for level in [1, 2, 5, 10]:
            bid_cum = book.cumulative_depth("bid", level)
            ask_cum = book.cumulative_depth("ask", level)
            features[f"cum_bid_depth_{level}"] = float(bid_cum)
            features[f"cum_ask_depth_{level}"] = float(ask_cum)

        return features

    def _extract_imbalance_features(self, book: OrderBook) -> dict[str, float]:
        """Extract order book imbalance features."""
        features = {}

        for level in [1, 2, 5, 10]:
            imbalance = book.depth_imbalance(level)
            features[f"depth_imbalance_{level}"] = imbalance

        # Weighted mid price deviation
        mid = book.mid_price()
        wmid = book.weighted_mid_price(5)
        if mid and wmid:
            features["wmid_deviation_bps"] = float((wmid - mid) / mid * 10000)
        else:
            features["wmid_deviation_bps"] = 0.0

        return features

    def _extract_microstructure_features(self, book: OrderBook) -> dict[str, float]:
        """Extract microstructure features (requires historical data)."""
        return {}  # Placeholder - requires trade stream integration

    def get_feature_names(self) -> list[str]:
        """Get ordered list of feature names for vectorization."""
        names = []

        # Price features
        names.extend(["mid_price", "microprice", "spread", "spread_bps"])

        # Depth features
        for i in range(self.depth_levels):
            names.extend([f"bid_price_{i}", f"bid_qty_{i}"])
        for i in range(self.depth_levels):
            names.extend([f"ask_price_{i}", f"ask_qty_{i}"])

        # Imbalance features
        for level in [1, 2, 5, 10]:
            names.append(f"depth_imbalance_{level}")
        names.append("wmid_deviation_bps")

        # Microstructure
        if self.include_microstructure:
            pass  # Add microstructure feature names

        return names

    def num_features(self) -> int:
        return len(self.get_feature_names())


class MicrostructureFeatures:
    """
    Microstructure features from trade stream.

    Requires:
    - Recent trades
    - Order flow
    - Trade timing
    """

    def __init__(self, window: int = 100):
        self.window = window
        self._trades: list[dict] = []
        self._order_flow: list[int] = []  # +1 buy, -1 sell

    def add_trade(self, price: float, quantity: float, side: str, timestamp: float) -> None:
        """Add a trade to the history."""
        trade = {
            "price": price,
            "quantity": quantity,
            "side": side,
            "timestamp": timestamp,
            "value": price * quantity,
        }
        self._trades.append(trade)
        self._order_flow.append(1 if side == "buy" else -1)

        # Trim to window
        if len(self._trades) > self.window:
            self._trades = self._trades[-self.window:]
            self._order_flow = self._order_flow[-self.window:]

    def compute(self) -> dict[str, float]:
        """Compute microstructure features."""
        if not self._trades:
            return self._default_features()

        features = {}

        # Trade intensity
        if len(self._trades) > 1:
            time_span = self._trades[-1]["timestamp"] - self._trades[0]["timestamp"]
            features["trade_intensity"] = len(self._trades) / max(time_span, 1)
        else:
            features["trade_intensity"] = 0.0

        # Order flow imbalance
        ofi = sum(self._order_flow)
        total_trades = len(self._order_flow)
        features["order_flow_imbalance"] = ofi / max(total_trades, 1)

        # Buy/sell volume imbalance
        buy_vol = sum(t["value"] for t in self._trades if t["side"] == "buy")
        sell_vol = sum(t["value"] for t in self._trades if t["side"] == "sell")
        total_vol = buy_vol + sell_vol
        if total_vol > 0:
            features["volume_imbalance"] = (buy_vol - sell_vol) / total_vol
        else:
            features["volume_imbalance"] = 0.0

        # Average trade size
        avg_size = np.mean([t["quantity"] for t in self._trades])
        features["avg_trade_size"] = avg_size

        # Trade size variance
        if len(self._trades) > 1:
            features["trade_size_std"] = np.std([t["quantity"] for t in self._trades])
        else:
            features["trade_size_std"] = 0.0

        # Price impact proxy (requires price series)
        if len(self._trades) > 1:
            prices = [t["price"] for t in self._trades]
            features["price_impact"] = (max(prices) - min(prices)) / max(prices) if max(prices) > 0 else 0.0
        else:
            features["price_impact"] = 0.0

        return features

    def _default_features(self) -> dict[str, float]:
        return {
            "trade_intensity": 0.0,
            "order_flow_imbalance": 0.0,
            "volume_imbalance": 0.0,
            "avg_trade_size": 0.0,
            "trade_size_std": 0.0,
            "price_impact": 0.0,
        }

    def reset(self) -> None:
        self._trades.clear()
        self._order_flow.clear()


class FeatureEngine:
    """
    Combined feature engine for the RL environment.

    Combines:
    - LOB features
    - Microstructure features
    - Agent state features
    """

    def __init__(
        self,
        depth_levels: int = 10,
        include_microstructure: bool = True,
        include_agent_state: bool = True,
    ):
        self.lob_features = LOBFeatures(depth_levels, include_microstructure)
        self.micro_features = MicrostructureFeatures()
        self.include_agent_state = include_agent_state

        # Cache feature names
        self._feature_names = self.lob_features.get_feature_names()

    def compute(
        self,
        book: OrderBook,
        agent_state: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Compute all features as a numpy array.

        Args:
            book: OrderBook instance
            agent_state: Optional agent state dict with keys like
                inventory, cash, pnl, etc.

        Returns:
            numpy array of features
        """
        features = {}

        # LOB features
        lob_feats = self.lob_features.extract(book)
        features.update(lob_feats)

        # Microstructure features
        if self.include_microstructure:
            micro_feats = self.micro_features.compute()
            features.update(micro_feats)

        # Agent state features
        if self.include_agent_state and agent_state:
            features.update(self._compute_agent_features(agent_state))

        # Convert to ordered array
        return self._to_array(features)

    def _compute_agent_features(self, state: dict) -> dict[str, float]:
        """Compute agent state features."""
        features = {}

        inventory = state.get("inventory", 0.0)
        max_inventory = state.get("max_inventory", 1.0)
        cash = state.get("cash", 0.0)
        pnl = state.get("pnl", 0.0)
        unrealized_pnl = state.get("unrealized_pnl", 0.0)
        num_orders = state.get("num_orders", 0)
        last_fill_time = state.get("time_since_fill", 0.0)

        features["inventory"] = inventory
        features["inventory_ratio"] = inventory / max_inventory if max_inventory > 0 else 0.0
        features["cash"] = cash
        features["pnl"] = pnl
        features["unrealized_pnl"] = unrealized_pnl
        features["num_orders"] = num_orders
        features["time_since_fill"] = last_fill_time
        features["inventory_abs"] = abs(inventory)

        return features

    def _to_array(self, features: dict[str, float]) -> np.ndarray:
        """Convert feature dict to ordered numpy array."""
        return np.array([features.get(name, 0.0) for name in self._feature_names], dtype=np.float32)

    def num_features(self) -> int:
        """Get total number of features."""
        base = self.lob_features.num_features()
        if self.include_microstructure:
            base += 6  # Microstructure features
        if self.include_agent_state:
            base += 8  # Agent state features
        return base

    def get_feature_names(self) -> list[str]:
        """Get all feature names in order."""
        names = self._feature_names.copy()
        if self.include_microstructure:
            names.extend([
                "trade_intensity",
                "order_flow_imbalance",
                "volume_imbalance",
                "avg_trade_size",
                "trade_size_std",
                "price_impact",
            ])
        if self.include_agent_state:
            names.extend([
                "inventory",
                "inventory_ratio",
                "cash",
                "pnl",
                "unrealized_pnl",
                "num_orders",
                "time_since_fill",
                "inventory_abs",
            ])
        return names

    def reset(self) -> None:
        """Reset all feature state."""
        self.micro_features.reset()