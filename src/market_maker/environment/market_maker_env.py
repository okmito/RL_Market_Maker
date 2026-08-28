from __future__ import annotations
import time
from decimal import Decimal
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from loguru import logger

from market_maker.exchange.simulated_exchange import SimulatedExchange
from market_maker.exchange.base import Order, OrderSide, OrderType
from market_maker.environment.reward import RewardCalculator, RewardConfig
from market_maker.market_data.order_book import OrderBook

class MarketMakingEnv(gym.Env):
    """
    Gymnasium environment for market making.
    Observation: LOB features + agent state (inventory, cash, pnl, etc.)
    Action: continuous vector [bid_offset_bps, ask_offset_bps, bid_size_ratio, ask_size_ratio, inventory_skew]
    Action is mapped to quote parameters relative to mid.
    """
    metadata={"render_modes":["human"]}

    def __init__(
        self,
        exchange: Optional[SimulatedExchange]=None,
        max_steps: int=10000,
        max_inventory: float=1.0,
        tick_size: float=0.01,
        lot_size: float=0.0001,
        reward_config: Optional[RewardConfig]=None,
        depth_levels: int=10,
        seed: int=42,
    ):
        super().__init__()
        self.exchange = exchange or SimulatedExchange(seed=seed, initial_cash=100000, initial_inventory=0)
        self.max_steps=max_steps
        self.max_inventory=max_inventory
        self.tick_size=Decimal(str(tick_size))
        self.lot_size=Decimal(str(lot_size))
        self.reward_calc=RewardCalculator(reward_config or RewardConfig())
        self.depth_levels=depth_levels
        self._rng=np.random.default_rng(seed)
        self._step_count=0
        self._prev_portfolio=None

        # Action space: 5 continuous values in [-1,1] mapped to quotes
        self.action_space=spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        # Observation: depth*4 + mid/spread/imbalance + agent state ~ 50 dims
        # Observation: depth_levels*4 (bid/ask price+qty) + 10 (3 imbalance + 7 misc) + 8 (agent) = 58 with depth=10. Documented as 58-d (not 50) — see reports/rl_pipeline_audit.md.
        obs_dim = depth_levels*4 + 10 + 8
        self.observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _get_obs(self)->np.ndarray:
        book=self.exchange.order_book
        # Price features
        mid = float(book.mid_price() or self.exchange.mid_price)
        spread_bps = book.spread_bps() or 10.0
        micro = float(book.microprice() or mid)
        # Depth features normalize relative to mid
        feats=[]
        bids=book.get_bids(self.depth_levels)
        asks=book.get_asks(self.depth_levels)
        for i in range(self.depth_levels):
            if i < len(bids):
                px,qty=bids[i]
                feats.append(float(px)/mid -1 if mid else 0)
                feats.append(float(qty)/5.0)
            else:
                feats.extend([0,0])
            if i < len(asks):
                px,qty=asks[i]
                feats.append(float(px)/mid -1 if mid else 0)
                feats.append(float(qty)/5.0)
            else:
                feats.extend([0,0])
        # Imbalance & misc
        for lvl in [1,5,10]:
            feats.append(book.depth_imbalance(lvl))
        feats.extend([
            spread_bps/100,
            float(micro)/mid -1 if mid else 0,
            float(self.exchange.inventory)/self.max_inventory,
            float(self.exchange.cash)/100000,
            float(self.exchange.portfolio_value - 100000)/100000,
            float(self.exchange.inventory) * mid/100000,
            len([o for o in self.exchange._orders.values() if o.is_active])/10,
            self._step_count/self.max_steps,
            float(book.weighted_mid_price(5) or mid)/mid -1 if mid else 0,
        ])
        # Pad/truncate to obs_dim
        target_dim=self.observation_space.shape[0]
        if len(feats) < target_dim:
            feats.extend([0]*(target_dim-len(feats)))
        elif len(feats) > target_dim:
            feats=feats[:target_dim]
        arr=np.array(feats, dtype=np.float32)
        # Check NaN/Inf
        arr=np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        return arr

    def _map_action(self, action: np.ndarray):
        # Clamp
        action=np.clip(action, -1, 1)
        bid_offset_bps = float(action[0]*50)  # -50..50 bps
        ask_offset_bps = float(action[1]*50)
        bid_size_ratio = float((action[2]+1)/2 *1.9 +0.1)  # 0.1..2.0
        ask_size_ratio = float((action[3]+1)/2 *1.9 +0.1)
        # inventory skew not separately used; incorporated via offsets already but keep for compat
        skew = float(action[4])  # -1..1
        return bid_offset_bps, ask_offset_bps, bid_size_ratio, ask_size_ratio, skew

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng=np.random.default_rng(seed)
        self._step_count=0
        self.exchange.inventory=Decimal("0")
        self.exchange.cash=Decimal("100000")
        self.exchange._orders.clear()
        self.exchange._queue_position.clear()
        self.exchange.mid_price=Decimal("50000")
        # reset book
        self.exchange._init_book(self.exchange.mid_price - Decimal("2.5"), self.exchange.mid_price + Decimal("2.5"))
        self.reward_calc.reset()
        self._prev_portfolio=float(self.exchange.portfolio_value)
        obs=self._get_obs()
        return obs, {}

    def step(self, action):
        self._step_count+=1
        mid = float(self.exchange.order_book.mid_price() or self.exchange.mid_price)
        bid_off, ask_off, bid_ratio, ask_ratio, skew = self._map_action(np.array(action, dtype=np.float32))

        # Inventory skew adjustment: reduce size on side that increases inventory risk
        inv = float(self.exchange.inventory)
        inv_ratio = inv/self.max_inventory
        # If long, reduce bid size/increase ask aggressiveness
        if inv_ratio > 0.5:
            bid_ratio *= max(0.2, 1 - inv_ratio)
            ask_off -= skew*5  # skew negative to encourage selling
        elif inv_ratio < -0.5:
            ask_ratio *= max(0.2, 1 + inv_ratio)
            bid_off += skew*5

        # Risk limits: restrict dangerous side entirely if at max
        if abs(inv_ratio) >= 0.95:
            if inv_ratio>0:
                bid_ratio=0
            else:
                ask_ratio=0

        # Generate quotes via QuoteManager (centralized, verified tick/lot handling)
        from market_maker.execution.quote_manager import QuoteManager
        qm = QuoteManager(tick_size=self.tick_size, lot_size=self.lot_size)
        base_qty=Decimal("0.01")
        bid_price, ask_price, bid_qty, ask_qty = qm.generate(
            mid=Decimal(str(mid)), bid_offset_bps=bid_off, ask_offset_bps=ask_off,
            bid_qty=base_qty*Decimal(str(bid_ratio)), ask_qty=base_qty*Decimal(str(ask_ratio))
        )
        # Place orders (cancel previous first)
        # Simple: cancel all then place new quotes if qty>0
        import asyncio
        # synchronous cancel: direct manipulation
        for oid, o in list(self.exchange._orders.items()):
            if o.is_active:
                o.status=o.status.__class__.CANCELED
        # Place bid/ask
        fills_before=len([o for o in self.exchange._orders.values() if o.is_filled])
        if bid_ratio>0.05:
            o=Order(symbol=self.exchange.symbol, side=OrderSide.BUY, type=OrderType.LIMIT, quantity=bid_qty, price=bid_price)
            o.order_id=self.exchange._next_order_id; self.exchange._next_order_id+=1
            o.status=o.status.__class__.NEW
            self.exchange._orders[o.order_id]=o
            self.exchange._queue_position[o.order_id]=self.exchange._rng.uniform(0.3,0.9)
        if ask_ratio>0.05:
            o=Order(symbol=self.exchange.symbol, side=OrderSide.SELL, type=OrderType.LIMIT, quantity=ask_qty, price=ask_price)
            o.order_id=self.exchange._next_order_id; self.exchange._next_order_id+=1
            o.status=o.status.__class__.NEW
            self.exchange._orders[o.order_id]=o
            self.exchange._queue_position[o.order_id]=self.exchange._rng.uniform(0.3,0.9)

        # Advance market
        self.exchange.step_market(steps=1)

        # Compute reward — pnl_delta already includes fees via cash (SimulatedExchange cash -= commission), so transaction_cost is 0 to avoid double-count. See reward.py and reports/initial_audit.md.
        portfolio=float(self.exchange.portfolio_value)
        spread_bps=self.exchange.order_book.spread_bps() or 0
        transaction_cost=0.0  # Do NOT sum commissions here — already in portfolio_value via cash
        # adverse selection proxy: if inventory and price moved against
        adv=0.0
        if len(self.exchange._price_history)>=2:
            price_move = self.exchange._price_history[-1] - self.exchange._price_history[-2]
            adv = - float(inv) * price_move / mid * 0.01

        risk_violation = abs(float(self.exchange.inventory)/self.max_inventory) > 0.95
        reward, info = self.reward_calc.compute(portfolio_value=portfolio, cash=float(self.exchange.cash), inventory=float(self.exchange.inventory), mid_price=mid, spread_bps=spread_bps, transaction_cost=transaction_cost, num_cancels=0, adverse_selection_cost=adv, risk_violation=risk_violation)
        # Scale reward
        reward = float(np.clip(reward/100, -10, 10))

        obs=self._get_obs()
        terminated=False
        truncated=self._step_count>=self.max_steps
        # Hard inventory violation -> truncated with penalty
        if abs(float(self.exchange.inventory)) > self.max_inventory*1.5:
            reward-=5
            truncated=True

        info.update({"portfolio":portfolio, "inventory":float(self.exchange.inventory), "mid":mid, "bid_price":float(bid_price), "ask_price":float(ask_price)})
        return obs, reward, terminated, truncated, info

    def render(self):
        mid=float(self.exchange.order_book.mid_price() or self.exchange.mid_price)
        print(f"Step {self._step_count} mid={mid:.2f} inv={self.exchange.inventory} cash={self.exchange.cash} port={self.exchange.portfolio_value}")
