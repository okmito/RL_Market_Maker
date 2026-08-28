from __future__ import annotations
from decimal import Decimal
from market_maker.exchange.simulated_exchange import SimulatedExchange
from market_maker.exchange.base import Order, OrderSide, OrderType
import numpy as np

class FixedSpreadStrategy:
    def __init__(self, spread_bps: float=10, qty: float=0.01):
        self.spread_bps=spread_bps
        self.qty=Decimal(str(qty))
    def act(self, mid: float)->tuple[Decimal,Decimal]:
        bid = Decimal(str(mid*(1-self.spread_bps/10000/2)))
        ask = Decimal(str(mid*(1+self.spread_bps/10000/2)))
        return bid, ask
    def run(self, env_steps: int=1000, seed: int=42)->dict:
        ex=SimulatedExchange(seed=seed)
        hist=[]
        inv_hist=[]
        for _ in range(env_steps):
            mid=float(ex.order_book.mid_price() or ex.mid_price)
            bid,ask=self.act(mid)
            # cancel and place
            for oid,o in list(ex._orders.items()):
                if o.is_active:
                    o.status=o.status.__class__.CANCELED
            for side, price in [(OrderSide.BUY,bid),(OrderSide.SELL,ask)]:
                o=Order(symbol=ex.symbol, side=side, type=OrderType.LIMIT, quantity=self.qty, price=price)
                o.order_id=ex._next_order_id; ex._next_order_id+=1
                ex._orders[o.order_id]=o
                ex._queue_position[o.order_id]=ex._rng.uniform(0.3,0.9)
            ex.step_market(1)
            hist.append(float(ex.portfolio_value))
            inv_hist.append(float(ex.inventory))
        from market_maker.evaluation.metrics import compute_metrics
        return compute_metrics(hist, inv_hist, [], 0)

class InventorySkewStrategy:
    def __init__(self, base_spread_bps: float=10, qty: float=0.01, skew_coeff: float=5):
        self.base=base_spread_bps
        self.qty=Decimal(str(qty))
        self.skew_coeff=skew_coeff
    def act(self, mid: float, inventory: float, max_inv: float=1.0)->tuple[Decimal,Decimal]:
        skew = -inventory/max_inv * self.skew_coeff  # bps
        bid_off = -self.base/2 + skew
        ask_off = self.base/2 + skew
        bid = Decimal(str(mid*(1+bid_off/10000)))
        ask = Decimal(str(mid*(1+ask_off/10000)))
        return bid,ask
    def run(self, env_steps: int=1000, seed: int=42)->dict:
        ex=SimulatedExchange(seed=seed)
        hist=[]; inv_hist=[]
        for _ in range(env_steps):
            mid=float(ex.order_book.mid_price() or ex.mid_price)
            inv=float(ex.inventory)
            bid,ask=self.act(mid, inv)
            for oid,o in list(ex._orders.items()):
                if o.is_active:
                    o.status=o.status.__class__.CANCELED
            for side, price in [(OrderSide.BUY,bid),(OrderSide.SELL,ask)]:
                o=Order(symbol=ex.symbol, side=side, type=OrderType.LIMIT, quantity=self.qty, price=price)
                o.order_id=ex._next_order_id; ex._next_order_id+=1
                ex._orders[o.order_id]=o
                ex._queue_position[o.order_id]=ex._rng.uniform(0.3,0.9)
            ex.step_market(1)
            hist.append(float(ex.portfolio_value))
            inv_hist.append(float(ex.inventory))
        from market_maker.evaluation.metrics import compute_metrics
        return compute_metrics(hist, inv_hist, [], 0)
