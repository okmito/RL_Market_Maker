from decimal import Decimal
from market_maker.exchange.simulated_exchange import SimulatedExchange
from market_maker.exchange.base import Order, OrderSide, OrderType
import time

def test_pnl_reconciliation():
    ex=SimulatedExchange(seed=42, initial_cash=100000, initial_inventory=0, maker_fee_bps=1, taker_fee_bps=5)
    mid=ex.order_book.mid_price()
    # Buy 0.1 BTC at 50000
    o=Order(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.LIMIT, quantity=Decimal("0.1"), price=Decimal("60000"))
    o.order_id=ex._next_order_id; ex._next_order_id+=1
    ex._orders[o.order_id]=o
    ex._queue_position[o.order_id]=0.0  # front
    ex.step_market(1)
    # Check cash+inv*mid == portfolio
    port=float(ex.portfolio_value)
    cash=float(ex.cash)
    inv=float(ex.inventory)
    mid_f=float(ex.order_book.mid_price() or ex.mid_price)
    assert abs(port - (cash + inv*mid_f)) < 1e-6
    # Fees deducted from cash
    assert cash < 100000 or inv!=0

def test_equity_formula():
    ex=SimulatedExchange(seed=1)
    ex.cash=Decimal("50000")
    ex.inventory=Decimal("1.0")
    ex.mid_price=Decimal("50000")
    ex.order_book.initialize_from_snapshot(bids=[(Decimal("49999"),Decimal("1"))], asks=[(Decimal("50001"),Decimal("1"))], timestamp=time.time(), sequence=1)
    assert float(ex.portfolio_value) == 50000 + 1*50000
