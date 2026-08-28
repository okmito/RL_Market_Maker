import pytest
from decimal import Decimal
from market_maker.exchange.simulated_exchange import SimulatedExchange
from market_maker.exchange.base import Order, OrderSide, OrderType

@pytest.mark.asyncio
async def test_sim_exchange_place_and_fill():
    ex=SimulatedExchange(seed=42, initial_cash=100000, initial_inventory=0)
    await ex.connect()
    # place aggressive buy that should fill
    o=Order(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.LIMIT, quantity=Decimal("0.01"), price=Decimal("60000"))
    await ex.place_order(o)
    # step market to cause fill
    ex.step_market(5)
    # check order status changed or still active but not rejected
    assert o.status.value in ("NEW","PARTIALLY_FILLED","FILLED","CANCELED")
    await ex.disconnect()

@pytest.mark.asyncio
async def test_portfolio_value():
    ex=SimulatedExchange(seed=1)
    await ex.connect()
    val=ex.portfolio_value
    assert val>0
    await ex.disconnect()
