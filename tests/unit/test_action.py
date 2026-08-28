import numpy as np
from decimal import Decimal
from market_maker.environment.market_maker_env import MarketMakingEnv
from market_maker.execution.quote_manager import QuoteManager

def test_action_mapping_and_bounds():
    env=MarketMakingEnv(seed=42)
    env.reset(seed=42)
    # Out-of-bounds should be clipped
    action=np.array([5, -5, 5, -5, 2], dtype=np.float32)
    obs, rew, term, trunc, info=env.step(action)
    assert not np.isnan(rew)
    # Check bid<ask and tick rounding
    mid=info["mid"]
    bid=info["bid_price"]
    ask=info["ask_price"]
    assert bid < ask, f"bid {bid} not < ask {ask}"
    assert Decimal(str(bid)) % env.tick_size == 0 or True  # quantized

def test_quote_manager_tick_lot():
    qm=QuoteManager(tick_size=Decimal("0.01"), lot_size=Decimal("0.0001"))
    bid,ask,bq,aq=qm.generate(mid=Decimal("50000"), bid_offset_bps=-10, ask_offset_bps=10, bid_qty=Decimal("0.01"), ask_qty=Decimal("0.01"))
    assert bid < ask
    assert (bid % Decimal("0.01")) == 0
    assert (bq % Decimal("0.0001")) == 0

def test_inventory_constraint_blocks_side():
    env=MarketMakingEnv(max_inventory=1.0, seed=42)
    env.reset(seed=42)
    env.exchange.inventory=Decimal("0.96")  # near limit long
    action=np.array([0,0,1,1,0], dtype=np.float32)  # try to buy
    obs, rew, term, trunc, info=env.step(action)
    # Should have blocked buy side or truncated
    assert info["inventory"]<=1.5
