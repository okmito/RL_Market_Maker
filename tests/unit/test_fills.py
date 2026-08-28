import random
from market_maker.environment.fills import FillConfig, should_fill

def test_should_fill():
    cfg=FillConfig(model="queue_aware")
    rng=random.Random(42)
    fill,maker=should_fill(order_price=50001, market_bid=50000, market_ask=50001, side="BUY", queue_position=0.5, config=cfg, rng=rng)
    assert isinstance(fill,bool)
