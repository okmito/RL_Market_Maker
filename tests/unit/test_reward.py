from market_maker.environment.reward import RewardCalculator, RewardConfig

def test_reward_basic():
    cfg=RewardConfig()
    rc=RewardCalculator(cfg)
    r,info=rc.compute(portfolio_value=100000, cash=100000, inventory=0, mid_price=50000, spread_bps=10)
    assert r==0
    r2,info2=rc.compute(portfolio_value=100100, cash=100050, inventory=0.1, mid_price=50050, spread_bps=10)
    # should have pnl delta
    assert info2["pnl_delta"]==100

def test_inventory_penalty():
    cfg=RewardConfig(inventory_penalty_weight=1.0)
    rc=RewardCalculator(cfg)
    rc.compute(portfolio_value=100000, cash=100000, inventory=0, mid_price=50000, spread_bps=10)
    r,info=rc.compute(portfolio_value=100000, cash=100000, inventory=1.0, mid_price=50000, spread_bps=10)
    assert info["inventory_penalty"]>0
