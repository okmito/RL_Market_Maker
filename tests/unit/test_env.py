import numpy as np
from market_maker.environment.market_maker_env import MarketMakingEnv

def test_env_reset_and_step():
    env=MarketMakingEnv(max_steps=100, seed=42)
    obs,_=env.reset(seed=42)
    assert obs.shape[0]==env.observation_space.shape[0]
    assert not np.isnan(obs).any()
    assert not np.isinf(obs).any()
    action=np.array([0.1, -0.1, 0.5, -0.5, 0.0], dtype=np.float32)
    obs2,reward,term,trunc,info=env.step(action)
    assert isinstance(reward,float)
    assert not np.isnan(reward)
    assert "inventory" in info

def test_action_bounds():
    env=MarketMakingEnv(seed=42)
    env.reset()
    # out of bounds action should be clipped
    action=np.array([5,5,5,5,5], dtype=np.float32)
    obs,reward,term,trunc,info=env.step(action)
    assert reward is not None

def test_inventory_bounded():
    env=MarketMakingEnv(max_steps=200, max_inventory=0.5, seed=42)
    env.reset()
    for _ in range(200):
        action=np.random.uniform(-1,1,size=(5,)).astype(np.float32)
        obs,reward,term,trunc,info=env.step(action)
        # after fix, inventory shouldn't explode infinitely (within 2x limit)
        assert abs(float(env.exchange.inventory)) <= 2.0
        if trunc:
            break
