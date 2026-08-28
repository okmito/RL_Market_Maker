import numpy as np
from market_maker.environment.market_maker_env import MarketMakingEnv

def test_observation_shape_and_finite():
    env=MarketMakingEnv(seed=42, depth_levels=10)
    obs,_=env.reset(seed=42)
    # Actual is 58, not 50 — documented in rl_pipeline_audit.md
    assert obs.shape[0]==58, f"Expected 58, got {obs.shape[0]}"
    assert obs.dtype==np.float32
    assert not np.isnan(obs).any(), "Observation contains NaN"
    assert not np.isinf(obs).any(), "Observation contains Inf"
    assert np.isfinite(obs).all()

def test_observation_table():
    # Verify documented features count: depth*4=40 +10 misc +8 agent =58
    env=MarketMakingEnv(depth_levels=10)
    assert env.observation_space.shape[0]==58
    env5=MarketMakingEnv(depth_levels=5)
    assert env5.observation_space.shape[0]==5*4+10+8
