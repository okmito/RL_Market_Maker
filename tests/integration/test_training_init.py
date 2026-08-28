import pytest

def test_ppo_config():
    try:
        import ray
        from market_maker.environment.market_maker_env import MarketMakingEnv
        from market_maker.agents.ppo import get_ppo_config
        def maker(cfg=None):
            return MarketMakingEnv(seed=42)
        cfg={"num_workers":0, "framework":"torch", "lr":3e-4}
        c=get_ppo_config(maker, cfg)
        assert c is not None
    except ImportError:
        pytest.skip("Ray not installed")

def test_env_gymnasium_api():
    from market_maker.environment.market_maker_env import MarketMakingEnv
    import numpy as np
    env=MarketMakingEnv(seed=42)
    assert hasattr(env, "observation_space")
    assert hasattr(env, "action_space")
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    obs,info=env.reset()
    assert isinstance(obs, np.ndarray)
