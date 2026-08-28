import ray
ray.init(ignore_reinit_error=True)
print('init', ray.is_initialized())
from market_maker.agents.ppo import get_ppo_config
from market_maker.environment.market_maker_env import MarketMakingEnv
def mk():
    return MarketMakingEnv(seed=42)
cfg=get_ppo_config(lambda c: mk(), {'num_workers':0})
print('cfg', cfg)
algo=cfg.build()
print('built')
r=algo.train()
print('reward', r.get('episode_reward_mean'))
ray.shutdown()
