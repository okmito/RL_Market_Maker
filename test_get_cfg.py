import ray
from market_maker.agents.ppo import get_ppo_config
from market_maker.environment.market_maker_env import MarketMakingEnv
ray.init(ignore_reinit_error=True, include_dashboard=False)
cfg=get_ppo_config(lambda c: MarketMakingEnv(seed=42), {'num_workers':0, 'train_batch_size':4000, 'sgd_minibatch_size':128, 'num_sgd_iter':10, 'lr':3e-4, 'gamma':0.99, 'lambda':0.95, 'clip_param':0.2, 'entropy_coeff':0.01})
print(type(cfg))
print(cfg)
try:
    algo=cfg.build()
    print("built ok")
    r=algo.train()
    print(r.get('episode_reward_mean'))
except Exception as e:
    import traceback; traceback.print_exc()
finally:
    ray.shutdown()
