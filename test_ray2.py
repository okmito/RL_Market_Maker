import ray, traceback
from ray.rllib.algorithms.ppo import PPOConfig
from market_maker.environment.market_maker_env import MarketMakingEnv
from ray.tune.registry import register_env
def _wrap(cfg):
    return MarketMakingEnv(seed=42)
register_env("mm_env", lambda c: _wrap(c))
ray.init(ignore_reinit_error=True)
try:
    cfg = PPOConfig()
    cfg = cfg.environment(env="mm_env")
    cfg = cfg.framework(framework="torch")
    cfg = cfg.env_runners(num_env_runners=0)
    cfg = cfg.training(train_batch_size=4000, sgd_minibatch_size=128, num_sgd_iter=3, lr=3e-4, gamma=0.99)
    print("cfg built")
    algo=cfg.build()
    print("algo built", algo)
    r=algo.train()
    print("reward", r.get('episode_reward_mean'))
except Exception as e:
    traceback.print_exc()
finally:
    ray.shutdown()
