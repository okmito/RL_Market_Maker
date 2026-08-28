from ray.rllib.algorithms.ppo import PPOConfig
from market_maker.environment.market_maker_env import MarketMakingEnv
from ray.tune.registry import register_env
import ray
register_env("mm_env", lambda c: MarketMakingEnv(seed=42))
ray.init(ignore_reinit_error=True, include_dashboard=False)
try:
    cfg=PPOConfig()
    cfg=cfg.environment(env="mm_env")
    cfg=cfg.framework(framework="torch")
    cfg=cfg.env_runners(num_env_runners=0)
    cfg=cfg.training(train_batch_size=4000, minibatch_size=128, num_epochs=10, lr=[(0,0.0003),(1000000,0.0003)], gamma=0.99, lambda_=0.95, clip_param=0.2, entropy_coeff=0.01)
    print("cfg training ok")
    algo=cfg.build()
    print("built")
except Exception as e:
    import traceback; traceback.print_exc()
finally:
    ray.shutdown()
