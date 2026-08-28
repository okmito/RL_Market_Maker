import yaml, ray, traceback
from market_maker.agents.ppo import get_ppo_config
from market_maker.environment.market_maker_env import MarketMakingEnv
with open("configs/base.yaml") as f:
    cfg_dict=yaml.safe_load(f)["ppo"]
print(cfg_dict)
ray.init(ignore_reinit_error=True, include_dashboard=False)
try:
    cfg=get_ppo_config(lambda c: MarketMakingEnv(seed=42), cfg_dict)
    print("cfg type", type(cfg))
    algo=cfg.build()
    print("built")
except Exception as e:
    traceback.print_exc()
finally:
    ray.shutdown()
