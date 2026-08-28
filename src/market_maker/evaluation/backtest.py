from __future__ import annotations
from market_maker.evaluation.benchmark import FixedSpreadStrategy, InventorySkewStrategy
from market_maker.environment.market_maker_env import MarketMakingEnv
import numpy as np

def backtest_rl_policy(policy_fn, env_steps: int=1000, seed: int=42)->dict:
    env=MarketMakingEnv(seed=seed)
    obs,_=env.reset(seed=seed)
    hist=[]; inv_hist=[]
    for _ in range(env_steps):
        action=policy_fn(obs)
        obs, reward, term, trunc, info=env.step(action)
        hist.append(info["portfolio"])
        inv_hist.append(info["inventory"])
        if term or trunc:
            break
    from market_maker.evaluation.metrics import compute_metrics
    return compute_metrics(hist, inv_hist, [], 0)

def compare_all(env_steps: int=2000, seed: int=42)->dict:
    fixed=FixedSpreadStrategy().run(env_steps, seed)
    skew=InventorySkewStrategy().run(env_steps, seed)
    # RL random policy baseline for now (uniform)
    def random_policy(obs):
        return np.random.uniform(-1,1,size=(5,)).astype(np.float32)
    rl=backtest_rl_policy(random_policy, env_steps, seed)
    return {"fixed_spread":fixed, "inventory_skew":skew, "rl_random":rl}
