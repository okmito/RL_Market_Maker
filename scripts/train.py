import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
#!/usr/bin/env python
"""Train RL agent."""
import argparse, pathlib, yaml
from market_maker.environment.market_maker_env import MarketMakingEnv
from market_maker.exchange.simulated_exchange import SimulatedExchange

def make_env(seed=42):
    def _creator(config=None):
        ex=SimulatedExchange(seed=seed)
        return MarketMakingEnv(exchange=ex, max_steps=1000, seed=seed)
    return _creator

def main():
    parser=argparse.ArgumentParser(description="Train PPO")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--checkpoint-dir", type=str, default="models")
    args=parser.parse_args()

    # Load config
    import yaml
    with open(args.config) as f:
        cfg=yaml.safe_load(f)
    ppo_cfg=cfg.get("ppo", {})
    # Force single-process for Windows/local without cluster
    ppo_cfg["num_workers"] = 0
    ppo_cfg["num_envs_per_worker"] = 1

    print(f"Training for {args.iters} iterations")
    try:
        from market_maker.agents.ppo import train_ppo
        ckpt=train_ppo(make_env(), ppo_cfg, total_iters=args.iters, checkpoint_dir=args.checkpoint_dir)
        print(f"Done checkpoint {ckpt}")
    except Exception as e:
        print(f"Training failed or RLlib not available: {e}")
        # Fallback: simple sanity training without Ray
        print("Running sanity env loop instead")
        env=make_env()(None)
        obs,_=env.reset()
        for i in range(args.iters*1000):
            import numpy as np
            action=np.random.uniform(-1,1,size=(5,))
            obs,reward,term,trunc,_=env.step(action)
            if term or trunc:
                obs,_=env.reset()
        print("Sanity loop completed")
        pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{args.checkpoint_dir}/sanity.txt").write_text("sanity completed")

if __name__=="__main__":
    main()

