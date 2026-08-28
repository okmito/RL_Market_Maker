from __future__ import annotations
import os
from pathlib import Path

def get_ppo_config(env_creator, config_dict: dict):
    """Build RLlib PPO config dict verified against installed Ray version."""
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        # New API
        cfg = PPOConfig()
        # Resolve env: support callable creator or class
        env_spec = env_creator
        env_config = {}
        # if env_creator is a function returning env class instance, wrap via registered env
        if callable(env_creator):
            try:
                # try calling to see if it returns an Env class or instance? if it returns a Gym env, need to register
                from ray.tune.registry import register_env
                def _wrap(cfg_dict):
                    # env_creator is _creator(config) from train.py make_env
                    try:
                        inner = env_creator(cfg_dict)
                    except TypeError:
                        inner = env_creator()
                    # if returned is still callable (double-wrap), unwrap
                    if callable(inner) and not isinstance(inner, type) and hasattr(inner, '__code__'):
                        try:
                            inner = inner(cfg_dict)
                        except Exception:
                            pass
                    return inner
                register_env("mm_env", lambda c: _wrap(c))
                env_spec = "mm_env"
            except Exception:
                # fallback to class
                from market_maker.environment.market_maker_env import MarketMakingEnv
                env_spec = MarketMakingEnv
        cfg = cfg.environment(env=env_spec, env_config=env_config)
        cfg.framework(framework=config_dict.get("framework","torch"))
        try:
            cfg.env_runners(num_env_runners=config_dict.get("num_workers",0), num_envs_per_env_runner=config_dict.get("num_envs_per_worker",1))
        except TypeError:
            try:
                cfg.rollouts(num_rollout_workers=config_dict.get("num_workers",0), num_envs_per_env_runner=config_dict.get("num_envs_per_worker",1))
            except TypeError:
                cfg.rollouts(num_rollout_workers=config_dict.get("num_workers",0), num_envs_per_worker=config_dict.get("num_envs_per_worker",1))
        # Map old keys to new RLlib 2.58 names; lr must be schedule list in new API
        lr_val = config_dict.get("lr",3e-4)
        if isinstance(lr_val, str):
            try:
                lr_val = float(lr_val)
            except ValueError:
                pass
        if isinstance(lr_val, (int,float)):
            lr_cfg = [(0, float(lr_val)), (1000000, float(lr_val))]
        else:
            lr_cfg = lr_val
            if isinstance(lr_cfg, list) and len(lr_cfg) == 1:
                lr_cfg = [lr_cfg[0], (lr_cfg[0][0]+1000000, lr_cfg[0][1])]
        # Try full training config, fallback to minimal if validation fails
        try:
            cfg.training(
                train_batch_size=config_dict.get("train_batch_size",4000),
                minibatch_size=config_dict.get("sgd_minibatch_size", config_dict.get("minibatch_size",128)),
                num_epochs=config_dict.get("num_sgd_iter", config_dict.get("num_epochs",10)),
                lr=lr_cfg,
                gamma=config_dict.get("gamma",0.99),
                lambda_=config_dict.get("lambda", config_dict.get("lambda_",0.95)),
                clip_param=config_dict.get("clip_param",0.2),
                entropy_coeff=config_dict.get("entropy_coeff",0.01),
            )
        except Exception:
            # Minimal training config for new API
            cfg.training(train_batch_size=config_dict.get("train_batch_size",4000), lr=lr_cfg)
        return cfg
    except ImportError as e:
        raise RuntimeError(f"RLLib not installed: {e}")
    except Exception as e:
        # Fallback dict config for older Ray
        return {"env": env_creator, "framework": "torch", **config_dict}

def train_ppo(env_creator, ppo_config: dict, total_iters: int=10, checkpoint_dir: str="./models"):
    import os
    # Remove stale RAY_ADDRESS=auto from .env so local ray.init starts a new cluster
    if os.getenv("RAY_ADDRESS") == "auto":
        os.environ.pop("RAY_ADDRESS", None)
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    cfg = get_ppo_config(env_creator, ppo_config)
    if isinstance(cfg, dict):
        from ray.rllib.algorithms.ppo import PPO
        algo = PPO(config=cfg)
    else:
        algo = cfg.build()
    ckpt_dir = Path(checkpoint_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for i in range(total_iters):
        result = algo.train()
        # New API stores metrics under env_runners
        rew = result.get('episode_reward_mean')
        if rew is None:
            rew = result.get('env_runners', {}).get('episode_return_mean')
        leng = result.get('episode_len_mean')
        if leng is None:
            leng = result.get('env_runners', {}).get('episode_len_mean')
        print(f"Iter {i}: reward={rew} len={leng}")
        if (i+1)%5==0:
            try:
                ckpt = algo.save(str(ckpt_dir))
                print(f"Checkpoint {ckpt}")
            except Exception as e:
                print(f"Checkpoint save failed: {e}")
                ckpt = str(ckpt_dir / f"iter_{i}")
    try:
        ckpt = algo.save(str(ckpt_dir))
    except Exception as e:
        print(f"Final save failed: {e}")
        ckpt = str(ckpt_dir)
    ray.shutdown()
    return ckpt
