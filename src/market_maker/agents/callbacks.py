from __future__ import annotations
try:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    class MarketMakerCallback(DefaultCallbacks):
        def on_episode_start(self, *, episode, base_env, policies, env_index, **kwargs):
            pass
        def on_episode_end(self, *, episode, base_env, policies, env_index, **kwargs):
            # log inventory variance
            info=episode.last_info_for()
            if info:
                episode.custom_metrics["inventory"] = info.get("inventory",0)
                episode.custom_metrics["portfolio"] = info.get("portfolio",0)
except Exception:
    class MarketMakerCallback:
        pass
