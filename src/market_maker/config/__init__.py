from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

CONFIG_DIR = Path(__file__).parent.parent / "configs"


class ExchangeConfig(BaseModel):
    type: str = "simulated"
    testnet_base_url: str = "https://testnet.binance.vision"
    testnet_ws_url: str = "wss://testnet.binance.vision/ws"
    production_base_url: str = "https://api.binance.com"
    production_ws_url: str = "wss://stream.binance.com:9443/ws"
    recv_window: int = 5000
    request_timeout: int = 10


class MarketDataConfig(BaseModel):
    depth_levels: int = 20
    update_speed: str = "100ms"
    snapshot_interval: int = 3600
    max_reconnect_attempts: int = 10
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 60.0
    heartbeat_interval: int = 30


class LOBConfig(BaseModel):
    max_price_levels: int = 100
    price_precision: int = 8
    quantity_precision: int = 8
    validate_sequence: bool = True
    allow_crossed_book: bool = False
    stale_threshold_ms: int = 5000


class FeaturesConfig(BaseModel):
    lookback_window: int = 50
    normalization_window: int = 1000
    depth_levels_for_features: int = 10
    include_microstructure: bool = True
    include_agent_state: bool = True


class SimulatorConfig(BaseModel):
    initial_price: float = 50000.0
    initial_spread_bps: float = 10.0
    tick_size: float = 0.01
    lot_size: float = 0.0001
    min_order_size: float = 0.0001
    max_order_size: float = 10.0
    maker_fee_bps: float = 1.0
    taker_fee_bps: float = 5.0
    latency_ms: int = 10
    fill_model: str = "queue_aware"
    volatility: float = 0.02
    drift: float = 0.0
    mean_reversion: float = 0.1
    jump_probability: float = 0.001
    jump_size: float = 0.05


class RiskConfig(BaseModel):
    max_inventory: float = 1.0
    max_inventory_ratio: float = 0.8
    target_inventory: float = 0.0
    max_position_usd: float = 100000.0
    max_daily_loss_usd: float = 5000.0
    max_order_count_per_minute: int = 100
    max_open_orders: int = 20
    inventory_penalty_coeff: float = 0.1
    adverse_selection_penalty_coeff: float = 0.05


class RLEnvConfig(BaseModel):
    observation_type: str = "continuous"
    action_type: str = "continuous"
    max_episode_steps: int = 10000
    reward_scaling: float = 1.0
    include_inventory_in_obs: bool = True
    include_pnl_in_obs: bool = True
    include_open_orders_in_obs: bool = True


class ActionSpaceConfig(BaseModel):
    bid_offset_min_bps: float = -50.0
    bid_offset_max_bps: float = 50.0
    ask_offset_min_bps: float = -50.0
    ask_offset_max_bps: float = 50.0
    size_min_ratio: float = 0.1
    size_max_ratio: float = 2.0
    inventory_skew_min: float = -1.0
    inventory_skew_max: float = 1.0


class RewardConfig(BaseModel):
    pnl_weight: float = 1.0
    transaction_cost_weight: float = 1.0
    inventory_penalty_weight: float = 0.5
    adverse_selection_weight: float = 0.3
    cancel_penalty_weight: float = 0.01
    risk_penalty_weight: float = 0.2
    realized_pnl_weight: float = 1.0
    unrealized_pnl_weight: float = 0.1


class PPOConfig(BaseModel):
    algorithm: str = "PPO"
    framework: str = "torch"
    num_workers: int = 2
    num_envs_per_worker: int = 1
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 128
    num_sgd_iter: int = 10
    lr: float = 3e-4
    gamma: float = 0.99
    lambda_: float = Field(0.95, alias="lambda")
    clip_param: float = 0.2
    vf_clip_param: float = 10.0
    entropy_coeff: float = 0.01
    entropy_coeff_schedule: list | None = None
    kl_coeff: float = 0.2
    kl_target: float = 0.01
    grad_clip: float = 0.5
    model: dict = Field(default_factory=lambda: {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        "vf_share_layers": False,
        "free_log_std": True,
    })


class TrainingConfig(BaseModel):
    total_timesteps: int = 1000000
    checkpoint_freq: int = 10
    evaluation_interval: int = 50
    evaluation_duration: int = 10
    evaluation_duration_unit: str = "episodes"
    seed: int = 42
    experiment_name: str = "rl_market_maker"
    storage_path: str = "./models"


class EvaluationConfig(BaseModel):
    num_episodes: int = 100
    baseline_strategies: list[str] = Field(default_factory=lambda: ["fixed_spread", "inventory_skew"])
    metrics: list[str] = Field(default_factory=lambda: [
        "total_pnl", "realized_pnl", "unrealized_pnl", "sharpe_ratio",
        "max_drawdown", "inventory_variance", "fill_rate",
        "spread_captured", "adverse_selection", "transaction_costs"
    ])


class BacktestConfig(BaseModel):
    data_source: str = "synthetic"
    start_date: str | None = None
    end_date: str | None = None
    initial_cash: float = 100000.0
    initial_inventory: float = 0.0
    commission_bps: float = 1.0
    slippage_bps: float = 0.5


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/market_maker.log"
    rotation: str = "1 day"
    retention: str = "7 days"
    compression: str = "gz"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    enable_dashboard: bool = True


class Config(BaseModel):
    environment: str = "simulation"
    symbol: str = "BTCUSDT"
    base_asset: str = "BTC"
    quote_asset: str = "USDT"
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    lob: LOBConfig = Field(default_factory=LOBConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    rl_env: RLEnvConfig = Field(default_factory=RLEnvConfig)
    action_space: ActionSpaceConfig = Field(default_factory=ActionSpaceConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    # Extra fields from environment-specific configs
    synthetic_data: dict = Field(default_factory=dict)
    testnet: dict = Field(default_factory=dict)
    callbacks: list = Field(default_factory=list)
    logger_config: dict = Field(default_factory=dict)
    tune: dict = Field(default_factory=dict)


def load_config(config_name: str = "base") -> Config:
    """Load configuration from YAML files."""
    base_path = CONFIG_DIR / "base.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found at {base_path}")

    with open(base_path) as f:
        base_config = yaml.safe_load(f) or {}

    if config_name != "base":
        env_path = CONFIG_DIR / f"{config_name}.yaml"
        if env_path.exists():
            with open(env_path) as f:
                env_config = yaml.safe_load(f) or {}
            base_config = _deep_merge(base_config, env_config)

    # Override with environment variables
    base_config = _apply_env_overrides(base_config)

    return Config(**base_config)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: dict) -> dict:
    """Apply environment variable overrides."""
    env_mappings = {
        "ENVIRONMENT": ("environment", str),
        "SYMBOL": ("symbol", str),
        "BINANCE_TESTNET_API_KEY": ("testnet", "api_key", str),
        "BINANCE_TESTNET_API_SECRET": ("testnet", "api_secret", str),
        "DATABASE_URL": ("database_url", str),
        "LOG_LEVEL": ("logging", "level", str),
        "RAY_ADDRESS": ("ray_address", str),
        "RAY_NUM_CPUS": ("ray_num_cpus", int),
        "RAY_NUM_GPUS": ("ray_num_gpus", int),
    }

    for env_var, path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            if len(path) == 2:
                key, type_fn = path
                config[key] = type_fn(value)
            elif len(path) == 3:
                section, key, type_fn = path
                if section not in config:
                    config[section] = {}
                config[section][key] = type_fn(value)

    return config


# Global config instance
_config: Config | None = None


def get_config(config_name: str | None = None) -> Config:
    """Get global config instance."""
    global _config
    if _config is None or config_name is not None:
        name = config_name or os.getenv("ENVIRONMENT", "simulation")
        _config = load_config(name)
    return _config


def set_config(config: Config) -> None:
    """Set global config instance."""
    global _config
    _config = config