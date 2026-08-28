from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class RewardConfig:
    pnl_weight: float=1.0
    transaction_cost_weight: float=1.0
    inventory_penalty_weight: float=0.5
    adverse_selection_weight: float=0.3
    cancel_penalty_weight: float=0.01
    risk_penalty_weight: float=0.2
    realized_pnl_weight: float=1.0
    unrealized_pnl_weight: float=0.1

class RewardCalculator:
    """
    reward = ΔPnL - transaction_costs - inventory_penalty - adverse_selection - cancel_penalty - risk_penalty
    where inventory_penalty = |inventory| * coeff * spread
    """
    def __init__(self, config: RewardConfig):
        self.config=config
        self._prev_portfolio_value: float|None=None
        self._prev_inventory: float=0

    def reset(self):
        self._prev_portfolio_value=None
        self._prev_inventory=0

    def compute(
        self,
        portfolio_value: float,
        cash: float,
        inventory: float,
        mid_price: float,
        spread_bps: float,
        transaction_cost: float=0,
        num_cancels: int=0,
        adverse_selection_cost: float=0,
        risk_violation: bool=False,
    ) -> tuple[float, dict]:
        if self._prev_portfolio_value is None:
            self._prev_portfolio_value=portfolio_value
            pnl_delta=0.0
        else:
            pnl_delta=portfolio_value - self._prev_portfolio_value
            self._prev_portfolio_value=portfolio_value

        inventory_penalty = abs(inventory) * self.config.inventory_penalty_weight * (spread_bps/10000) * 0.1
        # scale by inventory ratio
        cancel_penalty = num_cancels * self.config.cancel_penalty_weight
        risk_penalty = self.config.risk_penalty_weight if risk_violation else 0.0

        # adverse selection penalty already passed
        reward = (
            self.config.pnl_weight * pnl_delta
            - self.config.transaction_cost_weight * transaction_cost
            - inventory_penalty
            - self.config.adverse_selection_weight * adverse_selection_cost
            - cancel_penalty
            - risk_penalty
        )
        info={
            "pnl_delta": pnl_delta,
            "inventory_penalty": inventory_penalty,
            "transaction_cost": transaction_cost,
            "cancel_penalty": cancel_penalty,
            "adverse_selection": adverse_selection_cost,
            "risk_penalty": risk_penalty,
            "reward": reward,
        }
        return reward, info
