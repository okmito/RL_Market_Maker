from __future__ import annotations
from decimal import Decimal
from dataclasses import dataclass

@dataclass
class RiskLimits:
    max_inventory: float=1.0
    max_position_usd: float=100000
    max_daily_loss: float=5000
    max_open_orders: int=20

class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.limits=limits
        self.daily_pnl: float=0
        self.start_portfolio: float|None=None
    def check_inventory(self, inventory: float)->bool:
        return abs(inventory) <= self.limits.max_inventory
    def check_position(self, inventory: float, mid: float)->bool:
        return abs(inventory*mid) <= self.limits.max_position_usd
    def check_loss(self, portfolio: float)->bool:
        if self.start_portfolio is None:
            self.start_portfolio=portfolio
        self.daily_pnl=portfolio-self.start_portfolio
        return self.daily_pnl >= -self.limits.max_daily_loss
    def check_orders(self, open_orders: int)->bool:
        return open_orders <= self.limits.max_open_orders
    def validate(self, inventory: float, mid: float, portfolio: float, open_orders: int)->tuple[bool,str]:
        if not self.check_inventory(inventory):
            return False,"INVENTORY_LIMIT"
        if not self.check_position(inventory, mid):
            return False,"POSITION_LIMIT"
        if not self.check_loss(portfolio):
            return False,"LOSS_LIMIT"
        if not self.check_orders(open_orders):
            return False,"ORDER_LIMIT"
        return True,"OK"
