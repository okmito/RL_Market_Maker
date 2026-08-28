from __future__ import annotations
from decimal import Decimal

class InventoryManager:
    def __init__(self, max_inventory: float=1.0, target: float=0.0):
        self.max_inventory=Decimal(str(max_inventory))
        self.target=Decimal(str(target))
    def ratio(self, inventory: Decimal)->float:
        if self.max_inventory==0:
            return 0.0
        return float(inventory/self.max_inventory)
    def is_at_limit(self, inventory: Decimal)->bool:
        return abs(inventory) >= self.max_inventory*Decimal("0.95")
    def skew(self, inventory: Decimal)->float:
        r=self.ratio(inventory)
        return max(-1,min(1,-r))
    def allowed_side(self, inventory: Decimal, side: str)->bool:
        r=self.ratio(inventory)
        if r>0.8 and side=="BUY":
            return False
        if r<-0.8 and side=="SELL":
            return False
        return True
