from decimal import Decimal
from market_maker.execution.inventory_manager import InventoryManager
from market_maker.execution.risk_manager import RiskManager, RiskLimits

def test_inventory():
    im=InventoryManager(max_inventory=1.0)
    assert im.ratio(Decimal("0.5"))==0.5
    assert im.is_at_limit(Decimal("0.96"))==True
    assert im.allowed_side(Decimal("0.9"), "BUY")==False

def test_risk():
    rm=RiskManager(RiskLimits(max_inventory=1.0, max_position_usd=50000))
    ok,_=rm.validate(inventory=0.1, mid=50000, portfolio=100000, open_orders=5)
    assert ok==True
    ok,reason=rm.validate(inventory=2.0, mid=50000, portfolio=100000, open_orders=5)
    assert ok==False
