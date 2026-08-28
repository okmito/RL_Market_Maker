from __future__ import annotations
from decimal import Decimal, ROUND_DOWN

class QuoteManager:
    def __init__(self, tick_size: Decimal, lot_size: Decimal):
        self.tick_size=tick_size
        self.lot_size=lot_size
    def round_price(self, price: Decimal)->Decimal:
        return (price/self.tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN)*self.tick_size
    def round_qty(self, qty: Decimal)->Decimal:
        return (qty/self.lot_size).quantize(Decimal("1"), rounding=ROUND_DOWN)*self.lot_size
    def generate(self, mid: Decimal, bid_offset_bps: float, ask_offset_bps: float, bid_qty: Decimal, ask_qty: Decimal)->tuple[Decimal,Decimal,Decimal,Decimal]:
        bid = self.round_price(mid*Decimal(str(1+bid_offset_bps/10000)))
        ask = self.round_price(mid*Decimal(str(1+ask_offset_bps/10000)))
        if bid>=ask:
            # Crossed or inverted: force valid spread around mid
            mid_r = self.round_price(mid)
            spread = max(self.tick_size*2, abs(bid-ask)+self.tick_size*2)
            bid = self.round_price(mid_r - spread/2)
            ask = self.round_price(mid_r + spread/2)
            if bid>=ask:
                ask = bid + self.tick_size
        return bid, ask, self.round_qty(bid_qty), self.round_qty(ask_qty)
