from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass
class FillConfig:
    model: Literal["conservative","queue_aware"]="queue_aware"
    queue_decay: float=0.05
    conservative_fill_prob: float=0.7
    queue_aware_base_prob: float=0.8

def should_fill(order_price: float, market_bid: float, market_ask: float, side: str, queue_position: float, config: FillConfig, rng) -> tuple[bool,bool]:
    """Return (should_fill, is_maker)."""
    if side=="BUY":
        if market_ask <= order_price:
            if config.model=="conservative":
                return (rng.random()<config.conservative_fill_prob, True)
            else:
                prob = config.queue_aware_base_prob * (1 - queue_position*0.6)
                return (rng.random()<prob, True)
        elif order_price >= (market_bid+market_ask)/2 + 5*0.01:
            return (rng.random()<0.9, False)
    else:
        if market_bid >= order_price:
            if config.model=="conservative":
                return (rng.random()<config.conservative_fill_prob, True)
            else:
                prob = config.queue_aware_base_prob * (1 - queue_position*0.6)
                return (rng.random()<prob, True)
        elif order_price <= (market_bid+market_ask)/2 - 5*0.01:
            return (rng.random()<0.9, False)
    return (False, True)
