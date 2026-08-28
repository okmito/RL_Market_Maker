from __future__ import annotations
import numpy as np

class MarketSimulator:
    """Synthetic LOB/mid price generator for training without live data."""
    def __init__(self, initial_price: float=50000.0, volatility: float=0.02, drift: float=0.0, mean_reversion: float=0.1, seed: int=42):
        self.initial_price=initial_price
        self.volatility=volatility
        self.drift=drift
        self.mean_reversion=mean_reversion
        self.rng=np.random.default_rng(seed)
        self.price=initial_price
        self.history=[initial_price]

    def reset(self):
        self.price=self.initial_price
        self.history=[self.initial_price]
        return self.price

    def step(self) -> float:
        dt=1.0
        prev=self.price
        # mean reversion term
        mr = -self.mean_reversion*(prev - self.initial_price)/self.initial_price if self.history else 0
        shock=self.rng.normal(0, self.volatility*np.sqrt(dt))
        if self.rng.random()<0.001:
            shock+= self.rng.choice([-1,1])*0.05
        new_price=prev*np.exp((self.drift+mr)*dt + shock)
        new_price=max(new_price, 1000)
        self.price=new_price
        self.history.append(new_price)
        return new_price

    def generate(self, n_steps: int) -> list[float]:
        self.reset()
        out=[self.price]
        for _ in range(n_steps-1):
            out.append(self.step())
        return out
