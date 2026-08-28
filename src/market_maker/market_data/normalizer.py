from __future__ import annotations

import time
from decimal import Decimal
from typing import Optional

import numpy as np

from market_maker.market_data.order_book import OrderBook


class FeatureNormalizer:
    """
    Online feature normalizer using running statistics.

    Uses Welford's algorithm for numerically stable mean/variance computation.
    """

    def __init__(self, num_features: int, window: int = 1000, eps: float = 1e-8):
        self.num_features = num_features
        self.window = window
        self.eps = eps

        self.count = 0
        self.mean = np.zeros(num_features, dtype=np.float64)
        self.m2 = np.zeros(num_features, dtype=np.float64)  # Sum of squared differences
        self.min_vals = np.full(num_features, np.inf, dtype=np.float64)
        self.max_vals = np.full(num_features, -np.inf, dtype=np.float64)

        # For sliding window
        self._buffer: list[np.ndarray] = []
        self._buffer_idx = 0

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new observation."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.num_features,):
            raise ValueError(f"Expected {self.num_features} features, got {x.shape}")

        # Update min/max
        self.min_vals = np.minimum(self.min_vals, x)
        self.max_vals = np.maximum(self.max_vals, x)

        # Welford's algorithm
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

        # Sliding window buffer
        if len(self._buffer) < self.window:
            self._buffer.append(x.copy())
        else:
            self._buffer[self._buffer_idx] = x.copy()
            self._buffer_idx = (self._buffer_idx + 1) % self.window

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize using running mean/std."""
        x = np.asarray(x, dtype=np.float64)
        if self.count < 2:
            return x - self.mean
        std = np.sqrt(self.m2 / (self.count - 1) + self.eps)
        return (x - self.mean) / std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize (inverse transform)."""
        if self.count < 2:
            return x + self.mean
        std = np.sqrt(self.m2 / (self.count - 1) + self.eps)
        return x * std + self.mean

    def get_stats(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "std": np.sqrt(self.m2 / max(1, self.count - 1)).tolist(),
            "min": self.min_vals.tolist(),
            "max": self.max_vals.tolist(),
        }

    def reset(self) -> None:
        self.count = 0
        self.mean.fill(0)
        self.m2.fill(0)
        self.min_vals.fill(np.inf)
        self.max_vals.fill(-np.inf)
        self._buffer.clear()
        self._buffer_idx = 0


class MinMaxNormalizer:
    """Min-max normalizer with configurable bounds."""

    def __init__(
        self,
        num_features: int,
        feature_min: Optional[np.ndarray] = None,
        feature_max: Optional[np.ndarray] = None,
        target_min: float = -1.0,
        target_max: float = 1.0,
        clip: bool = True,
    ):
        self.num_features = num_features
        self.target_min = target_min
        self.target_max = target_max
        self.clip = clip

        if feature_min is not None:
            self.feature_min = np.asarray(feature_min, dtype=np.float64)
        else:
            self.feature_min = np.full(num_features, -np.inf, dtype=np.float64)

        if feature_max is not None:
            self.feature_max = np.asarray(feature_max, dtype=np.float64)
        else:
            self.feature_max = np.full(num_features, np.inf, dtype=np.float64)

        self._fitted = feature_min is not None and feature_max is not None

    def fit(self, x: np.ndarray) -> None:
        """Fit min/max from data."""
        x = np.asarray(x, dtype=np.float64)
        self.feature_min = np.min(x, axis=0)
        self.feature_max = np.max(x, axis=0)
        self._fitted = True

    def partial_fit(self, x: np.ndarray) -> None:
        """Update min/max incrementally."""
        x = np.asarray(x, dtype=np.float64)
        if not self._fitted:
            self.feature_min = np.min(x, axis=0)
            self.feature_max = np.max(x, axis=0)
            self._fitted = True
        else:
            self.feature_min = np.minimum(self.feature_min, np.min(x, axis=0))
            self.feature_max = np.maximum(self.feature_max, np.max(x, axis=0))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not self._fitted:
            return x

        range_ = self.feature_max - self.feature_min
        range_ = np.where(range_ == 0, 1.0, range_)

        normalized = (x - self.feature_min) / range_
        normalized = normalized * (self.target_max - self.target_min) + self.target_min

        if self.clip:
            normalized = np.clip(normalized, self.target_min, self.target_max)

        return normalized

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return x

        range_ = self.feature_max - self.feature_min
        range_ = np.where(range_ == 0, 1.0, range_)

        x = (x - self.target_min) / (self.target_max - self.target_min)
        return x * range_ + self.feature_min


class ZScoreNormalizer:
    """Z-score normalizer with running statistics."""

    def __init__(self, num_features: int, eps: float = 1e-8, clip: float | None = 5.0):
        self.num_features = num_features
        self.eps = eps
        self.clip = clip

        self.mean = np.zeros(num_features, dtype=np.float64)
        self.std = np.ones(num_features, dtype=np.float64)
        self._fitted = False

    def fit(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0, ddof=1)
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        self._fitted = True

    def partial_fit(self, x: np.ndarray) -> None:
        if not self._fitted:
            self.fit(x)
        else:
            # Simple exponential moving average
            alpha = 0.01
            new_mean = np.mean(x, axis=0)
            new_std = np.std(x, axis=0, ddof=1)
            self.mean = (1 - alpha) * self.mean + alpha * new_mean
            self.std = (1 - alpha) * self.std + alpha * new_std
            self.std = np.where(self.std < self.eps, 1.0, self.std)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not self._fitted:
            return x

        normalized = (x - self.mean) / self.std
        if self.clip is not None:
            normalized = np.clip(normalized, -self.clip, self.clip)
        return normalized

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return x
        return x * self.std + self.mean


def create_normalizer(
    normalizer_type: str,
    num_features: int,
    **kwargs,
) -> FeatureNormalizer | MinMaxNormalizer | ZScoreNormalizer:
    """Factory function for normalizers."""
    if normalizer_type == "running":
        return FeatureNormalizer(num_features, **kwargs)
    elif normalizer_type == "minmax":
        return MinMaxNormalizer(num_features, **kwargs)
    elif normalizer_type == "zscore":
        return ZScoreNormalizer(num_features, **kwargs)
    else:
        raise ValueError(f"Unknown normalizer type: {normalizer_type}")