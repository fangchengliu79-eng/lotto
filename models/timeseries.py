"""
Time series models for lottery prediction - fully parameterized by cfg.

Implements ExponentialSmoothingModel for trend-aware prediction.
Works for any lottery type (DLT, SSQ, etc.) via cfg parameterization.
"""
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Any

import numpy as np

from utils.helpers import validate_numbers, get_logger


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class BaseTimeSeriesModel(ABC):
    """Abstract base for all time series prediction models."""

    def __init__(self, cfg, name: str = "ts_base"):
        self.cfg = cfg
        self.name = name
        self.logger = get_logger(cfg)
        self._fitted = False

    @abstractmethod
    def fit(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict_main(self, n_predictions: int = 1) -> List[List[int]]:
        ...

    @abstractmethod
    def predict_sub(self, n_predictions: int = 1) -> List[List[int]]:
        ...

    def predict(self, n_predictions: int = 1) -> List[Dict[str, Any]]:
        mains = self.predict_main(n_predictions)
        subs = self.predict_sub(n_predictions)
        return [
            {"main": mains[i], "sub": subs[i], "model": self.name}
            for i in range(n_predictions)
        ]


# ---------------------------------------------------------------------------
# Exponential Smoothing Model
# ---------------------------------------------------------------------------

class ExponentialSmoothingModel(BaseTimeSeriesModel):
    """Exponential smoothing for number appearance probability.

    Treats each number's appearance history as a binary time series
    (1 = appeared in draw, 0 = did not appear) and applies exponential
    smoothing to compute a smoothed probability. Numbers with higher
    smoothed probability are more likely to be selected.

    Fully parameterized by cfg - uses cfg.main_min/max/count and
    cfg.sub_min/max/count.
    """

    def __init__(self, cfg, alpha: float = 0.3):
        """Initialize exponential smoothing model.

        Parameters
        ----------
        cfg   : LotteryConfig instance.
        alpha : smoothing factor (0 < alpha <= 1). Higher = more weight on recent.
        """
        super().__init__(cfg, name="exponential_smoothing")
        self.alpha = max(0.01, min(1.0, alpha))
        self.main_smoothed: Dict[int, float] = {}
        self.sub_smoothed: Dict[int, float] = {}
        self.main_trend: Dict[int, float] = {}
        self.sub_trend: Dict[int, float] = {}

    def fit(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        """Fit exponential smoothing on historical draw data.

        For each number, computes an exponentially smoothed probability
        by scanning draws from oldest to newest.
        """
        total_draws = len(main_nums)

        # Initialize all numbers at 0.5 (neutral prior)
        for n in range(self.cfg.main_min, self.cfg.main_max + 1):
            self.main_smoothed[n] = 0.5
            self.main_trend[n] = 0.0

        for n in range(self.cfg.sub_min, self.cfg.sub_max + 1):
            self.sub_smoothed[n] = 0.5
            self.sub_trend[n] = 0.0

        # Smooth forwards through time
        for idx in range(total_draws):
            main_draw_set = set(main_nums[idx])
            sub_draw_set = set(sub_nums[idx])

            for n in range(self.cfg.main_min, self.cfg.main_max + 1):
                appeared = 1.0 if n in main_draw_set else 0.0
                prev = self.main_smoothed[n]
                self.main_smoothed[n] = self.alpha * appeared + (1 - self.alpha) * prev
                self.main_trend[n] = self.main_smoothed[n] - prev

            for n in range(self.cfg.sub_min, self.cfg.sub_max + 1):
                appeared = 1.0 if n in sub_draw_set else 0.0
                prev = self.sub_smoothed[n]
                self.sub_smoothed[n] = self.alpha * appeared + (1 - self.alpha) * prev
                self.sub_trend[n] = self.sub_smoothed[n] - prev

        self._fitted = True
        self.logger.info(
            "ExponentialSmoothingModel (alpha=%.2f) fitted on %d draws for %s",
            self.alpha, total_draws, self.cfg.name,
        )

    def predict_main(self, n_predictions: int = 1) -> List[List[int]]:
        """Select main numbers with highest smoothed probability + trend."""
        if not self._fitted:
            raise RuntimeError("ExponentialSmoothingModel not fitted. Call fit() first.")

        predictions = []
        for _ in range(n_predictions):
            # Score = current smoothed value + trend bonus
            scores = {}
            for n in range(self.cfg.main_min, self.cfg.main_max + 1):
                base = self.main_smoothed.get(n, 0.5)
                trend = self.main_trend.get(n, 0.0)
                # Add small noise to avoid ties
                scores[n] = base + trend * 0.5 + np.random.uniform(0, 0.001)

            ranked = sorted(scores.items(), key=lambda x: -x[1])
            pred = sorted([n for n, _ in ranked[:self.cfg.main_count]])
            predictions.append(pred)

        return predictions

    def predict_sub(self, n_predictions: int = 1) -> List[List[int]]:
        """Select sub numbers with highest smoothed probability + trend."""
        if not self._fitted:
            raise RuntimeError("ExponentialSmoothingModel not fitted. Call fit() first.")

        predictions = []
        for _ in range(n_predictions):
            scores = {}
            for n in range(self.cfg.sub_min, self.cfg.sub_max + 1):
                base = self.sub_smoothed.get(n, 0.5)
                trend = self.sub_trend.get(n, 0.0)
                scores[n] = base + trend * 0.5 + np.random.uniform(0, 0.001)

            ranked = sorted(scores.items(), key=lambda x: -x[1])
            pred = sorted([n for n, _ in ranked[:self.cfg.sub_count]])
            predictions.append(pred)

        return predictions
