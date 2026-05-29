"""
Statistical models for lottery prediction - fully parameterized by cfg.

Implements FrequencyModel, PoissonModel, and MonteCarloModel.
Works for any lottery type (DLT, SSQ, etc.) via cfg parameterization.
"""
import math
import random
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from utils.helpers import validate_numbers, get_logger


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class BaseStatisticalModel(ABC):
    """Abstract base for all statistical prediction models."""

    def __init__(self, cfg, name: str = "base"):
        self.cfg = cfg
        self.name = name
        self.logger = get_logger(cfg)
        self._fitted = False

    @abstractmethod
    def fit(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        """Fit the model on historical draw data."""
        ...

    @abstractmethod
    def predict_main(self, n_predictions: int = 1) -> List[List[int]]:
        """Generate main number predictions."""
        ...

    @abstractmethod
    def predict_sub(self, n_predictions: int = 1) -> List[List[int]]:
        """Generate sub number predictions."""
        ...

    def predict(self, n_predictions: int = 1) -> List[Dict[str, Any]]:
        """Generate full predictions (main + sub)."""
        mains = self.predict_main(n_predictions)
        subs = self.predict_sub(n_predictions)
        n = min(len(mains), len(subs))
        return [
            {"main": mains[i], "sub": subs[i], "model": self.name}
            for i in range(n)
        ]

    def _validate_prediction(self, main: List[int], sub: List[int]) -> bool:
        """Validate a single prediction against cfg rules."""
        ok_m, _ = validate_numbers(main, self.cfg, field="main")
        ok_s, _ = validate_numbers(sub, self.cfg, field="sub")
        return ok_m and ok_s


# ---------------------------------------------------------------------------
# Frequency Model
# ---------------------------------------------------------------------------

class FrequencyModel(BaseStatisticalModel):
    """Predict numbers based on historical appearance frequency.

    Assigns selection probability proportional to frequency.
    Uses cfg.main_min/max/range for main numbers and cfg.sub_min/max for sub.
    """

    def __init__(self, cfg):
        super().__init__(cfg, name="frequency")
        self.main_freq: Dict[int, int] = {}
        self.sub_freq: Dict[int, int] = {}
        self.main_probs: np.ndarray = np.array([])
        self.sub_probs: np.ndarray = np.array([])
        self.total_draws: int = 0

    def fit(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        """Compute frequency distributions from historical data."""
        self.total_draws = len(main_nums)

        # Main frequencies
        main_flat = main_nums.flatten()
        main_counter = Counter(main_flat.tolist())
        self.main_freq = {
            n: main_counter.get(n, 0)
            for n in range(self.cfg.main_min, self.cfg.main_max + 1)
        }

        # Sub frequencies
        sub_flat = sub_nums.flatten()
        sub_counter = Counter(sub_flat.tolist())
        self.sub_freq = {
            n: sub_counter.get(n, 0)
            for n in range(self.cfg.sub_min, self.cfg.sub_max + 1)
        }

        # Probability vectors with Laplace smoothing
        main_counts = np.array([
            self.main_freq[n] + 1 for n in range(self.cfg.main_min, self.cfg.main_max + 1)
        ], dtype=float)
        self.main_probs = main_counts / main_counts.sum()

        sub_counts = np.array([
            self.sub_freq[n] + 1 for n in range(self.cfg.sub_min, self.cfg.sub_max + 1)
        ], dtype=float)
        self.sub_probs = sub_counts / sub_counts.sum()

        self._fitted = True
        self.logger.info(
            "FrequencyModel fitted on %d draws for %s",
            self.total_draws, self.cfg.name,
        )

    def predict_main(self, n_predictions: int = 1) -> List[List[int]]:
        """Sample main numbers weighted by frequency.

        Uses weighted sampling without replacement per prediction.
        """
        if not self._fitted:
            raise RuntimeError("FrequencyModel not fitted. Call fit() first.")

        numbers = list(range(self.cfg.main_min, self.cfg.main_max + 1))
        predictions = []
        for _ in range(n_predictions):
            pred = list(np.random.choice(
                numbers,
                size=self.cfg.main_count,
                replace=False,
                p=self.main_probs,
            ))
            predictions.append(sorted(pred))
        return predictions

    def predict_sub(self, n_predictions: int = 1) -> List[List[int]]:
        """Sample sub numbers weighted by frequency."""
        if not self._fitted:
            raise RuntimeError("FrequencyModel not fitted. Call fit() first.")

        numbers = list(range(self.cfg.sub_min, self.cfg.sub_max + 1))
        predictions = []
        for _ in range(n_predictions):
            pred = list(np.random.choice(
                numbers,
                size=self.cfg.sub_count,
                replace=False,
                p=self.sub_probs,
            ))
            predictions.append(sorted(pred))
        return predictions


# ---------------------------------------------------------------------------
# Poisson Model
# ---------------------------------------------------------------------------

class PoissonModel(BaseStatisticalModel):
    """Model number appearances as Poisson processes.

    For each number, estimate lambda = (count / total_draws) * interval,
    then sample next appearance gaps and rank by expected appearance time.
    Uses cfg.main_min/max/range for main numbers.
    """

    def __init__(self, cfg):
        super().__init__(cfg, name="poisson")
        self.main_lambdas: Dict[int, float] = {}
        self.sub_lambdas: Dict[int, float] = {}
        self.main_last_seen: Dict[int, int] = {}
        self.sub_last_seen: Dict[int, int] = {}
        self.main_gaps: Dict[int, List[int]] = {}
        self.sub_gaps: Dict[int, List[int]] = {}
        self.total_draws: int = 0

    def fit(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        """Estimate Poisson parameters and gap distributions."""
        self.total_draws = len(main_nums)

        # Track last-seen draw index for each number
        main_range = list(range(self.cfg.main_min, self.cfg.main_max + 1))
        sub_range = list(range(self.cfg.sub_min, self.cfg.sub_max + 1))

        # Initialize
        for n in main_range:
            self.main_last_seen[n] = -1
            self.main_gaps[n] = []
            self.main_lambdas[n] = 0.0

        for n in sub_range:
            self.sub_last_seen[n] = -1
            self.sub_gaps[n] = []
            self.sub_lambdas[n] = 0.0

        # Track gaps and last seen positions
        for idx in range(self.total_draws):
            for n in main_nums[idx]:
                if self.main_last_seen[n] >= 0:
                    gap = idx - self.main_last_seen[n]
                    self.main_gaps[n].append(gap)
                self.main_last_seen[n] = idx

            for n in sub_nums[idx]:
                if self.sub_last_seen[n] >= 0:
                    gap = idx - self.sub_last_seen[n]
                    self.sub_gaps[n].append(gap)
                self.sub_last_seen[n] = idx

        # Estimate lambda (average gap)
        for n in main_range:
            gaps = self.main_gaps[n]
            self.main_lambdas[n] = np.mean(gaps) if gaps else self.total_draws

        for n in sub_range:
            gaps = self.sub_gaps[n]
            self.sub_lambdas[n] = np.mean(gaps) if gaps else self.total_draws

        self._fitted = True
        self.logger.info(
            "PoissonModel fitted on %d draws for %s",
            self.total_draws, self.cfg.name,
        )

    def predict_main(self, n_predictions: int = 1) -> List[List[int]]:
        """Pick numbers with highest Poisson probability of appearing soon.

        Uses the probability that the gap since last seen exceeds lambda
        as a proxy for due-ness. Numbers with larger gaps relative to
        their lambda are more likely to be predicted.
        """
        if not self._fitted:
            raise RuntimeError("PoissonModel not fitted. Call fit() first.")

        predictions = []
        for _ in range(n_predictions):
            scores = {}
            for n in range(self.cfg.main_min, self.cfg.main_max + 1):
                lam = self.main_lambdas.get(n, 1.0)
                if lam <= 0:
                    lam = 1.0
                # Number of draws since last seen
                last = self.main_last_seen.get(n, -1)
                if last < 0:
                    # Never seen: high priority
                    scores[n] = 100.0
                else:
                    gap = self.total_draws - 1 - last
                    # Poisson survival probability P(gap > current_gap | lambda)
                    # Higher survival = more overdue
                    surv_prob = math.exp(-gap / lam)
                    scores[n] = 1.0 - surv_prob  # probability it appears now

            # Pick top main_count numbers
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            pred = sorted([n for n, _ in ranked[:self.cfg.main_count]])
            predictions.append(pred)

        return predictions

    def predict_sub(self, n_predictions: int = 1) -> List[List[int]]:
        """Pick sub numbers with highest Poisson probability."""
        if not self._fitted:
            raise RuntimeError("PoissonModel not fitted. Call fit() first.")

        predictions = []
        for _ in range(n_predictions):
            scores = {}
            for n in range(self.cfg.sub_min, self.cfg.sub_max + 1):
                lam = self.sub_lambdas.get(n, 1.0)
                if lam <= 0:
                    lam = 1.0
                last = self.sub_last_seen.get(n, -1)
                if last < 0:
                    scores[n] = 100.0
                else:
                    gap = self.total_draws - 1 - last
                    surv_prob = math.exp(-gap / lam)
                    scores[n] = 1.0 - surv_prob

            ranked = sorted(scores.items(), key=lambda x: -x[1])
            pred = sorted([n for n, _ in ranked[:self.cfg.sub_count]])
            predictions.append(pred)

        return predictions


# ---------------------------------------------------------------------------
# Monte Carlo Model
# ---------------------------------------------------------------------------

class MonteCarloModel(BaseStatisticalModel):
    """Monte Carlo simulation for lottery number generation.

    Simulates many random draws and selects those matching historical
    statistical patterns (frequency distribution, odd/even ratio, sum range).
    Uses cfg.main_min/max/count and cfg.sub_min/max/count.
    """

    def __init__(self, cfg):
        super().__init__(cfg, name="monte_carlo")
        self.n_simulations: int = 50000
        self.main_freq: Dict[int, int] = {}
        self.sub_freq: Dict[int, int] = {}
        self.main_sum_mean: float = 0.0
        self.main_sum_std: float = 0.0
        self.sub_sum_mean: float = 0.0
        self.sub_sum_std: float = 0.0
        self.main_odd_pct: float = 0.5
        self.sub_odd_pct: float = 0.5

    def fit(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> None:
        """Learn statistical parameters from historical data."""
        # Frequencies
        main_flat = main_nums.flatten()
        main_counter = Counter(main_flat.tolist())
        self.main_freq = {
            n: main_counter.get(n, 0)
            for n in range(self.cfg.main_min, self.cfg.main_max + 1)
        }

        sub_flat = sub_nums.flatten()
        sub_counter = Counter(sub_flat.tolist())
        self.sub_freq = {
            n: sub_counter.get(n, 0)
            for n in range(self.cfg.sub_min, self.cfg.sub_max + 1)
        }

        # Sum statistics
        main_sums = np.sum(main_nums, axis=1)
        sub_sums = np.sum(sub_nums, axis=1)
        self.main_sum_mean = float(np.mean(main_sums))
        self.main_sum_std = float(np.std(main_sums))
        self.sub_sum_mean = float(np.mean(sub_sums))
        self.sub_sum_std = float(np.std(sub_sums))

        # Odd ratio
        main_odds = np.sum(main_nums % 2 == 1, axis=1)
        self.main_odd_pct = float(np.mean(main_odds) / self.cfg.main_count)

        sub_odds = np.sum(sub_nums % 2 == 1, axis=1)
        self.sub_odd_pct = float(np.mean(sub_odds) / self.cfg.sub_count) if self.cfg.sub_count > 0 else 0.5

        self._fitted = True
        self.logger.info(
            "MonteCarloModel fitted on %d draws for %s",
            len(main_nums), self.cfg.name,
        )

    def predict_main(self, n_predictions: int = 1) -> List[List[int]]:
        """Run Monte Carlo simulation and select best matching main sets."""
        if not self._fitted:
            raise RuntimeError("MonteCarloModel not fitted. Call fit() first.")

        candidates: List[Tuple[List[int], float]] = []
        main_range = list(range(self.cfg.main_min, self.cfg.main_max + 1))

        for _ in range(self.n_simulations):
            cand = sorted(random.sample(main_range, self.cfg.main_count))
            score = self._score_main_set(cand)
            candidates.append((cand, score))

        # Sort by score (lower is better - more "typical")
        candidates.sort(key=lambda x: x[1])

        # Return unique top predictions
        seen = set()
        predictions = []
        for cand, _ in candidates:
            key = tuple(cand)
            if key not in seen:
                seen.add(key)
                predictions.append(cand)
            if len(predictions) >= n_predictions:
                break

        return predictions

    def predict_sub(self, n_predictions: int = 1) -> List[List[int]]:
        """Run Monte Carlo simulation and select best matching sub sets."""
        if not self._fitted:
            raise RuntimeError("MonteCarloModel not fitted. Call fit() first.")

        candidates: List[Tuple[List[int], float]] = []
        sub_range = list(range(self.cfg.sub_min, self.cfg.sub_max + 1))

        for _ in range(self.n_simulations):
            cand = sorted(random.sample(sub_range, self.cfg.sub_count))
            score = self._score_sub_set(cand)
            candidates.append((cand, score))

        candidates.sort(key=lambda x: x[1])
        seen = set()
        predictions = []
        for cand, _ in candidates:
            key = tuple(cand)
            if key not in seen:
                seen.add(key)
                predictions.append(cand)
            if len(predictions) >= n_predictions:
                break

        return predictions

    def _score_main_set(self, cand: List[int]) -> float:
        """Score a candidate main set. Lower = more historically typical.

        Considers: sum proximity, odd/even ratio match, frequency score.
        """
        score = 0.0

        # Sum proximity (z-score absolute value)
        s = sum(cand)
        if self.main_sum_std > 0:
            z = abs(s - self.main_sum_mean) / self.main_sum_std
            score += z

        # Odd/even ratio
        odd_count = sum(1 for n in cand if n % 2 == 1)
        expected_odd = self.main_odd_pct * self.cfg.main_count
        score += abs(odd_count - expected_odd) * 0.5

        # Frequency score (prefer historically common numbers)
        total_freq = sum(self.main_freq.get(n, 0) for n in cand)
        avg_freq = total_freq / len(cand)
        score += (1.0 / (avg_freq + 1)) * 5  # encourage higher freq numbers

        return score

    def _score_sub_set(self, cand: List[int]) -> float:
        """Score a candidate sub set. Lower = more historically typical."""
        score = 0.0

        s = sum(cand)
        if self.sub_sum_std > 0:
            z = abs(s - self.sub_sum_mean) / self.sub_sum_std
            score += z

        odd_count = sum(1 for n in cand if n % 2 == 1)
        expected_odd = self.sub_odd_pct * self.cfg.sub_count
        score += abs(odd_count - expected_odd) * 0.5

        total_freq = sum(self.sub_freq.get(n, 0) for n in cand)
        avg_freq = total_freq / len(cand)
        score += (1.0 / (avg_freq + 1)) * 5

        return score
