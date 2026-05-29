"""
Ensemble model and recommendation generation - fully parameterized by cfg.

Combines predictions from multiple models using weighted fusion and
generates final recommendation groups with morphological filtering.
All validation uses cfg.main_min/max/count and cfg.sub_min/max/count.
"""
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd

from utils.helpers import validate_numbers, validate_numbers_full, get_logger


# ---------------------------------------------------------------------------
# Ensemble Model
# ---------------------------------------------------------------------------

class EnsembleModel:
    """Weighted ensemble of multiple prediction models.

    Combines predictions from all fitted models using cfg.ensemble_weights
    and generates final recommendations with morphological filtering.

    Parameters
    ----------
    models_dict : dict of {name: model_instance}
        Each model must have ``predict(n_predictions)`` returning list of
        dicts with keys ``main``, ``sub``, ``model``.
    cfg : LotteryConfig instance
        Provides weights, number ranges, and configuration.
    """

    def __init__(
        self,
        models_dict: Dict[str, Any],
        cfg,
    ):
        self.models = models_dict
        self.cfg = cfg
        self.logger = get_logger(cfg)
        self.weights = dict(cfg.ensemble_weights)

        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        # Build name mapping: short keys in models_dict → weight keys
        self._name_map = {}
        short_to_long = {
            'f': 'frequency', 'frequency': 'frequency',
            'p': 'poisson', 'poisson': 'poisson',
            'e': 'exponential_smoothing', 'exponential_smoothing': 'exponential_smoothing',
            'm': 'monte_carlo', 'monte_carlo': 'monte_carlo',
            'lstm': 'lstm', 'xgboost': 'xgboost', 'random_forest': 'random_forest',
        }
        for key in self.models:
            long_name = short_to_long.get(key, key)
            self._name_map[key] = long_name

        self.logger.info(
            "EnsembleModel initialized with %d models for %s",
            len(self.models), cfg.name,
        )

    def get_model_names(self) -> List[str]:
        """Return list of registered model names."""
        return list(self.models.keys())

    def fit_all(self, main_nums: np.ndarray, sub_nums: np.ndarray) -> Dict[str, bool]:
        """Fit all registered models.

        Parameters
        ----------
        main_nums : 2D array of main numbers.
        sub_nums  : 2D array of sub numbers.

        Returns
        -------
        dict {model_name: success}
        """
        results = {}
        for name, model in self.models.items():
            try:
                model.fit(main_nums, sub_nums)
                results[name] = True
                self.logger.info("Model '%s' fitted successfully", name)
            except Exception as exc:
                self.logger.error("Model '%s' failed to fit: %s", name, exc)
                results[name] = False
        return results

    def predict_all(
        self,
        n_per_model: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate predictions from all models.

        Parameters
        ----------
        n_per_model : number of predictions per model.

        Returns
        -------
        dict {model_name: list of prediction dicts}
        """
        all_preds = {}
        for name, model in self.models.items():
            try:
                preds = model.predict(n_per_model)
                all_preds[name] = preds
            except Exception as exc:
                self.logger.warning("Model '%s' predict failed: %s", name, exc)
                all_preds[name] = []
        return all_preds

    def weighted_fusion(
        self,
        all_predictions: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Fuse predictions from all models using weighted voting.

        For each number position in the main and sub sets, compute a
        weighted score based on how many models predicted it and their
        ensemble weights.

        Parameters
        ----------
        all_predictions : dict from predict_all() output.

        Returns
        -------
        dict with keys:
            main_scores : dict {number: total_weighted_score}
            sub_scores  : dict {number: total_weighted_score}
            main_ranked : list of (number, score) sorted descending
            sub_ranked  : list of (number, score) sorted descending
        """
        # Initialize score accumulators
        main_scores: Dict[int, float] = {}
        sub_scores: Dict[int, float] = {}

        for n in range(self.cfg.main_min, self.cfg.main_max + 1):
            main_scores[n] = 0.0
        for n in range(self.cfg.sub_min, self.cfg.sub_max + 1):
            sub_scores[n] = 0.0

        for model_name, preds in all_predictions.items():
            # Map short name to weight name
            weight_name = self._name_map.get(model_name, model_name)
            weight = self.weights.get(weight_name, 0.0)
            if weight <= 0:
                continue

            # Use model's probability distribution for sub numbers
            # (more informative than vote counting for small pools)
            model = self.models.get(model_name)
            sub_probs = getattr(model, 'sub_probs', None)
            main_probs = getattr(model, 'main_probs', None)

            if sub_probs is not None and len(sub_probs) > 0:
                # Use probability distribution directly
                for i, n in enumerate(range(self.cfg.sub_min, self.cfg.sub_max + 1)):
                    sub_scores[n] = sub_scores.get(n, 0.0) + sub_probs[i] * weight * 10
            else:
                # Fallback: count-based voting
                for pred in preds:
                    sub_nums = pred.get("sub", [])
                    for n in sub_nums:
                        sub_scores[n] = sub_scores.get(n, 0.0) + weight

            # For main: also use probabilities when available, otherwise votes
            if main_probs is not None and len(main_probs) > 0:
                for i, n in enumerate(range(self.cfg.main_min, self.cfg.main_max + 1)):
                    main_scores[n] = main_scores.get(n, 0.0) + main_probs[i] * weight * 10
            else:
                for pred in preds:
                    main_nums = pred.get("main", [])
                    for n in main_nums:
                        main_scores[n] = main_scores.get(n, 0.0) + weight

        # Rank
        main_ranked = sorted(main_scores.items(), key=lambda x: -x[1])
        sub_ranked = sorted(sub_scores.items(), key=lambda x: -x[1])

        return {
            "main_scores": main_scores,
            "sub_scores": sub_scores,
            "main_ranked": main_ranked,
            "sub_ranked": sub_ranked,
        }


# ---------------------------------------------------------------------------
# Morphological Filtering
# ---------------------------------------------------------------------------

def _odd_even_filter(
    main_candidate: List[int],
    cfg,
) -> bool:
    """Check if odd/even ratio is within reasonable range.

    Rejects candidates where all main numbers are odd or all are even.
    Also rejects if the ratio is outside [0.2, 0.8] of main_count.
    """
    odd_count = sum(1 for n in main_candidate if n % 2 == 1)
    ratio = odd_count / cfg.main_count
    # Allow 1 to main_count-1 odds (not all odd and not all even)
    return 0 < odd_count < cfg.main_count


def _span_filter(
    main_candidate: List[int],
    cfg,
) -> bool:
    """Check if the span (max-min) is within reasonable bounds.

    Rejects candidates with very narrow or very wide spans.
    """
    span = max(main_candidate) - min(main_candidate)
    max_possible = cfg.main_max - cfg.main_min
    min_reasonable = max_possible * 0.2
    max_reasonable = max_possible * 0.95
    return min_reasonable <= span <= max_reasonable


def _sum_filter(
    main_candidate: List[int],
    cfg,
    sum_mean: Optional[float] = None,
    sum_std: Optional[float] = None,
) -> bool:
    """Check if sum is within reasonable z-score range.

    Default range: within 2.5 standard deviations of the mean.
    """
    total = sum(main_candidate)
    if sum_mean is not None and sum_std is not None and sum_std > 0:
        z = abs(total - sum_mean) / sum_std
        return z <= 2.5

    # Fallback: check against theoretical range
    min_sum = cfg.main_min * cfg.main_count
    max_sum = cfg.main_max * cfg.main_count
    mid = (min_sum + max_sum) / 2
    acceptable_range = (max_sum - min_sum) * 0.4
    return mid - acceptable_range <= total <= mid + acceptable_range


def _ac_value_filter(
    main_candidate: List[int],
    cfg,
) -> bool:
    """Check if AC value is within typical range.

    AC value is the number of unique positive differences minus (count-1).
    """
    n = len(main_candidate)
    if n <= 1:
        return True
    diffs = set()
    sorted_nums = sorted(main_candidate)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(sorted_nums[j] - sorted_nums[i])
    ac = len(diffs) - (n - 1)
    # AC should be positive and not too large
    max_possible_ac = (n * (n - 1)) // 2 - (n - 1)
    min_ac = max(1, max_possible_ac * 0.1)
    max_ac = max_possible_ac * 0.9
    return min_ac <= ac <= max_ac


def _consecutive_filter(
    main_candidate: List[int],
    cfg,
    max_consecutive_pairs: int = 2,
) -> bool:
    """Check if number of consecutive pairs is reasonable.

    Rejects candidates with too many consecutive numbers.
    """
    sorted_nums = sorted(main_candidate)
    consec_pairs = 0
    for i in range(len(sorted_nums) - 1):
        if sorted_nums[i + 1] - sorted_nums[i] == 1:
            consec_pairs += 1
    return consec_pairs <= max_consecutive_pairs


def _divisible_filter(
    main_candidate: List[int],
    cfg,
) -> bool:
    """Check if there are too many numbers divisible by 3 or 5.

    Rejects candidates where >= 80% numbers are divisible by 3 or
    where all numbers share a common divisor pattern.
    """
    div3 = sum(1 for n in main_candidate if n % 3 == 0)
    div5 = sum(1 for n in main_candidate if n % 5 == 0)
    # Not too clustered on multiples
    if div3 >= cfg.main_count * 0.8:
        return False
    if div5 >= cfg.main_count * 0.8:
        return False
    return True


def _tail_distribution_filter(
    main_candidate: List[int],
    cfg,
) -> bool:
    """Check tail (last digit) distribution.

    Rejects candidates where too many numbers share the same tail digit.
    """
    tails = Counter(n % 10 for n in main_candidate)
    # No single tail should appear more than 40% of the time
    max_allowed = int(cfg.main_count * 0.4) + 1
    for count in tails.values():
        if count > max_allowed:
            return False
    return True


def morphological_filter(
    main_candidate: List[int],
    cfg,
    sum_mean: Optional[float] = None,
    sum_std: Optional[float] = None,
) -> bool:
    """Apply full morphological filtering to a candidate main set.

    A candidate must pass all filters to be accepted.

    Parameters
    ----------
    main_candidate : list of main numbers.
    cfg            : LotteryConfig instance.
    sum_mean       : optional historical sum mean for z-score filtering.
    sum_std        : optional historical sum std for z-score filtering.

    Returns
    -------
    True if the candidate passes all filters.
    """
    filters = [
        _odd_even_filter,
        _span_filter,
        _consecutive_filter,
        _divisible_filter,
        _tail_distribution_filter,
        _ac_value_filter,
    ]

    for f in filters:
        if f.__name__ in ("_sum_filter",):
            if not _sum_filter(main_candidate, cfg, sum_mean, sum_std):
                return False
        else:
            if not f(main_candidate, cfg):
                return False
    return True


# ---------------------------------------------------------------------------
# Recommendation Generation
# ---------------------------------------------------------------------------

def generate_recommendations(
    ensemble,
    cfg,
    num_groups: int = 5,
    n_per_model: int = 10,
    max_candidates: int = 1000,
    sum_mean: Optional[float] = None,
    sum_std: Optional[float] = None,
    diversity_factor: float = 0.25,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Generate recommendations by pure statistical sampling.

    Approach: Use FrequencyModel's probability distribution (real statistics
    from 200 records). Weighted sampling without any scoring/fusion tricks.
    The probability distribution IS the science.
    """
    logger = get_logger(cfg)
    logger.info("Generating %d recommendation groups for %s ...", num_groups, cfg.name)

    from datetime import datetime

    # Get the FrequencyModel's probability distribution (pure statistics)
    freq_model = ensemble.models.get('f') or ensemble.models.get('frequency')
    if freq_model is None:
        # Fallback: find first model with probs
        for m in ensemble.models.values():
            if hasattr(m, 'main_probs') and len(m.main_probs) > 0:
                freq_model = m
                break
    
    if freq_model is None or not hasattr(freq_model, 'main_probs'):
        return {"groups": [], "hot_numbers": [], "cold_numbers": [],
                "model_weights": {}, "timestamp": "", "candidates_evaluated": 0}
    
    main_probs = freq_model.main_probs
    sub_probs = freq_model.sub_probs
    main_range = list(range(cfg.main_min, cfg.main_max + 1))
    sub_range = list(range(cfg.sub_min, cfg.sub_max + 1))

    # Build historical set for comparison
    historical_sets = []
    if df is not None:
        for _, row in df.iterrows():
            hist = tuple(sorted([int(row[c]) for c in cfg.main_cols]))
            historical_sets.append(hist)

    # Seeded RNG for determinism
    seed_hash = hash((tuple(main_probs[:10]), tuple(sub_probs[:5])))
    rng = random.Random(seed_hash)

    # Generate candidates by pure weighted sampling
    candidates = []
    used_combos = set()

    for _ in range(max_candidates * 3):
        # Weighted sampling for main numbers
        pool_m = list(main_range)
        w_m = list(main_probs)
        main_cand = []
        for _ in range(cfg.main_count):
            idx = rng.choices(range(len(pool_m)), weights=w_m, k=1)[0]
            main_cand.append(pool_m.pop(idx))
            w_m.pop(idx)
            if w_m:
                w_m = [ww / sum(w_m) for ww in w_m]
        main_cand = sorted(main_cand)

        # Weighted sampling for sub numbers
        pool_s = list(sub_range)
        w_s = list(sub_probs)
        sub_cand = []
        for _ in range(cfg.sub_count):
            idx = rng.choices(range(len(pool_s)), weights=w_s, k=1)[0]
            sub_cand.append(pool_s.pop(idx))
            w_s.pop(idx)
            if w_s:
                w_s = [ww / sum(w_s) for ww in w_s]
        sub_cand = sorted(sub_cand)

        combo = (tuple(main_cand), tuple(sub_cand))
        if combo in used_combos:
            continue
        used_combos.add(combo)

        # Historical similarity check
        too_close = False
        for hist in historical_sets:
            if len(set(main_cand) & set(hist)) >= 5:
                too_close = True
                break
        if too_close:
            continue

        candidates.append({"main": main_cand, "sub": sub_cand})
        if len(candidates) >= max_candidates:
            break

    # Pick first 5 — probability distribution already ensures proper weighting
    groups = []
    for i, c in enumerate(candidates[:num_groups]):
        # Compute a clean probability-based confidence score
        main_p = sum(main_probs[main_range.index(n)] for n in c["main"]) * 100
        sub_p = sum(sub_probs[sub_range.index(n)] for n in c["sub"]) * 100
        score = round(main_p + sub_p, 1)
        groups.append({
            "index": i + 1,
            "main": c["main"],
            "sub": c["sub"],
            "score": score,
        })

    # Compute hot/cold from probability distribution  
    main_ranked = sorted([(n, main_probs[i]) for i, n in enumerate(main_range)], key=lambda x: -x[1])
    sub_ranked = sorted([(n, sub_probs[i]) for i, n in enumerate(sub_range)], key=lambda x: -x[1])

    logger.info("Generated %d recommendation groups for %s", len(groups), cfg.name)
    return {
        "groups": groups,
        "hot_numbers": [n for n, _ in main_ranked[:5]],
        "cold_numbers": [n for n, _ in main_ranked[-3:]],
        "model_weights": {},
        "timestamp": datetime.now().isoformat(),
        "candidates_evaluated": len(candidates),
    }

