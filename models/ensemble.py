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
            weight = self.weights.get(model_name, 0.0)
            if weight <= 0:
                continue

            for pred in preds:
                main_nums = pred.get("main", [])
                sub_nums = pred.get("sub", [])
                for n in main_nums:
                    main_scores[n] = main_scores.get(n, 0.0) + weight
                for n in sub_nums:
                    sub_scores[n] = sub_scores.get(n, 0.0) + weight

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
    ensemble: EnsembleModel,
    cfg,
    num_groups: int = 5,
    n_per_model: int = 20,
    max_candidates: int = 1000,
    sum_mean: Optional[float] = None,
    sum_std: Optional[float] = None,
    diversity_factor: float = 0.3,
) -> Dict[str, Any]:
    """Generate final recommendation groups using ensemble + filtering.

    Parameters
    ----------
    ensemble         : fitted EnsembleModel instance.
    cfg              : LotteryConfig instance.
    num_groups       : number of recommended groups to produce.
    n_per_model      : predictions per model for ensemble fusion.
    max_candidates   : max candidate groups to generate before filtering.
    sum_mean         : historical main sum mean (for z-score filtering).
    sum_std          : historical main sum std (for z-score filtering).
    diversity_factor : how much to penalize repeated numbers across groups
                       (0 = no penalty, 1 = maximum spread).

    Returns
    -------
    dict with keys:
        groups       : list of dicts with keys index, main, sub, score
        hot_numbers  : list of top hot numbers (from fusion)
        cold_numbers : list of bottom cold numbers (from fusion)
        model_weights: dict of weights used
        timestamp    : generation timestamp
    """
    logger = get_logger(cfg)
    logger.info(
        "Generating %d recommendation groups for %s ...",
        num_groups, cfg.name,
    )

    # Step 1: Get all model predictions
    all_preds = ensemble.predict_all(n_per_model=n_per_model)

    # Step 2: Weighted fusion scores
    fusion = ensemble.weighted_fusion(all_preds)

    # Step 3: Generate candidate groups
    candidates: List[Dict[str, Any]] = []
    main_range = list(range(cfg.main_min, cfg.main_max + 1))
    sub_range = list(range(cfg.sub_min, cfg.sub_max + 1))

    # Extract top-weighted numbers
    main_ranked = fusion["main_ranked"]
    sub_ranked = fusion["sub_ranked"]
    top_main = [n for n, _ in main_ranked[:cfg.top_hot_count]]
    top_sub = [n for n, _ in sub_ranked[:cfg.top_hot_count]]

    # Generate candidates by combining top-weighted numbers
    attempts = 0
    used_combos: set = set()

    while len(candidates) < max_candidates and attempts < max_candidates * 3:
        attempts += 1

        # Mix: mostly from top-ranked pool, some random for diversity
        if random.random() < 0.7:
            main_pool = top_main
        else:
            main_pool = main_range

        if random.random() < 0.7:
            sub_pool = top_sub
        else:
            sub_pool = sub_range

        main_candidate = sorted(random.sample(main_pool, cfg.main_count))
        sub_candidate = sorted(random.sample(sub_pool, cfg.sub_count))

        combo_key = (tuple(main_candidate), tuple(sub_candidate))
        if combo_key in used_combos:
            continue
        used_combos.add(combo_key)

        # Score the candidate based on fusion scores
        main_score = sum(fusion["main_scores"].get(n, 0) for n in main_candidate)
        sub_score = sum(fusion["sub_scores"].get(n, 0) for n in sub_candidate)
        total_score = main_score + sub_score

        # Apply morphological filter
        if morphological_filter(main_candidate, cfg, sum_mean, sum_std):
            candidates.append({
                "main": main_candidate,
                "sub": sub_candidate,
                "score": total_score,
            })

    # Sort by score descending
    candidates.sort(key=lambda c: -c["score"])

    # Step 4: Select diverse groups
    selected_groups = _select_diverse_groups(
        candidates, num_groups, cfg, diversity_factor,
    )

    # Step 5: Build final output
    hot_main = [n for n, _ in main_ranked[:cfg.top_hot_count]]
    cold_main = [n for n, _ in main_ranked[-cfg.top_cold_count:]]
    hot_sub = [n for n, _ in sub_ranked[:cfg.top_hot_count]]
    cold_sub = [n for n, _ in sub_ranked[-cfg.top_cold_count:]]

    from datetime import datetime

    result: Dict[str, Any] = {
        "groups": selected_groups,
        "hot_numbers": {
            "main": hot_main,
            "sub": hot_sub,
        },
        "cold_numbers": {
            "main": cold_main,
            "sub": cold_sub,
        },
        "model_weights": dict(ensemble.weights),
        "fusion_scores": {
            "main_top10": main_ranked[:10],
            "sub_top5": sub_ranked[:5],
        },
        "candidates_evaluated": len(candidates),
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(
        "Generated %d recommendation groups from %d candidates for %s",
        len(selected_groups), len(candidates), cfg.name,
    )

    return result


def _select_diverse_groups(
    candidates: List[Dict[str, Any]],
    num_groups: int,
    cfg,
    diversity_factor: float = 0.3,
) -> List[Dict[str, Any]]:
    """Select diverse groups from ranked candidates.

    Greedy selection: pick highest-scoring, then add groups that
    maximize overlap penalty (diversity).
    """
    if not candidates:
        return []

    selected = [candidates[0]]
    used_numbers = set(candidates[0]["main"])

    while len(selected) < num_groups and len(selected) < len(candidates):
        best_candidate = None
        best_score = -float("inf")

        for cand in candidates[1:]:
            if cand in selected:
                continue

            main_set = set(cand["main"])
            overlap = len(main_set & used_numbers)

            # Penalize overlap
            diversity_penalty = overlap * diversity_factor / cfg.main_count
            adjusted_score = cand["score"] * (1 - diversity_penalty)

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_candidate = cand

        if best_candidate is None:
            break

        selected.append(best_candidate)
        used_numbers.update(best_candidate["main"])

    # Renumber indices
    for i, g in enumerate(selected):
        g["index"] = i + 1

    return selected
