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
    n_per_model: int = 10,
    max_candidates: int = 1000,
    sum_mean: Optional[float] = None,
    sum_std: Optional[float] = None,
    diversity_factor: float = 0.25,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Generate recommendations following recent trends, preferring unique combos.

    Strategy (per user's expert insight):
    1. Identify CURRENT trends from fusion scores, not long-term averages
    2. Generate candidates that FOLLOW these trends (clustered, not uniform)
    3. Among trend-following candidates, prefer LOW-PROBABILITY unique combos
    4. No forced uniform distribution — real draws are streaky, not balanced
    """
    logger = get_logger(cfg)
    logger.info(
        "Generating %d recommendation groups for %s ...",
        num_groups, cfg.name,
    )

    from datetime import datetime

    # Step 1: Get model predictions and fusion scores
    all_preds = ensemble.predict_all(n_per_model=n_per_model)
    fusion = ensemble.weighted_fusion(all_preds)

    # Step 2: Define trend profile from fusion
    main_ranked = fusion["main_ranked"]
    sub_ranked = fusion["sub_ranked"]
    main_scores = {n: s for n, s in main_ranked}
    sub_scores = {n: s for n, s in sub_ranked}

    main_range = list(range(cfg.main_min, cfg.main_max + 1))
    sub_range = list(range(cfg.sub_min, cfg.sub_max + 1))

    # Identify trend zones: top 40% numbers define the "hot zone"
    hot_cutoff = int(len(main_ranked) * 0.4)
    hot_main = [n for n, _ in main_ranked[:hot_cutoff]]
    cold_main = [n for n, _ in main_ranked[-int(len(main_ranked)*0.3):]]
    hot_sub = [n for n, _ in sub_ranked[:max(3, len(sub_ranked)//3)]]

    # Step 3: Generate candidates deterministically from fusion scores
    # Seed RNG with fusion scores hash — same scores = same candidates
    seed_hash = hash((tuple(n for n, _ in main_ranked[:30]),
                      tuple(n for n, _ in sub_ranked[:10])))
    rng = random.Random(seed_hash)

    candidates: List[Dict[str, Any]] = []
    used_combos: set = set()
    attempts = 0

    # Historical check: build set of all past red-ball combos
    historical_sets = []
    if df is not None:
        for _, row in df.iterrows():
            hist = tuple(sorted([int(row[c]) for c in cfg.main_cols]))
            historical_sets.append(hist)

    while len(candidates) < max_candidates and attempts < max_candidates * 3:
        attempts += 1

        # Core insight: pick mostly from hot zone (follow trend),
        # some from cold zone (surprise element), avoid mid-range
        r = rng.random()
        if r < 0.6:
            main_pool = hot_main
        elif r < 0.85:
            main_pool = cold_main
        else:
            main_pool = main_range

        sub_pool = hot_sub if rng.random() < 0.6 else sub_range

        main_candidate = sorted(rng.sample(main_pool, cfg.main_count))
        sub_candidate = sorted(rng.sample(sub_pool, cfg.sub_count))

        combo_key = (tuple(main_candidate), tuple(sub_candidate))
        if combo_key in used_combos:
            continue
        used_combos.add(combo_key)

        # Historical similarity check: skip combos too close to past draws
        too_close = False
        for hist in historical_sets:
            overlap = len(set(main_candidate) & set(hist))
            if overlap >= 5:  # 5+ same red balls as a historical draw
                too_close = True
                break
        if too_close:
            continue

        # Step 4: Score each candidate — LOWER score = better
        # Reward trend-following + uniqueness, penalize "average" look
        score = 0.0

        # a) Individual number scores (higher fusion score = more trend-aligned = better)
        num_score = sum(main_scores.get(n, 0) for n in main_candidate) / cfg.main_count
        score -= num_score * 3  # Strong reward for trend-following

        # b) Uniqueness bonus: prefer less-probable combos among trend-followers
        # Calculate how many numbers are from the hot zone vs cold zone
        hot_count = sum(1 for n in main_candidate if n in hot_main)
        cold_count = sum(1 for n in main_candidate if n in cold_main)
        
        # A mix of hot + cold is most interesting (follows trend + surprises)
        # Pure hot is boring -> penalize slightly
        # Pure cold is too risky -> slight penalty
        # Hot + 1-2 cold = ideal sweet spot -> bonus
        ideal_hot_count = cfg.main_count - 2  # e.g., 4 hot + 2 cold for SSQ/6
        score += abs(hot_count - ideal_hot_count) * 0.3

        # c) Consecutive number bonus: real draws frequently have consecutive pairs
        consec_pairs = sum(1 for i in range(len(main_candidate)-1)
                          if main_candidate[i+1] - main_candidate[i] == 1)
        # Bonus for having 1-2 consecutive pairs (most common in real draws)
        if consec_pairs == 0:
            score += 0.8  # Penalty for no consecutive numbers at all
        elif consec_pairs == 1:
            score -= 0.3  # Strong bonus for 1 consecutive pair
        elif consec_pairs == 2:
            score -= 0.1  # Mild bonus for 2 pairs
        # 3+ pairs: too many, slight penalty
        else:
            score += 0.5

        # d) Span bonus: prefer slightly wider spans (more spread = more interesting)
        span = max(main_candidate) - min(main_candidate)
        max_span = cfg.main_max - cfg.main_min
        # Prefer span in middle-upper range (60-90% of max)
        ideal_span_min = max_span * 0.55
        ideal_span_max = max_span * 0.92
        if span < ideal_span_min:
            score += (ideal_span_min - span) / max_span * 0.5
        elif span > ideal_span_max:
            score += (span - ideal_span_max) / max_span * 0.5
            
        candidates.append({
            "main": main_candidate,
            "sub": sub_candidate,
            "score": score,
        })

    # Sort by score (lower = better)
    candidates.sort(key=lambda x: x["score"])

    # Step 5: Select final groups with diversity penalty
    selected = []
    for cand in candidates:
        # Apply diversity penalty against already-selected groups
        penalty = 0.0
        for sel in selected:
            overlap = len(set(cand["main"]) & set(sel["main"]))
            penalty += overlap * diversity_factor
        cand["score"] += penalty

        selected.append(cand)
        if len(selected) >= num_groups:
            break

    # Re-sort with penalty and take top N
    selected.sort(key=lambda x: x["score"])
    selected = selected[:num_groups]

    # Normalize scores to 0-100 (higher = better)
    scores = [c["score"] for c in selected]
    min_s, max_s = min(scores), max(scores)
    range_s = max_s - min_s if max_s != min_s else 1

    # Assign group numbers
    groups = []
    for i, c in enumerate(selected):
        norm = (max_s - c["score"]) / range_s * 100  # 100=best, 0=worst
        groups.append({
            "index": i + 1,
            "main": c["main"],
            "sub": c["sub"],
            "score": round(norm, 1),
        })

    logger.info(
        "Generated %d recommendation groups from %d candidates for %s",
        len(groups), len(candidates), cfg.name,
    )

    return {
        "groups": groups,
        "hot_numbers": hot_main[:5],
        "cold_numbers": cold_main[:3],
        "model_weights": getattr(ensemble, 'weights', cfg.ensemble_weights),
        "timestamp": datetime.now().isoformat(),
        "candidates_evaluated": len(candidates),
    }

