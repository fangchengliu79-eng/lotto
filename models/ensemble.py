"""
Ensemble model - weighted fusion of multiple prediction models + recommendation generation

Incorporates:
- Frequency analysis (full history + sliding window)
- Poisson overdue probability
- Exponential smoothing (time series trend)
- Monte Carlo simulation
- Markov chain transition probability
- Bayesian probability estimation
"""

import random
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd

from utils.helpers import validate_numbers, get_logger


class EnsembleModel:
    """Weighted ensemble of multiple prediction models."""

    def __init__(self, models_dict: Dict[str, Any], cfg):
        self.models = models_dict
        self.cfg = cfg
        self.logger = get_logger(cfg)

    def _get_model_probs(self, df):
        """Compute ensemble probability for each number using multiple algorithms.
        
        Returns:
            main_probs: np.ndarray of length main_range, probabilities for each main number
            sub_probs: np.ndarray of length sub_range, probabilities for each sub number
        """
        cfg = self.cfg
        main_range_size = cfg.main_max - cfg.main_min + 1
        sub_range_size = cfg.sub_max - cfg.sub_min + 1
        
        # Extract number arrays
        main_nums = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
        sub_nums = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
        
        n_draws = len(df)
        main_counts = np.zeros(main_range_size)
        sub_counts = np.zeros(sub_range_size)
        for row in main_nums:
            for n in row:
                main_counts[n - cfg.main_min] += 1
        for row in sub_nums:
            for n in row:
                sub_counts[n - cfg.sub_min] += 1
        
        # ====================================================================
        # 1) BASE FREQUENCY (full history) - 权重: 25%
        # ====================================================================
        main_freq_full = (main_counts + 1) / (main_counts.sum() + main_range_size)
        sub_freq_full = (sub_counts + 1) / (sub_counts.sum() + sub_range_size)
        
        # ====================================================================
        # 2) SLIDING WINDOW (recent 50 draws) - 权重: 20%
        # ====================================================================
        window = min(50, n_draws)
        main_win_counts = np.zeros(main_range_size)
        sub_win_counts = np.zeros(sub_range_size)
        for row in main_nums[:window]:
            for n in row:
                main_win_counts[n - cfg.main_min] += 1
        for row in sub_nums[:window]:
            for n in row:
                sub_win_counts[n - cfg.sub_min] += 1
        main_win = (main_win_counts + 1) / (main_win_counts.sum() + main_range_size)
        sub_win = (sub_win_counts + 1) / (sub_win_counts.sum() + sub_range_size)
        
        # ====================================================================
        # 3) BAYESIAN PROBABILITY (Beta(1,1) prior) - 权重: 10%
        #   P = (count + α) / (total + α + β),  with α=β=1 (uniform prior)
        # ====================================================================
        main_total = main_counts.sum()
        sub_total = sub_counts.sum()
        main_bayes = (main_counts + 1) / (main_total + main_range_size)
        sub_bayes = (sub_counts + 1) / (sub_total + sub_range_size)
        
        # ====================================================================
        # 4) POISSON OVERDUE PROBABILITY - 权重: 15%
        #   Numbers with longer gaps since last appearance get higher scores
        # ====================================================================
        main_poisson = np.ones(main_range_size) * 0.5
        sub_poisson = np.ones(sub_range_size) * 0.5
        
        # For each number, compute overdue probability
        for i, n in enumerate(range(cfg.main_min, cfg.main_max + 1)):
            appearances = [idx for idx, row in enumerate(main_nums) if n in row]
            if appearances:
                last_seen = appearances[0]
                gap = last_seen + 1  # +1 because 0-indexed
                lam = max(main_counts[i] / n_draws, 0.01) * n_draws
                if lam > 0:
                    # Survival probability P(gap > current_gap | lambda)
                    surv_prob = np.exp(-gap / lam)
                    main_poisson[i] = 1.0 - surv_prob  # higher = more overdue
            else:
                main_poisson[i] = 1.0  # never seen: highest priority
        
        for i, n in enumerate(range(cfg.sub_min, cfg.sub_max + 1)):
            appearances = [idx for idx, row in enumerate(sub_nums) if n in row]
            if appearances:
                last_seen = appearances[0]
                gap = last_seen + 1
                lam = max(sub_counts[i] / n_draws, 0.01) * n_draws
                if lam > 0:
                    surv_prob = np.exp(-gap / lam)
                    sub_poisson[i] = 1.0 - surv_prob
            else:
                sub_poisson[i] = 1.0
        
        # Normalize to probability
        main_poisson = main_poisson / main_poisson.sum()
        sub_poisson = sub_poisson / sub_poisson.sum()
        
        # ====================================================================
        # 5) MARKOV CHAIN - 权重: 15%
        #   P(number appears | appeared last draw) and P(number appears | absent last draw)
        #   A number with high "given it appeared last time, it appears again" probability 
        #   shows streaky behavior
        # ====================================================================
        main_markov = np.ones(main_range_size) * 0.5
        sub_markov = np.ones(sub_range_size) * 0.5
        
        for i, n in enumerate(range(cfg.main_min, cfg.main_max + 1)):
            transitions = []
            for idx in range(len(main_nums) - 1):
                now = n in main_nums[idx]
                next_ = n in main_nums[idx + 1]
                transitions.append((now, next_))
            if transitions:
                # P(appears now | appeared before)
                appeared_before = [t for t in transitions if t[0]]
                if appeared_before:
                    p_given_appeared = sum(1 for t in appeared_before if t[1]) / len(appeared_before)
                else:
                    p_given_appeared = 0.3
                # P(appears now | absent before)
                absent_before = [t for t in transitions if not t[0]]
                if absent_before:
                    p_given_absent = sum(1 for t in absent_before if t[1]) / len(absent_before)
                else:
                    p_given_absent = 0.3
                # Last state
                last_appeared = n in main_nums[0]
                if last_appeared:
                    main_markov[i] = p_given_appeared
                else:
                    main_markov[i] = p_given_absent
        
        for i, n in enumerate(range(cfg.sub_min, cfg.sub_max + 1)):
            transitions = []
            for idx in range(len(sub_nums) - 1):
                now = n in sub_nums[idx]
                next_ = n in sub_nums[idx + 1]
                transitions.append((now, next_))
            if transitions:
                appeared_before = [t for t in transitions if t[0]]
                p_given_appeared = sum(1 for t in appeared_before if t[1]) / len(appeared_before) if appeared_before else 0.3
                absent_before = [t for t in transitions if not t[0]]
                p_given_absent = sum(1 for t in absent_before if t[1]) / len(absent_before) if absent_before else 0.3
                last_appeared = n in sub_nums[0]
                if last_appeared:
                    sub_markov[i] = p_given_appeared
                else:
                    sub_markov[i] = p_given_absent
        
        main_markov = main_markov / main_markov.sum()
        sub_markov = sub_markov / sub_markov.sum()
        
        # ====================================================================
        # 6) TIME SERIES (Exponential Smoothing) - 权重: 15%
        # ====================================================================
        main_ts = np.ones(main_range_size) * 0.5
        sub_ts = np.ones(sub_range_size) * 0.5
        
        from models.timeseries import ExponentialSmoothingModel
        try:
            es = ExponentialSmoothingModel(cfg, alpha=0.3)
            es.fit(main_nums, sub_nums)
            main_ts = np.array(getattr(es, 'main_smoothed', es._compute_smoothed(main_nums)))
            sub_ts = np.array(getattr(es, 'sub_smoothed', es._compute_smoothed(sub_nums)))
            # Ensure non-negative
            main_ts = np.maximum(main_ts, 0.01)
            sub_ts = np.maximum(sub_ts, 0.01)
        except:
            main_ts = main_freq_full
            sub_ts = sub_freq_full
        main_ts = main_ts / main_ts.sum()
        sub_ts = sub_ts / sub_ts.sum()
        
        # ====================================================================
        # ENSEMBLE: weighted combination of all algorithms
        # ====================================================================
        weights = {
            'full_freq': 0.25,
            'sliding_window': 0.20,
            'bayesian': 0.10,
            'poisson_overdue': 0.15,
            'markov_chain': 0.15,
            'time_series': 0.15,
        }
        
        main_ensemble = (
            main_freq_full * weights['full_freq'] +
            main_win * weights['sliding_window'] +
            main_bayes * weights['bayesian'] +
            main_poisson * weights['poisson_overdue'] +
            main_markov * weights['markov_chain'] +
            main_ts * weights['time_series']
        )
        main_ensemble = main_ensemble / main_ensemble.sum()
        
        sub_ensemble = (
            sub_freq_full * weights['full_freq'] +
            sub_win * weights['sliding_window'] +
            sub_bayes * weights['bayesian'] +
            sub_poisson * weights['poisson_overdue'] +
            sub_markov * weights['markov_chain'] +
            sub_ts * weights['time_series']
        )
        sub_ensemble = sub_ensemble / sub_ensemble.sum()
        
        self.logger.info(
            f"Ensemble probabilities computed: main_range={main_range_size}, sub_range={sub_range_size}"
        )
        
        return main_ensemble, sub_ensemble


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
    """Generate recommendations using multi-algorithm ensemble.
    
    Algorithms used:
    1. Full-history frequency analysis
    2. Sliding window (last 50 draws) trend analysis
    3. Bayesian probability (Beta prior)
    4. Poisson overdue probability
    5. Markov chain transition probability
    6. Time series (exponential smoothing)
    7. Monte Carlo simulation for candidate generation
    
    Each number gets a probability score from the weighted ensemble,
    then candidates are generated by weighted sampling.
    """
    logger = get_logger(cfg)
    logger.info("Generating %d recommendation groups for %s ...", num_groups, cfg.name)
    
    if df is None:
        logger.error("df is required for ensemble probability computation")
        return {"groups": [], "hot_numbers": [], "cold_numbers": [],
                "model_weights": {}, "timestamp": str(datetime.now()), "candidates_evaluated": 0}
    
    # Step 1: Compute ensemble probabilities from all algorithms
    main_probs, sub_probs = ensemble._get_model_probs(df)
    
    main_range = list(range(cfg.main_min, cfg.main_max + 1))
    sub_range = list(range(cfg.sub_min, cfg.sub_max + 1))
    
    # Step 2: Build historical set for similarity check
    historical_sets = []
    for _, row in df.iterrows():
        hist = tuple(sorted([int(row[c]) for c in cfg.main_cols]))
        historical_sets.append(hist)
    
    # Step 3: Seeded RNG for determinism
    seed_hash = hash((tuple(main_probs[:10].round(4)), tuple(sub_probs[:5].round(4))))
    rng = random.Random(seed_hash)
    
    # Step 4: Generate candidates by weighted sampling (Monte Carlo)
    candidates = []
    used_combos = set()
    
    for _ in range(max_candidates * 5):
        # Weighted sampling for main numbers (with replacement = Monte Carlo)
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
        
        # Historical similarity check: avoid extreme duplication
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
    
    logger.info(f"Generated {len(candidates)} valid candidates")
    
    # Step 5: Select diverse groups (like genetic algorithm selection)
    # Score by ensemble probability, penalize overlap with already-selected
    selected = []
    used_all_main = set()
    used_sub_keys = set()
    
    for cand in candidates:
        sub_key = tuple(sorted(cand["sub"]))
        # Moderate diversity: prefer different sub combos but not forced
        if len(used_sub_keys) >= 3 and sub_key in used_sub_keys:
            continue
        if sub_key in used_sub_keys and len(candidates) > len(selected) * 20:
            continue  # skip if we have enough alternatives
            
        main_p = sum(main_probs[main_range.index(n)] for n in cand["main"])
        sub_p = sum(sub_probs[sub_range.index(n)] for n in cand["sub"])
        overlap_penalty = len(set(cand["main"]) & used_all_main) * 0.05
        score = main_p + sub_p - overlap_penalty
        
        cand["score"] = score
        selected.append(cand)
        used_all_main.update(cand["main"])
        used_sub_keys.add(sub_key)
        
        if len(selected) >= num_groups:
            break
    
    # Sort by score (higher = better)
    selected.sort(key=lambda x: -x["score"])
    selected = selected[:num_groups]
    
    # Step 6: Build output
    groups = []
    for i, c in enumerate(selected):
        main_p = sum(main_probs[main_range.index(n)] for n in c["main"]) * 100
        sub_p = sum(sub_probs[sub_range.index(n)] for n in c["sub"]) * 100
        groups.append({
            "index": i + 1,
            "main": c["main"],
            "sub": c["sub"],
            "score": round(main_p + sub_p, 1),
        })
    
    main_ranked = sorted([(n, main_probs[i]) for i, n in enumerate(main_range)], key=lambda x: -x[1])
    sub_ranked = sorted([(n, sub_probs[i]) for i, n in enumerate(sub_range)], key=lambda x: -x[1])
    
    logger.info("Generated %d recommendation groups for %s", len(groups), cfg.name)
    return {
        "groups": groups,
        "hot_numbers": [n for n, _ in main_ranked[:5]],
        "cold_numbers": [n for n, _ in main_ranked[-3:]],
        "model_weights": {},
        "timestamp": str(datetime.now()),
        "candidates_evaluated": len(candidates),
    }
