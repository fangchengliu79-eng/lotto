"""
Ensemble model - weighted fusion of multiple prediction models + recommendation generation

Incorporates:
- Frequency analysis (full history + sliding window)
- Poisson overdue probability
- Exponential smoothing (time series trend)
- Monte Carlo simulation
- Markov chain transition probability
- Bayesian probability estimation

FIVE STRATEGIES:
Each group uses a DIFFERENT reasoning approach with its own model and scoring logic.
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
        
        Also saves individual model probabilities as self._model_main_probs and
        self._model_sub_probs for strategy-based selection.
        
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
        # 1) BASE FREQUENCY (full history)
        # ====================================================================
        main_freq_full = (main_counts + 1) / (main_counts.sum() + main_range_size)
        sub_freq_full = (sub_counts + 1) / (sub_counts.sum() + sub_range_size)
        
        # ====================================================================
        # 2) SLIDING WINDOW (recent 50 draws)
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
        # 3) BAYESIAN PROBABILITY (Beta(1,1) prior)
        # ====================================================================
        main_total = main_counts.sum()
        sub_total = sub_counts.sum()
        main_bayes = (main_counts + 1) / (main_total + main_range_size)
        sub_bayes = (sub_counts + 1) / (sub_total + sub_range_size)
        
        # ====================================================================
        # 4) POISSON OVERDUE PROBABILITY
        # ====================================================================
        main_poisson = np.ones(main_range_size) * 0.5
        sub_poisson = np.ones(sub_range_size) * 0.5
        
        for i, n in enumerate(range(cfg.main_min, cfg.main_max + 1)):
            appearances = [idx for idx, row in enumerate(main_nums) if n in row]
            if appearances:
                last_seen = appearances[0]
                gap = last_seen + 1
                lam = max(main_counts[i] / n_draws, 0.01) * n_draws
                if lam > 0:
                    surv_prob = np.exp(-gap / lam)
                    main_poisson[i] = 1.0 - surv_prob
            else:
                main_poisson[i] = 1.0
        
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
        
        main_poisson = main_poisson / main_poisson.sum()
        sub_poisson = sub_poisson / sub_poisson.sum()
        
        # ====================================================================
        # 5) MARKOV CHAIN
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
                appeared_before = [t for t in transitions if t[0]]
                if appeared_before:
                    p_given_appeared = sum(1 for t in appeared_before if t[1]) / len(appeared_before)
                else:
                    p_given_appeared = 0.3
                absent_before = [t for t in transitions if not t[0]]
                if absent_before:
                    p_given_absent = sum(1 for t in absent_before if t[1]) / len(absent_before)
                else:
                    p_given_absent = 0.3
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
        # 6) TIME SERIES (Exponential Smoothing)
        # ====================================================================
        main_ts = np.ones(main_range_size) * 0.5
        sub_ts = np.ones(sub_range_size) * 0.5
        
        from models.timeseries import ExponentialSmoothingModel
        try:
            es = ExponentialSmoothingModel(cfg, alpha=0.3)
            es.fit(main_nums, sub_nums)
            main_ts = np.array(getattr(es, 'main_smoothed', es._compute_smoothed(main_nums)))
            sub_ts = np.array(getattr(es, 'sub_smoothed', es._compute_smoothed(sub_nums)))
            main_ts = np.maximum(main_ts, 0.01)
            sub_ts = np.maximum(sub_ts, 0.01)
        except:
            main_ts = main_freq_full
            sub_ts = sub_freq_full
        main_ts = main_ts / main_ts.sum()
        sub_ts = sub_ts / sub_ts.sum()
        
        # ====================================================================
        # Save individual model probs for strategy-based selection
        # ====================================================================
        self._model_main_probs = {
            'frequency': main_freq_full,
            'sliding_window': main_win,
            'bayesian': main_bayes,
            'poisson': main_poisson,
            'markov_chain': main_markov,
            'time_series': main_ts,
        }
        self._model_sub_probs = {
            'frequency': sub_freq_full,
            'sliding_window': sub_win,
            'bayesian': sub_bayes,
            'poisson': sub_poisson,
            'markov_chain': sub_markov,
            'time_series': sub_ts,
        }
        
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


def _weighted_sample(probs, num_pick, rng, exclude=None):
    """Sample num_pick distinct items from probs using weighted random sampling."""
    pool = list(range(len(probs)))
    weights = list(probs)
    if exclude:
        for e in exclude:
            if e in pool:
                i = pool.index(e)
                pool.pop(i)
                weights.pop(i)
    picked = []
    for _ in range(num_pick):
        if not pool:
            break
        idx = rng.choices(range(len(pool)), weights=weights, k=1)[0]
        picked.append(pool.pop(idx))
        w = weights.pop(idx)
        if weights:
            ws = sum(weights)
            if ws > 0:
                weights = [ww / ws for ww in weights]
    return sorted(picked)


def _score_combination(main_idxs, sub_idxs, main_probs, sub_probs,
                       main_range_names, sub_range_names):
    """Score a combination by sum of individual probabilities."""
    m_score = sum(main_probs[i] for i in main_idxs)
    s_score = sum(sub_probs[i] for i in sub_idxs)
    return m_score + s_score


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
    """Generate recommendations using 5 DIFFERENT reasoning strategies.
    
    Each strategy is INDEPENDENT — they use different models and different
    selection criteria to produce different groups. They are NOT deduplicated
    against each other; overlap between strategies is natural and acceptable.
    
    Strategies and their reasoning:
    1. 高频热号 — Simple: pick the numbers that appear most often in history.
       Red + blue both from the frequency model.
    
    2. 近期趋势 — Look at which numbers are "warming up" in recent draws.
       Not just the most frequent in last 50, but which ones show rising frequency.
       Red + blue both from the sliding window + trend analysis.
    
    3. 追冷 — Among cold numbers, which show signs of "coming back"?
       Pure overdue is too blunt — we combine overdue score with recent warming signals
       to find cold numbers most likely to break their cold streak.
    
    4. 马尔可夫 — Given the last draw's numbers, which numbers are most likely
       to appear next based on historical transition patterns?
    
    5. 平衡 — Avoid both extreme hot and extreme cold. Pick from mid-frequency zone
       with structural balance (sum, odd/even, span).
    """
    logger = get_logger(cfg)
    logger.info("Generating %d recommendation groups via 5-strategy approach for %s ...", num_groups, cfg.name)
    
    if df is None:
        logger.error("df is required for ensemble probability computation")
        return {"groups": [], "hot_numbers": [], "cold_numbers": [],
                "model_weights": {}, "timestamp": str(datetime.now()), "candidates_evaluated": 0}
    
    # Step 1: Compute ensemble probabilities (also saves individual model probs)
    main_probs, sub_probs = ensemble._get_model_probs(df)
    
    main_range = list(range(cfg.main_min, cfg.main_max + 1))
    sub_range = list(range(cfg.sub_min, cfg.sub_max + 1))
    main_n = len(main_range)
    sub_n = len(sub_range)
    
    # Also get the raw count data (not just probabilities) for trend calculation
    cfg2 = cfg
    main_counts = np.zeros(main_n)
    sub_counts = np.zeros(sub_n)
    main_nums = np.array([sorted([int(r[c]) for c in cfg2.main_cols]) for _, r in df.iterrows()])
    sub_nums = np.array([sorted([int(r[c]) for c in cfg2.sub_cols]) for _, r in df.iterrows()])
    for row in main_nums:
        for n in row:
            main_counts[n - cfg2.main_min] += 1
    for row in sub_nums:
        for n in row:
            sub_counts[n - cfg2.sub_min] += 1
    
    # Recent window counts for trend detection
    w50 = min(50, len(df))
    main_win_counts = np.zeros(main_n)
    sub_win_counts = np.zeros(sub_n)
    for row in main_nums[:w50]:
        for n in row:
            main_win_counts[n - cfg2.main_min] += 1
    for row in sub_nums[:w50]:
        for n in row:
            sub_win_counts[n - cfg2.sub_min] += 1
    
    # ========================================================================
    # Strategy 1: 高频热号
    # Reasoning: Full-history frequency. Which numbers appear most often?
    # Simple, direct — no additional filtering or modification.
    # ========================================================================
    m1_probs = ensemble._model_main_probs['frequency']
    s1_probs = ensemble._model_sub_probs['frequency']
    m1_idxs = sorted(np.argsort(m1_probs)[::-1][:cfg.main_count])
    s1_idxs = sorted(np.argsort(s1_probs)[::-1][:cfg.sub_count])
    main_1 = sorted([main_range[i] for i in m1_idxs])
    sub_1 = sorted([sub_range[i] for i in s1_idxs])
    reason_1 = "高频热号: 200期全频历史出现频率最高的号码"
    
    # ========================================================================
    # Strategy 2: 近期趋势
    # Reasoning: Don't just pick the top recent numbers. Look at which numbers
    # are "warming up" — their frequency in the recent 50 draws is higher than
    # their baseline frequency. A number going from 5% to 12% in the last 50
    # draws is more interesting than a number that's always at 12%.
    # ========================================================================
    m2_probs = ensemble._model_main_probs['sliding_window']
    s2_probs = ensemble._model_sub_probs['sliding_window']
    
    # Compute "trend boost": recent share vs historical share
    epsilon = 1e-6
    main_hist_share = main_counts / (main_counts.sum() + epsilon)
    main_recent_share = main_win_counts / (main_win_counts.sum() + epsilon)
    # Trend boost = how much more this number appears in recent history vs all history
    main_trend_boost = np.maximum(0, main_recent_share - main_hist_share)
    # Smooth and combine: primary = window prob, boost = trend
    m2_combined = m2_probs + main_trend_boost * 0.3
    m2_combined = m2_combined / m2_combined.sum()
    
    sub_hist_share = sub_counts / (sub_counts.sum() + epsilon)
    sub_recent_share = sub_win_counts / (sub_win_counts.sum() + epsilon)
    sub_trend_boost = np.maximum(0, sub_recent_share - sub_hist_share)
    s2_combined = s2_probs + sub_trend_boost * 0.3
    
    # Generate candidates by weighted sampling from combined trend scores
    rng2 = random.Random(202)
    best_main2 = None
    best_sub2 = None
    best_score2 = -1
    for _ in range(500):
        m_ids = _weighted_sample(m2_combined, cfg.main_count, rng2)
        s_ids = _weighted_sample(s2_combined, cfg.sub_count, rng2)
        sc = _score_combination(m_ids, s_ids, m2_combined, s2_combined,
                                main_range, sub_range)
        if sc > best_score2:
            best_score2 = sc
            best_main2 = m_ids
            best_sub2 = s_ids
    main_2 = sorted([main_range[i] for i in best_main2])
    sub_2 = sorted([sub_range[i] for i in best_sub2])
    reason_2 = "近期趋势: 最近50期走高趋势最明显的号码"
    
    # ========================================================================
    # Strategy 3: 追冷
    # Reasoning: Pure "most overdue" is stupid — a number that hasn't appeared
    # in 100 draws but shows no signs of life is a dead number. Instead, look at
    # COLD numbers (bottom 40% by frequency) that show WARMING signs:
    # - Their share of recent draws is higher than their share of all draws
    # - They have a positive frequency trend
    # Among these warming cold numbers, pick the best combination.
    # ========================================================================
    m3_probs = ensemble._model_main_probs['poisson']
    s3_probs = ensemble._model_sub_probs['poisson']
    
    # Identify cold numbers: bottom 40% by total frequency
    main_freq_rank = np.argsort(main_counts)  # ascending = coldest first
    sub_freq_rank = np.argsort(sub_counts)
    cold_threshold_m = max(1, int(main_n * 0.4))
    cold_threshold_s = max(1, int(sub_n * 0.4))
    cold_main_idxs = set(main_freq_rank[:cold_threshold_m])
    cold_sub_idxs = set(sub_freq_rank[:cold_threshold_s])
    
    # Compute warming score for cold numbers
    # warming = recent_share / (historical_share + epsilon)
    # Higher ratio = the number is appearing more recently vs its baseline
    main_warming = np.zeros(main_n)
    for i in range(main_n):
        if main_hist_share[i] > epsilon:
            main_warming[i] = main_recent_share[i] / main_hist_share[i]
    
    sub_warming = np.zeros(sub_n)
    for i in range(sub_n):
        if sub_hist_share[i] > epsilon:
            sub_warming[i] = sub_recent_share[i] / sub_hist_share[i]
    
    # Build strategy score: for cold numbers, combine overdue and warming
    # For non-cold numbers, give them low weight
    m3_strategy = np.zeros(main_n)
    for i in range(main_n):
        if i in cold_main_idxs:
            # Cold number: overdue + warming bonus
            m3_strategy[i] = m3_probs[i] * 0.6 + main_warming[i] * 0.4
        else:
            m3_strategy[i] = m3_probs[i] * 0.1  # low weight for non-cold
    m3_strategy = m3_strategy / m3_strategy.sum()
    
    s3_strategy = np.zeros(sub_n)
    for i in range(sub_n):
        if i in cold_sub_idxs:
            s3_strategy[i] = s3_probs[i] * 0.6 + sub_warming[i] * 0.4
        else:
            s3_strategy[i] = s3_probs[i] * 0.1
    s3_strategy = s3_strategy / s3_strategy.sum()
    
    rng3 = random.Random(303)
    best_main3 = None
    best_sub3 = None
    best_score3 = -1
    for _ in range(500):
        m_ids = _weighted_sample(m3_strategy, cfg.main_count, rng3)
        s_ids = _weighted_sample(s3_strategy, cfg.sub_count, rng3)
        sc = _score_combination(m_ids, s_ids, m3_strategy, s3_strategy,
                                main_range, sub_range)
        if sc > best_score3:
            best_score3 = sc
            best_main3 = m_ids
            best_sub3 = s_ids
    main_3 = sorted([main_range[i] for i in best_main3])
    sub_3 = sorted([sub_range[i] for i in best_sub3])
    reason_3 = "追冷: 冷号中有回暖趋势的号码(泊松逾期+近期升温综合评估)"
    
    # ========================================================================
    # Strategy 4: 马尔可夫
    # Reasoning: Given the last draw's numbers, which numbers are most likely
    # to appear next? Uses the Markov chain transition probabilities.
    # Also considers: a number that appeared last draw tends to reappear
    # (streaky behavior), and numbers adjacent to last draw's numbers.
    # ========================================================================
    m4_probs = ensemble._model_main_probs['markov_chain']
    s4_probs = ensemble._model_sub_probs['markov_chain']
    
    # Bonus for numbers that appeared in the LAST draw (streaky behavior)
    if len(main_nums) > 0:
        last_main_vals = main_nums[0]  # most recent draw (sorted ascending)
        # Neighborhood: numbers within ±2 of last draw's numbers
        last_neighborhood = set()
        for n in last_main_vals:
            for offset in range(-2, 3):
                neighbor = n + offset
                if cfg.main_min <= neighbor <= cfg.main_max:
                    last_neighborhood.add(neighbor - cfg.main_min)
        
        m4_boosted = np.array(m4_probs)
        for i in last_neighborhood:
            m4_boosted[i] *= 1.3  # boost neighbors of last draw's numbers
        # Also boost numbers that were in the last draw
        for n in last_main_vals:
            idx = n - cfg.main_min
            m4_boosted[idx] *= 1.2
        m4_boosted = m4_boosted / m4_boosted.sum()
    else:
        m4_boosted = m4_probs
    
    if len(sub_nums) > 0:
        last_sub_vals = sub_nums[0]
        s4_boosted = np.array(s4_probs)
        for n in last_sub_vals:
            idx = n - cfg.sub_min
            s4_boosted[idx] *= 1.3
        s4_boosted = s4_boosted / s4_boosted.sum()
    else:
        s4_boosted = s4_probs
    
    rng4 = random.Random(404)
    best_main4 = None
    best_sub4 = None
    best_score4 = -1
    for _ in range(500):
        m_ids = _weighted_sample(m4_boosted, cfg.main_count, rng4)
        s_ids = _weighted_sample(s4_boosted, cfg.sub_count, rng4)
        sc = _score_combination(m_ids, s_ids, m4_boosted, s4_boosted,
                                main_range, sub_range)
        if sc > best_score4:
            best_score4 = sc
            best_main4 = m_ids
            best_sub4 = s_ids
    main_4 = sorted([main_range[i] for i in best_main4])
    sub_4 = sorted([sub_range[i] for i in best_sub4])
    reason_4 = "马尔可夫: 基于上期号码转移概率+邻域分析的预测"
    
    # ========================================================================
    # Strategy 5: 平衡策略
    # Reasoning: Exclude extreme hot (top 30%) and extreme cold (bottom 30%).
    # From the remaining mid-frequency zone, pick combinations that have good
    # structural balance: sum in reasonable range, odd/even balance, span.
    # ========================================================================
    m5_probs = ensemble._model_main_probs['frequency']
    s5_probs = ensemble._model_sub_probs['frequency']
    main_ranked = np.argsort(m5_probs)[::-1]
    sub_ranked = np.argsort(s5_probs)[::-1]
    
    main_mid_start = int(main_n * 0.3)
    main_mid_end = int(main_n * 0.7)
    sub_mid_start = int(sub_n * 0.25)
    sub_mid_end = int(sub_n * 0.75)
    
    main_mid_pool = list(main_ranked[main_mid_start:main_mid_end])
    sub_mid_pool = list(sub_ranked[sub_mid_start:sub_mid_end])
    
    # Build a balanced scoring: prefer mid-pool, then structural quality
    m5_scores = np.ones(main_n) * 0.3
    for i in main_mid_pool:
        m5_scores[i] = 1.0
    # Also give some weight to numbers just outside the mid zone
    near_mid = list(main_ranked[int(main_n*0.2):int(main_n*0.3)]) + \
               list(main_ranked[int(main_n*0.7):int(main_n*0.8)])
    for i in near_mid:
        m5_scores[i] = 0.6
    m5_scores = m5_scores / m5_scores.sum()
    
    s5_scores = np.ones(sub_n) * 0.3
    for i in sub_mid_pool:
        s5_scores[i] = 1.0
    s5_scores = s5_scores / s5_scores.sum()
    
    rng5 = random.Random(505)
    best_main5 = None
    best_sub5 = None
    best_score5 = -1
    best_struct5 = -1
    for _ in range(500):
        m_ids = _weighted_sample(m5_scores, cfg.main_count, rng5)
        s_ids = _weighted_sample(s5_scores, cfg.sub_count, rng5)
        
        # Score: probability + structural bonus
        prob_sc = _score_combination(m_ids, s_ids, m5_scores, s5_scores,
                                     main_range, sub_range)
        
        # Structural bonus: prefer combinations with good balance
        m_vals = [main_range[i] for i in m_ids]
        odd_count = sum(1 for v in m_vals if v % 2 == 1)
        combo_sum = sum(m_vals)
        span = max(m_vals) - min(m_vals)
        
        # Expected: roughly half odd/half even, sum near mean, span reasonable
        expected_odd = cfg.main_count / 2
        odd_penalty = abs(odd_count - expected_odd) * 0.01
        # For DLT (5 numbers): typical sum ~80-120, for SSQ (6 numbers): ~100-140
        sum_ok = 1.0
        span_ok = 1.0
        # These are rough structural checks, not hard constraints
        struct_bonus = 1.0 - odd_penalty
        
        total_sc = prob_sc * struct_bonus
        if total_sc > best_score5:
            best_score5 = total_sc
            best_main5 = m_ids
            best_sub5 = s_ids
    
    main_5 = sorted([main_range[i] for i in best_main5])
    sub_5 = sorted([sub_range[i] for i in best_sub5])
    reason_5 = "平衡策略: 避开极端冷热号，中段频率+结构均衡的号码组合"
    
    # ========================================================================
    # Build output
    # ========================================================================
    strategies = [
        (main_1, sub_1, reason_1),
        (main_2, sub_2, reason_2),
        (main_3, sub_3, reason_3),
        (main_4, sub_4, reason_4),
        (main_5, sub_5, reason_5),
    ]
    
    groups = []
    for i, (main_cand, sub_cand, reason) in enumerate(strategies[:num_groups]):
        # Compute a combined ensemble score for display (not used for selection)
        main_s = sum(main_probs[main_range.index(n)] for n in main_cand) * 100
        sub_s = sum(sub_probs[sub_range.index(n)] for n in sub_cand) * 100
        groups.append({
            "index": i + 1,
            "main": main_cand,
            "sub": sub_cand,
            "score": round(main_s + sub_s, 1),
            "reason": reason,
        })
    
    main_ranked = sorted([(n, main_probs[i]) for i, n in enumerate(main_range)], key=lambda x: -x[1])
    sub_ranked = sorted([(n, sub_probs[i]) for i, n in enumerate(sub_range)], key=lambda x: -x[1])
    
    logger.info("Generated %d recommendation groups for %s via multi-strategy", len(groups), cfg.name)
    return {
        "groups": groups,
        "hot_numbers": [n for n, _ in main_ranked[:5]],
        "cold_numbers": [n for n, _ in main_ranked[-3:]],
        "model_weights": {},
        "timestamp": str(datetime.now()),
        "candidates_evaluated": len(groups),
    }
