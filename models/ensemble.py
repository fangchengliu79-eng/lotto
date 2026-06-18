"""
Ensemble model - enhanced with international best practices

Incorporates:
- Frequency analysis (full history + sliding window)
- Decay-weighted frequency (0.98^weeks)  ← NEW
- Gap/recency analysis                   ← NEW
- Poisson overdue probability
- Exponential smoothing (time series trend)
- Markov chain transition probability
- Bayesian log-space fusion              ← NEW
- CRF-like global combination scoring    ← NEW

FIVE STRATEGIES:
Each group uses a DIFFERENT reasoning approach with its own model and scoring logic.
"""

import math
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
        """Compute probability distributions using multiple algorithms.
        
        NEW: Decay-weighted frequency + gap/recency analysis added as signals.
        Ensemble uses Bayesian log-space fusion instead of linear average.
        
        Also saves individual model probabilities as self._model_main_probs and
        self._model_sub_probs for strategy-based selection.
        
        Returns:
            main_probs: np.ndarray of length main_range
            sub_probs: np.ndarray of length sub_range
        """
        cfg = self.cfg
        main_range_size = cfg.main_max - cfg.main_min + 1
        sub_range_size = cfg.sub_max - cfg.sub_min + 1
        
        # Extract number arrays (chronological: index 0 = newest draw)
        main_nums = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
        sub_nums = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
        n_draws = len(df)
        
        # Raw counts
        main_counts = np.zeros(main_range_size)
        sub_counts = np.zeros(sub_range_size)
        for row in main_nums:
            for n in row:
                main_counts[n - cfg.main_min] += 1
        for row in sub_nums:
            for n in row:
                sub_counts[n - cfg.sub_min] += 1
        
        # ====================================================================
        # 1) BASE FREQUENCY (full history, equal weight)
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
        # 3) NEW: DECAY-WEIGHTED FREQUENCY (0.98^draws)
        #    Recent draws contribute more, old draws fade out.
        #    This naturally de-emphasizes ancient hot numbers.
        # ====================================================================
        decay_factor = 0.98
        main_decay_counts = np.zeros(main_range_size)
        sub_decay_counts = np.zeros(sub_range_size)
        for idx in range(n_draws):
            w = decay_factor ** idx  # idx=0(newest) gets highest weight
            for n in main_nums[idx]:
                main_decay_counts[n - cfg.main_min] += w
            for n in sub_nums[idx]:
                sub_decay_counts[n - cfg.sub_min] += w
        main_decay = (main_decay_counts + 1) / (main_decay_counts.sum() + main_range_size)
        sub_decay = (sub_decay_counts + 1) / (sub_decay_counts.sum() + sub_range_size)
        
        # ====================================================================
        # 4) BAYESIAN PROBABILITY (Beta(1,1) prior)
        # ====================================================================
        main_total = main_counts.sum()
        sub_total = sub_counts.sum()
        main_bayes = (main_counts + 1) / (main_total + main_range_size)
        sub_bayes = (sub_counts + 1) / (sub_total + sub_range_size)
        
        # ====================================================================
        # 5) POISSON OVERDUE PROBABILITY
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
        # 6) MARKOV CHAIN
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
                p_given_appeared = sum(1 for t in appeared_before if t[1]) / len(appeared_before) if appeared_before else 0.3
                absent_before = [t for t in transitions if not t[0]]
                p_given_absent = sum(1 for t in absent_before if t[1]) / len(absent_before) if absent_before else 0.3
                last_appeared = n in main_nums[0]
                main_markov[i] = p_given_appeared if last_appeared else p_given_absent
        
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
                sub_markov[i] = p_given_appeared if last_appeared else p_given_absent
        
        main_markov = main_markov / main_markov.sum()
        sub_markov = sub_markov / sub_markov.sum()
        
        # ====================================================================
        # 7) TIME SERIES (Exponential Smoothing)
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
        # 8) NEW: GAP/RECENCY ANALYSIS
        #    Track the current absence gap for each number.
        #    Numbers with unusually long gaps (vs historical) get a boost.
        #    This is independent of the Poisson model — it measures "how unusual
        #    is this gap" rather than "how overdue is this number".
        # ====================================================================
        main_gap = np.zeros(main_range_size)
        sub_gap = np.zeros(sub_range_size)
        
        for i, n in enumerate(range(cfg.main_min, cfg.main_max + 1)):
            appearances = [idx for idx, row in enumerate(main_nums) if n in row]
            if appearances:
                current_gap = appearances[0]  # index 0 = newest draw
                # Historical gaps between appearances
                hist_gaps = []
                for j in range(len(appearances) - 1):
                    hist_gaps.append(appearances[j+1] - appearances[j])
                if hist_gaps:
                    mean_gap = np.mean(hist_gaps)
                    std_gap = np.std(hist_gaps) + 0.01
                    # Z-score: how many std devs is current gap from historical mean?
                    z = (current_gap - mean_gap) / std_gap
                    # Sigmoid: z=0→0.5, z=2→0.88, z=-2→0.12
                    main_gap[i] = 1.0 / (1.0 + math.exp(-z * 0.8))
                else:
                    main_gap[i] = 0.5
            else:
                # Never appeared — give moderate boost (not extreme)
                main_gap[i] = 0.7
        
        for i, n in enumerate(range(cfg.sub_min, cfg.sub_max + 1)):
            appearances = [idx for idx, row in enumerate(sub_nums) if n in row]
            if appearances:
                current_gap = appearances[0]
                hist_gaps = []
                for j in range(len(appearances) - 1):
                    hist_gaps.append(appearances[j+1] - appearances[j])
                if hist_gaps:
                    mean_gap = np.mean(hist_gaps)
                    std_gap = np.std(hist_gaps) + 0.01
                    z = (current_gap - mean_gap) / std_gap
                    sub_gap[i] = 1.0 / (1.0 + math.exp(-z * 0.8))
                else:
                    sub_gap[i] = 0.5
            else:
                sub_gap[i] = 0.7
        
        main_gap = main_gap / main_gap.sum()
        sub_gap = sub_gap / sub_gap.sum()
        
        # ====================================================================
        # Save individual model probs for strategy-based selection
        # ====================================================================
        self._model_main_probs = {
            'frequency': main_freq_full,
            'decay_frequency': main_decay,
            'gap_recency': main_gap,
            'sliding_window': main_win,
            'bayesian': main_bayes,
            'poisson': main_poisson,
            'markov_chain': main_markov,
            'time_series': main_ts,
        }
        self._model_sub_probs = {
            'frequency': sub_freq_full,
            'decay_frequency': sub_decay,
            'gap_recency': sub_gap,
            'sliding_window': sub_win,
            'bayesian': sub_bayes,
            'poisson': sub_poisson,
            'markov_chain': sub_markov,
            'time_series': sub_ts,
        }
        
        # ====================================================================
        # ENSEMBLE: Bayesian log-space fusion
        #    P(n) ∝ Π P(n|model_i)^w_i
        #    Unlike linear average, log-space fusion naturally handles
        #    conflicting evidence: if one model says 0.01 and another says 0.99,
        #    their geometric mean is ~0.1, not 0.5.
        # ====================================================================
        weights = {
            'decay_frequency': 0.20,  # decay-weighted > raw frequency
            'sliding_window': 0.18,
            'gap_recency': 0.12,      # NEW signal
            'poisson_overdue': 0.15,
            'markov_chain': 0.15,
            'time_series': 0.12,
            'bayesian': 0.08,
        }
        
        # Bayesian fusion: log-space weighted product
        eps = 1e-10
        main_log = np.zeros(main_range_size)
        sub_log = np.zeros(sub_range_size)
        model_map = {
            'decay_frequency': (main_decay, sub_decay),
            'sliding_window': (main_win, sub_win),
            'gap_recency': (main_gap, sub_gap),
            'poisson_overdue': (main_poisson, sub_poisson),
            'markov_chain': (main_markov, sub_markov),
            'time_series': (main_ts, sub_ts),
            'bayesian': (main_bayes, sub_bayes),
        }
        for model_name, (mp, sp) in model_map.items():
            w = weights.get(model_name, 0.10)
            main_log += w * np.log(mp + eps)
            sub_log += w * np.log(sp + eps)
        
        main_ensemble = np.exp(main_log)
        sub_ensemble = np.exp(sub_log)
        main_ensemble = main_ensemble / main_ensemble.sum()
        sub_ensemble = sub_ensemble / sub_ensemble.sum()
        
        self.logger.info(
            f"Bayesian ensemble probs computed: main={main_range_size}, sub={sub_range_size}"
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


def _bayesian_fusion(model_probs_dict, weights_dict, n, eps=1e-10):
    """Bayesian log-space fusion of multiple probability distributions.
    
    P(n) ∝ exp(Σ w_i * log(P_i(n)))
    """
    log_p = np.zeros(n)
    for name, prob in model_probs_dict.items():
        w = weights_dict.get(name, 0.1)
        log_p += w * np.log(prob + eps)
    result = np.exp(log_p)
    return result / result.sum()


def _check_dlt_zone(main_vals, cold_mode=False):
    """大乐透前区区间分布过滤：返回区间分布模式的优先级分值。
    
    通用模式（cold_mode=False, 策略1/2/4/5）：
      Priority 2 (☆最优): ABBBC(A=1,B=3,C=1,D=0) 或 ABBBD(A=1,B=3,C=0,D=1)
                          → B区3个相近（差值≤5）
      Priority 1 (✓可接受): ABBCD(A=1,B=2,C=1,D=1)
    
    追冷模式（cold_mode=True, 策略3）：
      Priority 2 (☆最优): ABBCC(A=1,B=2,C=2,D=0), ABCCC(A=1,B=1,C=3,D=0), 
                          ACCCD(A=1,C=3,D=1)
      Priority 1 (✓可接受): ABBBC(A=1,B=3,C=1,D=0) 或 ABBBD(A=1,B=3,C=0,D=1)
    """
    a = sum(1 for n in main_vals if 1 <= n <= 9)
    b = sum(1 for n in main_vals if 10 <= n <= 20)
    c = sum(1 for n in main_vals if 21 <= n <= 29)
    d = sum(1 for n in main_vals if 30 <= n <= 35)

    if a + b + c + d != 5:
        return 0
    if a != 1:          # A区固定1个
        return 0

    if cold_mode:
        # ─── 追冷模式 ───
        # Priority 2: C区偏多的冷号分布
        if c >= 2:
            # ABBCC: B=2,C=2,D=0
            if b == 2 and c == 2 and d == 0:
                return 2
            # ABCCC: B=1,C=3,D=0
            if b == 1 and c == 3 and d == 0:
                return 2
            # ACCCD: B=0,C=3,D=1
            if b == 0 and c == 3 and d == 1:
                return 2
            # ABCCD: B=1,C=2,D=1 (变体)
            if b == 1 and c == 2 and d == 1:
                return 2
        # Priority 1: 标准BBB+模式
        if b == 3:
            b_vals = [n for n in main_vals if 10 <= n <= 20]
            if max(b_vals) - min(b_vals) <= 5:
                if (c == 1 and d == 0) or (c == 0 and d == 1):
                    return 1  # ABBBC 或 ABBBD
        # 也接受标准BBC
        if b == 2 and c == 1 and d == 1:
            return 1
        return 0

    else:
        # ─── 通用模式 ───
        # Priority 2: BBB 模式（B=3, 相近）
        if b == 3:
            b_vals = [n for n in main_vals if 10 <= n <= 20]
            if max(b_vals) - min(b_vals) <= 5:
                if (c == 1 and d == 0) or (c == 0 and d == 1):
                    return 2  # ABBBC ☆ 或 ABBBD ☆
        # Priority 1: BBC 模式
        if b == 2 and c == 1 and d == 1:
            return 1  # ABBCD ✓
        # B=3但范围不满足相近条件的，降低优先级
        if b == 3:
            if (c == 1 and d == 0) or (c == 0 and d == 1):
                return 1  # B区跨度>5也算可接受但不是最优

        return 0


def _crf_score(main_vals, sub_vals, main_probs, sub_probs, cfg):
    """CRF-like global combination scoring.
    
    Not just sum of individual probabilities — evaluates the FULL combination
    for structural coherence. This is the key insight from CRF decoding:
    the best individual numbers don't necessarily make the best combination.
    
    Scoring factors:
    1. Base: sum of individual number probabilities from the strategy's model
    2. Consecutive penalty: too many consecutive numbers = unnatural
    3. Sum fitness: reward sums in the typical range
    4. Odd/even balance: reward roughly balanced parity
    5. Span fitness: reward reasonable spread across the number range
    6. Section distribution: reward even spread across low/mid/high sections
    """
    m = sorted(main_vals)
    s = sorted(sub_vals)
    count = len(m)
    
    # 1. Base probability score
    try:
        main_range = list(range(cfg.main_min, cfg.main_max + 1))
        sub_range = list(range(cfg.sub_min, cfg.sub_max + 1))
        base = sum(main_probs[main_range.index(n)] for n in m)
        base += sum(sub_probs[sub_range.index(n)] for n in s)
    except (ValueError, IndexError):
        base = 0.0
    
    # 2. Consecutive penalty
    consec_streaks = 1
    max_streak = 1
    for i in range(1, len(m)):
        if m[i] == m[i-1] + 1:
            consec_streaks += 1
            max_streak = max(max_streak, consec_streaks)
        else:
            consec_streaks = 1
    # 2 in a row is OK, 3+ is penalized progressively
    consec_ok = max_streak - 1  # 0 for no consec, 1 for pair, 2 for triple...
    consec_penalty = max(0, consec_ok - 1) * 0.04  # only penalize triple+
    
    # 3. Sum fitness
    total = sum(m)
    # Expected sum range differs by game
    mid_point = (cfg.main_min + cfg.main_max) / 2
    expected_sum = mid_point * count
    sum_dev = abs(total - expected_sum) / expected_sum
    sum_penalty = sum_dev * 0.03
    
    # 4. Odd/even balance
    odd = sum(1 for v in m if v % 2 == 1)
    expected_odd = count / 2
    odd_dev = abs(odd - expected_odd) / expected_odd if expected_odd > 0 else 0
    odd_penalty = min(odd_dev * 0.03, 0.03)  # cap at 0.03
    
    # 5. Span fitness
    span = m[-1] - m[0]
    range_size = cfg.main_max - cfg.main_min
    expected_span = range_size * 0.6
    span_dev = abs(span - expected_span) / expected_span if expected_span > 0 else 0
    span_penalty = span_dev * 0.02
    
    # 6. Section distribution: divide range into 3 sections, penalize clustering
    section_size = (cfg.main_max - cfg.main_min + 1) / 3
    sections = [0, 0, 0]
    for v in m:
        s_idx = min(2, int((v - cfg.main_min) / section_size))
        sections[s_idx] += 1
    # Ideal: roughly count/3 per section
    section_cluster = max(sections) - min(sections)
    section_penalty = (section_cluster / count) * 0.02
    
    final = base - consec_penalty - sum_penalty - odd_penalty - span_penalty - section_penalty
    return final


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
    """Generate recommendations using 5 strategies + CRF global scoring.
    
    Each strategy:
    1. Builds its own fused probability distribution (Bayesian fusion of relevant signals)
    2. Generates candidate combinations via weighted sampling
    3. Scores each combination using CRF-like global scoring (structural coherence)
    4. Picks the best combination
    
    Strategies:
    1. 🔥高频热号 — Decay-weighted frequency (recent > old)
    2. 📈近期趋势 — Sliding window + trend boost
    3. ❄️追冷 — Cold numbers with gap recency signal
    4. 🔗马尔可夫 — Markov transition + neighborhood
    5. ⚖️平衡 — Mid-frequency with structural balance
    """
    logger = get_logger(cfg)
    logger.info("CRF-enhanced 5-strategy generation for %s ...", cfg.name)
    
    if df is None:
        logger.error("df is required")
        return {"groups": [], "hot_numbers": [], "cold_numbers": [],
                "model_weights": {}, "timestamp": str(datetime.now()), "candidates_evaluated": 0}
    
    # Compute ensemble probabilities (also saves individual model probs)
    main_probs, sub_probs = ensemble._get_model_probs(df)
    
    main_range = list(range(cfg.main_min, cfg.main_max + 1))
    sub_range = list(range(cfg.sub_min, cfg.sub_max + 1))
    main_n = len(main_range)
    sub_n = len(sub_range)
    
    # Precompute helper data
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
    
    w50 = min(50, len(df))
    main_win_counts = np.zeros(main_n)
    sub_win_counts = np.zeros(sub_n)
    for row in main_nums[:w50]:
        for n in row:
            main_win_counts[n - cfg2.main_min] += 1
    for row in sub_nums[:w50]:
        for n in row:
            sub_win_counts[n - cfg2.sub_min] += 1
    
    epsilon = 1e-6
    main_hist_share = main_counts / (main_counts.sum() + epsilon)
    main_recent_share = main_win_counts / (main_win_counts.sum() + epsilon)
    sub_hist_share = sub_counts / (sub_counts.sum() + epsilon)
    sub_recent_share = sub_win_counts / (sub_win_counts.sum() + epsilon)
    
    # DLT 区间分布过滤器（仅对大乐透生效，双色球保持原样）
    is_dlt = (cfg.short == "dlt")
    dlt_filter = _check_dlt_zone if is_dlt else None
    dlt_cold_filter = (lambda v: _check_dlt_zone(v, cold_mode=True)) if is_dlt else None
    dlt_candidates = 1200 if is_dlt else 500  # DLT需要更多候选以找到符合区间分布的

    def _run_strategy(strategy_probs_m, strategy_probs_s, rng_seed, n_candidates=500, pattern_filter=None):
        """Run one strategy: generate candidates, score with CRF, pick best.
        
        If pattern_filter is provided, only candidates passing the filter are scored.
        """
        rng = random.Random(rng_seed)
        best_combo = None
        best_score = -999
        # DLT multi-pass: track best per priority level
        best_by_priority = {}  # priority -> (score, combo)
        
        for _ in range(n_candidates):
            m_ids = _weighted_sample(strategy_probs_m, cfg.main_count, rng)
            s_ids = _weighted_sample(strategy_probs_s, cfg.sub_count, rng)
            m_vals = [main_range[i] for i in m_ids]
            s_vals = [sub_range[i] for i in s_ids]
            
            # Pattern filter with priority scoring (for DLT)
            if pattern_filter is not None:
                pri = pattern_filter(m_vals) if callable(pattern_filter) else (1 if pattern_filter(m_vals) else 0)
                if pri == 0:
                    continue
                sc = _crf_score(m_vals, s_vals, strategy_probs_m, strategy_probs_s, cfg)
                # Track best per priority level
                if pri not in best_by_priority or sc > best_by_priority[pri][0]:
                    best_by_priority[pri] = (sc, (m_ids, s_ids))
                if sc > best_score:
                    best_score = sc
                    best_combo = (m_ids, s_ids)
            else:
                sc = _crf_score(m_vals, s_vals, strategy_probs_m, strategy_probs_s, cfg)
                if sc > best_score:
                    best_score = sc
                    best_combo = (m_ids, s_ids)
        
        # For DLT: prefer highest priority, fall back to lower
        if pattern_filter is not None and best_by_priority:
            for pri in sorted(best_by_priority.keys(), reverse=True):
                sc, combo = best_by_priority[pri]
                best_combo = combo
                if pri >= 2:  # BBB found! prefer this
                    break
        
        # If no valid combo found after filtering, try without filter
        if best_combo is None and pattern_filter is not None:
            logger.warning("未找到符合区间分布的组合，放宽约束重试")
            return _run_strategy(strategy_probs_m, strategy_probs_s, rng_seed, n_candidates * 2, pattern_filter=None)
        elif best_combo is None:
            # Shouldn't happen, but fallback
            m_ids = _weighted_sample(strategy_probs_m, cfg.main_count, rng)
            s_ids = _weighted_sample(strategy_probs_s, cfg.sub_count, rng)
            best_combo = (m_ids, s_ids)
        
        m_ids, s_ids = best_combo
        return sorted([main_range[i] for i in m_ids]), sorted([sub_range[i] for i in s_ids])
    
    # ========================================================================
    # Strategy 1: 🔥高频热号 (Decay-weighted)
    #    Uses decay_frequency + gap_recency via Bayesian fusion.
    #    CRF scoring ensures the combination is structurally sound, not just
    #    a collection of the hottest individual numbers.
    # ========================================================================
    m1_dict = {
        'decay': ensemble._model_main_probs['decay_frequency'],
        'gap': ensemble._model_main_probs['gap_recency'],
    }
    s1_dict = {
        'decay': ensemble._model_sub_probs['decay_frequency'],
        'gap': ensemble._model_sub_probs['gap_recency'],
    }
    w1 = {'decay': 0.7, 'gap': 0.3}
    m1_fused = _bayesian_fusion(m1_dict, w1, main_n)
    s1_fused = _bayesian_fusion(s1_dict, w1, sub_n)
    
    main_1, sub_1 = _run_strategy(m1_fused, s1_fused, 101, n_candidates=dlt_candidates, pattern_filter=dlt_filter)
    reason_1 = "高频热号: 衰减加权频率+间隔分析贝叶斯融合，CRF全局评分优选"
    
    # ========================================================================
    # Strategy 2: 📈近期趋势
    #    Sliding window + trend boost (recent share - historical share).
    #    Numbers with rising frequency get priority.
    # ========================================================================
    m2_raw = ensemble._model_main_probs['sliding_window']
    s2_raw = ensemble._model_sub_probs['sliding_window']
    
    main_trend_boost = np.maximum(0, main_recent_share - main_hist_share)
    m2_boosted = m2_raw + main_trend_boost * 0.4
    
    sub_trend_boost = np.maximum(0, sub_recent_share - sub_hist_share)
    s2_boosted = s2_raw + sub_trend_boost * 0.4
    
    m2_fused = _bayesian_fusion(
        {'window_boost': m2_boosted, 'decay': ensemble._model_main_probs['decay_frequency']},
        {'window_boost': 0.65, 'decay': 0.35}, main_n)
    s2_fused = _bayesian_fusion(
        {'window_boost': s2_boosted, 'decay': ensemble._model_sub_probs['decay_frequency']},
        {'window_boost': 0.65, 'decay': 0.35}, sub_n)
    
    main_2, sub_2 = _run_strategy(m2_fused, s2_fused, 202, n_candidates=dlt_candidates, pattern_filter=dlt_filter)
    reason_2 = "近期趋势: 滑动窗口+趋势涨幅+衰减频率贝叶斯融合"
    
    # ========================================================================
    # Strategy 3: ❄️追冷
    #    Cold numbers (bottom 40% by frequency) with gap recency signal.
    #    Gap recency measures "unusually long absence" better than raw Poisson.
    # ========================================================================
    main_freq_rank = np.argsort(main_counts)
    sub_freq_rank = np.argsort(sub_counts)
    cold_th_m = max(1, int(main_n * 0.4))
    cold_th_s = max(1, int(sub_n * 0.4))
    cold_m = set(main_freq_rank[:cold_th_m])
    cold_s = set(sub_freq_rank[:cold_th_s])
    
    m3_gap = ensemble._model_main_probs['gap_recency']
    s3_gap = ensemble._model_sub_probs['gap_recency']
    
    # Cold strategy: emphasize cold numbers, boost ones with high gap scores
    m3_strat = np.zeros(main_n)
    for i in range(main_n):
        if i in cold_m:
            m3_strat[i] = m3_gap[i] * 0.7 + main_recent_share[i] * 0.3
        else:
            m3_strat[i] = m3_gap[i] * 0.15
    m3_strat = m3_strat / m3_strat.sum()
    
    s3_strat = np.zeros(sub_n)
    for i in range(sub_n):
        if i in cold_s:
            s3_strat[i] = s3_gap[i] * 0.7 + sub_recent_share[i] * 0.3
        else:
            s3_strat[i] = s3_gap[i] * 0.15
    s3_strat = s3_strat / s3_strat.sum()
    
    main_3, sub_3 = _run_strategy(m3_strat, s3_strat, 303, n_candidates=dlt_candidates, pattern_filter=dlt_cold_filter)
    reason_3 = "追冷: 冷号中回补信号最强的号码(间隔异常度+近期升温)"
    
    # ========================================================================
    # Strategy 4: 🔗马尔可夫
    #    Markov transition + neighborhood of last draw's numbers.
    #    Boost numbers in ±2 range of last draw's winners.
    # ========================================================================
    m4_raw = ensemble._model_main_probs['markov_chain']
    s4_raw = ensemble._model_sub_probs['markov_chain']
    
    m4_boosted = np.array(m4_raw)
    s4_boosted = np.array(s4_raw)
    
    if len(main_nums) > 0:
        last_main_vals = main_nums[0]
        for n in last_main_vals:
            idx = n - cfg.main_min
            m4_boosted[idx] *= 1.2  # repeat boost
            for offset in range(-2, 3):
                neighbor = n + offset
                if cfg.main_min <= neighbor <= cfg.main_max:
                    m4_boosted[neighbor - cfg.main_min] *= 1.15
        m4_boosted = m4_boosted / m4_boosted.sum()
    
    if len(sub_nums) > 0:
        for n in sub_nums[0]:
            s4_boosted[n - cfg.sub_min] *= 1.3
        s4_boosted = s4_boosted / s4_boosted.sum()
    
    # Bayesian fusion with decay frequency for stability
    m4_fused = _bayesian_fusion(
        {'markov': m4_boosted, 'decay': ensemble._model_main_probs['decay_frequency']},
        {'markov': 0.7, 'decay': 0.3}, main_n)
    s4_fused = _bayesian_fusion(
        {'markov': s4_boosted, 'decay': ensemble._model_sub_probs['decay_frequency']},
        {'markov': 0.7, 'decay': 0.3}, sub_n)
    
    main_4, sub_4 = _run_strategy(m4_fused, s4_fused, 404, n_candidates=dlt_candidates, pattern_filter=dlt_filter)
    reason_4 = "马尔可夫: 转移概率+上期邻域增强+衰减频率贝叶斯融合"
    
    # ========================================================================
    # Strategy 5: ⚖️平衡
    #    Exclude extreme hot (top 30%) and extreme cold (bottom 30%).
    #    Use CRF scoring heavily weighted toward structural coherence.
    #    This strategy's primary goal is STRUCTURAL SOUNDNESS, not hot/cold.
    # ========================================================================
    m5_raw = ensemble._model_main_probs['decay_frequency']
    s5_raw = ensemble._model_sub_probs['decay_frequency']
    main_ranked = np.argsort(m5_raw)[::-1]
    sub_ranked = np.argsort(s5_raw)[::-1]
    
    main_mid_start = int(main_n * 0.3)
    main_mid_end = int(main_n * 0.7)
    sub_mid_start = int(sub_n * 0.25)
    sub_mid_end = int(sub_n * 0.75)
    
    # Build mid-zone preference
    m5_scores = np.ones(main_n) * 0.2
    for i in main_ranked[main_mid_start:main_mid_end]:
        m5_scores[i] = 1.0
    for i in list(main_ranked[int(main_n*0.2):int(main_n*0.3)]) + list(main_ranked[int(main_n*0.7):int(main_n*0.8)]):
        m5_scores[i] = 0.5
    m5_scores = m5_scores / m5_scores.sum()
    
    s5_scores = np.ones(sub_n) * 0.2
    for i in sub_ranked[sub_mid_start:sub_mid_end]:
        s5_scores[i] = 1.0
    s5_scores = s5_scores / s5_scores.sum()
    
    # For balanced strategy: run CRF scoring with EXTRA structural weight
    rng5 = random.Random(505)
    best_main5 = None
    best_sub5 = None
    best_score5 = -999
    # DLT: track best per priority level
    s5_best_by_pri = {}  # priority -> (score, main_vals, sub_vals)
    s5_n_candidates = dlt_candidates if is_dlt else 500
    for _ in range(s5_n_candidates):
        m_ids = _weighted_sample(m5_scores, cfg.main_count, rng5)
        s_ids = _weighted_sample(s5_scores, cfg.sub_count, rng5)
        m_vals = [main_range[i] for i in m_ids]
        s_vals = [sub_range[i] for i in s_ids]

        # DLT 区间分布优先过滤
        if is_dlt:
            pri = _check_dlt_zone(m_vals)
            if pri == 0:
                continue
            sc = _crf_score(m_vals, s_vals, m5_scores, s5_scores, cfg)
            # Track best per priority
            if pri not in s5_best_by_pri or sc > s5_best_by_pri[pri][0]:
                s5_best_by_pri[pri] = (sc, m_vals, s_vals)
            if sc > best_score5:
                best_score5 = sc
                best_main5 = m_vals
                best_sub5 = s_vals
        else:
            sc = _crf_score(m_vals, s_vals, m5_scores, s5_scores, cfg)
            if sc > best_score5:
                best_score5 = sc
                best_main5 = m_vals
                best_sub5 = s_vals

    # For DLT: prefer highest priority (BBB=2 > BBC=1)
    if is_dlt and s5_best_by_pri:
        for pri in sorted(s5_best_by_pri.keys(), reverse=True):
            sc, m_vals, s_vals = s5_best_by_pri[pri]
            best_main5, best_sub5 = m_vals, s_vals
            if pri >= 2:  # BBB found! prefer this
                break

    # If no valid combo found for DLT, relax filter
    if is_dlt and not s5_best_by_pri:
        logger.warning("大乐透平衡策略未找到符合区间分布的组合，放宽约束")
        for _ in range(1000):
            m_ids = _weighted_sample(m5_scores, cfg.main_count, rng5)
            s_ids = _weighted_sample(s5_scores, cfg.sub_count, rng5)
            m_vals = [main_range[i] for i in m_ids]
            s_vals = [sub_range[i] for i in s_ids]
            sc = _crf_score(m_vals, s_vals, m5_scores, s5_scores, cfg)
            if sc > best_score5:
                best_score5 = sc
                best_main5 = m_vals
                best_sub5 = s_vals
    
    main_5 = sorted(best_main5)
    sub_5 = sorted(best_sub5)
    reason_5 = "平衡策略: 避开极端冷热号，CRF结构评分选最优组合"
    
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
    
    logger.info("CRF-enhanced: %d groups for %s", len(groups), cfg.name)
    return {
        "groups": groups,
        "hot_numbers": [n for n, _ in main_ranked[:5]],
        "cold_numbers": [n for n, _ in main_ranked[-3:]],
        "model_weights": {},
        "timestamp": str(datetime.now()),
        "candidates_evaluated": len(groups),
    }
