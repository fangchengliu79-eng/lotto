"""
Statistical analysis module - fully parameterized by cfg.

Provides functions for extracting draw numbers and performing
comprehensive statistical analysis on lottery draw data.
Works for any lottery type (DLT, SSQ, etc.) via cfg parameterization.
"""
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from utils.helpers import get_logger, parse_draw_row


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_main_numbers(df: pd.DataFrame, cfg) -> np.ndarray:
    """Extract main (front/red) ball numbers from DataFrame using cfg.main_cols.

    Parameters
    ----------
    df  : DataFrame with columns matching cfg.main_cols.
    cfg : LotteryConfig instance.

    Returns
    -------
    2D numpy array of shape (n_draws, cfg.main_count).
    """
    cols = [c for c in cfg.main_cols if c in df.columns]
    if not cols:
        raise ValueError(
            f"None of cfg.main_cols {cfg.main_cols} found in DataFrame columns {list(df.columns)}"
        )
    return df[cols].to_numpy(dtype=int)


def extract_sub_numbers(df: pd.DataFrame, cfg) -> np.ndarray:
    """Extract sub (back/blue) ball numbers from DataFrame using cfg.sub_cols.

    Parameters
    ----------
    df  : DataFrame with columns matching cfg.sub_cols.
    cfg : LotteryConfig instance.

    Returns
    -------
    2D numpy array of shape (n_draws, cfg.sub_count).
    """
    cols = [c for c in cfg.sub_cols if c in df.columns]
    if not cols:
        raise ValueError(
            f"None of cfg.sub_cols {cfg.sub_cols} found in DataFrame columns {list(df.columns)}"
        )
    return df[cols].to_numpy(dtype=int)


# ---------------------------------------------------------------------------
# Frequency analysis
# ---------------------------------------------------------------------------

def frequency_analysis(
    main_nums: np.ndarray,
    sub_nums: np.ndarray,
    cfg,
) -> Dict[str, Any]:
    """Compute frequency statistics for main and sub numbers.

    Parameters
    ----------
    main_nums : 2D array of main numbers, shape (n_draws, main_count).
    sub_nums  : 2D array of sub numbers, shape (n_draws, sub_count).
    cfg       : LotteryConfig instance.

    Returns
    -------
    dict with keys:
        main_frequencies  : dict {number: count}
        sub_frequencies   : dict {number: count}
        main_frequency_pct: dict {number: percentage}
        sub_frequency_pct : dict {number: percentage}
        total_draws       : int
    """
    total_draws = len(main_nums)

    # Flatten and count
    main_flat = main_nums.flatten()
    sub_flat = sub_nums.flatten()

    main_counter = Counter(main_flat.tolist())
    sub_counter = Counter(sub_flat.tolist())

    # Ensure all numbers in range are represented
    main_freq = {}
    for n in range(cfg.main_min, cfg.main_max + 1):
        main_freq[n] = main_counter.get(n, 0)

    sub_freq = {}
    for n in range(cfg.sub_min, cfg.sub_max + 1):
        sub_freq[n] = sub_counter.get(n, 0)

    # Percentages
    main_pct = {n: round(cnt / max(total_draws, 1) * 100, 2) for n, cnt in main_freq.items()}
    sub_pct = {n: round(cnt / max(total_draws, 1) * 100, 2) for n, cnt in sub_freq.items()}

    return {
        "main_frequencies": main_freq,
        "sub_frequencies": sub_freq,
        "main_frequency_pct": main_pct,
        "sub_frequency_pct": sub_pct,
        "total_draws": total_draws,
    }


# ---------------------------------------------------------------------------
# Hot / Cold numbers
# ---------------------------------------------------------------------------

def hot_cold_analysis(
    main_nums: np.ndarray,
    sub_nums: np.ndarray,
    cfg,
    hot_threshold: Optional[float] = None,
    cold_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Classify numbers as hot, warm, or cold based on appearance frequency.

    Parameters
    ----------
    main_nums        : 2D array of main numbers.
    sub_nums         : 2D array of sub numbers.
    cfg              : LotteryConfig instance.
    hot_threshold    : frequency fraction above which a number is "hot"
                       (default: 1.5 / expected_frequency).
    cold_threshold   : frequency fraction below which a number is "cold"
                       (default: 0.5 / expected_frequency).

    Returns
    -------
    dict with keys: main_hot, main_warm, main_cold, sub_hot, sub_warm, sub_cold,
                    main_freq_pct, sub_freq_pct
    """
    total_draws = len(main_nums)
    if total_draws == 0:
        return _empty_hot_cold(cfg)

    # Expected frequency per number if uniform
    main_expected = total_draws * cfg.main_count / (cfg.main_max - cfg.main_min + 1)
    sub_expected = total_draws * cfg.sub_count / (cfg.sub_max - cfg.sub_min + 1)

    hot_th = hot_threshold if hot_threshold is not None else 1.5 / main_expected
    cold_th = cold_threshold if cold_threshold is not None else 0.5 / main_expected

    freq = frequency_analysis(main_nums, sub_nums, cfg)

    main_pct = freq["main_frequency_pct"]
    sub_pct = freq["sub_frequency_pct"]

    result = {
        "main_hot": sorted([n for n, p in main_pct.items() if p >= hot_th * 100]),
        "main_warm": sorted([
            n for n, p in main_pct.items() if cold_th * 100 < p < hot_th * 100
        ]),
        "main_cold": sorted([n for n, p in main_pct.items() if p <= cold_th * 100]),
        "sub_hot": sorted([n for n, p in sub_pct.items() if p >= hot_th * 100]),
        "sub_warm": sorted([
            n for n, p in sub_pct.items() if cold_th * 100 < p < hot_th * 100
        ]),
        "sub_cold": sorted([n for n, p in sub_pct.items() if p <= cold_th * 100]),
        "main_freq_pct": main_pct,
        "sub_freq_pct": sub_pct,
    }
    return result


def _empty_hot_cold(cfg) -> Dict[str, Any]:
    """Return empty hot/cold structure."""
    return {
        "main_hot": [], "main_warm": [], "main_cold": list(range(cfg.main_min, cfg.main_max + 1)),
        "sub_hot": [], "sub_warm": [], "sub_cold": list(range(cfg.sub_min, cfg.sub_max + 1)),
        "main_freq_pct": {}, "sub_freq_pct": {},
    }


# ---------------------------------------------------------------------------
# Odd / Even analysis
# ---------------------------------------------------------------------------

def odd_even_analysis(main_nums: np.ndarray, sub_nums: np.ndarray, cfg) -> Dict[str, Any]:
    """Analyze odd/even distribution of main and sub numbers.

    Parameters
    ----------
    main_nums : 2D array of main numbers.
    sub_nums  : 2D array of sub numbers.
    cfg       : LotteryConfig instance.

    Returns
    -------
    dict with keys:
        main_odd_even_ratio: list of (odd_count, even_count) per draw
        sub_odd_even_ratio : list of (odd_count, even_count) per draw
        main_odd_even_stats: {odd_mean, even_mean, odd_min, odd_max, ...}
        sub_odd_even_stats : similar
        main_most_common   : most common (odd, even) pair for main
        sub_most_common    : most common (odd, even) pair for sub
    """
    main_odd_counts = np.sum(main_nums % 2 == 1, axis=1)
    main_even_counts = cfg.main_count - main_odd_counts

    sub_odd_counts = np.sum(sub_nums % 2 == 1, axis=1)
    sub_even_counts = cfg.sub_count - sub_odd_counts

    main_ratios = list(zip(main_odd_counts.tolist(), main_even_counts.tolist()))
    sub_ratios = list(zip(sub_odd_counts.tolist(), sub_even_counts.tolist()))

    main_common = Counter(main_ratios).most_common(1)
    sub_common = Counter(sub_ratios).most_common(1)

    return {
        "main_odd_even_ratio": main_ratios,
        "sub_odd_even_ratio": sub_ratios,
        "main_odd_even_stats": {
            "odd_mean": float(np.mean(main_odd_counts)),
            "even_mean": float(np.mean(main_even_counts)),
            "odd_min": int(np.min(main_odd_counts)),
            "odd_max": int(np.max(main_odd_counts)),
            "even_min": int(np.min(main_even_counts)),
            "even_max": int(np.max(main_even_counts)),
        },
        "sub_odd_even_stats": {
            "odd_mean": float(np.mean(sub_odd_counts)),
            "even_mean": float(np.mean(sub_even_counts)),
            "odd_min": int(np.min(sub_odd_counts)),
            "odd_max": int(np.max(sub_odd_counts)),
            "even_min": int(np.min(sub_even_counts)),
            "even_max": int(np.max(sub_even_counts)),
        },
        "main_most_common": list(main_common[0][0]) if main_common else None,
        "sub_most_common": list(sub_common[0][0]) if sub_common else None,
    }


# ---------------------------------------------------------------------------
# Sum analysis
# ---------------------------------------------------------------------------

def sum_analysis(main_nums: np.ndarray, sub_nums: np.ndarray, cfg) -> Dict[str, Any]:
    """Analyze the sum of numbers per draw.

    Parameters
    ----------
    main_nums : 2D array of main numbers.
    sub_nums  : 2D array of sub numbers.
    cfg       : LotteryConfig instance.

    Returns
    -------
    dict with keys:
        main_sums: list of per-draw main sums
        sub_sums : list of per-draw sub sums
        total_sums: list of per-draw total sums
        main_sum_stats: {mean, std, min, max, median}
        sub_sum_stats : {mean, std, min, max, median}
        total_sum_stats: {mean, std, min, max, median}
    """
    main_sums = np.sum(main_nums, axis=1)
    sub_sums = np.sum(sub_nums, axis=1)
    total_sums = main_sums + sub_sums

    def _stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "median": float(np.median(arr)),
        }

    return {
        "main_sums": main_sums.tolist(),
        "sub_sums": sub_sums.tolist(),
        "total_sums": total_sums.tolist(),
        "main_sum_stats": _stats(main_sums),
        "sub_sum_stats": _stats(sub_sums),
        "total_sum_stats": _stats(total_sums),
    }


# ---------------------------------------------------------------------------
# Span analysis (max - min)
# ---------------------------------------------------------------------------

def span_analysis(main_nums: np.ndarray, sub_nums: np.ndarray, cfg) -> Dict[str, Any]:
    """Analyze the span (max - min) of numbers per draw.

    Parameters
    ----------
    main_nums : 2D array of main numbers.
    sub_nums  : 2D array of sub numbers.
    cfg       : LotteryConfig instance.

    Returns
    -------
    dict with keys:
        main_spans: list of per-draw main spans
        sub_spans : list of per-draw sub spans
        main_span_stats: {mean, std, min, max, median}
        sub_span_stats : {mean, std, min, max, median}
    """
    main_spans = np.max(main_nums, axis=1) - np.min(main_nums, axis=1)
    sub_spans = np.max(sub_nums, axis=1) - np.min(sub_nums, axis=1)

    def _stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "median": float(np.median(arr)),
        }

    return {
        "main_spans": main_spans.tolist(),
        "sub_spans": sub_spans.tolist(),
        "main_span_stats": _stats(main_spans),
        "sub_span_stats": _stats(sub_spans),
    }


# ---------------------------------------------------------------------------
# AC value (complexity value)
# ---------------------------------------------------------------------------

def _ac_value(numbers: np.ndarray) -> int:
    """Compute the AC value (complexity/abstraction coefficient) for a set of numbers.

    AC value = number of unique positive differences between sorted numbers
              minus (number_count - 1).
    Reference: Common in Chinese lottery analysis.
    """
    n = len(numbers)
    if n <= 1:
        return 0
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(abs(numbers[i] - numbers[j]))
    return len(diffs) - (n - 1)


def ac_value_analysis(main_nums: np.ndarray, sub_nums: np.ndarray, cfg) -> Dict[str, Any]:
    """Analyze AC values of each draw.

    Parameters
    ----------
    main_nums : 2D array of main numbers.
    sub_nums  : 2D array of sub numbers.
    cfg       : LotteryConfig instance.

    Returns
    -------
    dict with keys:
        main_ac_values: list of per-draw main AC values
        main_ac_stats : {mean, std, min, max, median, most_common}
    """
    main_ac = np.array([_ac_value(sorted(row)) for row in main_nums])

    def _stats(arr):
        c = Counter(arr.tolist())
        most_common = c.most_common(3) if c else []
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "median": float(np.median(arr)),
            "most_common": most_common,
        }

    return {
        "main_ac_values": main_ac.tolist(),
        "main_ac_stats": _stats(main_ac),
    }


# ---------------------------------------------------------------------------
# Consecutive number analysis
# ---------------------------------------------------------------------------

def consecutive_analysis(main_nums: np.ndarray, sub_nums: np.ndarray, cfg) -> Dict[str, Any]:
    """Analyze consecutive numbers in each draw.

    Parameters
    ----------
    main_nums : 2D array of main numbers.
    sub_nums  : 2D array of sub numbers.
    cfg       : LotteryConfig instance.

    Returns
    -------
    dict with keys:
        main_consecutive_counts: number of consecutive pairs per draw for main
        sub_consecutive_counts : number of consecutive pairs per draw for sub
        main_has_consecutive   : boolean list for main
        sub_has_consecutive    : boolean list for sub
        main_consec_pct        : percentage of draws with consecutive main numbers
        sub_consec_pct         : percentage of draws with consecutive sub numbers
        most_common_main_consec: most common consecutive pair(s)
    """
    def _count_consecutive(arr_sorted):
        count = 0
        for i in range(len(arr_sorted) - 1):
            if arr_sorted[i + 1] - arr_sorted[i] == 1:
                count += 1
        return count

    main_consec = []
    main_pairs = Counter()
    for row in main_nums:
        sorted_row = sorted(row)
        consec_pairs = 0
        for i in range(len(sorted_row) - 1):
            if sorted_row[i + 1] - sorted_row[i] == 1:
                consec_pairs += 1
                main_pairs[(sorted_row[i], sorted_row[i + 1])] += 1
        main_consec.append(consec_pairs)

    sub_consec = [_count_consecutive(sorted(row)) for row in sub_nums]

    total = len(main_nums)
    return {
        "main_consecutive_counts": main_consec,
        "sub_consecutive_counts": sub_consec,
        "main_has_consecutive": [c > 0 for c in main_consec],
        "sub_has_consecutive": [c > 0 for c in sub_consec],
        "main_consec_pct": round(sum(c > 0 for c in main_consec) / max(total, 1) * 100, 2),
        "sub_consec_pct": round(sum(c > 0 for c in sub_consec) / max(total, 1) * 100, 2),
        "most_common_main_consec": main_pairs.most_common(5),
    }


# ---------------------------------------------------------------------------
# Comprehensive analysis
# ---------------------------------------------------------------------------

def comprehensive_analysis(df: pd.DataFrame, cfg) -> Dict[str, Any]:
    """Run a full statistical analysis on the draw data.

    Parameters
    ----------
    df  : DataFrame with columns per cfg.main_cols and cfg.sub_cols.
    cfg : LotteryConfig instance.

    Returns
    -------
    dict containing all analysis results:
        frequency  : output of frequency_analysis
        hot_cold   : output of hot_cold_analysis
        odd_even   : output of odd_even_analysis
        sum        : output of sum_analysis
        span       : output of span_analysis
        ac_value   : output of ac_value_analysis
        consecutive: output of consecutive_analysis
    """
    logger = get_logger(cfg)
    logger.info("Running comprehensive analysis for %s ...", cfg.name)

    main_nums = extract_main_numbers(df, cfg)
    sub_nums = extract_sub_numbers(df, cfg)

    result = {
        "total_draws": len(df),
        "period_range": {
            "first": str(df["period"].iloc[-1]) if "period" in df.columns and len(df) > 0 else "",
            "last": str(df["period"].iloc[0]) if "period" in df.columns and len(df) > 0 else "",
        },
        "frequency": frequency_analysis(main_nums, sub_nums, cfg),
        "hot_cold": hot_cold_analysis(main_nums, sub_nums, cfg),
        "odd_even": odd_even_analysis(main_nums, sub_nums, cfg),
        "sum": sum_analysis(main_nums, sub_nums, cfg),
        "span": span_analysis(main_nums, sub_nums, cfg),
        "ac_value": ac_value_analysis(main_nums, sub_nums, cfg),
        "consecutive": consecutive_analysis(main_nums, sub_nums, cfg),
    }

    logger.info("Analysis complete for %s (%d draws)", cfg.name, len(df))
    return result
