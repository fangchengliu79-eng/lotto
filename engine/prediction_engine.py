"""
Advanced Probabilistic Lottery Forecasting Platform - Core Engine.

Integrates:
- Walk-Forward Backtesting
- Adaptive Ensemble Weighting
- Genetic Algorithm Optimization
- Machine Learning Prediction Layer
- Performance Database
- Comprehensive Recommendation Generation

Architecture:
  Data Layer -> Feature Engineering -> ML Models -> Statistical Models
       -> Ensemble (adaptive weights) -> GA Optimizer -> Recommendations
       -> Backtesting -> Performance DB -> Weight Adaptation (feedback loop)
"""
import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd

from utils.helpers import get_logger
from engine.feature_engineering import build_features, compute_feature_importance
from engine.walk_forward import WalkForwardBacktester
from engine.genetic_algorithm import GAOptimizer
from engine.ml_predictor import MLPredictor
from engine.performance_db import PerformanceDB


class AdvancedPredictionEngine:
    """
    Core prediction engine that orchestrates all components.

    1. Fits all statistical models on historical data
    2. Trains ML models on engineered features
    3. Loads performance DB for adaptive weighting
    4. Runs GA optimization for best combinations
    5. Outputs comprehensive recommendations with all metadata
    """

    def __init__(self, cfg, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df.sort_values("period", ascending=False).reset_index(drop=True)
        self.logger = get_logger(cfg)

        # Data arrays (oldest first for ML training)
        self.main_nums = np.array([
            sorted([int(r[c]) for c in cfg.main_cols])
            for _, r in self.df.iterrows()
        ])
        self.sub_nums = np.array([
            sorted([int(r[c]) for c in cfg.sub_cols])
            for _, r in self.df.iterrows()
        ])
        # Chronological order (oldest first)
        self.main_nums_chrono = self.main_nums[::-1].copy()
        self.sub_nums_chrono = self.sub_nums[::-1].copy()

        self.n_draws = len(self.df)
        self.n_main = cfg.main_max - cfg.main_min + 1
        self.n_sub = cfg.sub_max - cfg.sub_min + 1
        self.main_range = list(range(cfg.main_min, cfg.main_max + 1))
        self.sub_range = list(range(cfg.sub_min, cfg.sub_max + 1))

        # Components
        self.performance_db = PerformanceDB(cfg)
        self.ml_predictor = MLPredictor(cfg)
        self.ga_optimizer = None
        self._feature_X = None
        self._feature_names = None

        # Model probability cache
        self._model_probs = {}  # model_name -> (main_probs, sub_probs)

        # Default ensemble weights
        self.base_weights = {
            "frequency": 0.18,
            "sliding_window": 0.14,
            "bayesian": 0.08,
            "poisson": 0.14,
            "markov_chain": 0.14,
            "exponential_smoothing": 0.12,
            "monte_carlo": 0.08,
            "ml_ensemble": 0.12,
        }

    def fit_all(self, train_ml: bool = True, verbose: bool = False):
        """
        Fit all models: statistical + ML.

        Parameters
        ----------
        train_ml : whether to train ML models (can be slow)
        verbose  : print progress
        """
        self.logger.info("Fitting all models for %s (%d draws)...", self.cfg.name, self.n_draws)

        if verbose:
            print(f"Fitting statistical models...")

        self._compute_statistical_probs()

        if train_ml and self.n_draws >= 100:
            if verbose:
                print(f"Building feature matrix...")
            X, y, fnames = build_features(self.df, self.cfg)
            if len(X) > 10:
                self._feature_X = X
                self._feature_names = fnames
                if verbose:
                    print(f"Training ML models on {len(X)} samples with {len(fnames)} features...")
                self.ml_predictor.fit(X, y, fnames)
                if verbose:
                    imp = self.ml_predictor.get_feature_importance(10)
                    if imp:
                        print("Top 10 features:")
                        for k, v in imp.items():
                            print(f"  {k}: {v:.1f}%")
            else:
                self.logger.warning("Not enough samples (%d) for ML training", len(X))
        elif train_ml:
            self.logger.warning("Need >=100 draws for ML, have %d", self.n_draws)

        # Compute ensemble with adaptive weights
        adaptive_w = self.performance_db.compute_adaptive_weights(self.base_weights)
        self._ensemble_probs = self._compute_ensemble_probs(adaptive_w)

        if verbose:
            print("\nAdaptive ensemble weights:")
            for mname, w in sorted(adaptive_w.items(), key=lambda x: -x[1]):
                print(f"  {mname}: {w:.1%}")

        self.logger.info("All models fitted")

    def _compute_statistical_probs(self):
        """Compute all statistical model probability distributions."""
        n_draws = self.n_draws
        m_idx = lambda n: n - self.cfg.main_min
        s_idx = lambda n: n - self.cfg.sub_min

        # Precompute presence arrays (chronological - oldest first)
        main_presence = np.zeros((n_draws, self.n_main), dtype=np.float64)
        sub_presence = np.zeros((n_draws, self.n_sub), dtype=np.float64)
        for t in range(n_draws):
            for n in self.main_nums_chrono[t]:
                main_presence[t, m_idx(n)] = 1.0
            for n in self.sub_nums_chrono[t]:
                sub_presence[t, s_idx(n)] = 1.0

        # Full counts
        main_counts = main_presence.sum(axis=0)
        sub_counts = sub_presence.sum(axis=0)

        # 1. FREQUENCY (full history)
        mf = (main_counts + 1) / (main_counts.sum() + self.n_main)
        sf = (sub_counts + 1) / (sub_counts.sum() + self.n_sub)
        self._model_probs["frequency"] = (mf, sf)

        # 2. SLIDING WINDOW (last 50)
        w50 = min(50, n_draws)
        mw = main_presence[-w50:].sum(axis=0)
        sw = sub_presence[-w50:].sum(axis=0)
        mw_p = (mw + 1) / (mw.sum() + self.n_main)
        sw_p = (sw + 1) / (sw.sum() + self.n_sub)
        self._model_probs["sliding_window"] = (mw_p, sw_p)

        # 3. BAYESIAN
        mb = (main_counts + 1) / (main_counts.sum() + self.n_main)
        sb = (sub_counts + 1) / (sub_counts.sum() + self.n_sub)
        self._model_probs["bayesian"] = (mb, sb)

        # 4. POISSON OVERDUE
        mp = np.ones(self.n_main) * 0.5
        sp = np.ones(self.n_sub) * 0.5
        for i in range(self.n_main):
            appearances = np.where(main_presence[:, i] > 0)[0]
            if len(appearances) > 0:
                last_seen = appearances[-1]
                gap = n_draws - 1 - last_seen
                lam = max(main_counts[i] / n_draws * n_draws, 0.5)
                surv = math.exp(-gap / lam) if lam > 0 else 0
                mp[i] = 1.0 - surv
        for i in range(self.n_sub):
            appearances = np.where(sub_presence[:, i] > 0)[0]
            if len(appearances) > 0:
                last_seen = appearances[-1]
                gap = n_draws - 1 - last_seen
                lam = max(sub_counts[i] / n_draws * n_draws, 0.5)
                surv = math.exp(-gap / lam) if lam > 0 else 0
                sp[i] = 1.0 - surv
        mp = mp / mp.sum()
        sp = sp / sp.sum()
        self._model_probs["poisson"] = (mp, sp)

        # 5. MARKOV CHAIN
        mm = np.ones(self.n_main) * 0.5
        sm = np.ones(self.n_sub) * 0.5
        for i in range(self.n_main):
            count_11 = count_10 = count_01 = 0
            for t in range(1, n_draws):
                prev = main_presence[t - 1, i] > 0
                curr = main_presence[t, i] > 0
                if prev and curr: count_11 += 1
                elif prev and not curr: count_10 += 1
                elif not prev and curr: count_01 += 1
            p_ga = count_11 / max(count_11 + count_10, 1)
            p_ab = count_01 / max(count_01 + (n_draws - count_11 - count_10 - count_10), 1)
            mm[i] = p_ga if main_presence[-1, i] > 0 else p_ab
        for i in range(self.n_sub):
            count_11 = count_10 = count_01 = 0
            for t in range(1, n_draws):
                prev = sub_presence[t - 1, i] > 0
                curr = sub_presence[t, i] > 0
                if prev and curr: count_11 += 1
                elif prev and not curr: count_10 += 1
                elif not prev and curr: count_01 += 1
            p_ga = count_11 / max(count_11 + count_10, 1)
            p_ab = count_01 / max(count_01 + (n_draws - count_11 - count_10 - count_10), 1)
            sm[i] = p_ga if sub_presence[-1, i] > 0 else p_ab
        mm = mm / mm.sum()
        sm = sm / sm.sum()
        self._model_probs["markov_chain"] = (mm, sm)

        # 6. EXPONENTIAL SMOOTHING
        mt = np.ones(self.n_main) * 0.5
        st_s = np.ones(self.n_sub) * 0.5
        alpha = 0.3
        for i in range(self.n_main):
            s = 0.5
            for t in range(n_draws):
                s = alpha * main_presence[t, i] + (1 - alpha) * s
            mt[i] = s
        for i in range(self.n_sub):
            s = 0.5
            for t in range(n_draws):
                s = alpha * sub_presence[t, i] + (1 - alpha) * s
            st_s[i] = s
        mt = mt / mt.sum()
        st_s = st_s / st_s.sum()
        self._model_probs["exponential_smoothing"] = (mt, st_s)

        # 7. ML PROBS (if fitted)
        if self.ml_predictor._is_fitted and self._feature_X is not None:
            last_X = self._feature_X[-1:].reshape(1, -1)
            ml_result = self.ml_predictor.predict_proba(last_X)
            self._model_probs["ml_ensemble"] = (ml_result["main_probs"], ml_result["sub_probs"])
            self._ml_confidence = ml_result["ml_confidence"]
        else:
            self._model_probs["ml_ensemble"] = (
                np.ones(self.n_main) / self.n_main,
                np.ones(self.n_sub) / self.n_sub,
            )
            self._ml_confidence = 0.0

    def _compute_ensemble_probs(self, weights: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute weighted ensemble probability distribution."""
        men = np.zeros(self.n_main)
        sen = np.zeros(self.n_sub)
        for mname, (mp_, sp_) in self._model_probs.items():
            w = weights.get(mname, self.base_weights.get(mname, 0.05))
            men += mp_ * w
            sen += sp_ * w
        men = men / men.sum()
        sen = sen / sen.sum()
        return men, sen

    # ====================================================================
    # RECOMMENDATION GENERATION
    # ====================================================================

    def generate_recommendations(
        self,
        num_groups: int = 5,
        use_ga: bool = True,
        ga_generations: int = 50,
        ga_population: int = 200,
        max_candidates: int = 5000,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive recommendations.

        Parameters
        ----------
        num_groups : number of recommendation groups
        use_ga     : use genetic algorithm optimization
        ga_generations : GA generations
        ga_population  : GA population size
        max_candidates : fallback candidate count

        Returns
        -------
        dict with:
            groups: [{index, main, sub, score, fitness, reason}]
            model_weights: current adaptive weights
            ml_confidence: ML model confidence
            ga_stats: GA evolution statistics
            feature_importance: top features
            performance_summary: model performance overview
            ensemble_breakdown: per-model contribution
        """
        self.logger.info("Generating %d recommendations...", num_groups)

        groups = []

        if use_ga:
            # ---- GENETIC ALGORITHM APPROACH ----
            ga = GAOptimizer(
                self.cfg,
                population_size=ga_population,
                generations=min(ga_generations, 30),  # cap to maintain diversity
                elite_ratio=0.08,  # lower elite ratio for more diversity
                crossover_rate=0.7,
                mutation_rate=0.2,  # higher mutation for diversity
            )
            self.ga_optimizer = ga

            prob_scores = {
                "main_probs": self._ensemble_probs[0],
                "sub_probs": self._ensemble_probs[1],
            }

            ga_result = ga.evolve(
                prob_scores=prob_scores,
                verbose=verbose,
            )

            # Select diverse groups from GA top candidates
            ga_candidates = ga_result["top_n"]
            
            # Use _select_diverse across GA candidates + weighted sampling for real diversity
            ga_combo_set = set()
            all_candidates = []
            for entry in ga_candidates:
                key = (tuple(entry["main"]), tuple(entry["sub"]))
                if key not in ga_combo_set:
                    ga_combo_set.add(key)
                    all_candidates.append({"main": entry["main"], "sub": entry["sub"]})
            
            # Supplement with weighted sampling if not enough
            if len(all_candidates) < num_groups * 3:
                extra = self._generate_candidates(3000)
                for c in extra:
                    key = (tuple(c["main"]), tuple(c["sub"]))
                    if key not in ga_combo_set:
                        ga_combo_set.add(key)
                        all_candidates.append(c)
            
            selected = self._select_diverse(all_candidates, num_groups)
            for i, cand in enumerate(selected):
                score_val = self._score_combination(cand["main"], cand["sub"])
                risk_val = self._compute_risk_score(cand["main"], cand["sub"])
                struct_val = self._compute_structure_score(cand["main"], cand["sub"])
                groups.append({
                    "index": i + 1,
                    "main": cand["main"],
                    "sub": cand["sub"],
                    "score": round(score_val, 2),
                    "fitness": round(sum(self._ensemble_probs[0][n - self.cfg.main_min] for n in cand["main"]) + sum(self._ensemble_probs[1][n - self.cfg.sub_min] for n in cand["sub"]), 2),
                    "risk_score": round(risk_val, 2),
                    "structure_score": round(struct_val, 2),
                    "source": "ga_diverse",
                })

            ga_stats = {
                "generations_run": ga_result["generations_run"],
                "final_diversity": ga_result["final_diversity"],
                "evolution_history": ga_result["evolution_history"][::5],  # sample every 5
                "best_fitness": ga_result["best_fitness"],
            }
        else:
            # ---- WEIGHTED SAMPLING APPROACH ----
            ga_stats = None
            candidates = self._generate_candidates(max_candidates)
            selected = self._select_diverse(candidates, num_groups)

            for i, cand in enumerate(selected):
                score = self._score_combination(cand["main"], cand["sub"])
                risk = self._compute_risk_score(cand["main"], cand["sub"])
                struct = self._compute_structure_score(cand["main"], cand["sub"])
                groups.append({
                    "index": i + 1,
                    "main": cand["main"],
                    "sub": cand["sub"],
                    "score": round(score, 2),
                    "fitness": round(score, 2),
                    "risk_score": round(risk, 2),
                    "structure_score": round(struct, 2),
                    "source": "weighted_sampling",
                })

        # Compute model contributions
        adaptive_w = self.performance_db.compute_adaptive_weights(self.base_weights)
        ensemble_breakdown = {}
        for mname, w in sorted(adaptive_w.items(), key=lambda x: -x[1]):
            if mname in self._model_probs:
                mp, sp = self._model_probs[mname]
                # Contribution = weight * avg probability score
                contribution = w * float(mp.mean() + sp.mean())
                ensemble_breakdown[mname] = {
                    "weight": round(w * 100, 1),
                    "contribution": round(contribution * 100, 2),
                    "mean_prob": float(mp.mean()),
                }

        # Compile result
        result = {
            "groups": groups,
            "model_weights": {k: round(v * 100, 1) for k, v in sorted(adaptive_w.items(), key=lambda x: -x[1])},
            "ensemble_breakdown": ensemble_breakdown,
            "ml_confidence": round(self._ml_confidence * 100, 1),
            "ga_stats": ga_stats,
            "feature_importance": self.ml_predictor.get_feature_importance(15),
            "performance_summary": self.performance_db.get_performance_summary(),
            "candidates_evaluated": len(groups),
            "timestamp": datetime.now().isoformat(),
        }

        return result

    def _generate_candidates(self, max_count: int) -> List[Dict]:
        """Generate candidate combinations by weighted sampling."""
        main_probs, sub_probs = self._ensemble_probs
        candidates = []
        used = set()
        rng = random.Random(42)

        for _ in range(max_count * 10):
            # Weighted sampling without replacement
            pool_m = list(self.main_range)
            w_m = list(main_probs)
            main_cand = []
            for _ in range(self.cfg.main_count):
                idx = rng.choices(range(len(pool_m)), weights=w_m, k=1)[0]
                main_cand.append(pool_m.pop(idx))
                w_m.pop(idx)
                if w_m:
                    w_m = [ww / sum(w_m) for ww in w_m]
            main_cand = sorted(main_cand)

            pool_s = list(self.sub_range)
            w_s = list(sub_probs)
            sub_cand = []
            for _ in range(self.cfg.sub_count):
                idx = rng.choices(range(len(pool_s)), weights=w_s, k=1)[0]
                sub_cand.append(pool_s.pop(idx))
                w_s.pop(idx)
                if w_s:
                    w_s = [ww / sum(w_s) for ww in w_s]
            sub_cand = sorted(sub_cand)

            key = (tuple(main_cand), tuple(sub_cand))
            if key in used:
                continue
            used.add(key)

            candidates.append({"main": main_cand, "sub": sub_cand})
            if len(candidates) >= max_count:
                break

        return candidates

    def _select_diverse(self, candidates: List[Dict], num_groups: int) -> List[Dict]:
        """Select diverse top candidates with per-number diversity."""
        if not candidates:
            return []
            
        # 先按概率评分排序
        scored = []
        for c in candidates:
            main_p = sum(self._ensemble_probs[0][n - self.cfg.main_min] for n in c["main"])
            sub_p = sum(self._ensemble_probs[1][n - self.cfg.sub_min] for n in c["sub"])
            score = main_p + sub_p
            scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        
        # 按sub号码分组：每个不同的sub组合只保留得分最高的一组
        best_by_sub = {}
        for score, c in scored:
            sub_key = tuple(sorted(c["sub"]))
            if sub_key not in best_by_sub:
                best_by_sub[sub_key] = (score, c)
        
        # 优先选sub号码不同的组合
        selected = []
        used_sub_nums = set()
        fallback_pool = []
        
        # 第一轮：选sub号码完全不重复的
        for sub_key, (score, c) in sorted(best_by_sub.items(), key=lambda x: -x[1][0]):
            if not used_sub_nums or not (set(c["sub"]) & used_sub_nums):
                selected.append(c)
                used_sub_nums.update(c["sub"])
                if len(selected) >= num_groups:
                    break
            else:
                fallback_pool.append((score, c))
        
        # 第二轮：如果不够，从剩余的里面选，允许少量重复
        if len(selected) < num_groups:
            for score, c in fallback_pool:
                if c not in selected:
                    selected.append(c)
                    if len(selected) >= num_groups:
                        break
        
        # 第三轮：如果还不到5组，从原始候选池补充
        if len(selected) < num_groups:
            for score, c in scored:
                if c not in selected:
                    selected.append(c)
                    if len(selected) >= num_groups:
                        break
        
        return selected

    def _score_combination(self, main: List[int], sub: List[int]) -> float:
        """Comprehensive scoring for a combination (0-100)."""
        mprobs, sprobs = self._ensemble_probs
        main_score = sum(mprobs[n - self.cfg.main_min] for n in main)
        sub_score = sum(sprobs[n - self.cfg.sub_min] for n in sub)

        expected_m = self.cfg.main_count / self.n_main
        expected_s = self.cfg.sub_count / max(self.n_sub, 1)

        main_ratio = main_score / max(expected_m, 0.001)
        sub_ratio = sub_score / max(expected_s, 0.001)

        combined = (main_ratio * self.cfg.main_count + sub_ratio * self.cfg.sub_count)
        combined /= (self.cfg.main_count + self.cfg.sub_count)

        return min(100, combined * 50)

    def _compute_risk_score(self, main: List[int], sub: List[int]) -> float:
        """Risk score: higher = riskier combination."""
        risk = 0.0

        # 1. Consecutive numbers = moderate risk
        consec = sum(1 for i in range(len(main) - 1) if main[i + 1] - main[i] == 1)
        risk += consec * 5

        # 2. All odd or all even = high risk
        odds = sum(1 for n in main if n % 2 == 1)
        if odds == 0 or odds == len(main):
            risk += 20

        # 3. Extreme sum = higher risk
        main_sum = sum(main)
        expected_sum = (self.cfg.main_min + self.cfg.main_max) / 2 * self.cfg.main_count
        sum_ratio = abs(main_sum - expected_sum) / expected_sum
        risk += sum_ratio * 15

        # 4. All hot or all cold = risk
        # (handled by diversity in fitness)

        return min(100, risk)

    def _compute_structure_score(self, main: List[int], sub: List[int]) -> float:
        """Structural quality score (0-100)."""
        score = 50.0

        # 1. AC value proximity to expected
        ac = self._ac_value(main)
        expected_ac = self.n_main * 0.65 - 5
        score += max(0, 15 - abs(ac - expected_ac) * 2)

        # 2. Span reasonableness
        span = max(main) - min(main)
        expected_span = self.cfg.main_max - self.cfg.main_min
        span_ratio = span / expected_span
        if 0.5 <= span_ratio <= 0.85:
            score += 15
        elif 0.3 <= span_ratio <= 0.95:
            score += 8

        # 3. Decade coverage
        decades = set(n // 10 for n in main)
        score += len(decades) * 5

        # 4. Gap uniformity
        s_main = sorted(main)
        gaps = [s_main[i + 1] - s_main[i] for i in range(len(s_main) - 1)]
        if gaps:
            cv = np.std(gaps) / max(np.mean(gaps), 0.1)
            score += max(0, 10 - cv * 3)

        return min(100, max(0, score))

    def _ac_value(self, nums: list) -> int:
        n = len(nums)
        if n <= 1:
            return 0
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(abs(nums[i] - nums[j]))
        return len(diffs) - (n - 1)

    # ====================================================================
    # BACKTESTING
    # ====================================================================

    def run_backtest(
        self,
        initial_train: int = 100,
        test_window: int = 50,
        step: int = 1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest and update performance DB.

        Returns comprehensive backtest statistics.
        """
        bt = WalkForwardBacktester(
            self.df, self.cfg,
            initial_train=initial_train,
            test_window=test_window,
            step=step,
        )
        stats = bt.run(verbose=verbose)

        # Sync backtest results to performance DB
        if "error" not in stats:
            for entry in bt.performance_db["by_period"]:
                self.performance_db.record_period(
                    entry["period"], entry["model_results"]
                )
            self.performance_db.save()

        return stats

    # ====================================================================
    # SINGLE NUMBER PROBABILITY REPORT
    # ====================================================================

    def get_number_probability_report(self) -> Dict[str, Any]:
        """Generate comprehensive per-number probability report."""
        main_probs, sub_probs = self._ensemble_probs
        main_ranked = np.argsort(main_probs)[::-1]
        sub_ranked = np.argsort(sub_probs)[::-1]

        report = {
            "main_numbers": [],
            "sub_numbers": [],
            "hot_numbers": [],
            "cold_numbers": [],
            "probability_distribution": {
                "main_mean": float(main_probs.mean()),
                "main_std": float(main_probs.std()),
                "main_max": float(main_probs.max()),
                "main_min": float(main_probs.min()),
                "sub_mean": float(sub_probs.mean()),
                "sub_std": float(sub_probs.std()),
            },
        }

        for rank, idx in enumerate(main_ranked):
            num = idx + self.cfg.main_min
            prob = float(main_probs[idx])
            report["main_numbers"].append({
                "number": num, "rank": rank + 1,
                "probability": round(prob * 100, 2),
            })

        for rank, idx in enumerate(sub_ranked):
            num = idx + self.cfg.sub_min
            prob = float(sub_probs[idx])
            report["sub_numbers"].append({
                "number": num, "rank": rank + 1,
                "probability": round(prob * 100, 2),
            })

        # Hot/cold: top/bottom 20%
        hot_count = max(3, self.n_main // 5)
        cold_count = max(3, self.n_main // 5)
        report["hot_numbers"] = [
            idx + self.cfg.main_min for idx in main_ranked[:hot_count]
        ]
        report["cold_numbers"] = [
            idx + self.cfg.main_min for idx in main_ranked[-cold_count:]
        ]

        return report
