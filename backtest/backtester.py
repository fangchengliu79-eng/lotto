"""
回测模块 - 对历史数据进行回测，评估模型预测能力
"""
from typing import Dict, List, Optional
from copy import deepcopy

import numpy as np
import pandas as pd

from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
from models.ensemble import EnsembleModel, generate_recommendations
from utils.helpers import get_logger


class Backtester:
    """滚动回测引擎"""

    def __init__(self, df: pd.DataFrame, cfg, test_window: int = 30):
        self.df = df.sort_values("period", ascending=True).reset_index(drop=True)
        self.cfg = cfg
        self.test_window = test_window
        self.logger = get_logger(cfg)

    def run(self, verbose: bool = False) -> Dict:
        total = len(self.df)
        if total < self.test_window + 20:
            return {"error": f"数据不足，需要至少 {self.test_window + 20} 期，当前 {total} 期"}

        results = []
        for i in range(self.test_window, total):
            train_df = self.df.iloc[:i]
            test_row = self.df.iloc[i]

            actual_main = sorted([int(test_row[c]) for c in self.cfg.main_cols])
            actual_sub = sorted([int(test_row[c]) for c in self.cfg.sub_cols])

            try:
                models = {
                    "frequency": FrequencyModel(train_df, self.cfg),
                    "poisson": PoissonModel(train_df, self.cfg),
                    "monte_carlo": MonteCarloModel(train_df, self.cfg, num_simulations=5000),
                    "exponential_smoothing": ExponentialSmoothingModel(train_df, self.cfg, alpha=0.3),
                }
                ensemble = EnsembleModel(models, self.cfg)
                recs = generate_recommendations(ensemble, self.cfg, num_groups=5)

                best_hits = 0
                for rec in recs:
                    main_hits = len([n for n in rec["main"] if n in actual_main])
                    sub_hits = len([n for n in rec["sub"] if n in actual_sub])
                    best_hits = max(best_hits, main_hits + sub_hits)

                results.append(best_hits)
            except Exception as e:
                self.logger.warning(f"回测第 {i} 期失败: {e}")
                continue

        if not results:
            return {"error": "所有回测都失败了"}

        results = np.array(results)
        total_tests = len(results)
        best_hits_count = np.sum(results == max(results)) if len(results) > 0 else 0

        stats = {
            "total_tests": total_tests,
            "total_hits": {
                "mean": float(np.mean(results)),
                "median": float(np.median(results)),
                "max": int(np.max(results)) if len(results) > 0 else 0,
                "min": int(np.min(results)) if len(results) > 0 else 0,
            },
            "best_hits_count": int(best_hits_count),
            "best_hits_rate": float(best_hits_count / total_tests * 100) if total_tests > 0 else 0,
        }
        return stats
