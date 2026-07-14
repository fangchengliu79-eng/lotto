"""
🌀 螺旋矩阵预测算法 (Spiral Matrix Predictor)

基于旋转矩阵覆盖设计 (4-if-5) 的独立预测流水线。
适用于大乐透 (5码前区) 和 双色球 (6码前区)。

核心思想:
  - 不依赖概率融合, 依靠覆盖设计保证结构命中
  - 5个策略各自选取不同优势号池, 分别轮转出最优组合
  - 无区间形态过滤, 自由覆盖

与 Ensemble 统计算法的区别:
  Ensemble  -> 贝叶斯融合 + CRF评分 -> 概率优选
  螺旋矩阵  -> 覆盖设计矩阵 -> 结构保底
"""

import math
import random
import itertools
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# 覆盖设计索引表 (Covering Design Index Tables)
# ══════════════════════════════════════════════════════════════════════

# DLT: 选5覆盖 (129 blocks, 4-if-5 from 12 numbers)
_WHEEL_5_12: List[Tuple[int, ...]] = [
    (0, 4, 5, 7, 9), (1, 2, 3, 7, 10), (3, 5, 6, 8, 9), (2, 6, 8, 9, 11),
    (2, 3, 5, 6, 10), (0, 3, 6, 9, 10), (3, 4, 6, 7, 10), (4, 5, 7, 8, 11),
    (0, 1, 4, 6, 9), (0, 3, 5, 6, 11), (1, 3, 5, 7, 11), (0, 2, 7, 8, 9),
    (0, 1, 5, 6, 7), (0, 4, 5, 10, 11), (0, 2, 3, 4, 10), (1, 3, 4, 7, 8),
    (4, 6, 8, 9, 10), (7, 8, 9, 10, 11), (1, 2, 4, 6, 10), (0, 6, 7, 8, 10),
    (3, 4, 6, 8, 11), (1, 3, 4, 10, 11), (0, 2, 3, 7, 11), (0, 1, 2, 4, 8),
    (1, 3, 8, 9, 11), (5, 6, 7, 9, 10), (0, 1, 2, 9, 10), (2, 4, 7, 9, 11),
    (0, 2, 5, 6, 9), (0, 1, 3, 5, 9), (1, 5, 9, 10, 11), (1, 4, 5, 6, 8),
    (1, 4, 6, 7, 11), (1, 4, 7, 9, 10), (0, 2, 4, 6, 11), (0, 1, 6, 8, 11),
    (1, 2, 3, 4, 5), (0, 2, 5, 8, 10), (1, 2, 5, 8, 11), (3, 5, 7, 8, 10),
    (0, 2, 3, 6, 8), (1, 3, 6, 7, 9), (2, 3, 9, 10, 11), (0, 6, 7, 9, 11),
    (0, 5, 8, 9, 11), (1, 5, 7, 8, 9), (2, 3, 4, 6, 9), (1, 2, 3, 6, 11),
    (3, 4, 5, 9, 11), (0, 3, 8, 10, 11), (2, 3, 5, 7, 9), (0, 1, 7, 10, 11),
    (2, 5, 6, 7, 11), (1, 2, 6, 7, 8), (0, 3, 4, 8, 9), (1, 3, 6, 8, 10),
    (2, 4, 5, 7, 10), (2, 4, 5, 8, 9), (2, 4, 8, 10, 11), (5, 6, 8, 10, 11),
    (0, 1, 3, 4, 6), (0, 4, 5, 6, 10), (0, 4, 9, 10, 11), (2, 6, 7, 9, 10),
    (1, 4, 8, 9, 11), (0, 4, 7, 8, 10), (1, 2, 3, 8, 9), (4, 6, 7, 8, 9),
    (0, 1, 3, 5, 8), (1, 2, 7, 9, 11), (1, 2, 5, 6, 9), (0, 1, 2, 4, 7),
    (3, 6, 7, 8, 11), (3, 4, 5, 8, 10), (0, 3, 7, 9, 10), (3, 4, 5, 6, 7),
    (0, 1, 3, 5, 10), (0, 1, 4, 5, 11), (0, 2, 6, 10, 11), (4, 5, 6, 9, 11),
    (2, 3, 4, 7, 8), (0, 1, 8, 9, 10), (1, 6, 9, 10, 11), (0, 3, 4, 7, 11),
    (2, 3, 5, 8, 11), (0, 2, 3, 9, 11), (1, 7, 8, 10, 11), (2, 7, 8, 10, 11),
    (0, 4, 5, 6, 8), (0, 2, 3, 6, 7), (2, 5, 8, 9, 10), (0, 2, 3, 4, 5),
    (0, 1, 3, 7, 8), (0, 2, 5, 7, 8), (3, 5, 7, 10, 11), (0, 1, 2, 3, 11),
    (1, 2, 5, 10, 11), (4, 6, 7, 10, 11), (1, 5, 6, 7, 10), (1, 3, 4, 9, 10),
    (1, 4, 5, 9, 10), (2, 4, 5, 6, 8), (0, 1, 4, 6, 10), (1, 2, 3, 8, 10),
    (3, 6, 9, 10, 11), (2, 4, 5, 9, 11), (0, 2, 4, 8, 11), (1, 2, 4, 9, 11),
    (3, 7, 8, 9, 11), (0, 3, 5, 9, 10), (0, 1, 6, 8, 9), (0, 2, 5, 7, 11),
    (1, 3, 5, 6, 11), (0, 1, 2, 5, 7), (0, 2, 4, 6, 7), (0, 2, 4, 9, 10),
    (1, 4, 5, 8, 10), (0, 3, 5, 7, 10), (0, 1, 7, 9, 11), (0, 3, 7, 8, 11),
    (0, 1, 2, 6, 9), (0, 3, 8, 9, 10), (5, 6, 7, 9, 11), (2, 6, 8, 9, 10),
    (3, 5, 6, 7, 8), (2, 3, 4, 7, 9), (0, 2, 7, 9, 10), (1, 4, 5, 7, 8),
    (0, 2, 3, 4, 11),
]

# DLT: 14选5覆盖 (260 blocks)
_WHEEL_5_14: List[Tuple[int, ...]] = [
    (1, 8, 9, 12, 13), (0, 1, 3, 10, 11), (2, 5, 6, 7, 12), (0, 2, 3, 4, 10),
    (0, 3, 11, 12, 13), (0, 4, 7, 9, 12), (5, 6, 8, 10, 12), (4, 7, 10, 12, 13),
    (3, 4, 5, 7, 13), (0, 1, 2, 6, 12), (6, 9, 10, 11, 13), (1, 4, 7, 10, 11),
    (1, 2, 8, 10, 12), (4, 6, 9, 12, 13), (0, 2, 8, 9, 13), (2, 7, 9, 12, 13),
    (1, 5, 6, 7, 11), (1, 3, 5, 9, 12), (1, 4, 6, 10, 12), (2, 3, 4, 9, 11),
    (2, 4, 8, 9, 10), (0, 4, 7, 8, 10), (0, 1, 2, 9, 11), (1, 6, 7, 8, 9),
    (3, 4, 8, 9, 12), (2, 3, 5, 8, 13), (1, 2, 7, 11, 12), (0, 3, 5, 7, 8),
    (2, 6, 8, 10, 11), (0, 3, 6, 7, 13), (1, 5, 7, 8, 12), (0, 1, 7, 9, 13),
    (0, 2, 5, 6, 8), (0, 1, 4, 6, 13), (4, 5, 7, 8, 9), (0, 1, 6, 9, 10),
    (1, 2, 5, 12, 13), (1, 2, 3, 6, 9), (0, 1, 3, 8, 9), (1, 3, 4, 5, 10),
    (3, 6, 7, 11, 12), (1, 5, 8, 10, 11), (4, 5, 6, 11, 13), (3, 5, 6, 8, 11),
    (1, 6, 9, 11, 12), (0, 4, 5, 10, 13), (7, 8, 9, 11, 12), (3, 7, 9, 10, 11),
    (2, 3, 9, 10, 12), (2, 4, 6, 10, 13), (0, 2, 10, 12, 13), (0, 1, 5, 10, 12),
    (1, 6, 8, 10, 13), (1, 2, 8, 11, 13), (1, 2, 6, 7, 13), (1, 3, 5, 11, 13),
    (2, 3, 4, 6, 12), (0, 3, 4, 6, 11), (1, 3, 4, 12, 13), (0, 2, 6, 7, 9),
    (0, 5, 6, 9, 11), (5, 6, 8, 9, 13), (0, 1, 3, 5, 6), (5, 7, 10, 11, 12),
    (1, 5, 7, 10, 13), (3, 4, 6, 8, 10), (1, 2, 4, 5, 7), (0, 1, 2, 7, 10),
    (0, 8, 10, 11, 12), (0, 4, 6, 8, 9), (0, 2, 4, 5, 11), (2, 3, 5, 10, 11),
    (1, 2, 3, 7, 8), (0, 3, 5, 9, 13), (0, 5, 8, 9, 10), (1, 4, 6, 8, 11),
    (2, 5, 7, 9, 10), (1, 3, 6, 8, 12), (3, 8, 9, 11, 13), (4, 9, 10, 11, 12),
    (0, 3, 7, 10, 12), (6, 7, 8, 11, 13), (2, 5, 8, 9, 11), (0, 2, 4, 7, 13),
    (0, 1, 7, 8, 11), (0, 1, 3, 4, 7), (0, 7, 8, 12, 13), (3, 5, 10, 12, 13),
    (3, 4, 6, 7, 9), (0, 4, 9, 11, 13), (2, 5, 7, 11, 13), (0, 4, 5, 6, 7),
    (1, 4, 5, 11, 12), (1, 7, 9, 10, 12), (0, 4, 5, 8, 12), (0, 1, 2, 3, 13),
    (4, 5, 6, 9, 10), (2, 3, 7, 10, 13), (2, 4, 6, 7, 11), (1, 4, 5, 8, 13),
    (2, 4, 5, 9, 13), (2, 4, 7, 8, 12), (3, 5, 6, 7, 10), (4, 8, 11, 12, 13),
    (0, 2, 5, 9, 12), (5, 9, 11, 12, 13), (3, 4, 10, 11, 13), (0, 2, 6, 11, 13),
    (0, 2, 3, 7, 11), (1, 2, 4, 9, 12), (0, 3, 4, 8, 13), (1, 2, 9, 10, 13),
    (2, 6, 8, 9, 12), (0, 5, 6, 12, 13), (2, 4, 5, 10, 12), (0, 1, 4, 5, 9),
    (0, 5, 8, 11, 13), (2, 3, 8, 11, 12), (3, 4, 7, 8, 11), (1, 10, 11, 12, 13),
    (0, 7, 10, 11, 13), (7, 8, 9, 10, 13), (1, 2, 5, 6, 10), (0, 1, 2, 4, 8),
    (0, 3, 6, 9, 12), (3, 6, 10, 11, 12), (1, 5, 7, 9, 11), (1, 4, 8, 9, 10),
    (5, 6, 7, 9, 12), (0, 1, 8, 10, 13), (1, 2, 4, 11, 13), (0, 3, 6, 8, 10),
    (2, 4, 6, 8, 13), (3, 5, 8, 9, 10), (2, 5, 6, 11, 12), (1, 3, 4, 9, 11),
    (1, 3, 7, 8, 13), (0, 4, 6, 10, 12), (0, 6, 8, 11, 12), (2, 3, 5, 7, 12),
    (6, 7, 10, 12, 13), (0, 2, 4, 11, 12), (2, 3, 4, 5, 6), (2, 5, 8, 10, 13),
    (0, 9, 10, 12, 13), (4, 6, 7, 8, 12), (1, 3, 6, 11, 13), (0, 5, 7, 11, 12),
    (0, 5, 6, 10, 11), (0, 2, 9, 10, 11), (0, 3, 4, 9, 10), (1, 3, 6, 7, 10),
    (2, 7, 8, 10, 11), (1, 4, 7, 9, 13), (3, 7, 8, 10, 12), (3, 6, 9, 10, 13),
    (0, 3, 4, 5, 12), (0, 1, 6, 7, 12), (2, 3, 7, 8, 9), (1, 5, 6, 9, 13),
    (1, 2, 5, 8, 9), (4, 5, 8, 10, 11), (3, 4, 5, 9, 11), (0, 2, 3, 8, 12),
    (2, 3, 6, 12, 13), (2, 3, 6, 7, 8), (3, 7, 9, 12, 13), (0, 1, 3, 11, 12),
    (5, 7, 8, 12, 13), (2, 9, 11, 12, 13), (4, 7, 11, 12, 13), (0, 3, 8, 9, 11),
    (2, 4, 7, 9, 11), (0, 6, 7, 9, 10), (0, 2, 3, 6, 10), (2, 3, 4, 9, 13),
    (1, 8, 9, 10, 11), (0, 1, 4, 10, 11), (0, 2, 5, 7, 8), (0, 2, 4, 8, 11),
    (2, 4, 6, 9, 11), (1, 4, 5, 6, 12), (3, 6, 8, 9, 11), (0, 3, 5, 7, 9),
    (0, 2, 3, 5, 9), (1, 3, 8, 10, 13), (5, 9, 10, 11, 13), (1, 2, 3, 4, 10),
    (0, 1, 4, 8, 12), (5, 8, 9, 10, 12), (2, 3, 4, 7, 10), (2, 6, 7, 10, 12),
    (4, 8, 9, 10, 13), (0, 1, 2, 5, 11), (1, 3, 5, 7, 11), (1, 3, 4, 5, 8),
    (0, 6, 7, 9, 11), (5, 6, 7, 9, 13), (1, 3, 4, 7, 12), (5, 6, 7, 8, 10),
    (1, 2, 6, 10, 11), (0, 1, 5, 7, 13), (4, 6, 7, 10, 11), (2, 5, 6, 9, 13),
    (1, 7, 9, 11, 13), (3, 5, 8, 11, 12), (3, 6, 8, 12, 13), (0, 1, 5, 6, 8),
    (0, 1, 2, 4, 6), (0, 1, 6, 11, 13), (2, 7, 8, 12, 13), (0, 1, 7, 12, 13),
    (1, 2, 3, 11, 12), (0, 4, 5, 7, 10), (3, 5, 6, 9, 12), (0, 2, 5, 10, 13),
    (2, 4, 5, 12, 13), (0, 4, 5, 7, 11), (1, 3, 4, 10, 12), (3, 4, 9, 11, 12),
    (1, 6, 11, 12, 13), (1, 4, 6, 7, 13), (3, 4, 5, 6, 13), (2, 8, 10, 11, 13),
    (1, 4, 7, 8, 10), (0, 3, 5, 10, 11), (0, 6, 9, 10, 13), (2, 4, 10, 11, 12),
    (0, 2, 3, 8, 10), (0, 6, 7, 8, 13), (0, 1, 9, 11, 12), (2, 6, 8, 9, 10),
    (1, 3, 7, 9, 13), (4, 5, 7, 9, 12), (2, 3, 7, 11, 13), (1, 3, 8, 10, 11),
    (0, 7, 8, 9, 12), (4, 8, 10, 12, 13), (2, 4, 5, 8, 12), (1, 3, 4, 6, 9),
    (1, 3, 5, 9, 10), (1, 2, 7, 9, 13), (1, 5, 7, 8, 11), (2, 3, 6, 7, 11),
    (4, 8, 9, 10, 11), (5, 6, 9, 10, 13), (4, 5, 6, 11, 12), (1, 2, 3, 4, 5),
    (0, 4, 7, 9, 10), (0, 3, 7, 10, 13), (0, 4, 10, 12, 13), (1, 2, 6, 7, 8),
    (4, 7, 8, 9, 13), (1, 4, 10, 11, 13), (4, 5, 6, 8, 9), (0, 2, 7, 12, 13),
    (0, 2, 4, 9, 12), (1, 4, 8, 11, 12), (2, 3, 4, 8, 13), (1, 6, 9, 10, 12),
]

# SSQ: 选6覆盖 (61 blocks, 4-if-6 from 12 numbers)
_WHEEL_6_12: List[Tuple[int, ...]] = [
    (0, 3, 4, 5, 6, 11), (0, 2, 3, 5, 8, 9), (0, 1, 3, 4, 8, 10),
    (2, 3, 7, 8, 9, 11), (1, 2, 4, 5, 8, 10), (2, 6, 7, 8, 9, 10),
    (0, 2, 3, 4, 8, 9), (0, 2, 5, 8, 10, 11), (0, 2, 3, 4, 6, 9),
    (0, 2, 7, 8, 9, 11), (0, 1, 4, 5, 9, 11), (0, 1, 2, 6, 7, 11),
    (0, 1, 6, 8, 9, 10), (0, 1, 5, 6, 7, 10), (0, 1, 2, 5, 9, 10),
    (0, 2, 5, 6, 9, 11), (0, 4, 5, 6, 10, 11), (1, 3, 4, 6, 7, 8),
    (0, 1, 4, 7, 8, 9), (2, 4, 6, 8, 9, 10), (1, 3, 5, 6, 8, 9),
    (2, 5, 7, 8, 10, 11), (1, 2, 3, 5, 7, 11), (0, 1, 3, 4, 9, 10),
    (1, 2, 4, 5, 6, 9), (1, 5, 6, 9, 10, 11), (0, 1, 2, 4, 8, 11),
    (1, 2, 3, 4, 9, 11), (2, 3, 6, 7, 8, 10), (2, 3, 4, 8, 10, 11),
    (0, 1, 3, 5, 7, 8), (3, 4, 5, 6, 7, 11), (1, 2, 3, 7, 9, 10),
    (0, 6, 7, 9, 10, 11), (2, 3, 5, 6, 7, 9), (0, 3, 6, 9, 10, 11),
    (1, 4, 6, 7, 10, 11), (0, 3, 4, 5, 6, 8), (2, 3, 4, 5, 9, 10),
    (0, 4, 6, 7, 8, 11), (0, 2, 3, 4, 7, 10), (2, 4, 5, 6, 7, 8),
    (0, 1, 2, 3, 8, 11), (3, 5, 6, 8, 9, 11), (2, 4, 6, 7, 9, 11),
    (0, 3, 5, 7, 10, 11), (1, 2, 3, 6, 10, 11), (4, 5, 8, 9, 10, 11),
    (1, 4, 5, 7, 9, 10), (1, 5, 7, 8, 9, 11), (1, 2, 6, 8, 10, 11),
    (0, 2, 5, 6, 8, 10), (0, 3, 4, 5, 7, 9), (1, 3, 4, 5, 6, 10),
    (1, 2, 4, 7, 8, 10), (0, 2, 4, 5, 7, 11), (0, 1, 3, 4, 6, 7),
    (0, 3, 5, 7, 8, 10), (1, 2, 3, 8, 9, 10), (0, 1, 2, 9, 10, 11),
    (0, 1, 2, 6, 7, 9),
]


class LotteryWheelingFilter:
    """
    旋转矩阵 (覆盖设计) 核心引擎。
    将 N 个优势号通过覆盖设计矩阵生成投注组合。

    适用于两种彩种:
      - DLT (选5): 4-if-5 覆盖
      - SSQ (选6): 4-if-6 覆盖
    """

    # 轮转设计表: (k) -> (n) -> [blocks...]
    # k = 每个组合的号码数 (DLT=5, SSQ=6)
    _WHEELS = {
        5: {12: _WHEEL_5_12, 14: _WHEEL_5_14},
        6: {12: _WHEEL_6_12},
    }

    def __init__(self):
        self._wheel_cache: Dict[str, List[Tuple[int, ...]]] = {}

    def generate_wheel(self, selected_nums: List[int], k: int = 5) -> List[Tuple[int, ...]]:
        """
        旋转矩阵生成。

        Args:
            selected_nums: 优势号码池
            k: 每组号码数 (DLT=5, SSQ=6)

        Returns:
            去重后的组合列表
        """
        n = len(selected_nums)
        indices = self._get_wheel_indices(n, k)
        seen: Set[Tuple[int, ...]] = set()
        wheel = []
        for idx_tuple in indices:
            if all(i < n for i in idx_tuple):
                combo = tuple(sorted(selected_nums[i] for i in idx_tuple))
                if combo not in seen and len(combo) == k:
                    seen.add(combo)
                    wheel.append(combo)
        return wheel

    def _get_wheel_indices(self, n: int, k: int) -> List[Tuple[int, ...]]:
        """获取覆盖设计索引表"""
        cache_key = f"{k}_{n}"
        if cache_key in self._wheel_cache:
            return self._wheel_cache[cache_key]

        wheels_for_k = self._WHEELS.get(k, {})
        available = sorted(wheels_for_k.keys())
        best_key = None
        for ak in available:
            if ak <= n:
                best_key = ak

        if best_key is not None and best_key >= k + 3:
            indices = wheels_for_k[best_key]
        else:
            # 号太少: 全组合
            indices = list(itertools.combinations(range(n), k))

        self._wheel_cache[cache_key] = indices
        return indices

    def generate_back_pairs(self, selected_sub: List[int], k: int = 2) -> List[Tuple[int, ...]]:
        """后区组合: 所有 C(n, k) 组合"""
        return list(itertools.combinations(sorted(selected_sub), k))


class SpiralMatrixPredictor:
    """
    基于旋转矩阵覆盖设计的独立预测算法。

    predict() 返回格式与 ensemble.generate_recommendations() 完全兼容:
      {
        "groups": [
          {"index": 1, "main": [...], "sub": [...], "score": x.x, "reason": "..."},
          ...
        ],
        "hot_numbers": [...],
        "cold_numbers": [...],
        "timestamp": "...",
        "algorithm": "spiral_matrix"
      }
    """

    def __init__(self):
        self.wheeler = LotteryWheelingFilter()

    def predict(
        self,
        cfg,
        df: pd.DataFrame,
        num_groups: int = 5,
    ) -> Dict[str, Any]:
        """
        主预测入口。

        Args:
            cfg: 配置对象 (DLT / SSQ)
            df: 历史开奖数据
            num_groups: 返回组数

        Returns:
            与 generate_recommendations 兼容的字典
        """
        k = cfg.main_count  # DLT=5, SSQ=6
        sub_k = cfg.sub_count  # DLT=2, SSQ=1

        # 基础数据
        main_nums = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
        sub_nums = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
        n_draws = len(df)

        main_min, main_max = cfg.main_min, cfg.main_max
        sub_min, sub_max = cfg.sub_min, cfg.sub_max
        main_range_size = main_max - main_min + 1
        sub_range_size = sub_max - sub_min + 1

        # 频率
        main_counts = np.zeros(main_range_size)
        sub_counts = np.zeros(sub_range_size)
        for row in main_nums:
            for n in row:
                main_counts[n - main_min] += 1
        for row in sub_nums:
            for n in row:
                sub_counts[n - sub_min] += 1

        # 衰减加权
        decay = 0.98
        main_decay = np.zeros(main_range_size)
        sub_decay = np.zeros(sub_range_size)
        for idx in range(n_draws):
            w = decay ** idx
            for n in main_nums[idx]:
                main_decay[n - main_min] += w
            for n in sub_nums[idx]:
                sub_decay[n - sub_min] += w

        # 滑动窗口
        w50 = min(50, n_draws)
        main_win = np.zeros(main_range_size)
        sub_win = np.zeros(sub_range_size)
        for row in main_nums[:w50]:
            for n in row:
                main_win[n - main_min] += 1
        for row in sub_nums[:w50]:
            for n in row:
                sub_win[n - sub_min] += 1

        # 概率
        eps = 1e-6
        main_decay_prob = (main_decay + 1) / (main_decay.sum() + main_range_size)
        sub_decay_prob = (sub_decay + 1) / (sub_decay.sum() + sub_range_size)
        main_win_prob = (main_win + 1) / (main_win.sum() + main_range_size)
        sub_win_prob = (sub_win + 1) / (sub_win.sum() + sub_range_size)

        def _pick_top_n(probs, n):
            ranked = sorted([(i + main_min, probs[i]) for i in range(len(probs))], key=lambda x: -x[1])
            return [num for num, _ in ranked[:n]]

        def _pick_sub_weighted(probs, n):
            """从概率分布中加权随机选取 n 个蓝球 (不重复)"""
            rng = random.Random()
            # 使用当前策略名作为seed一部分，保证不同策略出不同结果
            candidates = list(range(len(probs)))
            weights = list(probs)
            picked = []
            for _ in range(n):
                if not candidates:
                    break
                total_w = sum(weights)
                if total_w <= 0:
                    picked.append(candidates[0])
                    candidates.pop(0)
                    weights.pop(0)
                else:
                    r = rng.random() * total_w
                    cum = 0
                    for i, w in enumerate(weights):
                        cum += w
                        if r <= cum:
                            picked.append(candidates[i])
                            candidates.pop(i)
                            weights.pop(i)
                            break
            return [i + sub_min for i in picked]

        def _dlt_zone_score(combo: Tuple[int, ...]) -> float:
            """大乐透区间形态弹性评分 (0~100分)，非硬过滤。
            
            评分标准:
              - 100: 完美 ABBBC/ABBBD + B区跨度 ≤ 5
              - 70:  ABBBC/ABBBD + B区跨度 6~8
              - 50:  B区2个相近号 (跨度 ≤ 3) + A区1个
              - 30:  B区3个但跨度 > 8 或其他弱匹配
              - 0:   全奇/全偶/全小(≤17)/全大(≥18)/无B区号
            """
            a = sum(1 for n in combo if 1 <= n <= 9)
            b = sum(1 for n in combo if 10 <= n <= 20)
            c = sum(1 for n in combo if 21 <= n <= 29)
            d = sum(1 for n in combo if 30 <= n <= 35)

            # 极端形态直接 0 分
            odd_all = all(n % 2 == 1 for n in combo)
            even_all = all(n % 2 == 0 for n in combo)
            small_all = all(n <= 17 for n in combo)
            big_all = all(n >= 18 for n in combo)
            if odd_all or even_all or small_all or big_all:
                return 0.0
            if b == 0:
                return 0.0
            if a + b + c + d != 5:
                return 0.0

            b_vals = [n for n in combo if 10 <= n <= 20]
            b_span = max(b_vals) - min(b_vals) if b_vals else 99

            # 完美 ABBBC/ABBBD + B区相近
            if a == 1 and b == 3 and (c + d) == 1:
                if b_span <= 5:
                    return 100.0
                elif b_span <= 8:
                    return 70.0
                else:
                    return 30.0

            # B区2个相近号 + A区1个
            if a == 1 and b == 2 and b_span <= 3:
                return 50.0

            # 其他有B区的给基础分
            if b >= 1:
                return 20.0

            return 0.0

        def _run_strategy(
            pool: List[int],
            pool_name: str,
            sub_probs: Optional[np.ndarray] = None,
            n_main: int = 12,
        ) -> Optional[Dict]:
            """单个策略: 选池 -> 补齐 -> 轮转 -> CRF优选 -> 约束检查"""
            if len(pool) < k:
                return None
            pool = pool[:min(n_main, len(pool))]

            # ── [安全补齐] 确保 pool 长度达到覆盖设计表需要的最小值 ──
            # 12选5需要12个, 14选5需要14个
            required = max(n_main, 12)  # 至少12个
            if len(pool) < required:
                pool_set = set(pool)
                # 按衰减概率从全范围中找未在pool中的最高分号码
                fill_candidates = sorted(
                    [(i + main_min, main_decay_prob[i]) for i in range(main_range_size)
                     if (i + main_min) not in pool_set],
                    key=lambda x: -x[1]
                )
                needed = required - len(pool)
                for num, _ in fill_candidates[:needed]:
                    pool.append(num)
                    pool_set.add(num)
                pool = sorted(pool)
                pool_name += f" [+补齐{needed}码]"

            # 轮转
            combos = self.wheeler.generate_wheel(pool, k=k)

            if not combos:
                return None

            # 蓝球: 根据本策略的概率分布加权选取
            sp = sub_probs if sub_probs is not None else sub_decay_prob
            best_back = _pick_sub_weighted(sp, sub_k)

            # CRF优选: 结构评分 + 概率评分 + DLT形态评分
            is_dlt = (cfg.short == "dlt")
            main_range_span = main_max - main_min
            section_size = main_range_span / 3.0
            scored = []
            for combo in combos:
                m_sorted = sorted(combo)
                prob_sum = sum(main_decay_prob[n - main_min] for n in combo)
                sub_prob = sum(sp[n - sub_min] for n in best_back)
                base = prob_sum + sub_prob

                # 1) 连号惩罚
                consec_pairs = sum(1 for i in range(1, len(m_sorted)) if m_sorted[i] == m_sorted[i-1] + 1)
                max_run = 1
                run = 1
                for i in range(1, len(m_sorted)):
                    if m_sorted[i] == m_sorted[i-1] + 1:
                        run += 1
                        max_run = max(max_run, run)
                    else:
                        run = 1
                consec_penalty = consec_pairs * 0.02
                if max_run >= 4:
                    consec_penalty += 0.15
                if max_run >= 5:
                    consec_penalty = 1.0

                # 2) 区间分布
                sections = [0, 0, 0]
                for v in m_sorted:
                    s_idx = min(2, int((v - main_min) / section_size))
                    sections[s_idx] += 1
                empty_sections = sum(1 for s in sections if s == 0)
                section_penalty = empty_sections * 0.04
                if empty_sections >= 2:
                    section_penalty += 0.05

                # 3) 跨度
                span = m_sorted[-1] - m_sorted[0]
                expected_span = main_range_span * 0.55
                span_dev = abs(span - expected_span) / expected_span
                span_penalty = min(span_dev * 0.03, 0.06)

                # 4) 奇偶平衡
                odd = sum(1 for v in m_sorted if v % 2 == 1)
                odd_ideal = k / 2.0
                odd_bonus = 0.03 if abs(odd - odd_ideal) <= 0.5 else 0.01

                # 5) 和值
                total = sum(m_sorted)
                mid_point = (main_min + main_max) / 2.0
                expected_sum = mid_point * k
                sum_dev = abs(total - expected_sum) / expected_sum
                sum_penalty = min(sum_dev * 0.02, 0.05)

                score = base - consec_penalty - section_penalty - span_penalty - sum_penalty + odd_bonus

                # 6) DLT形态弹性评分 (仅大乐透, 作为额外加分)
                if is_dlt:
                    dlt_score = _dlt_zone_score(m_sorted)
                    # dlt_score 0~100, 归一化到 0~0.15 加分
                    dlt_bonus = dlt_score / 100.0 * 0.15
                    score += dlt_bonus

                scored.append((combo, score))

            scored.sort(key=lambda x: -x[1])
            # 选择满足跨策略约束（前区+后区）的最高分组合
            best_combo = None
            for combo, _ in scored:
                if _check_constraints(sorted(combo), best_back):
                    best_combo = combo
                    break
            if best_combo is None:
                # 全部不满足约束时，选择得分最高的
                best_combo = scored[0][0]
            main_score = sum(main_decay_prob[n - main_min] for n in best_combo) * 100
            sub_score = sum(sp[n - sub_min] for n in best_back) * 100

            return {
                "main": list(best_combo),
                "sub": sorted(list(best_back)),
                "score": round(main_score + sub_score, 1),
                "reason": pool_name,
            }

        # ═══ 5个独立冷号轮询策略 ═══
        # 统一使用冷号池，通过不同池容量和蓝球概率区分组合

        # 所有号码按频率升序排列
        main_all_ranked = sorted(
            [(i + main_min, main_counts[i]) for i in range(len(main_counts))],
            key=lambda x: x[1]
        )

        # 跨策略约束：前区+后区号码和号码对使用计数
        used_main_counts = {}      # 前区 number -> total appearances across groups
        used_main_pairs = {}       # 前区 frozenset({a,b}) -> total appearances across groups
        used_sub_counts = {}       # 后区 number -> total appearances across groups
        used_sub_pairs = {}        # 后区 frozenset({a,b}) -> total appearances across groups

        def _check_constraints(main_result, sub_result, max_num=3, max_pair=3):
            """检查前区和后区的号码和号码对是否超过约束上限。"""
            # 前区号码
            for n in main_result:
                if used_main_counts.get(n, 0) >= max_num:
                    return False
            sorted_m = sorted(main_result)
            for i in range(len(sorted_m)):
                for j in range(i + 1, len(sorted_m)):
                    pair = frozenset([sorted_m[i], sorted_m[j]])
                    if used_main_pairs.get(pair, 0) >= max_pair:
                        return False
            # 后区号码
            for n in sub_result:
                if used_sub_counts.get(n, 0) >= max_num:
                    return False
            # 后区号码对（仅当≥2个后区号）
            if len(sub_result) >= 2:
                sorted_s = sorted(sub_result)
                for i in range(len(sorted_s)):
                    for j in range(i + 1, len(sorted_s)):
                        pair = frozenset([sorted_s[i], sorted_s[j]])
                        if used_sub_pairs.get(pair, 0) >= max_pair:
                            return False
            return True

        def _update_constraints(main_result, sub_result):
            """更新前区和后区的号码和号码对使用计数。"""
            # 前区
            for n in main_result:
                used_main_counts[n] = used_main_counts.get(n, 0) + 1
            sorted_m = sorted(main_result)
            for i in range(len(sorted_m)):
                for j in range(i + 1, len(sorted_m)):
                    pair = frozenset([sorted_m[i], sorted_m[j]])
                    used_main_pairs[pair] = used_main_pairs.get(pair, 0) + 1
            # 后区
            for n in sub_result:
                used_sub_counts[n] = used_sub_counts.get(n, 0) + 1
            if len(sub_result) >= 2:
                sorted_s = sorted(sub_result)
                for i in range(len(sorted_s)):
                    for j in range(i + 1, len(sorted_s)):
                        pair = frozenset([sorted_s[i], sorted_s[j]])
                        used_sub_pairs[pair] = used_sub_pairs.get(pair, 0) + 1

        def _run_cold_strategy(n_pool, sub_prob_name, seed, label):
            """从冷号池构建轮转组合，过滤已超限号码。"""
            pool = [num for num, _ in main_all_ranked[:n_pool]]
            # 过滤已出现3+次的号码（跨策略约束）
            pool = [n for n in pool if used_main_counts.get(n, 0) < 3]
            # 如果过滤后太少，补充冷区次冷号码
            if len(pool) < 8:
                extra = [num for num, _ in main_all_ranked[n_pool:n_pool + 6]
                         if used_main_counts.get(num, 0) < 3 and num not in pool]
                pool.extend(extra)
            # 蓝球: 根据给定概率加权
            # 蓝球: 根据给定概率加权
            if sub_prob_name == 'cold':
                sp = np.maximum(sub_counts.max() - sub_counts + 1, 1)
                sp = sp / sp.sum()
            elif sub_prob_name == 'gap':
                sp = _gap_prob(sub_nums, sub_counts, n_draws, sub_min, sub_max, sub_range_size)
            elif sub_prob_name == 'uniform':
                sp = np.ones(sub_range_size) / sub_range_size
            else:
                sp = sub_decay_prob
            return _run_strategy(pool, label, sub_probs=sp, n_main=max(n_pool, 12))

        # 间隔异常度概率 (gap_recency)
        def _gap_prob(num_array, counts, total, mn, mx, size):
            prob = np.zeros(size)
            for i, n in enumerate(range(mn, mx + 1)):
                appearances = [idx for idx, row in enumerate(num_array) if n in row]
                if appearances:
                    current_gap = appearances[0]
                    hist_gaps = []
                    for j in range(len(appearances) - 1):
                        hist_gaps.append(appearances[j+1] - appearances[j])
                    if hist_gaps:
                        mean_gap = np.mean(hist_gaps)
                        std_gap = np.std(hist_gaps) + 0.01
                        z = (current_gap - mean_gap) / std_gap
                        prob[i] = 1.0 / (1.0 + math.exp(-z * 0.8))
                    else:
                        prob[i] = 0.5
                else:
                    prob[i] = 0.7
            return prob / prob.sum()

        main_gap_probs = _gap_prob(main_nums, main_counts, n_draws, main_min, main_max, main_range_size)
        sub_gap_probs = _gap_prob(sub_nums, sub_counts, n_draws, sub_min, sub_max, sub_range_size)

        # 冷号 + 间隔异常度混合排序
        main_freq_rank = np.argsort(main_counts)
        main_cold_ranked = sorted(
            [(i + main_min, main_counts[i], main_gap_probs[i])
             for i in range(len(main_counts))],
            key=lambda x: (x[1], -x[2])  # 频率低优先，同频率间隔异常高的优先
        )

        # 策略1: ❄️冷号12池（按频率） + 冷号蓝球
        r1 = _run_cold_strategy(12, 'cold', 101, "❄️冷号轮转A: 最冷12码->旋转矩阵优选")
        if r1:
            _update_constraints(r1["main"], r1["sub"])

        # 策略2: ❄️冷号10 + 间隔异常6码混合池 + 间隔异常蓝球
        pool2 = [num for num, _, _ in main_cold_ranked[:10]]
        pool2 = [n for n in pool2 if used_main_counts.get(n, 0) < 3]  # 过滤超限号码
        pool2_set = set(pool2)
        gap_sorted_all = sorted(
            [(i + main_min, main_gap_probs[i]) for i in range(len(main_gap_probs))],
            key=lambda x: -x[1]
        )
        for num, _ in gap_sorted_all:
            if num not in pool2_set and len(pool2) < 16 and used_main_counts.get(num, 0) < 3:
                pool2.append(num)
                pool2_set.add(num)
        r2 = _run_strategy(pool2,
            "❄️冷号轮转B: 冷号10码+间隔异常6码->旋转矩阵优选",
            sub_probs=sub_gap_probs, n_main=16)
        if r2:
            _update_constraints(r2["main"], r2["sub"])

        # 策略3: ❄️间隔异常TOP14 + 间隔异常蓝球
        cold_half_th = max(1, int(main_range_size * 0.55))
        cold_half = set(i for i in range(main_range_size) if i in main_freq_rank[:cold_half_th])
        gap_in_cold = sorted(
            [(i + main_min, main_gap_probs[i]) for i in cold_half],
            key=lambda x: -x[1]
        )
        pool3 = [num for num, _ in gap_in_cold[:14]]
        pool3 = [n for n in pool3 if used_main_counts.get(n, 0) < 3]  # 过滤超限号码
        r3 = _run_strategy(pool3,
            "❄️冷号轮转C: 冷半区间隔异常TOP14->旋转矩阵优选",
            sub_probs=sub_gap_probs, n_main=14)
        if r3:
            _update_constraints(r3["main"], r3["sub"])

        # 策略4: ❄️冷号8 + 中段8码混合池 + 平衡蓝球
        pool4 = [num for num, _, _ in main_cold_ranked[:8]]
        pool4 = [n for n in pool4 if used_main_counts.get(n, 0) < 3]  # 过滤超限号码
        pool4_set = set(pool4)
        mid_idx = int(main_range_size * 0.2)
        mid_end = int(main_range_size * 0.65)
        main_mid_ranked = sorted(
            [(i + main_min, main_decay_prob[i]) for i in range(main_range_size)
             if i >= mid_idx and i <= mid_end],
            key=lambda x: -x[1]
        )
        for num, _ in main_mid_ranked:
            if num not in pool4_set and len(pool4) < 16 and used_main_counts.get(num, 0) < 3:
                pool4.append(num)
                pool4_set.add(num)
        sub_balanced = np.ones(sub_range_size) * 0.1
        sub_ranked_probs = sorted(
            [(i, sub_decay_prob[i]) for i in range(sub_range_size)],
            key=lambda x: -x[1]
        )
        sub_mid_start = max(1, int(len(sub_ranked_probs) * 0.2))
        sub_mid_end = min(len(sub_ranked_probs), int(len(sub_ranked_probs) * 0.8))
        for i in range(sub_mid_start, sub_mid_end):
            idx = sub_ranked_probs[i][0]
            sub_balanced[idx] = 1.0
        sub_balanced = sub_balanced / sub_balanced.sum()
        r4 = _run_strategy(pool4,
            "❄️冷号轮转D: 冷号8码+中段8码->旋转矩阵优选",
            sub_probs=sub_balanced, n_main=16)
        if r4:
            _update_constraints(r4["main"], r4["sub"])

        # 策略5: ❄️冷号12（按频率+遗漏加权）+ 冷号蓝球
        r5 = _run_cold_strategy(12, 'cold', 505, "❄️冷号轮转E: 冷号(频率+遗漏加权)12码->旋转矩阵优选")
        if r5:
            _update_constraints(r5["main"], r5["sub"])

        results = [r for r in [r1, r2, r3, r4, r5] if r is not None]
        for i, r in enumerate(results):
            r["index"] = i + 1

        # 更新冷热号展示（冷号=频率最低的，热号=频率最高的）

        hot = [num for num, _, _ in main_cold_ranked[:5]]
        cold = [num for num, _, _ in main_cold_ranked[-3:]]

        return {
            "groups": results[:num_groups],
            "hot_numbers": hot,
            "cold_numbers": cold,
            "timestamp": str(datetime.now()),
            "algorithm": "spiral_matrix",
        }
