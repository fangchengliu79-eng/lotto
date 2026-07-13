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
            """单个策略: 选池 -> 补齐 -> 轮转 -> CRF优选"""
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
            best_combo = scored[0][0]
            main_score = sum(main_decay_prob[n - main_min] for n in best_combo) * 100
            sub_score = sum(sp[n - sub_min] for n in best_back) * 100

            return {
                "main": list(best_combo),
                "sub": sorted(list(best_back)),
                "score": round(main_score + sub_score, 1),
                "reason": pool_name,
            }

        # ═══ 5个独立策略 ═══
        # 每个策略使用不同的蓝球概率分布

        # 策略1: 🔥热号轮转 (蓝球: 衰减加权热号)
        pool1 = _pick_top_n(main_decay_prob, 12)
        r1 = _run_strategy(pool1,
            f"🔥热号轮转: 衰减加权前{min(12, len(pool1))}码->旋转矩阵(4-if-{k})优选",
            sub_probs=sub_decay_prob, n_main=12)

        # 策略2: 📈趋势轮转 (蓝球: 近期窗口热号)
        pool2_all = _pick_top_n(main_win_prob, 14)
        if len(main_nums) > 0:
            last_main = main_nums[0]
            neighbors = set()
            for n in last_main:
                for offset in range(-2, 3):
                    nn = n + offset
                    if main_min <= nn <= main_max:
                        neighbors.add(nn)
            pool2 = list(dict.fromkeys(list(neighbors) + pool2_all))[:12]
        else:
            pool2 = pool2_all[:12]
        r2 = _run_strategy(pool2,
            f"📈趋势轮转: 近期窗口+上期邻域->旋转矩阵优选",
            sub_probs=sub_win_prob, n_main=12)

        # 策略3: ❄️冷号轮转 (蓝球: 低频冷号)
        main_ranked_asc = sorted(
            [(i + main_min, main_counts[i]) for i in range(len(main_counts))],
            key=lambda x: x[1]
        )
        pool3 = [num for num, _ in main_ranked_asc[:14]]
        # 蓝球冷号概率: 反比例于频率
        sub_cold_prob = np.maximum(sub_counts.max() - sub_counts + 1, 1)
        sub_cold_prob = sub_cold_prob / sub_cold_prob.sum()
        r3 = _run_strategy(pool3,
            f"❄️冷号轮转: 低频后{min(14, len(pool3))}码->旋转矩阵优选",
            sub_probs=sub_cold_prob, n_main=14)

        # 策略4: 🔗邻域轮转 (蓝球: 上期蓝球+附近)
        if len(main_nums) > 0:
            last_main = main_nums[0]
            pool4_base = set(last_main)
            for n in last_main:
                for offset in range(-1, 2):
                    nn = n + offset
                    if main_min <= nn <= main_max:
                        pool4_base.add(nn)
            hot_all = _pick_top_n(main_decay_prob, 20)
            for n in hot_all:
                if len(pool4_base) >= 14:
                    break
                pool4_base.add(n)
            pool4 = sorted(pool4_base)[:14]
        else:
            pool4 = _pick_top_n(main_decay_prob, 12)
        # 蓝球: 上期蓝球邻域加权
        sub_nearby_prob = np.ones(sub_range_size) * 0.2
        if len(sub_nums) > 0:
            last_sub = sub_nums[0]
            for n in last_sub:
                idx = n - sub_min
                sub_nearby_prob[idx] *= 3.0
                for offset in range(-1, 2):
                    ni = n + offset
                    if sub_min <= ni <= sub_max:
                        sub_nearby_prob[ni - sub_min] *= 1.5
        sub_nearby_prob = sub_nearby_prob / sub_nearby_prob.sum()
        r4 = _run_strategy(pool4,
            f"🔗邻域轮转: 上期号码+邻域补全->旋转矩阵优选",
            sub_probs=sub_nearby_prob, n_main=14)

        # 策略5: ⚖️平衡轮转 (蓝球: 避开极端冷热)
        main_ranked_desc = sorted(
            [(i + main_min, main_decay_prob[i]) for i in range(len(main_decay_prob))],
            key=lambda x: -x[1]
        )
        total_nums = len(main_ranked_desc)
        start = max(1, int(total_nums * 0.2))
        end = min(total_nums, int(total_nums * 0.7))
        pool5 = [num for num, _ in main_ranked_desc[start:end]][:14]
        # 蓝球: 剔除极热极冷, 取中间
        sub_ranked = sorted([(i, sub_decay_prob[i]) for i in range(len(sub_decay_prob))], key=lambda x: -x[1])
        sub_mid_start = max(1, int(len(sub_ranked) * 0.2))
        sub_mid_end = min(len(sub_ranked), int(len(sub_ranked) * 0.8))
        sub_balanced = np.ones(sub_range_size) * 0.1
        for i in range(sub_mid_start, sub_mid_end):
            idx = sub_ranked[i][0]
            sub_balanced[idx] = 1.0
        sub_balanced = sub_balanced / sub_balanced.sum()
        r5 = _run_strategy(pool5,
            f"⚖️平衡轮转: 避开极端冷热,中间{min(14, len(pool5))}码->旋转矩阵优选",
            sub_probs=sub_balanced, n_main=12)

        results = [r for r in [r1, r2, r3, r4, r5] if r is not None]
        for i, r in enumerate(results):
            r["index"] = i + 1

        hot = [num for num, _ in main_ranked_desc[:5]]
        cold = [num for num, _ in main_ranked_asc[:3]]

        return {
            "groups": results[:num_groups],
            "hot_numbers": hot,
            "cold_numbers": cold,
            "timestamp": str(datetime.now()),
            "algorithm": "spiral_matrix",
        }
