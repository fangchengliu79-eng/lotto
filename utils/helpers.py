"""
Lottery prediction helpers - fully parameterized by cfg.

Provides validation, parsing, formatting, logging, and disclaimer
utilities used across all lottery types (DLT, SSQ, etc.).
"""
import logging
from typing import List, Tuple, Sequence, Optional, Any


def get_logger(cfg) -> logging.Logger:
    """Get a logger named after cfg.short (e.g. 'dlt', 'ssq')."""
    name = f"lotto.{cfg.short}"
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def validate_numbers(
    nums: Sequence[int],
    cfg,
    field: str = "main",
) -> Tuple[bool, str]:
    """Validate a list of numbers against cfg range + count rules.

    Parameters
    ----------
    nums : sequence of integers to validate.
    cfg  : LotteryConfig instance.
    field : ``"main"`` (front/red) or ``"sub"`` (back/blue).

    Returns
    -------
    (is_valid, message)
    """
    if field == "main":
        min_v, max_v, count = cfg.main_min, cfg.main_max, cfg.main_count
        label = cfg.main_label
    elif field == "sub":
        min_v, max_v, count = cfg.sub_min, cfg.sub_max, cfg.sub_count
        label = cfg.sub_label
    else:
        return False, f"Unknown field '{field}'"

    if len(nums) != count:
        return (
            False,
            f"{label}需要{count}个号码，提供了{len(nums)}个",
        )

    sorted_nums = sorted(nums)
    # Check duplicates
    if len(set(sorted_nums)) != len(sorted_nums):
        return False, f"{label}中存在重复号码"

    # Check range
    for n in sorted_nums:
        if not (min_v <= n <= max_v):
            return (
                False,
                f"号码{n}超出{label}范围[{min_v}, {max_v}]",
            )

    return True, f"{label}号码验证通过"


def validate_numbers_full(
    main: Sequence[int],
    sub: Sequence[int],
    cfg,
) -> Tuple[bool, str]:
    """Validate both main and sub numbers against cfg."""
    ok_main, msg_main = validate_numbers(main, cfg, field="main")
    if not ok_main:
        return False, msg_main
    ok_sub, msg_sub = validate_numbers(sub, cfg, field="sub")
    if not ok_sub:
        return False, msg_sub
    return True, "号码验证通过"


def parse_draw_row(row, cfg) -> Tuple[List[int], List[int]]:
    """Parse main/sub numbers from a DataFrame row using cfg.main_cols/sub_cols.

    Returns
    -------
    (main_numbers, sub_numbers)  each as sorted list of ints.
    """
    main = sorted([int(row[c]) for c in cfg.main_cols if c in row.index])
    sub = sorted([int(row[c]) for c in cfg.sub_cols if c in row.index])
    return main, sub


def format_numbers(
    main: Sequence[int],
    sub: Sequence[int],
    cfg,
    delimiter: str = "  ",
    item_sep: str = " ",
) -> str:
    """Format main + sub numbers with cfg labels.

    Example (DLT)::
        前区: 05 12 23 28 34  后区: 07 11

    Example (SSQ)::
        红球: 05 12 23 28 31 34  蓝球: 07
    """
    main_str = item_sep.join(f"{n:02d}" for n in sorted(main))
    sub_str = item_sep.join(f"{n:02d}" for n in sorted(sub))
    return f"{cfg.main_label}: {main_str}{delimiter}{cfg.sub_label}: {sub_str}"


def total_combinations(cfg) -> int:
    """Return total lottery combinations for this cfg."""
    return cfg.total_combinations()


def print_disclaimer(cfg) -> None:
    """Print a lottery disclaimer."""
    logger = get_logger(cfg)
    lines = [
        f"{'=' * 60}",
        f"⚠️  {cfg.name}数据分析与预测系统 ⚠️",
        f"{'=' * 60}",
        f"",
        f"📌 本系统仅供学习研究使用，所有预测结果仅供参考。",
        f"📌 彩票是一种随机游戏，没有任何方法可以保证中奖。",
        f"📌 请理性购彩，量力而行，切勿沉迷。",
        f"📌 数据分析基于历史开奖数据，结果具有随机性。",
        f"",
        f"🔢 {cfg.name}规则：",
        f"   {cfg.main_label}: {cfg.main_min}-{cfg.main_max}选{cfg.main_count}个",
        f"   {cfg.sub_label}: {cfg.sub_min}-{cfg.sub_max}选{cfg.sub_count}个",
        f"   总组合数: {cfg.total_combinations():,}",
        f"",
        f"{'=' * 60}",
    ]
    for line in lines:
        logger.info(line)
