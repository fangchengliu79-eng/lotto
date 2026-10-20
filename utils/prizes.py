"""
奖级与奖金计算模块（双色球 1-6 等奖 / 大乐透 1-9 等奖）

用法：
    from utils.prizes import evaluate, evaluate_prediction
    r = evaluate(cfg, main_hits=4, sub_hits=1)     # 单注
    s = evaluate_prediction(pred_dict, cfg)        # 整期 5 组

说明：
- 一等奖 / 二等奖 为浮动奖（按奖池和销量分配），无法离线精确得知，
  这里使用 EST_FLOATING 中的保守估算值，并标记 floating=True 供 UI 提示。
- 其余奖级为国家规定的固定奖金。
"""

from typing import Dict, List, Any

# 单注投注金额（双色球、大乐透基本投注均为 2 元/注）
BET_COST = 2

# 浮动奖估算值（元）—— 仅供展示参考，不构成任何承诺
EST_FLOATING = {
    "ssq": {1: 5_000_000, 2: 150_000},
    "dlt": {1: 8_000_000, 2: 300_000},
}

# ── 双色球：6 红 + 1 蓝 ──────────────────────────────────────────────
SSQ_LEVELS = {
    1: ("一等奖", None, "6红+1蓝（浮动奖）"),
    2: ("二等奖", None, "6红+0蓝（浮动奖）"),
    3: ("三等奖", 3000, "5红+1蓝"),
    4: ("四等奖", 200, "5红+0蓝 / 4红+1蓝"),
    5: ("五等奖", 10, "4红+0蓝 / 3红+1蓝"),
    6: ("六等奖", 5, "2红+1蓝 / 1红+1蓝 / 0红+1蓝"),
}

# ── 大乐透：5 前区 + 2 后区 ─────────────────────────────────────────
DLT_LEVELS = {
    1: ("一等奖", None, "5+2（浮动奖）"),
    2: ("二等奖", None, "5+1（浮动奖）"),
    3: ("三等奖", 10000, "5+0"),
    4: ("四等奖", 3000, "4+2"),
    5: ("五等奖", 300, "4+1"),
    6: ("六等奖", 200, "3+2"),
    7: ("七等奖", 100, "4+0"),
    8: ("八等奖", 15, "3+1 / 2+2"),
    9: ("九等奖", 5, "3+0 / 2+1 / 1+2 / 0+2"),
}


def _ssq_level(m: int, s: int) -> int:
    if m == 6 and s >= 1:
        return 1
    if m == 6:
        return 2
    if m == 5 and s >= 1:
        return 3
    if m == 5 or (m == 4 and s >= 1):
        return 4
    if m == 4 or (m == 3 and s >= 1):
        return 5
    if s >= 1:          # 0/1/2 红 + 蓝
        return 6
    return 0


def _dlt_level(m: int, s: int) -> int:
    if m == 5 and s == 2:
        return 1
    if m == 5 and s == 1:
        return 2
    if m == 5:
        return 3
    if m == 4 and s == 2:
        return 4
    if m == 4 and s == 1:
        return 5
    if m == 3 and s == 2:
        return 6
    if m == 4:
        return 7
    if (m == 3 and s == 1) or (m == 2 and s == 2):
        return 8
    if m == 3 or (m == 2 and s == 1) or (m == 1 and s == 2) or s == 2:
        return 9
    return 0


def prize_level(cfg, main_hits: int, sub_hits: int) -> int:
    """返回奖级（0 = 未中奖）"""
    if getattr(cfg, "short", "ssq") == "dlt":
        return _dlt_level(main_hits, sub_hits)
    return _ssq_level(main_hits, sub_hits)


def evaluate(cfg, main_hits: int, sub_hits: int) -> Dict[str, Any]:
    """单注奖金评估。

    Returns dict:
        level       奖级 0..N（0=未中奖）
        level_name  '一等奖' / '未中奖'
        amount      金额（元），浮动奖为估算值
        floating    是否浮动奖
        desc        中奖条件说明
    """
    key = "dlt" if getattr(cfg, "short", "ssq") == "dlt" else "ssq"
    table = DLT_LEVELS if key == "dlt" else SSQ_LEVELS
    lv = prize_level(cfg, main_hits, sub_hits)
    if lv <= 0:
        return {"level": 0, "level_name": "未中奖", "amount": 0,
                "floating": False, "desc": ""}
    name, fixed, desc = table[lv]
    if fixed is None:
        amount = EST_FLOATING[key].get(lv, 0)
        floating = True
    else:
        amount = fixed
        floating = False
    return {"level": lv, "level_name": name, "amount": amount,
            "floating": floating, "desc": desc}


def evaluate_prediction(pred: Dict, cfg) -> Dict[str, Any]:
    """对一条已完成预测的全部推荐组计算奖金。

    数据来源优先级：
      1. pred['matches']（比对结果，含 main_hits/sub_hits）
      2. pred['summary'] 中的 actual_main / actual_sub + recommendations 现场比对（旧记录兜底）

    Returns dict:
        groups   [{group, main_hits, sub_hits, level, level_name, amount, floating, desc}, ...]
        total_amount  总奖金（元）
        cost          总投入（元）
        profit        盈亏（元）
        win_groups    中奖组数
        best_level    最好奖级（0=未中奖）
        has_floating  是否包含浮动奖估算
    """
    groups: List[Dict[str, Any]] = []
    matches = {m.get("group"): m for m in (pred.get("matches") or [])}
    summary = pred.get("summary") or {}
    act_main = summary.get("actual_main")
    act_sub = summary.get("actual_sub")

    for rec in pred.get("recommendations", []):
        g = rec.get("group", rec.get("index"))
        m = matches.get(g)
        if m is not None:
            mh = int(m.get("main_hits", len(m.get("main_matches", []))))
            sh = int(m.get("sub_hits", len(m.get("sub_matches", []))))
        elif act_main is not None and act_sub is not None:
            mh = len(set(rec.get("main", [])) & set(act_main))
            sh = len(set(rec.get("sub", [])) & set(act_sub))
        else:
            continue
        p = evaluate(cfg, mh, sh)
        groups.append({"group": g, "main_hits": mh, "sub_hits": sh, **p})

    total = sum(x["amount"] for x in groups)
    n = len(groups)
    winning = [x for x in groups if x["level"] > 0]
    return {
        "groups": groups,
        "total_amount": total,
        "cost": n * BET_COST,
        "profit": total - n * BET_COST,
        "win_groups": len(winning),
        "best_level": min((x["level"] for x in winning), default=0),
        "has_floating": any(x["floating"] for x in winning),
    }


def fmt_money(v: float) -> str:
    """金额格式化：¥1,234,567 / ¥5"""
    try:
        v = float(v)
    except Exception:
        return "-"
    if v == int(v):
        return "¥{:,.0f}".format(v)
    return "¥{:,.2f}".format(v)


def level_badge(level: int) -> str:
    """奖级短标签，用于卡片角标"""
    if level <= 0:
        return ""
    names = ["", "一等", "二等", "三等", "四等", "五等", "六等", "七等", "八等", "九等"]
    return names[level] if level < len(names) else f"{level}等"
