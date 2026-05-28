"""
预测封存与比对模块 - 完全参数化
"""
import json
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .helpers import get_logger


def _load_all(cfg) -> List[Dict]:
    """加载所有预测"""
    path = cfg.predictions_file
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        get_logger(cfg).warning(f"读取预测历史失败: {e}")
        return []


def _save_all(predictions: List[Dict], cfg):
    """保存所有预测"""
    path = cfg.predictions_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


def _convert(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _convert(obj.tolist())
    return obj


def save_prediction(period: str, recommendations: list, cfg, models_used: list = None):
    """封存一组预测"""
    predictions = _load_all(cfg)
    pred_id = f"pred_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    recommendations = _convert(recommendations)
    entry = {
        "id": pred_id,
        "period": str(period),
        "draw_date": "待定",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "active",
        "models_used": models_used or ["frequency", "poisson", "exponential_smoothing", "monte_carlo"],
        "recommendations": [
                {
                    "group": i + 1,
                    "main": r["main"],
                    "sub": r["sub"],
                    "score": round(r["score"], 4),
                    "reason": r.get("reason", f"综合评分 {round(r['score'], 1)}"),
                }
                for i, r in enumerate(recommendations)
            ],
        "actual_draw": None,
        "matches": None,
        "summary": None,
    }
    existing = [p for p in predictions if p["period"] == str(period) and p["status"] != "archived"]
    if existing:
        predictions = [p for p in predictions if p["id"] != existing[0]["id"]]
    predictions.append(entry)
    _save_all(predictions, cfg)
    return pred_id


def get_latest_prediction(cfg) -> Optional[Dict]:
    """获取最新的预测记录"""
    predictions = _load_all(cfg)
    if not predictions:
        return None
    predictions.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    return predictions[0]


def get_prediction_by_period(period: str, cfg) -> Optional[Dict]:
    """根据期号获取预测"""
    predictions = _load_all(cfg)
    for p in predictions:
        if p["period"] == str(period):
            return p
    return None


def get_all_predictions(cfg) -> List[Dict]:
    """获取所有预测（按时间倒序）"""
    predictions = _load_all(cfg)
    predictions.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    return predictions


def compare_with_draw(prediction: Dict, df: pd.DataFrame, cfg) -> Optional[Dict]:
    """将预测与实际开奖号码比对"""
    period = prediction["period"]

    draw_row = df[df["period"] == period]
    if draw_row.empty:
        get_logger(cfg).info(f"期号 {period} 的开奖数据尚未获取")
        return prediction

    actual_main = sorted([int(draw_row.iloc[0][c]) for c in cfg.main_cols])
    actual_sub = sorted([int(draw_row.iloc[0][c]) for c in cfg.sub_cols])

    matches = []
    total_main_hits = 0
    total_sub_hits = 0
    best_group = 0
    best_hits = 0

    for rec in prediction["recommendations"]:
        pred_main = rec["main"]
        pred_sub = rec["sub"]

        main_matches = sorted([n for n in pred_main if n in actual_main])
        sub_matches = sorted([n for n in pred_sub if n in actual_sub])
        total = len(main_matches) + len(sub_matches)

        match_info = {
            "group": rec["group"],
            "main_matches": main_matches,
            "sub_matches": sub_matches,
            "main_hits": len(main_matches),
            "sub_hits": len(sub_matches),
            "total_hits": total,
        }
        matches.append(match_info)
        total_main_hits += len(main_matches)
        total_sub_hits += len(sub_matches)
        if total > best_hits:
            best_hits = total
            best_group = rec["group"]

    num_groups = len(prediction["recommendations"])

    summary = {
        "actual_main": actual_main,
        "actual_sub": actual_sub,
        "best_group": best_group,
        "best_hits": best_hits,
        "avg_main_hits": round(total_main_hits / num_groups, 2) if num_groups > 0 else 0,
        "avg_sub_hits": round(total_sub_hits / num_groups, 2) if num_groups > 0 else 0,
        "avg_total_hits": round((total_main_hits + total_sub_hits) / num_groups, 2) if num_groups > 0 else 0,
    }

    prediction["actual_draw"] = {"main": actual_main, "sub": actual_sub}
    prediction["matches"] = matches
    prediction["summary"] = summary
    prediction["status"] = "completed"

    # 保存更新
    all_predictions = _load_all(cfg)
    for i, p in enumerate(all_predictions):
        if p["id"] == prediction["id"]:
            all_predictions[i] = prediction
            break
    _save_all(all_predictions, cfg)

    get_logger(cfg).info(f"预测 {period} 比对完成: 最佳命中 {best_hits} 个")
    return prediction


def auto_compare_latest(df: pd.DataFrame, cfg) -> Optional[Dict]:
    """自动查找最新可比对的预测并比对"""
    predictions = _load_all(cfg)
    for pred in predictions:
        if pred["status"] == "active":
            period = pred["period"]
            if period in df["period"].values:
                get_logger(cfg).info(f"发现可比对预测: {period}")
                return compare_with_draw(pred, df, cfg)
    return None
