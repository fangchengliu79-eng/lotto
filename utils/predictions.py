"""
预测封存与比对模块 - 完全参数化
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .helpers import get_logger


def _pred_dir(cfg) -> Path:
    """每期独立预测文件目录（数据级的容灾备份）"""
    d = cfg.data_dir / "predictions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_all(cfg) -> List[Dict]:
    """加载所有预测 — 按 id 去重，单文件损坏不影响其他期"""
    pred_dir = _pred_dir(cfg)
    files = sorted(pred_dir.glob("*.json"))
    if files:
        seen_ids = set()
        predictions = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    record = json.load(fh)
                rid = record.get("id")
                if rid and rid in seen_ids:
                    continue
                if rid:
                    seen_ids.add(rid)
                predictions.append(record)
            except Exception:
                get_logger(cfg).warning(f"读取预测文件失败: {f.name}")
        # 也尝试从旧式单文件加载并合并（迁移兼容）
        path = cfg.predictions_file
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    legacy = json.load(f)
                if isinstance(legacy, list):
                    for p in legacy:
                        rid = p.get("id")
                        if rid and rid not in seen_ids:
                            seen_ids.add(rid)
                            predictions.append(p)
            except Exception:
                pass
        return predictions

    # 纯旧式单文件回退
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
    """保存所有预测 — 同时写联合文件 + 每期独立文件，最大保留20条"""
    MAX_HISTORY = 20
    active = [p for p in predictions if p.get("status") == "active"]
    completed = [p for p in predictions if p.get("status") == "completed"]
    completed.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    kept = active + completed[:max(0, MAX_HISTORY - len(active))]

    # 写联合文件（兼容旧版读取）
    path = cfg.predictions_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    # 写每期独立文件（容灾备份）
    pred_dir = _pred_dir(cfg)
    kept_keys = set()
    for p in kept:
        alg = p.get("algorithm", "ensemble")
        key = f"{p['period']}_{alg}"
        kept_keys.add(key)
        pfile = pred_dir / f"{key}.json"
        with open(pfile, "w", encoding="utf-8") as f:
            json.dump(p, f, ensure_ascii=False, indent=2)

    # 清理超出的独立文件
    for f in pred_dir.glob("*.json"):
        # 兼容旧文件: period.json 或 period_algorithm.json
        stem = f.stem
        if stem in kept_keys:
            continue
        # 旧格式兼容: 如果 stem 是纯数字期号, 检查是否有任何算法用此期号
        if stem.isdigit() and any(k.startswith(stem + "_") for k in kept_keys):
            continue
        try:
            f.unlink()
        except Exception:
            pass


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


def get_next_draw_date(cfg) -> str:
    """根据开奖排期计算下一期预计开奖日期"""
    from datetime import datetime, timedelta
    now = datetime.now()
    wd = now.weekday()
    dh, dm = map(int, cfg.draw_time.split(':'))
    draw_passed = (now.hour > dh or (now.hour == dh and now.minute >= dm))

    for days_ahead in range(7):
        check = (wd + days_ahead) % 7
        if check in cfg.draw_days:
            if days_ahead == 0 and draw_passed:
                continue  # 今日已开奖，找下一期
            next_date = now + timedelta(days=days_ahead)
            return next_date.strftime("%Y-%m-%d")
    return "待定"


def save_prediction(period: str, recommendations: list, cfg, models_used: list = None, algorithm: str = "ensemble"):
    """封存一组预测"""
    predictions = _load_all(cfg)
    pred_id = f"{algorithm}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    recommendations = _convert(recommendations)
    entry = {
        "id": pred_id,
        "algorithm": algorithm,
        "period": str(period),
        "draw_date": get_next_draw_date(cfg),
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
    existing = [p for p in predictions if p["period"] == str(period) and p["status"] != "archived"
                and (p.get("algorithm") == algorithm or (algorithm == "ensemble" and "algorithm" not in p))]
    if existing:
        predictions = [p for p in predictions if p["id"] != existing[0]["id"]]
    predictions.append(entry)

    # 归档同一算法下更早的活跃预测（避免同算法多期待开奖）
    for p in predictions:
        if p["id"] != entry["id"] and p.get("status") == "active" and p.get("algorithm") == algorithm:
            p["status"] = "archived"

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


def get_all_predictions(cfg, include_archived: bool = False) -> List[Dict]:
    """获取所有预测（按时间倒序），默认排除 archived 记录"""
    predictions = _load_all(cfg)
    if not include_archived:
        predictions = [p for p in predictions if p.get("status") != "archived"]
    predictions.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    return predictions


def compare_with_draw(prediction: Dict, df: pd.DataFrame, cfg) -> Optional[Dict]:
    """将预测与实际开奖号码比对"""
    period = str(prediction["period"])

    # 类型安全：确保 period 类型匹配
    df_period = df["period"].astype(str)
    draw_row = df[df_period == period]
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


def auto_compare_latest(df: pd.DataFrame, cfg) -> int:
    """自动查找所有可比对的预测并比对，返回比对数量"""
    predictions = _load_all(cfg)
    df_periods = set(df["period"].astype(str).values)
    compared_count = 0
    for pred in predictions:
        if pred["status"] != "active":
            continue
        period = str(pred["period"])
        if period not in df_periods:
            continue
        get_logger(cfg).info(f"发现可比对预测: {period} (algorithm={pred.get('algorithm','ensemble')})")
        updated = compare_with_draw(pred, df, cfg)
        if updated:
            # 确保更新持久化到文件
            all_p = _load_all(cfg)
            for i, p in enumerate(all_p):
                if p.get("id") == updated.get("id"):
                    all_p[i] = updated
                    break
            _save_all(all_p, cfg)
            compared_count += 1
    return compared_count


def get_recent_draws_html(df: pd.DataFrame, cfg, n: int = 10) -> str:
    """生成最近 n 期实际开奖号码的 HTML"""
    recent = df.head(n).copy()
    if recent.empty:
        return "<p style='color:#94a3b8;font-size:0.85rem;'>暂无开奖数据</p>"

    lines = []
    for _, row in recent.iterrows():
        period = int(row["period"])
        main_nums = sorted([int(row[c]) for c in cfg.main_cols])
        sub_nums = sorted([int(row[c]) for c in cfg.sub_cols])
        main_balls = "".join(
            f"<span class='number-ball {cfg.main_label_en}' style='width:34px;height:34px;font-size:0.85rem;'>{n:02d}</span>"
            for n in main_nums
        )
        sub_balls = "".join(
            f"<span class='number-ball {cfg.sub_label_en}' style='width:34px;height:34px;font-size:0.85rem;'>{n:02d}</span>"
            for n in sub_nums
        )
        lines.append(
            f"<div style='display:flex;align-items:center;gap:0.6rem;padding:0.35rem 0;border-bottom:1px solid #f1f5f9;'>"
            f"<span style='font-weight:700;font-size:0.85rem;color:#64748b;min-width:3.5rem;'>#{period}</span>"
            f"<span>{main_balls}</span>"
            f"<span style='color:#94a3b8;font-size:0.7rem;'>{cfg.sub_label}</span>"
            f"<span>{sub_balls}</span>"
            f"</div>"
        )
    return "".join(lines)


def force_check_overdue(cfg) -> tuple:
    """
    强制从网页拉取最新开奖数据，检查所有 active 预测是否逾期。
    返回 (fresh_df, changed_count, messages) —— 用于 UI 展示。
    """
    from data_fetcher import update_data

    logger = get_logger(cfg)
    messages = []
    changed_count = 0

    # 1. 强制从网页刷新数据
    logger.info(f"{cfg.name}: 强制刷新数据...")
    fresh_df = update_data(cfg, force_refresh=True)
    if fresh_df is None or fresh_df.empty:
        fresh_df = pd.DataFrame()
        return fresh_df, 0, ["⚠️ 无法从网页获取最新数据"]

    fresh_periods = set(fresh_df["period"].astype(str).values)
    predictions = _load_all(cfg)
    updated_any = False

    for pred in predictions:
        if pred["status"] != "active":
            continue
        period = str(pred["period"])
        if period in fresh_periods:
            logger.info(f"{cfg.name}: 发现逾期预测 {period}，开始比对...")
            updated = compare_with_draw(pred, fresh_df, cfg)
            if updated:
                all_p = _load_all(cfg)
                for i, p in enumerate(all_p):
                    if p.get("id") == updated.get("id"):
                        all_p[i] = updated
                        break
                _save_all(all_p, cfg)
                updated_any = True
                changed_count += 1
                hit = updated.get("summary", {}).get("best_hits", 0)
                messages.append(f"✅ 期 {period} 比对完成，最佳命中 {hit} 个")

    if not updated_any:
        messages.append(f"ℹ️ 未发现新的可比对期号")

    return fresh_df, changed_count, messages
