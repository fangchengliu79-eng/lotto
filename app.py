#!/usr/bin/env python3
"""
双色球 + 大乐透 统一预测系统
通过页面顶部 tabs 切换彩种（像浏览器标签页一样）
"""
import sys, os
from io import StringIO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

from config_dlt import DLT
from config_ssq import SSQ
from data_fetcher import update_data
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
from models.ensemble import EnsembleModel, generate_recommendations
from models.spiral_matrix import SpiralMatrixPredictor
from utils.predictions import (
    get_all_predictions, get_latest_prediction, save_prediction,
    auto_compare_latest, force_check_overdue, get_recent_draws_html
)
from utils.helpers import print_disclaimer

st.set_page_config(page_title="彩票预测系统", page_icon="🎯", layout="wide")

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
.stApp { background: #f0f2f5; }
header[data-testid="stHeader"], .stApp > header { display: none; }
html, body { color: #1e293b; font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-weight: 700; letter-spacing: -0.02em; color: #1e293b; }
h1 { font-size: 1.5rem !important; }
[data-testid="stAppViewBlockContainer"] { max-width: 1400px; padding: 1rem 2rem; }
.stTabs [data-baseweb="tab-list"] { gap: 2px; background: #fff; border-radius: 10px; padding: 4px; border: 1px solid #e2e8f0; margin-bottom: 1rem; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 0.5rem 1.5rem; font-weight: 600; font-size: 0.9rem; color: #64748b; }
.stTabs [aria-selected="true"] { background: #eff6ff !important; color: #1d4ed8 !important; border-color: #2563eb !important; }
.glass-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.glass-card.glow { border-color: #2563eb; box-shadow: 0 2px 8px rgba(37,99,235,0.10); }
.metric-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem; text-align: center; }
.metric-card .value { font-size: 1.8rem; font-weight: 800; color: #1e293b; line-height: 1.2; }
.metric-card .label { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.25rem; }
.number-ball { display: inline-flex; align-items: center; justify-content: center; width: 46px; height: 46px; border-radius: 50%; font-size: 1rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin: 0 3px; }
.number-ball.main { background: linear-gradient(135deg,#dc2626,#b91c1c); color: #fff; box-shadow: 0 2px 6px rgba(220,38,38,0.25); }
.number-ball.sub { background: linear-gradient(135deg,#2563eb,#1d4ed8); color: #fff; box-shadow: 0 2px 6px rgba(37,99,235,0.25); }
.number-ball.hit { box-shadow: 0 0 0 3px #fff, 0 0 0 5px #eab308; border: none; }
.stButton > button[kind="primary"] { background: #2563eb; border: none; color: #fff; }
.stButton > button[kind="primary"]:hover { background: #1d4ed8; }
.stRadio div[role="radiogroup"] label { border-radius: 8px; padding: 0.5rem 0.75rem; background: #fff; border: 1px solid #e2e8f0; font-size: 0.85rem; }
.stRadio div[role="radiogroup"] label[data-checked="true"] { border-color: #2563eb; background: #eff6ff; color: #1d4ed8; font-weight: 600; }
.streamlit-expanderHeader { background: #fff !important; border-radius: 10px !important; border: 1px solid #e2e8f0 !important; }
[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrame"] thead tr th { background: #f8fafc !important; color: #64748b !important; font-size: 0.75rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid #e2e8f0 !important; }
[data-testid="stDataFrame"] tbody tr { background: #fff !important; }
[data-testid="stDataFrame"] tbody tr:nth-child(even) { background: #f8fafc !important; }
[data-testid="stDataFrame"] tbody td { border-bottom: 1px solid #f1f5f9 !important; color: #334155 !important; }
[data-testid="stAlert"] { border-radius: 10px; border-left: 4px solid #2563eb; background: #fff !important; }
.stAlert.success { border-left-color: #059669; } .stAlert.info { border-left-color: #2563eb; }
.stAlert.warning { border-left-color: #d97706; } .stAlert.error { border-left-color: #dc2626; }
hr { border: none; height: 1px; background: #e2e8f0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── 辅助函数 ──
def ball(n, kind="main", hit=False):
    return f"<span class='number-ball {kind}{' hit' if hit else ''}'>{n:02d}</span>"

def metric_card(v, l, d=None):
    return f"<div class='metric-card'><div class='value'>{v}</div><div class='label'>{l}</div>{'<div style=font-size:0.7rem;color:#059669;margin-top:0.15rem>'+d+'</div>' if d else ''}</div>"

def pred_card(rec, matches=None, active=True, cfg=None):
    g = rec.get("index", rec.get("group", 1))
    gm = matches.get(g, {}) if matches else {}
    mh = set(gm.get("main_matches", [])) if gm else set()
    sh = set(gm.get("sub_matches", [])) if gm else set()
    sc = rec["score"]
    mb = "".join(ball(n, "main", n in mh) for n in rec["main"])
    sb = "".join(ball(n, "sub", n in sh) for n in rec["sub"])
    ht = ""
    reason = rec.get("reason", "")
    strategy_name = reason.split(":")[0] if ":" in reason else "综合推荐"
    strategy_icons = {"追冷A": "❄️", "追冷B": "❄️", "追冷C": "❄️", "追冷D": "❄️", "追冷E": "❄️"}
    strategy_colors = {"追冷A": "#2563eb", "追冷B": "#0891b2", "追冷C": "#4f46e5", "追冷D": "#0284c7", "追冷E": "#475569"}
    icon = strategy_icons.get(strategy_name, "🎯")
    color = strategy_colors.get(strategy_name, "#64748b")
    if gm and gm.get("total_hits", 0) > 0:
        m = gm
        ht = (f"<div style='margin-top:0.5rem;font-size:0.85rem;color:#eab308;font-weight:600;'>🎯 命中 {m['main_hits']}{cfg.main_label} + {m['sub_hits']}{cfg.sub_label} = {m['total_hits']}个</div>")
    elif active:
        ht = "<div style='margin-top:0.5rem;font-size:0.85rem;color:#64748b;'>⏳ 待开奖</div>"
    return (f"<div class='glass-card glow' style='padding:1rem;border-top:3px solid {color};'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;'>"
            f"<div style='display:flex;align-items:center;gap:0.5rem;'>"
            f"<span style='background:{color};color:#fff;border-radius:6px;padding:0.15rem 0.5rem;font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;'>{icon} {strategy_name}</span>"
            f"<span style='font-size:0.7rem;font-weight:600;color:#94a3b8;'>#{g}</span></div>"
            f"<span style='font-size:0.75rem;color:{color};font-weight:600;font-family:JetBrains Mono,monospace;'>{sc:.1f}</span></div>"
            f"<div style='margin-bottom:0.5rem;'><div style='font-size:0.7rem;color:#64748b;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.05em;'>{cfg.main_label} · {cfg.main_count}码</div><div>{mb}</div></div>"
            f"<div style='margin-bottom:0.5rem;'><div style='font-size:0.7rem;color:#64748b;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.05em;'>{cfg.sub_label} · {cfg.sub_count}码</div><div>{sb}</div></div>"
            f"<div style='font-size:0.7rem;color:#94a3b8;margin-top:0.35rem;'>{reason}</div>{ht}</div>")

def hist_row(rec, mm=None, cfg=None):
    g = rec.get("index", rec.get("group", 1))
    m = mm.get(g, {}) if mm else {}
    mh = set(m.get("main_matches", [])) if m else set()
    sh = set(m.get("sub_matches", [])) if m else set()
    mb = "".join(ball(n, "main", n in mh) for n in rec["main"])
    sb = "".join(ball(n, "sub", n in sh) for n in rec["sub"])
    t = m.get("total_hits", 0) if m else 0
    bdg = f"<span style='color:#eab308;font-weight:700;font-size:0.85rem;'>🎯 命中 {t}个</span>" if t > 0 else ""
    return (f"<div style='background:#fafbfc;border:1px solid #e2e8f0;border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:0.4rem;display:flex;align-items:center;flex-wrap:wrap;gap:0.5rem;'>"
            f"<span style='font-weight:700;color:#94a3b8;font-size:0.75rem;min-width:2.5rem;'>第{g}组</span>"
            f"<span style='font-size:0.7rem;color:#94a3b8;'>{cfg.main_label}</span>{mb}"
            f"<span style='font-size:0.7rem;color:#94a3b8;margin-left:0.3rem;'>{cfg.sub_label}</span>{sb}"
            f"{'&nbsp;'+bdg if bdg else ''}</div>")

def spiral_card(rec, cfg):
    """螺旋矩阵结果卡片 - 紫金配色"""
    g = rec.get("index", 1)
    mb = "".join(ball(n, "main") for n in rec["main"])
    sb = "".join(ball(n, "sub") for n in rec["sub"])
    sc = rec.get("score", 0)
    reason = rec.get("reason", "")
    strategy_name = reason.split(":")[0] if ":" in reason else "螺旋矩阵"
    colors = {"冷号轮转A":"#2563eb","冷号轮转B":"#0891b2","冷号轮转C":"#4f46e5","冷号轮转D":"#0284c7","冷号轮转E":"#475569"}
    icons = {"冷号轮转A":"❄️","冷号轮转B":"❄️","冷号轮转C":"❄️","冷号轮转D":"❄️","冷号轮转E":"❄️"}
    icon = icons.get(strategy_name, "🌀")
    color = colors.get(strategy_name, "#7c3aed")
    return (f"<div class='glass-card glow' style='padding:1rem;border-top:3px solid {color};background:linear-gradient(135deg,#faf5ff,#f3e8ff);'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;'>"
            f"<div style='display:flex;align-items:center;gap:0.5rem;'>"
            f"<span style='background:{color};color:#fff;border-radius:6px;padding:0.15rem 0.5rem;font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;'>{icon} {strategy_name}</span>"
            f"<span style='font-size:0.7rem;font-weight:600;color:#94a3b8;'>#{g}</span></div>"
            f"<span style='font-size:0.75rem;color:{color};font-weight:600;font-family:JetBrains Mono,monospace;'>{sc:.1f}</span></div>"
            f"<div style='margin-bottom:0.5rem;'><div style='font-size:0.7rem;color:#64748b;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.05em;'>{cfg.main_label} · {cfg.main_count}码</div><div>{mb}</div></div>"
            f"<div style='margin-bottom:0.5rem;'><div style='font-size:0.7rem;color:#64748b;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.05em;'>{cfg.sub_label} · {cfg.sub_count}码</div><div>{sb}</div></div>"
            f"<div style='font-size:0.7rem;color:#94a3b8;margin-top:0.35rem;'>{reason}</div></div>")


@st.cache_data(ttl=3600, show_spinner="正在加载数据...")
def load_lottery(cfg_key, _mtime=0):
    _cfg = DLT if cfg_key == "dlt" else SSQ
    return update_data(_cfg)


def _generate_both_algorithms(cfg, df, period):
    """生成并保存统计算法 + 螺旋矩阵预测"""
    from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
    from models.timeseries import ExponentialSmoothingModel
    from models.ensemble import EnsembleModel, generate_recommendations
    from models.spiral_matrix import SpiralMatrixPredictor

    md = {"f": FrequencyModel(cfg), "p": PoissonModel(cfg),
          "e": ExponentialSmoothingModel(cfg, alpha=0.3), "m": MonteCarloModel(cfg)}
    md["m"].n_simulations = 20000
    mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
    sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
    for m in md.values(): m.fit(mn, sn)

    # 统计算法
    ensemble = EnsembleModel(md, cfg)
    r_ensemble = generate_recommendations(ensemble, cfg, num_groups=5, df=df)
    save_prediction(period=str(period), recommendations=r_ensemble.get("groups", []), cfg=cfg, algorithm="ensemble")

    # 螺旋矩阵
    try:
        predictor = SpiralMatrixPredictor()
        r_spiral = predictor.predict(cfg, df, num_groups=5)
        if r_spiral.get("groups"):
            save_prediction(period=str(period), recommendations=r_spiral["groups"], cfg=cfg, algorithm="spiral_matrix")
    except Exception as e:
        import logging
        logging.getLogger(cfg.name).warning(f"螺旋矩阵预测生成失败: {e}")


def render_lottery(cfg):
    """渲染单个彩种内容"""
    import os
    csv_path = str(cfg.history_csv)
    mtime = os.path.getmtime(csv_path) if os.path.exists(csv_path) else 0
    df = load_lottery(cfg.short, _mtime=mtime)
    if df is None or len(df) == 0:
        st.error(f"无法获取{cfg.name}数据"); st.stop()
    st.info(f"✅ 已加载 {len(df):,} 期 {cfg.name} 数据 · {cfg.total_combinations():,} 种组合")

    auto_compare_latest(df, cfg)

    # ── 逾期自动检查 ──
    from datetime import datetime as _dt
    _now = _dt.now()
    _init_key = f"_overdue_inited_{cfg.short}"
    if _init_key not in st.session_state:
        st.session_state[_init_key] = False

    col_a, col_b = st.columns([3, 1])
    with col_b:
        force_btn = st.button("🔍 强制检查逾期比对", key=f"force_{cfg.short}",
                              help=f"强制从网页拉取最新{cfg.name}开奖数据并自动比对",
                              use_container_width=True)

    active_missing = []
    for p in get_all_predictions(cfg):
        if p["status"] == "active":
            pp = str(p["period"])
            if pp not in set(df["period"].astype(str).values):
                active_missing.append(pp)

    should_check = force_btn
    if not should_check and not st.session_state[_init_key] and active_missing:
        _wd = _now.weekday()
        _h, _m = _now.hour, _now.minute
        _dh, _dm = map(int, cfg.draw_time.split(':'))
        _draw_passed = (_h > _dh or (_h == _dh and _m >= _dm))
        if _wd in cfg.draw_days and _draw_passed:
            should_check = True
        _yesterday = (_wd - 1) % 7
        if _yesterday in cfg.draw_days and not _draw_passed:
            should_check = True
    st.session_state[_init_key] = True

    if should_check:
        if active_missing:
            with st.spinner(f"🔍 正在联网获取最新{cfg.name}开奖数据并自动比对..."):
                fresh_df, count, msgs = force_check_overdue(cfg)
                if fresh_df is not None and not fresh_df.empty:
                    df = fresh_df
                if count > 0:
                    for msg in msgs:
                        st.success(msg)
                    st.rerun()
                else:
                    for msg in msgs:
                        st.info(msg)
        elif force_btn:
            with st.spinner(f"🔍 正在强制检查..."):
                fresh_df, count, msgs = force_check_overdue(cfg)
                if fresh_df is not None and not fresh_df.empty:
                    df = fresh_df
                if count > 0:
                    for msg in msgs:
                        st.success(msg)
                    st.rerun()
                else:
                    for msg in msgs:
                        st.info(msg)

    # ── 自动生成预测（如无活跃预测）──
    all_preds_now = get_all_predictions(cfg)
    active_ensemble = [p for p in all_preds_now if p["status"] == "active" and p.get("algorithm", "ensemble") == "ensemble"]
    active_spiral = [p for p in all_preds_now if p["status"] == "active" and p.get("algorithm") == "spiral_matrix"]

    # 如果任一种算法的活跃预测缺失，补齐 (每个 session 只触发一次)
    gen_key = f"_gen_done_{cfg.short}"
    if gen_key not in st.session_state:
        st.session_state[gen_key] = False

    if not st.session_state[gen_key] and (not active_ensemble or not active_spiral):
        st.session_state[gen_key] = True
        existing_active_periods = [int(p["period"]) for p in all_preds_now if p["status"] == "active"]
        completed = [int(p["period"]) for p in all_preds_now if p["status"] == "completed"]
        latest_period_int = int(df.iloc[0]["period"])
        if existing_active_periods:
            next_period = max(existing_active_periods)
        elif completed:
            next_period = max(completed) + 1
            if latest_period_int >= next_period:
                next_period = latest_period_int + 1
        else:
            next_period = latest_period_int + 1
        with st.spinner(f"正在生成{cfg.name}统计预测 + 螺旋矩阵预测 (期{next_period})..."):
            _generate_both_algorithms(cfg, df, next_period)
        st.rerun()

    page = st.radio("", ["🎯 预测结果", "📜 预测历史"],
                    horizontal=True, label_visibility="collapsed", key=f"nav_{cfg.short}")

    # ──── 预测结果 ────
    if page == "🎯 预测结果":
        all_p = get_all_predictions(cfg)

        # 统计结果
        st.markdown(f"<h1>📊 统计算法 (Ensemble) {cfg.icon}</h1>", unsafe_allow_html=True)
        active_ens = [p for p in all_p if p["status"] == "active" and p.get("algorithm", "ensemble") == "ensemble"]
        lp_ens = active_ens[0] if active_ens else None

        if lp_ens is None:
            st.info("⏳ 等待生成统计算法预测...")
        else:
            p, nr, cr = lp_ens["period"], len(lp_ens["recommendations"]), lp_ens.get("created_at", "")
            dd = lp_ens.get("draw_date", cr[:10] if cr else "-")
            st.info(f"⏳ **期{p}** 未开奖 · 开奖后自动比对并生成下一期")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(metric_card(f"#{p}", "期号"), unsafe_allow_html=True)
            with c2: st.markdown(metric_card(f"{nr}", "推荐组数"), unsafe_allow_html=True)
            with c3: st.markdown(f"<div class='metric-card'><div class='value' style='font-size:1.2rem;color:#d97706'>⏳ 待开奖</div><div class='label'>状态</div></div>", unsafe_allow_html=True)
            with c4: st.markdown(metric_card(dd, "开奖日期"), unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            if dd and dd != "待定" and len(dd) == 10:
                try:
                    from datetime import datetime, timedelta
                    draw_dt = datetime.strptime(dd, "%Y-%m-%d")
                    deadline = draw_dt + timedelta(days=60)
                    st.info(f"📅 **兑奖截止:** {deadline.strftime('%Y-%m-%d')}（自开奖日起 **60天**）")
                except Exception:
                    pass
            st.markdown(f"<h3>📊 统计推荐号码 <span style='color:#64748b;font-weight:400;font-size:0.85rem;'>({nr} 组)</span></h3>", unsafe_allow_html=True)
            cols = st.columns(min(5, nr))
            for i, r in enumerate(lp_ens["recommendations"]):
                with cols[i]: st.markdown(pred_card(r, None, True, cfg), unsafe_allow_html=True)

        # 螺旋矩阵结果
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h1>🌀 螺旋矩阵算法 (覆盖设计)</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#7c3aed;font-size:0.85rem;'>基于 4-if-5/6 旋转矩阵覆盖设计，与统计算法完全独立的第二套计算方法</p>", unsafe_allow_html=True)

        active_sp = [p for p in all_p if p["status"] == "active" and p.get("algorithm") == "spiral_matrix"]
        lp_sp = active_sp[0] if active_sp else None

        if lp_sp is None:
            # 兜底: 实时计算一次
            with st.spinner("🌀 正在实时计算螺旋矩阵预测..."):
                try:
                    predictor = SpiralMatrixPredictor()
                    r_sp = predictor.predict(cfg, df, num_groups=5)
                    if r_sp.get("groups"):
                        lp_sp = {
                            "period": str(int(df.iloc[0]["period"]) + 1),
                            "recommendations": r_sp["groups"],
                        }
                except Exception as e:
                    st.warning(f"🌀 螺旋矩阵计算异常: {e}")

        if lp_sp and lp_sp.get("recommendations"):
            sp_groups = lp_sp["recommendations"]
            p_sp = lp_sp.get("period", "?")
            st.markdown(f"<h3>🌀 螺旋矩阵推荐号码 <span style='color:#64748b;font-weight:400;font-size:0.85rem;'>({len(sp_groups)} 组 · 期{p_sp})</span></h3>", unsafe_allow_html=True)
            cols2 = st.columns(min(5, len(sp_groups)))
            for i, r in enumerate(sp_groups):
                with cols2[i]: st.markdown(spiral_card(r, cfg), unsafe_allow_html=True)
            st.markdown("<div style='margin-top:0.5rem;padding:0.8rem 1rem;background:#faf5ff;border-radius:10px;border:1px solid #e9d5ff;font-size:0.8rem;color:#6b21a8;'>"
                        f"<b>🌀 螺旋矩阵说明:</b> 每组的 {cfg.main_count} 个前区号码由旋转矩阵从策略优势号池中生成"
                        f"（4-if-{cfg.main_count}覆盖保证），无区间形态限制，自由覆盖。"
                        f"<br><b>💡 比对方法:</b> 开奖后分别计算两组算法的命中率，可直观比较。"
                        f"</div>", unsafe_allow_html=True)
        else:
            st.info("🌀 暂无螺旋矩阵预测")

        # 重新计算按钮（含增量数据刷新）
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("🔄 重新计算（统计算法 + 螺旋矩阵）", key=f"recalc_{cfg.short}",
                      help="增量刷新最新开奖数据后，重新生成两种算法的预测"):
            # 第一步：增量数据刷新
            with st.spinner("正在检查最新开奖数据..."):
                from data_fetcher import fetch_from_500_html, fetch_ssq_cwl
                from pathlib import Path
                csv_path = Path(cfg.history_csv)
                if csv_path.exists():
                    local_df = pd.read_csv(csv_path)
                    local_periods = set(local_df["period"].astype(str).values)
                    latest_local = int(local_df.iloc[0]["period"])
                else:
                    local_df = pd.DataFrame()
                    local_periods = set()
                    latest_local = 0
                try:
                    if cfg.short == "ssq":
                        web_df = fetch_ssq_cwl(max_draws=10)
                    else:
                        web_df = fetch_from_500_html(cfg, max_draws=10)
                    if web_df is not None and not web_df.empty:
                        new_rows = web_df[~web_df["period"].astype(str).isin(local_periods)]
                        if not new_rows.empty:
                            merged = pd.concat([web_df, local_df[~local_df["period"].astype(str).isin(web_df["period"].astype(str))]], ignore_index=True)
                            merged = merged.drop_duplicates(subset=["period"])
                            merged = merged.sort_values("period", ascending=False).reset_index(drop=True)
                            merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
                            fresh_df = merged
                            st.info(f"📥 新增 {len(new_rows)} 期开奖数据（最新期 {int(web_df.iloc[0]['period'])}）")
                        else:
                            fresh_df = local_df
                            st.info(f"✅ 数据已是最新（最新期 {latest_local}），无需更新")
                    else:
                        fresh_df = local_df
                        st.info(f"⚠️ 网页拉取失败，使用本地数据（最新期 {latest_local}）")
                except Exception as e:
                    fresh_df = local_df
                    st.info(f"⚠️ 网络获取异常: {e}，使用本地数据（最新期 {latest_local}）")

            # 第二步：用最新数据重新生成两种算法
            with st.spinner("正在使用最新数据重新计算两种算法..."):
                md = {"f": FrequencyModel(cfg), "p": PoissonModel(cfg),
                      "e": ExponentialSmoothingModel(cfg, alpha=0.3), "m": MonteCarloModel(cfg)}
                md["m"].n_simulations = 20000
                mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in fresh_df.iterrows()])
                sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in fresh_df.iterrows()])
                for m in md.values(): m.fit(mn, sn)

                # 确定期号: 当前活跃预测的最大期号+1, 或最新数据期号+1
                all_p_now = get_all_predictions(cfg)
                active_periods = [int(p["period"]) for p in all_p_now if p["status"] == "active"]
                latest_data_period = int(fresh_df.iloc[0]["period"])
                if active_periods:
                    target_p = str(max(active_periods))
                    # 如果最新数据期号大于活跃期号, 说明开奖已过, 需要更新到下一期
                    if latest_data_period >= int(target_p):
                        target_p = str(latest_data_period + 1)
                else:
                    target_p = str(latest_data_period + 1)

                # 1) 统计算法
                ensemble = EnsembleModel(md, cfg)
                r2 = generate_recommendations(ensemble, cfg, num_groups=5, df=fresh_df)
                new_groups = r2.get("groups", [])
                all_p = get_all_predictions(cfg)
                remaining = [p for p in all_p if not (p["period"] == target_p and p.get("algorithm", "ensemble") == "ensemble")]
                from utils.predictions import _save_all
                _save_all(remaining, cfg)
                save_prediction(period=target_p, recommendations=new_groups, cfg=cfg, algorithm="ensemble")
                st.success("✅ 统计算法已重新生成（最新冷号分析）")

                # 2) 螺旋矩阵
                try:
                    predictor = SpiralMatrixPredictor()
                    r_sp = predictor.predict(cfg, fresh_df, num_groups=5)
                    if r_sp.get("groups"):
                        all_p2 = get_all_predictions(cfg)
                        remaining2 = [p for p in all_p2 if not (p["period"] == target_p and p.get("algorithm") == "spiral_matrix")]
                        from utils.predictions import _save_all
                        _save_all(remaining2, cfg)
                        save_prediction(period=target_p, recommendations=r_sp["groups"], cfg=cfg, algorithm="spiral_matrix")
                        st.success("🌀 螺旋矩阵已重新生成（最新冷号分析）")
                except Exception as e:
                    st.warning(f"🌀 螺旋矩阵刷新失败: {e}")
                st.rerun()

    # ──── 预测历史 ────
    elif page == "📜 预测历史":
        st.markdown(f"<h1>📜 预测历史记录</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size:1.1rem;color:#1e293b;margin-top:0.5rem;'>📋 最近10期开奖号码</h3>", unsafe_allow_html=True)
        recent_html = get_recent_draws_html(df, cfg, n=10)
        st.markdown(f"<div class='glass-card' style='padding:0.8rem 1rem;'>{recent_html}</div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        all_p = get_all_predictions(cfg)
        if not all_p:
            st.markdown("<div class='glass-card' style='text-align:center;padding:3rem;'><p style='color:#64748b;'>暂无预测历史记录</p></div>", unsafe_allow_html=True)
        else:
            # 按算法分组显示
            ensemble_hist = [p for p in all_p if p.get("algorithm", "ensemble") == "ensemble"]
            spiral_hist = [p for p in all_p if p.get("algorithm") == "spiral_matrix"]

            st.markdown(f"<p style='color:#475569;'>📊 统计算法: {len(ensemble_hist)} 条记录 · 🌀 螺旋矩阵: {len(spiral_hist)} 条记录</p>", unsafe_allow_html=True)

            # 统计历史
            if ensemble_hist:
                st.markdown("<h3 style='font-size:1rem;color:#dc2626;'>📊 统计算法历史</h3>", unsafe_allow_html=True)
                for pred in ensemble_hist:
                    ps, pp, pc, pr_ = pred["status"], pred["period"], pred.get("draw_date", pred.get("created_at","")[:10] or ""), len(pred.get("recommendations",[]))
                    if ps == "completed" and pred.get("summary"):
                        s = pred["summary"]
                        with st.expander(f"**期 {pp}** ✅ 已比对 · 最佳 {s['best_hits']} 个 · {pc}", expanded=False):
                            if pred.get("actual_draw"):
                                am = "".join(ball(n,"main") for n in pred["actual_draw"]["main"])
                                as_ = "".join(ball(n,"sub") for n in pred["actual_draw"]["sub"])
                                st.markdown(f"<p style='color:#94a3b8;'>📌 实际开奖: {am} {as_}</p>", unsafe_allow_html=True)
                            ca,cb,cc = st.columns(3)
                            with ca: st.markdown(metric_card(f"{s['best_hits']} 个", "最佳命中"), unsafe_allow_html=True)
                            with cb: st.markdown(metric_card(f"{s['avg_main_hits']}", f"{cfg.main_label}平均"), unsafe_allow_html=True)
                            with cc: st.markdown(metric_card(f"{s['avg_sub_hits']}", f"{cfg.sub_label}平均"), unsafe_allow_html=True)
                            if pred.get("matches"):
                                st.markdown("<h4 style='font-size:0.95rem;margin:1rem 0 0.5rem;color:#475569;'>各组预测号码</h4>", unsafe_allow_html=True)
                                mm = {m["group"]: m for m in pred["matches"]}
                                for r in pred["recommendations"]: st.markdown(hist_row(r, mm, cfg), unsafe_allow_html=True)
                    else:
                        with st.expander(f"**期 {pp}** ⏳ 待开奖 · {pr_}组 · {pc}", expanded=False):
                            if pred.get("recommendations"):
                                for r in pred["recommendations"]: st.markdown(hist_row(r, None, cfg), unsafe_allow_html=True)

            # 螺旋矩阵历史
            if spiral_hist:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h3 style='font-size:1rem;color:#7c3aed;'>🌀 螺旋矩阵历史</h3>", unsafe_allow_html=True)
                for pred in spiral_hist:
                    ps, pp, pc, pr_ = pred["status"], pred["period"], pred.get("draw_date", pred.get("created_at","")[:10] or ""), len(pred.get("recommendations",[]))
                    if ps == "completed" and pred.get("summary"):
                        s = pred["summary"]
                        with st.expander(f"🌀 **期 {pp}** ✅ 已比对 · 最佳 {s['best_hits']} 个 · {pc}", expanded=False):
                            if pred.get("actual_draw"):
                                am = "".join(ball(n,"main") for n in pred["actual_draw"]["main"])
                                as_ = "".join(ball(n,"sub") for n in pred["actual_draw"]["sub"])
                                st.markdown(f"<p style='color:#94a3b8;'>📌 实际开奖: {am} {as_}</p>", unsafe_allow_html=True)
                            ca,cb,cc = st.columns(3)
                            with ca: st.markdown(metric_card(f"{s['best_hits']} 个", "最佳命中"), unsafe_allow_html=True)
                            with cb: st.markdown(metric_card(f"{s['avg_main_hits']}", f"{cfg.main_label}平均"), unsafe_allow_html=True)
                            with cc: st.markdown(metric_card(f"{s['avg_sub_hits']}", f"{cfg.sub_label}平均"), unsafe_allow_html=True)
                            if pred.get("matches"):
                                st.markdown("<h4 style='font-size:0.95rem;margin:1rem 0 0.5rem;color:#475569;'>各组预测号码</h4>", unsafe_allow_html=True)
                                mm = {m["group"]: m for m in pred["matches"]}
                                for r in pred["recommendations"]: st.markdown(hist_row(r, mm, cfg), unsafe_allow_html=True)
                    else:
                        with st.expander(f"🌀 **期 {pp}** ⏳ 待开奖 · {pr_}组 · {pc}", expanded=False):
                            if pred.get("recommendations"):
                                for r in pred["recommendations"]: st.markdown(hist_row(r, None, cfg), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    print_disclaimer(cfg)
    st.markdown(f"<p style='text-align:center;font-size:0.7rem;color:#94a3b8;margin-top:1rem;'>{cfg.short.upper()} v2.0 · Python · Streamlit · scikit-learn · 螺旋矩阵 v1.0</p>", unsafe_allow_html=True)


# ═══════ 主界面 ═══════
tab1, tab2 = st.tabs(["🎯 大乐透", "🔴 双色球"])

with tab1:
    render_lottery(DLT)

with tab2:
    render_lottery(SSQ)
