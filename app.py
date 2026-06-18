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
/* Tabs 样式 */
.stTabs [data-baseweb="tab-list"] { gap: 2px; background: #fff; border-radius: 10px; padding: 4px; border: 1px solid #e2e8f0; margin-bottom: 1rem; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 0.5rem 1.5rem; font-weight: 600; font-size: 0.9rem; color: #64748b; }
.stTabs [aria-selected="true"] { background: #eff6ff !important; color: #1d4ed8 !important; border-color: #2563eb !important; }
/* 卡片 */
.glass-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.glass-card.glow { border-color: #2563eb; box-shadow: 0 2px 8px rgba(37,99,235,0.10); }
.metric-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem; text-align: center; }
.metric-card .value { font-size: 1.8rem; font-weight: 800; color: #1e293b; line-height: 1.2; }
.metric-card .label { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.25rem; }
/* 号码球 */
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
    
    # Strategy badge
    reason = rec.get("reason", "")
    strategy_name = reason.split(":")[0] if ":" in reason else "综合推荐"
    strategy_icons = {
        "高频热号": "🔥", "近期趋势": "📈", "追冷": "❄️",
        "马尔可夫": "🔗", "平衡策略": "⚖️",
    }
    strategy_colors = {
        "高频热号": "#dc2626", "近期趋势": "#059669", "追冷": "#2563eb",
        "马尔可夫": "#7c3aed", "平衡策略": "#d97706",
    }
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
            f"<span style='font-size:0.7rem;font-weight:600;color:#94a3b8;'>#{g}</span>"
            f"</div>"
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
    # Strategy badge for history
    reason = rec.get("reason", "")
    strategy_name = reason.split(":")[0] if ":" in reason else ""
    strategy_colors = {"高频热号":"#dc2626","近期趋势":"#059669","追冷":"#2563eb","马尔可夫":"#7c3aed","平衡策略":"#d97706"}
    strategy_icons = {"高频热号":"🔥","近期趋势":"📈","追冷":"❄️","马尔可夫":"🔗","平衡策略":"⚖️"}
    scolor = strategy_colors.get(strategy_name, "#94a3b8")
    sicon = strategy_icons.get(strategy_name, "")
    strat_tag = f"<span style='background:{scolor};color:#fff;border-radius:4px;padding:0.1rem 0.4rem;font-size:0.6rem;font-weight:700;'>{sicon} {strategy_name}</span>" if strategy_name else ""
    return (f"<div style='background:#fafbfc;border:1px solid #e2e8f0;border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:0.4rem;display:flex;align-items:center;flex-wrap:wrap;gap:0.5rem;'>"
            f"<span style='font-weight:700;color:#94a3b8;font-size:0.75rem;min-width:2.5rem;'>第{g}组</span>"
            f"{strat_tag}"
            f"<span style='font-size:0.7rem;color:#94a3b8;'>{cfg.main_label}</span>{mb}"
            f"<span style='font-size:0.7rem;color:#94a3b8;margin-left:0.3rem;'>{cfg.sub_label}</span>{sb}"
            f"{'&nbsp;'+bdg if bdg else ''}</div>")


@st.cache_data(ttl=3600, show_spinner="正在加载数据...")
def load_lottery(cfg_key, _mtime=0):
    """加载数据，_mtime强制缓存随文件修改时间失效"""
    _cfg = DLT if cfg_key == "dlt" else SSQ
    return update_data(_cfg)


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

    # ── 逾期自动检查与手动强制比对 ──
    from datetime import datetime as _dt
    _now = _dt.now()
    _init_key = f"_overdue_inited_{cfg.short}"
    if _init_key not in st.session_state:
        st.session_state[_init_key] = False

    # 手动强制检查按钮（放在顶部，明显可见）
    col_a, col_b = st.columns([3, 1])
    with col_b:
        force_btn = st.button("🔍 强制检查逾期比对", key=f"force_{cfg.short}",
                              help=f"强制从网页拉取最新{cfg.name}开奖数据并自动比对",
                              use_container_width=True)

    # 收集当前 active 预测中不在数据里的期号
    active_missing = []
    for p in get_all_predictions(cfg):
        if p["status"] == "active":
            pp = str(p["period"])
            if pp not in set(df["period"].astype(str).values):
                active_missing.append(pp)

    should_check = force_btn  # 手动按钮总是触发
    if not should_check and not st.session_state[_init_key] and active_missing:
        # 自动检查：判断当前时间是否已过开奖时间
        _wd = _now.weekday()
        _h, _m = _now.hour, _now.minute
        _dh, _dm = map(int, cfg.draw_time.split(':'))
        _draw_passed = (_h > _dh or (_h == _dh and _m >= _dm))

        if _wd in cfg.draw_days and _draw_passed:
            should_check = True  # 今天是开奖日，已过开奖时间
        _yesterday = (_wd - 1) % 7
        if _yesterday in cfg.draw_days and not _draw_passed:
            should_check = True  # 昨天开奖了，今天还没到下次开奖时间

    st.session_state[_init_key] = True

    if should_check:
        if active_missing:
            with st.spinner(f"🔍 正在联网获取最新{cfg.name}开奖数据并自动比对..."):
                fresh_df, count, msgs = force_check_overdue(cfg)
                if fresh_df is not None and not fresh_df.empty:
                    df = fresh_df  # 即使 count=0 也要更新最新数据
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
                    df = fresh_df  # 即使 count=0 也要更新最新数据
                if count > 0:
                    for msg in msgs:
                        st.success(msg)
                    st.rerun()
                else:
                    for msg in msgs:
                        st.info(msg)

    # ── 确保当前始终有未开奖的活跃预测 ──
    # 比对完毕后（预测变为completed），自动生成下一期
    all_preds_now = get_all_predictions(cfg)
    active_now = [p for p in all_preds_now if p["status"] == "active"]
    
    if not active_now:
        # 确定下一期期号
        completed_periods = [int(p["period"]) for p in all_preds_now if p["status"] == "completed"]
        latest_data_period_int = int(df.iloc[0]["period"])
        if completed_periods:
            next_period = max(completed_periods) + 1
            # 如果最新数据期号已经超过了最新已完成预测的下一期
            if latest_data_period_int >= next_period:
                next_period = latest_data_period_int + 1
        else:
            # 完全没有历史预测，以最新数据期号为基准
            next_period = latest_data_period_int + 1
        
        # 检查是否已存在该期号的预测（避免重复生成）
        all_periods = set(p["period"] for p in all_preds_now)
        if str(next_period) not in all_periods:
            with st.spinner(f"检测到第{next_period}期待预测，正在生成{cfg.name}预测..."):
                ensemble = EnsembleModel({}, cfg)
                r2 = generate_recommendations(ensemble, cfg, num_groups=5, df=df)
                save_prediction(period=str(next_period), recommendations=r2.get("groups",[]), cfg=cfg)
                st.rerun()

    # 子页面导航
    page = st.radio("", ["🎯 预测结果", "📜 预测历史"], 
                    horizontal=True, label_visibility="collapsed", key=f"nav_{cfg.short}")
    num_groups = 5

    # ──── 预测结果 ────
    if page == "🎯 预测结果":
        st.markdown(f"<h1>{cfg.icon} 当前预测结果</h1>", unsafe_allow_html=True)
        # 始终显示活跃的（未开奖）预测
        all_p = get_all_predictions(cfg)
        active_p = [p for p in all_p if p["status"] == "active"]
        lp = active_p[0] if active_p else None
        
        if lp is None:
            with st.spinner(f"正在首次生成{cfg.name}预测..."):
                md = {"f": FrequencyModel(cfg), "p": PoissonModel(cfg),
                      "e": ExponentialSmoothingModel(cfg, alpha=0.3), "m": MonteCarloModel(cfg)}
                md["m"].n_simulations = 20000
                mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
                sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
                for m in md.values(): m.fit(mn, sn)
                ensemble = EnsembleModel(md, cfg)
                r2 = generate_recommendations(ensemble, cfg, num_groups=5, df=df)
                save_prediction(period=str(int(df.iloc[0]["period"])+1), recommendations=r2.get("groups",[]), cfg=cfg)
                st.rerun()
        else:
            p, s, nr, cr = lp["period"], lp["status"], len(lp["recommendations"]), lp.get("created_at","")
            dd = lp.get("draw_date", cr[:10] if cr else "-")
            st.info(f"⏳ **期{p}** 未开奖 · 开奖后自动比对并生成下一期")
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown(metric_card(f"#{p}", "期号"), unsafe_allow_html=True)
            with c2: st.markdown(metric_card(f"{nr}", "推荐组数"), unsafe_allow_html=True)
            with c3: st.markdown(f"<div class='metric-card'><div class='value' style='font-size:1.2rem;color:#d97706'>⏳ 待开奖</div><div class='label'>状态</div></div>", unsafe_allow_html=True)
            with c4: st.markdown(metric_card(dd, "开奖日期"), unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            # 兑奖截止日期（60天）
            if dd and dd != "待定" and len(dd) == 10:
                try:
                    from datetime import datetime, timedelta
                    draw_dt = datetime.strptime(dd, "%Y-%m-%d")
                    deadline = draw_dt + timedelta(days=60)
                    st.info(f"📅 **兑奖截止:** {deadline.strftime('%Y-%m-%d')}（自开奖日起 **60天**）")
                except Exception:
                    pass

            st.markdown(f"<h3>推荐号码 <span style='color:#64748b;font-weight:400;font-size:0.85rem;'>({nr} 组)</span></h3>", unsafe_allow_html=True)
            cols = st.columns(min(5, len(lp["recommendations"])))
            for i, r in enumerate(lp["recommendations"]):
                with cols[i]: st.markdown(pred_card(r, None, True, cfg), unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            if st.button("🔄 重新计算", key=f"recalc_{cfg.short}", help="用最新算法重新评估，排名有变化才更新"):
                with st.spinner("正在重新计算..."):
                    # 用最新算法重新生成
                    md = {"f": FrequencyModel(cfg), "p": PoissonModel(cfg),
                          "e": ExponentialSmoothingModel(cfg, alpha=0.3), "m": MonteCarloModel(cfg)}
                    md["m"].n_simulations = 20000
                    mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
                    sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
                    for m in md.values(): m.fit(mn, sn)
                    ensemble = EnsembleModel(md, cfg)
                    r2 = generate_recommendations(ensemble, cfg, num_groups=5, df=df)
                    new_groups = r2.get("groups", [])

                    # 确定要保存的期号：保持活动预测原有期号，不要用最新数据期号
                    current_pred = get_latest_prediction(cfg) if get_latest_prediction(cfg) else None
                    if current_pred and current_pred.get("status") == "active":
                        target_p = current_pred["period"]
                    else:
                        target_p = str(int(df.iloc[0]["period"]) + 1)

                    if current_pred and current_pred.get("recommendations"):
                        old_groups = current_pred["recommendations"]
                        merged_groups = []
                        changed_count = 0
                        max_groups = max(len(old_groups), len(new_groups))

                        for i in range(max_groups):
                            old_g = old_groups[i] if i < len(old_groups) else None
                            new_g = new_groups[i] if i < len(new_groups) else None

                            if old_g is None:
                                merged_groups.append(new_g)
                                changed_count += 1
                            elif new_g is None:
                                merged_groups.append(old_g)
                            else:
                                old_key = (tuple(sorted(old_g["main"])), tuple(sorted(old_g["sub"])))
                                new_key = (tuple(sorted(new_g["main"])), tuple(sorted(new_g["sub"])))
                                old_score = old_g.get("score", 0)
                                new_score = new_g.get("score", 0)

                                if old_key == new_key:
                                    # 号码完全相同, 分数也可能细微变化, 保留旧的
                                    merged_groups.append(old_g)
                                elif new_score > old_score:
                                    # 号码变了且分数提高了 → 替换这一组
                                    merged_groups.append(new_g)
                                    changed_count += 1
                                else:
                                    # 号码变了但分数没提高 → 保持旧的
                                    merged_groups.append(old_g)

                        if changed_count > 0:
                            all_p = get_all_predictions(cfg)
                            remaining = [p for p in all_p if p["period"] != target_p]
                            from utils.predictions import _save_all
                            _save_all(remaining, cfg)
                            save_prediction(period=target_p, recommendations=merged_groups, cfg=cfg)
                            st.success(f"✅ {changed_count} 组有优化更新，推荐号码已刷新")
                        else:
                            st.info("ℹ️ 算法评分排名无变化，推荐号码保持不变")
                    else:
                        # 没有旧预测，直接保存
                        save_prediction(period=target_p, recommendations=new_groups, cfg=cfg)
                        st.success("✅ 预测已生成")
                    st.rerun()



    # ──── 预测历史 ────
    elif page == "📜 预测历史":
        st.markdown("<h1>📜 预测历史记录</h1>", unsafe_allow_html=True)

        # ── 最近10期实际开奖号码 ──
        st.markdown("<h3 style='font-size:1.1rem;color:#1e293b;margin-top:0.5rem;'>📋 最近10期开奖号码</h3>", unsafe_allow_html=True)
        recent_html = get_recent_draws_html(df, cfg, n=10)
        st.markdown(f"<div class='glass-card' style='padding:0.8rem 1rem;'>{recent_html}</div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        all_p = get_all_predictions(cfg)
        if not all_p: st.markdown("<div class='glass-card' style='text-align:center;padding:3rem;'><p style='color:#64748b;'>暂无预测历史记录</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:#475569;'>{len(all_p)} 条预测记录</p>", unsafe_allow_html=True)
            for pred in all_p:
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



    st.markdown("<hr>", unsafe_allow_html=True)
    print_disclaimer(cfg)
    st.markdown(f"<p style='text-align:center;font-size:0.7rem;color:#94a3b8;margin-top:1rem;'>{cfg.short.upper()} v2.0 · Python · Streamlit · scikit-learn</p>", unsafe_allow_html=True)


# ═══════ 主界面 ═══════
tab1, tab2 = st.tabs(["🎯 大乐透", "🔴 双色球"])

with tab1:
    render_lottery(DLT)

with tab2:
    render_lottery(SSQ)
