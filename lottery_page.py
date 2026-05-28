"""
彩票页面组件 - 含 CSS、辅助函数、完整 4 页面渲染
"""
import sys, os
from io import StringIO
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

from data_fetcher import update_data
from analysis.statistics import comprehensive_analysis
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
from models.ml_models import LSTMSequenceModel, XGBoostModel, RandomForestModel
from models.ensemble import EnsembleModel, generate_recommendations
from backtest.backtester import Backtester
from utils.predictions import (
    get_all_predictions, get_latest_prediction, save_prediction, auto_compare_latest
)


def apply_css():
    """注入通用 CSS"""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
.stApp { background: #f0f2f5; }
header[data-testid="stHeader"], .stApp > header { display: none; }
html, body, [data-testid="stAppViewContainer"] { color: #1e293b; font-family: 'Inter', -apple-system, sans-serif; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #e2e8f0; }
::-webkit-scrollbar-thumb { background: #94a3b8; border-radius: 3px; }
h1, h2, h3 { font-weight: 700; letter-spacing: -0.02em; }
[data-testid="stAppViewBlockContainer"] { max-width: 1400px; padding: 1.5rem 2rem; }

/* ── 侧边栏 ── */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #475569; }
[data-testid="stSidebar"] .sidebar-brand { padding: 1rem 0 0.5rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem; }
[data-testid="stSidebar"] .sidebar-brand h2 { color: #1e293b; font-size: 1.3rem; margin: 0; }
[data-testid="stSidebar"] .sidebar-brand p { font-size: 0.75rem; color: #94a3b8; margin: 0.2rem 0 0 0; }

/* ── 卡片 ── */
.glass-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04); transition: all 0.2s ease; }
.glass-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.06); border-color: #cbd5e1; }
.glass-card.glow { border-color: #2563eb; box-shadow: 0 2px 8px rgba(37,99,235,0.10); }
.metric-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem 1.25rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.metric-card .value { font-size: 1.8rem; font-weight: 800; color: #1e293b; line-height: 1.2; }
.metric-card .label { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.25rem; }

/* ── 号码球 ── */
.number-ball { display: inline-flex; align-items: center; justify-content: center; width: 46px; height: 46px; border-radius: 50%; font-size: 1rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin: 0 3px; }
.number-ball.main { background: linear-gradient(135deg, #dc2626, #b91c1c); color: #fff; box-shadow: 0 2px 6px rgba(220,38,38,0.25); }
.number-ball.sub { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: #fff; box-shadow: 0 2px 6px rgba(37,99,235,0.25); }
.number-ball.hit { box-shadow: 0 0 0 3px #fff, 0 0 0 5px #eab308; border: none; }

/* ── 按钮 ── */
.stButton > button, div[data-testid="stButton"] > button { border-radius: 8px; font-weight: 600; font-size: 0.85rem; transition: all 0.2s ease; border: 1px solid #d1d5db; background: #ffffff; color: #1e293b; }
.stButton > button:hover { border-color: #2563eb; background: #f8fafc; color: #2563eb; }
div[data-testid="stButton"] > button[kind="primary"] { background: #2563eb; border: none; color: #fff; box-shadow: 0 2px 8px rgba(37,99,235,0.2); }
div[data-testid="stButton"] > button[kind="primary"]:hover { background: #1d4ed8; box-shadow: 0 4px 14px rgba(37,99,235,0.3); transform: translateY(-1px); }
.stSlider > div > div > div { background: #2563eb !important; }

/* ── 导航选择栏 ── */
.stRadio div[role="radiogroup"] { gap: 0.25rem; }
.stRadio div[role="radiogroup"] label { border-radius: 8px; padding: 0.5rem 0.75rem; background: #ffffff; border: 1px solid #e2e8f0; transition: all 0.2s ease; font-size: 0.85rem; }
.stRadio div[role="radiogroup"] label:hover { border-color: #93c5fd; background: #f0f7ff; }
.stRadio div[role="radiogroup"] label[data-checked="true"] { border-color: #2563eb; background: #eff6ff; color: #1d4ed8; font-weight: 600; }
.streamlit-expanderHeader { background: #ffffff !important; border-radius: 10px !important; border: 1px solid #e2e8f0 !important; }
.streamlit-expanderHeader:hover { border-color: #93c5fd !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrame"] thead tr th { background: #f8fafc !important; color: #64748b !important; font-size: 0.75rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid #e2e8f0 !important; }
[data-testid="stDataFrame"] tbody tr { background: #ffffff !important; }
[data-testid="stDataFrame"] tbody tr:nth-child(even) { background: #f8fafc !important; }
[data-testid="stDataFrame"] tbody td { border-bottom: 1px solid #f1f5f9 !important; color: #334155 !important; font-size: 0.8rem !important; }

/* ── Alert ── */
[data-testid="stAlert"] { border-radius: 10px; border-left: 4px solid #2563eb; background: #ffffff !important; }
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] { color: #1e293b; }
.stAlert.success { border-left-color: #059669; } .stAlert.info { border-left-color: #2563eb; }
.stAlert.warning { border-left-color: #d97706; } .stAlert.error { border-left-color: #dc2626; }
hr { border: none; height: 1px; background: #e2e8f0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── 辅助函数 ──────────────────────────────────────────────────
def render_ball(n, kind="main", is_hit=False):
    css = f"number-ball {kind}"
    if is_hit:
        css += " hit"
    return f"<span class='{css}'>{n:02d}</span>"


def render_metric(value, label, delta=None):
    d = f"<div style='font-size:0.7rem;color:#059669;margin-top:0.15rem'>{delta}</div>" if delta else ""
    return f"<div class='metric-card'><div class='value'>{value}</div><div class='label'>{label}</div>{d}</div>"


def render_prediction_card(rec, matches=None, is_active=True, cfg=None):
    g = rec.get("index", rec.get("group", 1))
    gm = matches.get(g, {}) if matches else {}
    mh = set(gm.get("main_matches", [])) if gm else set()
    sh = set(gm.get("sub_matches", [])) if gm else set()
    main_balls = "".join(render_ball(n, "main", n in mh) for n in rec["main"])
    sub_balls = "".join(render_ball(n, "sub", n in sh) for n in rec["sub"])
    score_pct = rec["score"] * 100
    hit_text = ""
    if gm and gm.get("total_hits", 0) > 0:
        m = gm
        hit_text = (f"<div style='margin-top:0.5rem;font-size:0.85rem;color:#eab308;font-weight:600;'>🎯 命中 {m['main_hits']}{cfg.main_label} + {m['sub_hits']}{cfg.sub_label} = {m['total_hits']}个</div>")
    elif is_active:
        hit_text = f"<div style='margin-top:0.5rem;font-size:0.85rem;color:#64748b;'>⏳ 待开奖</div>"
    return (f"<div class='glass-card glow pred-card' style='padding:1rem;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;'>"
            f"<span style='font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;margin-bottom:0.5rem;'>第 {g} 组</span>"
            f"<span style='font-size:0.75rem;color:#2563eb;font-weight:600;font-family:JetBrains Mono,monospace;'>综合得分 {score_pct:.2f}%</span></div>"
            f"<div style='margin-bottom:0.5rem;'><div style='font-size:0.7rem;color:#64748b;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.05em;'>{cfg.main_label} · {cfg.main_count}码</div><div>{main_balls}</div></div>"
            f"<div style='margin-bottom:0.5rem;'><div style='font-size:0.7rem;color:#64748b;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.05em;'>{cfg.sub_label} · {cfg.sub_count}码</div><div>{sub_balls}</div></div>"
            f"<div style='font-size:0.7rem;color:#64748b;margin-top:0.35rem;'>📊 {rec.get('reason', '综合评分 ' + str(round(rec['score'], 1)))}</div>{hit_text}</div>")


def render_history_group(rec, matches_map=None, cfg=None):
    g = rec.get("index", rec.get("group", 1))
    m = matches_map.get(g, {}) if matches_map else {}
    mh = set(m.get("main_matches", [])) if m else set()
    sh = set(m.get("sub_matches", [])) if m else set()
    main_balls = "".join(render_ball(n, "main", n in mh) for n in rec["main"])
    sub_balls = "".join(render_ball(n, "sub", n in sh) for n in rec["sub"])
    total = m.get("total_hits", 0) if m else 0
    badge = f"<span style='color:#eab308;font-weight:700;font-size:0.85rem;'>🎯 命中 {total}个</span>" if total > 0 else ""
    return (f"<div style='background:#fafbfc;border:1px solid #e2e8f0;border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:0.4rem;display:flex;align-items:center;flex-wrap:wrap;gap:0.5rem;'>"
            f"<span style='font-weight:700;color:#94a3b8;font-size:0.75rem;min-width:2.5rem;'>第{g}组</span>"
            f"<span style='font-size:0.7rem;color:#94a3b8;'>{cfg.main_label}</span>{main_balls}"
            f"<span style='font-size:0.7rem;color:#94a3b8;margin-left:0.3rem;'>{cfg.sub_label}</span>{sub_balls}"
            f"{'&nbsp;' + badge if badge else ''}</div>")


# ── 数据加载 ──────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="正在加载数据...")
def load_data_for(cfg_key):
    from config_dlt import DLT
    from config_ssq import SSQ
    _cfg = DLT if cfg_key == "dlt" else SSQ
    return update_data(_cfg)


# ── 主渲染函数 ───────────────────────────────────────────────
def render_page(cfg):
    """渲染单个彩种的全部 4 个页面"""
    apply_css()

    # 数据加载
    with st.spinner(f"正在加载{cfg.name}历史数据..."):
        df = load_data_for(cfg.short)
    if df is None or len(df) == 0:
        st.error(f"无法获取{cfg.name}数据")
        st.stop()

    # 侧边栏
    with st.sidebar:
        st.markdown(
            f"<div class='sidebar-brand'><h2>{cfg.icon} {cfg.name}</h2>"
            f"<p>{cfg.total_combinations():,}种组合 · {cfg.short.upper()}</p></div>",
            unsafe_allow_html=True)

        st.markdown("<p style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;'>导航</p>", unsafe_allow_html=True)
        page = st.radio("nav", ["🎯 预测结果", "📊 数据分析", "📜 预测历史", "📈 回测"],
                        label_visibility="collapsed", index=0)

        st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;'>控制面板</p>", unsafe_allow_html=True)
        if st.button("🔄 刷新数据", use_container_width=True):
            st.cache_data.clear(); st.rerun()
        num_groups = st.slider("推荐组数", 1, 20, cfg.recommend_groups)
        st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)
        st.caption("⚠️ 仅供娱乐参考 · 理性购彩")
        st.sidebar.success(f"✅ 已加载 {len(df):,} 期数据")

    auto_compare_latest(df, cfg)

    # ═══════ 预测结果 ═══════
    if page == "🎯 预测结果":
        latest_pred = get_latest_prediction(cfg)
        st.markdown(f"<h1 style='background:linear-gradient(135deg,#2563eb 0%,#1d4ed8 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2rem;font-weight:800;'>{cfg.icon} 当前预测结果</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#475569;font-size:0.85rem;margin-bottom:1.5rem;'>{cfg.name} · 多模型集成 · 冻结不刷新</p>", unsafe_allow_html=True)

        if latest_pred is None:
            st.markdown("<div class='glass-card' style='text-align:center;padding:3rem;'>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#64748b;'>暂无预测记录</p>", unsafe_allow_html=True)
            if st.button("🚀 生成首期预测", type="primary", use_container_width=True):
                with st.spinner("正在生成预测..."):
                    md = {"frequency": FrequencyModel(cfg), "poisson": PoissonModel(cfg),
                          "exponential_smoothing": ExponentialSmoothingModel(cfg, alpha=0.3),
                          "monte_carlo": MonteCarloModel(cfg)}
                    md["monte_carlo"].n_simulations = 20000
                    mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
                    sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
                    for m in md.values(): m.fit(mn, sn)
                    ensemble = EnsembleModel(md, cfg)
                    r2 = generate_recommendations(ensemble, cfg, num_groups=num_groups)
                    recs = r2.get("groups", [])
                    next_period = str(int(df.iloc[0]["period"]) + 1)
                    save_prediction(period=next_period, recommendations=recs, cfg=cfg)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            period = latest_pred["period"]
            status = latest_pred["status"]
            num_recs = len(latest_pred["recommendations"])
            created = latest_pred.get("created_at", "")
            if status == "completed":
                s = latest_pred["summary"]
                st.success(f"✅ **期 {period}** 已开奖 · 最佳命中 **{s['best_hits']}** 个（第 {s['best_group']} 组）")
            else:
                st.info(f"⏳ **期 {period}** 预测已封存 · 开奖后自动比对")

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(render_metric(f"#{period}", "期号"), unsafe_allow_html=True)
            with c2: st.markdown(render_metric(f"{num_recs}", "推荐组数"), unsafe_allow_html=True)
            with c3:
                sl = "✅ 已比对" if status == "completed" else "⏳ 待开奖"
                sc = "#059669" if status == "completed" else "#d97706"
                st.markdown(f"<div class='metric-card'><div class='value' style='font-size:1.2rem;color:{sc};'>{sl}</div><div class='label'>状态</div></div>", unsafe_allow_html=True)
            with c4: st.markdown(render_metric(created[:10] if created else "-", "日期"), unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)

            if status == "completed" and latest_pred.get("actual_draw"):
                act = latest_pred["actual_draw"]
                st.markdown("<h3 style='font-size:1.1rem;color:#1e293b;'>实际开奖号码</h3>", unsafe_allow_html=True)
                am = "".join(render_ball(n, "main") for n in act["main"])
                as_ = "".join(render_ball(n, "sub") for n in act["sub"])
                st.markdown(f"<div><span style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-right:0.5rem;'>{cfg.main_label}</span>{am}</div><div><span style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-right:0.5rem;'>{cfg.sub_label}</span>{as_}</div>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown(f"<h3 style='font-size:1.1rem;color:#1e293b;'>推荐号码 <span style='color:#64748b;font-weight:400;font-size:0.85rem;'>({num_recs} 组)</span></h3>", unsafe_allow_html=True)
            matches = {m["group"]: m for m in (latest_pred.get("matches") or [])}
            recs = latest_pred["recommendations"]
            cols = st.columns(min(5, len(recs)))
            for i, rec in enumerate(recs):
                with cols[i]:
                    st.markdown(render_prediction_card(rec, matches, status != "completed", cfg), unsafe_allow_html=True)

            if status == "completed" and latest_pred.get("summary"):
                s = latest_pred["summary"]
                st.markdown("<hr>", unsafe_allow_html=True)
                c_a, c_b, c_c, c_d = st.columns(4)
                with c_a: st.markdown(render_metric(f"{s['best_hits']} 个", "最佳命中", f"第{s['best_group']}组"), unsafe_allow_html=True)
                with c_b: st.markdown(render_metric(f"{s['avg_main_hits']:.1f}", f"{cfg.main_label}平均"), unsafe_allow_html=True)
                with c_c: st.markdown(render_metric(f"{s['avg_sub_hits']:.1f}", f"{cfg.sub_label}平均"), unsafe_allow_html=True)
                with c_d: st.markdown(render_metric(f"{s['avg_total_hits']:.1f}", "综合平均"), unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            latest_period = str(df.iloc[0]["period"])
            has_pred = any(p["period"] == latest_period and p["status"] in ("active", "completed") for p in get_all_predictions(cfg))
            if has_pred: st.info(f"✅ 期 {latest_period} 已有预测记录")
            if st.button("📅 生成下一期预测", type="primary", use_container_width=True):
                next_period = str(int(df.iloc[0]["period"]) + 1)
                existing = [p for p in get_all_predictions(cfg) if p["period"] == next_period]
                if existing: st.warning(f"期 {next_period} 已有预测")
                else:
                    with st.spinner("正在生成新预测..."):
                        md = {"frequency": FrequencyModel(cfg), "poisson": PoissonModel(cfg),
                              "exponential_smoothing": ExponentialSmoothingModel(cfg, alpha=0.3),
                              "monte_carlo": MonteCarloModel(cfg)}
                        md["monte_carlo"].n_simulations = 20000
                        mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
                        sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
                        for m in md.values(): m.fit(mn, sn)
                        ensemble = EnsembleModel(md, cfg)
                        r2 = generate_recommendations(ensemble, cfg, num_groups=num_groups)
                        recs_new = r2.get("groups", [])
                        save_prediction(period=next_period, recommendations=recs_new, cfg=cfg)
                        st.rerun()

    # ═══════ 数据分析 ═══════
    elif page == "📊 数据分析":
        st.markdown(f"<h1 style='background:linear-gradient(135deg,#2563eb 0%,#1d4ed8 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2rem;font-weight:800;'>📊 {cfg.name}数据分析</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#475569;font-size:0.85rem;margin-bottom:1.5rem;'>基于 {len(df):,} 期历史数据</p>", unsafe_allow_html=True)
        analysis = comprehensive_analysis(df, cfg)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(render_metric(f"{len(df):,}", "历史期数"), unsafe_allow_html=True)
        with c2: st.markdown(render_metric(" ".join(f"{n:02d}" for n in analysis["hot_cold"]["main_hot"][:5]), f"{cfg.main_label}热号 TOP5"), unsafe_allow_html=True)
        with c3: st.markdown(render_metric(" ".join(f"{n:02d}" for n in analysis["hot_cold"]["main_cold"][:3]), f"{cfg.main_label}冷号 TOP3"), unsafe_allow_html=True)
        with c4: st.markdown(render_metric(f"{len(analysis['hot_cold']['main_hot'])}", "热号总数"), unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(f"<h3 style='font-size:1.1rem;color:#1e293b;'>🔢 号码频率分析</h3>", unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            fq = analysis["frequency"]
            mf = pd.DataFrame([(n, fq["main_frequencies"][n], fq["main_frequency_pct"][n]) for n in range(cfg.main_min, cfg.main_max + 1)], columns=["号码", "出现次数", "频率(%)"]).sort_values("出现次数", ascending=False).reset_index(drop=True)
            mf.index = mf.index + 1
            st.dataframe(mf, width='stretch', height=360)
        with cb:
            sf = pd.DataFrame([(n, fq["sub_frequencies"][n], fq["sub_frequency_pct"][n]) for n in range(cfg.sub_min, cfg.sub_max + 1)], columns=["号码", "出现次数", "频率(%)"]).sort_values("出现次数", ascending=False).reset_index(drop=True)
            sf.index = sf.index + 1
            st.dataframe(sf, width='stretch', height=360)
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown(f"<h3 style='font-size:1.1rem;color:#1e293b;'>📋 统计分析摘要</h3>", unsafe_allow_html=True)
        ss = analysis["sum"]["main_sum_stats"]
        sp = analysis["span"]["main_span_stats"]
        ac = analysis["ac_value"]["main_ac_stats"]
        conc = analysis["consecutive"]
        oe = analysis["odd_even"]
        cs1, cs2, cs3 = st.columns(3)
        with cs1: st.markdown(render_metric(f"{ss['mean']:.1f}", f"{cfg.main_label}和值均值"), unsafe_allow_html=True); st.markdown(render_metric(f"{ss['min']} ~ {ss['max']}", "和值范围"), unsafe_allow_html=True)
        with cs2: st.markdown(render_metric(f"{sp['mean']:.1f}", f"{cfg.main_label}跨度均值"), unsafe_allow_html=True); st.markdown(render_metric(f"{ac['mean']:.1f}", "AC 值均值"), unsafe_allow_html=True)
        with cs3: st.markdown(render_metric(f"{conc['main_consec_pct']:.1f}%", "连号出现率"), unsafe_allow_html=True); st.markdown(render_metric(str(oe["main_most_common"]), "常见奇偶比"), unsafe_allow_html=True)

    # ═══════ 预测历史 ═══════
    elif page == "📜 预测历史":
        st.markdown(f"<h1 style='background:linear-gradient(135deg,#2563eb 0%,#1d4ed8 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2rem;font-weight:800;'>📜 预测历史记录</h1>", unsafe_allow_html=True)
        all_preds = get_all_predictions(cfg)
        if not all_preds:
            st.markdown("<div class='glass-card' style='text-align:center;padding:3rem;'><p style='color:#64748b;'>暂无预测历史记录</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:#475569;font-size:0.85rem;'>共 {len(all_preds)} 条记录</p>", unsafe_allow_html=True)
            for pred in all_preds:
                ps, pp, pc, pr = pred["status"], pred["period"], pred.get("created_at", ""), len(pred.get("recommendations", []))
                if ps == "completed" and pred.get("summary"):
                    s = pred["summary"]
                    with st.expander(f"**期 {pp}** ✅ 已比对 · 最佳 {s['best_hits']} 个 · {pc}", expanded=False):
                        if pred.get("actual_draw"):
                            am = "".join(render_ball(n, "main") for n in pred["actual_draw"]["main"])
                            as_ = "".join(render_ball(n, "sub") for n in pred["actual_draw"]["sub"])
                            st.markdown(f"<p style='color:#94a3b8;font-size:0.85rem;'>📌 实际开奖: {am} {as_}</p>", unsafe_allow_html=True)
                        c_a, c_b, c_c = st.columns(3)
                        with c_a: st.markdown(render_metric(f"{s['best_hits']} 个", "最佳命中"), unsafe_allow_html=True)
                        with c_b: st.markdown(render_metric(f"{s['avg_main_hits']}", f"{cfg.main_label}平均"), unsafe_allow_html=True)
                        with c_c: st.markdown(render_metric(f"{s['avg_sub_hits']}", f"{cfg.sub_label}平均"), unsafe_allow_html=True)
                        if pred.get("matches"):
                            st.markdown("<h4 style='font-size:0.95rem;margin:1rem 0 0.5rem;color:#475569;'>各组预测号码</h4>", unsafe_allow_html=True)
                            mm = {m["group"]: m for m in pred["matches"]}
                            for rec in pred["recommendations"]:
                                st.markdown(render_history_group(rec, mm, cfg), unsafe_allow_html=True)
                else:
                    with st.expander(f"**期 {pp}** ⏳ 待开奖 · {pr}组 · {pc}", expanded=False):
                        if pred.get("recommendations"):
                            for rec in pred["recommendations"]:
                                st.markdown(render_history_group(rec, None, cfg), unsafe_allow_html=True)

    # ═══════ 回测 ═══════
    elif page == "📈 回测":
        st.markdown(f"<h1 style='background:linear-gradient(135deg,#2563eb 0%,#1d4ed8 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2rem;font-weight:800;'>📈 历史回测</h1>", unsafe_allow_html=True)
        if st.button("▶️ 开始回测", type="primary", use_container_width=True):
            with st.spinner("正在运行回测..."):
                o = sys.stdout; sys.stdout = captured = StringIO()
                bt = Backtester(df, cfg, test_window=30)
                stats = bt.run(verbose=False)
                sys.stdout = o
            if "error" in stats: st.error(f"回测失败: {stats['error']}")
            else:
                st.success("✅ 回测完成")
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.markdown(render_metric(f"{stats['total_tests']}", "测试期数"), unsafe_allow_html=True)
                with c2: st.markdown(render_metric(f"{stats['total_hits']['mean']:.2f}", "综合平均命中"), unsafe_allow_html=True)
                with c3: st.markdown(render_metric(f"{stats['best_hits_count']}", "最优次数"), unsafe_allow_html=True)
                with c4: st.markdown(render_metric(f"{stats['best_hits_rate']:.1f}%", "最优命中率"), unsafe_allow_html=True)

    # 页脚
    st.markdown("<hr>", unsafe_allow_html=True)
    from utils.helpers import print_disclaimer
    print_disclaimer(cfg)
    st.markdown("<p style='text-align:center;font-size:0.7rem;color:#94a3b8;margin-top:1rem;'>{}</p>".format(f"{cfg.short.upper()} Predictor v2.0 · Python · Streamlit · scikit-learn · TensorFlow"), unsafe_allow_html=True)
