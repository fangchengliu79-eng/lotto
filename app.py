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
    get_all_predictions, get_latest_prediction, save_prediction, auto_compare_latest
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

    if gm and gm.get("total_hits", 0) > 0:
        m = gm
        ht = (f"<div style='margin-top:0.5rem;font-size:0.85rem;color:#eab308;font-weight:600;'>🎯 命中 {m['main_hits']}{cfg.main_label} + {m['sub_hits']}{cfg.sub_label} = {m['total_hits']}个</div>")
    elif active:
        ht = "<div style='margin-top:0.5rem;font-size:0.85rem;color:#64748b;'>⏳ 待开奖</div>"
    return (f"<div class='glass-card glow' style='padding:1rem;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;'>"
            f"<span style='font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;'>第{g}组</span>"
            f"<span style='font-size:0.75rem;color:#2563eb;font-weight:600;font-family:JetBrains Mono,monospace;'>综合得分 {sc:.1f}</span></div>"
            f"<div style='margin-bottom:0.5rem;'><div style='font-size:0.7rem;color:#64748b;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.05em;'>{cfg.main_label} · {cfg.main_count}码</div><div>{mb}</div></div>"
            f"<div style='margin-bottom:0.5rem;'><div style='font-size:0.7rem;color:#64748b;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.05em;'>{cfg.sub_label} · {cfg.sub_count}码</div><div>{sb}</div></div>"
            f"<div style='font-size:0.7rem;color:#64748b;margin-top:0.35rem;'>📊 {rec.get('reason','综合评分 '+str(round(rec['score'],1)))}</div>{ht}</div>")

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

    # 检测是否有新数据加入：如果最新数据的期号 > 最新预测的期号，自动生成新预测
    latest_data_period = str(df.iloc[0]["period"])
    lp_check = get_latest_prediction(cfg)
    needs_new_prediction = False
    if lp_check is None:
        needs_new_prediction = True
    else:
        try:
            if int(latest_data_period) > int(lp_check["period"]):
                needs_new_prediction = True
        except (ValueError, KeyError):
            pass
    
    if needs_new_prediction:
        with st.spinner(f"检测到新数据，正在生成{cfg.name}预测..."):
            ensemble = EnsembleModel({}, cfg)
            r2 = generate_recommendations(ensemble, cfg, num_groups=5, df=df)
            next_period = str(int(latest_data_period) + 1)
            save_prediction(period=next_period, recommendations=r2.get("groups",[]), cfg=cfg)
            st.rerun()

    # 子页面导航
    page = st.radio("", ["🎯 预测结果", "📜 预测历史"], 
                    horizontal=True, label_visibility="collapsed", key=f"nav_{cfg.short}")
    num_groups = 5

    # ──── 预测结果 ────
    if page == "🎯 预测结果":
        st.markdown(f"<h1>{cfg.icon} 当前预测结果</h1>", unsafe_allow_html=True)
        lp = get_latest_prediction(cfg)
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
            if s == "completed": st.success(f"✅ **期{p}** 已开奖 · 最佳命中 **{lp['summary']['best_hits']}** 个")
            else: st.info(f"⏳ **期{p}** 预测已封存 · 开奖后自动比对")
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown(metric_card(f"#{p}", "期号"), unsafe_allow_html=True)
            with c2: st.markdown(metric_card(f"{nr}", "推荐组数"), unsafe_allow_html=True)
            stat_color = "#059669" if s == "completed" else "#d97706"
            stat_label = "✅ 已比对" if s == "completed" else "⏳ 待开奖"
            with c3: st.markdown(f"<div class='metric-card'><div class='value' style='font-size:1.2rem;color:{stat_color}'>{stat_label}</div><div class='label'>状态</div></div>", unsafe_allow_html=True)
            with c4: st.markdown(metric_card(cr[:10] if cr else "-", "日期"), unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)

            if s == "completed" and lp.get("actual_draw"):
                a = lp["actual_draw"]
                am = "".join(ball(n,"main") for n in a["main"])
                as_ = "".join(ball(n,"sub") for n in a["sub"])
                st.markdown(f"<h3>实际开奖号码</h3><div><span style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-right:0.5rem;'>{cfg.main_label}</span>{am}</div><div><span style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-right:0.5rem;'>{cfg.sub_label}</span>{as_}</div>", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown(f"<h3>推荐号码 <span style='color:#64748b;font-weight:400;font-size:0.85rem;'>({nr} 组)</span></h3>", unsafe_allow_html=True)
            mt = {m["group"]: m for m in (lp.get("matches") or [])}
            cols = st.columns(min(5, len(lp["recommendations"])))
            for i, r in enumerate(lp["recommendations"]):
                with cols[i]: st.markdown(pred_card(r, mt, s != "completed", cfg), unsafe_allow_html=True)

            if s == "completed" and lp.get("summary"):
                sm = lp["summary"]
                st.markdown("<hr>", unsafe_allow_html=True)
                ca,cb,cc,cd = st.columns(4)
                with ca: st.markdown(metric_card(f"{sm['best_hits']} 个", "最佳命中", f"第{sm['best_group']}组"), unsafe_allow_html=True)
                with cb: st.markdown(metric_card(f"{sm['avg_main_hits']:.1f}", f"{cfg.main_label}平均"), unsafe_allow_html=True)
                with cc: st.markdown(metric_card(f"{sm['avg_sub_hits']:.1f}", f"{cfg.sub_label}平均"), unsafe_allow_html=True)
                with cd: st.markdown(metric_card(f"{sm['avg_total_hits']:.1f}", "综合平均"), unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            lp2 = str(df.iloc[0]["period"])
            if any(p["period"]==lp2 and p["status"] in ("active","completed") for p in get_all_predictions(cfg)):
                st.info(f"✅ 期 {lp2} 已有预测记录")
            if st.button("📅 生成下一期预测", type="primary", key=f"gen_next_{cfg.short}"):
                np_ = str(int(df.iloc[0]["period"])+1)
                if any(p["period"]==np_ for p in get_all_predictions(cfg)):
                    st.warning(f"期 {np_} 已有预测")
                else:
                    with st.spinner("正在生成新预测..."):
                        md = {"f": FrequencyModel(cfg), "p": PoissonModel(cfg),
                              "e": ExponentialSmoothingModel(cfg, alpha=0.3), "m": MonteCarloModel(cfg)}
                        md["m"].n_simulations = 20000
                        mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
                        sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
                        for m in md.values(): m.fit(mn, sn)
                        ensemble = EnsembleModel(md, cfg)
                        r2 = generate_recommendations(ensemble, cfg, num_groups=5, df=df)
                        save_prediction(period=np_, recommendations=r2.get("groups",[]), cfg=cfg)
                        st.rerun()
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
        all_p = get_all_predictions(cfg)
        if not all_p: st.markdown("<div class='glass-card' style='text-align:center;padding:3rem;'><p style='color:#64748b;'>暂无预测历史记录</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:#475569;'>{len(all_p)} 条记录</p>", unsafe_allow_html=True)
            for pred in all_p:
                ps, pp, pc, pr_ = pred["status"], pred["period"], pred.get("created_at",""), len(pred.get("recommendations",[]))
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
