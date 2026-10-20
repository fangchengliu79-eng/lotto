"""跨期惩罚强度参数扫描（不保存任何预测）：找出既压制过度重复号码、又不让 5 组红球雷同的 k 组合。"""
import sys, os, itertools, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config_ssq import SSQ
from data_fetcher import update_data
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
import models.ensemble as E
from models.ensemble import EnsembleModel, generate_recommendations, _load_prediction_history_counts
from utils.predictions import _load_all, _save_all

cfg = SSQ
df = update_data(cfg)

# 先移除 active 记录（与真实重新生成时一致），只保留 completed
all_p = _load_all(cfg)
active = [p for p in all_p if p.get("status") == "active"]
_save_all([p for p in all_p if p.get("status") != "active"], cfg)
print("已临时移除 active 记录:", len(active), "（本次扫描不保存任何预测）")

m, s, mp, sp, rm, rs = _load_prediction_history_counts(cfg, 10, 3)
overused_main = [n for n, c in sorted(m.items(), key=lambda x: -x[1])[:5]]
print("近10期红球最常推荐:", [(n, c) for n, c in sorted(m.items(), key=lambda x: -x[1])[:5]])
print("近10期蓝球最常推荐:", [(n, c) for n, c in sorted(s.items(), key=lambda x: -x[1])[:5]])

md = {"f": FrequencyModel(cfg), "p": PoissonModel(cfg),
      "e": ExponentialSmoothingModel(cfg, alpha=0.3), "m": MonteCarloModel(cfg)}
md["m"].n_simulations = 20000
mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
for mm in md.values():
    mm.fit(mn, sn)

combos = [
    (0.0, 2.2, 0.0, 3.0),   # 红球完全不压制（只看蓝球）
    (1.0, 2.2, 1.2, 3.0),
    (1.3, 2.2, 1.6, 3.0),
    (1.8, 2.2, 2.0, 3.0),
    (2.5, 2.2, 3.0, 3.0),
]

for klm, kls, krm, krs in combos:
    E.HIST_K_LONG_MAIN, E.HIST_K_LONG_SUB = klm, kls
    E.HIST_K_RECENT_MAIN, E.HIST_K_RECENT_SUB = krm, krs
    r = generate_recommendations(EnsembleModel(md, cfg), cfg, num_groups=5, df=df)
    gs = r["groups"]
    mains = [tuple(g["main"]) for g in gs]
    flat_m = [n for g in gs for n in g["main"]]
    flat_s = [n for g in gs for n in g["sub"]]
    cm = collections.Counter(flat_m)
    cs = collections.Counter(flat_s)
    print(f"\n--- k_main(long/recent)={klm}/{krm}  k_sub={kls}/{krs}")
    for g in gs:
        print("   ", g["main"], "蓝", g["sub"])
    print(f"    唯一红球组合 {len(set(mains))}/5 · 红球max重复 {max(cm.values())} · "
          f"蓝球唯一 {len(cs)}/5 max {max(cs.values())} · 含04 {4 in cs} · "
          f"过度号命中 {[n for n in overused_main if cm.get(n,0)>=2]}")
