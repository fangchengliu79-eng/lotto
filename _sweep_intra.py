"""跨期内降权系数扫描（不保存预测）：消除 5 组红球雷同。"""
import sys, os, collections
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
all_p = _load_all(cfg)
active = [p for p in all_p if p.get("status") == "active"]
if active:
    _save_all([p for p in all_p if p.get("status") != "active"], cfg)
    print("临时移除 active:", len(active))

m, s, *_ = _load_prediction_history_counts(cfg, 10, 3)
overused = [n for n, c in sorted(m.items(), key=lambda x: -x[1])[:5]]
sub_over = [n for n, c in sorted(s.items(), key=lambda x: -x[1])[:3]]
print("过度红球:", overused, "过度蓝球:", sub_over)

md = {"f": FrequencyModel(cfg), "p": PoissonModel(cfg),
      "e": ExponentialSmoothingModel(cfg, alpha=0.3), "m": MonteCarloModel(cfg)}
md["m"].n_simulations = 20000
mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
for mm in md.values():
    mm.fit(mn, sn)

for i1, i2, i3, pair in [(0.55, 0.10, 0.01, 0.15),
                         (0.30, 0.05, 0.01, 0.10),
                         (0.15, 0.03, 0.005, 0.05)]:
    E.INTRA_MAIN_1, E.INTRA_MAIN_2, E.INTRA_MAIN_3, E.INTRA_PAIR = i1, i2, i3, pair
    r = generate_recommendations(EnsembleModel(md, cfg), cfg, num_groups=5, df=df)
    gs = r["groups"]
    mains = [tuple(g["main"]) for g in gs]
    fm = [n for g in gs for n in g["main"]]
    fs = [n for g in gs for n in g["sub"]]
    cm, cs = collections.Counter(fm), collections.Counter(fs)
    print(f"\n--- 组内降权 {i1}/{i2}/{i3} 对{pair}")
    for g in gs:
        print("   ", g["main"], "蓝", g["sub"], "|", g["reason"].split(":")[0])
    print(f"    唯一红球组合 {len(set(mains))}/5 · 红max {max(cm.values())} · 蓝唯一 {len(cs)}/5 · "
          f"含04 {4 in cs} · 过度红球命中 {[n for n in overused if cm.get(n,0)>=2]} · "
          f"过度蓝球命中 {[n for n in sub_over if n in cs]}")
