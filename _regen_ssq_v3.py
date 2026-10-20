"""按蓝球规则 v3 重新生成双色球当前期预测（一次性维护脚本）。

- 不删除任何 completed 历史；save_prediction 会原位替换同一期同一算法的 active 记录
- 生成 统计算法(ensemble) + 螺旋矩阵(spiral_matrix)
- 打印蓝球校验：上期猜中的蓝球本期必须 0 次出现
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config_ssq import SSQ
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
from models.ensemble import (EnsembleModel, generate_recommendations,
                             _load_blue_prev_hits, _blue_caps_hit_only)
from models.spiral_matrix import SpiralMatrixPredictor
from utils.predictions import _load_all, save_prediction

cfg = SSQ
df = pd.read_csv(cfg.history_csv, encoding="utf-8-sig")
df = df.sort_values("period", ascending=False).reset_index(drop=True)
latest = int(df.iloc[0]["period"])
drawn = [int(df.iloc[0][c]) for c in cfg.sub_cols]

all_p = _load_all(cfg)
active = [p for p in all_p if p.get("status") == "active"]
target = max([int(p["period"]) for p in active], default=latest + 1)
print(f"最新开奖 {latest} (蓝={drawn})  目标期号 {target}  active记录 {len(active)}")

hits_ens = _load_blue_prev_hits(cfg, latest, drawn, "ensemble")
hits_sp = _load_blue_prev_hits(cfg, latest, drawn, "spiral_matrix")
print(f"上期猜中蓝球: ensemble={sorted(hits_ens) or '无'}  spiral={sorted(hits_sp) or '无'}")

md = {"f": FrequencyModel(cfg), "p": PoissonModel(cfg),
      "e": ExponentialSmoothingModel(cfg, alpha=0.3), "m": MonteCarloModel(cfg)}
md["m"].n_simulations = 20000
mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
for m in md.values():
    m.fit(mn, sn)

r1 = generate_recommendations(EnsembleModel(md, cfg), cfg, num_groups=5, df=df)
r2 = SpiralMatrixPredictor().predict(cfg, df, num_groups=5)

save_prediction(period=str(target), recommendations=r1["groups"], cfg=cfg, algorithm="ensemble")
save_prediction(period=str(target), recommendations=r2["groups"], cfg=cfg, algorithm="spiral_matrix")

ok = True
for name, res in (("ensemble", r1), ("spiral_matrix", r2)):
    subs = [n for g in res["groups"] for n in g["sub"]]
    cm = collections.Counter(n for g in res["groups"] for n in g["main"])
    cs = collections.Counter(subs)
    print(f"\n[{name}] 期{target}")
    for g in res["groups"]:
        print("  ", g["main"], "蓝", g["sub"], "|", str(g.get("reason", "")).split(":")[0])
    print(f"  蓝球分布 {dict(cs)} 最大同期重复 {max(cs.values())} / 红球最大重复 {cm.most_common(1)}")
    blocked = hits_ens if name == "ensemble" else hits_sp
    bad = sorted(set(subs) & set(blocked))
    print(f"  上期猜中号{ sorted(blocked) } 本期出现: {bad if bad else '无 ✓'}")
    ok = ok and not bad and max(cs.values()) <= 3

print("\n校验:", "通过 ✓" if ok else "失败 ✗")
sys.exit(0 if ok else 1)
