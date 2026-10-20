"""一次性维护脚本：按新算法（跨期记忆 + 近3期重罚）重新生成双色球当前期预测。

- 只删除 status="active" 的 SSQ 预测（completed 历史全部保留）
- 重新生成 统计算法 + 螺旋矩阵 两种算法
- 打印蓝球/红球重复度校验
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import collections
from config_ssq import SSQ
from data_fetcher import update_data
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
from models.ensemble import EnsembleModel, generate_recommendations, _load_prediction_history_counts
from models.spiral_matrix import SpiralMatrixPredictor
from utils.predictions import (_load_all, _save_all, save_prediction,
                               get_all_predictions)

cfg = SSQ
df = update_data(cfg)
print("数据期数:", len(df), "最新期:", int(df.iloc[0]["period"]))

# 历史推荐统计（改造前状态，用于对照）
mc, sc, mp, sp, rm, rs = _load_prediction_history_counts(cfg, 10, 3)
print("近10期蓝球推荐次数 TOP6:", sorted(sc.items(), key=lambda x: -x[1])[:6])
print("近3期蓝球推荐次数 TOP6:", sorted(rs.items(), key=lambda x: -x[1])[:6])

# 目标期号：现有 active 期号，否则 最新数据期+1
all_p = _load_all(cfg)
active = [p for p in all_p if p.get("status") == "active"]
if active:
    target = max(int(p["period"]) for p in active)
else:
    target = int(df.iloc[0]["period"]) + 1
print("目标期号:", target, "（删除前 active 记录数:", len(active), "）")

kept = [p for p in all_p if p.get("status") != "active"]
print("保留（completed/其它）记录数:", len(kept))
_save_all(kept, cfg)

# 生成
md = {"f": FrequencyModel(cfg), "p": PoissonModel(cfg),
      "e": ExponentialSmoothingModel(cfg, alpha=0.3), "m": MonteCarloModel(cfg)}
md["m"].n_simulations = 20000
import numpy as np
mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
for m in md.values():
    m.fit(mn, sn)

ens = EnsembleModel(md, cfg)
r1 = generate_recommendations(ens, cfg, num_groups=5, df=df)
save_prediction(period=str(target), recommendations=r1["groups"], cfg=cfg, algorithm="ensemble")

sp_pred = SpiralMatrixPredictor()
r2 = sp_pred.predict(cfg, df, num_groups=5)
save_prediction(period=str(target), recommendations=r2["groups"], cfg=cfg, algorithm="spiral_matrix")

print("\n=== 新预测 统计算法 ===")
for g in r1["groups"]:
    print(" ", g["main"], "蓝", g["sub"], "|", g["reason"].split(":")[0])

print("=== 新预测 螺旋矩阵 ===")
for g in r2["groups"]:
    print(" ", g["main"], "蓝", g["sub"], "|", g["reason"].split(":")[0])

# 校验
for name, res in (("ensemble", r1), ("spiral_matrix", r2)):
    subs = [n for g in res["groups"] for n in g["sub"]]
    mains = [n for g in res["groups"] for n in g["main"]]
    cs = collections.Counter(subs)
    cm = collections.Counter(mains)
    print(f"\n[{name}] 蓝球分布 {dict(cs)} 最大重复 {max(cs.values())}")
    print(f"[{name}] 红球最大重复 {cm.most_common(3)}")
    bad = [n for n, c in cs.items() if c >= 3]
    print(f"[{name}] 蓝球出现3次的号码: {bad if bad else '无 ✓'}")
    print(f"[{name}] 蓝球是否含04: {'是' if 4 in cs else '否 ✓'}")
