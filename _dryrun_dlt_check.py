"""DLT 干跑校验（不写入任何预测文件）：确认跨期惩罚改造后大乐透仍正常出号。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collections
import numpy as np
from config_dlt import DLT
from data_fetcher import update_data
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
from models.ensemble import EnsembleModel, generate_recommendations
from models.spiral_matrix import SpiralMatrixPredictor

cfg = DLT
df = update_data(cfg)
print("DLT 数据期数:", len(df), "最新期:", int(df.iloc[0]["period"]))

md = {"f": FrequencyModel(cfg), "p": PoissonModel(cfg),
      "e": ExponentialSmoothingModel(cfg, alpha=0.3), "m": MonteCarloModel(cfg)}
md["m"].n_simulations = 20000
mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
for m in md.values():
    m.fit(mn, sn)

r1 = generate_recommendations(EnsembleModel(md, cfg), cfg, num_groups=5, df=df)
print("\n=== DLT 统计算法（干跑，未保存）===")
for g in r1["groups"]:
    print("  前区", g["main"], "后区", g["sub"], "|", g["reason"].split(":")[0])
fronts = [n for g in r1["groups"] for n in g["main"]]
backs = [n for g in r1["groups"] for n in g["sub"]]
print("前区最大重复", collections.Counter(fronts).most_common(3), "后区最大重复", collections.Counter(backs).most_common(3))
print("前区是否含 07/17:", 7 in fronts, 17 in fronts)

r2 = SpiralMatrixPredictor().predict(cfg, df, num_groups=5)
print("\n=== DLT 螺旋矩阵（干跑，未保存）===")
for g in r2["groups"]:
    print("  前区", g["main"], "后区", g["sub"], "|", g["reason"].split(":")[0])
