"""验收：相邻期蓝球去重（同一期内可重复，相邻期不能接着堆）。

不写入任何真实预测文件：monkeypatch `_load_blue_period_history`，用内存历史模拟连续 3 期。
判定标准：
  1. 每期蓝球出现次数 <= 该号码的配额
  2. 配额只由"过往期推荐次数"决定（上一期推过 -> 最多1组；近3期>=4次 -> 0组）
  3. 同一期内仍允许重复（配额=3 时最多 3 组），规则不禁止同期重复
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config_ssq import SSQ
from data_fetcher import update_data
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
import models.ensemble as E
from models.ensemble import (EnsembleModel, generate_recommendations,
                             _load_blue_period_history, _blue_caps)
from utils.predictions import get_all_predictions

cfg = SSQ
df = update_data(cfg)
latest = int(df.iloc[0]["period"])

# 真实历史（只读）：取已开奖期的 ensemble 记录
real = [p for p in get_all_predictions(cfg)
        if p.get("algorithm", "ensemble") == "ensemble"
        and p.get("status") in ("active", "completed")
        and int(str(p.get("period", "0"))) <= latest]
real.sort(key=lambda p: str(p["period"]), reverse=True)

sim = [{"period": p["period"], "recommendations": p["recommendations"]} for p in real[:3]]


def fake_loader(cfg_, algorithm="ensemble", periods=3, max_period=None):
    prev, recent = {}, {}
    for rank, p in enumerate(sim[:periods]):
        for g in p["recommendations"]:
            for n in g.get("sub", []):
                recent[n] = recent.get(n, 0) + 1
                if rank == 0:
                    prev[n] = prev.get(n, 0) + 1
    return prev, recent


E._load_blue_period_history = fake_loader

prev, recent = fake_loader(cfg)
caps = _blue_caps(prev, recent, cfg.sub_min, cfg.sub_max, 3, cfg.sub_count * 5)
print(f"上一期(期{sim[0]['period']})蓝球推荐:", {n: c for n, c in prev.items()})
print(f"最近3期累计:", {n: c for n, c in recent.items()})
print(f"本期蓝球配额:", {n: caps[n] for n in range(1, 17)})
print("  （配额只看次数：上一期推过->1组；近3期>=4次->0组；其余->3组，同期限重复不变）\n")

md = {"f": FrequencyModel(cfg), "p": PoissonModel(cfg),
      "e": ExponentialSmoothingModel(cfg, alpha=0.3), "m": MonteCarloModel(cfg)}
md["m"].n_simulations = 20000
mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
for m in md.values():
    m.fit(mn, sn)

start = latest + 1
for step in range(3):
    period = start + step
    r = generate_recommendations(EnsembleModel(md, cfg), cfg, num_groups=5, df=df)
    subs = [g["sub"][0] for g in r["groups"]]
    c = collections.Counter(subs)
    viol = [n for n in c if c[n] > caps.get(n, 3)]
    capped_here = {n: (c.get(n, 0), caps.get(n, 3)) for n in sorted(set(list(c) + list(prev)))
                   if caps.get(n, 3) < 3}
    print(f"期{period} 蓝球 {subs}   红球最大重复 "
          f"{max(collections.Counter([n for g in r['groups'] for n in g['main']]).values())}")
    print(f"   受限号(实际次数/配额): {capped_here}   超配额: {viol if viol else '无 ✓'}")
    sim.insert(0, {"period": str(period), "recommendations": r["groups"]})
    prev, recent = fake_loader(cfg)
    caps = _blue_caps(prev, recent, cfg.sub_min, cfg.sub_max, 3, cfg.sub_count * 5)

print("\n同期限重复是否仍允许：配额=3 的号码本期最多可占 3 组（未被规则禁止）")
