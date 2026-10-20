"""验收脚本：证明蓝球"跨期不过度集中"是与具体号码无关的通用规则（不写入任何真实预测文件）。

做法：
1. 打印当前真实历史算出的「蓝球跨期配额表」（r3 = 近3期被推荐次数 → 本期允许组数）
2. 用内存假历史模拟连续 5 期：每期生成后把结果并入假历史，再生成下一期
   —— 全程 monkeypatch 历史加载器，不碰 data/ 下任何文件
3. 打印每期蓝球分布 + 是否有人超过配额 + 跨 5 期的单号最大累计次数
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config_ssq import SSQ
from data_fetcher import update_data
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
import models.ensemble as E
from models.ensemble import EnsembleModel, generate_recommendations, _build_sub_quota
from utils.predictions import get_all_predictions

cfg = SSQ
df = update_data(cfg)

# ── 真实历史（只读）──
real = [p for p in get_all_predictions(cfg)
        if p.get("algorithm", "ensemble") == "ensemble"
        and p.get("status") in ("active", "completed")]
real.sort(key=lambda p: str(p["period"]), reverse=True)
real = real[:10]

fake = [{"period": p["period"], "recommendations": p["recommendations"],
         "status": "completed", "algorithm": "ensemble"} for p in real]


def fake_loader(cfg_, max_records=10, recent_records=3):
    main_counts, sub_counts, main_pairs, sub_pairs = {}, {}, {}, {}
    recent_main, recent_sub = {}, {}
    for rank, p in enumerate(fake[:max_records]):
        in_recent = rank < recent_records
        for g in p.get("recommendations", []):
            for n in g.get("main", []):
                main_counts[n] = main_counts.get(n, 0) + 1
                if in_recent:
                    recent_main[n] = recent_main.get(n, 0) + 1
            for n in g.get("sub", []):
                sub_counts[n] = sub_counts.get(n, 0) + 1
                if in_recent:
                    recent_sub[n] = recent_sub.get(n, 0) + 1
            sm = sorted(g.get("main", []))
            for i in range(len(sm)):
                for j in range(i + 1, len(sm)):
                    pr = frozenset((sm[i], sm[j]))
                    main_pairs[pr] = main_pairs.get(pr, 0) + 1
            ss = sorted(g.get("sub", []))
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    pr = frozenset((ss[i], ss[j]))
                    sub_pairs[pr] = sub_pairs.get(pr, 0) + 1
    return main_counts, sub_counts, main_pairs, sub_pairs, recent_main, recent_sub


E._load_prediction_history_counts = fake_loader

# ── 1. 当前配额表 ──
_, _, _, _, _, r3_now = fake_loader(cfg)
q = _build_sub_quota(r3_now, cfg.sub_min, cfg.sub_max, 3, cfg.sub_count * 5)
print("近3期蓝球推荐次数:", {n: c for n, c in sorted(r3_now.items(), key=lambda x: -x[1])})
print("本期蓝球配额:    ", {n: q[n] for n in range(1, 17)})
print("（配额只看次数，不看是哪个号码；r3>=6 的号本期 0 组）\n")

# ── 2. 连续 5 期模拟 ──
md = {"f": FrequencyModel(cfg), "p": PoissonModel(cfg),
      "e": ExponentialSmoothingModel(cfg, alpha=0.3), "m": MonteCarloModel(cfg)}
md["m"].n_simulations = 20000
mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
for m in md.values():
    m.fit(mn, sn)

acc = collections.Counter()
start = int(df.iloc[0]["period"]) + 1
for step in range(5):
    period = start + step
    r = generate_recommendations(EnsembleModel(md, cfg), cfg, num_groups=5, df=df)
    subs = [g["sub"][0] for g in r["groups"]]
    c = collections.Counter(subs)
    acc.update(subs)
    over = [n for n in c if c[n] > q.get(n, 3)]
    print(f"期{period} 蓝球 {subs}  单号最多 {max(c.values())} 次  超配额: {over if over else '无 ✓'}")
    fake.insert(0, {"period": str(period), "recommendations": r["groups"],
                    "status": "completed", "algorithm": "ensemble"})
    _, _, _, _, _, r3 = fake_loader(cfg)
    q = _build_sub_quota(r3, cfg.sub_min, cfg.sub_max, 3, cfg.sub_count * 5)

print("\n跨 5 期单号累计:", acc.most_common(6))
print("5期×5组=25 个蓝球位，平均每号 25/16=1.6 次；最大累计",
      max(acc.values()), "次 →", "合理 ✓" if max(acc.values()) <= 6 else "仍偏集中 ✗")
