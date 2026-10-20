"""验收：蓝球规则 v3（2026-10-20）

唯一违规情形：上一期蓝球已经猜中（该期实际开出的蓝球在我们推荐里），本期还推荐同一个蓝球。
其余情况全部放行（上期推过没中、近3期推得多、同期重复）。

不写入任何预测文件：只用 generate_recommendations 在内存里算，不调用 save_prediction。
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from config_ssq import SSQ
from models.statistical import FrequencyModel, PoissonModel, MonteCarloModel
from models.timeseries import ExponentialSmoothingModel
import models.ensemble as E
from models.ensemble import (EnsembleModel, generate_recommendations,
                             _load_blue_prev_hits, _blue_caps_hit_only)

cfg = SSQ
df = pd.read_csv(cfg.history_csv, encoding="utf-8-sig")
df = df.sort_values("period", ascending=False).reset_index(drop=True)
latest = int(df.iloc[0]["period"])
drawn = [int(df.iloc[0][c]) for c in cfg.sub_cols]
print(f"最新开奖期 {latest}，实际蓝球 {drawn}")

# ── 1) 命中识别：上期推荐的蓝球里是否含实际开出的号 ──
for alg in ("ensemble", "spiral_matrix"):
    hits = _load_blue_prev_hits(cfg, latest, drawn, alg)
    print(f"  [{alg}] 上期({latest})猜中的蓝球 -> {sorted(hits) or '无'}")
hits_ens = _load_blue_prev_hits(cfg, latest, drawn, "ensemble")
caps = _blue_caps_hit_only(hits_ens, cfg.sub_min, cfg.sub_max, base_cap=3)
print(f"  本期配额: 被猜中的号={ {n: caps[n] for n in sorted(hits_ens)} }  "
      f"其余号={sorted(set(caps.values()) - {0})} 组")
assert all(caps[n] == 0 for n in hits_ens), "猜中号必须 0 组"
assert all(caps[n] == 3 for n in caps if n not in hits_ens), "未猜中号必须不受限(3组)"
print("  ✓ 配额规则正确：猜中号 0 组，其余号 3 组")

# ── 2) 用真实历史跑本期预测，确认猜中号不再出现 ──
md = {"f": FrequencyModel(cfg), "p": PoissonModel(cfg),
      "e": ExponentialSmoothingModel(cfg, alpha=0.3), "m": MonteCarloModel(cfg)}
md["m"].n_simulations = 5000
mn = np.array([sorted([int(r[c]) for c in cfg.main_cols]) for _, r in df.iterrows()])
sn = np.array([sorted([int(r[c]) for c in cfg.sub_cols]) for _, r in df.iterrows()])
for m in md.values():
    m.fit(mn, sn)

r = generate_recommendations(EnsembleModel(md, cfg), cfg, num_groups=5, df=df)
subs = [g["sub"][0] for g in r["groups"]]
cnt = collections.Counter(subs)
print(f"\n期{latest + 1} 五组蓝球: {subs}   红球: {[g['main'] for g in r['groups']]}")
print(f"  同期重复统计: {dict(cnt)}")
for n in hits_ens:
    assert n not in subs, f"违规: 上期猜中的蓝球 {n} 又在下一期出现了"
print(f"  ✓ 上期猜中的 {sorted(hits_ens)} 本期 0 次出现")
print(f"  ✓ 同期限重复仍允许（单号最多3组），本例最大重复 {max(cnt.values())} 次")
assert max(cnt.values()) <= 3, "单号同期不得超过3组"
print(f"  ✓ 其他蓝球未受任何跨期压制（只在猜中情形下限量）")

# ── 3) 未猜中场景：不施加任何限制，猜中号之外全部可推 ──
caps_miss = _blue_caps_hit_only(set(), cfg.sub_min, cfg.sub_max, base_cap=3)
assert set(caps_miss.values()) == {3}, "没有猜中时必须完全不限制"
print("\n  ✓ 上期没猜中 -> 本期所有蓝球均为 3 组上限（无跨期限制）")

# ── 4) 连续 3 期链路：猜中后仅禁 1 期，第 3 期自动解禁 ──
x = sorted(hits_ens)[0] if hits_ens else 4
chain = [_blue_caps_hit_only({x}, cfg.sub_min, cfg.sub_max, 3),      # N+1: 禁
         _blue_caps_hit_only(set(), cfg.sub_min, cfg.sub_max, 3),    # N+2: 解禁
         _blue_caps_hit_only(set(), cfg.sub_min, cfg.sub_max, 3)]    # N+3: 解禁
print(f"  期{latest+1}(猜中后) {x} 配额={chain[0][x]}  "
      f"期{latest+2} {x} 配额={chain[1][x]}  期{latest+3} {x} 配额={chain[2][x]}")
assert chain[0][x] == 0 and chain[1][x] == 3 and chain[2][x] == 3
print("  ✓ 只禁紧接着的一期，之后自动恢复")

print("\n全部通过 ✓")
