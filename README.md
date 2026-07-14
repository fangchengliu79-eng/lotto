# 🎯 彩票预测系统 · 部署说明

> 版本: **v4.0 (fourstage)**
> Git tag: `fourstage`
> 支持: **大乐透 (DLT)** + **双色球 (SSQ)**

---

## 📦 快速部署（AI agent / 新机器）

### 1. 克隆仓库

```bash
git clone https://github.com/fangchengliu79-eng/lotto.git
cd lotto
git checkout master
```

### 2. 环境准备

依赖库（系统 Python 3.10+ 即可，无需 virtualenv）：

```bash
pip install streamlit pandas numpy matplotlib requests beautifulsoup4
```

### 3. 启动服务

```bash
cd /root/lotto
streamlit run app.py --server.port 8502 --server.headless true
```

访问地址：`http://localhost:8502`

> ⚠️ **端口说明**：8501 已被量化分析系统 (quant-analyzer) 占用，lotto 固定使用 **8502**。

### 4. 开机自启

已配置 crontab @reboot，自动执行 `/root/lotto/start.sh`。

```bash
# 检查是否已配置
crontab -l | grep start.sh
```

如未配置，手动添加：

```bash
chmod +x /root/lotto/start.sh
(crontab -l 2>/dev/null; echo "@reboot /root/lotto/start.sh") | crontab -
```

---

## 🏗 系统架构

```
/root/lotto/
├── app.py                  # Streamlit 主界面（双彩种 tabs）
├── config_base.py          # 彩票配置基类
├── config_dlt.py           # 大乐透配置（开奖日: 一/三/六 21:25）
├── config_ssq.py           # 双色球配置（开奖日: 二/四/日 21:15）
├── data_fetcher.py         # 数据获取（500.com HTML / CWL API）
├── models/
│   ├── ensemble.py         # 5策略推理引擎 + CRF评分
│   ├── statistical.py      # 频率/泊松/蒙特卡洛模型
│   └── timeseries.py       # 指数平滑模型
├── utils/
│   ├── predictions.py      # 预测封存/比对/历史管理
│   └── helpers.py          # 日志/验证工具
├── data/
│   ├── dlt/
│   │   ├── dlt_history.csv      # 开奖历史数据
│   │   ├── predictions.json     # 联合预测文件
│   │   └── predictions/         # 每期独立文件（容灾备份）
│   │       ├── 26060.json
│   │       └── ...
│   └── ssq/
│       ├── ssq_history.csv
│       ├── predictions.json
│       └── predictions/
│           ├── 26062.json
│           └── ...
├── start.sh                # 开机自启脚本
├── README.md               # 本文件
└── REQUIREMENTS.md         # 原始需求文档
```

---

## 🔄 核心工作流

### 预测生命周期

```
① 产生预测（active, 未开奖）
    ↓ 开奖日到来
② 刷新页面 / 点"强制检查逾期比对"
    ↓ 自动联网拉取最新开奖号码
③ 比对完成 → 预测变为 completed → 进入"预测历史"
    ↓ 自动检测无活跃预测
④ 自动生成下一期预测（active, 未开奖）
    ↓ 页面始终只显示未开奖的预测
```

### 强制检查逾期比对

点击 **"🔍 强制检查逾期比对"** 按钮：
1. 从 500.com / CWL API 拉取最新开奖数据
2. 对比当前 active 预测的期号
3. 如果该期已开奖 → 更新命中数据 → 标记为 completed
4. 自动检测无 active 预测 → 自动生成下一期
5. 页面始终显示最新的未开奖预测

---

## 📊 页面说明

### 预测结果（默认页）
- 始终显示 **下一期未开奖** 的预测号码
        - 5 组推荐号码，全部基于追冷算法推理：
        - ❄️ **追冷A-E**：不同冷号阈值 + 间隔异常度 + 近期升温信号

### 预测历史
- **最近10期开奖号码**：从 CSV 数据自动滚动更新
- **我的预测记录**：所有历史预测（active + completed），保留最多 20 条

---

## 💾 数据保护机制

预测数据有三层保护，**不会因单文件删除而丢失**：

| 层级 | 存储方式 | 路径 |
|------|---------|------|
| 🟢 主文件 | 联合 JSON | `data/*/predictions.json` |
| 🟡 独立文件 | 每期一个 JSON | `data/*/predictions/{期号}.json` |
| 🔵 Git 跟踪 | predictions.json 已入版本控制 | 每次 `git commit` 备份 |

**恢复方法**：`_load_all()` 优先从独立文件恢复，再合并联合文件。某个文件损坏不影响其他期。

---

## 🧪 常见操作

### 重新计算推荐号码
点击 **"🔄 重新计算"** 按钮，用最新算法重新评估。仅当号码或分数有变化时才更新，无变化则保留原有推荐。

### 数据源
- 双色球：优先使用 CWL 政府 API（`cwl.gov.cn`），500.com 作为备选
- 大乐透：使用 500.com HTML 解析
- 数据首次加载后缓存到 CSV，后续启动秒开

### 开奖排期
| 彩种 | 开奖日 | 开奖时间 |
|------|--------|---------|
| 大乐透 (DLT) | 周一、周三、周六 | 21:25 |
| 双色球 (SSQ) | 周二、周四、周日 | 21:15 |

---
---
## 🔖 版本控制与回滚

### 已打标签

| 标签名 | 对应阶段 | 日期 |
|--------|---------|------|
| `firststage` | v1.0 双色球+大乐透统一预测 | 2026-05 |
| `secondstage` | 多算法融合+预测生命周期 | 2026-05 |
| `fourstage` | 预测生命周期自动流转+数据容灾 | 2026-06 |
| `spiral-matrix-v2` | **当前** — 螺旋矩阵覆盖设计完整版 | 2026-07-13 |

### 一键回滚命令

**回滚到上一个稳定版本（fourstage）：**
```bash
cd /root/lotto
git checkout fourstage
# 重启服务
pkill -f "streamlit.*lotto" 2>/dev/null; sleep 1
streamlit run app.py --server.port 8502 --server.headless true &
```

**回滚到指定标签：**
```bash
cd /root/lotto
git checkout <tag-name>
# 同上重启
```

**查看所有可回滚版本：**
```bash
cd /root/lotto && git tag -l
```

**保留当前数据但回滚代码：**
```bash
cd /root/lotto
# 先备份当前数据
cp data/dlt/predictions.json data/dlt/predictions.json.bak
cp data/ssq/predictions.json data/ssq/predictions.json.bak
# 回滚代码但保留工作区文件
git checkout <tag-name> -- .
# 恢复最新数据
cp data/dlt/predictions.json.bak data/dlt/predictions.json
cp data/ssq/predictions.json.bak data/ssq/predictions.json
rm data/dlt/predictions.json.bak data/ssq/predictions.json.bak
# 重启
```

> 只需告诉我 **"回滚到 <标签名>"**，我会自动执行以上命令。

---

## ❗ 注意事项

1. **不要删除 `data/*/predictions/` 目录下的独立文件**，它们是历史数据保护的关键
2. **不要手动编辑 predictions.json**，由系统自动管理
3. 第 2 次启动后数据自动缓存，无需联网
4. 系统仅供学习研究，彩票是随机游戏，没有任何方法保证中奖
