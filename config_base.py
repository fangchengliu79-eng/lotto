"""
彩票配置基类 - 定义大乐透/双色球通用配置接口
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class LotteryConfig:
    """彩种配置 - 所有差异参数在此定义"""
    # 标识
    name: str               # "大乐透" / "双色球"
    short: str              # "dlt" / "ssq"
    code: str               # "dlt" / "ssq"
    icon: str               # "🎯"

    # 号码规则
    main_min: int           # 前区/红球范围最小值
    main_max: int           # 前区/红球范围最大值
    main_count: int         # 前区/红球选号数
    sub_min: int            # 后区/蓝球范围最小值
    sub_max: int            # 后区/蓝球范围最大值
    sub_count: int          # 后区/蓝球选号数

    # 标签
    main_label: str         # "前区" / "红球"
    sub_label: str          # "后区" / "蓝球"
    main_label_en: str      # "front" / "red"
    sub_label_en: str       # "back" / "blue"

    # CSV 列名
    main_cols: List[str] = field(default_factory=list)
    sub_cols: List[str] = field(default_factory=list)

    # 目录
    root_dir: Path = Path(__file__).parent.absolute()
    data_dir: Path = None
    output_dir: Path = None
    history_csv: Path = None
    model_dir: Path = None
    predictions_file: Path = None

    # 数据源
    data_sources: Dict[str, str] = field(default_factory=dict)
    default_source: str = "500com"
    zhcw_path: str = ""

    # 默认参数
    min_history_draws: int = 100
    recommend_groups: int = 5
    top_hot_count: int = 12
    top_cold_count: int = 8

    # 模型权重
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "frequency": 0.25, "poisson": 0.15, "monte_carlo": 0.10,
        "exponential_smoothing": 0.10, "lstm": 0.15,
        "xgboost": 0.15, "random_forest": 0.10,
    })

    # 机器学习参数
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 200, "max_depth": 6,
        "learning_rate": 0.1, "random_state": 42,
    })

    # Streamlit 配置
    streamlit_title: str = ""
    streamlit_icon: str = "🎯"

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.root_dir / "data" / self.short
        if self.output_dir is None:
            self.output_dir = self.root_dir / "output" / self.short
        if self.history_csv is None:
            self.history_csv = self.data_dir / f"{self.short}_history.csv"
        if self.model_dir is None:
            self.model_dir = self.data_dir / "models"
        if self.predictions_file is None:
            self.predictions_file = self.data_dir / "predictions.json"
        if not self.main_cols:
            self.main_cols = [f"{self.main_label_en}_{i}" for i in range(1, self.main_count + 1)]
        if not self.sub_cols:
            self.sub_cols = [f"{self.sub_label_en}_{i}" for i in range(1, self.sub_count + 1)]
        if not self.streamlit_title:
            self.streamlit_title = f"{self.icon} {self.name}数据分析与预测系统"

        # 确保目录存在
        for d in [self.data_dir, self.output_dir, self.model_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def total_combinations(self) -> int:
        """总组合数"""
        from math import comb
        return comb(self.main_max - self.main_min + 1, self.main_count) * \
               comb(self.sub_max - self.sub_min + 1, self.sub_count)
