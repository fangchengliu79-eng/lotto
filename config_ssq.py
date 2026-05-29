"""
双色球 (SSQ) 配置
"""
from config_base import LotteryConfig

SSQ = LotteryConfig(
    name="双色球",
    short="ssq",
    code="ssq",
    icon="🔴",
    main_min=1, main_max=33, main_count=6,
    sub_min=1, sub_max=16, sub_count=1,
    main_label="红球", sub_label="蓝球",
    main_label_en="red", sub_label_en="blue",
    zhcw_path="https://www.zhcw.com/kjxx/ssq/",
    data_sources={
        "500com": "https://datachart.500.com/ssq/history/newinc/history.php?start={start}&end={end}",
        "cwl_gov": "https://www.lottery.gov.cn/historykj/history.jspx?_ltype=ssq",
    },
    streamlit_title="🔴 双色球数据分析与预测系统",
    streamlit_icon="🔴",
    top_hot_count=18,  # 双色球33个红球，覆盖一半以上
)
