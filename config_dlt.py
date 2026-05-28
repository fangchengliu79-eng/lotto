"""
大乐透 (DLT) 配置
"""
from config_base import LotteryConfig

DLT = LotteryConfig(
    name="大乐透",
    short="dlt",
    code="dlt",
    icon="🎯",
    main_min=1, main_max=35, main_count=5,
    sub_min=1, sub_max=12, sub_count=2,
    main_label="前区", sub_label="后区",
    main_label_en="front", sub_label_en="back",
    zhcw_path="https://www.zhcw.com/kjxx/dlt/",
    data_sources={
        "500com": "https://datachart.500.com/dlt/history/newinc/history.php?start={start}&end={end}",
        "cwl_gov": "https://www.lottery.gov.cn/historykj/history.jspx?_ltype=dlt",
    },
    streamlit_title="🎯 大乐透数据分析与预测系统",
    streamlit_icon="🎯",
)
