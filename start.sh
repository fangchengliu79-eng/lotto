#!/bin/bash
# 开机自启：大乐透+双色球 预测服务
cd /root/lotto || exit 1
exec streamlit run app.py --server.port 8502 --server.headless true >> /root/lotto/streamlit.log 2>&1
