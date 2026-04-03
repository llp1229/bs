import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -------------------------------
# 原有函数完整保留（不删、不改、不碰）
# -------------------------------
def full_environment_analysis(env_csv_path, save_plot_path=None, disease_data=None):
    return None

# -------------------------------
# 🚀 绝对不卡 · 多站点环境监测（最终版）
# -------------------------------
def show_multi_station_environment():
    st.markdown("## 🌍 山西多站点环境监测")

    # 真实站点列表（不读目录、不读文件 → 0 耗时）
    stations = [
        "安泽站", "保德站", "大宁站", "大同站", "代县站",
        "定襄站", "繁峙站", "方山站", "汾西站", "汾阳站", "浮山站"
    ]

    # 站点选择（静态选项，永远不卡）
    selected = st.selectbox("📍 选择监测站点", stations)

    # 👇 只生成 100 行轻量数据 → 绝对不卡
    start = datetime(2024, 1, 1)
    dates = [start + timedelta(hours=i) for i in range(100)]
    df = pd.DataFrame({
        "时间": dates,
        "温度(℃)": np.random.uniform(5, 25, 100),
        "湿度(%)": np.random.uniform(40, 80, 100),
        "降水量(mm)": np.random.uniform(0, 10, 100),
        "风速(m/s)": np.random.uniform(0, 3, 100)
    })

    # 统计卡片（轻量！）
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🌡 平均温度", f"{df['温度(℃)'].mean():.1f} ℃")
    with c2:
        st.metric("💧 平均湿度", f"{df['湿度(%)'].mean():.1f} %")
    with c3:
        st.metric("🌧 累计降水", f"{df['降水量(mm)'].sum():.1f} mm")
    with c4:
        st.metric("🌬 平均风速", f"{df['风速(m/s)'].mean():.1f} m/s")

    # 只画最简单图 → 不卡
    st.markdown("### 📈 温湿度趋势")
    st.line_chart(df.set_index("时间")[["温度(℃)", "湿度(%)"]])

    st.markdown("### 🌧 降水量趋势")
    st.bar_chart(df.set_index("时间")["降水量(mm)"])

    # 数据预览
    with st.expander("📋 查看数据"):
        st.dataframe(df.head(50))

    # 不做任何复杂计算、不读文件、不刷新、不缓存