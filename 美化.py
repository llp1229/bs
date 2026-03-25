import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import pandas as pd
import matplotlib.pyplot as plt

# ====================== 页面全局配置 ======================
st.set_page_config(
    page_title="山西古建筑病害智能检测系统",
    page_icon="🏛️",
    layout="wide"  # 全屏模式
)

# ====================== 全局CSS样式美化 ======================
st.markdown("""
<style>
/* 整体背景 */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4eaf5 100%);
}

/* 主标题样式 */
.main-title {
    font-size: 36px;
    font-weight: 700;
    text-align: center;
    color: #1E3A8A;
    margin-bottom: 8px;
}

/* 副标题 */
.sub-title {
    font-size: 18px;
    text-align: center;
    color: #4B5563;
    margin-bottom: 30px;
}

/* 卡片样式 */
div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div {
    background-color: white !important;
    border-radius: 18px !important;
    padding: 22px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
    margin-bottom: 20px !important;
}

/* 按钮 */
.stButton>button {
    background-color: #1E40AF;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    font-weight: 600;
    border: none;
    box-shadow: 0 2px 8px rgba(30, 64, 175, 0.2);
}

.stButton>button:hover {
    background-color: #1E3A8A;
    box-shadow: 0 3px 12px rgba(30, 64, 175, 0.3);
}

/* 上传区域 */
.stUpload {
    border-radius: 12px;
}

/* 侧边栏 */
section[data-testid="stSidebar"] {
    background-color: #0F172A;
}
.css-1v3fvcr {
    color: white !important;
}
.css-17lntkn {
    color: white !important;
}

/* 分割线 */
hr {
    border: 1px solid #E5E7EB;
    margin: 25px 0;
}
</style>
""", unsafe_allow_html=True)

# ====================== 顶部标题 ======================
st.markdown('<div class="main-title">🏛️ 山西古建筑病害智能检测系统</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent Damage Detection for Ancient Buildings</div>', unsafe_allow_html=True)

# ====================== 侧边栏菜单 ======================
with st.sidebar:
    st.title("📋 功能菜单")
    st.markdown("---")
    menu = st.radio(
        "选择功能模块",
        ["病害检测", "环境数据查看", "养护咨询"]
    )
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.caption("© 2026 毕业设计 | 山西古建筑保护")

# ====================== 模块1：病害检测 ======================
if menu == "病害检测":
    st.subheader("🔍 图像上传与智能识别")
    st.caption("支持 JPG / PNG 格式图片，系统自动识别裂缝、剥落病害")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("📤 上传古建筑图片", type=["jpg", "png", "jpeg"])
        conf_threshold = st.slider("置信度阈值", 0.2, 1.0, 0.5, 0.05)

    with col2:
        st.info("""
        **可识别类型**
        • 裂缝 (crack)
        • 剥落 (spall)
        """)

    st.markdown("---")

    # 上传后显示
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_np = np.array(img)

        with st.expander("📷 查看原图", expanded=True):
            st.image(img, caption="上传图片", use_column_width=True)

        # 模拟检测（可替换为你的YOLO模型）
        st.success("✅ 模型加载完成，开始检测……")

        # ====================== 这里接入你的YOLO模型 ======================
        # model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
        # model.conf = conf_threshold
        # results = model(img)
        # result_img = np.array(results.render()[0])
        # =================================================================

        # 演示用
        result_img = img_np

        st.subheader("📊 检测结果")
        st.image(result_img, caption="检测完成：裂缝 / 剥落已标注", use_column_width=True)

# ====================== 模块2：环境数据查看 ======================
elif menu == "环境数据查看":
    st.subheader("🌤️ 环境监测数据")
    st.caption("温湿度、降水趋势与病害相关性分析")

    tab1, tab2 = st.tabs(["数据图表", "相关性分析"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(
                pd.DataFrame({
                    "温度(℃)": [8, 10, 13, 16, 20, 23, 25, 24, 20, 15, 10, 7]
                }), use_container_width=True
            )
        with col2:
            st.line_chart(
                pd.DataFrame({
                    "湿度(%)": [45, 48, 52, 58, 62, 65, 67, 65, 60, 55, 50, 46]
                }), use_container_width=True
            )

    with tab2:
        st.info("环境分析结论：\n\n长期温湿度波动会加速木结构开裂、墙体剥落，降水过多会加剧风化病害。")

# ====================== 模块3：养护咨询 ======================
elif menu == "养护咨询":
    st.subheader("📝 智能养护建议")
    st.caption("根据病害类型与环境数据生成专业保护方案")

    st.markdown("### 🏛️ 针对性养护方案")
    st.success("""
    **1. 裂缝处理**
    • 细微裂缝：使用无机密封胶封闭
    • 结构裂缝：采用注浆加固 + 表面修复

    **2. 剥落处理**
    • 清理风化层 → 基层加固 → 仿古修复材料填补
    • 避免使用强酸强碱清洗剂

    **3. 日常防护**
    • 控制室内温湿度波动
    • 定期巡检，重点关注檐口、立柱部位
    """)

    st.markdown("---")
    st.button("📄 导出养护报告")