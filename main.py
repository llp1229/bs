# 基础库导入
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import requests
import datetime
import os
import base64
# 新增：处理路径，适配不同操作系统
import pathlib
import warnings

warnings.filterwarnings('ignore')

# 解决Matplotlib中文显示问题（适配山西古建筑中文标注）
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150


# ---------------------- 工具函数（整合utils模块逻辑） ----------------------
# 1. 图片预处理（适配YOLO输入+山西古建筑图片特性）
def preprocess_for_yolo(uploaded_file):
    """一站式图片预处理：读取→缩放→归一化"""
    try:
        # 读取上传文件
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("上传的图片无法解析，请确认是jpg/png格式")

        # 缩放为YOLO标准尺寸（800x800，匹配训练时的尺寸），记录缩放比例和偏移
        h, w = img.shape[:2]
        scale = min(800 / w, 800 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 补边到800x800（保持比例）
        img_padded = np.zeros((800, 800, 3), dtype=np.uint8)
        offset_x, offset_y = (800 - new_w) // 2, (800 - new_h) // 2
        img_padded[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = img_resized

        # 转换为RGB（YOLO默认输入）
        img_processed = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        return img_processed, scale, (offset_x, offset_y), img  # 返回原始图片用于对比
    except Exception as e:
        st.error(f"图片预处理失败：{str(e)}")
        return None, None, None, None


# 2. 图片转字节（Streamlit展示用）
def image_to_bytes(img):
    """将OpenCV图片转为Streamlit可展示的字节流"""
    try:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        _, img_encoded = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return img_encoded.tobytes()
    except Exception as e:
        st.error(f"图片转字节失败：{str(e)}")
        return b''


# 3. 环境数据全量分析（适配山西气候特点）
def full_environment_analysis(env_csv_path, save_plot_path=None, disease_data=None):
    """环境数据：统计+可视化+相关性分析"""
    try:
        # 读取数据
        if not os.path.exists(env_csv_path):
            raise FileNotFoundError("环境数据文件不存在，请先运行'环境数据查看'生成")
        env_data = pd.read_csv(env_csv_path)
        env_data["时间"] = pd.to_datetime(env_data["时间"], errors='coerce')

        # 去除无效时间数据
        env_data = env_data.dropna(subset=["时间"])
        if len(env_data) == 0:
            raise ValueError("环境数据中无有效时间信息")

        # 1. 数据概览（统计值）
        stats = {
            "数据时间范围": f"{env_data['时间'].min()} 至 {env_data['时间'].max()}",
            "平均温度(℃)": round(env_data["温度(℃)"].mean(), 1),
            "平均湿度(%)": round(env_data["湿度(%)"].mean(), 1),
            "累计降水量(mm)": round(env_data["降水量(mm)"].sum(), 1),
            "温度波动范围(℃)": f"{env_data['温度(℃)'].min()} ~ {env_data['温度(℃)'].max()}",
            "湿度波动范围(%)": f"{env_data['湿度(%)'].min()} ~ {env_data['湿度(%)'].max()}"
        }

        # 2. 趋势图绘制（适配山西古建筑场景配色）
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        # 温度趋势
        ax1.plot(env_data["时间"], env_data["温度(℃)"], color="#c0392b", linewidth=2, label="温度", marker='o',
                 markersize=3)
        ax1.fill_between(env_data["时间"], env_data["温度(℃)"], alpha=0.2, color="#c0392b")
        ax1.set_title("山西古建筑周边温度变化趋势", fontsize=14, fontweight="bold", pad=15)
        ax1.set_ylabel("温度(℃)", fontsize=12)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        # 湿度趋势
        ax2.plot(env_data["时间"], env_data["湿度(%)"], color="#2980b9", linewidth=2, label="湿度", marker='s',
                 markersize=3)
        ax2.fill_between(env_data["时间"], env_data["湿度(%)"], alpha=0.2, color="#2980b9")
        ax2.set_title("山西古建筑周边湿度变化趋势", fontsize=14, fontweight="bold", pad=15)
        ax2.set_ylabel("湿度(%)", fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        # 降水趋势
        ax3.bar(env_data["时间"], env_data["降水量(mm)"], color="#27ae60", alpha=0.8, label="降水量",
                edgecolor="#219653",
                linewidth=0.5)
        ax3.set_title("山西古建筑周边降水量变化趋势", fontsize=14, fontweight="bold", pad=15)
        ax3.set_ylabel("降水量(mm)", fontsize=12)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        plt.tight_layout(pad=2.0)

        # 保存图片（可选）
        if save_plot_path:
            os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
            plt.savefig(save_plot_path, dpi=150, bbox_inches="tight", facecolor='white')

        # 3. 相关性分析（温湿度/降水，适配古建筑病害）
        corr = env_data[["温度(℃)", "湿度(%)", "降水量(mm)"]].corr()
        corr_text = f"""
        🔍 山西古建筑环境相关性分析：
        - 温湿度相关系数：{round(corr.loc['温度(℃)', '湿度(%)'], 2)}（负相关易致木材开裂）
        - 温度-降水相关系数：{round(corr.loc['温度(℃)', '降水量(mm)'], 2)}
        - 湿度-降水相关系数：{round(corr.loc['湿度(%)', '降水量(mm)'], 2)}（正相关易致砖石剥落）
        """

        # 返回结果
        return {
            "数据概览": stats,
            "趋势图": fig,
            "相关性分析": corr_text
        }
    except Exception as e:
        st.error(f"环境数据分析失败：{str(e)}")
        return None


# 4. 养护咨询API调用（强化山西古建筑特性）
def get_consult_suggestion(disease_type, disease_count, api_type="tongyi", api_key="", env_stats=None):
    """调用大模型生成山西古建筑专属养护建议"""
    try:
        # 构建Prompt（融合山西古建筑特性：木构/砖石、晋北/晋南气候差异）
        env_desc = f"""
        山西古建筑周边环境：
        - 平均温度{env_stats['平均温度(℃)']}℃，温度波动{env_stats['温度波动范围(℃)']}
        - 平均湿度{env_stats['平均湿度(%)']}%，湿度波动{env_stats['湿度波动范围(%)']}
        - 累计降水量{env_stats['累计降水量(mm)']}mm
        """ if env_stats else "无环境数据（默认按山西晋中地区气候）"

        prompt = f"""
        你是山西古建筑（木构/砖石）养护专家，针对以下情况给出实用建议：
        1. 检测病害：{disease_type}，数量：{disease_count}
        2. 环境背景：{env_desc}
        3. 建筑类型：山西传统木构/砖石古建筑（如应县木塔、晋祠、平遥古城类）
        要求：
        - 严格贴合山西当地气候和建筑材料特性；
        - 分点说明，每条不超20字，非专业人员可操作；
        - 聚焦病害修复+预防，最多4条核心建议；
        - 语言通俗，避免专业术语。
        """

        # 通义千问API调用（稳定版）
        if api_type == "tongyi":
            if not api_key:
                raise ValueError("请输入通义千问API Key（前往https://dashscope.aliyun.com/获取）")
            url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "qwen-turbo",
                "input": {"messages": [{"role": "user", "content": prompt}]},
                "parameters": {
                    "result_format": "text",
                    "max_tokens": 300,
                    "temperature": 0.7,
                    "top_p": 0.8
                }
            }
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["output"]["choices"][0]["message"]["content"]
        else:
            raise NotImplementedError("暂仅支持通义千问API")
    except requests.exceptions.Timeout:
        st.error("API请求超时，请检查网络或稍后重试")
        return None
    except ValueError as e:
        st.error(f"参数错误：{str(e)}")
        return None
    except Exception as e:
        st.error(f"API调用失败：{str(e)}")
        return None


# 5. 读取本地背景图/视频并编码（适配zy文件夹相对路径）
def get_local_file_base64(file_path):
    """读取本地文件（图/视频）并转换为base64编码，适配相对路径"""
    try:
        if not os.path.exists(file_path):
            return ""
        with open(file_path, "rb") as f:
            file_base64 = base64.b64encode(f.read()).decode()
        return file_base64
    except Exception as e:
        st.warning(f"读取文件失败：{str(e)}")
        return ""


# ---------------------- 页面基础设置（美化核心，适配zy文件夹） ----------------------
def setup_page_style():
    """设置页面样式，背景图优先读取项目根目录/zy/背景.jpg"""
    # 背景图路径：优先项目根目录的zy文件夹（你放文件的目录），相对路径！
    bg_image_paths = [
        "zy/背景.jpg",
        "zy/系统网页背景图.png",
        "背景.jpg",
        "系统网页背景图.png"
    ]

    bg_image_path = ""
    for path in bg_image_paths:
        if os.path.exists(path):
            bg_image_path = path
            break

    bg_img_base64 = get_local_file_base64(bg_image_path)

    # 构建完整的CSS样式
    if bg_img_base64:
        css_style = """
        <style>
        /* 全局重置 */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        /* 全局背景设置 */
        .stApp {
            background-image: url('data:image/png;base64,""" + bg_img_base64 + """');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-color: #f8f9fa; /* 备用背景色 */
            min-height: 100vh;
        }
        /* 主内容容器 */
        .main .block-container {
            background-color: rgba(255, 255, 255, 0.92);
            border-radius: 16px;
            padding: 2.5rem;
            margin-top: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.8);
        }
        /* 侧边栏美化 */
        .stSidebar {
            background-color: rgba(240, 242, 246, 0.95);
            border-radius: 12px;
            margin: 1rem;
            padding: 1.5rem !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }
        /* 侧边栏标题 */
        .stSidebar h2, .stSidebar h3 {
            color: #2c3e50;
            font-weight: 700;
            margin-bottom: 1rem;
            border-bottom: 2px solid #e74c3c;
            padding-bottom: 0.5rem;
        }
        /* 按钮美化 */
        .stButton>button {
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(231, 76, 60, 0.15);
        }
        .stButton>button:hover {
            background-color: #c0392b;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(231, 76, 60, 0.2);
        }
        .stButton>button:active {
            transform: translateY(0);
        }
        /* 主要按钮（primary） */
        .stButton>button[type="primary"] {
            background-color: #2980b9;
            box-shadow: 0 4px 6px rgba(41, 128, 185, 0.15);
        }
        .stButton>button[type="primary"]:hover {
            background-color: #1f618d;
            box-shadow: 0 6px 12px rgba(41, 128, 185, 0.2);
        }
        /* 滑块美化 */
        .stSlider {
            padding: 1rem 0;
        }
        .stSlider [data-baseweb="slider"] {
            margin: 0;
        }
        .stSlider [data-baseweb="slider"] .thumb {
            background-color: #e74c3c;
            border: 2px solid white;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        .stSlider [data-baseweb="slider"] .track-1 {
            background-color: #e74c3c;
        }
        .stSlider [data-baseweb="slider"] .track-2 {
            background-color: #e0e0e0;
        }
        /* 输入框美化 */
        .stTextInput>div>div>input, .stSelectbox>div>div>select {
            border-radius: 8px;
            border: 1px solid #ddd;
            padding: 0.6rem 1rem;
            font-size: 14px;
            transition: border 0.3s ease;
        }
        .stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
            border-color: #2980b9;
            outline: none;
            box-shadow: 0 0 0 2px rgba(41, 128, 185, 0.2);
        }
        /* 文件上传组件 */
        .stFileUploader {
            padding: 1.5rem;
            border: 2px dashed #ddd;
            border-radius: 12px;
            background-color: rgba(255, 255, 255, 0.8);
            transition: all 0.3s ease;
        }
        .stFileUploader:hover {
            border-color: #2980b9;
            background-color: rgba(245, 250, 255, 0.9);
        }
        /* 下载按钮 */
        .stDownloadButton>button {
            background-color: #27ae60;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stDownloadButton>button:hover {
            background-color: #219653;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(39, 174, 96, 0.2);
        }
        /* 卡片/容器样式 */
        .dataframe, .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        /* 文字样式优化 */
        h1 {
            color: #2c3e50 !important;
            font-weight: 800 !important;
            line-height: 1.2 !important;
            margin-bottom: 0.5rem !important;
        }
        h2 {
            color: #34495e !important;
            font-weight: 700 !important;
            margin: 1.5rem 0 1rem 0 !important;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #f1c40f;
        }
        h3 {
            color: #34495e !important;
            font-weight: 600 !important;
            margin: 1rem 0 0.8rem 0 !important;
        }
        h4 {
            color: #7f8c8d !important;
            font-weight: 500 !important;
        }
        p, div, span {
            color: #34495e !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
        }
        /* 视频样式 */
        .custom-video {
            width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .custom-video:hover {
            transform: scale(1.02);
        }
        /* 加载动画美化 */
        .stSpinner > div > div {
            border-top-color: #e74c3c !important;
            width: 2rem !important;
            height: 2rem !important;
        }
        /* 提示框样式 */
        .stSuccess, .stInfo, .stWarning, .stError {
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid;
        }
        .stSuccess {
            background-color: rgba(46, 204, 113, 0.1);
            border-left-color: #27ae60;
        }
        .stInfo {
            background-color: rgba(52, 152, 219, 0.1);
            border-left-color: #2980b9;
        }
        .stWarning {
            background-color: rgba(241, 196, 15, 0.1);
            border-left-color: #f39c12;
        }
        .stError {
            background-color: rgba(231, 76, 60, 0.1);
            border-left-color: #e74c3c;
        }
        /* 分割线 */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(to right, transparent, #ddd, transparent);
            margin: 2rem 0;
        }
        /* 响应式适配 */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1.5rem;
                margin-top: 1rem;
            }
            .stSidebar {
                margin: 0.5rem;
                padding: 1rem !important;
            }
            h1 {
                font-size: 28px !important;
            }
            h2 {
                font-size: 20px !important;
            }
        }
        </style>
        """
    else:
        # 无背景图的情况
        css_style = """
        <style>
        /* 全局重置 */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        /* 全局背景设置 */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
            min-height: 100vh;
        }
        /* 主内容容器 */
        .main .block-container {
            background-color: rgba(255, 255, 255, 0.92);
            border-radius: 16px;
            padding: 2.5rem;
            margin-top: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.8);
        }
        /* 侧边栏美化 */
        .stSidebar {
            background-color: rgba(240, 242, 246, 0.95);
            border-radius: 12px;
            margin: 1rem;
            padding: 1.5rem !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }
        /* 侧边栏标题 */
        .stSidebar h2, .stSidebar h3 {
            color: #2c3e50;
            font-weight: 700;
            margin-bottom: 1rem;
            border-bottom: 2px solid #e74c3c;
            padding-bottom: 0.5rem;
        }
        /* 按钮美化 */
        .stButton>button {
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(231, 76, 60, 0.15);
        }
        .stButton>button:hover {
            background-color: #c0392b;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(231, 76, 60, 0.2);
        }
        .stButton>button:active {
            transform: translateY(0);
        }
        /* 主要按钮（primary） */
        .stButton>button[type="primary"] {
            background-color: #2980b9;
            box-shadow: 0 4px 6px rgba(41, 128, 185, 0.15);
        }
        .stButton>button[type="primary"]:hover {
            background-color: #1f618d;
            box-shadow: 0 6px 12px rgba(41, 128, 185, 0.2);
        }
        /* 滑块美化 */
        .stSlider {
            padding: 1rem 0;
        }
        .stSlider [data-baseweb="slider"] {
            margin: 0;
        }
        .stSlider [data-baseweb="slider"] .thumb {
            background-color: #e74c3c;
            border: 2px solid white;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        .stSlider [data-baseweb="slider"] .track-1 {
            background-color: #e74c3c;
        }
        .stSlider [data-baseweb="slider"] .track-2 {
            background-color: #e0e0e0;
        }
        /* 输入框美化 */
        .stTextInput>div>div>input, .stSelectbox>div>div>select {
            border-radius: 8px;
            border: 1px solid #ddd;
            padding: 0.6rem 1rem;
            font-size: 14px;
            transition: border 0.3s ease;
        }
        .stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
            border-color: #2980b9;
            outline: none;
            box-shadow: 0 0 0 2px rgba(41, 128, 185, 0.2);
        }
        /* 文件上传组件 */
        .stFileUploader {
            padding: 1.5rem;
            border: 2px dashed #ddd;
            border-radius: 12px;
            background-color: rgba(255, 255, 255, 0.8);
            transition: all 0.3s ease;
        }
        .stFileUploader:hover {
            border-color: #2980b9;
            background-color: rgba(245, 250, 255, 0.9);
        }
        /* 下载按钮 */
        .stDownloadButton>button {
            background-color: #27ae60;
            color: white;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stDownloadButton>button:hover {
            background-color: #219653;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(39, 174, 96, 0.2);
        }
        /* 卡片/容器样式 */
        .dataframe, .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        /* 文字样式优化 */
        h1 {
            color: #2c3e50 !important;
            font-weight: 800 !important;
            line-height: 1.2 !important;
            margin-bottom: 0.5rem !important;
        }
        h2 {
            color: #34495e !important;
            font-weight: 700 !important;
            margin: 1.5rem 0 1rem 0 !important;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #f1c40f;
        }
        h3 {
            color: #34495e !important;
            font-weight: 600 !important;
            margin: 1rem 0 0.8rem 0 !important;
        }
        h4 {
            color: #7f8c8d !important;
            font-weight: 500 !important;
        }
        p, div, span {
            color: #34495e !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
        }
        /* 视频样式 */
        .custom-video {
            width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .custom-video:hover {
            transform: scale(1.02);
        }
        /* 加载动画美化 */
        .stSpinner > div > div {
            border-top-color: #e74c3c !important;
            width: 2rem !important;
            height: 2rem !important;
        }
        /* 提示框样式 */
        .stSuccess, .stInfo, .stWarning, .stError {
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid;
        }
        .stSuccess {
            background-color: rgba(46, 204, 113, 0.1);
            border-left-color: #27ae60;
        }
        .stInfo {
            background-color: rgba(52, 152, 219, 0.1);
            border-left-color: #2980b9;
        }
        .stWarning {
            background-color: rgba(241, 196, 15, 0.1);
            border-left-color: #f39c12;
        }
        .stError {
            background-color: rgba(231, 76, 60, 0.1);
            border-left-color: #e74c3c;
        }
        /* 分割线 */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(to right, transparent, #ddd, transparent);
            margin: 2rem 0;
        }
        /* 响应式适配 */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1.5rem;
                margin-top: 1rem;
            }
            .stSidebar {
                margin: 0.5rem;
                padding: 1rem !important;
            }
            h1 {
                font-size: 28px !important;
            }
            h2 {
                font-size: 20px !important;
            }
        }
        </style>
        """

    # 应用CSS样式
    st.markdown(css_style, unsafe_allow_html=True)


# ---------------------- 加载自定义YOLO模型（核心：默认相对路径，自动解析） ----------------------
@st.cache_resource
def load_custom_yolo_model(model_path):
    """加载训练好的山西古建筑病害检测模型，自动解析相对路径为当前脚本绝对路径"""
    try:
        # 关键：将输入的路径（无论相对/绝对）解析为【当前main.py所在目录】的绝对路径
        model_path_obj = pathlib.Path(model_path).resolve()

        # 检查模型文件是否存在
        if not model_path_obj.exists():
            st.error(f"❌ 自定义模型文件不存在！解析后的绝对路径：{model_path_obj}")
            return None

        # 加载模型
        model = YOLO(str(model_path_obj))
        st.success(f"✅ 山西古建筑病害模型加载成功！")
        st.info(f"📌 模型识别类别：{model.names} | 模型路径：{model_path_obj}")
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        st.error("💡 建议：检查模型文件是否完整（需为YOLO的.pt格式），或确认路径正确")
        return None


# ---------------------- 主程序 ----------------------
def main():
    # 页面基础设置
    st.set_page_config(
        page_title="山西古建筑健康监测与养护咨询系统",
        page_icon="🏯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 设置页面样式（适配zy文件夹背景/视频）
    setup_page_style()

    # 标题+本地视频分栏（美化布局，视频优先读取项目根目录/zy/古建筑.mp4）
    col_title, col_video = st.columns([3, 1.2], gap="large")  # 调整比例，增加间距
    with col_title:
        # 主标题（渐变文字+装饰）
        st.markdown(
            """
            <h1 style='font-size: 40px; font-weight: 800; margin-bottom: 12px; 
                       background: linear-gradient(90deg, #e74c3c, #c0392b);
                       -webkit-background-clip: text;
                       color: transparent;'>
                🏯 山西古建筑健康监测与养护咨询系统
            </h1>
            """,
            unsafe_allow_html=True
        )
        # 副标题1（精致样式）
        st.markdown(
            "<h3 style='font-size: 22px; margin-bottom: 10px; color: #34495e; font-weight: 600;'>基于Python+Streamlit | 自定义YOLO模型</h3>",
            unsafe_allow_html=True
        )
        # 副标题2（轻量样式）
        st.markdown(
            "<h4 style='font-size: 16px; color: #7f8c8d; line-height: 1.6;'>适配山西木构/砖石古建筑：裂缝、剥落病害检测 + 环境分析 + AI养护建议</h4>",
            unsafe_allow_html=True
        )

    with col_video:
        # 视频路径：优先项目根目录的zy文件夹（你放文件的目录），纯相对路径！
        video_paths = [
            "zy/古建筑.mp4",
            "古建筑.mp4"
        ]
        video_path = ""
        for path in video_paths:
            if os.path.exists(path):
                video_path = path
                break

        if video_path:
            try:
                # 读取视频为base64（0.5倍速播放）
                video_base64 = get_local_file_base64(video_path)
                video_data_url = f"data:video/mp4;base64,{video_base64}"
                # 自定义视频标签
                st.markdown(f"""
                <div style="border-radius: 12px; overflow: hidden; box-shadow: 0 6px 20px rgba(0,0,0,0.1);">
                    <video class="custom-video" controls autoplay loop muted playsinline preload="metadata">
                        <source src="{video_data_url}" type="video/mp4">
                        你的浏览器不支持HTML5视频播放，请更换浏览器后重试。
                    </video>
                </div>
                <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    const video = document.querySelector('.custom-video');
                    video.playbackRate = 0.5;
                    video.play();
                }});
                </script>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ 视频加载失败：{str(e)}")
                st.info("💡 建议：检查视频为MP4格式，且放在zy文件夹下")
        else:
            st.warning(f"📹 未找到视频文件！请将古建筑.mp4放在项目根目录的zy文件夹中")

    # ---------------------- 侧边栏功能选择（模型默认相对路径：model/best.pt） ----------------------
    with st.sidebar:
        # 侧边栏头部装饰
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #eee;">
            <span style="font-size: 20px; font-weight: 700; color: #e74c3c;">📋 功能菜单</span>
        </div>
        """, unsafe_allow_html=True)

        # 功能选择单选框
        function_choice = st.radio(
            "",
            ["病害检测", "环境数据查看", "养护咨询"],
            index=0,
            help="选择需要使用的功能模块"
        )

        st.divider()

        # 使用指南
        st.markdown("""
        <div style="background-color: rgba(52, 152, 219, 0.08); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <p style="font-weight: 600; color: #2980b9; margin-bottom: 0.5rem;">📝 使用指南：</p>
            <ol style="padding-left: 1.5rem; margin: 0; line-height: 1.8;">
                <li>病害检测：上传图片→调整阈值→点击检测</li>
                <li>环境数据：生成/导入CSV→分析温湿度/降水趋势</li>
                <li>养护咨询：输入病害+API Key→生成专属建议</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        # 模型配置（核心：默认相对路径 model/best.pt，推荐把模型放这里）
        st.markdown("""
        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee;">
            <span style="font-size: 18px; font-weight: 700; color: #2980b9;">⚙️ 模型配置</span>
        </div>
        """, unsafe_allow_html=True)

        model_rel_path = st.text_input(
            "",
            value="model/best.pt",  # 默认相对路径：项目根目录下的model文件夹里的best.pt
            help="模型相对/绝对路径均可，推荐：项目根目录/model/best.pt",
            placeholder="输入YOLO模型路径（.pt格式）..."
        )

        # 自动解析模型路径并显示绝对路径，方便排查
        model_abs_path = os.path.abspath(model_rel_path)
        st.markdown(f"""
        <div style="background-color: rgba(241, 196, 15, 0.08); padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0; font-size: 12px;">
            <span style="color: #f39c12; font-weight: 600;">📌 自动解析的绝对路径：</span><br>
            <code style="font-size: 11px; color: #7f8c8d;">{model_abs_path}</code>
        </div>
        """, unsafe_allow_html=True)

        if not os.path.exists(model_abs_path):
            st.error("❌ 模型文件不存在！请按提示放置模型或修改路径")

    # 初始化模型（自动解析相对路径，缓存加载）
    if "model" not in st.session_state:
        with st.spinner("🔧 加载山西古建筑病害检测模型中..."):
            st.session_state["model"] = load_custom_yolo_model(model_abs_path)
    model = st.session_state["model"]

    # ---------------------- 功能1：病害检测（优化交互，一键检测） ----------------------
    if function_choice == "病害检测":
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fdf2e9 0%, #fef5e7 100%); padding: 1rem; border-radius: 12px; margin-bottom: 2rem;">
            <h2 style="margin: 0 !important; padding: 0 !important; border: none !important; color: #e74c3c !important;">📷 山西古建筑病害检测（裂缝/剥落 | 自定义YOLO模型）</h2>
        </div>
        """, unsafe_allow_html=True)

        # 图片上传
        uploaded_file = st.file_uploader(
            "",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=False,
            help="推荐：上传山西古建筑木构/砖石清晰图片，分辨率≥640x640",
            label_visibility="collapsed"
        )

        # 检测阈值设置（分栏）
        if uploaded_file:
            st.markdown("<h3>🔧 检测阈值调节</h3>", unsafe_allow_html=True)
            col_conf, col_iou = st.columns(2, gap="medium")
            with col_conf:
                conf_threshold = st.slider("置信度阈值（越高检测越严格）", 0.1, 0.9, 0.3, 0.05)
            with col_iou:
                iou_threshold = st.slider("IOU阈值（越高重叠框越少）", 0.1, 0.9, 0.4, 0.05)

            # 一键检测按钮
            detect_btn = st.button("🚀 开始检测古建筑病害", type="primary", use_container_width=True)

            # 检测逻辑
            if detect_btn and model:
                with st.spinner("🔍 正在检测山西古建筑病害（裂缝/剥落）..."):
                    # 图片预处理
                    img_processed, scale, offset, img_original = preprocess_for_yolo(uploaded_file)
                    if img_processed is None:
                        st.stop()

                    # YOLO检测（CPU运行，适配所有环境）
                    results = model.predict(
                        img_processed,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        imgsz=800,
                        device="cpu",
                        verbose=False
                    )

                    # 处理检测结果，还原到原始图片坐标
                    result = results[0]
                    detections = result.boxes if result.boxes is not None else []
                    img_with_boxes = img_original.copy()
                    disease_count = {}

                    if len(detections) > 0:
                        for box in detections:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            # 还原缩放和偏移，匹配原始图片尺寸
                            x1 = (x1 - offset[0]) / scale
                            y1 = (y1 - offset[1]) / scale
                            x2 = (x2 - offset[0]) / scale
                            y2 = (y2 - offset[1]) / scale
                            # 确保坐标在图片范围内
                            x1, y1 = max(0, int(x1)), max(0, int(y1))
                            x2, y2 = min(img_original.shape[1], int(x2)), min(img_original.shape[0], int(y2))
                            # 获取病害类型和置信度
                            cls = int(box.cls[0])
                            cls_name = model.names[cls]
                            conf = round(float(box.conf[0]), 2)
                            # 统计病害数量
                            disease_count[cls_name] = disease_count.get(cls_name, 0) + 1
                            # 绘制检测框和标签
                            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img_with_boxes, f"{cls_name} {conf}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 展示原始图+检测结果图（分栏）
                    st.markdown("<h3>🎯 病害检测结果对比</h3>", unsafe_allow_html=True)
                    col_ori, col_det = st.columns(2, gap="medium")
                    with col_ori:
                        st.markdown("<p style='text-align: center; font-weight: 600;'>原始图片</p>",
                                    unsafe_allow_html=True)
                        st.image(image_to_bytes(img_original), use_container_width=True)
                    with col_det:
                        st.markdown("<p style='text-align: center; font-weight: 600;'>标注结果图</p>",
                                    unsafe_allow_html=True)
                        st.image(image_to_bytes(img_with_boxes), use_container_width=True)

                    # 病害统计结果
                    st.markdown("<h3>📊 病害检测统计</h3>", unsafe_allow_html=True)
                    if disease_count:
                        total = sum(disease_count.values())
                        st.success(
                            f"✅ 共检测到 <span style='font-weight:700; color:#e74c3c;'>{total}</span> 处古建筑病害！",
                            unsafe_allow_html=True)
                        for dis, cnt in disease_count.items():
                            st.info(f"📌 {dis}：{cnt} 处")
                    else:
                        st.info("ℹ️ 未检测到病害，可适当降低「置信度阈值」后重试")

                    # 结果下载
                    st.markdown("<h3>📥 检测结果下载</h3>", unsafe_allow_html=True)
                    st.download_button(
                        label="下载病害标注图片",
                        data=image_to_bytes(img_with_boxes),
                        file_name=f"山西古建筑病害检测_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )

    # ---------------------- 功能2：环境数据查看（自动生成示例数据） ----------------------
    elif function_choice == "环境数据查看":
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ebf5fb 0%, #e8f4f8 100%); padding: 1rem; border-radius: 12px; margin-bottom: 2rem;">
            <h2 style="margin: 0 !important; padding: 0 !important; border: none !important; color: #2980b9 !important;">🌡️ 山西古建筑周边环境数据（温湿度/降水 | 全量分析）</h2>
        </div>
        """, unsafe_allow_html=True)

        # 环境数据文件路径（默认data文件夹）
        env_csv_path = st.text_input(
            "环境数据CSV路径",
            value="data/environment_data.csv",
            help="相对/绝对路径均可，无文件将自动生成山西晋中模拟数据"
        )
        os.makedirs(os.path.dirname(env_csv_path), exist_ok=True)

        # 生成示例数据按钮
        if st.button("📄 生成山西地区模拟环境数据", use_container_width=True):
            with st.spinner("📊 生成山西晋中冬季环境模拟数据..."):
                dates = pd.date_range(start="2026-01-01", end="2026-01-31", freq="D")
                env_data = pd.DataFrame({
                    "时间": dates,
                    "温度(℃)": np.random.uniform(-8, 12, len(dates)),  # 山西冬季温度
                    "湿度(%)": np.random.uniform(25, 60, len(dates)),  # 山西冬季湿度
                    "降水量(mm)": np.random.choice([0, 0.2, 0.5, 1], len(dates), p=[0.9, 0.05, 0.03, 0.02])  # 少降水
                })
                env_data.to_csv(env_csv_path, index=False, encoding="utf-8")
            st.success(f"✅ 山西地区环境模拟数据已生成：{os.path.abspath(env_csv_path)}")

        # 开始分析按钮
        if st.button("🚀 开始环境数据分析", type="primary", use_container_width=True):
            with st.spinner("🔍 分析温湿度/降水趋势及相关性..."):
                analysis_result = full_environment_analysis(
                    env_csv_path=env_csv_path,
                    save_plot_path="data/plots/shanxi_env_trend.png"
                )
                if analysis_result:
                    # 数据概览
                    st.markdown("<h3>📊 环境数据概览</h3>", unsafe_allow_html=True)
                    stats_html = "<div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-bottom: 1.5rem;'>"
                    for key, value in analysis_result["数据概览"].items():
                        stats_html += f"""
                        <div style='display: flex; justify-content: space-between; padding: 0.8rem 0; border-bottom: 1px solid #eee;'>
                            <span style='font-weight: 600; color: #34495e;'>{key}：</span>
                            <span style='color: #2980b9; font-weight: 700;'>{value}</span>
                        </div>
                        """
                    stats_html += "</div>"
                    st.markdown(stats_html, unsafe_allow_html=True)

                    # 趋势图
                    st.markdown("<h3>📈 环境趋势图（山西古建筑专属）</h3>", unsafe_allow_html=True)
                    st.pyplot(analysis_result["趋势图"], use_container_width=True)

                    # 相关性分析
                    st.markdown("<h3>🔗 环境相关性分析（适配古建筑病害）</h3>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="background-color: rgba(46, 204, 113, 0.08); padding: 1.2rem; border-radius: 8px; border-left: 4px solid #27ae60;">
                        {analysis_result["相关性分析"]}
                    </div>
                    """, unsafe_allow_html=True)

                    # 下载趋势图
                    if os.path.exists("data/plots/shanxi_env_trend.png"):
                        with open("data/plots/shanxi_env_trend.png", "rb") as f:
                            st.download_button(
                                label="📥 下载环境趋势图",
                                data=f,
                                file_name=f"山西古建筑环境趋势_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )

    # ---------------------- 功能3：养护咨询（结合病害+环境，AI生成建议） ----------------------
    elif function_choice == "养护咨询":
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f2f9e9 0%, #eafaf1 100%); padding: 1rem; border-radius: 12px; margin-bottom: 2rem;">
            <h2 style="margin: 0 !important; padding: 0 !important; border: none !important; color: #27ae60 !important;">💡 山西古建筑智能养护咨询（结合病害+环境）</h2>
        </div>
        """, unsafe_allow_html=True)

        # 输入参数分栏
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown("<p style='font-weight: 600; color: #2c3e50;'>📌 病害信息</p>", unsafe_allow_html=True)
            disease_type = st.selectbox(
                "",
                ["裂缝", "剥落", "裂缝+剥落", "其他"],
                help="选择检测到的主要病害类型",
                label_visibility="collapsed"
            )
            disease_count = st.number_input(
                "病害数量（处）",
                min_value=1, max_value=100, value=3, step=1,
                help="输入检测到的病害总数量"
            )
        with col2:
            st.markdown("<p style='font-weight: 600; color: #2c3e50;'>🔧 环境/API配置</p>", unsafe_allow_html=True)
            use_env_data = st.checkbox("结合环境数据（更精准）", value=True)
            api_key = st.text_input(
                "通义千问API Key",
                value="",
                type="password",
                help="前往https://dashscope.aliyun.com/获取，免费额度足够使用"
            )

        # 生成建议按钮
        if st.button("🚀 生成山西古建筑专属养护建议", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ 请输入通义千问API Key（免费获取，步骤简单）")
                st.stop()
            with st.spinner("🧠 AI分析病害+环境，生成养护建议..."):
                # 加载环境数据（如果勾选）
                env_stats = None
                if use_env_data:
                    env_csv_path = "data/environment_data.csv"
                    if os.path.exists(env_csv_path):
                        env_analysis = full_environment_analysis(env_csv_path)
                        env_stats = env_analysis["数据概览"] if env_analysis else None
                    else:
                        st.warning("⚠️ 未找到环境数据，将按山西晋中默认气候生成建议")
                # 调用API生成建议
                suggestion = get_consult_suggestion(
                    disease_type=disease_type,
                    disease_count=disease_count,
                    api_key=api_key,
                    env_stats=env_stats
                )
                # 展示建议并提供下载
                # 展示建议并提供下载
                if suggestion:
                    st.markdown("<h3>💡 山西古建筑专属养护建议（AI生成）</h3>", unsafe_allow_html=True)
                    # 先把换行处理好，避免在f-string里用反斜杠
                    formatted_suggestion = suggestion.replace('\n', '<br>')
                    st.markdown(f"""
                    <div style="background-color: rgba(46, 204, 113, 0.08); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #27ae60; line-height: 1.8;">
                    {formatted_suggestion}
                    </div>
                    """, unsafe_allow_html=True)

                    # 下载建议
                    st.download_button(
                        label="📥 下载养护建议（TXT）",
                        data=suggestion,
                        file_name=f"山西古建筑养护建议_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )


# 运行主程序
if __name__ == "__main__":
    main()