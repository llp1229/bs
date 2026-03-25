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

# 解决Matplotlib中文显示问题（适配山西古建筑中文标注）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------- 工具函数（整合utils模块逻辑） ----------------------
# 1. 图片预处理（适配YOLO输入+山西古建筑图片特性）
def preprocess_for_yolo(uploaded_file):
    """一站式图片预处理：读取→缩放→归一化"""
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


# 2. 图片转字节（Streamlit展示用）
def image_to_bytes(img):
    """将OpenCV图片转为Streamlit可展示的字节流"""
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    _, img_encoded = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return img_encoded.tobytes()


# 3. 环境数据全量分析（适配山西气候特点）
def full_environment_analysis(env_csv_path, save_plot_path=None, disease_data=None):
    """环境数据：统计+可视化+相关性分析"""
    # 读取数据
    if not os.path.exists(env_csv_path):
        raise FileNotFoundError("环境数据文件不存在，请先运行'环境数据查看'生成")
    env_data = pd.read_csv(env_csv_path)
    env_data["时间"] = pd.to_datetime(env_data["时间"])

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
    ax3.bar(env_data["时间"], env_data["降水量(mm)"], color="#27ae60", alpha=0.8, label="降水量", edgecolor="#219653",
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


# 4. 养护咨询API调用（强化山西古建筑特性）
def get_consult_suggestion(disease_type, disease_count, api_type="tongyi", api_key="", env_stats=None):
    """调用大模型生成山西古建筑专属养护建议"""
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
        try:
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["output"]["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            raise TimeoutError("API请求超时，请检查网络或稍后重试")
        except Exception as e:
            raise Exception(f"API调用失败：{str(e)}")
    else:
        raise NotImplementedError("暂仅支持通义千问API")


# 5. 读取本地背景图并编码
def get_local_bg_image(image_path):
    """读取本地背景图并转换为base64编码"""
    # 先检查文件是否存在
    if not os.path.exists(image_path):
        st.error(f"❌ 背景图文件不存在！检查路径：{image_path}")
        return ""  # 返回空字符串，避免程序崩溃
    # 读取并编码图片
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()
    return img_base64


# ---------------------- 页面基础设置（美化核心） ----------------------
# 自定义页面样式（全面美化：配色、动效、布局、组件样式）
st.markdown("""
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
.main .block-container {{
    background-color: rgba(255, 255, 255, 0.92);
    border-radius: 16px;
    padding: 2.5rem;
    margin-top: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.8);
}}

/* 侧边栏美化 */
.stSidebar {{
    background-color: rgba(240, 242, 246, 0.95);
    border-radius: 12px;
    margin: 1rem;
    padding: 1.5rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
}}

/* 侧边栏标题 */
.stSidebar h2, .stSidebar h3 {{
    color: #2c3e50;
    font-weight: 700;
    margin-bottom: 1rem;
    border-bottom: 2px solid #e74c3c;
    padding-bottom: 0.5rem;
}}

/* 按钮美化 */
.stButton>button {{
    background-color: #e74c3c;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(231, 76, 60, 0.15);
}}
.stButton>button:hover {{
    background-color: #c0392b;
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(231, 76, 60, 0.2);
}}
.stButton>button:active {{
    transform: translateY(0);
}}

/* 主要按钮（primary） */
.stButton>button[type="primary"] {{
    background-color: #2980b9;
    box-shadow: 0 4px 6px rgba(41, 128, 185, 0.15);
}}
.stButton>button[type="primary"]:hover {{
    background-color: #1f618d;
    box-shadow: 0 6px 12px rgba(41, 128, 185, 0.2);
}}

/* 滑块美化 */
.stSlider {{
    padding: 1rem 0;
}}
.stSlider [data-baseweb="slider"] {{
    margin: 0;
}}
.stSlider [data-baseweb="slider"] .thumb {{
    background-color: #e74c3c;
    border: 2px solid white;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}}
.stSlider [data-baseweb="slider"] .track-1 {{
    background-color: #e74c3c;
}}
.stSlider [data-baseweb="slider"] .track-2 {{
    background-color: #e0e0e0;
}}

/* 输入框美化 */
.stTextInput>div>div>input, .stSelectbox>div>div>select {{
    border-radius: 8px;
    border: 1px solid #ddd;
    padding: 0.6rem 1rem;
    font-size: 14px;
    transition: border 0.3s ease;
}}
.stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus {{
    border-color: #2980b9;
    outline: none;
    box-shadow: 0 0 0 2px rgba(41, 128, 185, 0.2);
}}

/* 文件上传组件 */
.stFileUploader {{
    padding: 1.5rem;
    border: 2px dashed #ddd;
    border-radius: 12px;
    background-color: rgba(255, 255, 255, 0.8);
    transition: all 0.3s ease;
}}
.stFileUploader:hover {{
    border-color: #2980b9;
    background-color: rgba(245, 250, 255, 0.9);
}}

/* 下载按钮 */
.stDownloadButton>button {{
    background-color: #27ae60;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: all 0.3s ease;
}}
.stDownloadButton>button:hover {{
    background-color: #219653;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(39, 174, 96, 0.2);
}}

/* 卡片/容器样式 */
.dataframe, .stDataFrame {{
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}}

/* 文字样式优化 */
h1 {{
    color: #2c3e50 !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
    margin-bottom: 0.5rem !important;
}}
h2 {{
    color: #34495e !important;
    font-weight: 700 !important;
    margin: 1.5rem 0 1rem 0 !important;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #f1c40f;
}}
h3 {{
    color: #34495e !important;
    font-weight: 600 !important;
    margin: 1rem 0 0.8rem 0 !important;
}}
h4 {{
    color: #7f8c8d !important;
    font-weight: 500 !important;
}}
p, div, span {{
    color: #34495e !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}}

/* 视频样式 */
.custom-video {{
    width: 100%;
    height: auto;
    border-radius: 12px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease;
}}
.custom-video:hover {{
    transform: scale(1.02);
}}

/* 加载动画美化 */
.stSpinner > div > div {{
    border-top-color: #e74c3c !important;
    width: 2rem !important;
    height: 2rem !important;
}}

/* 提示框样式 */
.stSuccess, .stInfo, .stWarning, .stError {{
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    border-left: 4px solid;
}}
.stSuccess {{
    background-color: rgba(46, 204, 113, 0.1);
    border-left-color: #27ae60;
}}
.stInfo {{
    background-color: rgba(52, 152, 219, 0.1);
    border-left-color: #2980b9;
}}
.stWarning {{
    background-color: rgba(241, 196, 15, 0.1);
    border-left-color: #f39c12;
}}
.stError {{
    background-color: rgba(231, 76, 60, 0.1);
    border-left-color: #e74c3c;
}}

/* 分割线 */
hr {{
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #ddd, transparent);
    margin: 2rem 0;
}}

/* 响应式适配 */
@media (max-width: 768px) {{
    .main .block-container {{
        padding: 1.5rem;
        margin-top: 1rem;
    }}
    .stSidebar {{
        margin: 0.5rem;
        padding: 1rem !important;
    }}
    h1 {{
        font-size: 28px !important;
    }}
    h2 {{
        font-size: 20px !important;
    }}
}}
</style>
""", unsafe_allow_html=True)

# 标题+本地视频分栏（美化布局）
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
    # 使用在线古建筑图片替代本地视频
    st.markdown("""
    <div style="border-radius: 12px; overflow: hidden; box-shadow: 0 6px 20px rgba(0,0,0,0.1); background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; text-align: center;">
        <div style="font-size: 60px; margin-bottom: 10px;">🏯</div>
        <div style="font-size: 14px; color: #666; font-weight: 600;">山西古建筑</div>
        <div style="font-size: 12px; color: #999; margin-top: 5px;">木构/砖石病害检测</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------- 侧边栏功能选择（美化） ----------------------
with st.sidebar:
    # 侧边栏头部装饰
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #eee;">
        <span style="font-size: 20px; font-weight: 700; color: #e74c3c;">📋 功能菜单</span>
    </div>
    """, unsafe_allow_html=True)

    # 美化后的单选框
    function_choice = st.radio(
        "",
        ["病害检测", "环境数据查看", "养护咨询"],
        index=0,
        help="选择需要使用的功能模块"
    )

    st.divider()

    # 使用指南美化
    st.markdown("""
    <div style="background-color: rgba(52, 152, 219, 0.08); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <p style="font-weight: 600; color: #2980b9; margin-bottom: 0.5rem;">📝 使用指南：</p>
        <ol style="padding-left: 1.5rem; margin: 0; line-height: 1.8;">
            <li>病害检测：上传图片→调整阈值→识别病害</li>
            <li>环境数据：查看温湿度/降水趋势+相关性分析</li>
            <li>养护咨询：结合病害+环境生成专属建议</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # 模型配置美化
    st.markdown("""
    <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee;">
        <span style="font-size: 18px; font-weight: 700; color: #2980b9;">⚙️ 模型配置</span>
    </div>
    """, unsafe_allow_html=True)

    model_rel_path = st.text_input(
        "",
        value="runs/detect/train_combined/weights/best.pt",
        help="修改为你的模型实际路径",
        placeholder="输入模型文件路径..."
    )

    # 模型路径提示美化
    model_abs_path = os.path.abspath(model_rel_path)
    st.markdown(f"""
    <div style="background-color: rgba(241, 196, 15, 0.08); padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0; font-size: 12px;">
        <span style="color: #f39c12; font-weight: 600;">📌 模型绝对路径：</span><br>
        <code style="font-size: 11px; color: #7f8c8d;">{model_abs_path}</code>
    </div>
    """, unsafe_allow_html=True)

    if not os.path.exists(model_abs_path):
        st.error("❌ 模型文件不存在！请检查路径")


# ---------------------- 加载自定义YOLO模型（核心修复：路径问题） ----------------------
@st.cache_resource
def load_custom_yolo_model(model_path):
    """加载训练好的山西古建筑病害检测模型"""
    # 转换为路径对象，适配不同系统
    model_path_obj = pathlib.Path(model_path).resolve()

    # 检查模型文件是否存在
    if not model_path_obj.exists():
        st.error(f"❌ 自定义模型文件不存在！路径：{model_path_obj}")
        st.stop()

    # 加载模型
    try:
        model = YOLO(str(model_path_obj))
        st.success(f"✅ 山西古建筑病害模型加载成功！")
        st.info(f"📌 模型识别类别：{model.names}")
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        st.error("💡 建议：检查模型文件是否完整，或重新训练模型")
        st.stop()


# 初始化模型（使用侧边栏配置的路径）
if "model" not in st.session_state:
    with st.spinner("🔧 加载山西古建筑病害检测模型中..."):
        st.session_state["model"] = load_custom_yolo_model(model_abs_path)
model = st.session_state["model"]

# ---------------------- 功能1：病害检测（美化+交互优化） ----------------------
if function_choice == "病害检测":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fdf2e9 0%, #fef5e7 100%); padding: 1rem; border-radius: 12px; margin-bottom: 2rem;">
        <h2 style="margin: 0 !important; padding: 0 !important; border: none !important; color: #e74c3c !important;">📷 山西古建筑病害检测（裂缝/剥落 | 自定义YOLO模型）</h2>
    </div>
    """, unsafe_allow_html=True)

    # 图片上传（美化）
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=False,
        help="建议：上传清晰的建筑表面图片（木构/砖石），分辨率≥640x640",
        key="file_uploader_disease",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            # 1. 图片预处理（调用工具函数）
            with st.spinner("🔧 图片预处理中（适配YOLO输入）..."):
                img_processed, scale, offset, img_original = preprocess_for_yolo(uploaded_file)

                # 展示原始+预处理图片对比（美化布局）
                st.markdown("<h3>🖼️ 图片预处理对比</h3>", unsafe_allow_html=True)
                col1, col2 = st.columns(2, gap="medium")
                with col1:
                    st.markdown("<p style='text-align: center; font-weight: 600; margin-bottom: 0.5rem;'>原始图片</p>",
                                unsafe_allow_html=True)
                    st.image(image_to_bytes(img_original), width='stretch', use_container_width=True)
                with col2:
                    st.markdown(
                        "<p style='text-align: center; font-weight: 600; margin-bottom: 0.5rem;'>800x800标准化图片（YOLO输入）</p>",
                        unsafe_allow_html=True)
                    st.image(image_to_bytes(img_processed), width='stretch', use_container_width=True)

            # 阈值调整（美化布局）
            st.markdown("<h3>🔧 检测阈值设置</h3>", unsafe_allow_html=True)
            col_conf, col_iou = st.columns(2, gap="medium")
            with col_conf:
                st.markdown(
                    "<p style='color: #e74c3c; font-weight: 600; margin-bottom: 0.5rem;'>置信度阈值（越高越严格）</p>",
                    unsafe_allow_html=True)
                conf_threshold = st.slider(
                    "",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    help="过滤低置信度的检测结果：\n- 调高→漏检增多、误检减少\n- 调低→漏检减少、误检增多\n建议范围：0.2-0.5",
                    label_visibility="collapsed"
                )
            with col_iou:
                st.markdown(
                    "<p style='color: #2980b9; font-weight: 600; margin-bottom: 0.5rem;'>IOU阈值（越高重叠框越少）</p>",
                    unsafe_allow_html=True)
                iou_threshold = st.slider(
                    "",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.4,
                    step=0.05,
                    help="过滤重叠的检测框：\n- 调高→保留的框越少\n- 调低→保留的框越多\n建议范围：0.3-0.6",
                    label_visibility="collapsed"
                )

            # 2. 自定义模型检测
            with st.spinner("🔍 使用YOLO模型检测山西古建筑病害中..."):
                results = model(
                    img_processed,
                    conf=conf_threshold,
                    iou_threshold=iou_threshold
                )
                # 生成标注后的图片
                detected_img = results[0].plot(line_width=3, font_size=14)  # 加大标注线宽和字体
                detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

            # 3. 展示检测结果
            st.markdown("<h3>🎯 病害检测结果（山西古建筑专属标注）</h3>", unsafe_allow_html=True)
            st.image(image_to_bytes(detected_img_rgb), caption="裂缝/剥落病害标注结果", width='stretch',
                     use_container_width=True)

            # 4. 提取病害信息
            detections = results[0].boxes.data
            disease_count = len(detections)
            disease_types = []
            disease_details = []

            for idx, box in enumerate(detections):
                x1, y1, x2, y2, conf, cls_idx = box
                cls_idx = int(cls_idx)
                cls_name = model.names[cls_idx]
                conf = round(float(conf), 3)

                # 转换坐标到原始图片
                x1_ori = (x1 - offset[0]) / scale
                y1_ori = (y1 - offset[1]) / scale
                x2_ori = (x2 - offset[0]) / scale
                y2_ori = (y2 - offset[1]) / scale

                disease_types.append(cls_name)
                disease_details.append({
                    "病害编号": idx + 1,
                    "病害类型": cls_name,
                    "置信度": conf,
                    "原始图片坐标": f"({int(x1_ori)},{int(y1_ori)})-({int(x2_ori)},{int(y2_ori)})"
                })

            # 5. 统计展示（美化）
            st.success(
                f"✅ 检测完成！共识别到 <span style='font-weight: 700; color: #e74c3c;'>{disease_count}</span> 处山西古建筑病害",
                unsafe_allow_html=True)

            if disease_details:
                st.markdown("<h3>📊 病害详细统计</h3>", unsafe_allow_html=True)
                disease_df = pd.DataFrame(disease_details)
                st.dataframe(disease_df, width='stretch', use_container_width=True)

                # 病害类型汇总（美化图表）
                st.markdown("<h3>📈 病害类型汇总</h3>", unsafe_allow_html=True)
                disease_summary = pd.Series(disease_types).value_counts().reset_index()
                disease_summary.columns = ["病害类型", "数量"]
                st.bar_chart(
                    disease_summary.set_index("病害类型"),
                    width='stretch',
                    use_container_width=True,
                    color=["#e74c3c", "#2980b9", "#27ae60"]
                )

                # 保存检测结果
                save_path = f"detection_results/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                os.makedirs(save_path, exist_ok=True)
                # 保存标注图片
                cv2.imwrite(f"{save_path}/detected.jpg", cv2.cvtColor(detected_img_rgb, cv2.COLOR_RGB2BGR))
                # 保存统计结果
                disease_df.to_csv(f"{save_path}/disease_stats.csv", index=False, encoding="utf-8")
                st.info(f"💾 检测结果已保存至：<code>{os.path.abspath(save_path)}</code>", unsafe_allow_html=True)

                # 下载按钮（美化布局）
                st.markdown("<h3>📥 结果下载</h3>", unsafe_allow_html=True)
                col1, col2 = st.columns(2, gap="medium")
                with col1:
                    st.download_button(
                        label="下载标注图片",
                        data=image_to_bytes(detected_img_rgb),
                        file_name=f"山西古建筑病害检测_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
                with col2:
                    st.download_button(
                        label="下载病害统计（CSV）",
                        data=disease_df.to_csv(index=False, encoding="utf-8"),
                        file_name=f"山西古建筑病害统计_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("ℹ️ 未检测到裂缝/剥落病害，建议降低置信度阈值后重试")

        except Exception as e:
            st.error(f"❌ 检测失败：{str(e)}")
            st.info("💡 建议：检查图片格式，或重新上传清晰的山西古建筑图片")

# ---------------------- 功能2：环境数据查看（美化） ----------------------
elif function_choice == "环境数据查看":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ebf5fb 0%, #e8f4f8 100%); padding: 1rem; border-radius: 12px; margin-bottom: 2rem;">
        <h2 style="margin: 0 !important; padding: 0 !important; border: none !important; color: #2980b9 !important;">🌡️ 山西古建筑周边环境数据（温湿度/降水 | 全量分析）</h2>
    </div>
    """, unsafe_allow_html=True)

    # 生成/加载环境数据
    env_data_path = "data/environment_data.csv"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(env_data_path):
        with st.spinner("📊 首次运行，生成山西晋中地区模拟环境数据..."):
            dates = pd.date_range(start="2026-01-01", end="2026-01-10", freq="H")
            # 模拟山西冬季气候
            env_data = pd.DataFrame({
                "时间": dates,
                "温度(℃)": np.random.uniform(-5, 10, len(dates)),
                "湿度(%)": np.random.uniform(20, 50, len(dates)),
                "降水量(mm)": np.random.choice([0, 0.1, 0.3], len(dates), p=[0.95, 0.04, 0.01])
            })
            env_data.to_csv(env_data_path, index=False, encoding="utf-8")
        st.success("✅ 山西地区环境模拟数据生成完成")

    # 调用全量分析函数
    try:
        analysis_result = full_environment_analysis(
            env_csv_path=env_data_path,
            save_plot_path="data/plots/shanxi_env_trend.png"
        )
        # 展示结果（美化）
        st.markdown("<h3>📊 山西古建筑环境数据概览</h3>", unsafe_allow_html=True)
        # 美化数据展示
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

        st.markdown("<h3>📈 山西古建筑环境趋势图</h3>", unsafe_allow_html=True)
        st.pyplot(analysis_result["趋势图"], use_container_width=True)

        st.markdown("<h3>🔗 环境相关性分析（适配古建筑病害）</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background-color: rgba(46, 204, 113, 0.08); padding: 1.2rem; border-radius: 8px; border-left: 4px solid #27ae60;">
            {analysis_result["相关性分析"]}
        </div>
        """, unsafe_allow_html=True)

        # 展示保存的图片
        if os.path.exists("data/plots/shanxi_env_trend.png"):
            st.markdown("<h3>💾 保存的环境趋势图</h3>", unsafe_allow_html=True)
            st.image("data/plots/shanxi_env_trend.png", width='stretch', use_container_width=True)

            # 下载趋势图
            with open("data/plots/shanxi_env_trend.png", "rb") as f:
                st.download_button(
                    label="📥 下载环境趋势图",
                    data=f,
                    file_name=f"山西古建筑环境趋势_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
    except Exception as e:
        st.error(f"❌ 环境分析失败：{str(e)}")

# ---------------------- 功能3：养护咨询（美化） ----------------------
elif function_choice == "养护咨询":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f2f9e9 0%, #eafaf1 100%); padding: 1rem; border-radius: 12px; margin-bottom: 2rem;">
        <h2 style="margin: 0 !important; padding: 0 !important; border: none !important; color: #27ae60 !important;">💡 山西古建筑智能养护咨询（结合病害+环境）</h2>
    </div>
    """, unsafe_allow_html=True)

    # 1. 基础输入（美化布局）
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("<p style='font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;'>📌 病害信息</p>",
                    unsafe_allow_html=True)
        disease_type = st.selectbox(
            "",
            ["裂缝", "剥落", "裂缝+剥落"],
            help="选择检测到的山西古建筑主要病害类型",
            label_visibility="collapsed"
        )
        disease_count = st.text_input(
            "",
            value="裂缝3处，剥落2处",
            help="填写检测到的具体病害数量",
            placeholder="例：裂缝5处，剥落3处",
            label_visibility="collapsed"
        )
        st.markdown(
            "<p style='font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem; margin-top: 1.5rem;'>🔑 API配置</p>",
            unsafe_allow_html=True)
        api_key = st.text_input(
            "",
            type="password",
            help="前往https://dashscope.aliyun.com/获取免费API Key",
            placeholder="输入通义千问API Key...",
            label_visibility="collapsed"
        )
    with col2:
        st.markdown("<p style='font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;'>🌡️ 环境数据</p>",
                    unsafe_allow_html=True)
        # 加载环境数据统计
        env_csv_path = "data/environment_data.csv"
        env_stats = None
        if os.path.exists(env_csv_path):
            try:
                env_result = full_environment_analysis(env_csv_path)
                env_stats = env_result["数据概览"]
                st.success("✅ 已加载山西古建筑周边环境数据统计")
                # 美化环境数据展示
                env_html = "<div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; font-size: 12px;'>"
                for k, v in env_stats.items():
                    env_html += f"<div style='padding: 0.3rem 0;'><strong>{k}：</strong>{v}</div>"
                env_html += "</div>"
                st.markdown(env_html, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"⚠️ 环境数据加载失败：{str(e)}，将按山西默认气候生成建议")
        else:
            st.warning("⚠️ 未找到环境数据，将按山西晋中地区气候生成建议")

    # 2. 生成建议按钮（美化）
    st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
    if st.button("🚀 生成山西古建筑专属养护建议", type="primary", use_container_width=True):
        if not api_key:
            st.warning("⚠️ 请输入通义千问API Key（免费获取，支持多次调用）")
        else:
            with st.spinner("🤖 AI生成山西古建筑养护建议中..."):
                try:
                    # 调用咨询函数
                    suggestion = get_consult_suggestion(
                        disease_type=disease_type,
                        disease_count=disease_count,
                        api_type="tongyi",
                        api_key=api_key,
                        env_stats=env_stats
                    )
                    # 展示建议（美化）
                    st.markdown("<h3>📋 山西古建筑定制化养护建议</h3>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f7f9fb 0%, #f5f7fa 100%); 
                                padding: 1.8rem; border-radius: 12px; 
                                border-left: 5px solid #27ae60;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                        <div style="line-height: 2.0; font-size: 15px;">{suggestion}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 导出建议
                    st.download_button(
                        label="📥 导出养护建议（TXT）",
                        data=suggestion,
                        file_name=f"山西古建筑养护建议_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="download_suggestion"
                    )
                except TimeoutError:
                    st.error("❌ API请求超时，请检查网络或稍后重试")
                except Exception as e:
                    st.error(f"❌ 生成建议失败：{str(e)}")
                    st.info("💡 检查API Key是否正确，或网络是否通畅")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- 结尾提示（美化） ----------------------
st.divider()
st.markdown(f"""
<div style="text-align: center; color: #7f8c8d; font-size: 13px; padding: 1rem 0;">
    💡 当前使用模型：<code style="color: #95a5a6;">{model_abs_path}</code> | 
    📅 更新时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
    🏯 适配场景：山西木构/砖石古建筑（应县木塔、晋祠、平遥古城等）
</div>
""", unsafe_allow_html=True)