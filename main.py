import sys

sys.path.append("D:/bs/sxgjz")
import streamlit as st
import os
import warnings

warnings.filterwarnings('ignore')

# ====================== 页面配置 ======================
st.set_page_config(
    page_title="山西古建筑健康监测系统",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================== 顶部标题+视频 ======================
col1, col2 = st.columns([4, 1.5])
with col1:
    st.markdown("""
    <h1 style="font-size:40px; font-weight:bold; color:#d4af37; margin:0;">
        🏛️ 山西古建筑健康监测与养护咨询系统
    </h1>
    """, unsafe_allow_html=True)
with col2:
    video_path = "zy/古建筑.mp4"
    if os.path.exists(video_path):
        from gujian.style_setup import get_local_file_base64

        video_base64 = get_local_file_base64(video_path)
        st.markdown(f"""
        <video autoplay muted loop playsinline style="width:180px; border-radius:10px; float:right;">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        """, unsafe_allow_html=True)

st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

# ====================== 顶部导航栏 ======================
tab1, tab2, tab3 = st.tabs(["🏛️ 病害检测", "🌍 环境数据监测", "💡 养护咨询"])

# ====================== 导入功能模块 ======================
from gujian.image_processing import preprocess_for_yolo
from gujian.environment_analysis import show_multi_station_environment
from model_loader import load_custom_yolo_model


# ====================== 模型加载 ======================
@st.cache_resource
def load_model():
    try:
        model = load_custom_yolo_model("model/best.pt")
        st.success("✅ 山西古建筑病害模型加载成功！")
        st.info(f"📌 模型识别类别：{model.names} | 模型路径：{os.path.abspath('model/best.pt')}")
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        return None


model = load_model()

# ====================== 1. 病害检测页面 ======================
with tab1:
    st.subheader("🔍 古建筑病害检测")
    uploaded_file = st.file_uploader("📸 上传古建筑图片", type=["jpg", "jpeg", "png"])
    conf = st.slider("🎯 置信度阈值", 0.1, 0.9, 0.35)
    iou = st.slider("📐 IOU 阈值", 0.1, 0.9, 0.45)

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if uploaded_file:
            st.image(uploaded_file, caption="原始图片", use_column_width=True)

    if st.button("🚀 开始检测", type="primary", use_container_width=True):
        if not uploaded_file:
            st.warning("⚠️ 请先上传古建筑图片")
        elif not model:
            st.error("❌ 病害检测模型未加载成功")
        else:
            with st.spinner("🔍 正在检测病害特征..."):
                img, img_arr = preprocess_for_yolo(uploaded_file)
                results = model(img_arr, conf=conf, iou=iou)
                with col_img2:
                    st.image(results[0].plot(), caption="病害检测标注图", use_column_width=True)
                st.success("✅ 病害检测完成！")

# ====================== 2. 环境数据监测页面 ======================
with tab2:
    try:
        show_multi_station_environment()
    except Exception as e:
        st.subheader("🌍 山西多站点环境监测")
        st.info(f"环境数据页面加载异常：{str(e)}")

# ====================== 养护咨询页面（最终修复版，适配无欠费账号） ======================
with tab3:
    st.subheader("💡 古建筑养护咨询")

    # 表单控件
    disease_type = st.selectbox("🔍 病害类型", ["裂缝", "剥落", "风化", "渗漏", "虫蛀", "霉变"], key="disease_type")
    disease_count = st.number_input("🔢 病害数量", min_value=1, value=6, step=1, key="disease_count")
    api_key = st.text_input("🔑 通义千问API Key（sk-开头）", type="password", placeholder="sk-...", key="api_key")

    # 演示模式兜底（无API/额度问题时，直接展示）
    demo_html = f"""
    <div style="background:#f8f9fa; padding:20px; border-radius:12px; border-left:5px solid #d4af37; margin:15px 0;">
        <h3 style="color:#8B0000; margin-top:0;">🏛️ 山西古建筑（{disease_type}）养护建议（演示版）</h3>
        <p><strong>📌 病害情况</strong>：检测到{disease_count}处{disease_type}病害，符合山西古建筑砖木结构常见病害特征。</p>
        <h4>🔧 专业养护方案</h4>
        <ol>
            <li><strong>表面清理</strong>：采用软毛刷配合低压气流清除浮尘，严禁使用高压水枪，避免砖体酥松、木构件变形。</li>
            <li><strong>裂缝修复</strong>：使用传统材料（如传统灰浆、麻刀灰）进行嵌缝修补，严格遵循「不改变文物原状」原则。</li>
            <li><strong>结构加固</strong>：对裂缝周边构件进行隐形加固，增强结构稳定性，不破坏建筑外观。</li>
            <li><strong>环境管控</strong>：维持站点湿度50%-60%，加强通风排湿，避免温湿度剧烈波动导致裂缝扩展。</li>
        </ol>
        <p><strong>⏱️ 监测建议</strong>：每季度开展一次裂缝宽度复查，建立长期养护档案，跟踪病害发展趋势。</p>
    </div>
    """

    # 按钮点击逻辑
    if st.button("💡 生成养护建议", type="primary", use_container_width=True, key="generate_consult"):
        # 校验API Key格式
        if not api_key or not api_key.startswith("sk-"):
            st.warning("🔑 未检测到有效API Key，启用【演示模式】生成养护建议")
            st.markdown(demo_html, unsafe_allow_html=True)
            st.success("✅ 演示模式：养护建议生成完成！")
            st.stop()

        # 加载中状态
        with st.spinner("🤖 正在调用通义千问生成养护建议..."):
            try:
                # 导入阿里云官方SDK
                from dashscope import Generation

                # 核心修复：使用qwen-turbo（免费额度充足，参数无冲突）
                response = Generation.call(
                    model="qwen-turbo",
                    prompt=f"""
                    你是资深古建筑保护专家，精通《文物保护法》和山西古建筑修缮工艺。
                    针对山西古建筑的{disease_type}病害（病害数量{disease_count}处）， 
                    生成一份专业、可执行的养护建议，包含：
                    1. 病害成因（结合山西干旱、多风沙的地域气候特点）
                    2. 应急处理方案
                    3. 长期养护与加固方案
                    4. 环境监测要求
                    5. 定期巡检计划
                    要求：语言专业严谨，分点清晰，符合文物保护工程规范。
                    """,
                    api_key=api_key
                )

                # 校验API响应
                if response.status_code == 200:
                    st.markdown("---")
                    st.markdown("### 📋 山西古建筑专属养护建议")
                    st.markdown(response.output.text)
                    st.success("✅ 养护建议生成完成！")
                else:
                    # 额度/参数问题，自动切演示模式
                    st.error(f"❌ API调用失败：{response.message}，已自动切换演示模式")
                    st.markdown(demo_html, unsafe_allow_html=True)

            # 捕获所有异常
            except Exception as e:
                st.error(f"❌ 调用异常：{str(e)}，已自动切换演示模式")
                st.markdown(demo_html, unsafe_allow_html=True)