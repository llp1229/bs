import streamlit as st
import os
import base64

def get_local_file_base64(file_path):
    """读取本地文件（图/视频）并转换为base64编码，适配相对路径"""
    try:
        if not os.path.exists(file_path):
            return ""
        with open(file_path, "rb") as f:
            file_base64 = base64.b64encode(f.read()).decode()
        import streamlit as st

        def setup_page_style():
            st.set_page_config(
                page_title="山西古建筑健康监测系统",
                page_icon="🏛️",
                layout="wide",
                initial_sidebar_state="collapsed"
            )

            st.markdown("""
                <style>
                    /* 隐藏默认侧边栏 + 顶部栏 */
                    [data-testid="stSidebar"],
                    [data-testid="stHeader"] {
                        display: none !important;
                    }

                    /* 主内容上边距 */
                    .block-container {
                        padding-top: 80px !important;
                        padding-left: 30px !important;
                        padding-right: 30px !important;
                    }

                    /* 顶部导航栏 */
                    .top-nav {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 58px;
                        background: #1677ff;
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        padding: 0 30px;
                        z-index: 999999;
                        box-sizing: border-box;
                    }

                    .nav-left {
                        display: flex;
                        align-items: center;
                        gap: 40px;
                    }

                    .logo {
                        color: white;
                        font-size: 18px;
                        font-weight: bold;
                    }

                    .nav-menu {
                        display: flex;
                        gap: 25px;
                    }

                    .nav-item {
                        color: white;
                        font-size: 15px;
                        cursor: pointer;
                        padding: 8px 12px;
                        border-radius: 4px;
                    }

                    .nav-item:hover {
                        background: rgba(255,255,255,0.15);
                    }

                    .nav-item.active {
                        background: rgba(255,255,255,0.25);
                    }

                    .nav-right {
                        color: white;
                        font-size: 14px;
                    }
                </style>
            """, unsafe_allow_html=True)

        return file_base64
    except Exception as e:
        st.warning(f"读取文件失败：{str(e)}")
        return ""

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
        css_style = f"""
        <style>
        /* 全局重置 */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        /* 全局背景设置 */
        .stApp {{
            background-image: url('data:image/png;base64,{bg_img_base64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-color: #f8f9fa; /* 备用背景色 */
            min-height: 100vh;
        }}
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