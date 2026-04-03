import streamlit as st
from ultralytics import YOLO
import pathlib
import os

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