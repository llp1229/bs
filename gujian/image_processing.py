import cv2
import numpy as np
import streamlit as st

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