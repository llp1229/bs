import cv2
import numpy as np
from PIL import Image
import io
import os


def load_image(image_source):
    """
    加载图片（支持本地文件路径/Streamlit上传文件/字节流）
    :param image_source: 图片路径(str) / Streamlit上传的文件对象 / 字节流
    :return: OpenCV格式图片(numpy.ndarray)，RGB通道
    """
    try:
        # 处理Streamlit上传的文件
        if hasattr(image_source, 'read'):
            image_bytes = image_source.read()
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        # 处理本地文件路径
        elif isinstance(image_source, str) and os.path.exists(image_source):
            img = cv2.imread(image_source)
        # 处理字节流
        else:
            img = cv2.imdecode(np.frombuffer(image_source, np.uint8), cv2.IMREAD_COLOR)

        # 转换为RGB通道（OpenCV默认BGR）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

    except Exception as e:
        raise ValueError(f"图片加载失败：{str(e)}")


def resize_image(img, target_size=(640, 640), keep_ratio=True):
    """
    调整图片尺寸（适配YOLO模型输入，默认640×640）
    :param img: OpenCV格式图片(RGB)
    :param target_size: 目标尺寸 (width, height)
    :param keep_ratio: 是否保持宽高比（避免拉伸变形）
    :return: 调整后的图片，缩放比例
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size

    if keep_ratio:
        # 计算缩放比例（取最小比例，避免超出目标尺寸）
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 缩放图片
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建空白画布，填充黑色背景
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        # 计算居中填充位置
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

        return canvas, scale, (x_offset, y_offset)
    else:
        # 直接拉伸（不推荐，易变形）
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        return img_resized, 1.0, (0, 0)


def remove_noise(img, method='gaussian'):
    """
    去除图片噪声（提升病害识别精度）
    :param img: OpenCV格式图片(RGB)
    :param method: 去噪方法 - gaussian(高斯模糊)/median(中值滤波)/bilateral(双边滤波)
    :return: 去噪后的图片
    """
    if method == 'gaussian':
        # 高斯模糊：适合去除高斯噪声，保留边缘
        img_denoised = cv2.GaussianBlur(img, (3, 3), 0)
    elif method == 'median':
        # 中值滤波：适合去除椒盐噪声（如图片中的白点/黑点）
        img_denoised = cv2.medianBlur(img, 3)
    elif method == 'bilateral':
        # 双边滤波：去噪同时保留更多细节（速度较慢）
        img_denoised = cv2.bilateralFilter(img, 9, 75, 75)
    else:
        raise ValueError(f"不支持的去噪方法：{method}")

    return img_denoised


def enhance_contrast(img, alpha=1.2, beta=10):
    """
    增强图片对比度（突出病害纹理，如裂缝/剥落）
    :param img: OpenCV格式图片(RGB)
    :param alpha: 对比度系数（1.0为原图，>1增强）
    :param beta: 亮度调整（0为原图）
    :return: 增强后的图片
    """
    # 对比度增强公式：dst = alpha * img + beta
    img_enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img_enhanced


def image_to_bytes(img, format='PNG'):
    """
    将OpenCV图片转换为字节流（用于Streamlit展示/保存）
    :param img: OpenCV格式图片(RGB)
    :param format: 输出格式 - PNG/JPG
    :return: 字节流
    """
    # 转换为BGR（OpenCV编码需要）
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 编码为字节流
    is_success, buffer = cv2.imencode(f'.{format.lower()}', img_bgr)
    if not is_success:
        raise RuntimeError("图片编码失败")
    byte_data = io.BytesIO(buffer)
    return byte_data


def save_processed_image(img, save_path):
    """
    保存处理后的图片到本地
    :param img: OpenCV格式图片(RGB)
    :param save_path: 保存路径（如 'data/uploads/processed_1.jpg'）
    """
    try:
        # 创建保存目录（若不存在）
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 转换为BGR并保存
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)
    except Exception as e:
        raise ValueError(f"图片保存失败：{str(e)}")


# 示例：一站式预处理函数（适用于YOLO模型输入）
def preprocess_for_yolo(image_source, target_size=(640, 640)):
    """
    一站式图片预处理（加载→去噪→增强对比度→调整尺寸）
    :param image_source: 图片路径/Streamlit上传文件/字节流
    :param target_size: YOLO模型输入尺寸
    :return: 预处理后的图片，缩放比例，偏移量
    """
    # 1. 加载图片
    img = load_image(image_source)
    # 2. 去噪
    img_denoised = remove_noise(img, method='gaussian')
    # 3. 增强对比度
    img_enhanced = enhance_contrast(img_denoised)
    # 4. 调整尺寸
    img_resized, scale, offset = resize_image(img_enhanced, target_size)

    return img_resized, scale, offset