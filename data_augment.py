import os
import cv2
import random
import numpy as np
from PIL import Image

# ---------------------- 核心配置（适配你的目录结构：images/labels子目录） ----------------------
# 根数据集目录
ROOT_DATA_DIR = r"C:\Users\小刘\PycharmProjects\ancient_building_system\data\disease_dataset"
# 需要增强的类别（spall和crack）
CLASSES = ["spall", "crack"]
# 需要增强的数据集（train和val）
DATA_SPLITS = ["train", "val"]
# 每张图生成的增强图数量
AUG_NUM_PER_IMG = 2
# 增强后数据的保存子目录（会自动创建在原images/labels同级）
AUG_SUBDIR = "augmented"

# 增强参数（可根据需求调整）
ROTATION_ANGLE = [-15, 15]  # 随机旋转角度范围
BRIGHTNESS_RANGE = [0.7, 1.3]  # 随机亮度调整范围
FLIP_PROB = 0.5  # 水平翻转概率
BLUR_PROB = 0.2  # 高斯模糊概率（增加纹理鲁棒性）


# ---------------------- 数据增强工具函数 ----------------------
def augment_single_image(img_path, label_path, aug_img_dir, aug_label_dir, aug_idx):
    """对单张图片+标签进行增强并保存（兼容中文路径、跳过损坏图片）"""
    # 1. 读取图片（改用PIL，兼容中文路径）
    try:
        with Image.open(img_path) as img_pil:
            # 转换为RGB格式（避免透明通道问题）
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            img = np.array(img_pil)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换为cv2的BGR格式
    except Exception as e:
        print(f"⚠️  无法读取图片：{img_path}，错误：{str(e)}")
        return

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        print(f"⚠️  图片尺寸异常：{img_path}")
        return

    # 2. 读取标签（YOLO格式）
    if not os.path.exists(label_path):
        print(f"⚠️  标签文件不存在：{label_path}")
        return
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        if not labels:
            print(f"⚠️  标签文件为空：{label_path}")
            return
    except Exception as e:
        print(f"⚠️  读取标签失败：{label_path}，错误：{str(e)}")
        return

    # 3. 执行增强操作
    # 随机旋转
    angle = random.uniform(ROTATION_ANGLE[0], ROTATION_ANGLE[1])
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img_aug = cv2.warpAffine(img, M_rot, (w, h))

    # 随机调整亮度
    img_aug = img_aug.astype(np.float32)
    brightness = random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
    img_aug = img_aug * brightness
    img_aug = np.clip(img_aug, 0, 255).astype(np.uint8)

    # 随机水平翻转
    if random.random() < FLIP_PROB:
        img_aug = cv2.flip(img_aug, 1)
        # 同步翻转标签的x坐标
        flipped_labels = []
        for label in labels:
            try:
                cls, cx, cy, bw, bh = label.split()
                cx = str(1 - float(cx))  # 中心点x坐标取反
                flipped_labels.append(f"{cls} {cx} {cy} {bw} {bh}")
            except:
                print(f"⚠️  标签格式错误，跳过翻转：{label}")
                flipped_labels.append(label)
        labels = flipped_labels

    # 随机高斯模糊
    if random.random() < BLUR_PROB:
        ksize = (5, 5) if random.random() > 0.5 else (3, 3)
        img_aug = cv2.GaussianBlur(img_aug, ksize, 0)

    # 4. 保存增强后的图片和标签
    img_basename = os.path.basename(img_path)
    img_name_noext = os.path.splitext(img_basename)[0]
    aug_img_name = f"{img_name_noext}_aug{aug_idx}.jpg"
    aug_label_name = f"{img_name_noext}_aug{aug_idx}.txt"

    try:
        # 保存图片（强制转为jpg格式）
        cv2.imwrite(os.path.join(aug_img_dir, aug_img_name), img_aug)
        # 保存标签
        with open(os.path.join(aug_label_dir, aug_label_name), "w", encoding="utf-8") as f:
            f.write("\n".join(labels))
        print(f"✅ 生成增强数据：{os.path.join(aug_img_dir, aug_img_name)}")
    except Exception as e:
        print(f"⚠️  保存增强数据失败：{img_name_noext}，错误：{str(e)}")


# ---------------------- 批量增强主逻辑 ----------------------
if __name__ == "__main__":
    # 遍历每个类别（spall和crack）
    for cls in CLASSES:
        # 遍历每个数据集（train和val）
        for split in DATA_SPLITS:
            # 原数据目录（适配images/labels子目录）
            src_img_dir = os.path.join(ROOT_DATA_DIR, cls, split, "images")
            src_label_dir = os.path.join(ROOT_DATA_DIR, cls, split, "labels")

            # 增强后数据的保存目录（创建在split目录下的augmented）
            aug_root_dir = os.path.join(ROOT_DATA_DIR, cls, split, AUG_SUBDIR)
            aug_img_dir = os.path.join(aug_root_dir, "images")
            aug_label_dir = os.path.join(aug_root_dir, "labels")
            os.makedirs(aug_img_dir, exist_ok=True)
            os.makedirs(aug_label_dir, exist_ok=True)

            # 获取原目录下的图片文件
            img_extensions = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]
            if not os.path.exists(src_img_dir):
                print(f"⚠️  {src_img_dir} 目录不存在，跳过增强")
                continue

            img_files = [
                f for f in os.listdir(src_img_dir)
                if os.path.splitext(f)[1] in img_extensions
            ]

            if not img_files:
                print(f"⚠️  {src_img_dir} 下无图片，跳过增强")
                continue

            # 对每张图生成增强数据
            print(f"\n=== 开始增强 {cls}/{split} 目录下的 {len(img_files)} 张图片 ===")
            for img_file in img_files:
                img_path = os.path.join(src_img_dir, img_file)
                label_file = os.path.splitext(img_file)[0] + ".txt"
                label_path = os.path.join(src_label_dir, label_file)

                # 提前检查图片是否有效，跳过损坏文件
                try:
                    with Image.open(img_path) as temp_img:
                        temp_img.verify()  # 验证图片完整性
                except:
                    print(f"⚠️  跳过损坏/无效图片：{img_path}")
                    continue

                for aug_idx in range(AUG_NUM_PER_IMG):
                    augment_single_image(img_path, label_path, aug_img_dir, aug_label_dir, aug_idx)

    print("\n🎉 所有类别+数据集的增强已完成！")