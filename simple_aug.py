import os
import random
from PIL import Image, ImageEnhance, ImageOps

# 配置
SRC_IMG_DIR = r"C:\Users\小刘\PycharmProjects\ancient_building_system\data\disease_dataset\crack\train\images"
SRC_LABEL_DIR = r"C:\Users\小刘\PycharmProjects\ancient_building_system\data\disease_dataset\crack\train\labels"
AUG_IMG_DIR = r"C:\Users\小刘\PycharmProjects\ancient_building_system\data\disease_dataset\crack\train\augmented\images"
AUG_LABEL_DIR = r"C:\Users\小刘\PycharmProjects\ancient_building_system\data\disease_dataset\crack\train\augmented\labels"
os.makedirs(AUG_IMG_DIR, exist_ok=True)
os.makedirs(AUG_LABEL_DIR, exist_ok=True)


def augment_img_label(img_path, label_path, aug_idx):
    """纯PIL实现增强（水平翻转+亮度调整）"""
    # 读取图片
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        print(f"跳过：{img_path}")
        return
    # 读取标签
    if not os.path.exists(label_path):
        print(f"标签缺失：{label_path}")
        return
    with open(label_path, "r") as f:
        labels = f.readlines()

    # 1. 水平翻转
    img_aug = ImageOps.mirror(img)
    # 翻转标签（YOLO格式）
    new_labels = []
    for line in labels:
        line = line.strip()
        if not line:
            continue
        cls, cx, cy, bw, bh = line.split()
        cx = str(1 - float(cx))  # 中心点x取反
        new_labels.append(f"{cls} {cx} {cy} {bw} {bh}")

    # 2. 随机亮度调整
    enhancer = ImageEnhance.Brightness(img_aug)
    brightness = random.uniform(0.7, 1.3)
    img_aug = enhancer.enhance(brightness)

    # 保存增强后的图片和标签
    img_name = os.path.basename(img_path).replace(".jpg", f"_aug{aug_idx}.jpg")
    label_name = os.path.basename(label_path).replace(".txt", f"_aug{aug_idx}.txt")
    img_aug.save(os.path.join(AUG_IMG_DIR, img_name), "JPEG")
    with open(os.path.join(AUG_LABEL_DIR, label_name), "w") as f:
        f.write("\n".join(new_labels))
    print(f"生成：{img_name}")


# 批量处理
for img_file in os.listdir(SRC_IMG_DIR):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(SRC_IMG_DIR, img_file)
        label_path = os.path.join(SRC_LABEL_DIR, img_file.replace(".jpg", ".txt"))
        augment_img_label(img_path, label_path, 0)