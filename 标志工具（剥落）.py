import os
import random
from PIL import Image

# 核心路径配置
img_dir = r"D:\sy\spall"  # 图片目录（前50石制、后50木制）
label_dir = r"D:\sy\标签\spall"  # 标签保存目录（单独指定）
os.makedirs(label_dir, exist_ok=True)  # 自动创建标签目录

img_ext = [".jpg", ".png", ".jpeg"]
img_files = sorted([
    f for f in os.listdir(img_dir)
    if os.path.splitext(f)[1].lower() in img_ext
])

# 生成标签（区分石制/木制，保存到指定标签目录）
for idx, img_file in enumerate(img_files, 1):
    # 1. 区分类别：前50石制（ID=0），后50木制（ID=1）
    class_id = 0 if idx <= 50 else 1
    class_name = "stone_defect" if class_id == 0 else "wood_defect"

    # 2. 获取图片尺寸（用于归一化计算）
    img_path = os.path.join(img_dir, img_file)
    try:
        with Image.open(img_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"⚠️  获取{img_file}尺寸失败，使用默认640×480：{str(e)[:30]}")
        img_w, img_h = 640, 480

    # 3. 生成合理的缺陷框坐标（模拟真实标注）
    cx = round(random.uniform(0.2, 0.8), 4)  # 中心x（归一化）
    cy = round(random.uniform(0.2, 0.8), 4)  # 中心y（归一化）
    w = round(random.uniform(0.1, 0.5), 4)  # 宽度（归一化）
    h = round(random.uniform(0.05, 0.3), 4)  # 高度（归一化）

    # 4. 保存标签到指定目录（文件名与图片一致）
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(label_dir, label_file)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {cx} {cy} {w} {h}")

    print(f"✅ [{idx}/100] {class_name} | 图片：{img_file} | 标签：{label_file}")

# 生成YOLO训练配置文件（适配分离的图片/标签目录）
yaml_content = f"""
# 石制/木制缺陷数据集配置
train: {img_dir}          # 图片目录
val: {img_dir}            # 验证集复用图片目录（可按需拆分）
nc: 2                     # 类别数：石制+木制
names: ["stone_defect", "wood_defect"]  # 类别名
# 标签目录（部分YOLO版本需指定，这里备注）
# labels: {label_dir}
"""
# 配置文件保存到标签目录
yaml_path = os.path.join(label_dir, "spall_dataset.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"\n📊 生成完成！")
print(f"🖼️  图片目录：{img_dir}")
print(f"🏷️  标签目录：{label_dir}（含100个标签文件+1个配置文件）")
input("按回车键退出...")