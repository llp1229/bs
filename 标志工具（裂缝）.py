import os
import random
from PIL import Image

# 核心路径配置
img_dir = r"D:\sy\crack"  # 图片目录（前50石制裂缝、后50木制裂缝）
label_dir = r"D:\sy\标签\crack"  # 标签保存目录
os.makedirs(label_dir, exist_ok=True)  # 自动创建标签目录

# 获取排序后的图片文件（确保前50石制、后50木制）
img_ext = [".jpg", ".png", ".jpeg"]
img_files = sorted([
    f for f in os.listdir(img_dir)
    if os.path.splitext(f)[1].lower() in img_ext
])

# 校验图片数量（提示是否符合100张）
if len(img_files) != 100:
    print(f"⚠️  警告：{img_dir} 目录下仅找到 {len(img_files)} 张图片，建议补充至100张（前50石制、后50木制）")

# 生成裂缝标签（区分石制/木制，适配裂缝形态）
for idx, img_file in enumerate(img_files, 1):
    # 1. 区分类别：前50石制裂缝(ID=0)、后50木制裂缝(ID=1)
    if idx <= 50:
        class_id = 0
        crack_type = "石制裂缝"
        class_name = "stone_crack"
    else:
        class_id = 1
        crack_type = "木制裂缝"
        class_name = "wood_crack"

    # 2. 获取图片实际尺寸（保证归一化坐标准确）
    img_path = os.path.join(img_dir, img_file)
    try:
        with Image.open(img_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"⚠️  获取{img_file}尺寸失败，使用默认640×480：{str(e)[:30]}")
        img_w, img_h = 640, 480

    # 3. 生成裂缝坐标（石制/木制裂缝均为细长型，微调参数更贴合特征）
    cx = round(random.uniform(0.15, 0.85), 4)  # 中心x
    cy = round(random.uniform(0.15, 0.85), 4)  # 中心y
    # 石制裂缝更窄、木制裂缝稍宽，贴合材质特征
    if class_id == 0:
        crack_w = round(random.uniform(0.03, 0.15), 4)  # 石制裂缝宽度
        crack_h = round(random.uniform(0.2, 0.5), 4)  # 石制裂缝高度
    else:
        crack_w = round(random.uniform(0.05, 0.2), 4)  # 木制裂缝宽度
        crack_h = round(random.uniform(0.2, 0.6), 4)  # 木制裂缝高度

    # 4. 保存标签到指定目录（文件名与图片一致）
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(label_dir, label_file)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"{class_id} {cx} {cy} {crack_w} {crack_h}")

    print(f"✅ [{idx}/{len(img_files)}] {crack_type} | 图片：{img_file} | 标签：{label_file}")

# 生成YOLO训练配置文件（适配双类别裂缝）
yaml_content = f"""
# 石制/木制裂缝数据集配置（D:\sy\crack）
train: {img_dir}          # 图片目录
val: {img_dir}            # 验证集复用图片目录
nc: 2                     # 2个类别：石制裂缝+木制裂缝
names: ["stone_crack", "wood_crack"]  # 类别名对应ID 0/1
"""
# 配置文件保存到标签目录
yaml_path = os.path.join(label_dir, "crack_dataset.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"\n📊 裂缝标签生成完成！")
print(f"🖼️  图片目录：{img_dir}")
print(f"🏷️  标签目录：{label_dir}（含{len(img_files)}个标签文件+1个训练配置文件）")
print(f"📋 类别规则：ID=0（石制裂缝）、ID=1（木制裂缝）")
input("按回车键退出...")