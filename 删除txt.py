import os
import random
from PIL import Image

# ---------------------- 核心配置（已适配你的需求） ----------------------
img_dir = r"D:\sy\木材"  # 改为木材图片目录
label_dir = r"D:\sy\标签\spall后"  # 标签保存目录不变
spall_class_id = 1  # spall对应的类别ID（可根据你的YOLO模型调整）
spall_class_name = "spall_defect"  # 统一标注为spall缺陷

# ---------------------- 初始化工作 ----------------------
# 自动创建标签目录（不存在则新建）
os.makedirs(label_dir, exist_ok=True)

# 获取图片文件（按文件名自然排序，兼容大小写后缀）
img_ext = [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]
img_files = sorted(
    [f for f in os.listdir(img_dir) if os.path.splitext(f)[1] in img_ext],
    key=lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else x
)  # 按文件名中的数字排序（如spall1051.jpg < spall1052.jpg）

# 校验图片数量
if len(img_files) == 0:
    print(f"❌ 错误：{img_dir} 目录下未找到任何图片！")
    exit(1)
print(f"✅ 找到 {len(img_files)} 张木材图片，开始生成标签...")

# ---------------------- 批量生成spall标签 ----------------------
for idx, img_file in enumerate(img_files, 1):
    # 1. 统一使用spall类别ID和名称（不再区分石制/木制）
    class_id = spall_class_id
    class_name = spall_class_name

    # 2. 获取图片真实尺寸（用于归一化标注）
    img_path = os.path.join(img_dir, img_file)
    try:
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        if img_w == 0 or img_h == 0:
            raise ValueError("图片尺寸为0")
    except Exception as e:
        print(f"⚠️  获取{img_file}尺寸失败，使用默认640×480：{str(e)[:30]}")
        img_w, img_h = 640, 480

    # 3. 生成合理的spall缺陷框（模拟真实标注，避免越界）
    cx = round(random.uniform(0.1, 0.9), 4)  # 中心点x（归一化）
    cy = round(random.uniform(0.1, 0.9), 4)  # 中心点y（归一化）
    w = round(random.uniform(0.05, min(0.8, 1 - cx)), 4)  # 宽度（归一化）
    h = round(random.uniform(0.05, min(0.6, 1 - cy)), 4)  # 高度（归一化）

    # 4. 保存标签文件（文件名与图片一致，保存到指定目录）
    label_filename = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(label_dir, label_filename)
    try:
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(f"{class_id} {cx} {cy} {w} {h}")
        print(f"✅ [{idx}/{len(img_files)}] {class_name} | 图片：{img_file} | 标签：{label_filename}")
    except Exception as e:
        print(f"❌ 保存{label_filename}失败：{str(e)[:30]}")

# ---------------------- 生成适配的YOLO训练配置文件 ----------------------
yaml_content = f"""# 木材spall缺陷数据集配置（自动生成）
train: {img_dir.replace('\\', '/')}  # 木材图片目录（转义反斜杠适配YOLO）
val: {img_dir.replace('\\', '/')}    # 验证集复用图片目录（可后续拆分）
nc: 1                                # 类别数量：仅spall缺陷
names: ["{spall_class_name}"]        # 类别名称
labels: {label_dir.replace('\\', '/')}  # 标签目录
"""

# 保存配置文件到标签目录
yaml_path = os.path.join(label_dir, "wood_spall_dataset.yaml")
try:
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\n📄 YOLO配置文件已生成：{yaml_path}")
except Exception as e:
    print(f"\n❌ 生成配置文件失败：{str(e)[:30]}")

# ---------------------- 最终统计 ----------------------
generated_labels = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
print(f"\n📊 生成完成！")
print(f"🖼️  处理木材图片总数：{len(img_files)} 张")
print(f"🏷️  生成spall标签总数：{len(generated_labels)} 个")
print(f"📌 标签保存路径：{label_dir}")
input("按回车键退出...")