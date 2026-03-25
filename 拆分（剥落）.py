import os
import shutil
import random

# ===================== 核心路径配置 =====================
source_img_dir = r"D:\sy\spall"  # 原始剥落图片目录（100张：前50石、后50木）
source_label_dir = r"D:\sy\标签\spall"  # 原始剥落标签目录
train_root = r"D:\sy\spall_dataset\train"  # 剥落训练集根目录
val_root = r"D:\sy\验证集\spall"  # 剥落验证集指定路径
val_ratio = 0.2  # 验证集比例（20%）
random_seed = 42  # 固定随机种子，结果可复现
# ==================================================

# 1. 创建目录结构（训练/验证集均按images/labels拆分）
train_img_dir = os.path.join(train_root, "images")
train_label_dir = os.path.join(train_root, "labels")
val_img_dir = os.path.join(val_root, "images")
val_label_dir = os.path.join(val_root, "labels")

for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
    os.makedirs(dir_path, exist_ok=True)

# 2. 按类别拆分数据（保证验证集类别均衡）
img_ext = [".jpg", ".png", ".jpeg"]
all_imgs = sorted([f for f in os.listdir(source_img_dir) if os.path.splitext(f)[1].lower() in img_ext])

# 拆分石制剥落（前50张）：40训练 + 10验证
stone_imgs = all_imgs[:50]
random.seed(random_seed)
val_stone_imgs = random.sample(stone_imgs, int(len(stone_imgs) * val_ratio))  # 10张验证
train_stone_imgs = [f for f in stone_imgs if f not in val_stone_imgs]  # 40张训练

# 拆分木制剥落（后50张）：40训练 + 10验证
wood_imgs = all_imgs[50:]
val_wood_imgs = random.sample(wood_imgs, int(len(wood_imgs) * val_ratio))  # 10张验证
train_wood_imgs = [f for f in wood_imgs if f not in val_wood_imgs]  # 40张训练

# 3. 复制训练集图片+标签
train_imgs = train_stone_imgs + train_wood_imgs
for img_file in train_imgs:
    # 复制图片到训练集
    src_img = os.path.join(source_img_dir, img_file)
    dst_img = os.path.join(train_img_dir, img_file)
    shutil.copy(src_img, dst_img)

    # 复制对应标签到训练集
    label_file = os.path.splitext(img_file)[0] + ".txt"
    src_label = os.path.join(source_label_dir, label_file)
    dst_label = os.path.join(train_label_dir, label_file)
    if os.path.exists(src_label):
        shutil.copy(src_label, dst_label)

# 4. 复制验证集图片+标签到指定路径（D:\sy\验证集\spall）
val_imgs = val_stone_imgs + val_wood_imgs
for img_file in val_imgs:
    # 复制图片到验证集
    src_img = os.path.join(source_img_dir, img_file)
    dst_img = os.path.join(val_img_dir, img_file)
    shutil.copy(src_img, dst_img)

    # 复制对应标签到验证集
    label_file = os.path.splitext(img_file)[0] + ".txt"
    src_label = os.path.join(source_label_dir, label_file)
    dst_label = os.path.join(val_label_dir, label_file)
    if os.path.exists(src_label):
        shutil.copy(src_label, dst_label)

# 5. 生成spall专属YOLO配置文件
yaml_content = f"""
# 石制/木制剥落数据集配置（验证集路径：D:\sy\验证集\spall）
train: {train_img_dir}  # 训练集图片路径
val: {val_img_dir}      # 验证集图片路径（指定到D:\sy\验证集\spall\images）
nc: 2                   # 类别数：石制剥落+木制剥落
names: ["stone_spall", "wood_spall"]  # 类别名对应ID 0/1
"""
# 配置文件保存到训练集根目录上层，方便调用
yaml_path = os.path.join(train_root, "../spall_dataset.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

# 6. 输出拆分结果
print("📊 剥落数据集拆分完成！")
print(f"🏋️  训练集目录：{train_root}")
print(f"   - 图片：{len(train_imgs)}张（{len(train_stone_imgs)}石制+{len(train_wood_imgs)}木制）")
print(f"   - 标签：{len(os.listdir(train_label_dir))}个")
print(f"✅ 验证集目录：{val_root}（指定路径）")
print(f"   - 图片：{len(val_imgs)}张（{len(val_stone_imgs)}石制+{len(val_wood_imgs)}木制）")
print(f"   - 标签：{len(os.listdir(val_label_dir))}个")
print(f"⚙️  YOLO配置文件：{yaml_path}")
input("按回车键退出...")