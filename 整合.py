import os
import shutil
import re

# 配置项（完全适配你的目录）
PROJECT_ROOT = r"C:\Users\小刘\PycharmProjects\ancient_building_system"
CRACK_PATH = os.path.join(PROJECT_ROOT, "data/disease_dataset/crack")
SPALL_PATH = os.path.join(PROJECT_ROOT, "data/disease_dataset/spall")
MERGED_PATH = os.path.join(PROJECT_ROOT, "data/disease_dataset/all_diseases")

# 工具函数：复制图片（直接从crack/train复制到all_diseases/images/train）
def copy_images(src_folder, dst_folder):
    create_dir_if_not_exist(dst_folder)
    for file in os.listdir(src_folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            src = os.path.join(src_folder, file)
            dst = os.path.join(dst_folder, file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    print(f"已复制图片到：{dst_folder}")

# 工具函数：复制并修改标签ID
def copy_labels(src_folder, dst_folder, target_id):
    create_dir_if_not_exist(dst_folder)
    for file in os.listdir(src_folder):
        if file.endswith(".txt"):
            src = os.path.join(src_folder, file)
            dst = os.path.join(dst_folder, file)
            with open(src, "r") as f:
                lines = f.readlines()
            new_lines = [re.sub(r"^\d+", str(target_id), line) for line in lines]
            with open(dst, "w") as f:
                f.writelines(new_lines)
    print(f"已复制标签到：{dst_folder}")

def create_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 主逻辑：手动复制train/val的图片和标签
if __name__ == "__main__":
    # 1. 复制裂缝的train图片和标签（ID=0）
    copy_images(
        src_folder=os.path.join(CRACK_PATH, "train"),
        dst_folder=os.path.join(MERGED_PATH, "images/train")
    )
    copy_labels(
        src_folder=os.path.join(CRACK_PATH, "train"),
        dst_folder=os.path.join(MERGED_PATH, "labels/train"),
        target_id=0
    )

    # 2. 复制裂缝的val图片和标签（ID=0）
    copy_images(
        src_folder=os.path.join(CRACK_PATH, "val"),
        dst_folder=os.path.join(MERGED_PATH, "images/val")
    )
    copy_labels(
        src_folder=os.path.join(CRACK_PATH, "val"),
        dst_folder=os.path.join(MERGED_PATH, "labels/val"),
        target_id=0
    )

    # 3. 复制剥落的train图片和标签（ID=1）
    copy_images(
        src_folder=os.path.join(SPALL_PATH, "train"),
        dst_folder=os.path.join(MERGED_PATH, "images/train")
    )
    copy_labels(
        src_folder=os.path.join(SPALL_PATH, "train"),
        dst_folder=os.path.join(MERGED_PATH, "labels/train"),
        target_id=1
    )

    # 4. 复制剥落的val图片和标签（ID=1）
    copy_images(
        src_folder=os.path.join(SPALL_PATH, "val"),
        dst_folder=os.path.join(MERGED_PATH, "images/val")
    )
    copy_labels(
        src_folder=os.path.join(SPALL_PATH, "val"),
        dst_folder=os.path.join(MERGED_PATH, "labels/val"),
        target_id=1
    )

    print("\n✅ 数据集合并完成！现在all_diseases/images/train里应该有图片了")