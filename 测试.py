import os
from PIL import Image

SRC_DIR = r"C:\Users\小刘\PycharmProjects\ancient_building_system\data\disease_dataset\spall\train\images"
for img_file in os.listdir(SRC_DIR):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(SRC_DIR, img_file)
        try:
            with Image.open(img_path) as img:
                # 强制转为JPEG并覆盖原文件
                img.convert("RGB").save(img_path, "JPEG", quality=95)
                print(f"✅ 重编码完成：{img_file}")
        except:
            print(f"❌ 无法重编码：{img_file}")