import os
import zipfile
from PIL import Image
from io import BytesIO

def convert_to_jpg(img_bytes, save_path):
    """将图片字节流转为JPG，透明背景填充白色"""
    try:
        with Image.open(img_bytes) as img:
            if img.mode in ('RGBA', 'P'):  # 处理透明格式
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            img.save(save_path, 'JPEG', quality=90)
        return True
    except Exception as e:
        print(f"转换失败：{str(e)}")
        return False

def extract_docx_images(docx_path, output_folder):
    # 创建目标文件夹
    os.makedirs(output_folder, exist_ok=True)
    # 解压docx（本质是ZIP文件）提取图片
    with zipfile.ZipFile(docx_path, 'r') as zipf:
        img_index = 1
        for file_name in zipf.namelist():
            if file_name.startswith('word/media/'):  # 嵌入图片存储路径
                # 读取图片字节流
                img_data = zipf.read(file_name)
                img_bytes = BytesIO(img_data)
                # 定义JPG保存路径
                jpg_name = f"嵌入图片_{img_index}.jpg"
                jpg_path = os.path.join(output_folder, jpg_name)
                # 转换并保存
                if convert_to_jpg(img_bytes, jpg_path):
                    print(f"已保存：{jpg_path}")
                img_index += 1

# 执行提取（路径需与需求一致）
if __name__ == "__main__":
    docx_path = r"D:\sy\图片链接汇总-20241005.docx"
    output_folder = r"D:\sy\木材"
    extract_docx_images(docx_path, output_folder)