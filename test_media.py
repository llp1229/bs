import os
import base64

# 测试背景图
print("测试背景图...")
bg_image_path = r"D:\sy\背景.jpg"
if os.path.exists(bg_image_path):
    print(f"✓ 背景图文件存在：{bg_image_path}")
    try:
        with open(bg_image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()
        print(f"✓ 背景图编码成功，长度：{len(img_base64)}")
    except Exception as e:
        print(f"✗ 背景图编码失败：{str(e)}")
else:
    print(f"✗ 背景图文件不存在：{bg_image_path}")

# 测试视频文件
print("\n测试视频文件...")
video_path = r"D:\sy\古建筑.mp4"
if os.path.exists(video_path):
    print(f"✓ 视频文件存在：{video_path}")
    file_size = os.path.getsize(video_path)
    print(f"✓ 视频文件大小：{file_size / 1024 / 1024:.2f} MB")
    try:
        with open(video_path, "rb") as f:
            # 只读取一部分进行测试
            video_data = f.read(1024 * 1024)  # 读取1MB
        print(f"✓ 视频文件读取成功，读取大小：{len(video_data) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"✗ 视频文件读取失败：{str(e)}")
else:
    print(f"✗ 视频文件不存在：{video_path}")

print("\n测试完成！")