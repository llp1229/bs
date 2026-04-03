import pandas as pd
import os

# 输入：原始xls文件目录
INPUT_DIR = "data/weather/raw"
# 输出：processed目录（你指定的）
OUTPUT_DIR = "data/weather/processed"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 遍历所有原始文件
for filename in os.listdir(INPUT_DIR):
    if filename.endswith((".xls", ".xlsx")):
        file_path = os.path.join(INPUT_DIR, filename)
        print(f"正在转换：{filename}")

        # 读取前2000行（平衡数据量和速度，不卡顿）
        df = pd.read_excel(file_path, nrows=2000, engine="xlrd" if filename.endswith(".xls") else "openpyxl")

        # 标准化列名（统一格式，避免后续报错）
        rename_dict = {}
        for col in df.columns:
            if "观测时间" in col:
                rename_dict[col] = "时间"
            elif "气温" in col:
                rename_dict[col] = "气温"
            elif "相对湿度" in col:
                rename_dict[col] = "相对湿度"
            elif "1小时降水量" in col:
                rename_dict[col] = "降水量"
            elif "2分钟平均风速" in col:
                rename_dict[col] = "风速"
        df.rename(columns=rename_dict, inplace=True)

        # 提取站点名，生成csv文件名
        station_name = filename.split("_小时PQC数据")[0]
        output_filename = f"{station_name}.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # 保存为utf-8编码的csv，避免中文乱码
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已保存：{output_filename}")

print("\n🎉 所有文件转换完成！CSV已全部放入 data/weather/processed 目录")