import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体（避免可视化时中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_environment_data(csv_path):
    """
    加载环境数据CSV文件，处理数据格式并做基础清洗
    :param csv_path: 环境数据文件路径（如 'data/environment_data.csv'）
    :return: 清洗后的DataFrame，包含列：日期(datetime)、温度(float)、湿度(float)、降水量(float)
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path, encoding='utf-8')

        # 基础列名检查（兼容常见列名格式）
        column_mapping = {
            '日期': 'date',
            '温度': 'temperature',
            '湿度': 'humidity',
            '降水量': 'precipitation',
            'Date': 'date',
            'Temperature': 'temperature',
            'Humidity': 'humidity',
            'Precipitation': 'precipitation'
        }
        df.rename(columns=column_mapping, inplace=True)

        # 必须包含的核心列
        required_cols = ['date', 'temperature', 'humidity', 'precipitation']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺失核心列：{missing_cols}，请检查CSV文件格式")

        # 数据类型转换与清洗
        # 1. 日期列转换为datetime格式
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # 2. 数值列转换为浮点数，替换异常值（如空值、非数字）
        for col in ['temperature', 'humidity', 'precipitation']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 填充空值（用均值，避免影响分析）
            df[col].fillna(df[col].mean(), inplace=True)
            # 过滤明显异常值（如温度>60℃或<-20℃，湿度>100%）
            if col == 'temperature':
                df = df[(df[col] >= -20) & (df[col] <= 60)]
            elif col == 'humidity':
                df = df[(df[col] >= 0) & (df[col] <= 100)]
            elif col == 'precipitation':
                df = df[df[col] >= 0]

        # 按日期排序
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"未找到环境数据文件：{csv_path}")
    except Exception as e:
        raise RuntimeError(f"加载环境数据失败：{str(e)}")


def get_statistical_analysis(df):
    """
    环境数据统计分析（均值、极值、波动等）
    :param df: 清洗后的环境数据DataFrame
    :return: 统计结果字典
    """
    stats = {
        '时间范围': {
            '开始日期': df['date'].min().strftime('%Y-%m-%d'),
            '结束日期': df['date'].max().strftime('%Y-%m-%d'),
            '数据天数': len(df)
        },
        '温度统计': {
            '平均温度(℃)': round(df['temperature'].mean(), 2),
            '最高温度(℃)': round(df['temperature'].max(), 2),
            '最低温度(℃)': round(df['temperature'].min(), 2),
            '温度标准差': round(df['temperature'].std(), 2)
        },
        '湿度统计': {
            '平均湿度(%)': round(df['humidity'].mean(), 2),
            '最高湿度(%)': round(df['humidity'].max(), 2),
            '最低湿度(%)': round(df['humidity'].min(), 2),
            '湿度标准差': round(df['humidity'].std(), 2)
        },
        '降水量统计': {
            '总降水量(mm)': round(df['precipitation'].sum(), 2),
            '日均降水量(mm)': round(df['precipitation'].mean(), 2),
            '最大单日降水量(mm)': round(df['precipitation'].max(), 2),
            '降雨天数': len(df[df['precipitation'] > 0])
        }
    }
    return stats


def plot_environment_trend(df, save_path=None):
    """
    绘制环境数据趋势图（温度/湿度/降水量时间序列）
    :param df: 清洗后的环境数据DataFrame
    :param save_path: 图片保存路径（None则返回图片对象）
    :return: Matplotlib图片对象（用于Streamlit展示）
    """
    # 创建画布（双Y轴：左侧温度/湿度，右侧降水量）
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制温度和湿度（左Y轴）
    ax1.set_xlabel('日期')
    ax1.set_ylabel('温度(℃)/湿度(%)', color='tab:blue')
    ax1.plot(df['date'], df['temperature'], color='tab:red', label='温度', linewidth=1.5)
    ax1.plot(df['date'], df['humidity'], color='tab:blue', label='湿度', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    # 绘制降水量（右Y轴）
    ax2 = ax1.twinx()
    ax2.set_ylabel('降水量(mm)', color='tab:green')
    ax2.bar(df['date'], df['precipitation'], color='tab:green', alpha=0.3, label='降水量')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.legend(loc='upper right')

    # 图表美化
    plt.title('古建筑环境数据趋势（温度/湿度/降水量）', fontsize=14, pad=20)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 保存图片或返回对象
    if save_path:
        # 创建保存目录
        import os
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def analyze_disease_env_correlation(df, disease_data=None):
    """
    分析病害与环境因素的相关性（可选：传入病害数据做关联）
    :param df: 环境数据DataFrame
    :param disease_data: 病害数据字典（如 {'crack_count': [5,3,8...], 'spall_count': [2,4,1...]}）
    :return: 相关性分析结果
    """
    # 基础环境因素相关性
    env_corr = df[['temperature', 'humidity', 'precipitation']].corr()

    # 若传入病害数据，分析病害与环境的相关性
    if disease_data and len(disease_data.get('crack_count', [])) == len(df):
        # 将病害数据合并到DataFrame
        df_combined = df.copy()
        df_combined['crack_count'] = disease_data['crack_count']
        df_combined['spall_count'] = disease_data['spall_count']

        # 计算病害与环境的相关性
        disease_env_corr = df_combined[
            ['temperature', 'humidity', 'precipitation', 'crack_count', 'spall_count']].corr()

        # 提取核心相关性结果
        correlation_result = {
            '环境因素间相关性': env_corr.round(2).to_dict(),
            '病害-环境相关性': {
                '裂缝-温度': round(disease_env_corr.loc['crack_count', 'temperature'], 2),
                '裂缝-湿度': round(disease_env_corr.loc['crack_count', 'humidity'], 2),
                '裂缝-降水量': round(disease_env_corr.loc['crack_count', 'precipitation'], 2),
                '剥落-温度': round(disease_env_corr.loc['spall_count', 'temperature'], 2),
                '剥落-湿度': round(disease_env_corr.loc['spall_count', 'humidity'], 2),
                '剥落-降水量': round(disease_env_corr.loc['spall_count', 'precipitation'], 2)
            }
        }
    else:
        correlation_result = {
            '环境因素间相关性': env_corr.round(2).to_dict(),
            '提示': '未传入病害数据，仅返回环境因素间相关性'
        }

    return correlation_result


# 示例：一站式环境数据分析函数
def full_environment_analysis(csv_path, save_plot_path=None, disease_data=None):
    """
    一站式环境数据分析（加载→统计→可视化→相关性）
    :param csv_path: 环境数据CSV路径
    :param save_plot_path: 趋势图保存路径
    :param disease_data: 病害数据（可选）
    :return: 综合分析结果字典
    """
    # 1. 加载数据
    df = load_environment_data(csv_path)
    # 2. 统计分析
    stats = get_statistical_analysis(df)
    # 3. 绘制趋势图
    fig = plot_environment_trend(df, save_plot_path)
    # 4. 相关性分析
    corr_result = analyze_disease_env_correlation(df, disease_data)

    # 整合结果
    result = {
        '数据概览': stats,
        '相关性分析': corr_result,
        '趋势图': fig
    }

    return result