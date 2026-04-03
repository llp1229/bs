import requests
import streamlit as st

def get_consult_suggestion(disease_type, disease_count, api_type="tongyi", api_key="", env_stats=None):
    """调用大模型生成山西古建筑专属养护建议"""
    try:
        # 构建Prompt（融合山西古建筑特性：木构/砖石、晋北/晋南气候差异）
        env_desc = f"""
        山西古建筑周边环境：
        - 平均温度{env_stats['平均温度(℃)']}℃，温度波动{env_stats['温度波动范围(℃)']}
        - 平均湿度{env_stats['平均湿度(%)']}%，湿度波动{env_stats['湿度波动范围(%)']}
        - 累计降水量{env_stats['累计降水量(mm)']}mm
        """ if env_stats else "无环境数据（默认按山西晋中地区气候）"

        prompt = f"""
        你是山西古建筑（木构/砖石）养护专家，针对以下情况给出实用建议：
        1. 检测病害：{disease_type}，数量：{disease_count}
        2. 环境背景：{env_desc}
        3. 建筑类型：山西传统木构/砖石古建筑（如应县木塔、晋祠、平遥古城类）
        要求：
        - 严格贴合山西当地气候和建筑材料特性；
        - 分点说明，每条不超20字，非专业人员可操作；
        - 聚焦病害修复+预防，最多4条核心建议；
        - 语言通俗，避免专业术语。
        """

        # 通义千问API调用（稳定版）
        if api_type == "tongyi":
            if not api_key:
                raise ValueError("请输入通义千问API Key（前往https://dashscope.aliyun.com/获取）")
            url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "qwen-turbo",
                "input": {"messages": [{"role": "user", "content": prompt}]},
                "parameters": {
                    "result_format": "text",
                    "max_tokens": 300,
                    "temperature": 0.7,
                    "top_p": 0.8
                }
            }
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["output"]["choices"][0]["message"]["content"]
        else:
            raise NotImplementedError("暂仅支持通义千问API")
    except requests.exceptions.Timeout:
        st.error("API请求超时，请检查网络或稍后重试")
        return None
    except ValueError as e:
        st.error(f"参数错误：{str(e)}")
        return None
    except Exception as e:
        st.error(f"API调用失败：{str(e)}")
        return None