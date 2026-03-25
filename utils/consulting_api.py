import requests
import json
import time
import warnings

warnings.filterwarnings('ignore')


class AncientBuildingConsultAPI:
    """古建筑病害养护建议API调用类"""

    def __init__(self, api_type="tongyi", api_key="", timeout=30):
        """
        初始化API配置
        :param api_type: 大模型类型 - tongyi(通义千问)/ernie(文心一言)/gpt(OpenAI GPT)
        :param api_key: 平台API密钥（需自行申请）
        :param timeout: 请求超时时间（秒）
        """
        self.api_type = api_type.lower()
        self.api_key = api_key
        self.timeout = timeout
        self.base_urls = {
            "tongyi": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            "ernie": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions",
            "gpt": "https://api.openai.com/v1/chat/completions"
        }
        # 检查必要配置
        if not self.api_key:
            raise ValueError(f"请配置{self.api_type}模型的API密钥")

    def _build_tongyi_request(self, prompt):
        """构建通义千问请求参数"""
        return {
            "model": "qwen-turbo",  # 轻量版模型，兼顾速度和效果
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是古建筑保护领域的资深专家，擅长根据病害类型、环境数据给出科学、可落地的养护建议。回答需简洁明了，分点说明，语言通俗易懂，符合《中国文物古迹保护准则》要求。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "result_format": "text",
                "temperature": 0.7,  # 生成多样性
                "top_p": 0.8,
                "max_tokens": 1000  # 最大回复长度
            }
        }

    def _build_ernie_request(self, prompt):
        """构建文心一言请求参数"""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "你是古建筑保护领域的资深专家，擅长根据病害类型、环境数据给出科学、可落地的养护建议。回答需简洁明了，分点说明，语言通俗易懂，符合《中国文物古迹保护准则》要求。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

    def _build_gpt_request(self, prompt):
        """构建GPT请求参数"""
        return {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a senior expert in ancient building protection, skilled at providing scientific and feasible maintenance suggestions based on disease types and environmental data. The answer should be concise, point-by-point, easy to understand, and comply with the 'China Principles for the Conservation of Cultural Heritage Relics'."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

    def get_maintenance_suggestion(self, disease_type, disease_count, env_stats=None):
        """
        获取古建筑病害养护建议
        :param disease_type: 病害类型（如 "裂缝/剥落/裂缝+剥落"）
        :param disease_count: 病害数量（如 "裂缝5处，剥落3处"）
        :param env_stats: 环境统计数据（可选，来自env_analysis的统计结果）
        :return: 养护建议文本（str）
        """
        # 构建提示词
        env_info = ""
        if env_stats:
            env_info = f"""
环境数据参考：
- 平均温度：{env_stats['温度统计']['平均温度(℃)']}℃，最高{env_stats['温度统计']['最高温度(℃)']}℃
- 平均湿度：{env_stats['湿度统计']['平均湿度(%)']}%，最高{env_stats['湿度统计']['最高湿度(%)']}%
- 近段时间总降水量：{env_stats['降水量统计']['总降水量(mm)']}mm，降雨天数{env_stats['降水量统计']['降雨天数']}天
            """

        prompt = f"""
请针对以下古建筑病害情况给出具体的养护建议：
1. 病害类型及数量：{disease_count}（{disease_type}）
2. {env_info}

要求：
1. 先分析病害产生的可能原因（结合环境数据，若有）；
2. 给出分阶段养护建议（紧急处理、短期维护、长期预防）；
3. 建议需符合古建筑"最小干预"原则，优先采用传统工艺+现代技术结合的方式；
4. 语言简洁，避免专业术语过多，便于基层保护人员执行。
        """

        # 构建请求参数
        if self.api_type == "tongyi":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = self._build_tongyi_request(prompt)
        elif self.api_type == "ernie":
            # 文心一言需先拼接API URL（含API Key）
            self.base_urls["ernie"] = f"{self.base_urls['ernie']}?access_token={self.api_key}"
            headers = {
                "Content-Type": "application/json"
            }
            data = self._build_ernie_request(prompt)
        elif self.api_type == "gpt":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = self._build_gpt_request(prompt)
        else:
            raise ValueError(f"不支持的模型类型：{self.api_type}")

        # 发送请求（带重试机制）
        max_retry = 3
        retry_count = 0
        while retry_count < max_retry:
            try:
                response = requests.post(
                    url=self.base_urls[self.api_type],
                    headers=headers,
                    data=json.dumps(data),
                    timeout=self.timeout
                )
                response.raise_for_status()  # 抛出HTTP异常
                result = response.json()

                # 解析不同模型的返回结果
                if self.api_type == "tongyi":
                    suggestion = result["output"]["choices"][0]["message"]["content"]
                elif self.api_type == "ernie":
                    suggestion = result["result"]
                elif self.api_type == "gpt":
                    suggestion = result["choices"][0]["message"]["content"]

                # 清理结果格式
                suggestion = suggestion.strip().replace("\n\n", "\n")
                return suggestion

            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count >= max_retry:
                    raise RuntimeError(f"API请求失败（重试{max_retry}次）：{str(e)}")
                time.sleep(1)  # 重试间隔1秒
            except KeyError as e:
                raise RuntimeError(f"API返回结果解析失败：缺失字段{e}，返回内容：{result}")

    @staticmethod
    def get_api_key_guide():
        """获取各平台API Key申请指南"""
        guide = {
            "通义千问": {
                "申请地址": "https://dashscope.console.aliyun.com/",
                "步骤": "1. 注册阿里云账号；2. 开通通义千问API；3. 创建API-KEY；4. 充值（有免费额度）"
            },
            "文心一言": {
                "申请地址": "https://console.bce.baidu.com/qianfan/",
                "步骤": "1. 注册百度智能云账号；2. 开通文心一言服务；3. 创建应用获取API Key和Secret；4. 生成Access Token"
            },
            "OpenAI GPT": {
                "申请地址": "https://platform.openai.com/api-keys",
                "步骤": "1. 注册OpenAI账号；2. 创建API Key；3. 绑定支付方式（有免费额度）"
            }
        }
        return guide


# 简化调用函数（新手友好）
def get_consult_suggestion(disease_type, disease_count, api_type="tongyi", api_key="", env_stats=None):
    """
    简化版养护建议获取函数
    :param disease_type: 病害类型
    :param disease_count: 病害数量
    :param api_type: 模型类型
    :param api_key: API密钥
    :param env_stats: 环境统计数据
    :return: 养护建议
    """
    try:
        api_client = AncientBuildingConsultAPI(api_type=api_type, api_key=api_key)
        suggestion = api_client.get_maintenance_suggestion(disease_type, disease_count, env_stats)
        return suggestion
    except Exception as e:
        return f"获取养护建议失败：{str(e)}\n请检查API密钥或网络配置，或参考申请指南：{AncientBuildingConsultAPI.get_api_key_guide()}"


# 示例测试（需替换为自己的API Key）
if __name__ == "__main__":
    # 示例环境数据（来自env_analysis的统计结果）
    sample_env_stats = {
        "温度统计": {"平均温度(℃)": 15.2, "最高温度(℃)": 28.5},
        "湿度统计": {"平均湿度(%)": 70.5, "最高湿度(%)": 85.2},
        "降水量统计": {"总降水量(mm)": 120.8, "降雨天数": 15}
    }

    # 调用示例（以通义千问为例，需替换为自己的API Key）
    # suggestion = get_consult_suggestion(
    #     disease_type="裂缝+剥落",
    #     disease_count="裂缝5处，剥落3处",
    #     api_type="tongyi",
    #     api_key="你的通义千问API Key",
    #     env_stats=sample_env_stats
    # )
    # print(suggestion)

    # 打印API Key申请指南
    print("API Key申请指南：")
    for platform, info in AncientBuildingConsultAPI.get_api_key_guide().items():
        print(f"\n{platform}：")
        print(f"申请地址：{info['申请地址']}")
        print(f"步骤：{info['步骤']}")