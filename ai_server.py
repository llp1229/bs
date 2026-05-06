from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json

app = Flask(__name__)
CORS(app)

# 通义千问API配置
API_KEY = "sk-abf81210d5bc4443b041f4ed25bfbe9d"  # 替换为你的实际API Key
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

# 古建筑养护知识库
KNOWLEDGE_BASE = {
    "裂缝": """
    古建筑裂缝修补方法：
    1. 表面裂缝：采用环氧树脂注浆封闭
    2. 结构裂缝：需专业评估，可能需要加固处理
    3. 温度裂缝：控制环境温湿度，防止继续扩展
    """,
    "潮湿": """
    古建筑潮湿发霉处理：
    1. 改善通风条件
    2. 控制环境湿度在40-60%
    3. 使用防霉剂处理已发霉部位
    4. 检查防水层是否破损
    """,
    "冬季": """
    古建筑冬季防冻保护：
    1. 水管排空防冻裂
    2. 木结构防潮防冻
    3. 石质构件防冻融破坏
    4. 室内温度控制
    """,
    "防虫": """
    木结构防虫蛀方法：
    1. 定期检查虫蛀痕迹
    2. 使用防虫剂处理
    3. 保持干燥环境
    4. 必要时更换受损构件
    """,
    "剥落": """
    砖墙剥落修缮方法：
    1. 清理剥落部位
    2. 使用原配比材料修补
    3. 做好防水处理
    4. 定期检查维护
    """
}

def get_knowledge_response(question):
    """从知识库获取相关回答"""
    question_lower = question.lower()
    for keyword, answer in KNOWLEDGE_BASE.items():
        if keyword in question_lower:
            return answer.strip()
    return None

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('message', '')
        
        # 先检查知识库
        kb_answer = get_knowledge_response(question)
        if kb_answer:
            return jsonify({
                'success': True,
                'response': f"根据古建筑养护知识库：\n\n{kb_answer}\n\n如需更详细指导，请咨询专业文物保护机构。"
            })
        
        # 调用通义千问API
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": "qwen-turbo",
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是山西古建筑养护专家，专注于古建筑保护、修缮和日常养护建议。回答要专业、实用、简洁。"
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get('output', {}).get('text', '抱歉，无法获取回答')
            return jsonify({
                'success': True,
                'response': ai_response
            })
        else:
            return jsonify({
                'success': False,
                'error': f'API调用失败: {response.status_code}'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })



@app.route('/api/ask', methods=['POST'])
def ask():
    # /api/ask - compatible with dashboard frontend
    try:
        data = request.json
        question = data.get('question', '') or data.get('message', '')
        if not question:
            return jsonify({'success': False, 'error': 'empty question'})

        # Check knowledge base first
        kb_answer = get_knowledge_response(question)
        if kb_answer:
            return jsonify({
                'success': True,
                'answer': "knowledge base:\n\n" + kb_answer.strip()
            })

        # Call Tongyi Qianwen API
        headers = {
            'Authorization': 'Bearer ' + API_KEY,
            'Content-Type': 'application/json'
        }
        payload = {
            "model": "qwen-turbo",
            "input": {
                "messages": [
                    {"role": "system", "content": "role: Shanxi ancient building conservation expert. Be professional, practical, concise."},
                    {"role": "user", "content": question}
                ]
            }
        }
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            ai_resp = result.get('output', {}).get('text', 'no answer')
            return jsonify({'success': True, 'answer': ai_resp})
        else:
            return jsonify({'success': False, 'error': 'API failed(' + str(resp.status_code) + ')'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': '古建筑AI养护助手'})

if __name__ == '__main__':
    print("启动古建筑AI养护服务器...")
    print("访问地址: http://localhost:5188")
    app.run(host='0.0.0.0', port=5188, debug=False)
