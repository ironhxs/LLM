from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download
import torch

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储模型
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    print("正在初始化模型...")
    
    # 使用本地绝对路径加载模型
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '..', 'models', 'Qwen', 'Qwen2___5-7B-Instruct-GPTQ-Int4')
    model_dir = os.path.abspath(model_dir)
    
    print(f"Loading model from: {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"模型加载完成！")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('query', '')
        history = data.get('history', []) # 获取客户端传来的历史对话
        
        if not query:
            return jsonify({"error": "No query provided"}), 400

        # 1. 构建系统提示词
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        
        # 2. 追加历史对话 (History 应该是 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}])
        if history:
            messages.extend(history)
            
        # 3. 追加当前问题
        messages.append({"role": "user", "content": query})
        
        # 4. 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,      # 关键修改：传入 input_ids 和 attention_mask
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                do_sample=True
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return jsonify({
            "response": response,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == '__main__':
    # 先加载模型
    load_model()
    # 启动服务，端口设为 6006
    print("API 服务已启动: http://localhost:6006")
    app.run(host='0.0.0.0', port=6006)
