from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from modelscope import snapshot_download
import os
import time

app = Flask(__name__)
CORS(app)

# 全局变量
tokenizer = None
model = None
vectorstore = None
embedding_model = None

def load_models():
    """加载 LLM 和向量数据库"""
    global tokenizer, model, vectorstore, embedding_model
    
    # 获取项目根目录（src的上级目录）
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")
    
    print("=" * 50)
    print("正在初始化 RAG 系统...")
    print(f"工作目录: {BASE_DIR}")
    print("=" * 50)
    
    # 1. 加载 Embedding 模型
    print("\n[1/3] 加载 Embedding 模型...")
    # 尝试从本地 models 目录加载，如果不存在则下载
    embedding_model_id = 'AI-ModelScope/text2vec-base-chinese'
    try:
        print(f"正在检查/下载 Embedding 模型到: {MODELS_DIR}")
        embedding_model_path = snapshot_download(embedding_model_id, cache_dir=MODELS_DIR)
        print(f"使用本地 Embedding 模型: {embedding_model_path}")
    except Exception as e:
        print(f"ModelScope 下载失败，尝试在线加载: {e}")
        embedding_model_path = "shibing624/text2vec-base-chinese"

    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={'device': 'cpu'}, # 推理服务通常显存紧张，Embedding 用 CPU 即可
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Embedding 模型加载完成")
    
    # 2. 加载 FAISS 向量库
    print("\n[2/3] 加载 FAISS 向量数据库...")
    print(f"向量库路径: {VECTOR_STORE_PATH}")
    
    if not os.path.exists(VECTOR_STORE_PATH):
        print("❌ 错误: 向量数据库不存在！")
        print(f"请先运行 02_rag_implementation.ipynb 构建向量库")
        raise FileNotFoundError(f"向量库路径不存在: {VECTOR_STORE_PATH}")
    
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print(f"✅ 向量数据库加载完成，包含 {vectorstore.index.ntotal} 个向量")
    
    # 3. 加载 Qwen 模型
    print("\n[3/3] 加载 Qwen2.5-7B 模型...")
    model_id = 'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4'
    # 使用绝对路径的 models 目录
    model_dir = snapshot_download(model_id, cache_dir=MODELS_DIR)
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ Qwen 模型加载完成")
    
    print("\n" + "=" * 50)
    print("🚀 RAG 系统初始化完成！")
    print("=" * 50)

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "vectorstore_loaded": vectorstore is not None,
        "model_loaded": model is not None,
        "vector_count": vectorstore.index.ntotal if vectorstore else 0
    })

@app.route('/rag_chat', methods=['POST'])
def rag_chat():
    """RAG 增强问答"""
    try:
        data = request.json
        query = data.get('query', '')
        history = data.get('history', [])
        k = data.get('k', 3)  # 检索文档数量
        
        if not query:
            return jsonify({"error": "查询内容不能为空"}), 400
        
        start_time = time.time()
        
        # 1. 检索相关文档
        retrieved_docs = vectorstore.similarity_search(query, k=k)
        
        # 2. 构建上下文
        context_parts = []
        sources = []
        
        for i, doc in enumerate(retrieved_docs):
            source_file = os.path.basename(doc.metadata.get('source', 'unknown'))
            sources.append({
                "index": i + 1,
                "source": source_file,
                "content": doc.page_content
            })
            context_parts.append(f"【参考资料 {i+1}】来源: {source_file}\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # 3. 构建提示词
        prompt = f"""你是合肥工业大学人工智能课程的智能助手。请根据以下参考资料回答学生的问题。

要求：
1. 如果参考资料中有相关信息，基于资料给出准确、详细的回答
2. 如果参考资料中没有相关信息，明确说明"参考资料中暂无相关内容"
3. 回答要专业、清晰、条理分明
4. 对于课程、实验相关问题，尽可能给出具体指导

{context}

【问题】
{query}

【回答】"""

        # 4. 构建对话历史
        messages = [{"role": "system", "content": "你是合肥工业大学人工智能课程的智能助手，专门为学生解答课程相关问题。"}]
        
        # 添加历史对话（最多保留最近 5 轮）
        for h in history[-5:]:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
        
        messages.append({"role": "user", "content": prompt})
        
        # 5. 调用模型生成
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            do_sample=True
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 6. 更新历史
        history.append([query, response])
        
        elapsed = time.time() - start_time
        
        return jsonify({
            "response": response,
            "sources": sources,
            "history": history,
            "elapsed_time": round(elapsed, 2)
        })
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def simple_chat():
    """普通对话（不使用 RAG）"""
    try:
        data = request.json
        query = data.get('query', '')
        history = data.get('history', [])
        
        if not query:
            return jsonify({"error": "查询内容不能为空"}), 400
        
        start_time = time.time()
        
        # 构建对话
        messages = [{"role": "system", "content": "你是一个专业的AI助手，擅长回答技术和学术相关问题。"}]
        
        for h in history[-5:]:
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})
        
        messages.append({"role": "user", "content": query})
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.8,
            do_sample=True
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        history.append([query, response])
        
        elapsed = time.time() - start_time
        
        return jsonify({
            "response": response,
            "history": history,
            "elapsed_time": round(elapsed, 2)
        })
        
    except Exception as e:
        print(f"错误: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_models()
    print("\n🌐 AI课程助手 RAG 服务器启动中...")
    print("📍 访问地址: http://localhost:6006")
    print("📘 RAG端点: POST /rag_chat (基于课程知识库)")
    print("💬 普通对话: POST /chat")
    print("❤️  健康检查: GET /health")
    app.run(host='0.0.0.0', port=6006, debug=False)
