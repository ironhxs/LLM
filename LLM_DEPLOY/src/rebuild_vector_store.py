import os
import sys
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from modelscope import snapshot_download

def rebuild_index():
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # src/
    PROJECT_ROOT = os.path.dirname(BASE_DIR) # LLM_DEPLOY/
    KB_DIR = os.path.join(PROJECT_ROOT, "knowledge_base")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "vector_store")

    print(f"正在重建向量库...")
    print(f"知识库目录: {KB_DIR}")
    
    # 1. 加载文档
    print("\n[1/4] 加载文档...")
    loaders = {
        ".txt": DirectoryLoader(KB_DIR, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}),
        ".md": DirectoryLoader(KB_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}),
    }
    
    documents = []
    for ext, loader in loaders.items():
        try:
            docs = loader.load()
            print(f"  加载 {ext} 文件: {len(docs)} 个")
            documents.extend(docs)
        except Exception as e:
            print(f"  加载 {ext} 失败 (可能没有此类文件): {e}")
            
    print(f"总计加载文档: {len(documents)} 个")
    
    if not documents:
        print("❌ 错误: 未找到任何文档，无法构建向量库")
        return

    # 2. 分割文档
    print("\n[2/4] 分割文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", ";", ",", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"生成文本块: {len(chunks)} 个")

    # 3. 加载 Embedding 模型
    print("\n[3/4] 加载 Embedding 模型...")
    embedding_model_path = os.path.join(MODELS_DIR, 'AI-ModelScope', 'text2vec-base-chinese')
    
    # 简单检查
    if not os.path.exists(os.path.join(embedding_model_path, 'config.json')):
        print("本地模型不完整，尝试下载...")
        try:
            embedding_model_path = snapshot_download('AI-ModelScope/text2vec-base-chinese', cache_dir=MODELS_DIR)
        except:
            embedding_model_path = "shibing624/text2vec-base-chinese"
            
    print(f"使用模型: {embedding_model_path}")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. 构建并保存索引
    print("\n[4/4] 构建并保存 FAISS 索引...")
    if not os.path.exists(VECTOR_STORE_PATH):
        os.makedirs(VECTOR_STORE_PATH)
        
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(VECTOR_STORE_PATH)
    
    print(f"✅ 向量库已重建并保存至: {VECTOR_STORE_PATH}")
    print("请重启 rag_server.py 以加载新数据")

if __name__ == "__main__":
    rebuild_index()
