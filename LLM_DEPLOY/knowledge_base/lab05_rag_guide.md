# 实验五：检索增强生成（RAG）系统构建

## 实验信息
- **实验名称**: RAG 系统构建与应用
- **实验学时**: 4 学时
- **实验类型**: 综合设计性实验
- **适用课程**: 人工智能原理与应用

---

## 一、实验目的

1. 理解检索增强生成（RAG）的基本原理与工作流程
2. 掌握文本向量化与相似度检索技术
3. 学习使用 LangChain 框架构建 RAG 应用
4. 熟悉向量数据库（FAISS）的使用方法
5. 实践大语言模型与知识库的集成方法

---

## 二、实验原理

### 2.1 RAG 概述

RAG（Retrieval-Augmented Generation）是一种结合信息检索与生成式模型的技术，通过从外部知识库检索相关文档，增强大语言模型的回答准确性。

### 2.2 工作流程

```
用户提问 
    ↓
文本向量化（Query Embedding）
    ↓
向量相似度检索（Vector Search）
    ↓
召回 Top-K 相关文档
    ↓
构建提示词（Prompt with Context）
    ↓
大语言模型生成回答
    ↓
返回结果 + 引用来源
```

### 2.3 关键技术

**文本向量化**
- 使用预训练的 Embedding 模型（如 text2vec-base-chinese）
- 将文本转换为高维向量表示（通常为 768 维）

**相似度检索**
- 使用 FAISS（Facebook AI Similarity Search）进行高效检索
- 支持 GPU 加速，检索速度快

**提示词工程**
- 将检索到的文档作为上下文注入提示词
- 引导模型基于参考资料回答问题

---

## 三、实验环境

### 3.1 软件环境
- Python 3.10
- PyTorch 2.1.2
- LangChain
- FAISS (GPU 版本)
- Transformers

### 3.2 硬件要求
- GPU: NVIDIA RTX 3060 或以上（显存 ≥ 8GB）
- 内存: ≥ 16GB
- 硬盘: ≥ 20GB（用于存储模型文件）

### 3.3 环境安装

```bash
# 创建虚拟环境
conda create -n rag_env python=3.10
conda activate rag_env

# 安装核心依赖
pip install langchain langchain-community langchain-huggingface
pip install sentence-transformers faiss-gpu
pip install transformers torch
```

---

## 四、实验内容

### 任务 1: 知识库准备（30 分钟）

**步骤**:
1. 准备文档：收集至少 5 篇相关文档（.txt / .md / .pdf）
2. 文档格式化：确保文档编码为 UTF-8
3. 文档存放：将文档放入 `knowledge_base/` 目录

**示例知识库主题**:
- 课程讲义与笔记
- 编程语言参考手册
- 技术文档与 FAQ
- 论文摘要与总结

**要求**:
- 文档总字数 ≥ 5000 字
- 内容具有一定的关联性
- 文档结构清晰，便于分块

---

### 任务 2: 文档加载与分块（30 分钟）

**代码示例**:

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 加载文档
loader = DirectoryLoader(
    'knowledge_base/',
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)
documents = loader.load()

print(f"加载文档数量: {len(documents)}")

# 2. 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块最大字符数
    chunk_overlap=50,    # 块之间重叠字符数
    length_function=len
)

chunks = text_splitter.split_documents(documents)
print(f"分块后数量: {len(chunks)}")

# 3. 查看分块示例
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- 分块 {i+1} ---")
    print(f"来源: {chunk.metadata['source']}")
    print(f"内容: {chunk.page_content[:100]}...")
```

**思考题**:
1. `chunk_size` 参数如何影响 RAG 效果？
2. 为什么需要设置 `chunk_overlap`？
3. 不同类型的文档是否需要不同的分块策略？

---

### 任务 3: 向量化与索引构建（40 分钟）

**代码示例**:

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 加载 Embedding 模型
embedding_model = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. 构建 FAISS 向量库
print("正在构建向量数据库...")
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

# 3. 保存向量库
vectorstore.save_local("vector_store/")
print("向量数据库已保存到 vector_store/")

# 4. 测试检索功能
query = "什么是 Transformer？"
retrieved_docs = vectorstore.similarity_search(query, k=3)

print(f"\n检索结果（Top-3）：")
for i, doc in enumerate(retrieved_docs):
    print(f"\n[{i+1}] 相似度分数: {doc.metadata.get('score', 'N/A')}")
    print(f"来源: {doc.metadata['source']}")
    print(f"内容: {doc.page_content[:200]}...")
```

**实验记录**:
- Embedding 模型加载时间: _____ 秒
- 向量化耗时: _____ 秒
- 索引大小: _____ MB
- 检索延迟: _____ 毫秒

---

### 任务 4: RAG 问答实现（50 分钟）

**代码示例**:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 加载大语言模型
model_name = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

# 2. 加载向量库
embedding_model = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={'device': 'cpu'}
)
vectorstore = FAISS.load_local(
    "vector_store/",
    embedding_model,
    allow_dangerous_deserialization=True
)

# 3. RAG 问答函数
def rag_query(query, k=3):
    # 检索相关文档
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    
    # 构建上下文
    context = "\n\n".join([
        f"【参考资料 {i+1}】\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    ])
    
    # 构建提示词
    prompt = f"""你是一个专业的技术助手。请根据以下参考资料回答问题。

参考资料：
{context}

问题：{query}

请基于参考资料给出准确回答。如果参考资料中没有相关信息，请明确说明。

回答："""

    # 调用模型生成
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8
    )
    
    response = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )
    
    return response, retrieved_docs

# 4. 测试问答
test_queries = [
    "Transformer 的自注意力机制是什么？",
    "如何评估机器学习模型的性能？",
    "深度学习中的反向传播算法是如何工作的？"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"问题: {query}")
    print(f"{'='*60}")
    
    response, sources = rag_query(query, k=3)
    
    print(f"\n回答:\n{response}")
    
    print(f"\n参考来源:")
    for i, doc in enumerate(sources):
        print(f"  [{i+1}] {doc.metadata['source']}")
```

**对比实验**:

设计对比实验，比较有/无 RAG 的回答质量：

| 问题 | 无 RAG 回答 | RAG 回答 | 准确性对比 |
|:---|:---|:---|:---|
| 课程特定问题 | | | |
| 通用知识问题 | | | |

---

### 任务 5: 性能优化（30 分钟）

**优化方向**:

1. **检索策略优化**
   - 调整 `k` 值（检索文档数量）
   - 使用 MMR（最大边际相关性）避免冗余

2. **提示词优化**
   - 优化上下文注入方式
   - 添加示例（Few-shot Learning）

3. **模型参数调整**
   - 调整 `temperature`（控制随机性）
   - 调整 `top_p`（核采样）

**实验记录**:

| 参数配置 | 回答质量 | 推理速度 | 综合评分 |
|:---|:---:|:---:|:---:|
| k=3, temp=0.7 | | | |
| k=5, temp=0.5 | | | |
| k=3, MMR | | | |

---

## 五、实验报告要求

### 5.1 报告结构

1. **实验目的**（简述）
2. **实验原理**（RAG 工作流程图）
3. **实验步骤**
   - 知识库准备
   - 向量化过程
   - RAG 实现代码
4. **实验结果**
   - 至少 3 个测试问题的问答结果
   - 有/无 RAG 的对比
   - 参数优化结果
5. **问题与讨论**
   - 遇到的问题及解决方法
   - RAG 的优势与局限
   - 改进建议
6. **实验总结**

### 5.2 提交内容

- 实验报告（PDF，包含截图与代码）
- 源代码（Python 脚本或 Jupyter Notebook）
- 知识库文档（至少 5 篇）
- 向量数据库文件（可选）

### 5.3 评分标准

| 评分项 | 分值 | 评分细则 |
|:---|:---:|:---|
| 代码实现 | 40 分 | 功能完整、代码规范 |
| 实验结果 | 30 分 | 结果正确、分析深入 |
| 报告质量 | 20 分 | 结构清晰、表达准确 |
| 创新点 | 10 分 | 优化方法、扩展功能 |

---

## 六、思考与扩展

### 6.1 思考题

1. RAG 相比直接使用大模型有哪些优势？存在哪些局限？
2. 如何选择合适的 Embedding 模型？中文和英文模型有何区别？
3. 向量数据库除了 FAISS，还有哪些选择？它们的特点是什么？
4. 如何处理长文档（超过模型最大输入长度）？
5. 如何评估 RAG 系统的检索准确性？

### 6.2 扩展任务（可选）

1. **多模态 RAG**: 支持图片与文档的混合检索
2. **实时更新**: 实现知识库的动态添加与删除
3. **Web 界面**: 开发基于 Flask/Streamlit 的交互界面
4. **性能监控**: 记录检索日志，分析用户查询模式

---

## 七、参考资料

- [LangChain 官方文档](https://python.langchain.com/)
- [FAISS 使用指南](https://github.com/facebookresearch/faiss/wiki)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- 论文: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

---

**实验指导教师**: 刘雪南教授  
**实验更新日期**: 2024年10月  
**联系方式**: ai_lab@hfut.edu.cn
