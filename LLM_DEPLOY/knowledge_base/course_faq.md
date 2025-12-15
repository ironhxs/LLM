# 人工智能课程 - 常见问题解答（FAQ）

## 目录
- [课程相关](#课程相关)
- [作业与考试](#作业与考试)
- [实验环境](#实验环境)
- [Python 与工具](#python-与工具)
- [模型部署](#模型部署)
- [RAG 系统](#rag-系统)

---

## 课程相关

### Q1: 这门课的先修课程有哪些？
**A**: 必需的先修课程包括：
- 数据结构（必须）
- 概率论与数理统计（必须）
- 线性代数（必须）
- Python 编程基础（推荐）

如果没有 Python 基础，建议提前自学基本语法。

---

### Q2: 课程使用什么编程语言？
**A**: 主要使用 **Python**。课程实验会用到以下库：
- NumPy, Pandas（数据处理）
- PyTorch（深度学习）
- Scikit-learn（传统机器学习）
- LangChain（LLM 应用开发）

---

### Q3: 需要自己准备 GPU 吗？
**A**: 不是必须的。学校实验室提供了配备 GPU 的工作站，可预约使用。如果需要在自己电脑上运行：
- 小模型（如 MNIST）可以使用 CPU
- 大模型（如 Qwen-7B）建议使用云服务器（如 AutoDL、阿里云）

---

### Q4: 课程设计和期末大作业有什么区别？
**A**: 
- **课程设计（20分）**: 大模型部署与 RAG 系统实现，侧重工程实践
- **期末大作业（40分）**: 论文精读与分析，侧重学术研究能力

两个作业各自独立，都需要完成。

---

## 作业与考试

### Q5: 平时作业如何提交？
**A**: 通过学校教务系统提交，格式要求：
- 文件命名: `学号_姓名_第X次作业.pdf`
- 代码打包为 `.zip` 文件
- 截止时间: 每周日晚 23:59

迟交会扣除 20% 分数。

---

### Q6: 期末考试是开卷还是闭卷？
**A**: **闭卷考试**。允许携带一张 A4 纸的手写笔记（双面）。

考试题型：
- 选择题（20分）
- 简答题（30分）
- 计算题（20分，如反向传播）
- 综合题（30分，如设计 RAG 系统）

---

### Q7: 课程设计可以多人合作吗？
**A**: **不可以**。课程设计和期末大作业都必须**独立完成**。发现抄袭将按学校规定严肃处理。

实验课可以讨论，但代码必须自己实现。

---

## 实验环境

### Q8: 如何安装 PyTorch？
**A**: 根据你的 CUDA 版本选择安装命令：

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU 版本（不推荐用于大模型）
pip install torch torchvision torchaudio
```

检查安装：
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # 应该输出 True
```

---

### Q9: Conda 环境和 venv 有什么区别？
**A**: 
- **Conda**: 管理 Python 版本和包，适合科学计算（推荐）
- **venv**: Python 内置，仅管理包

推荐使用 Conda：
```bash
conda create -n ai_course python=3.10
conda activate ai_course
```

---

### Q10: 实验室的 GPU 如何预约？
**A**: 
1. 登录实验室预约系统: `http://lab.cs.hfut.edu.cn`
2. 选择"AI 实验室"
3. 选择时间段（每次最多 4 小时）
4. 提交预约（需辅导员审批）

注意：逾期不归还会影响下次预约。

---

## Python 与工具

### Q11: 为什么我的 Jupyter Notebook 连接不上 kernel？
**A**: 常见原因：
1. 环境未激活：确保在正确的 Conda 环境中
2. ipykernel 未安装：`pip install ipykernel`
3. 端口被占用：重启 Jupyter 服务

解决方法：
```bash
conda activate ai_course
python -m ipykernel install --user --name ai_course --display-name "AI Course"
jupyter notebook
```

---

### Q12: 如何在 VSCode 中使用 Jupyter Notebook？
**A**: 
1. 安装 VSCode 插件: "Jupyter" 和 "Python"
2. 打开 `.ipynb` 文件
3. 右上角选择 Kernel（选择你的 Conda 环境）
4. 运行单元格: `Shift + Enter`

---

### Q13: NumPy 数组和 Python 列表有什么区别？
**A**: 

| 特性 | NumPy 数组 | Python 列表 |
|:---|:---|:---|
| 存储类型 | 固定类型（如 float32） | 任意类型混合 |
| 运算速度 | 快（C 实现） | 慢（Python 解释） |
| 向量化操作 | 支持 | 不支持 |
| 内存占用 | 小 | 大 |

示例：
```python
import numpy as np

# NumPy（推荐）
a = np.array([1, 2, 3])
b = a * 2  # 向量化操作

# Python 列表
c = [1, 2, 3]
d = [x * 2 for x in c]  # 需要循环
```

---

## 模型部署

### Q14: 如何选择合适的大模型？
**A**: 根据硬件资源选择：

| GPU 显存 | 推荐模型 | 量化方式 |
|:---:|:---|:---|
| 4-6 GB | Qwen2.5-3B | GPTQ-Int4 |
| 8-12 GB | Qwen2.5-7B | GPTQ-Int4 |
| 16-24 GB | Qwen2.5-14B | GPTQ-Int8 |
| 24 GB+ | Qwen2.5-32B | FP16 |

课程设计推荐使用 **Qwen2.5-7B-Instruct-GPTQ-Int4**。

---

### Q15: 什么是模型量化？为什么需要量化？
**A**: 
- **量化**: 降低模型权重的精度（如 FP32 → INT4）
- **目的**: 减少显存占用，加快推理速度
- **代价**: 轻微的精度损失（通常 < 1%）

常见量化方法：
- GPTQ: 适合生成任务
- AWQ: 保留关键权重
- GGUF: 适合 CPU 推理

---

### Q16: 模型下载速度很慢怎么办？
**A**: 使用国内镜像源：

**方法 1: ModelScope**（推荐）
```python
from modelscope import snapshot_download

model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4')
```

**方法 2: HuggingFace 镜像**
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
```

---

## RAG 系统

### Q17: 什么是 RAG？为什么需要 RAG？
**A**: 
- **RAG**: Retrieval-Augmented Generation（检索增强生成）
- **原理**: 从知识库检索相关文档，注入提示词，增强模型回答

**优势**:
- 回答私有数据问题（模型训练集中没有的信息）
- 减少幻觉（基于真实文档回答）
- 知识可更新（无需重新训练模型）

---

### Q18: 为什么 RAG 检索的文档不相关？
**A**: 可能的原因：
1. **Embedding 模型不合适**: 使用中文模型处理英文文档
2. **分块策略不当**: `chunk_size` 过大或过小
3. **查询表述不清**: 问题过于宽泛

改进方法：
- 使用专门的中文 Embedding 模型（如 text2vec-base-chinese）
- 调整分块参数（推荐 chunk_size=500, overlap=50）
- 优化查询表述（添加关键词）

---

### Q19: FAISS 和 Milvus 有什么区别？
**A**: 

| 特性 | FAISS | Milvus |
|:---|:---|:---|
| 类型 | 本地库 | 分布式数据库 |
| 部署难度 | 简单 | 复杂（需要 Docker） |
| 扩展性 | 单机 | 支持集群 |
| 适用场景 | 小规模（< 100万向量） | 大规模（> 1亿向量） |

课程设计推荐使用 **FAISS**。

---

### Q20: 如何评估 RAG 系统的效果？
**A**: 

**定量指标**:
- **召回率（Recall@K）**: 前 K 个检索结果中包含正确答案的比例
- **MRR（Mean Reciprocal Rank）**: 正确答案的平均排名倒数

**定性评估**:
- 人工评估回答的准确性
- 对比有/无 RAG 的回答质量
- 检查引用来源是否正确

---

### Q21: 知识库可以使用哪些格式的文档？
**A**: LangChain 支持多种格式：

| 格式 | 加载器 | 说明 |
|:---:|:---|:---|
| `.txt` | TextLoader | 纯文本（推荐） |
| `.pdf` | PyPDFLoader | 需要安装 pypdf |
| `.docx` | Docx2txtLoader | 需要安装 docx2txt |
| `.md` | UnstructuredMarkdownLoader | Markdown 文档 |
| `.csv` | CSVLoader | 表格数据 |

推荐使用 `.txt` 或 `.md` 格式，兼容性最好。

---

### Q22: 如何更新知识库？
**A**: 

**方法 1: 重新构建**（简单但慢）
```python
# 添加新文档后
vectorstore = FAISS.from_documents(new_documents, embedding_model)
vectorstore.save_local("vector_store/")
```

**方法 2: 增量添加**（推荐）
```python
# 加载现有向量库
vectorstore = FAISS.load_local("vector_store/", embedding_model)

# 添加新文档
vectorstore.add_documents(new_documents)
vectorstore.save_local("vector_store/")
```

---

## 其他问题

### Q23: 在哪里可以找到课程资料？
**A**: 
- 课程网站: `http://ai.cs.hfut.edu.cn`
- 课程 GitHub: 实验代码与示例
- 钉钉群: 课程通知与答疑

---

### Q24: 如何联系助教？
**A**: 
- 邮箱: `ai_ta@hfut.edu.cn`
- 答疑时间: 每周三下午 2:00-4:00
- 答疑地点: 翡翠湖校区 计算机楼 305

---

### Q25: 期末考试可以带什么？
**A**: 
- ✅ 允许: 一张 A4 纸手写笔记（双面）、文具
- ❌ 禁止: 电子设备、书籍、打印资料

建议在笔记上总结：
- 常用公式（反向传播、损失函数）
- 算法伪代码（A*, Q-Learning）
- 重要概念定义

---

**更新日期**: 2024年10月  
**如有其他问题，请联系课程助教或在钉钉群提问**
