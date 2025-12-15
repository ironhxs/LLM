# Python 机器学习代码示例集

## 目录
1. [数据预处理](#数据预处理)
2. [线性回归](#线性回归)
3. [逻辑回归](#逻辑回归)
4. [神经网络](#神经网络)
5. [自然语言处理](#自然语言处理)

---

## 数据预处理

### 1.1 加载和探索数据

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('data.csv')

# 查看基本信息
print(data.head())
print(data.info())
print(data.describe())

# 检查缺失值
print(data.isnull().sum())

# 数据可视化
plt.figure(figsize=(10, 6))
data['feature'].hist(bins=50)
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.title('Feature Distribution')
plt.show()
```

### 1.2 数据标准化

```python
from sklearn.preprocessing import StandardScaler

# 创建标准化器
scaler = StandardScaler()

# 拟合并转换数据
X_scaled = scaler.fit_transform(X)

# 或者使用 MinMaxScaler（归一化到 0-1）
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

### 1.3 处理缺失值

```python
# 方法1: 删除缺失值
data_clean = data.dropna()

# 方法2: 用均值填充
data['feature'].fillna(data['feature'].mean(), inplace=True)

# 方法3: 用中位数填充
data['feature'].fillna(data['feature'].median(), inplace=True)

# 方法4: 用众数填充（分类变量）
data['category'].fillna(data['category'].mode()[0], inplace=True)
```

### 1.4 划分训练集和测试集

```python
from sklearn.model_selection import train_test_split

# 划分数据（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
```

---

## 线性回归

### 2.1 简单线性回归

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 创建模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")
print(f"权重: {model.coef_}")
print(f"截距: {model.intercept_}")
```

### 2.2 多项式回归

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 创建多项式回归模型（2次）
model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

# 训练
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 可视化拟合曲线
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = model.predict(X_plot)

plt.scatter(X, y, alpha=0.5)
plt.plot(X_plot, y_plot, 'r-', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression (degree=2)')
plt.show()
```

### 2.3 岭回归（Ridge Regression）

```python
from sklearn.linear_model import Ridge

# 创建岭回归模型（L2 正则化）
model = Ridge(alpha=1.0)

# 训练
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

---

## 逻辑回归

### 3.1 二分类任务

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 创建模型
model = LogisticRegression(max_iter=1000)

# 训练
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # 预测概率

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
```

### 3.2 可视化决策边界

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):
    # 生成网格点
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # 预测
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘图
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

# 使用（假设 X 是 2 维特征）
plot_decision_boundary(model, X_test, y_test)
```

---

## 神经网络

### 4.1 使用 PyTorch 构建简单神经网络

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleNN(input_size=784, hidden_size=128, output_size=10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 准备数据
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 评估
model.eval()
with torch.no_grad():
    test_outputs = model(torch.FloatTensor(X_test))
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == torch.LongTensor(y_test)).float().mean()
    print(f"测试准确率: {accuracy:.4f}")
```

### 4.2 MNIST 手写数字识别

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 训练模型（与上面类似）
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环...
```

---

## 自然语言处理

### 5.1 文本预处理

```python
import re
from collections import Counter

def preprocess_text(text):
    # 转小写
    text = text.lower()
    
    # 去除特殊字符
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 分词
    words = text.split()
    
    return words

# 构建词汇表
def build_vocabulary(texts, min_freq=2):
    all_words = []
    for text in texts:
        all_words.extend(preprocess_text(text))
    
    word_counts = Counter(all_words)
    vocab = {word: idx for idx, (word, count) in enumerate(word_counts.items()) if count >= min_freq}
    vocab['<UNK>'] = len(vocab)  # 未知词
    vocab['<PAD>'] = len(vocab)  # 填充
    
    return vocab

# 示例
texts = ["This is a sample text.", "Natural language processing is fun!"]
vocab = build_vocabulary(texts)
print(f"词汇表大小: {len(vocab)}")
```

### 5.2 使用 Transformers 进行文本分类

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# 加载预训练模型
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 准备数据
texts = ["这是一个正面评价", "这个产品很差"]
labels = [1, 0]  # 1: 正面, 0: 负面

# 编码文本
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

# 创建数据集
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

dataset = TextDataset(encodings, labels)

# 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir='./logs',
)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

### 5.3 使用 LangChain 实现简单的 RAG

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# 加载文档
loader = TextLoader("knowledge.txt", encoding='utf-8')
documents = loader.load()

# 分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 向量化
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 检索
query = "什么是机器学习？"
retrieved_docs = vectorstore.similarity_search(query, k=3)

print("检索结果:")
for i, doc in enumerate(retrieved_docs):
    print(f"\n[{i+1}] {doc.page_content[:200]}...")
```

---

## 常用技巧

### 交叉验证

```python
from sklearn.model_selection import cross_val_score

# K 折交叉验证
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"交叉验证分数: {scores}")
print(f"平均准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### 超参数调优

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

# 网格搜索
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳分数: {grid_search.best_score_:.4f}")
```

---

**文档版本**: v1.0  
**更新日期**: 2024年10月  
**适用课程**: 人工智能原理与应用
