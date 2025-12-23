# 🎯 DoRA 论文复现 - Git Submodule 部署方案

## 📖 项目说明

本项目实现了 ICML 2024 Oral 论文 **DoRA (Weight-Decomposed Low-Rank Adaptation)** 的完整复现流程。

为了方便管理第三方代码依赖，采用 **Git Submodule** 方式集成 [NVlabs/DoRA](https://github.com/NVlabs/DoRA) 官方仓库。

---

## 🚀 快速开始

### 克隆本仓库（包含子模块）

```bash
# 在 GPU 服务器上执行
cd /root/autodl-tmp

# 方法A: 克隆时自动拉取子模块（推荐）
git clone --recurse-submodules https://github.com/ironhxs/LLM.git

# 方法B: 先克隆主仓库，再初始化子模块
git clone https://github.com/ironhxs/LLM.git
cd LLM
git submodule update --init --recursive
```

### 运行一键部署脚本

```bash
cd LLM/Final-Project
chmod +x DoRA_deploy.sh
nohup bash DoRA_deploy.sh > train.log 2>&1 &

# 监控训练进度
tail -f train.log
watch -n 1 nvidia-smi
```

---

## 🔍 技术实现：Git Submodule

本项目使用 Git Submodule 管理 DoRA 依赖，具有以下特点：

- **版本锁定**: 记录使用的 DoRA 具体版本（commit hash）
- **独立更新**: DoRA 可独立升级，不影响主仓库提交历史
- **自动拉取**: 克隆时使用 `--recurse-submodules` 自动获取完整代码
- **清晰分离**: 第三方代码与自己的代码逻辑分离

---

## 📂 仓库结构

```
LLM/
├── .gitmodules              # Submodule 配置
├── Final-Project/
│   ├── DoRA/                # 官方 DoRA 仓库（Submodule）
│   │   ├── commonsense_reasoning/
│   │   │   ├── finetune.py
│   │   │   ├── llama_7B_Dora.sh
│   │   │   └── ...
│   │   └── ...
│   ├── DoRA_deploy.sh       # 一键部署脚本
│   └── Submodule部署说明.md  # 本文档
└── ...
```

---

## 🔧 开发者参考

### 检查子模块状态

```bash
git submodule status
# 输出示例: +a1b2c3d Final-Project/DoRA (heads/main)
```

### 更新 DoRA 到最新版本

```bash
cd Final-Project/DoRA
git checkout main
git pull origin main
cd ../..
git add Final-Project/DoRA
git commit -m "更新 DoRA 到最新版本"
git push
```

### 克隆后手动初始化子模块

```bash
# 如果克隆时忘记加 --recurse-submodules
git submodule update --init --recursive
```

---

## ⚙️ 实验配置

| 参数 | 值 |
|-----|---|
| 模型 | LLaMA-7B |
| 方法 | DoRA |
| Rank | 8 (快速验证版本) |
| Alpha | 16 |
| 数据集 | commonsense_170k |
| 评测集 | BoolQ |
| GPU | RTX 4090 24GB |
| 训练时间 | ~60-90 分钟 |

---

## 📝 论文信息

- **标题**: DoRA: Weight-Decomposed Low-Rank Adaptation
- **会议**: ICML 2024 (Oral, 1.5% 接受率)
- **链接**: https://arxiv.org/abs/2402.09353
- **官方仓库**: https://github.com/NVlabs/DoRA

---

## 💡 设计思路

采用 Git Submodule 而非直接包含代码的原因：

1. **保持代码纯净**: 第三方代码不混入提交历史
2. **便于追溯**: 清楚记录使用的 DoRA 版本
3. **简化部署**: 一条命令完成所有依赖拉取
4. **易于维护**: DoRA 更新时只需拉取最新代码
