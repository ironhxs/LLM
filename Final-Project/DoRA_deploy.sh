#!/bin/bash
# DoRA 论文复现 - 一键部署脚本
# 论文: DoRA: Weight-Decomposed Low-Rank Adaptation (ICML 2024 Oral)
# 项目地址: https://github.com/ironhxs/LLM
# 
# 使用方法:
#   git clone --recurse-submodules <仓库地址>
#   cd LLM/Final-Project
#   bash DoRA_deploy.sh

set -e

echo "================================================="
echo "  DoRA 论文复现 - 自动化部署脚本"
echo "  预计总时间: 90-120 分钟"
echo "================================================="

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# === 阶段1: 环境配置 ===
echo -e "\n[1/4] 配置 Conda 环境..."
conda create -n dora_llama python=3.10 -y
source activate dora_llama || conda activate dora_llama

echo -e "\n[2/4] 安装 Python 依赖..."
cd DoRA/commonsense_reasoning
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# === 阶段2: 数据准备 ===
echo -e "\n[3/4] 下载数据集..."
echo "正在下载 commonsense_170k 训练集..."
wget -O commonsense_170k.json https://github.com/AGI-Edgerunners/LLM-Adapters/raw/main/ft-training_set/commonsense_170k.json

mkdir -p dataset
cd dataset
echo "正在下载 BoolQ 评测集..."
wget https://github.com/AGI-Edgerunners/LLM-Adapters/raw/main/dataset/boolq.json
cd ..

# === 阶段3: 模型训练 ===
echo -e "\n[4/4] 启动 DoRA 训练..."
echo "配置: rank=8, alpha=16 (快速验证版本)"
echo "预计耗时: 60-90 分钟"
echo "-------------------------------------------------"
bash llama_7B_Dora.sh 8 16 ./result 0

# === 阶段4: 模型评测 ===
echo -e "\n✅ 训练完成！开始评测..."
python commonsense_evaluate.py \
  --model LLaMA-7B \
  --adapter DoRA \
  --dataset boolq \
  --base_model 'yahma/llama-7b-hf' \
  --lora_weights ./result \
  --batch_size 8

echo -e "\n================================================="
echo "  🎉 DoRA 复现完成！"
echo "  训练权重: ./result/"
echo "  评测结果: experiment/LLaMA-7B-DoRA-boolq.json"
echo "================================================="
