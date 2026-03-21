#!/bin/bash

# 遇到错误立即停止执行
set -e

echo "🚀 开始构建 VLA4Rec 专属 Conda 环境..."

# 1. 创建基础 Python 环境
echo "📦 [1/4] 已创建基础环境vla4rec (Python 3.10)..."
#conda create -n vla4rec python=3.10 -y

# 激活环境 (确保在 bash 脚本中能正确执行 conda activate)
eval "$(conda shell.bash hook)"
conda activate vla4rec

# 2. 安装 PyTorch 体系 (适配 RTX 4090 的 CUDA 12.1)
echo "🔥 [2/4] 安装 PyTorch 与 CUDA 12.1..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 预先安装构建工具并编译 Flash Attention
echo "⚡ [3/4] 编译 Flash Attention (可能需要几分钟，请耐心等待)..."
pip install ninja
conda install -c nvidia cuda-toolkit=12.1 -y
pip cache purge
# 使用 --no-build-isolation 确保它能找到刚刚装好的 PyTorch
pip install psutil
pip install flash-attn --no-build-isolation

# 4. 安装大模型生态、数据科学包与 Agent 工具链
echo "🧠 [4/4] 安装 Transformers, 数据科学包与 LangChain..."
pip install transformers accelerate datasets diffusers
pip install pandas numpy scikit-learn tqdm wandb
pip install langchain langchain-community

echo "✅ 环境构建完成！"
echo "👉 请输入命令激活环境: conda activate vla4rec"