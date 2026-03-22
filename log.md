# VLA4Rec Baseline 项目日志

## 项目概述

本项目实现了一个基于 VLA (Vision-Language-Action) 架构和模仿学习 (Imitation Learning) 的音乐推荐系统 Baseline。核心思想是将推荐过程建模为具身智能体的"连续动作轨迹生成"，使用纯 ID 的行为克隆 (Behavioral Cloning) 进行训练。

**核心假设**：
- Action (动作) = 预测的下一首歌 (Semantic ID)
- State (状态) = 历史歌曲 ID 序列
- 训练范式 = Causal Transformer 自回归 Next-Token Prediction

---

## 更新日志

### 2026-03-21 - 大规模稀疏ID优化 (v2.0)

本次更新针对 Spotify 153 万量级 Action Space 进行了深度优化，使系统能够处理真实大规模推荐场景。

#### 核心优化内容

1. **模型优化 (model.py)**
   - 引入 **权重共享 (Weight Tying)** 策略
   - Embedding 权重与输出层 Linear 权重共享
   - 节省显存: `vocab_size * hidden_size * 4 bytes ≈ 393 MB`（单精度）
   - ActionHead 现在通过引用 embedding_layer 实现权重共享

2. **数据流水线优化 (dataset.py)**
   - 实现 **懒加载 (Lazy Loading)** 模式
   - 使用 Python 生成器逐行读取轨迹文件（747,510 行）
   - 新增 `Vocab` 类管理 URI 到 ID 的映射
   - 新增 `LazyPlaylistDataset` 支持流式数据处理
   - 新增 `StreamingDataLoader` 适配大规模训练

3. **训练流程优化 (train.py)**
   - 引入 **混合精度训练 (AMP)** 机制
   - 使用 `torch.cuda.amp.autocast` 和 `GradScaler`
   - 在 RTX 4090 上可节省约 50% 显存
   - 添加 **梯度累积** 支持
   - 新增 **负采样损失 (NegSamplingLoss)** 函数

#### 显存优化效果

| 优化项 | 节省显存 |
|--------|----------|
| 权重共享 (Weight Tying) | ~393 MB |
| 混合精度训练 (AMP) | ~50% |
| 负采样损失 (NegSamplingLoss) | O(vocab_size) → O(batch_size * num_neg_samples) |

**综合效果**: vocab_size=1,536,416 的模型可以在单卡 RTX 4090 上训练。

#### 核心类/函数变更

| 模块 | 类/函数 | 变更说明 |
|------|---------|----------|
| model.py | ActionHead | 新增 `embedding_layer` 参数实现权重共享 |
| model.py | VLA4RecBaseline | 新增 `use_weight_tying` 参数（默认开启）|
| model.py | print_model_memory_usage | 新增显存估算函数 |
| dataset.py | Vocab | 新增词表管理类 |
| dataset.py | LazyPlaylistDataset | 新增懒加载数据集类 |
| dataset.py | StreamingDataLoader | 新增流式加载器 |
| dataset.py | create_spotify_dataloaders | 新增 Spotify 数据加载器 |
| train.py | NegSamplingLoss | 新增负采样损失函数 |
| train.py | Trainer | 新增混合精度训练、梯度累积支持 |
| train.py | train | 新增多种优化参数 |
| train.py | get_gpu_memory_info | 新增显存监控函数 |

---

## 一、原始架构（保持不变）

### 1.1 创建的文件结构

```
vla4rec_baseline/
├── __init__.py       # 包导出，暴露对外接口
├── dataset.py        # 数据流水线模块
├── model.py          # 模型架构模块
├── metrics.py        # 评估指标模块
└── train.py          # 主训练循环模块
```

### 1.2 各模块实现内容

| 文件 | 主要内容 | 关键类/函数 |
|------|----------|-------------|
| `dataset.py` | 滑动窗口数据切分、虚拟数据生成 | `PlaylistDataset`, `get_dummy_dataloaders()` |
| `model.py` | 因果 Transformer、强制解耦架构 | `StateEncoder`, `ActionHead`, `VLA4RecBaseline` |
| `metrics.py` | HR@K 和 NDCG@K 指标计算 | `compute_hit_rate()`, `evaluate_model()` |
| `train.py` | 训练循环、验证循环、检查点管理 | `Trainer` 类, `train()` 函数 |

---

## 二、代码调用关系

### 2.1 模块依赖图

```
┌─────────────────────────────────────────────────────────────────────┐
│                              train.py                               │
│  (主入口：命令行参数解析、训练循环、模型保存)                          │
└─────────────────────────────────────────────────────────────────────┘
                    │ 导入依赖
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           model.py                                   │
│  VLA4RecBaseline = StateEncoder + ActionHead (强制解耦 + 权重共享)    │
└─────────────────────────────────────────────────────────────────────┘
                    │ 被调用
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          dataset.py                                  │
│  PlaylistDataset: 滑动窗口切分播放列表 → (输入序列, 目标) 样本          │
│  LazyPlaylistDataset: 懒加载版本，支持流式处理                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          metrics.py                                 │
│  根据模型输出的 logits 和真实目标，计算 HR@K 和 NDCG@K               │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 训练时的调用链

```
train() / Trainer.train_epoch()
    │
    ├── create_spotify_dataloaders()     [dataset.py]
    │       └── LazyPlaylistDataset      [dataset.py]
    │               └── StreamingDataLoader [dataset.py]
    │                       └── __iter__() → {'input_seq', 'target'}
    │
    ├── create_baseline_model()      [model.py]
    │       └── VLA4RecBaseline      [model.py]
    │               ├── StateEncoder
    │               │       └── TransformerEncoder (因果注意力)
    │               └── ActionHead
    │                       └── F.linear(state, embedding.weight, bias) [权重共享]
    │
    ├── model.forward(input_seq)     [model.py]
    │       └── output: [batch_size, vocab_size] 的 logits
    │
    ├── Trainer.train_step()        [train.py]
    │       ├── autocast()          [混合精度训练]
    │       └── criterion(output, target) [NegSamplingLoss 或 CrossEntropyLoss]
    │
    └── evaluate_model()             [metrics.py]
            ├── compute_hit_rate()   [metrics.py]
            └── compute_ndcg()       [metrics.py]
```

### 2.3 数据流向 (Tensor Shape)

```
输入轨迹文件: spotify_trajectories.txt (747,510 行)
                    │
                    ▼ [LazyPlaylistDataset + StreamingDataLoader]
训练样本流: (input_seq=[0,0,1], target=2), (input_seq=[0,1,2], target=3), ...
                    │
                    ▼ [DataLoader 批次化]
input_seq: [batch_size, max_seq_len]  # 例如 [64, 50]
target:    [batch_size]               # 例如 [64]
                    │
                    ▼ [VLA4RecBaseline.forward()]
StateEncoder:
  - token_embedding(input_seq): [64, 50] → [64, 50, 64]
  - + position_embedding:       [64, 50, 64]
  - TransformerEncoder:        [64, 50, 64] → [64, 50, 64]
ActionHead (权重共享):
  - 取最后一个时间步:            [64, 50, 64] → [64, 64]
  - F.linear():                 [64, 64] → [64, vocab_size] (使用 embedding.weight^T)
                    │
                    ▼
logits: [batch_size, vocab_size]  # 例如 [64, 1,536,416]
                    │
                    ▼ [NegSamplingLoss]
损失计算: 仅计算正样本 + num_neg_samples 个负样本
                    │
                    ▼ [evaluate_model()]
HR@10, HR@20, NDCG@10, NDCG@20
```

---

## 三、核心设计详解

### 3.1 强制解耦设计 + 权重共享 (model.py)

```
VLA4RecBaseline
├── StateEncoder      # 状态编码器
│   ├── Token Embedding (vocab_size → hidden_size) [权重共享给输出层]
│   ├── Positional Embedding (max_seq_len → hidden_size)
│   └── Transformer Encoder (因果注意力)
│
└── ActionHead        # 动作预测头
    └── F.linear(state, embedding.weight, bias) [权重共享，无独立参数]
```

**权重共享原理**：
- Embedding 层权重 shape: [vocab_size, hidden_size] = [1,536,416, 64]
- 输出层使用 embedding.weight^T 进行矩阵乘法
- 节省显存: 1,536,416 × 64 × 4 bytes ≈ 393 MB

### 3.2 懒加载数据流水线 (dataset.py)

```
spotify_trajectories.txt (747,510 行，每行一个播放列表)
                    │
                    ▼
┌─────────────────────────────────────────────┐
│         LazyPlaylistDataset                 │
│  - Vocab: 加载 uri_to_id.json               │
│  - _trajectory_generator(): 逐行读取文件      │
│  - _generate_samples_from_trajectory():     │
│    滑动窗口生成样本                          │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│         StreamingDataLoader                 │
│  - __iter__(): 迭代返回批次                  │
│  - 支持打乱轨迹顺序                          │
│  - 批次化样本                               │
└─────────────────────────────────────────────┘
```

### 3.3 混合精度训练 (train.py)

```
普通训练:
  forward: float32 → backward: float32 → 显存占用: 2x

混合精度训练 (AMP):
  forward: float16 → backward: float32 (scaled) → 显存占用: ~1.2x
```

### 3.4 负采样损失 (train.py)

```
标准 CrossEntropyLoss:
  logits: [batch_size, vocab_size] = [64, 1,536,416]
  计算所有类别的 softmax → O(vocab_size) 显存

NegSamplingLoss:
  logits: [batch_size, vocab_size] = [64, 1,536,416]
  采样 50 个负样本 → 计算 51 个类别的损失
  显存: O(batch_size * num_neg_samples)
```

---

## 四、使用方法

### 4.1 快速开始（使用 Spotify 真实数据）

```bash
# 激活环境
conda activate vla4rec

# 进入项目目录
cd /home/lzx/music_vla

# 运行训练（使用优化参数）
python -m vla4rec_baseline.train \
    --num_epochs 10 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --hidden_size 64 \
    --use_weight_tying \
    --use_mixed_precision
```

### 4.2 显存不足时的优化选项

```bash
# 选项1: 减小批次大小
python -m vla4rec_baseline.train \
    --batch_size 32

# 选项2: 使用梯度累积
python -m vla4rec_baseline.train \
    --batch_size 16 \
    --gradient_accumulation_steps 4

# 选项3: 使用负采样损失（最省显存）
python -m vla4rec_baseline.train \
    --batch_size 64 \
    --use_neg_sampling \
    --num_neg_samples 100

# 选项4: 组合优化
python -m vla4rec_baseline.train \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --use_mixed_precision \
    --use_neg_sampling
```

### 4.3 自定义参数训练

```bash
python -m vla4rec_baseline.train \
    --num_epochs 10 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --hidden_size 64 \
    --num_layers 2 \
    --num_heads 4 \
    --vocab_size 1536416 \
    --use_weight_tying \
    --use_mixed_precision \
    --gradient_accumulation_steps 1 \
    --save_dir ./checkpoints
```

**主要参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_epochs` | 10 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--learning_rate` | 0.001 | 学习率 |
| `--hidden_size` | 64 | 隐向量维度 |
| `--num_layers` | 2 | Transformer 层数 |
| `--num_heads` | 4 | 注意力头数 |
| `--use_weight_tying` | True | 是否使用权重共享 |
| `--use_mixed_precision` | True | 是否使用混合精度训练 |
| `--gradient_accumulation_steps` | 1 | 梯度累积步数 |
| `--use_neg_sampling` | False | 是否使用负采样损失 |
| `--num_neg_samples` | 50 | 负采样数量 |

### 4.4 使用 Python API

```python
from vla4rec_baseline.dataset import Vocab, LazyPlaylistDataset, create_spotify_dataloaders
from vla4rec_baseline.model import create_baseline_model, print_model_memory_usage
from vla4rec_baseline.metrics import evaluate_model, print_metrics
from vla4rec_baseline.train import Trainer, NegSamplingLoss

# 1. 准备数据（懒加载）
train_loader, val_loader = create_spotify_dataloaders(
    trajectories_file='spotify_trajectories.txt',
    vocab_file='uri_to_id.json',
    batch_size=64,
    max_seq_len=50,
    train_ratio=0.8
)

# 2. 创建模型（权重共享）
model, config = create_baseline_model(
    vocab_size=1536416,
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    use_weight_tying=True
)

# 3. 打印显存使用估算
print_model_memory_usage(model, batch_size=64, seq_len=50, device='cuda')

# 4. 创建训练器（混合精度）
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    learning_rate=1e-3,
    use_mixed_precision=True,
    use_neg_sampling=False
)

# 5. 开始训练
trainer.full_train(num_epochs=10, eval_interval=1000)

# 6. 保存模型
trainer.save_checkpoint("final_model.pt")
```

---

## 五、关键类/函数索引

### 5.1 model.py

| 函数/类 | 签名 | 说明 |
|---------|------|------|
| `StateEncoder` | `forward(input_ids, attention_mask) → states` | 状态编码器 |
| `ActionHead` | `forward(state_repr) → logits` | 动作预测头（支持权重共享）|
| `VLA4RecBaseline` | `forward(input_ids) → logits` | 主模型 |
| `create_baseline_model` | `(vocab_size, hidden_size, ...) → (model, config)` | 模型工厂函数 |
| `print_model_memory_usage` | `(model, batch_size, seq_len, device)` | 显存使用估算 |

### 5.2 dataset.py

| 函数/类 | 签名 | 说明 |
|---------|------|------|
| `Vocab` | `load(vocab_file)` | 词表管理类 |
| `LazyPlaylistDataset` | `__init__(...)` | 懒加载播放列表数据集 |
| `StreamingDataLoader` | `__iter__() → batch` | 流式数据加载器 |
| `PlaylistDataset` | `__init__(playlists, ...)` | 标准播放列表数据集（用于小规模）|
| `collate_fn` | `(batch) → Dict` | 批次整理函数 |
| `create_spotify_dataloaders` | `(trajectories_file, ...) → (train, val)` | 创建 Spotify DataLoader |
| `get_dummy_dataloaders` | `(num_users, ...) → (train, val)` | 生成虚拟数据 |

### 5.3 metrics.py

| 函数/类 | 签名 | 说明 |
|---------|------|------|
| `compute_hit_rate` | `(logits, targets, k) → hr` | 计算 HR@K |
| `compute_ndcg` | `(logits, targets, k) → ndcg` | 计算 NDCG@K |
| `evaluate_model` | `(model, dataloader, device) → metrics` | 完整评估 |

### 5.4 train.py

| 函数/类 | 签名 | 说明 |
|---------|------|------|
| `NegSamplingLoss` | `forward(logits, targets) → loss` | 负采样损失函数 |
| `Trainer` | `__init__(model, loaders, device, ...)` | 训练器封装类 |
| `train()` | `(num_epochs, batch_size, ...)` | 主训练函数（CLI 入口）|
| `get_gpu_memory_info` | `() → dict` | 获取GPU显存信息 |

---

## 六、大规模优化注意事项

1. **权重共享必须开启**：对于 vocab_size=1,536,416，关闭权重共享会直接导致显存溢出

2. **混合精度训练建议开启**：RTX 4090 使用 AMP 可节省约 50% 显存，这是最有效的优化手段

3. **负采样损失是保底选项**：当其他优化仍无法满足显存要求时，再启用负采样损失

4. **显存监控**：训练过程中会打印实时显存占用，便于及时发现问题

5. **批量大小调整策略**：
   - 优先降低批次大小（简单有效）
   - 其次增加梯度累积步数
   - 最后考虑负采样损失

---

## 七、后续扩展方向

1. **多模态特征融合**：在 `StateEncoder` 前后拼接视觉/文本编码器
2. **连续 Semantic ID**：将离散的 ID 替换为连续的特征向量
3. **强化学习增强**：在行为克隆基础上加入 RL 微调（如 DPO）
4. **更长上下文**：增加位置编码的上下文长度
5. **分布式训练**：多卡并行训练更大的模型

---

## 九、训练速度分析与优化（2026-03-21）

### 9.1 当前性能瓶颈分析

根据代码分析，一个 epoch 训练 1 小时的主要瓶颈：

| 瓶颈 | 原因 | 影响 |
|------|------|------|
| **数据遍历** | `StreamingDataLoader` 每次遍历全部 747,510 个播放列表 | ~1000 万样本/epoch |
| **单进程加载** | `num_workers=0`，无多进程数据预处理 | CPU 成为瓶颈 |
| **全量验证** | 每次验证遍历 50,000 样本（evaluate_model） | 每次验证耗时过长 |
| **计算效率** | 未使用 torch.compile、Flash Attention | GPU 利用率低 |
| **显存约束** | batch_size 受限，吞吐量低 | GPU 未充分利用 |

### 9.2 优化方案

#### 方案 A：数据流水线优化（最有效）

```python
# 1. 增加 num_workers（利用多核 CPU）
DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=4,           # 启用多进程
    prefetch_factor=2,        # 预取因子
    pin_memory=True,          # 加速数据传输
    persistent_workers=True   # 保持 worker 进程
)

# 2. 使用固定采样（每个 epoch 采样部分数据）
# 替代全遍历，假设每 epoch 只采样 10% 的播放列表
SAMPLE_RATIO = 0.1  # 每个 epoch 只用 10% 数据
num_sampled = int(num_trajectories * SAMPLE_RATIO)
```

#### 方案 B：模型编译优化

```python
# PyTorch 2.0+ 编译优化（可提速 20-50%）
model = torch.compile(model, mode="reduce-overhead")
```

#### 方案 C：训练策略优化

```bash
# 1. 增大 batch_size + 梯度累积（保持有效 batch_size）
python -m vla4rec_baseline.train \
    --batch_size 256 \
    --gradient_accumulation_steps 4  # 有效 batch_size = 1024

# 2. 减少验证频率
python -m vla4rec_baseline.train \
    --eval_interval 5000  # 从 1000 增加到 5000

# 3. 使用负采样损失（计算更快）
python -m vla4rec_baseline.train \
    --use_neg_sampling \
    --num_neg_samples 50
```

#### 方案 D：Flash Attention（Transformer 加速）

```python
# 在 TransformerEncoderLayer 中启用 Flash Attention
encoder_layer = nn.TransformerEncoderLayer(
    d_model=hidden_size,
    nhead=num_heads,
    dim_feedforward=intermediate_size or hidden_size * 4,
    dropout=dropout,
    activation='gelu',
    batch_first=True,
    norm_first=True,
    # PyTorch 2.0+ 支持
    # 使用 qkv_bias=True 配合 Flash Attention
)
```

### 9.3 推荐优化组合

**入门级（立即生效）**：
```bash
python -m vla4rec_baseline.train \
    --batch_size 128 \
    --gradient_accumulation_steps 4 \
    --num_workers 4 \
    --eval_interval 5000
```

**进阶级（需代码修改）**：
1. 添加 `torch.compile()` 编译模型
2. 启用 Flash Attention
3. 实现 epoch 采样机制

**预估效果**：
| 优化项 | 预期提速 | 实现难度 |
|--------|----------|----------|
| 增大 batch_size + 梯度累积 | 2-4x | 简单 |
| 多进程数据加载 | 1.5-2x | 简单 |
| torch.compile | 1.2-1.5x | 中等 |
| Epoch 采样 | 5-10x | 中等 |
| Flash Attention | 1.3-1.8x | 中等 |

### 9.4 未来多模态扩展的性能预估

当加入视觉、音频、歌词模态后，训练时间预计增长：

| 模态 | 预估时间增长 | 主要瓶颈 |
|------|--------------|----------|
| + 视觉特征 (CLIP) | 2-3x | 图像编码器 |
| + 音频特征 (CLAP) | 3-5x | 音频编码器 |
| + 歌词 (LLM) | 5-10x | 语言模型 |

**建议**：
1. 先在小规模数据上验证单模态
2. 使用预训练特征提取器（冻结权重）减少训练开销
3. 考虑使用更小的语言模型（如 DistilBERT）

---

## 十、文件变更历史

| 日期 | 版本 | 变更内容 |
|------|------|----------|
| 2026-03-21 | v1.0 | 初始版本，基础架构实现 |
| 2026-03-21 | v2.0 | 大规模稀疏ID优化：权重共享、懒加载、混合精度、负采样 |
| 2026-03-21 | v2.1 | 新增训练速度分析与优化方案 |

---

*最后更新：2026-03-21*
