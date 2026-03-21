# VLA4Rec Baseline 项目日志

## 项目概述

本项目实现了一个基于 VLA (Vision-Language-Action) 架构和模仿学习 (Imitation Learning) 的音乐推荐系统 Baseline。核心思想是将推荐过程建模为具身智能体的"连续动作轨迹生成"，使用纯 ID 的行为克隆 (Behavioral Cloning) 进行训练。

**核心假设**：
- Action (动作) = 预测的下一首歌 (Semantic ID)
- State (状态) = 历史歌曲 ID 序列
- 训练范式 = Causal Transformer 自回归 Next-Token Prediction

---

## 一、我做了什么

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
│  VLA4RecBaseline = StateEncoder + ActionHead (强制解耦)              │
└─────────────────────────────────────────────────────────────────────┘
                    │ 被调用
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          dataset.py                                  │
│  PlaylistDataset: 滑动窗口切分播放列表 → (输入序列, 目标) 样本        │
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
    ├── get_dummy_dataloaders()      [dataset.py]
    │       └── PlaylistDataset      [dataset.py]
    │               └── __getitem__() → {'input_seq', 'target'}
    │
    ├── create_baseline_model()      [model.py]
    │       └── VLA4RecBaseline      [model.py]
    │               ├── StateEncoder
    │               │       └── TransformerEncoder (因果注意力)
    │               └── ActionHead
    │                       └── Linear(hidden_size → vocab_size)
    │
    ├── model.forward(input_seq)     [model.py]
    │       └── output: [batch_size, vocab_size] 的 logits
    │
    ├── CrossEntropyLoss(output, target)    [内置]
    │
    └── evaluate_model()             [metrics.py]
            ├── compute_hit_rate()   [metrics.py]
            └── compute_ndcg()       [metrics.py]
```

### 2.3 数据流向 (Tensor Shape)

```
输入播放列表: [[1, 2, 3, 4, 5], [3, 7, 8, ...], ...]
                    │
                    ▼ [滑动窗口切分]
训练样本: (input_seq=[0,0,1], target=2), (input_seq=[0,1,2], target=3), ...
                    │
                    ▼ [DataLoader 批次化]
input_seq: [batch_size, max_seq_len]  # 例如 [32, 50]
target:    [batch_size]               # 例如 [32]
                    │
                    ▼ [VLA4RecBaseline.forward()]
StateEncoder:
  - token_embedding(input_seq): [32, 50] → [32, 50, 64]
  - + position_embedding:       [32, 50, 64]
  - TransformerEncoder:        [32, 50, 64] → [32, 50, 64]
ActionHead:
  - 取最后一个时间步:            [32, 50, 64] → [32, 64]
  - linear():                    [32, 64] → [32, vocab_size]
                    │
                    ▼
logits: [batch_size, vocab_size]  # 例如 [32, 1000]
                    │
                    ▼ [evaluate_model()]
HR@10, HR@20, NDCG@10, NDCG@20
```

---

## 三、核心设计详解

### 3.1 强制解耦设计 (model.py)

```python
VLA4RecBaseline
├── StateEncoder      # 状态编码器
│   ├── Token Embedding (vocab_size → hidden_size)
│   ├── Positional Embedding (max_seq_len → hidden_size)
│   └── Transformer Encoder (因果注意力)
│
└── ActionHead        # 动作预测头
    └── Linear (hidden_size → vocab_size)
```

**为什么要解耦？**
- 后续可在 `StateEncoder` 前后拼接视觉/文本编码器
- 可将离散的 Semantic ID 替换为连续的特征向量
- 便于单独微调某个模块

### 3.2 滑动窗口切分 (dataset.py)

播放列表 `[1, 2, 3, 4, 5]`，`max_seq_len=3`，`padding_id=0`：

| 输入序列 | 目标 |
|----------|------|
| `[0, 0, 1]` | `2` |
| `[0, 1, 2]` | `3` |
| `[1, 2, 3]` | `4` |
| `[2, 3, 4]` | `5` |

### 3.3 因果注意力 (model.py)

使用 `nn.TransformerEncoder` + `src_key_padding_mask` 实现单向注意力，确保模型只能看到当前位置及之前的信息。

---

## 四、使用方法

### 4.1 快速开始（使用虚拟数据）

```bash
# 激活环境
conda activate vla4rec

# 进入项目目录
cd /home/lzx/music_vla

# 运行训练（使用默认参数）
python -m vla4rec_baseline.train
```

### 4.2 自定义参数训练

```bash
# 方式1：命令行参数
python -m vla4rec_baseline.train \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --hidden_size 64 \
    --num_layers 2 \
    --num_heads 4 \
    --num_users 1000 \
    --vocab_size 1000 \
    --save_dir ./checkpoints
```

**主要参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_epochs` | 10 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--learning_rate` | 0.001 | 学习率 |
| `--hidden_size` | 64 | 隐向量维度 |
| `--num_layers` | 2 | Transformer 层数 |
| `--num_heads` | 4 | 注意力头数 |
| `--num_users` | 1000 | 模拟用户数（虚拟数据） |
| `--vocab_size` | 1000 | 词表大小（歌曲 ID 数量） |
| `--max_seq_len` | 50 | 最大序列长度 |

### 4.3 使用 Python API

```python
from vla4rec_baseline.dataset import get_dummy_dataloaders, PlaylistDataset
from vla4rec_baseline.model import create_baseline_model
from vla4rec_baseline.metrics import evaluate_model, print_metrics
from vla4rec_baseline.train import Trainer

# 1. 准备数据
train_loader, val_loader = get_dummy_dataloaders(
    num_users=1000,
    vocab_size=1000,
    batch_size=32
)

# 2. 创建模型
model, config = create_baseline_model(
    vocab_size=1000,
    hidden_size=64,
    num_layers=2,
    num_heads=4
)

# 3. 创建训练器
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    learning_rate=1e-3
)

# 4. 开始训练
for epoch in range(1, 11):
    trainer.train_epoch(epoch)
    val_metrics = trainer.validate()
    print_metrics(val_metrics, prefix=f"Epoch {epoch}")

# 5. 保存模型
trainer.save_checkpoint("final_model.pt")
```

### 4.4 使用真实数据

如果使用自己的播放列表数据，只需替换数据加载部分：

```python
# 假设 your_playlists 是 List[List[int]]，每个子列表是一首歌单
your_playlists = [
    [101, 205, 333, 450, ...],  # 用户1的歌单
    [88, 156, 302, ...],        # 用户2的歌单
    ...
]

# 创建 DataLoader
from vla4rec_baseline.dataset import create_dataloaders
train_loader, val_loader = create_dataloaders(
    playlists=your_playlists,
    batch_size=32,
    max_seq_len=50,
    train_ratio=0.8
)
```

---

## 五、关键类/函数索引

### 5.1 dataset.py

| 函数/类 | 签名 | 说明 |
|---------|------|------|
| `PlaylistDataset` | `__init__(playlists, max_seq_len, padding_id)` | 播放列表数据集 |
| `collate_fn` | `(batch) → Dict` | 批次整理函数 |
| `create_dataloaders` | `(playlists, ...) → (train, val)` | 创建 DataLoader |
| `get_dummy_dataloaders` | `(num_users, ...) → (train, val)` | 生成虚拟数据 |

### 5.2 model.py

| 函数/类 | 签名 | 说明 |
|---------|------|------|
| `StateEncoder` | `forward(input_ids, attention_mask) → states` | 状态编码器 |
| `ActionHead` | `forward(state_repr) → logits` | 动作预测头 |
| `VLA4RecBaseline` | `forward(input_ids) → logits` | 主模型 |
| `create_baseline_model` | `(vocab_size, hidden_size, ...) → (model, config)` | 模型工厂函数 |

### 5.3 metrics.py

| 函数/类 | 签名 | 说明 |
|---------|------|------|
| `compute_hit_rate` | `(logits, targets, k) → hr` | 计算 HR@K |
| `compute_ndcg` | `(logits, targets, k) → ndcg` | 计算 NDCG@K |
| `evaluate_model` | `(model, dataloader, device) → metrics` | 完整评估 |

### 5.4 train.py

| 函数/类 | 签名 | 说明 |
|---------|------|------|
| `Trainer` | `__init__(model, loaders, device, ...)` | 训练器封装类 |
| `train()` | `(num_epochs, batch_size, ...)` | 主训练函数（CLI 入口） |

---

## 六、注意事项

1. **Padding ID**：代码中 `padding_id=0`，歌曲 ID 从 1 开始，0 保留为 padding
2. **GPU 支持**：代码会自动检测 CUDA，但当前环境不支持 CUDA，会自动回退到 CPU
3. **虚拟数据**：默认使用 `get_dummy_dataloaders()` 生成模拟数据，方便快速验证
4. **检查点保存**：训练过程中会自动保存 `best_model.pt` 和每个 epoch 的检查点

---

## 七、后续扩展方向

基于当前的强制解耦设计，可以轻松进行以下扩展：

1. **多模态特征融合**：在 `StateEncoder` 前拼接视觉/文本编码器
2. **连续 Semantic ID**：将离散的 ID 替换为连续的特征向量
3. **强化学习增强**：在行为克隆基础上加入 RL 微调（如 DPO）
4. **更长上下文**：增加位置编码的上下文长度

---

*最后更新：2026-03-21*
