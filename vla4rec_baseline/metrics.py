"""
================================================================================
VLA4Rec Baseline - 评估指标模块
================================================================================

本模块实现了推荐系统的标准评估指标，用于衡量模型的推荐质量。

评估指标：
- Hit Rate (HR@K): 命中率的衡量
- Normalized Discounted Cumulative Gain (NDCG@K): 排名质量衡量

核心假设：
- 输入是模型的预测 logits 和真实的下一首歌 ID
- 评估时不考虑 padding 位置

Author: VLA4Rec Team
Date: 2026-03-21
================================================================================
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def compute_hit_rate(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    padding_id: int = 0
) -> torch.Tensor:
    """
    计算 Hit Rate @ K

    Hit Rate 衡量真实标签是否出现在预测的 Top-K 列表中。
    HR@K = 1/K 命中数 / 总样本数
    或简化为: HR@K = 命中样本数 / 总样本数

    本实现采用后者（与大多数论文一致）。

    Args:
        logits: 模型预测的 logits
               Shape: [batch_size, vocab_size]
        targets: 真实的下一首歌 ID
               Shape: [batch_size]
        k: Top-K 中的 K 值
        padding_id: padding 的 ID（将被排除在评估外）

    Returns:
        hr: Hit Rate @ K (标量 tensor)
    """
    batch_size = logits.shape[0]

    # 获取 Top-K 预测索引
    # [batch_size, k]
    _, top_k_indices = torch.topk(logits, k=k, dim=-1)

    # 检查每个真实标签是否在 Top-K 预测中
    # targets: [batch_size] -> [batch_size, 1]
    targets_expanded = targets.unsqueeze(-1)

    # [batch_size, k] 中每个位置是否等于目标
    hits = (top_k_indices == targets_expanded).any(dim=-1)  # [batch_size]

    # 过滤掉 padding 样本
    valid_mask = targets != padding_id
    hits = hits & valid_mask

    # 计算 Hit Rate
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    hr = hits.float().sum() / valid_mask.float().sum()

    return hr


def compute_ndcg(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    padding_id: int = 0
) -> torch.Tensor:
    """
    计算 Normalized Discounted Cumulative Gain @ K (NDCG@K)

    NDCG 是衡量推荐排名质量的指标，考虑了：
    1. 相关性（是否命中）
    2. 排名位置（越靠前的命中越重要）

    公式:
    DCG@K = Σ (rel_i / log2(i+1))，i 从 1 到 K
    NDCG@K = DCG@K / IDCG@K

    其中 rel_i = 1 如果命中，否则为 0

    Args:
        logits: 模型预测的 logits
               Shape: [batch_size, vocab_size]
        targets: 真实的下一首歌 ID
               Shape: [batch_size]
        k: Top-K 中的 K 值
        padding_id: padding 的 ID

    Returns:
        ndcg: NDCG@K (标量 tensor)
    """
    batch_size = logits.shape[0]

    # 过滤 padding 样本
    valid_mask = targets != padding_id
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    # 只对有效样本计算
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]

    if len(valid_indices) == 0:
        return torch.tensor(0.0, device=logits.device)

    valid_logits = logits[valid_indices]  # [num_valid, vocab_size]
    valid_targets = targets[valid_indices]  # [num_valid]

    # 获取 Top-K 预测索引
    _, top_k_indices = torch.topk(valid_logits, k=k, dim=-1)  # [num_valid, k]

    # 计算 DCG
    # 对于每个样本，检查目标是否在 Top-K 中以及其位置
    # targets: [num_valid] -> [num_valid, 1]
    targets_expanded = valid_targets.unsqueeze(-1)

    # [num_valid, k] - 是否命中
    is_hit = (top_k_indices == targets_expanded).float()  # [num_valid, k]

    # 排名折扣：位置 i (1-indexed) 的折扣为 1/log2(i+1)
    # positions: [1, 2, 3, ..., k]
    positions = torch.arange(1, k + 1, device=logits.device).float()
    discount = 1.0 / torch.log2(positions + 1)  # [k]

    # DCG = Σ (is_hit * discount)
    dcg = (is_hit * discount.unsqueeze(0)).sum(dim=-1)  # [num_valid]

    # IDCG = 1（完美情况下命中的 DCG）
    idcg = torch.ones_like(dcg)

    # NDCG = DCG / IDCG
    ndcg = dcg / idcg

    return ndcg.mean()


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    padding_id: int = 0,
    k_values: Tuple[int, ...] = (10, 20)
) -> Dict[str, float]:
    """
    在验证集上评估模型，计算多个 K 值下的 HR 和 NDCG

    Args:
        model: 待评估的模型
        dataloader: 验证数据的 DataLoader
        device: 计算设备
        padding_id: padding 的 ID
        k_values: 要评估的 K 值元组

    Returns:
        metrics: 指标字典，如 {'HR@10': 0.5, 'HR@20': 0.6, 'NDCG@10': 0.4, 'NDCG@20': 0.5}
    """
    model.eval()

    # 累计指标
    all_hits = {k: 0 for k in k_values}
    all_ndcg = {k: 0.0 for k in k_values}
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 获取数据
            input_seq = batch['input_seq'].to(device)
            targets = batch['target'].to(device)

            # 前向传播
            logits = model(input_seq)  # [batch_size, vocab_size]

            # 过滤掉目标为 padding 的样本
            valid_mask = targets != padding_id
            if valid_mask.sum() == 0:
                continue

            # 计算每个 K 值的指标
            for k in k_values:
                hr = compute_hit_rate(logits, targets, k=k, padding_id=padding_id)
                ndcg = compute_ndcg(logits, targets, k=k, padding_id=padding_id)

                # 累加（加权）
                num_valid = valid_mask.sum().item()
                all_hits[k] += hr.item() * num_valid
                all_ndcg[k] += ndcg.item() * num_valid

            total_samples += valid_mask.sum().item()

    # 平均
    if total_samples == 0:
        return {f'HR@{k}': 0.0 for k in k_values} | {f'NDCG@{k}': 0.0 for k in k_values}

    metrics = {}
    for k in k_values:
        metrics[f'HR@{k}'] = all_hits[k] / total_samples
        metrics[f'NDCG@{k}'] = all_ndcg[k] / total_samples

    return metrics


def print_metrics(
    metrics: Dict[str, float],
    prefix: str = "",
    decimals: int = 4
):
    """
    打印评估指标

    Args:
        metrics: 指标字典
        prefix: 前缀（如 "Validation"）
        decimals: 小数位数
    """
    if prefix:
        prefix = f"[{prefix}] "

    metric_strs = []
    for key, value in sorted(metrics.items()):
        metric_strs.append(f"{prefix}{key}: {value:.{decimals}f}")

    print(" | ".join(metric_strs))


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("测试评估指标")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模拟数据
    batch_size = 32
    vocab_size = 1000

    # 随机 logits
    logits = torch.randn(batch_size, vocab_size)
    # 真实目标（避免 0，因为 0 是 padding）
    targets = torch.randint(1, vocab_size, (batch_size,))

    # 测试 Hit Rate
    print("\nHit Rate 测试:")
    for k in [5, 10, 20]:
        hr = compute_hit_rate(logits, targets, k=k)
        print(f"  HR@{k}: {hr.item():.4f}")

    # 测试 NDCG
    print("\nNDCG 测试:")
    for k in [5, 10, 20]:
        ndcg = compute_ndcg(logits, targets, k=k)
        print(f"  NDCG@{k}: {ndcg.item():.4f}")

    # 测试评估函数
    print("\n完整评估测试:")

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(vocab_size, vocab_size)

        def forward(self, x):
            return self.fc(x)

    model = DummyModel().to(device)
    dummy_loader = [
        {'input_seq': torch.randint(1, 100, (batch_size, 50)), 'target': torch.randint(1, vocab_size, (batch_size,))}
    ]

    metrics = evaluate_model(model, dummy_loader, device, k_values=(10, 20))
    print_metrics(metrics, prefix="Test")

    print("\n" + "=" * 60)
    print("评估指标测试通过!")
    print("=" * 60)
