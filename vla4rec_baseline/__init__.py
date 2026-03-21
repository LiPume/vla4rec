"""
================================================================================
VLA4Rec Baseline - Python Package
================================================================================

基于 VLA (Vision-Language-Action) 架构的音乐推荐系统 Baseline

核心模块:
- dataset: 数据流水线（滑动窗口切分）
- model: Causal Transformer 模型架构（强制解耦设计）
- metrics: 推荐系统评估指标 (HR@K, NDCG@K)
- train: 主训练循环

Author: VLA4Rec Team
Date: 2026-03-21
================================================================================
"""

from .dataset import (
    PlaylistDataset,
    collate_fn,
    create_dataloaders,
    get_dummy_dataloaders
)

from .model import (
    StateEncoder,
    ActionHead,
    VLA4RecBaseline,
    create_baseline_model
)

from .metrics import (
    compute_hit_rate,
    compute_ndcg,
    evaluate_model,
    print_metrics
)

from .train import Trainer, train

__all__ = [
    # Dataset
    'PlaylistDataset',
    'collate_fn',
    'create_dataloaders',
    'get_dummy_dataloaders',
    # Model
    'StateEncoder',
    'ActionHead',
    'VLA4RecBaseline',
    'create_baseline_model',
    # Metrics
    'compute_hit_rate',
    'compute_ndcg',
    'evaluate_model',
    'print_metrics',
    # Train
    'Trainer',
    'train'
]

__version__ = '0.1.0'
