"""
================================================================================
VLA4Rec Baseline - Python Package
================================================================================

基于 VLA (Vision-Language-Action) 架构的音乐推荐系统 Baseline

核心模块:
- dataset: 数据流水线（滑动窗口切分，支持懒加载和索引优化）
- model: Causal Transformer 模型架构（强制解耦设计）
- metrics: 推荐系统评估指标 (HR@K, NDCG@K)
- train: 主训练循环（支持 torch.compile 和采样训练）

性能优化:
- 索引优化文件读取
- Epoch 采样训练
- torch.compile 模型编译
- 多进程数据加载

Author: VLA4Rec Team
Date: 2026-03-21
================================================================================
"""

from .dataset import (
    PlaylistDataset,
    collate_fn,
    create_dataloaders,
    get_dummy_dataloaders,
    create_spotify_dataloaders,
    LazyPlaylistDataset,
    StreamingDataLoader,
    SampledPlaylistDataset,
    Vocab
)

from .model import (
    StateEncoder,
    ActionHead,
    VLA4RecBaseline,
    create_baseline_model,
    print_model_memory_usage
)

from .metrics import (
    compute_hit_rate,
    compute_ndcg,
    evaluate_model,
    print_metrics
)

from .train import Trainer, train, NegSamplingLoss, get_gpu_memory_info

__all__ = [
    # Dataset
    'PlaylistDataset',
    'collate_fn',
    'create_dataloaders',
    'get_dummy_dataloaders',
    'create_spotify_dataloaders',
    'LazyPlaylistDataset',
    'StreamingDataLoader',
    'SampledPlaylistDataset',
    'Vocab',
    # Model
    'StateEncoder',
    'ActionHead',
    'VLA4RecBaseline',
    'create_baseline_model',
    'print_model_memory_usage',
    # Metrics
    'compute_hit_rate',
    'compute_ndcg',
    'evaluate_model',
    'print_metrics',
    # Train
    'Trainer',
    'train',
    'NegSamplingLoss',
    'get_gpu_memory_info'
]

__version__ = '0.2.0'
