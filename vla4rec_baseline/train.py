"""
================================================================================
VLA4Rec Baseline - 主训练循环 (大规模稀疏ID优化版)
================================================================================

本模块实现了完整的训练和验证流程，针对大规模稀疏ID（153万量级）进行了深度优化。

核心优化：
1. 混合精度训练 (AMP): 使用 torch.cuda.amp.autocast，在 RTX 4090 上可节省约 50% 显存
2. 梯度累积: 当显存不足时，通过累积多个小批次来实现有效的大批次训练
3. 负采样损失: 可选的稀疏损失函数，避免全量 CrossEntropyLoss 的显存压力
4. 动态批次大小: 根据可用显存自动调整批次大小

核心训练范式：行为克隆 (Behavioral Cloning)
- 通过专家轨迹（高质量播放列表）进行模仿学习
- 使用 CrossEntropyLoss 进行 Next-Token Prediction

Author: VLA4Rec Team
Date: 2026-03-21
================================================================================
"""

import os
import sys
import time
import glob
import gc
from typing import Optional, Tuple, Dict, List, Union
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import matplotlib.pyplot as plt

# 导入本地模块
from .dataset import (
    get_dummy_dataloaders,
    PlaylistDataset,
    create_dataloaders,
    create_spotify_dataloaders,
    LazyPlaylistDataset,
    StreamingDataLoader,
    Vocab
)
from .model import VLA4RecBaseline, create_baseline_model, print_model_memory_usage
from .metrics import evaluate_model, print_metrics


class NegSamplingLoss(nn.Module):
    """
    负采样损失函数

    针对大规模 vocab_size（153万）的优化：当 target 对应的类别数太多时，
    直接使用 CrossEntropyLoss 会因为计算所有类别的 softmax 而导致显存爆炸。

    负采样通过只计算正样本和少量负样本的损失来解决这个问题。
    """

    def __init__(
        self,
        vocab_size: int,
        num_neg_samples: int = 50,
        padding_idx: int = 0
    ):
        """
        初始化负采样损失

        Args:
            vocab_size: 词表大小
            num_neg_samples: 负采样数量（默认50，与正样本比例约1:50）
            padding_idx: padding索引，损失中忽略该类别
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_neg_samples = num_neg_samples
        self.padding_idx = padding_idx

        # LogQ修正因子缓存
        self._logq_cache = {}
        self.register_buffer('uniform_prob', torch.ones(vocab_size) / vocab_size)

    def _sample_negatives(
        self,
        targets: torch.Tensor,
        num_neg: int
    ) -> torch.Tensor:
        """
        对目标类别采样负样本

        Args:
            targets: 目标类别张量，形状为 [batch_size]
            num_neg: 每个目标采样的负样本数量

        Returns:
            负样本张量，形状为 [batch_size, num_neg]
        """
        batch_size = targets.shape[0]

        # 为每个样本生成不同的负样本
        neg_samples = torch.randint(
            1,  # 从1开始避免padding
            self.vocab_size,
            (batch_size, num_neg),
            device=targets.device,
            dtype=targets.dtype
        )

        return neg_samples

    def _compute_logq(self, targets: torch.Tensor) -> torch.Tensor:
        """
        计算LogQ修正因子
        """
        if targets.device not in self._logq_cache:
            freq = torch.bincount(targets, minlength=self.vocab_size).float()
            freq = freq + 1  # 平滑
            prob = freq / freq.sum()
            self._logq_cache[targets.device] = torch.log(prob)

        return self._logq_cache[targets.device][targets]

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算负采样损失

        Args:
            logits: 模型输出的logits，形状为 [batch_size, vocab_size]
            targets: 目标类别，形状为 [batch_size]

        Returns:
            损失标量
        """
        batch_size = targets.shape[0]
        device = logits.device

        # 采样负样本
        neg_samples = self._sample_negatives(targets, self.num_neg_samples)

        # 计算正样本的logit
        pos_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 计算负样本的logit
        neg_logits = logits.gather(1, neg_samples)

        # 使用 sigmoid binary cross entropy（等价于 NCE）
        # 正样本标签为1，负样本标签为0
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits,
            torch.ones_like(pos_logits),
            reduction='mean'
        )

        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits,
            torch.zeros_like(neg_logits),
            reduction='mean'
        )

        loss = pos_loss + neg_loss

        return loss


def get_next_run_id(base_dir: str) -> int:
    """获取下一个run的编号"""
    existing_runs = glob.glob(os.path.join(base_dir, "run*"))
    if not existing_runs:
        return 1

    run_ids = []
    for run_path in existing_runs:
        try:
            run_id = int(os.path.basename(run_path).replace("run", ""))
            run_ids.append(run_id)
        except ValueError:
            continue

    return max(run_ids) + 1 if run_ids else 1


def create_run_directory(base_dir: str = "./checkpoints") -> str:
    """创建新的run目录"""
    os.makedirs(base_dir, exist_ok=True)
    run_id = get_next_run_id(base_dir)
    run_dir = os.path.join(base_dir, f"run{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_metrics_to_csv(metrics_history: List[Dict], filepath: str):
    """保存指标历史到CSV文件"""
    if not metrics_history:
        return

    df = pd.DataFrame(metrics_history)
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")


def plot_training_curves(train_history: List[Dict], val_history: List[Dict], save_dir: str):
    """绘制训练曲线"""
    if not val_history:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Curves', fontsize=14, fontweight='bold')

    steps = [m.get('step', i) for i, m in enumerate(val_history)]
    val_losses = [m.get('loss', 0) for m in val_history]

    axes[0, 0].plot(steps, val_losses, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Validation Loss')
    axes[0, 0].set_title('Validation Loss')
    axes[0, 0].grid(True, alpha=0.3)

    for k in [10, 20]:
        recall_key = f'recall@{k}'
        recalls = [m.get(recall_key, 0) for m in val_history]
        if any(r != 0 for r in recalls):
            axes[0, 1].plot(steps, recalls, linewidth=2, marker='s', markersize=4, label=f'Recall@{k}')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Recall@K')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    for k in [10, 20]:
        ndcg_key = f'ndcg@{k}'
        ndcgs = [m.get(ndcg_key, 0) for m in val_history]
        if any(n != 0 for n in ndcgs):
            axes[1, 0].plot(steps, ndcgs, linewidth=2, marker='^', markersize=4, label=f'NDCG@{k}')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('NDCG')
    axes[1, 0].set_title('NDCG@K')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    train_losses = [m.get('loss', 0) for m in train_history]
    if train_losses:
        train_steps = list(range(len(train_losses)))
        axes[1, 1].plot(train_steps, train_losses, 'g-', linewidth=2, label='Train Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Training Loss')
        axes[1, 1].set_title('Training Loss per Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {plot_path}")


def get_gpu_memory_info() -> Dict[str, float]:
    """获取GPU显存信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}


class Trainer:
    """
    训练器类

    封装训练和验证的所有逻辑，包括：
    - 模型管理
    - 优化器管理
    - 学习率调度
    - 梯度累积
    - 训练循环
    - 验证循环
    - 混合精度训练 (AMP)
    - 检查点保存（仅保留 latest 和 best）
    - 指标记录与可视化
    """

    def __init__(
        self,
        model: VLA4RecBaseline,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        warmup_steps: int = 100,
        gradient_clip_val: float = 1.0,
        padding_id: int = 0,
        log_interval: int = 50,
        eval_interval: int = 500,
        save_dir: str = "./checkpoints",
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        use_neg_sampling: bool = False,
        num_neg_samples: int = 50
    ):
        """
        初始化训练器
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.padding_id = padding_id
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        self.run_dir = create_run_directory(save_dir)
        self.save_dir = self.run_dir
        print(f"Run directory: {self.run_dir}")

        self.gradient_clip_val = gradient_clip_val
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_neg_sampling = use_neg_sampling

        # 优化器：AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 学习率调度器：带预热的余弦衰减
        self.scheduler = self._create_scheduler(warmup_steps)

        # 损失函数
        vocab_size = model.vocab_size
        if use_neg_sampling and vocab_size > 100000:
            print(f"\n[优化] 使用负采样损失函数，负样本数: {num_neg_samples}")
            self.criterion = NegSamplingLoss(
                vocab_size=vocab_size,
                num_neg_samples=num_neg_samples,
                padding_idx=padding_id
            )
        else:
            print(f"\n[优化] 使用标准 CrossEntropyLoss (vocab_size={vocab_size:,})")
            self.criterion = nn.CrossEntropyLoss(ignore_index=padding_id, reduction='mean')

        # 混合精度训练
        if self.use_mixed_precision and device.type == 'cuda':
            print("[优化] 开启混合精度训练 (AMP)")
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_metric = {}

        # 统计信息
        self.train_history = []
        self.val_history = []

        # 打印显存信息
        if device.type == 'cuda':
            print(f"\n[显存] 初始显存占用: {get_gpu_memory_info()['allocated']:.2f} GB")

    def _create_scheduler(self, warmup_steps: int):
        """创建学习率调度器"""
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(max(1, warmup_steps))
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict) -> float:
        """执行单个训练步骤"""
        self.model.train()

        input_seq = batch['input_seq'].to(self.device)
        targets = batch['target'].to(self.device)

        # 混合精度前向传播
        if self.scaler is not None:
            with autocast():
                output = self.model(input_seq)
                loss = self.criterion(output, targets)
                loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
        else:
            output = self.model(input_seq)
            loss = self.criterion(output, targets)
            loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_val
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

        return loss.item() * self.gradient_accumulation_steps

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.current_epoch = epoch
        epoch_loss = 0.0
        num_batches = 0

        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.train_step(batch)
            epoch_loss += loss
            num_batches += 1
            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                current_lr = self.optimizer.param_groups[0]['lr']

                mem_info = get_gpu_memory_info()
                print(
                    f"Epoch [{epoch}] "
                    f"Step [{self.global_step}] "
                    f"Loss: {loss:.4f} "
                    f"Avg Loss: {avg_loss:.4f} "
                    f"LR: {current_lr:.6f} "
                    f"GPU: {mem_info['allocated']:.2f}GB"
                )

            if self.global_step % self.eval_interval == 0 and self.global_step > 0:
                val_metrics = self.validate()
                val_metrics['step'] = self.global_step
                self.val_history.append(val_metrics)
                print(f"\n>> Validation at Step {self.global_step}:")
                print_metrics(val_metrics, prefix="Validation")
                print()

                # 保存 latest checkpoint
                self.save_checkpoint("latest.pt")

                # 保存 best checkpoint
                val_loss = val_metrics.get('loss', float('inf'))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_metric = val_metrics.copy()
                    self.save_checkpoint("best.pt")
                    print(f">> Best model saved! Val Loss: {val_loss:.4f}\n")

        avg_epoch_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start_time

        epoch_metrics = {
            'loss': avg_epoch_loss,
            'time': epoch_time,
            'num_batches': num_batches
        }

        self.train_history.append(epoch_metrics)

        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """在验证集上评估模型"""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_seq = batch['input_seq'].to(self.device)
            targets = batch['target'].to(self.device)

            # 混合精度验证
            if self.scaler is not None:
                with autocast():
                    output = self.model(input_seq)
                    loss = self.criterion(output, targets)
            else:
                output = self.model(input_seq)
                loss = self.criterion(output, targets)

            total_loss += loss.item()
            num_batches += 1

        avg_val_loss = total_loss / num_batches

        eval_metrics = evaluate_model(
            self.model,
            self.val_loader,
            self.device,
            padding_id=self.padding_id,
            k_values=(10, 20)
        )

        eval_metrics['loss'] = avg_val_loss

        return eval_metrics

    def save_checkpoint(self, filename: str):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_metric': self.best_metric,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载模型检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_metric = checkpoint.get('best_metric', {})
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Checkpoint loaded from {filepath}")

    def save_results(self):
        """保存训练结果"""
        # 保存验证指标历史到CSV
        if self.val_history:
            val_df = pd.DataFrame(self.val_history)
            val_csv_path = os.path.join(self.save_dir, 'validation_metrics.csv')
            val_df.to_csv(val_csv_path, index=False)
            print(f"Validation metrics saved to {val_csv_path}")

        # 保存训练指标历史到CSV
        if self.train_history:
            train_df = pd.DataFrame(self.train_history)
            train_csv_path = os.path.join(self.save_dir, 'training_metrics.csv')
            train_df.to_csv(train_csv_path, index=False)
            print(f"Training metrics saved to {train_csv_path}")

        # 绘制训练曲线
        plot_training_curves(self.train_history, self.val_history, self.save_dir)

        # 保存最佳模型指标摘要
        if self.best_metric:
            summary_path = os.path.join(self.save_dir, 'best_model_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("Best Model Summary\n")
                f.write("=" * 50 + "\n")
                f.write(f"Best Validation Loss: {self.best_val_loss:.4f}\n\n")
                f.write("Metrics at Best Checkpoint:\n")
                for key, value in self.best_metric.items():
                    if key != 'step':
                        f.write(f"  {key}: {value}\n")
            print(f"Best model summary saved to {summary_path}")

    def full_train(
        self,
        num_epochs: int = 10,
        eval_interval: int = 1000
    ):
        """
        完整的训练流程

        Args:
            num_epochs: 训练轮数
            eval_interval: 评估间隔
        """
        self.eval_interval = eval_interval

        print("\n" + "=" * 60)
        print("开始训练!")
        print("=" * 60)

        for epoch in range(1, num_epochs + 1):
            print(f"\n>>> Epoch {epoch}/{num_epochs}")
            print("-" * 40)

            # 训练一个 epoch
            train_metrics = self.train_epoch(epoch)

            print(f"\n[Epoch {epoch}] Training completed:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Time: {train_metrics['time']:.2f}s")

            # 验证
            print(f"\n[Epoch {epoch}] Validating...")
            val_metrics = self.validate()
            val_metrics['step'] = self.global_step
            self.val_history.append(val_metrics)

            print(f"\n[Epoch {epoch}] Validation results:")
            print_metrics(val_metrics, prefix="Validation")

            # 保存 latest checkpoint（每个epoch结束）
            self.save_checkpoint("latest.pt")

            # 检查是否是最佳模型
            val_loss = val_metrics.get('loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_metric = val_metrics.copy()
                self.save_checkpoint("best.pt")
                print(f">> Best model saved! Val Loss: {val_loss:.4f}")

            # 每个epoch后清理显存
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)


def train(
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    max_seq_len: int = 50,
    vocab_size: int = 1536416,
    num_users: int = 1000,
    warmup_steps: int = 100,
    gradient_clip_val: float = 1.0,
    log_interval: int = 50,
    eval_interval: int = 1000,
    save_dir: str = "./checkpoints",
    use_mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
    use_neg_sampling: bool = False,
    num_neg_samples: int = 50,
    use_weight_tying: bool = True
):
    """
    主训练函数

    一键启动训练流程。
    """
    print("=" * 60)
    print("VLA4Rec Baseline Training (大规模稀疏ID优化版)")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()

    # 尝试加载词表
    data_dir = 'spotify_data'
    try:
        vocab = Vocab(os.path.join(data_dir, 'uri_to_id.json'))
        vocab_size = len(vocab)
        print(f"从词表文件加载: vocab_size = {vocab_size:,}", flush=True)
    except Exception as e:
        print(f"无法加载词表，使用默认 vocab_size: {vocab_size:,}", flush=True)
        import traceback
        traceback.print_exc()

    # 创建数据加载器
    trajectories_file = os.path.join(data_dir, 'spotify_trajectories.txt')
    vocab_file = os.path.join(data_dir, 'uri_to_id.json')
    if os.path.exists(trajectories_file):
        print(f"\n使用 Spotify 真实数据: {trajectories_file}", flush=True)
        train_loader, val_loader = create_spotify_dataloaders(
            trajectories_file=trajectories_file,
            vocab_file=vocab_file,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            train_ratio=0.8,
            cache_val_samples=True
        )
    else:
        print("\n使用虚拟数据")
        train_loader, val_loader = get_dummy_dataloaders(
            num_users=num_users,
            vocab_size=vocab_size,
            batch_size=batch_size,
            max_seq_len=max_seq_len
        )

    print()

    # 创建模型
    print("Creating model...")
    model, config = create_baseline_model(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        device=device,
        use_weight_tying=use_weight_tying
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()

    # 打印显存使用估算
    print_model_memory_usage(model, batch_size, max_seq_len, str(device))
    print()

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        gradient_clip_val=gradient_clip_val,
        log_interval=log_interval,
        eval_interval=eval_interval,
        save_dir=save_dir,
        use_mixed_precision=use_mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_neg_sampling=use_neg_sampling,
        num_neg_samples=num_neg_samples
    )

    # 训练
    trainer.full_train(num_epochs=num_epochs, eval_interval=eval_interval)

    # 保存结果
    print("\nSaving results...")
    trainer.save_results()
    print(f"\nAll results saved to: {trainer.run_dir}")

    return trainer


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLA4Rec Baseline Training (大规模稀疏ID优化版)")

    # 数据参数
    parser.add_argument("--num_users", type=int, default=1000, help="Number of simulated users")
    parser.add_argument("--vocab_size", type=int, default=1536416, help="Vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=50, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    # 模型参数
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--use_weight_tying", action="store_true", default=True,
                        help="Use weight tying between embedding and output layer")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping threshold")

    # 优化参数
    parser.add_argument("--use_mixed_precision", action="store_true", default=True,
                        help="Use mixed precision training (AMP)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--use_neg_sampling", action="store_true", default=False,
                        help="Use negative sampling loss for large vocab")
    parser.add_argument("--num_neg_samples", type=int, default=50,
                        help="Number of negative samples for each positive")

    # 其他参数
    parser.add_argument("--log_interval", type=int, default=50, help="Log interval")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Evaluation interval")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Save directory")

    args = parser.parse_args()

    # 启动训练
    train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        num_users=args.num_users,
        warmup_steps=args.warmup_steps,
        gradient_clip_val=args.gradient_clip_val,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_dir=args.save_dir,
        use_mixed_precision=args.use_mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_neg_sampling=args.use_neg_sampling,
        num_neg_samples=args.num_neg_samples,
        use_weight_tying=args.use_weight_tying
    )
