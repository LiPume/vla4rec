"""
================================================================================
VLA4Rec Baseline - 主训练循环
================================================================================

本模块实现了完整的训练和验证流程，包括：
- 数据加载
- 模型实例化
- 优化器配置
- 训练循环
- 验证循环
- 指标评估

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
from typing import Optional, Tuple, Dict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入本地模块
from .dataset import get_dummy_dataloaders, PlaylistDataset, create_dataloaders
from .model import VLA4RecBaseline, create_baseline_model
from .metrics import evaluate_model, print_metrics


class Trainer:
    """
    训练器类

    封装训练和验证的所有逻辑，包括：
    - 模型管理
    - 优化器管理
    - 学习率调度
    - 训练循环
    - 验证循环
    - 检查点保存
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
        save_dir: str = "./checkpoints"
    ):
        """
        初始化训练器

        Args:
            model: VLA4Rec 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 计算设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_steps: 预热步数
            gradient_clip_val: 梯度裁剪阈值
            padding_id: padding 的 ID（用于忽略 padding loss）
            log_interval: 日志打印间隔（步数）
            eval_interval: 验证间隔（步数）
            save_dir: 检查点保存目录
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.padding_id = padding_id
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.gradient_clip_val = gradient_clip_val

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

        # 损失函数：CrossEntropyLoss
        # 忽略 padding 的 loss（将 padding 位置的 loss 设为 0）
        self.criterion = nn.CrossEntropyLoss(ignore_index=padding_id, reduction='mean')

        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 统计信息
        self.train_history = []
        self.val_history = []

    def _create_scheduler(self, warmup_steps: int):
        """创建学习率调度器"""
        # 使用 Linear Warmup + Cosine Decay
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # 线性预热
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # 余弦衰减
                progress = float(current_step - warmup_steps) / float(max(1, warmup_steps))
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict) -> float:
        """
        执行单个训练步骤

        Args:
            batch: 包含 'input_seq' 和 'target' 的批次

        Returns:
            loss: 当前步骤的损失值
        """
        self.model.train()

        # 将数据移动到设备
        input_seq = batch['input_seq'].to(self.device)  # [batch_size, seq_len]
        targets = batch['target'].to(self.device)  # [batch_size]

        # 前向传播
        # output: [batch_size, vocab_size]
        output = self.model(input_seq)

        # 计算损失
        # 注意：这里我们只预测最后一个位置的歌曲
        # 如果模型输出是 [batch_size, vocab_size]，直接与 targets 计算
        loss = self.criterion(output, targets)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_val
            )

        # 更新参数
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个 epoch

        Args:
            epoch: 当前 epoch 编号

        Returns:
            epoch_metrics: 当前 epoch 的平均指标
        """
        self.current_epoch = epoch
        epoch_loss = 0.0
        num_batches = 0

        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # 训练步骤
            loss = self.train_step(batch)
            epoch_loss += loss
            num_batches += 1
            self.global_step += 1

            # 打印日志
            if self.global_step % self.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                current_lr = self.optimizer.param_groups[0]['lr']

                print(
                    f"Epoch [{epoch}] "
                    f"Step [{self.global_step}] "
                    f"Loss: {loss:.4f} "
                    f"Avg Loss: {avg_loss:.4f} "
                    f"LR: {current_lr:.6f}"
                )

            # 定期验证
            if self.global_step % self.eval_interval == 0 and self.global_step > 0:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)
                print(f"\n>> Validation at Step {self.global_step}:")
                print_metrics(val_metrics, prefix="Validation")
                print()

                # 保存最佳模型
                val_loss = val_metrics.get('loss', float('inf'))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
                    print(f">> Best model saved! Val Loss: {val_loss:.4f}\n")

        # 计算 epoch 平均指标
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
        """
        在验证集上评估模型

        Returns:
            val_metrics: 验证指标字典
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            # 获取数据
            input_seq = batch['input_seq'].to(self.device)
            targets = batch['target'].to(self.device)

            # 前向传播
            output = self.model(input_seq)

            # 计算损失
            loss = self.criterion(output, targets)

            total_loss += loss.item()
            num_batches += 1

        # 平均损失
        avg_val_loss = total_loss / num_batches

        # 计算推荐指标
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
            'train_history': self.train_history,
            'val_history': self.val_history
        }

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
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])

        print(f"Checkpoint loaded from {filepath}")


def train(
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    max_seq_len: int = 50,
    vocab_size: int = 1000,
    num_users: int = 1000,
    warmup_steps: int = 100,
    gradient_clip_val: float = 1.0,
    log_interval: int = 50,
    eval_interval: int = 500,
    save_dir: str = "./checkpoints"
):
    """
    主训练函数

    一键启动训练流程。
    """
    print("=" * 60)
    print("VLA4Rec Baseline 训练")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()

    # 生成虚拟数据
    print("准备数据...")
    train_loader, val_loader = get_dummy_dataloaders(
        num_users=num_users,
        vocab_size=vocab_size,
        batch_size=batch_size,
        max_seq_len=max_seq_len
    )
    print()

    # 创建模型
    print("创建模型...")
    model, config = create_baseline_model(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        device=device
    )

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()

    # 打印配置
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  num_epochs: {num_epochs}")
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
        save_dir=save_dir
    )

    # 训练循环
    print("开始训练...")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        print(f"\n>>> Epoch {epoch}/{num_epochs}")
        print("-" * 40)

        # 训练一个 epoch
        train_metrics = trainer.train_epoch(epoch)

        print(f"\n[Epoch {epoch}] 训练完成:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  时间: {train_metrics['time']:.2f}s")

        # 验证
        print("\n[Epoch {epoch}] 验证中...")
        val_metrics = trainer.validate()
        trainer.val_history.append(val_metrics)

        print(f"\n[Epoch {epoch}] 验证结果:")
        print_metrics(val_metrics, prefix="Validation")

        # 保存检查点
        checkpoint_path = os.path.join(save_dir, f"epoch_{epoch}.pt")
        trainer.save_checkpoint(f"epoch_{epoch}.pt")

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLA4Rec Baseline 训练")

    # 数据参数
    parser.add_argument("--num_users", type=int, default=1000, help="模拟用户数")
    parser.add_argument("--vocab_size", type=int, default=1000, help="词表大小")
    parser.add_argument("--max_seq_len", type=int, default=50, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")

    # 模型参数
    parser.add_argument("--hidden_size", type=int, default=64, help="隐向量维度")
    parser.add_argument("--num_layers", type=int, default=2, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="梯度裁剪阈值")

    # 其他参数
    parser.add_argument("--log_interval", type=int, default=50, help="日志打印间隔")
    parser.add_argument("--eval_interval", type=int, default=500, help="验证间隔")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="保存目录")

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
        save_dir=args.save_dir
    )
