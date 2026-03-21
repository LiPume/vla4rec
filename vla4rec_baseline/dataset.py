"""
================================================================================
VLA4Rec Baseline - 数据流水线模块
================================================================================

本模块实现了基于滑动窗口的推荐数据切分逻辑，将高质量播放列表转化为
可用于因果Transformer进行Next-Token Prediction的训练样本。

核心思想：
- 推荐任务被建模为智能体的行为克隆问题
- 历史歌曲序列（Semantic IDs）构成状态 (State)
- 预测的下一首歌构成动作 (Action)
- 使用滑动窗口切分播放列表，生成 (输入序列, 目标) 样本对

Author: VLA4Rec Team
Date: 2026-03-21
================================================================================
"""

import random
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class PlaylistDataset(Dataset):
    """
    播放列表数据集类

    该数据集接收用户的高质量播放列表列表（每个播放列表是一个歌曲ID序列），
    使用滑动窗口机制将其切分为多个训练样本。

    滑动窗口切分示例（max_seq_len=3, padding_id=0）：
        原始播放列表: [1, 2, 3, 4, 5]
        切分结果:
            输入序列: [0, 0, 1] -> 目标: 2
            输入序列: [0, 1, 2] -> 目标: 3
            输入序列: [1, 2, 3] -> 目标: 4
            输入序列: [2, 3, 4] -> 目标: 5
    """

    def __init__(
        self,
        playlists: List[List[int]],
        max_seq_len: int = 50,
        padding_id: int = 0,
        min_playlist_len: int = 2
    ):
        """
        初始化数据集

        Args:
            playlists: 用户播放列表列表，每个子列表是一个播放列表（歌曲ID序列）
            max_seq_len: 最大序列长度（输入序列的长度）
            padding_id: 用于填充的ID（默认为0，在语义ID体系中0通常表示padding或特殊符号）
            min_playlist_len: 播放列表最小长度，短于此长度的播放列表将被跳过
        """
        self.max_seq_len = max_seq_len
        self.padding_id = padding_id

        # 调用滑动窗口切分逻辑，生成所有训练样本
        # 每个样本格式: (input_sequence, target_id)
        self.samples = self._create_sliding_window_samples(
            playlists, min_playlist_len
        )

    def _create_sliding_window_samples(
        self,
        playlists: List[List[int]],
        min_playlist_len: int
    ) -> List[Tuple[List[int], int]]:
        """
        使用滑动窗口切分播放列表

        对于每个播放列表 [s1, s2, s3, ..., sn]，生成以下样本：
            - 窗口长度 = max_seq_len
            - 窗口从左到右滑动，每次移动1步
            - 最后一个位置的歌曲ID作为预测目标

        Args:
            playlists: 播放列表列表
            min_playlist_len: 最小播放列表长度

        Returns:
            样本列表，每个元素为 (input_seq, target_id) 元组
        """
        samples = []

        for playlist in playlists:
            # 过滤过短的播放列表（至少需要2首歌才能生成1个有效样本）
            if len(playlist) < min_playlist_len:
                continue

            # 对每个有效位置生成一个样本
            # i 是窗口最后一个位置（目标位置）
            for i in range(1, len(playlist)):
                # 计算窗口起始位置，确保窗口长度不超过 max_seq_len
                start_idx = max(0, i - self.max_seq_len)

                # 提取输入序列（窗口内容，可能前面用padding填充）
                input_seq = playlist[start_idx:i]

                # 左侧填充 padding_id 使序列达到 max_seq_len 长度
                padding_len = self.max_seq_len - len(input_seq)
                input_seq = [self.padding_id] * padding_len + input_seq

                # 目标ID：窗口最后一个位置的下一首歌
                target_id = playlist[i]

                samples.append((input_seq, target_id))

        return samples

    def __len__(self) -> int:
        """返回数据集中样本的数量"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            包含 'input_seq' 和 'target' 的字典，类型为 torch.Tensor
            - input_seq: [max_seq_len] 形状的LongTensor，表示输入序列
            - target: [] 形状的LongTensor（标量），表示目标歌曲ID
        """
        input_seq, target_id = self.samples[idx]

        return {
            'input_seq': torch.tensor(input_seq, dtype=torch.long),
            'target': torch.tensor(target_id, dtype=torch.long)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    自定义批处理整理函数（Collate Function）

    将 DataLoader 收集的单个样本整理成一个批次。
    由于我们使用固定长度的输入序列（通过padding实现），这里直接stack即可。

    Args:
        batch: 样本列表，每个样本是 {'input_seq': Tensor, 'target': Tensor}

    Returns:
        批次字典，包含:
            - input_seq: [batch_size, max_seq_len]
            - target: [batch_size]
            - target_mask: [batch_size, max_seq_len] (用于标识哪些位置是有效目标)
    """
    # Stack操作将列表中的tensor堆叠成更高维度的tensor
    # input_seq: list of [max_seq_len] -> [batch_size, max_seq_len]
    # target: list of [] -> [batch_size]
    input_seq = torch.stack([item['input_seq'] for item in batch])
    target = torch.stack([item['target'] for item in batch])

    return {
        'input_seq': input_seq,
        'target': target
    }


def create_dataloaders(
    playlists: List[List[int]],
    batch_size: int = 32,
    max_seq_len: int = 50,
    train_ratio: float = 0.8,
    padding_id: int = 0,
    num_workers: int = 4,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练集和验证集的DataLoader

    Args:
        playlists: 播放列表列表
        batch_size: 批次大小
        max_seq_len: 最大序列长度
        train_ratio: 训练集比例（剩余部分作为验证集）
        padding_id: padding的ID值
        num_workers: DataLoader的工作进程数
        shuffle: 是否打乱数据

    Returns:
        (train_loader, val_loader) 元组
    """
    # 随机打乱播放列表顺序
    if shuffle:
        random.seed(42)  # 设置种子以保证可复现性
        random.shuffle(playlists)

    # 划分训练集和验证集
    split_idx = int(len(playlists) * train_ratio)
    train_playlists = playlists[:split_idx]
    val_playlists = playlists[split_idx:]

    # 创建数据集
    train_dataset = PlaylistDataset(
        playlists=train_playlists,
        max_seq_len=max_seq_len,
        padding_id=padding_id
    )
    val_dataset = PlaylistDataset(
        playlists=val_playlists,
        max_seq_len=max_seq_len,
        padding_id=padding_id
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True  # 加速GPU数据传输
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


def get_dummy_dataloaders(
    num_users: int = 1000,
    vocab_size: int = 1000,
    avg_playlist_len: int = 30,
    max_seq_len: int = 50,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    生成虚拟数据用于快速测试

    该函数生成模拟的用户播放列表数据，用于：
    1. 快速验证模型架构是否正确
    2. 测试训练循环是否能正常运行
    3. 调试数据处理流程

    生成逻辑：
    - 每个用户生成一个播放列表
    - 播放列表长度服从正态分布（均值=avg_playlist_len，标准差=5）
    - 歌曲ID从0到vocab_size-1中选择
    - 通过马尔可夫链模拟用户的音乐偏好（相邻歌曲有一定相关性）

    Args:
        num_users: 模拟的用户数量
        vocab_size: 歌曲ID的范围（词表大小）
        avg_playlist_len: 平均播放列表长度
        max_seq_len: 最大序列长度（与模型保持一致）
        batch_size: DataLoader的批次大小

    Returns:
        (train_loader, val_loader) 元组
    """
    print(f"=" * 60)
    print(f"生成虚拟数据: {num_users} 个用户, 词表大小: {vocab_size}")
    print(f"=" * 60)

    playlists = []

    for user_id in range(num_users):
        # 随机生成长度
        playlist_len = max(5, int(np.random.normal(avg_playlist_len, 5)))

        # 生成播放列表（使用简单的自回归模型模拟用户偏好）
        playlist = []

        # 第一个歌曲随机选择
        first_song = random.randint(1, vocab_size - 1)  # 从1开始，0保留为padding
        playlist.append(first_song)

        # 后续歌曲以一定概率与前一首相关（模拟流派/风格偏好）
        for _ in range(playlist_len - 1):
            if random.random() < 0.7:
                # 70%概率：选择与当前歌曲"相似"的歌曲
                # 通过加一个小随机数模拟
                next_song = (playlist[-1] + random.randint(-50, 50)) % (vocab_size - 1) + 1
            else:
                # 30%概率：随机探索新歌曲
                next_song = random.randint(1, vocab_size - 1)

            # 确保歌曲ID有效（避免与padding冲突）
            next_song = max(1, min(next_song, vocab_size - 1))
            playlist.append(next_song)

        playlists.append(playlist)

    print(f"生成完成! 共有 {len(playlists)} 个播放列表")

    # 统计信息
    total_songs = sum(len(p) for p in playlists)
    avg_len = total_songs / len(playlists) if playlists else 0
    print(f"平均播放列表长度: {avg_len:.2f}")

    # 创建DataLoader
    train_loader, val_loader = create_dataloaders(
        playlists=playlists,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        train_ratio=0.8,
        padding_id=0,
        num_workers=0,  # 测试时用0避免多进程问题
        shuffle=True
    )

    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"=" * 60)

    return train_loader, val_loader


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("测试滑动窗口切分逻辑")
    print("=" * 60)

    # 测试用例
    test_playlist = [1, 2, 3, 4, 5]
    max_seq_len = 3
    padding_id = 0

    dataset = PlaylistDataset(
        playlists=[test_playlist],
        max_seq_len=max_seq_len,
        padding_id=padding_id
    )

    print(f"\n原始播放列表: {test_playlist}")
    print(f"最大序列长度: {max_seq_len}")
    print(f"\n切分后的样本:")

    for i, sample in enumerate(dataset):
        input_seq = sample['input_seq'].tolist()
        target = sample['target'].item()
        print(f"  样本 {i+1}: 输入 {input_seq} -> 目标 {target}")

    print("\n" + "=" * 60)
    print("测试虚拟数据生成")
    print("=" * 60)

    # 测试虚拟数据
    train_loader, val_loader = get_dummy_dataloaders(
        num_users=100,
        vocab_size=100,
        batch_size=8
    )

    # 遍历一个批次验证
    print("\n测试DataLoader批次:")
    for batch in train_loader:
        print(f"  input_seq shape: {batch['input_seq'].shape}")
        print(f"  target shape: {batch['target'].shape}")
        print(f"  input_seq dtype: {batch['input_seq'].dtype}")
        print(f"  target dtype: {batch['target'].dtype}")
        break

    print("\n" + "=" * 60)
    print("数据流水线测试通过!")
    print("=" * 60)
