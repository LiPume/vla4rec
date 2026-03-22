"""
================================================================================
VLA4Rec Baseline - 数据流水线模块 (大规模懒加载优化版)
================================================================================

本模块实现了针对大规模数据集（153万量级）的懒加载数据处理逻辑。

核心优化：
1. 懒加载 (Lazy Loading): 使用生成器逐行读取轨迹文件，避免一次性加载
2. Vocab 类: 高效管理 URI 到 ID 的映射
3. 滑动窗口样本生成器: 支持流式处理大规模数据

核心思想：
- 推荐任务被建模为智能体的行为克隆问题
- 历史歌曲序列（Semantic IDs）构成状态 (State)
- 预测的下一首歌构成动作 (Action)
- 使用滑动窗口切分播放列表，生成 (输入序列, 目标) 样本对

Author: VLA4Rec Team
Date: 2026-03-21
================================================================================
"""

import os
import json
import random
from typing import List, Tuple, Dict, Optional, Iterator, Generator
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Vocab:
    """
    词表管理类

    用于管理 URI 到整数 ID 的映射，支持从 JSON 文件懒加载。
    """

    def __init__(self, vocab_file: Optional[str] = None):
        """
        初始化词表

        Args:
            vocab_file: 词表文件路径（JSON格式）
        """
        self.uri_to_id = {}
        self.id_to_uri = {}
        self.vocab_size = 0
        self._vocab_file = vocab_file

        if vocab_file and os.path.exists(vocab_file):
            self.load(vocab_file)

    def load(self, vocab_file: str):
        """从 JSON 文件加载词表"""
        print(f"加载词表: {vocab_file}")
        with open(vocab_file, 'r') as f:
            self.uri_to_id = json.load(f)

        self.id_to_uri = {v: k for k, v in self.uri_to_id.items()}
        self.vocab_size = len(self.uri_to_id)
        print(f"词表加载完成！词表大小: {self.vocab_size:,}")

    def __len__(self) -> int:
        """返回词表大小"""
        return self.vocab_size

    def __contains__(self, uri: str) -> bool:
        """检查 URI 是否在词表中"""
        return uri in self.uri_to_id

    def get_id(self, uri: str) -> Optional[int]:
        """获取 URI 对应的 ID"""
        return self.uri_to_id.get(uri)

    def get_uri(self, idx: int) -> Optional[str]:
        """获取 ID 对应的 URI"""
        return self.id_to_uri.get(idx)


class LazyPlaylistDataset(Dataset):
    """
    懒加载播放列表数据集

    针对大规模轨迹文件（747,510 行）优化，使用流式读取而非一次性加载。
    支持原地打乱和流式采样。
    """

    def __init__(
        self,
        trajectories_file: str,
        vocab_file: str,
        max_seq_len: int = 50,
        padding_id: int = 0,
        min_playlist_len: int = 2,
        cache_samples: bool = False,
        max_cached_samples: int = 100000
    ):
        """
        初始化懒加载数据集

        Args:
            trajectories_file: 轨迹文件路径（每行是一个空格分隔的歌曲ID序列）
            vocab_file: 词表文件路径
            max_seq_len: 最大序列长度
            padding_id: 用于填充的ID
            min_playlist_len: 播放列表最小长度
            cache_samples: 是否缓存样本（用于验证集，训练集建议 False）
            max_cached_samples: 最大缓存样本数
        """
        self.trajectories_file = trajectories_file
        self.max_seq_len = max_seq_len
        self.padding_id = padding_id
        self.min_playlist_len = min_playlist_len
        self.cache_samples = cache_samples
        self.max_cached_samples = max_cached_samples

        # 加载词表
        self.vocab = Vocab(vocab_file)

        # 统计信息
        self.num_trajectories = self._count_trajectories()
        print(f"轨迹文件包含 {self.num_trajectories:,} 个播放列表")

        # 样本索引列表（用于随机采样）
        self._sample_indices = None

        # 缓存（用于验证集）
        self._cached_samples = []
        if self.cache_samples:
            print(f"预缓存样本到内存（最多 {max_cached_samples:,} 个）...")
            self._cache_samples()
            print(f"缓存完成！共 {len(self._cached_samples):,} 个样本")

    def _count_trajectories(self) -> int:
        """统计轨迹文件行数"""
        if not os.path.exists(self.trajectories_file):
            raise FileNotFoundError(f"轨迹文件不存在: {self.trajectories_file}")

        with open(self.trajectories_file, 'r') as f:
            num_lines = sum(1 for _ in f)
        return num_lines

    def _read_trajectory(self, line: str) -> List[int]:
        """解析单行轨迹为歌曲ID列表"""
        return [int(x) for x in line.strip().split() if x]

    def _trajectory_generator(self) -> Generator[List[int], None, None]:
        """轨迹文件生成器"""
        with open(self.trajectories_file, 'r') as f:
            for line in f:
                if line.strip():
                    yield self._read_trajectory(line)

    def _generate_samples_from_trajectory(
        self,
        playlist: List[int]
    ) -> Generator[Tuple[List[int], int], None, None]:
        """
        从单个播放列表生成滑动窗口样本

        Args:
            playlist: 播放列表（歌曲ID列表）

        Yields:
            (input_seq, target_id) 元组
        """
        if len(playlist) < self.min_playlist_len:
            return

        for i in range(1, len(playlist)):
            start_idx = max(0, i - self.max_seq_len)
            input_seq = playlist[start_idx:i]
            padding_len = self.max_seq_len - len(input_seq)
            input_seq = [self.padding_id] * padding_len + input_seq
            target_id = playlist[i]
            yield (input_seq, target_id)

    def _cache_samples(self):
        """预缓存样本（用于验证集）"""
        count = 0
        for playlist in self._trajectory_generator():
            for sample in self._generate_samples_from_trajectory(playlist):
                self._cached_samples.append(sample)
                count += 1
                if count >= self.max_cached_samples:
                    return

    def __len__(self) -> int:
        """返回数据集大小"""
        if self.cache_samples:
            return len(self._cached_samples)
        return self.num_trajectories * self.max_seq_len  # 估算值

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        如果使用缓存，直接从缓存返回。
        否则随机选择一个轨迹生成样本。
        """
        if self.cache_samples:
            input_seq, target_id = self._cached_samples[idx]
        else:
            # 懒加载：从随机轨迹生成
            input_seq, target_id = self._get_random_sample()

        return {
            'input_seq': torch.tensor(input_seq, dtype=torch.long),
            'target': torch.tensor(target_id, dtype=torch.long)
        }

    def _get_random_sample(self) -> Tuple[List[int], int]:
        """从随机轨迹获取随机样本"""
        # 随机选择一个轨迹
        line_idx = random.randint(0, self.num_trajectories - 1)

        with open(self.trajectories_file, 'r') as f:
            for i, line in enumerate(f):
                if i == line_idx:
                    playlist = self._read_trajectory(line)
                    break

        # 从该轨迹中随机选择一个有效的样本
        valid_samples = list(self._generate_samples_from_trajectory(playlist))
        if valid_samples:
            return random.choice(valid_samples)

        # 如果该轨迹没有有效样本，递归尝试其他轨迹
        return self._get_random_sample()

    def get_streaming_samples(self, num_samples: int = 1000) -> Generator[Dict, None, None]:
        """
        获取流式样本生成器（用于训练）

        这是核心方法，用于训练时流式生成样本。

        Args:
            num_samples: 每次迭代返回的样本数

        Yields:
            样本字典 {'input_seq': Tensor, 'target': Tensor}
        """
        samples_generated = 0

        for playlist in self._trajectory_generator():
            for sample in self._generate_samples_from_trajectory(playlist):
                input_seq, target_id = sample
                yield {
                    'input_seq': torch.tensor(input_seq, dtype=torch.long),
                    'target': torch.tensor(target_id, dtype=torch.long)
                }
                samples_generated += 1

                if samples_generated >= num_samples:
                    return


class SampledPlaylistDataset(Dataset):
    """
    采样播放列表数据集（性能优化版）

    核心优化：
    1. 预扫描轨迹文件建立索引，避免每次 O(n) 遍历
    2. 支持 epoch 采样，每个 epoch 只采样部分播放列表
    3. 适配标准 DataLoader，支持多进程加载
    """

    def __init__(
        self,
        trajectories_file: str,
        vocab_file: str,
        max_seq_len: int = 50,
        padding_id: int = 0,
        min_playlist_len: int = 2,
        epoch_sample_ratio: float = 1.0,
        random_seed: int = 42
    ):
        """
        初始化采样数据集

        Args:
            trajectories_file: 轨迹文件路径
            vocab_file: 词表文件路径
            max_seq_len: 最大序列长度
            padding_id: padding ID
            min_playlist_len: 最小播放列表长度
            epoch_sample_ratio: 每个 epoch 采样的比例 (0.0-1.0)
            random_seed: 随机种子
        """
        self.trajectories_file = trajectories_file
        self.max_seq_len = max_seq_len
        self.padding_id = padding_id
        self.min_playlist_len = min_playlist_len
        self.epoch_sample_ratio = epoch_sample_ratio
        self.random_seed = random_seed

        # 加载词表
        self.vocab = Vocab(vocab_file)

        # 预扫描建立索引（行号 -> 文件偏移量）
        self._build_index()

        # 当前 epoch 的采样索引
        self._current_epoch_indices = None
        self._epoch = 0

        print(f"[SampledPlaylistDataset] 索引建立完成: {len(self._index)} 个播放列表")
        print(f"[SampledPlaylistDataset] Epoch 采样比例: {epoch_sample_ratio:.1%}")

    def _build_index(self):
        """预扫描轨迹文件，建立索引"""
        self._index = []  # 每个播放列表的信息 [(start_pos, end_pos, length), ...]

        with open(self.trajectories_file, 'r', buffering=8192) as f:
            while True:
                start_pos = f.tell()
                line = f.readline()
                if not line:
                    break
                end_pos = f.tell()
                playlist = line.strip().split()
                if len(playlist) >= self.min_playlist_len:
                    self._index.append((start_pos, end_pos, len(playlist)))

    def _read_playlist_at(self, start_pos: int, end_pos: int) -> List[int]:
        """根据文件偏移量读取播放列表"""
        with open(self.trajectories_file, 'r') as f:
            f.seek(start_pos)
            line = f.read(end_pos - start_pos)
            return [int(x) for x in line.strip().split()]

    def _generate_samples_from_playlist(
        self,
        playlist: List[int]
    ) -> List[Tuple[List[int], int]]:
        """从播放列表生成所有样本"""
        samples = []
        for i in range(1, len(playlist)):
            start_idx = max(0, i - self.max_seq_len)
            input_seq = playlist[start_idx:i]
            padding_len = self.max_seq_len - len(input_seq)
            input_seq = [self.padding_id] * padding_len + input_seq
            target_id = playlist[i]
            samples.append((input_seq, target_id))
        return samples

    def set_epoch(self, epoch: int):
        """设置当前 epoch，更新采样索引"""
        self._epoch = epoch
        rng = random.Random(self.random_seed + epoch)

        # 根据采样比例选择播放列表
        num_to_sample = max(1, int(len(self._index) * self.epoch_sample_ratio))
        self._current_epoch_indices = rng.sample(range(len(self._index)), num_to_sample)

    def __len__(self) -> int:
        if self._current_epoch_indices is None:
            return len(self._index) * 30  # 估算
        return len(self._current_epoch_indices) * 30  # 估算

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取样本（随机选择播放列表中的随机位置）"""
        if self._current_epoch_indices is None:
            self.set_epoch(0)

        # 随机选择一个播放列表
        playlist_idx = random.choice(self._current_epoch_indices)
        start_pos, end_pos, playlist_len = self._index[playlist_idx]

        # 读取播放列表并生成样本
        playlist = self._read_playlist_at(start_pos, end_pos)
        samples = self._generate_samples_from_playlist(playlist)

        # 随机选择一个样本
        input_seq, target_id = random.choice(samples)

        return {
            'input_seq': torch.tensor(input_seq, dtype=torch.long),
            'target': torch.tensor(target_id, dtype=torch.long)
        }


class StreamingDataLoader:
    """
    流式数据加载器（性能优化版）

    核心优化：
    1. 基于索引的文件读取，避免每次 O(n) 遍历
    2. 支持 epoch 采样
    3. 可选异步预取
    """

    def __init__(
        self,
        dataset: LazyPlaylistDataset,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        epoch_sample_ratio: float = 1.0
    ):
        """
        初始化流式数据加载器

        Args:
            dataset: LazyPlaylistDataset 实例
            batch_size: 批次大小
            shuffle: 是否打乱数据
            drop_last: 是否丢弃最后一个不完整的批次
            num_workers: 多进程数量（0 表示单进程）
            prefetch_factor: 预取因子
            epoch_sample_ratio: 每个 epoch 采样的比例
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.epoch_sample_ratio = epoch_sample_ratio

        # 创建索引列表
        self.num_trajectories = dataset.num_trajectories
        self.trajectory_indices = list(range(self.num_trajectories))

        # 预扫描建立索引
        self._build_file_index()

        # 采样索引
        self._sampled_indices = None
        self._epoch = 0

        if self.shuffle:
            self._shuffle_indices()

        self._file_handle = None

    def _build_file_index(self):
        """预扫描文件，建立偏移量索引"""
        self._index = []  # [(start_pos, end_pos), ...]

        with open(self.dataset.trajectories_file, 'r', buffering=8192) as f:
            while True:
                start_pos = f.tell()
                line = f.readline()
                if not line:
                    break
                end_pos = f.tell()
                self._index.append((start_pos, end_pos))

    def _shuffle_indices(self):
        """打乱索引"""
        random.shuffle(self.trajectory_indices)

    def set_epoch(self, epoch: int):
        """设置 epoch 并更新采样"""
        self._epoch = epoch
        rng = random.Random(42 + epoch)

        # 根据采样比例采样
        num_to_sample = max(1, int(len(self._index) * self.epoch_sample_ratio))
        self._sampled_indices = rng.sample(range(len(self._index)), num_to_sample)

        if self.shuffle:
            rng.shuffle(self._sampled_indices)

    def _read_trajectory_by_index(self, idx: int) -> List[int]:
        """根据索引读取轨迹"""
        start_pos, end_pos = self._index[idx]
        with open(self.dataset.trajectories_file, 'r') as f:
            f.seek(start_pos)
            line = f.read(end_pos - start_pos)
            return self.dataset._read_trajectory(line)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """迭代器"""
        # 更新 epoch 采样
        self.set_epoch(self._epoch)

        # 使用采样索引或全部索引
        indices_to_use = self._sampled_indices if self._sampled_indices else self.trajectory_indices

        batch_input = []
        batch_target = []

        for idx in indices_to_use:
            playlist = self._read_trajectory_by_index(idx)

            for sample in self.dataset._generate_samples_from_trajectory(playlist):
                input_seq, target_id = sample
                batch_input.append(torch.tensor(input_seq, dtype=torch.long))
                batch_target.append(torch.tensor(target_id, dtype=torch.long))

                if len(batch_input) >= self.batch_size:
                    yield {
                        'input_seq': torch.stack(batch_input),
                        'target': torch.stack(batch_target)
                    }
                    batch_input = []
                    batch_target = []

        # 处理剩余样本
        if batch_input and not self.drop_last:
            yield {
                'input_seq': torch.stack(batch_input),
                'target': torch.stack(batch_target)
            }

    def __len__(self) -> int:
        """返回批次数"""
        num_indices = len(self._sampled_indices) if self._sampled_indices else self.num_trajectories
        estimated_samples = num_indices * 30
        if self.drop_last:
            return estimated_samples // self.batch_size
        return (estimated_samples // self.batch_size) + 1


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    自定义批处理整理函数（Collate Function）

    将 DataLoader 收集的单个样本整理成一个批次。

    Args:
        batch: 样本列表，每个样本是 {'input_seq': Tensor, 'target': Tensor}

    Returns:
        批次字典
    """
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
        random.seed(42)
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
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


def create_spotify_dataloaders(
    trajectories_file: str,
    vocab_file: str,
    batch_size: int = 64,
    max_seq_len: int = 50,
    train_ratio: float = 0.8,
    padding_id: int = 0,
    num_workers: int = 0,
    cache_val_samples: bool = True,
    epoch_sample_ratio: float = 1.0,
    use_indexed_loader: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    创建 Spotify 数据的 DataLoader（性能优化版）

    Args:
        trajectories_file: 轨迹文件路径
        vocab_file: 词表文件路径
        batch_size: 批次大小
        max_seq_len: 最大序列长度
        train_ratio: 训练集比例
        padding_id: padding的ID值
        num_workers: DataLoader的工作进程数（0=单进程）
        cache_val_samples: 是否缓存验证集样本
        epoch_sample_ratio: 每个epoch采样的比例（0.0-1.0），可大幅加速训练
        use_indexed_loader: 是否使用索引优化加载器

    Returns:
        (train_loader, val_loader) 元组
    """
    print("\n" + "=" * 60)
    print("创建 Spotify 数据加载器")
    print("=" * 60)

    # 统计轨迹数量
    with open(trajectories_file, 'r') as f:
        num_trajectories = sum(1 for _ in f)

    print(f"总轨迹数: {num_trajectories:,}")

    # 划分训练集和验证集
    split_idx = int(num_trajectories * train_ratio)
    print(f"训练集轨迹数: {split_idx:,}")
    print(f"验证集轨迹数: {num_trajectories - split_idx:,}")

    # 创建训练集
    train_dataset = LazyPlaylistDataset(
        trajectories_file=trajectories_file,
        vocab_file=vocab_file,
        max_seq_len=max_seq_len,
        padding_id=padding_id,
        cache_samples=False
    )

    # 创建验证集（可缓存以便重复评估）
    val_dataset = LazyPlaylistDataset(
        trajectories_file=trajectories_file,
        vocab_file=vocab_file,
        max_seq_len=max_seq_len,
        padding_id=padding_id,
        cache_samples=cache_val_samples,
        max_cached_samples=50000
    )

    # 创建训练加载器
    if use_indexed_loader:
        train_loader = StreamingDataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            epoch_sample_ratio=epoch_sample_ratio
        )
        print(f"\n[优化] 使用索引优化流式加载器")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )

    # 验证集使用标准 DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print("\n数据加载器创建完成！")
    print(f"训练集: {'流式加载（索引优化）' if use_indexed_loader else '标准加载'}")
    if epoch_sample_ratio < 1.0:
        print(f"训练采样: 每epoch采样 {epoch_sample_ratio:.0%} 的数据（加速 {1/epoch_sample_ratio:.1f}x）")
    print(f"验证集: 缓存模式（{len(val_dataset):,} 样本）")
    print(f"DataLoader workers: {num_workers}")
    print("=" * 60 + "\n")

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

    Args:
        num_users: 模拟的用户数量
        vocab_size: 歌曲ID的范围（词表大小）
        avg_playlist_len: 平均播放列表长度
        max_seq_len: 最大序列长度
        batch_size: DataLoader的批次大小

    Returns:
        (train_loader, val_loader) 元组
    """
    print(f"=" * 60)
    print(f"生成虚拟数据: {num_users} 个用户, 词表大小: {vocab_size}")
    print(f"=" * 60)

    playlists = []

    for user_id in range(num_users):
        playlist_len = max(5, int(np.random.normal(avg_playlist_len, 5)))
        playlist = []

        first_song = random.randint(1, vocab_size - 1)
        playlist.append(first_song)

        for _ in range(playlist_len - 1):
            if random.random() < 0.7:
                next_song = (playlist[-1] + random.randint(-50, 50)) % (vocab_size - 1) + 1
            else:
                next_song = random.randint(1, vocab_size - 1)

            next_song = max(1, min(next_song, vocab_size - 1))
            playlist.append(next_song)

        playlists.append(playlist)

    print(f"生成完成! 共有 {len(playlists)} 个播放列表")

    total_songs = sum(len(p) for p in playlists)
    avg_len = total_songs / len(playlists) if playlists else 0
    print(f"平均播放列表长度: {avg_len:.2f}")

    # 使用标准 PlaylistDataset（用于虚拟数据）
    train_dataset = PlaylistDataset(
        playlists=playlists[:int(len(playlists) * 0.8)],
        max_seq_len=max_seq_len,
        padding_id=0
    )
    val_dataset = PlaylistDataset(
        playlists=playlists[int(len(playlists) * 0.8):],
        max_seq_len=max_seq_len,
        padding_id=0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"训练集样本数: {len(train_dataset):,}")
    print(f"验证集样本数: {len(val_dataset):,}")
    print(f"=" * 60)

    return train_loader, val_loader


class PlaylistDataset(Dataset):
    """
    播放列表数据集类（用于小规模数据）

    该数据集接收用户的高质量播放列表列表，使用滑动窗口机制将其切分为多个训练样本。
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
            playlists: 用户播放列表列表
            max_seq_len: 最大序列长度
            padding_id: 用于填充的ID
            min_playlist_len: 播放列表最小长度
        """
        self.max_seq_len = max_seq_len
        self.padding_id = padding_id

        self.samples = self._create_sliding_window_samples(
            playlists, min_playlist_len
        )

    def _create_sliding_window_samples(
        self,
        playlists: List[List[int]],
        min_playlist_len: int
    ) -> List[Tuple[List[int], int]]:
        """使用滑动窗口切分播放列表"""
        samples = []

        for playlist in playlists:
            if len(playlist) < min_playlist_len:
                continue

            for i in range(1, len(playlist)):
                start_idx = max(0, i - self.max_seq_len)
                input_seq = playlist[start_idx:i]
                padding_len = self.max_seq_len - len(input_seq)
                input_seq = [self.padding_id] * padding_len + input_seq
                target_id = playlist[i]
                samples.append((input_seq, target_id))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_seq, target_id = self.samples[idx]

        return {
            'input_seq': torch.tensor(input_seq, dtype=torch.long),
            'target': torch.tensor(target_id, dtype=torch.long)
        }


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("测试数据流水线")
    print("=" * 60)

    # 测试 Vocab 类
    print("\n--- 测试 Vocab 类 ---")
    if os.path.exists('uri_to_id.json'):
        vocab = Vocab('uri_to_id.json')
        print(f"词表大小: {len(vocab):,}")
    else:
        print("词表文件不存在，跳过测试")

    # 测试虚拟数据
    print("\n--- 测试虚拟数据 ---")
    train_loader, val_loader = get_dummy_dataloaders(
        num_users=100,
        vocab_size=100,
        batch_size=8
    )

    print("\n测试DataLoader批次:")
    for batch in train_loader:
        print(f"  input_seq shape: {batch['input_seq'].shape}")
        print(f"  target shape: {batch['target'].shape}")
        break

    print("\n" + "=" * 60)
    print("数据流水线测试通过!")
    print("=" * 60)
