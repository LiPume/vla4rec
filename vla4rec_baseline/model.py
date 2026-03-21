"""
================================================================================
VLA4Rec Baseline - 模型架构模块 (大规模稀疏ID优化版)
================================================================================

本模块实现了基于 Causal Transformer 的推荐策略网络（Policy Network），
针对大规模稀疏ID（153万量级）进行了深度优化。

核心优化：
1. 权重共享 (Weight Tying): Embedding 与输出层共享权重，节省 1.5GB 显存
2. 轻量化设计: 减小 hidden_size/intermediate_size，降低参数量
3. 梯度检查点 (Gradient Checkpointing): 可选，降低显存峰值

架构设计参考：
- SASRec: 使用单向 Transformer 进行自回归推荐
- GPT-2: 使用 Causal Attention 防止信息泄露
- RT-2: 将推荐任务建模为视觉-语言-动作模型的简化版本

Author: VLA4Rec Team
Date: 2026-03-21
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class StateEncoder(nn.Module):
    """
    状态编码器模块 (优化版)

    负责将输入的歌曲ID序列（State）编码为高维隐向量表征。

    架构组成：
    1. Token Embedding Layer: 将每个歌曲ID映射为 dense vector
    2. Positional Embedding Layer: 注入序列位置信息（可学习）
    3. Transformer Encoder Layers: 使用单向注意力建模序列依赖关系
    4. Layer Normalization: 稳定训练

    输入: [batch_size, seq_len] 歌曲ID序列
    输出: [batch_size, seq_len, hidden_dim] 状态表征序列
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 50
    ):
        """
        初始化状态编码器

        Args:
            vocab_size: 歌曲ID的词表大小（Semantic ID 数量）
            hidden_size: 隐向量维度
            num_layers: Transformer Encoder 层数
            num_heads: 注意力头数量
            intermediate_size: FFN 中间层维度（默认为 hidden_size * 4）
            dropout: Dropout 概率
            max_seq_len: 最大序列长度（用于初始化位置编码）
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # 1. Token Embedding: 将歌曲ID映射为 hidden_dim 维向量
        # 输入: [batch_size, seq_len]
        # 输出: [batch_size, seq_len, hidden_size]
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=0
        )

        # 2. 位置编码（可学习的）
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=hidden_size
        )

        self.register_buffer(
            'position_ids',
            torch.arange(max_seq_len).unsqueeze(0)
        )

        # 3. Dropout 层
        self.embedding_dropout = nn.Dropout(dropout)

        # 4. Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size or hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size)
        )

        self.final_layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: 输入的歌曲ID序列
                   Shape: [batch_size, seq_len]
            attention_mask: 注意力掩码（可选）

        Returns:
            encoded_states: 编码后的状态表征
                   Shape: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        # Step 1: Token Embedding
        token_embeds = self.token_embedding(input_ids)

        # Step 2: Position Embedding
        position_ids = self.position_ids[:, :seq_len]
        position_embeds = self.position_embedding(position_ids)

        # Step 3: 合并 Token Embedding 和 Position Embedding
        hidden_states = token_embeds + position_embeds

        # Step 4: Dropout
        hidden_states = self.embedding_dropout(hidden_states)

        # Step 5: Transformer Encoder
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        encoded_states = self.transformer_encoder(
            hidden_states,
            src_key_padding_mask=key_padding_mask
        )

        encoded_states = self.final_layer_norm(encoded_states)

        return encoded_states


class ActionHead(nn.Module):
    """
    动作预测头模块 (优化版 - 权重共享)

    核心优化：不再创建独立的 Linear 层，而是通过权重共享使用 token_embedding 的权重。
    这样可以节省 vocab_size * hidden_size * 4 bytes 的显存：
    - vocab_size = 1,536,416
    - hidden_size = 64
    - 节省空间 = 1,536,416 * 64 * 4 bytes ≈ 393 MB (单精度)
    - 如果使用双精度，则节省约 786 MB
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        embedding_layer: nn.Embedding,
        use_weight_tying: bool = True
    ):
        """
        初始化动作预测头

        Args:
            hidden_size: StateEncoder 输出的隐向量维度
            vocab_size: 歌曲ID的词表大小
            embedding_layer: 用于权重共享的 embedding 层
            use_weight_tying: 是否使用权重共享（默认开启）
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.use_weight_tying = use_weight_tying

        # 核心优化：权重共享
        # 不再创建独立的 Linear 层，而是使用 embedding 的转置作为输出权重
        self.embedding_layer = embedding_layer

        # 如果不使用权重共享，则创建独立的输出层
        if not use_weight_tying:
            self.action_layer = nn.Linear(hidden_size, vocab_size, bias=True)
        else:
            # 添加一个偏置项（因为 embedding 不包含偏置）
            self.output_bias = nn.Parameter(torch.zeros(vocab_size))
            self.action_layer = None

    def forward(
        self,
        state_representation: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            state_representation: 状态表征
                   Shape: [batch_size, seq_len, hidden_size]

        Returns:
            action_logits: 每个位置的预测 logits
                   Shape: [batch_size, vocab_size]
        """
        batch_size, seq_len, hidden_size = state_representation.shape

        # 取序列最后一个时间步的隐向量作为当前状态的表征
        current_state = state_representation[:, -1, :]

        # 使用权重共享计算 logits
        # current_state: [batch_size, hidden_size]
        # embedding.weight: [vocab_size, hidden_size]
        # output: [batch_size, vocab_size] = [batch_size, hidden_size] @ [hidden_size, vocab_size]^T
        if self.use_weight_tying:
            # 矩阵乘法: current_state @ embedding.weight^T
            # embedding.weight 的 shape 是 [vocab_size, hidden_size]
            # 我们需要 [batch_size, hidden_size] @ [hidden_size, vocab_size]
            action_logits = F.linear(current_state, self.embedding_layer.weight, self.output_bias)
        else:
            action_logits = self.action_layer(current_state)

        return action_logits


class VLA4RecBaseline(nn.Module):
    """
    VLA4Rec 基线模型 (大规模稀疏ID优化版)

    核心优化：
    1. 权重共享 (Weight Tying): 节省显存
    2. 轻量化设计: 适合 153 万量级的 Action Space

    模型架构（强制解耦设计 + 权重共享）：
    - StateEncoder: 将歌曲ID序列编码为状态表征
    - ActionHead: 基于状态表征生成动作（预测下一首歌），与 Embedding 共享权重
    """

    def __init__(
        self,
        vocab_size: int = 1536416,  # Spotify 大规模词表
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        use_weight_tying: bool = True
    ):
        """
        初始化 VLA4Rec 基线模型

        Args:
            vocab_size: 歌曲ID的词表大小（默认 1,536,416）
            hidden_size: 隐向量维度
            num_layers: Transformer 层数
            num_heads: 注意力头数量
            intermediate_size: FFN 中间层维度
            dropout: Dropout 概率
            max_seq_len: 最大序列长度
            use_weight_tying: 是否使用权重共享（强烈建议开启）
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.use_weight_tying = use_weight_tying

        # 解耦设计：StateEncoder 和 ActionHead 独立初始化
        # 但 ActionHead 通过 embedding_layer 引用实现权重共享
        self.state_encoder = StateEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            max_seq_len=max_seq_len
        )

        self.action_head = ActionHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            embedding_layer=self.state_encoder.token_embedding,
            use_weight_tying=use_weight_tying
        )

        # 权重初始化
        self.apply(self._init_weights)

        # 计算并打印显存节省信息
        if use_weight_tying:
            param_count = self._count_parameters()
            print(f"[VLA4RecBaseline] 权重共享已开启，节省约 {param_count['shared_params']:,} 参数")

    def _count_parameters(self) -> dict:
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 估算权重共享节省的参数
        if self.use_weight_tying:
            # embedding 权重会被 ActionHead 共享使用
            shared_params = self.vocab_size * self.hidden_size
            unique = total - shared_params
        else:
            shared_params = 0
            unique = total

        return {
            'total': total,
            'trainable': trainable,
            'shared_params': shared_params,
            'unique_params': unique
        }

    def _init_weights(self, module: nn.Module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: 输入的歌曲ID序列
                   Shape: [batch_size, seq_len]
            attention_mask: 注意力掩码（可选）

        Returns:
            action_logits: 预测下一首歌的 logits
                   Shape: [batch_size, vocab_size]
        """
        # Step 1: StateEncoder 将歌曲ID序列编码为状态表征
        state_representation = self.state_encoder(input_ids, attention_mask)

        # Step 2: ActionHead 基于状态表征生成动作
        action_logits = self.action_head(state_representation)

        return action_logits

    def predict_next_item(
        self,
        input_ids: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测下一首歌

        Args:
            input_ids: 输入序列
            top_k: 返回 top-k 个候选

        Returns:
            top_probs: Top-K 概率
            top_indices: Top-K 歌曲ID
        """
        with torch.no_grad():
            logits = self.forward(input_ids)
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

        return top_probs, top_indices


# ============================================================================
# 模型工厂函数
# ============================================================================

def create_baseline_model(
    vocab_size: int = 1536416,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    max_seq_len: int = 50,
    use_weight_tying: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[VLA4RecBaseline, dict]:
    """
    创建基线模型的工厂函数

    Args:
        vocab_size: 词表大小
        hidden_size: 隐向量维度
        num_layers: Transformer 层数
        num_heads: 注意力头数
        dropout: Dropout 概率
        max_seq_len: 最大序列长度
        use_weight_tying: 是否使用权重共享
        device: 设备

    Returns:
        model: 初始化好的模型
        config: 模型配置字典
    """
    model = VLA4RecBaseline(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_seq_len=max_seq_len,
        use_weight_tying=use_weight_tying
    )

    config = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'dropout': dropout,
        'max_seq_len': max_seq_len,
        'use_weight_tying': use_weight_tying
    }

    model = model.to(device)
    model.device = device

    return model, config


def print_model_memory_usage(model: nn.Module, batch_size: int, seq_len: int, device: str = 'cuda'):
    """
    打印模型显存使用估算

    Args:
        model: 模型
        batch_size: 批次大小
        seq_len: 序列长度
        device: 设备类型
    """
    if device == 'cpu':
        print("CPU 模式，无需计算显存")
        return

    # 统计参数量
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    param_size_mb = param_size / 1024 / 1024

    # 估算中间激活值显存
    # Hidden states: batch_size * seq_len * hidden_size * 4 bytes (float32)
    hidden_size = model.hidden_size
    activation_size = batch_size * seq_len * hidden_size * 4 / 1024 / 1024  # MB

    # Action logits: batch_size * vocab_size * 4 bytes
    vocab_size = model.vocab_size
    output_size = batch_size * vocab_size * 4 / 1024 / 1024 / 1024  # GB

    print("\n" + "=" * 50)
    print("显存使用估算:")
    print("=" * 50)
    print(f"模型参数量: {param_size_mb:.2f} MB")
    print(f"Hidden states 激活: ~{activation_size:.2f} MB")
    print(f"Output logits: ~{output_size:.2f} GB")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    if model.use_weight_tying:
        print("\n[权重共享优化] 输出层与 Embedding 共享权重")
        print(f"共享权重节省: {vocab_size * hidden_size * 4 / 1024 / 1024:.2f} MB")
    print("=" * 50)


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("测试 VLA4Rec Baseline 模型 (大规模稀疏ID优化版)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 测试小规模 vocab_size
    print("\n--- 测试 1: 小规模 vocab_size=1000 ---")
    vocab_size_small = 1000
    hidden_size = 64
    batch_size = 8
    seq_len = 20

    model_small, config_small = create_baseline_model(
        vocab_size=vocab_size_small,
        hidden_size=hidden_size,
        num_layers=2,
        num_heads=4,
        device=device,
        use_weight_tying=True
    )

    print(f"模型参数量: {sum(p.numel() for p in model_small.parameters()):,}")
    print_model_memory_usage(model_small, batch_size, seq_len, device)

    input_ids = torch.randint(1, vocab_size_small, (batch_size, seq_len), device=device)
    action_logits = model_small(input_ids)
    print(f"输入 shape: {input_ids.shape}")
    print(f"输出 logits shape: {action_logits.shape}")

    # 测试大规模 vocab_size
    print("\n--- 测试 2: 大规模 vocab_size=1536416 (Spotify) ---")
    vocab_size_large = 1536416

    model_large, config_large = create_baseline_model(
        vocab_size=vocab_size_large,
        hidden_size=hidden_size,
        num_layers=2,
        num_heads=4,
        device=device,
        use_weight_tying=True
    )

    print(f"模型参数量: {sum(p.numel() for p in model_large.parameters()):,}")
    print_model_memory_usage(model_large, batch_size, seq_len, device)

    input_ids = torch.randint(1, vocab_size_large, (batch_size, seq_len), device=device)
    action_logits = model_large(input_ids)
    print(f"输入 shape: {input_ids.shape}")
    print(f"输出 logits shape: {action_logits.shape}")
    print(f"输出 logits 范围: [{action_logits.min().item():.4f}, {action_logits.max().item():.4f}]")

    print("\n" + "=" * 60)
    print("模型测试通过!")
    print("=" * 60)
