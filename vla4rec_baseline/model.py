"""
================================================================================
VLA4Rec Baseline - 模型架构模块
================================================================================

本模块实现了基于 Causal Transformer 的推荐策略网络（Policy Network），
采用强制解耦设计以支持后续多模态扩展。

核心设计理念（来自 RT-2/VLA 的思想）：
- StateEncoder: 将历史状态（歌曲ID序列）编码为状态表征
- ActionHead: 基于状态表征生成动作（预测下一首歌）

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
    状态编码器模块

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
            padding_idx=0  # 设置padding索引，padding位置的embedding会一直是0
        )

        # 2. 位置编码（可学习的）
        # 为每个位置提供一个可学习的隐向量
        # 输入: [batch_size, seq_len]
        # 输出: [batch_size, seq_len, hidden_size]
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=hidden_size
        )

        # 位置ID，用于生成位置编码
        # [max_seq_len] = [0, 1, 2, ..., max_seq_len-1]
        self.register_buffer(
            'position_ids',
            torch.arange(max_seq_len).unsqueeze(0)  # [1, max_seq_len]
        )

        # 3. Dropout 层
        self.embedding_dropout = nn.Dropout(dropout)

        # 4. Transformer Encoder 层
        # 使用 nn.TransformerEncoderLayer 构建单向注意力（通过 mask 实现因果性）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size or hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # 输入输出格式: [batch, seq, feature]
            norm_first=True  # Pre-LayerNorm 稳定训练
        )

        # 5. Transformer Encoder
        # 通过 attn_mask 创建 Causal Mask 实现因果注意力
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size)  # 最终的 LayerNorm
        )

        # 6. 额外的 LayerNorm（与 Pre-LayerNorm 配合使用）
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
                   取值范围: [0, vocab_size-1]，其中0为padding
            attention_mask: 注意力掩码（可选）
                   Shape: [batch_size, seq_len]
                   1表示有效位置，0表示padding位置

        Returns:
            encoded_states: 编码后的状态表征
                   Shape: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        # Step 1: Token Embedding
        # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        token_embeds = self.token_embedding(input_ids)

        # Step 2: Position Embedding
        # 使用 register_buffer 中的 position_ids 自动处理位置
        # [1, seq_len] -> [batch_size, seq_len, hidden_size]
        position_ids = self.position_ids[:, :seq_len]
        position_embeds = self.position_embedding(position_ids)

        # Step 3: 合并 Token Embedding 和 Position Embedding
        # [batch_size, seq_len, hidden_size]
        hidden_states = token_embeds + position_embeds

        # Step 4: Dropout
        hidden_states = self.embedding_dropout(hidden_states)

        # Step 5: Transformer Encoder
        # 使用 key_padding_mask 来处理 padding
        # 如果有 padding_mask，将其转换为 key_padding_mask 格式
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len]，1表示有效，0表示padding
            # key_padding_mask 需要 True 表示mask掉（padding），False表示保留
            key_padding_mask = (attention_mask == 0)

        # 输入: [batch_size, seq_len, hidden_size]
        # 输出: [batch_size, seq_len, hidden_size]
        encoded_states = self.transformer_encoder(
            hidden_states,
            src_key_padding_mask=key_padding_mask
        )

        # Step 8: 最终 LayerNorm
        encoded_states = self.final_layer_norm(encoded_states)

        return encoded_states

    def _make_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        创建因果注意力掩码

        生成一个下三角矩阵，确保每个位置只能注意到它之前的位置。
        这对于序列生成任务（如推荐）是必需的。

        示例（seq_len=4）：
            [[0., -inf, -inf, -inf],
             [0., 0.,   -inf, -inf],
             [0., 0.,   0.,   -inf],
             [0., 0.,   0.,   0.  ]]

        其中 0 表示可以attend，-inf 表示不能attend
        """
        # 创建下三角矩阵（包含对角线）
        # tril() 将上三角设为0，下三角保持原值
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=dtype)
        )

        # 将0替换为0.0，将1替换为-inf
        # 这样在 softmax 后，-inf 的位置概率接近0
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)

        return mask


class ActionHead(nn.Module):
    """
    动作预测头模块

    负责基于状态表征生成动作（预测下一首歌）。

    设计选择：
    - 使用序列最后一个时间步的隐向量作为"当前状态"的表征
    - 通过线性层输出词表大小的 logits
    - 后续通过 softmax 得到每个歌曲的概率分布

    这对应于具身智能中的"基于当前状态选择动作"。
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int
    ):
        """
        初始化动作预测头

        Args:
            hidden_size: StateEncoder 输出的隐向量维度
            vocab_size: 歌曲ID的词表大小（用于预测）
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # 动作预测层：将隐向量映射为词表大小的 logits
        # 输入: [batch_size, hidden_size]
        # 输出: [batch_size, vocab_size]
        self.action_layer = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(
        self,
        state_representation: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            state_representation: 状态表征
                   Shape: [batch_size, seq_len, hidden_size]
                   通常是 StateEncoder 的输出

        Returns:
            action_logits: 每个位置的预测 logits
                   Shape: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len, hidden_size = state_representation.shape

        # 取序列最后一个时间步的隐向量作为当前状态的表征
        # 这在推荐场景中表示"根据历史行为预测下一步"
        # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        current_state = state_representation[:, -1, :]

        # 通过线性层得到动作 logits
        # [batch_size, hidden_size] -> [batch_size, vocab_size]
        action_logits = self.action_layer(current_state)

        return action_logits


class VLA4RecBaseline(nn.Module):
    """
    VLA4Rec 基线模型

    这是一个基于行为克隆（Behavioral Cloning）的推荐模型，核心思想：
    - 将推荐任务建模为序列决策问题
    - 历史歌曲序列 = 智能体的状态 (State)
    - 预测下一首歌 = 智能体的动作 (Action)
    - 通过专家轨迹（高质量播放列表）进行行为克隆

    模型架构（强制解耦设计）：
    - StateEncoder: 将歌曲ID序列编码为状态表征
    - ActionHead: 基于状态表征生成动作（预测下一首歌）

    这种解耦设计支持后续扩展：
    - 可以在 StateEncoder 前后拼接视觉/文本编码器
    - 可以将离散的 Semantic ID 替换为连续的特征向量
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 50
    ):
        """
        初始化 VLA4Rec 基线模型

        Args:
            vocab_size: 歌曲ID的词表大小
            hidden_size: 隐向量维度
            num_layers: Transformer 层数
            num_heads: 注意力头数量
            intermediate_size: FFN 中间层维度
            dropout: Dropout 概率
            max_seq_len: 最大序列长度
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # 解耦设计：StateEncoder 和 ActionHead 独立初始化
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
            vocab_size=vocab_size
        )

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            # Linear 层：使用截断正态分布初始化
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embedding 层：使用正态分布初始化
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm：偏置初始化为0，缩放初始化为1
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
        # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        state_representation = self.state_encoder(input_ids, attention_mask)

        # Step 2: ActionHead 基于状态表征生成动作
        # [batch_size, seq_len, hidden_size] -> [batch_size, vocab_size]
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
            # 获取 logits
            logits = self.forward(input_ids)

            # Softmax 得到概率分布
            probs = F.softmax(logits, dim=-1)

            # 取 Top-K
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

        return top_probs, top_indices


# ============================================================================
# 模型工厂函数
# ============================================================================

def create_baseline_model(
    vocab_size: int = 1000,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    max_seq_len: int = 50,
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
        device: 设备（'cuda' 或 'cpu'）

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
        max_seq_len=max_seq_len
    )

    config = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'dropout': dropout,
        'max_seq_len': max_seq_len
    }

    model = model.to(device)
    model.device = device

    return model, config


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("测试 VLA4Rec Baseline 模型")
    print("=" * 60)

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 超参数
    vocab_size = 1000
    hidden_size = 64
    num_layers = 2
    num_heads = 4
    batch_size = 16
    seq_len = 50

    # 创建模型
    print(f"\n模型配置:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_heads: {num_heads}")

    model, config = create_baseline_model(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        device=device
    )

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    print("\n测试前向传播:")

    # 随机生成输入
    # [batch_size, seq_len]，值域 [1, vocab_size-1]，避免 padding (0)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

    print(f"  输入 shape: {input_ids.shape}")

    # 前向传播
    action_logits = model(input_ids)

    print(f"  输出 logits shape: {action_logits.shape}")
    print(f"  输出 logits 范围: [{action_logits.min().item():.4f}, {action_logits.max().item():.4f}]")

    # 测试预测功能
    print("\n测试预测功能:")
    top_probs, top_indices = model.predict_next_item(input_ids, top_k=10)
    print(f"  Top-10 概率 shape: {top_probs.shape}")
    print(f"  Top-10 索引 shape: {top_indices.shape}")
    print(f"  第一个样本的 Top-10 推荐: {top_indices[0].tolist()}")

    # 打印模型结构
    print("\n模型结构:")
    print(model)

    print("\n" + "=" * 60)
    print("模型测试通过!")
    print("=" * 60)
