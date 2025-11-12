# /Tiny-RecUNet/recursive_module.py

import torch
import torch.nn as nn
import copy

class RecursiveReasoningModule(nn.Module):
    """
    TRM의 'L_level' 모듈을 모방한 재귀 모듈.
    (state, injection) -> state + injection -> (Transformer Blocks) -> new_state
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, num_layers: int = 1):
        """
        :param embed_dim: TransUnet의 embed_dim
        :param num_heads: TransUnet의 num_heads
        :param mlp_ratio: TransUnet의 mlp_ratio
        :param num_layers: 재귀 1스텝당 내부적으로 통과할 레이어 수 (TRM은 보통 1~2개)
        """
        super().__init__()
        
        # 재사용할 단일 트랜스포머 블록 정의
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True  # Pre-Normalization을 사용하는 것이 안정적일 수 있습니다.
        )
        
        # 이 블록을 num_layers 만큼 복제하여 하나의 '재귀 모듈'을 구성
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, hidden_state: torch.Tensor, input_injection: torch.Tensor) -> torch.Tensor:
        """
        :param hidden_state: 현재 은닉 상태 (예: z_L 또는 z_H)
        :param input_injection: 주입될 정보 (예: z_H + input_embeddings 또는 z_L)
        """
        
        # 1. TRM의 핵심: 두 입력을 더합니다.
        x = hidden_state + input_injection
        
        # 2. 내부 트랜스포머 블록을 순차적으로 통과
        for layer in self.layers:
            x = layer(x)
            
        return x