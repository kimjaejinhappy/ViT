# models/vit.py — ViT 모델 구조 (논문 구조 그대로)
#
# 논문: An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)
#
# 전체 흐름:
#   이미지
#   → PatchEmbedding  : 패치 분할 + [CLS] 토큰 + Positional Embedding
#   → TransformerEncoderBlock × L
#   → LayerNorm
#   → [CLS] 토큰 추출
#   → MLP Head
#   → 클래스 예측

import torch
import torch.nn as nn
import config


# ────────────────────────────────────────────────────────────
# 1. Patch Embedding
# ────────────────────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """
    이미지 → 패치 토큰 시퀀스

    논문 수식: z_0 = [x_class; x_p^1 E; ...; x_p^N E] + E_pos
    - Conv2d(kernel=patch_size, stride=patch_size) 로 패치 분할 + 선형 투영을 한 번에 처리
    - [CLS] 토큰과 Positional Embedding은 학습 가능한 파라미터
    """
    def __init__(self):
        super().__init__()

        # Conv2d로 패치 분할 + Linear Projection을 한 번에 수행
        # kernel_size=stride=patch_size → 겹치지 않는 패치 추출
        self.projection = nn.Sequential(
            nn.Conv2d(config.IN_CHANNELS, config.EMBED_DIM,
                      kernel_size=config.PATCH_SIZE, stride=config.PATCH_SIZE),
            nn.Flatten(2),   # (B, D, H/P, W/P) → (B, D, num_patches)
        )

        # [CLS] 토큰: 전체 이미지 표현을 담을 학습 가능한 벡터
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.EMBED_DIM))

        # Positional Embedding: 패치 순서 정보 (num_patches + 1은 [CLS] 자리)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.NUM_PATCHES + 1, config.EMBED_DIM)
        )

        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        # x: (B, C, H, W)
        B = x.size(0)

        x = self.projection(x).transpose(1, 2)          # (B, num_patches, D)

        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        x   = torch.cat([cls, x], dim=1)                 # (B, num_patches+1, D)

        x   = x + self.pos_embedding
        return self.dropout(x)


# ────────────────────────────────────────────────────────────
# 2. Multi-Head Self-Attention
# ────────────────────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    """
    논문 수식: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

    - Q, K, V를 하나의 Linear로 한 번에 계산 (3배 크기)
    - num_heads개 헤드 병렬 수행 후 concat
    """
    def __init__(self):
        super().__init__()
        assert config.EMBED_DIM % config.NUM_HEADS == 0

        self.num_heads = config.NUM_HEADS
        self.head_dim  = config.EMBED_DIM // config.NUM_HEADS
        self.scale     = self.head_dim ** -0.5            # 1 / sqrt(d_k)

        self.qkv      = nn.Linear(config.EMBED_DIM, config.EMBED_DIM * 3)
        self.out_proj = nn.Linear(config.EMBED_DIM, config.EMBED_DIM)
        self.dropout  = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        B, N, D = x.shape
        H = self.num_heads

        # Q, K, V 한 번에 계산 후 헤드 차원으로 분리
        qkv = self.qkv(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                          # 각 (B, H, N, head_dim)

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 가중합 후 헤드 합치기
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


# ────────────────────────────────────────────────────────────
# 3. MLP Block
# ────────────────────────────────────────────────────────────
class MLPBlock(nn.Module):
    """
    Transformer 내부 Feed-Forward Network

    논문: 'MLP contains two layers with a GELU non-linearity'
    구조: Linear → GELU → Dropout → Linear → Dropout
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.EMBED_DIM, config.MLP_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.MLP_DIM, config.EMBED_DIM),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


# ────────────────────────────────────────────────────────────
# 4. Transformer Encoder Block
# ────────────────────────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    """
    논문 수식:
        z'_l = MSA(LN(z_{l-1})) + z_{l-1}   ← Attention + Residual
        z_l  = MLP(LN(z'_l))   + z'_l       ← MLP + Residual

    Pre-LayerNorm 구조 (LN을 Attention/MLP 앞에 적용)
    """
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.EMBED_DIM)
        self.attn  = MultiHeadSelfAttention()

        self.norm2 = nn.LayerNorm(config.EMBED_DIM)
        self.mlp   = MLPBlock()

        self.drop  = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))   # Attention + Residual
        x = x + self.drop(self.mlp(self.norm2(x)))    # MLP + Residual
        return x


# ────────────────────────────────────────────────────────────
# 5. ViT 전체 모델
# ────────────────────────────────────────────────────────────
class ViT(nn.Module):
    """
    Vision Transformer (ViT)

    입력 흐름:
        (B, C, H, W)
        → PatchEmbedding         : (B, num_patches+1, D)
        → TransformerEncoder × L : (B, num_patches+1, D)
        → LayerNorm
        → [CLS] 토큰 추출         : (B, D)
        → MLP Head               : (B, num_classes)
    """
    def __init__(self):
        super().__init__()

        self.patch_embedding = PatchEmbedding()

        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock()
            for _ in range(config.NUM_LAYERS)
        ])

        self.norm = nn.LayerNorm(config.EMBED_DIM)

        # MLP Head: 처음부터 학습하는 경우 hidden layer 1개 사용
        self.mlp_head = nn.Sequential(
            nn.Linear(config.EMBED_DIM, config.MLP_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.MLP_DIM, config.NUM_CLASSES),
        )

    def forward(self, x):
        x = self.patch_embedding(x)   # 패치 임베딩
        x = self.transformer(x)       # Transformer Encoder
        x = self.norm(x)              # 마지막 LayerNorm
        cls = x[:, 0]                 # [CLS] 토큰만 추출
        return self.mlp_head(cls)     # 분류

    def count_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


__all__ = ['ViT', 'PatchEmbedding', 'MultiHeadSelfAttention',
           'MLPBlock', 'TransformerEncoderBlock']
