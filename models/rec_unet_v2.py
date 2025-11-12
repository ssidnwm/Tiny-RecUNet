import torch
import torch.nn as nn
import math
from .resnet_encoder import ResNetEncoder
from .recursive_module import RecursiveReasoningModule

class CNNEncoder(ResNetEncoder):
    """Wrapper alias for backward compatibility; now uses local ResNetEncoder."""
    def __init__(self, name: str = "resnet34", pretrained: bool = False, in_channels: int = 3):
        # pretrained is ignored in local implementation; kept for API compatibility
        super().__init__(name=name, in_channels=in_channels)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            # align spatial if off by 1 due to rounding
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Transformer2D(nn.Module):
    """Transformer over 2D feature maps [B, C, H, W] with 2D positional embeddings.

    This keeps the external API in image space without exposing token sequences.
    """

    def __init__(self, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], pos_embed: [1, C, H0, W0]
        B, C, H, W = x.shape
        if pos_embed is not None:
            pe = pos_embed
            if pe.shape[-2:] != (H, W):
                pe = nn.functional.interpolate(pe, size=(H, W), mode="bilinear", align_corners=False)
            x = x + pe
        # flatten spatial to sequence
        x_seq = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x_seq = self.encoder(x_seq)
        x = x_seq.transpose(1, 2).reshape(B, C, H, W)
        return x


class TinyRecUNet(nn.Module):
    """TransUNet의 트랜스포머를 TRM(Tiny Recursive Model) 구조로 대체한 모델.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        img_size: int = 224,
        backbone: str = "resnet34",
        pretrained_backbone: bool = False,
        embed_dim: int = 256,
        #depth: int = 6,
        recursive_steps: int= 6,
        num_recursive_layers: int= 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_grid=None,               # (gh, gw) over H/16,W/16 feature map; if None, defaults to full grid
        embed_dropout: float = 0.0,    # dropout on embeddings like the paper
        decoder_channels=(256, 128, 64, 16),
        head_channels: int = 512,
    ):
        super().__init__()

        self.encoder = CNNEncoder(name=backbone, pretrained=pretrained_backbone, in_channels=in_channels)

        # projection to transformer embed dim
        self.proj = nn.Conv2d(self.encoder.channels["c16"], embed_dim, kernel_size=1)
        
        # grid-based patch embedding (이하 동일)
        base_grid_h = max(1, img_size // 16)
        base_grid_w = base_grid_h
        if patch_grid is None:
            gh, gw = base_grid_h, base_grid_w
        else:
            assert isinstance(patch_grid, (tuple, list)) and len(patch_grid) == 2, "patch_grid must be (gh, gw)"
            gh, gw = int(patch_grid[0]), int(patch_grid[1])
            assert gh > 0 and gw > 0, "grid dims must be positive"
            assert base_grid_h % gh == 0 and base_grid_w % gw == 0, "img_size/16 must be divisible by grid"
        kh, kw = max(1, base_grid_h // gh), max(1, base_grid_w // gw)
        self.patch_embed = nn.Conv2d(embed_dim, embed_dim, kernel_size=(kh, kw), stride=(kh, kw))
        self.token_grid = (gh, gw)
        
        # 1D positional embeddings [1, N, E]
        N = gh * gw  # N = 토큰의 개수 (gh * gw)
        self.pos_embed_tok = nn.Parameter(torch.zeros(1, N, embed_dim))
        try:
            nn.init.trunc_normal_(self.pos_embed_tok, std=0.02)
        except Exception:
            nn.init.normal_(self.pos_embed_tok, std=0.02)
        self.emb_dropout = nn.Dropout(p=embed_dropout) if embed_dropout and embed_dropout > 0 else nn.Identity()

        # --- TRM 수정 부분 시작 ---
        
        # 1. 기존 standard Transformer 삭제
        # enc_layer = nn.TransformerEncoderLayer(...)
        # self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # 2. TRM 재귀 관련 파라미터 저장
        self.recursive_steps = recursive_steps
        
        # 3. 재사용할 재귀 모듈(L_level) 정의
        self.recursive_module = RecursiveReasoningModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_layers=num_recursive_layers # 모듈 내부는 2-layer
        )

        # 4. 재귀 상태(z_H, z_L) 초기값 정의
        #    (1, N, E) 크기의 학습 가능한 파라미터로 생성
        self.H_init = nn.Parameter(torch.zeros(1, N, embed_dim))
        self.L_init = nn.Parameter(torch.zeros(1, N, embed_dim))
        nn.init.trunc_normal_(self.H_init, std=0.02)
        nn.init.trunc_normal_(self.L_init, std=0.02)

        # --- TRM 수정 부분 끝 ---

        # decoder head conv (이하 디코더 부분 모두 동일)
        self.conv_more = nn.Sequential(
            nn.Conv2d(embed_dim, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )

        # build decoder blocks with skip channels
        c8, c4, c2 = self.encoder.channels["c8"], self.encoder.channels["c4"], self.encoder.channels["c2"]
        dec_ch = list(decoder_channels)
        self.dec1 = DecoderBlock(head_channels, c8, dec_ch[0])  # H/16 -> H/8 + skip f8
        self.dec2 = DecoderBlock(dec_ch[0], c4, dec_ch[1])      # H/8 -> H/4 + skip f4
        self.dec3 = DecoderBlock(dec_ch[1], c2, dec_ch[2])      # H/4 -> H/2 + skip f2
        self.dec4 = nn.Sequential(                              # H/2 -> H
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dec_ch[2], dec_ch[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(dec_ch[3]),
            nn.ReLU(inplace=True),
        ) if len(dec_ch) > 3 else nn.Identity()

        self.head = nn.Conv2d(dec_ch[-1], out_channels, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        f16, skips = self.encoder(x)
        z = self.proj(f16)
        z = self.patch_embed(z)
        B, E, gh, gw = z.shape
        
        tokens = z.flatten(2).transpose(1, 2) # [B, N, E]
        tokens = tokens + self.pos_embed_tok
        input_embeddings = self.emb_dropout(tokens) # TRM의 'input_embeddings' 역할

        # --- 1. 기존 self.transformer(tokens) 라인 삭제 ---
        # tokens = self.transformer(tokens)

        # --- 2. TRM 재귀 루프 적용 ---
        
        # 상태 초기화: H_init, L_init을 배치 크기(B)만큼 복제
        z_H = self.H_init.expand(B, -1, -1)
        z_L = self.L_init.expand(B, -1, -1)

        # 재귀 루프 실행 (TRM의 H_cycles)
        for _ in range(self.recursive_steps):
            
            # 1. z_L 업데이트 (TRM의 L_cycles 루프는 recursive_module 내부에 구현됨)
            z_L = self.recursive_module(
                hidden_state=z_L, 
                input_injection=(z_H + input_embeddings) # 원본 입력(input)과 H상태를 함께 주입
            )
            
            # 2. z_H 업데이트
            z_H = self.recursive_module(
                hidden_state=z_H,
                input_injection=z_L # L상태를 주입
            )

        # 최종 출력은 z_H를 사용
        tokens_out = z_H
        # --- 수정 끝 ---

        feat = tokens_out.transpose(1, 2).reshape(B, E, gh, gw) # 3. tokens -> tokens_out
        x = self.conv_more(feat)
        
        # ... (dec1, dec2, ... head, sigmoid(y) 등 디코더 부분 모두 동일) ...
        x = self.dec1(x, skips[0])
        x = self.dec2(x, skips[1])
        x = self.dec3(x, skips[2])
        x = self.dec4(x)
        if x.shape[-2:] != (H, W):
            x = nn.functional.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        y = self.head(x)
        return torch.sigmoid(y)