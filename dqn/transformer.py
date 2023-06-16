import numpy as np
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Type


class SimpleTransformer(nn.Module):

    def __init__(self, dim, num_layers=2, num_heads=2):
        super(SimpleTransformer, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.encoder = ValueEncoder(dim)
        self.pos_embed = PositionEmbeddingRandom(dim // 2)
        for i in range(num_layers):
            layer = Attention(dim, num_heads=num_heads)
            setattr(self, f'layer{i}', layer)
            layer = Mlp(dim, int(dim * 4.))
            setattr(self, f'mlp{i}', layer)
        self.proj = nn.Linear(dim, 1)
        self.cls_embed = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        B, H, W = x.shape
        x = self.encoder(x.unsqueeze(-1))  # B, H, W, C
        x = x + self.pos_embed((H, W)).permute(1, 2, 0).unsqueeze(0)  # B, H, W, C
        x = x.flatten(1, 2)  # B, H * W, C
        cls_embed = self.cls_embed.expand(B, -1, -1)  # B, 1, C
        x = torch.cat((cls_embed, x), dim=1)  # B, H * W + 1, C
        for i in range(self.num_layers):
            _input = x
            _layer = getattr(self, f'layer{i}')
            x = _layer(x)
            _mlp = getattr(self, f'mlp{i}')
            x = _mlp(x)
            x = x + _input
        # select the cls token
        x = x[:, 0]  # B, C
        x = self.proj(x)
        return x


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


class ValueEncoder(nn.Module):

    def __init__(self, dim):
        super(ValueEncoder, self).__init__()
        self.fc1 = nn.Linear(1, dim)

    def forward(self, x):
        x = self.fc1(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class Attention(nn.Module):
    """Multi-head Attention block"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3, B, num_heads, N, head_dim
        q, k, v = qkv.unbind(0) # B, num_heads, N, head_dim
        attn = (q * self.scale) @ k.transpose(-2, -1)  # B, num_heads, N, N
        attn = attn.softmax(dim=-1)  # B, num_heads, N, N
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)  # B, N, head_dim * num_heads
        x = self.proj(x)  # B, N, C
        return x