"""Generic PyTorch tabular classification architectures."""

from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise ImportError(
        "Optional package 'torch' is required for deep tabular models. "
        "Install PyTorch to use these models."
    ) from exc


class MambaAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.n1 = nn.LayerNorm(hidden_dim)
        self.d1 = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.n2 = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.n3 = nn.LayerNorm(hidden_dim // 2)
        self.d2 = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim // 2, num_classes)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.d1(F.gelu(self.n1(self.proj(x))))
        residual = x
        xq, _ = self.attn(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x = self.n2(xq.squeeze(1) + residual)
        return self.out(self.d2(F.gelu(self.n3(self.fc(x)))))


class FTTransformerClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        d_token: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.W, a=0.01)
        self.b = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        enc = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        tokens = x.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)
        cls = self.cls.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        output = self.enc(tokens)
        return self.head(self.norm(output[:, 0]))


class AutoIntClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        d: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d))
        nn.init.kaiming_uniform_(self.W, a=0.01)
        self.b = nn.Parameter(torch.zeros(n_features, d))
        self.attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    d,
                    n_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features * d, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)
        for attention, norm in zip(self.attn_layers, self.norms):
            residual, _ = attention(hidden, hidden, hidden)
            hidden = norm(hidden + residual)
        return self.head(hidden)


class TabResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int = 3,
        hidden: int = 256,
        n_blocks: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden * 2, hidden),
                    nn.Dropout(dropout / 2),
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)
