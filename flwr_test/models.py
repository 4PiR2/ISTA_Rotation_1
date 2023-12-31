from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class CNNEmbed2(nn.Module):
    def __init__(self, embed_y: bool, dim_y: int, embed_dim: int, device=None, in_channels: int = 3, n_kernels: int = 16):
        super().__init__()
        self.embed_y = embed_y
        layers = [
            nn.Conv2d(in_channels + embed_y * dim_y, n_kernels, 5),  # (B, C, 28, 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, C, 14, 14)
            nn.Conv2d(n_kernels, 2 * n_kernels, 5),  # (B, 2C, 10, 10)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 2C, 5, 5)
            nn.Flatten(start_dim=-3, end_dim=-1),  # (B, 50C)
            nn.Linear(2 * n_kernels * 5 * 5, 120),  # (B, 120)
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),  # (B, 84)
            nn.ReLU(inplace=True),
            nn.Linear(84, embed_dim),  # (B, E)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor | Tuple[torch.Tensor, torch.Tensor], y: Optional[torch.Tensor] = None):
        if not isinstance(x, torch.Tensor):
            x, y = x
        if self.embed_y:
            c = y[..., None, None].expand(*((-1,) * (y.dim() - 1)), -1, *x.shape[-2:])
            x = torch.cat((x, c), dim=-3)
        x = self.layers(x)
        return x * 16.


class MLPEmbed2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor | Tuple[torch.Tensor, torch.Tensor], y: Optional[torch.Tensor] = None):
        if not isinstance(x, torch.Tensor):
            _, y = x
        y = self.fc(y)
        return y * 16.


class Hyper(nn.Module):
    def __init__(
            self, example_state_dict: Dict[str, torch.Tensor], embedding_dim=26, hidden_dim=100, n_hidden=1,
            spec_norm=False,
    ):
        super().__init__()

        mlp_layers = []
        for i in range(n_hidden + 1):
            linear_layer = nn.Linear(embedding_dim if i == 0 else hidden_dim, hidden_dim)
            mlp_layers.append(spectral_norm(linear_layer) if spec_norm else linear_layer)
            if i != n_hidden:
                mlp_layers.append(nn.ReLU(inplace=True))
            # mlp_layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp_layers)

        layer_dict = {}
        for key, value in example_state_dict.items():
            linear_layer = nn.Linear(hidden_dim, value.numel())
            layer_dict[key.replace('.', '|')] = nn.Sequential(
                spectral_norm(linear_layer) if spec_norm else linear_layer,
                nn.Unflatten(dim=-1, unflattened_size=value.shape),
            )
        self.layer_dict = nn.ParameterDict(layer_dict)

    def forward(self, v):
        features = self.mlp(v)
        weights = {
            key.replace('|', '.'): self.layer_dict[key](features)
            for key in self.layer_dict.keys()
        }
        return weights


class HeadTarget(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int):
        super().__init__()
        layers = []
        if n_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        if n_layers >= 2:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
