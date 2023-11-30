from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


# class CNNHyper(nn.Module):
#     def __init__(
#             self, n_nodes=0, embedding_dim=26, in_channels=3, out_dim=10, n_kernels=16, hidden_dim=100,
#             spec_norm=False, n_hidden=1):
#         super().__init__()
#
#         self.in_channels = in_channels
#         self.out_dim = out_dim
#         self.n_kernels = n_kernels
#
#         layers = [
#             spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
#         ]
#         for _ in range(n_hidden):
#             layers.append(nn.ReLU(inplace=True))
#             layers.append(
#                 spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
#             )
#
#         self.mlp = nn.Sequential(*layers)
#
#         self.c1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 5 * 5)
#         self.c1_bias = nn.Linear(hidden_dim, self.n_kernels)
#         self.c2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 5 * 5)
#         self.c2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
#         self.l1_weights = nn.Linear(hidden_dim, 120 * 2 * self.n_kernels * 5 * 5)
#         self.l1_bias = nn.Linear(hidden_dim, 120)
#         self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
#         self.l2_bias = nn.Linear(hidden_dim, 84)
#         self.l3_weights = nn.Linear(hidden_dim, self.out_dim * 84)
#         self.l3_bias = nn.Linear(hidden_dim, self.out_dim)
#
#         if spec_norm:
#             self.c1_weights = spectral_norm(self.c1_weights)
#             self.c1_bias = spectral_norm(self.c1_bias)
#             self.c2_weights = spectral_norm(self.c2_weights)
#             self.c2_bias = spectral_norm(self.c2_bias)
#             self.l1_weights = spectral_norm(self.l1_weights)
#             self.l1_bias = spectral_norm(self.l1_bias)
#             self.l2_weights = spectral_norm(self.l2_weights)
#             self.l2_bias = spectral_norm(self.l2_bias)
#             self.l3_weights = spectral_norm(self.l3_weights)
#             self.l3_bias = spectral_norm(self.l3_bias)
#
#     def forward(self, v):
#         batch_dims = v.shape[:-1]
#         # emd = self.embeddings(idx)
#         features = self.mlp(v)
#
#         weights = {
#             "conv1.weight": self.c1_weights(features).view(*batch_dims, self.n_kernels, self.in_channels, 5, 5),
#             "conv1.bias": self.c1_bias(features).view(*batch_dims, self.n_kernels),
#             "conv2.weight": self.c2_weights(features).view(*batch_dims, 2 * self.n_kernels, self.n_kernels, 5, 5),
#             "conv2.bias": self.c2_bias(features).view(*batch_dims, 2 * self.n_kernels),
#             "fc1.weight": self.l1_weights(features).view(*batch_dims, 120, 2 * self.n_kernels * 5 * 5),
#             "fc1.bias": self.l1_bias(features).view(*batch_dims, 120),
#             "fc2.weight": self.l2_weights(features).view(*batch_dims, 84, 120),
#             "fc2.bias": self.l2_bias(features).view(*batch_dims, 84),
#             "fc3.weight": self.l3_weights(features).view(*batch_dims, self.out_dim, 84),
#             "fc3.bias": self.l3_bias(features).view(*batch_dims, self.out_dim),
#         }
#         return weights


# class CNNEmbed(nn.Module):
#     def __init__(self, embed_y, dim_y, embed_dim, device=None, in_channels=3, n_kernels=16):
#         super().__init__()
#
#         in_channels += embed_y * dim_y
#
#         self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
#         self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, embed_dim)
#
#         self.embed_y = embed_y
#
#     def forward(self, x, y):
#         batch_dims = y.shape[:-1]
#         if self.embed_y:
#             c = y[..., None, None].expand(*((-1,) * len(batch_dims)), -1, *x.shape[-2:])
#             inp = torch.cat((x, c), dim=-3)
#         else:
#             inp = x
#
#         x = self.pool(F.relu(self.conv1(inp)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(*batch_dims, -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class Hyper(nn.Module):
    def __init__(
            self, example_state_dict: Dict[str, torch.Tensor], embedding_dim=26, hidden_dim=100, n_hidden=1,
            spec_norm=False,
    ):
        super().__init__()

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        layer_param_dict = {}
        layer_shape_dict = {}
        for key, value in example_state_dict.items():
            if spec_norm:
                layer_param_dict[key] = spectral_norm(nn.Linear(hidden_dim, value.numel()))
            else:
                layer_param_dict[key] = nn.Linear(hidden_dim, value.numel())
            layer_shape_dict[key] = value.shape
        self.layer_dict = nn.ParameterDict(layer_param_dict)
        self.shape_dict = layer_shape_dict

    def forward(self, v):
        batch_dims = v.shape[:-1]
        features = self.mlp(v)

        weights = {
            key: self.layer_dict[key](features).view(*batch_dims, *self.shape_dict[key])
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
