"""GNN that predicts the lower-triangular FSAI factor G of the unit-diagonal-scaled matrix A_hat.

Output: G_ii = softplus(raw + c) with softplus(c)=1 (so G=I, i.e. Jacobi, at init), G_ij = delta_ij for i>j.
M^-1 = (G S)^T (G S) with S = diag(d^-1/2) undoes the scaling. SPD by construction since diag(G) > 0.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv

_C = math.log(math.e - 1.0)  # softplus(_C) == 1


class PrecondGNN(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 64, layers: int = 3, residual: bool = True):
        """residual=True: output is G = I + dG (starts exactly at Jacobi). residual=False: ablation, raw softplus diagonal
        and default init (starts from an arbitrary G)."""
        super().__init__()
        self.residual = residual
        self.enc = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)), edge_dim=1)
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * hidden + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        if residual:
            nn.init.zeros_(self.head[-1].weight)  # start exactly at G = I (Jacobi)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, data):
        h = F.relu(self.enc(data.x))
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.relu(norm(conv(h, data.edge_index, data.edge_attr)))
        i, j = data.pred_row, data.pred_col
        feat = torch.cat([h[i], h[j], data.pred_val[:, None], data.pred_is_diag[:, None].float()], dim=1)
        raw = self.head(feat).squeeze(-1)
        return torch.where(data.pred_is_diag, F.softplus(raw + (_C if self.residual else 0.0)), raw)  # values of G on the pattern


def unscale_columns(vals, col, scale):
    """G_A = G_hat S : multiply entry (i,j) by d_j^-1/2."""
    return vals * scale[col]
