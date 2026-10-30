"""Vanilla multi-layer GAT model."""

import torch.nn as nn
import torch.nn.functional as F

from models.layers import GATLayer


class GAT(nn.Module):
    def __init__(self, in_dim=6, head_dim=32, heads=4, n_layers=6):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(n_layers - 1):
            layers.append(GATLayer(dim, head_dim, heads=heads, concat=True,
                                   activation=F.elu))
            dim = head_dim * heads
        layers.append(GATLayer(dim, head_dim, heads=heads, concat=False))
        self.layers = nn.ModuleList(layers)
        decoder_hidden = head_dim * 2
        self.dec = nn.Sequential(
            nn.Linear(head_dim, decoder_hidden), nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, 1))
        nn.init.xavier_uniform_(self.dec[0].weight)
        nn.init.zeros_(self.dec[0].bias)
        nn.init.normal_(self.dec[2].weight, std=0.01)
        nn.init.zeros_(self.dec[2].bias)

    def forward(self, graph, features):
        row, col, n_nodes = graph['row'], graph['col'], graph['n']
        edge_feat = graph['edge_feat']
        hidden = features
        for layer in self.layers:
            hidden = layer(hidden, row, col, n_nodes, edge_feat)
        return self.dec(hidden).squeeze(-1)
