"""Coarse-to-fine AMG-GAT architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import GATLayer


class LevelGAT(nn.Module):
    def __init__(self, in_dim, head_dim, heads, n_layers):
        super().__init__()
        layers, dim = [], in_dim
        for _ in range(n_layers - 1):
            layers.append(GATLayer(dim, head_dim, heads=heads, concat=True,
                                   activation=F.elu))
            dim = head_dim * heads
        layers.append(GATLayer(dim, head_dim, heads=heads, concat=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, features, graph):
        hidden = features
        row, col, n_nodes = graph['row'], graph['col'], graph['n']
        for layer in self.layers:
            hidden = layer(hidden, row, col, n_nodes, graph['edge_feat'])
        return hidden


class A3GAT(nn.Module):
    """Standalone three-layer GAT that predicts only the A3 voltage drop."""

    def __init__(self, in_dim=5, head_dim=32, heads=4, n_layers=3):
        super().__init__()
        self.gat = LevelGAT(in_dim, head_dim, heads, n_layers)
        decoder_hidden = head_dim * 2
        self.decoder = nn.Sequential(
            nn.Linear(head_dim, decoder_hidden), nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, 1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, graph, features):
        """Return the normalized prediction of the A3 node drop u3."""
        h3 = self.gat(features, graph)
        return self.decoder(h3).squeeze(-1)


class AMGGAT(nn.Module):
    def __init__(self, in_dim=5, head_dim=32, heads=4,
                 layers_per_level=(3, 3, 3, 3)):
        super().__init__()
        self.n_levels = len(layers_per_level)
        self.head_dim = head_dim
        self.level_gats = nn.ModuleList([
            # The coarsest graph starts from its own five node features.
            # Every finer graph receives its own features plus P_l h_{l+1}.
            LevelGAT(in_dim if level == self.n_levels - 1 else in_dim + head_dim,
                     head_dim, heads, n_layers)
            for level, n_layers in enumerate(layers_per_level)])
        decoder_hidden = head_dim * 2
        self.dec = nn.Sequential(
            nn.Linear(head_dim, decoder_hidden), nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, 1))
        self.dec_coarsest = nn.Sequential(
            nn.Linear(head_dim, decoder_hidden), nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, 1))
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)
        nn.init.zeros_(self.dec_coarsest[-1].weight)
        nn.init.zeros_(self.dec_coarsest[-1].bias)

    def forward(self, graphs, features, prolongations, return_coarsest=False):
        """Predict L0 and, optionally, decode the standalone coarsest solution.

        For four AMG levels this starts with h3 = GAT3(F3), then decodes
        x3_pred = decoder3(h3).  The hidden h3 is prolongated into L2 only
        after this independent coarse prediction has been formed.
        """
        coarsest = self.n_levels - 1
        hidden = self.level_gats[coarsest](features[coarsest], graphs[coarsest])
        x_coarsest = self.dec_coarsest(hidden).squeeze(-1)

        for level in range(coarsest - 1, -1, -1):
            hidden = torch.sparse.mm(prolongations[level], hidden)
            hidden = torch.cat([features[level], hidden], dim=-1)
            hidden = self.level_gats[level](hidden, graphs[level])

        x_fine = self.dec(hidden).squeeze(-1)
        if return_coarsest:
            return x_fine, x_coarsest
        return x_fine
