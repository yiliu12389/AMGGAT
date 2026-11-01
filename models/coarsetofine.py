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
        if self.n_levels != 4:
            raise ValueError('AMGGAT coarse-to-fine model requires four levels')
        self.head_dim = head_dim
        self.feature_maps = nn.ModuleList([
            nn.Linear(in_dim, head_dim) for _ in range(self.n_levels)
        ])
        # L3 only receives its own mapped features.  Each finer level
        # receives a concatenation of its own mapped features and the
        # prolongated hidden state from the immediately coarser level.
        gat_input_dims = [2 * head_dim, 2 * head_dim, 2 * head_dim, head_dim]
        self.level_gats = nn.ModuleList([
            LevelGAT(gat_input_dims[level], head_dim, heads, n_layers)
            for level, n_layers in enumerate(layers_per_level)
        ])
        decoder_hidden = head_dim * 2
        self.dec = nn.Sequential(
            nn.Linear(head_dim, decoder_hidden), nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, 1))
        nn.init.zeros_(self.dec[-1].weight)
        nn.init.zeros_(self.dec[-1].bias)

    def forward(self, graphs, features, prolongations, return_trace=False):
        if len(graphs) != 4 or len(features) != 4:
            raise ValueError('graphs/features must contain L0, L1, L2, L3')
        if len(prolongations) != 3:
            raise ValueError('prolongations must contain P0, P1, P2')

        mapped = [feature_map(feature) for feature_map, feature in zip(
            self.feature_maps, features)]
        h3 = self.level_gats[3](mapped[3], graphs[3])

        h3_on_l2 = torch.sparse.mm(prolongations[2], h3)
        h2_input = torch.cat([mapped[2], h3_on_l2], dim=-1)
        h2 = self.level_gats[2](h2_input, graphs[2])

        h2_on_l1 = torch.sparse.mm(prolongations[1], h2)
        h1_input = torch.cat([mapped[1], h2_on_l1], dim=-1)
        h1 = self.level_gats[1](h1_input, graphs[1])

        h1_on_l0 = torch.sparse.mm(prolongations[0], h1)
        h0_input = torch.cat([mapped[0], h1_on_l0], dim=-1)
        h0 = self.level_gats[0](h0_input, graphs[0])
        prediction = self.dec(h0).squeeze(-1)

        if return_trace:
            return prediction, {
                'mapped_features': mapped,
                'h3': h3,
                'h3_on_l2': h3_on_l2,
                'h2_input': h2_input,
                'h2': h2,
                'h2_on_l1': h2_on_l1,
                'h1_input': h1_input,
                'h1': h1,
                'h1_on_l0': h1_on_l0,
                'h0_input': h0_input,
                'h0': h0,
            }
        return prediction
