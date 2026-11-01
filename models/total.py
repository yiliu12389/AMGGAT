import torch
import torch.nn as nn

from models.coarsetofine import LevelGAT


class TotalAMGGAT(nn.Module):
    """Fine-to-coarse transport followed by AMG-style hidden correction."""

    def __init__(self, n_levels=4, feature_dim=5, head_dim=32, heads=4,
                 gat_layers=3):
        super().__init__()
        self.n_levels = n_levels
        self.head_dim = head_dim
        self.embeddings = nn.ModuleList([
            nn.Linear(feature_dim, head_dim)
            for _ in range(n_levels)
        ])
        self.level_gats = nn.ModuleList([
            LevelGAT(head_dim, head_dim, heads, gat_layers)
            for _ in range(n_levels)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(head_dim, 2 * head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * head_dim, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def _restrict(prolongation, hidden):
        return torch.sparse.mm(
            prolongation.transpose(0, 1).coalesce(), hidden)

    def forward(self, graphs, features, prolongations, return_trace=False):
        if len(graphs) != self.n_levels or len(features) != self.n_levels:
            raise ValueError('graphs/features count must equal n_levels')
        if len(prolongations) != self.n_levels - 1:
            raise ValueError('need one prolongation between adjacent levels')

        mapped = [embedding(feature) for embedding, feature in zip(
            self.embeddings, features)]
        down_hidden = [self.level_gats[0](mapped[0], graphs[0])]
        for level in range(1, self.n_levels):
            restricted = self._restrict(
                prolongations[level - 1], down_hidden[level - 1])
            down_hidden.append(self.level_gats[level](
                mapped[level] + restricted, graphs[level]))

        up_hidden = [None] * self.n_levels
        up_hidden[-1] = down_hidden[-1]
        corrections = [None] * (self.n_levels - 1)
        for level in range(self.n_levels - 2, -1, -1):
            prolongation = prolongations[level]
            restricted_fine = self._restrict(
                prolongation, down_hidden[level])
            coarse_on_fine = torch.sparse.mm(
                prolongation, up_hidden[level + 1])
            corrected = (down_hidden[level] -
                         torch.sparse.mm(prolongation, restricted_fine) +
                         coarse_on_fine)
            corrections[level] = corrected
            up_hidden[level] = self.level_gats[level](
                corrected, graphs[level])

        prediction = self.decoder(up_hidden[0]).squeeze(-1)
        if return_trace:
            return prediction, {
                'down_hidden': down_hidden,
                'corrections': corrections,
                'up_hidden': up_hidden,
            }
        return prediction


TotalVCycle = TotalAMGGAT
