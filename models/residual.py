
"""Shared-parameter residual V-cycle GAT."""

import torch
import torch.nn as nn

from models.coarsetofine import LevelGAT


class SharedResidualVCycle(nn.Module):
    """Four-level residual V-cycle with one shared GAT and decoder.

    Residuals use r=A*x-b. A decoded residual solve is subtracted from the
    current solution. The residual feature callback appends one normalized
    residual channel to each level's five base features.
    """

    def __init__(self, n_levels=4, feature_dim=5, head_dim=32, heads=4,
                 gat_layers=3):
        super().__init__()
        if n_levels < 2:
            raise ValueError('residual V-cycle needs at least two levels')
        self.n_levels = n_levels
        self.input_dim = feature_dim + 1
        self.gat = LevelGAT(self.input_dim, head_dim, heads, gat_layers)
        self.decoder = nn.Sequential(
            nn.Linear(head_dim, 2 * head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * head_dim, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def _call(self, level, graph, features, residual, feature_fn, decode_fn):
        x = feature_fn(level, features, residual).to(features.dtype)
        h = self.gat(x, graph)
        raw = self.decoder(h).squeeze(-1)
        return decode_fn(level, raw) if decode_fn is not None else raw

    def forward(self, graphs, features, rhs0, prolongations,
                residual_feature_fn, decode_fn=None, return_trace=False):
        if len(graphs) != self.n_levels or len(features) != self.n_levels:
            raise ValueError('graphs/features count must equal n_levels')
        if len(prolongations) != self.n_levels - 1:
            raise ValueError('need one prolongation between adjacent levels')
        if residual_feature_fn is None:
            raise ValueError('residual_feature_fn is required')

        rhs0 = rhs0.reshape(-1)
        x0 = self._call(
            0, graphs[0], features[0], torch.zeros_like(rhs0),
            residual_feature_fn, decode_fn)
        residuals = [torch.sparse.mm(
            graphs[0]['A'], x0.unsqueeze(-1)).squeeze(-1) - rhs0]
        values = [x0]

        # Restrict A0*x0-b0 and solve a residual equation at every level.
        for level in range(1, self.n_levels):
            rhs = torch.sparse.mm(
                prolongations[level - 1].transpose(0, 1).coalesce(),
                residuals[-1].unsqueeze(-1)).squeeze(-1)
            e = self._call(
                level, graphs[level], features[level], rhs,
                residual_feature_fn, decode_fn)
            values.append(e)
            residuals.append(torch.sparse.mm(
                graphs[level]['A'], e.unsqueeze(-1)).squeeze(-1) - rhs)

        # Return one level at a time. At each level, combine the current
        # correction with the prolongated finer correction, then post-smooth
        # that level using the same shared GAT/decoder.
        # L3 is the deepest solve. It is immediately prolongated to L2;
        # there is no extra post-smoothing call at the deepest level.
        correction = values[-1]
        upward_values = [None] * self.n_levels
        upward_posts = [None] * self.n_levels
        upward_values[-1] = correction
        for level in range(self.n_levels - 2, 0, -1):
            correction = values[level] - torch.sparse.mm(
                prolongations[level], correction.unsqueeze(-1)
            ).squeeze(-1)
            upward_values[level] = correction
            correction_residual = torch.sparse.mm(
                graphs[level]['A'], correction.unsqueeze(-1)
            ).squeeze(-1) - (
                torch.sparse.mm(
                    prolongations[level - 1].transpose(0, 1).coalesce(),
                    residuals[level - 1].unsqueeze(-1)
                ).squeeze(-1)
                if level > 0 else residuals[level]
            )
            post = self._call(
                level, graphs[level], features[level], correction_residual,
                residual_feature_fn, decode_fn)
            correction = correction - post
            upward_posts[level] = post

        x = x0 - torch.sparse.mm(
            prolongations[0], correction.unsqueeze(-1)
        ).squeeze(-1)

        # Final fine-level post-smoothing, also with shared parameters.
        post_residual = torch.sparse.mm(
            graphs[0]['A'], x.unsqueeze(-1)).squeeze(-1) - rhs0
        post = self._call(
            0, graphs[0], features[0], post_residual,
            residual_feature_fn, decode_fn)
        x = x - post
        upward_values[0] = x
        upward_posts[0] = post

        if return_trace:
            return x, {
                'level_values': values,
                'pre_residuals': residuals,
                'post_residual': post_residual,
                'post_correction': post,
                'upward_values': upward_values,
                'upward_posts': upward_posts,
            }
        return x


ResidualAMGGAT = SharedResidualVCycle
