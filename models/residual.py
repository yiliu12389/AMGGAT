
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from data.features import build_features_from_rhs
from models.coarsetofine import LevelGAT


class ResidualVCycle(nn.Module):

    def __init__(self, n_levels=4, feature_dim=5, head_dim=32, heads=4,
                 gat_layers=3):
        super().__init__()
        if n_levels < 2:
            raise ValueError('residual V-cycle needs at least two levels')
        self.n_levels = n_levels
        self.input_dim = feature_dim
        self.level_gats = nn.ModuleList([
            LevelGAT(feature_dim, head_dim, heads, gat_layers)
            for _ in range(n_levels)
        ])
        self.down_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_dim, 2 * head_dim),
                nn.ReLU(inplace=True),
                nn.Linear(2 * head_dim, 1),
            )
            for _ in range(n_levels)
        ])
        self.up_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_dim, 2 * head_dim),
                nn.ReLU(inplace=True),
                nn.Linear(2 * head_dim, 1),
            )
            for _ in range(n_levels)
        ])
        nn.init.xavier_uniform_(self.down_decoders[0][-1].weight)
        nn.init.zeros_(self.down_decoders[0][-1].bias)
        for decoder in self.down_decoders[1:]:
            nn.init.zeros_(decoder[-1].weight)
            nn.init.zeros_(decoder[-1].bias)
        for decoder in self.up_decoders:
            nn.init.zeros_(decoder[-1].weight)
            nn.init.zeros_(decoder[-1].bias)

    def _solve_level(self, level, graph, rhs, layer_idx, feature_stats,
                     layer_bounds, decode_fn, initial=False, upward=False):
        features = build_features_from_rhs(
            graph, rhs, layer_idx[level], feature_stats, layer_bounds)
        if self.training:
            hidden = checkpoint(
                lambda x: self.level_gats[level](x, graph),
                features, use_reentrant=False)
        else:
            hidden = self.level_gats[level](features, graph)
        decoders = self.up_decoders if upward else self.down_decoders
        raw = decoders[level](hidden).squeeze(-1)
        if decode_fn is not None:
            value = decode_fn(level, raw, initial)
        else:
            value = raw
        return value, features

    def forward(self, graphs, layer_idx, rhs0, prolongations,
                node_stats=None, layer_bounds=None, decode_fn=None,
                return_trace=False, up_node_stats=None):
        if len(graphs) != self.n_levels or len(layer_idx) != self.n_levels:
            raise ValueError('graphs/layer_idx count must equal n_levels')
        if len(prolongations) != self.n_levels - 1:
            raise ValueError('need one prolongation between adjacent levels')
        if up_node_stats is None:
            up_node_stats = node_stats
        rhs0 = rhs0.reshape(-1)
        x0, f0 = self._solve_level(
            0, graphs[0], rhs0, layer_idx,
            None if node_stats is None else node_stats[0], layer_bounds,
            decode_fn, initial=True)
        residuals = [rhs0 - torch.sparse.mm(
            graphs[0]['A'], x0.unsqueeze(-1)).squeeze(-1)]
        values = [x0]
        rhs_levels = [rhs0]
        downward_features = [f0]


        for level in range(1, self.n_levels):
            rhs_level = torch.sparse.mm(
                prolongations[level - 1].transpose(0, 1).coalesce(),
                residuals[-1].unsqueeze(-1)).squeeze(-1)
            value, level_features = self._solve_level(
                level, graphs[level], rhs_level, layer_idx,
                None if node_stats is None else node_stats[level],
                layer_bounds, decode_fn)
            values.append(value)
            rhs_levels.append(rhs_level)
            downward_features.append(level_features)
            residuals.append(rhs_level - torch.sparse.mm(
                graphs[level]['A'], value.unsqueeze(-1)).squeeze(-1))


        correction = values[-1]
        upward_values = [None] * self.n_levels
        upward_features = [None] * self.n_levels
        upward_values[-1] = correction
        for level in range(self.n_levels - 2, 0, -1):
            corrected_level = values[level] + torch.sparse.mm(
                prolongations[level], correction.unsqueeze(-1)
            ).squeeze(-1)
            corrected_residual = rhs_levels[level] - torch.sparse.mm(
                graphs[level]['A'], corrected_level.unsqueeze(-1)
            ).squeeze(-1)
            post_correction, level_features = self._solve_level(
                level, graphs[level], corrected_residual, layer_idx,
                None if up_node_stats is None else up_node_stats[level],
                layer_bounds, decode_fn, upward=True)
            correction = corrected_level + post_correction
            upward_values[level] = correction
            upward_features[level] = level_features

        corrected_x0 = x0 + torch.sparse.mm(
            prolongations[0], correction.unsqueeze(-1)
        ).squeeze(-1)
        post_residual = rhs0 - torch.sparse.mm(
            graphs[0]['A'], corrected_x0.unsqueeze(-1)
        ).squeeze(-1)

        post_correction, level_features = self._solve_level(
            0, graphs[0], post_residual, layer_idx,
            None if up_node_stats is None else up_node_stats[0],
            layer_bounds, decode_fn, upward=True)
        x = corrected_x0 + post_correction
        upward_values[0] = x
        upward_features[0] = level_features

        if return_trace:
            return x, {
                'level_values': values,
                'pre_residuals': residuals,
                'rhs_levels': rhs_levels,
                'post_residual': post_residual,
                'upward_values': upward_values,
                'downward_features': downward_features,
                'upward_features': upward_features,
            }
        return x


SharedResidualVCycle = ResidualVCycle
ResidualAMGGAT = ResidualVCycle
