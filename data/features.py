"""Construct per-node AMG features from the matrix and restricted RHS."""

import numpy as np
import torch


def build_features(matrix, b_vdd, b_isource, layer_idx, device,
                   layer_bounds=None):
    """Build [VDD RHS, current RHS, log1p(Aii), log1p(degree), layer[-1,1]]."""
    n_nodes = matrix.shape[0]
    current = b_isource.astype(np.float32)
    diagonal = matrix.diagonal().astype(np.float32)
    coo = matrix.tocoo()
    off_diagonal = coo.row != coo.col
    degree = np.zeros(n_nodes, dtype=np.float32)
    np.add.at(degree, coo.row[off_diagonal], 1.0)
    diagonal_log = np.log1p(np.maximum(diagonal, 0.0))
    degree_log = np.log1p(degree)
    boundary_rhs = b_vdd.astype(np.float32)
    layer = np.asarray(layer_idx, dtype=np.float32)
    if layer_bounds is None:
        layer_min = float(layer.min()) if layer.size else 0.0
        layer_max = float(layer.max()) if layer.size else 0.0
    else:
        layer_min, layer_max = map(float, layer_bounds)
    if layer_max > layer_min:
        layer = 2.0 * (layer - layer_min) / (layer_max - layer_min) - 1.0
    else:
        layer = np.zeros_like(layer)
    columns = [boundary_rhs, current, diagonal_log, degree_log, layer]

    return torch.from_numpy(np.stack(columns, axis=-1).astype(np.float32)).to(device)


def build_features_from_rhs(graph, rhs, layer_idx, node_stats=None,
                            layer_bounds=None):
    rhs = torch.as_tensor(rhs, device=graph['A'].device,
                          dtype=graph['A'].dtype).reshape(-1)
    n_nodes = graph['n']
    if rhs.numel() != n_nodes:
        raise ValueError('rhs length must match the graph node count')

    matrix = graph['A'].to_sparse_coo().coalesce()
    row, col = matrix.indices()
    values = matrix.values()
    diagonal = torch.zeros(n_nodes, device=rhs.device, dtype=rhs.dtype)
    diagonal_mask = row == col
    diagonal.scatter_add_(0, row[diagonal_mask], values[diagonal_mask])
    degree = torch.zeros(n_nodes, device=rhs.device, dtype=rhs.dtype)
    off_diagonal = ~diagonal_mask
    degree.scatter_add_(0, row[off_diagonal], torch.ones_like(values[off_diagonal]))

    layer = torch.as_tensor(layer_idx, device=rhs.device,
                            dtype=rhs.dtype).reshape(-1)
    if layer.numel() != n_nodes:
        raise ValueError('layer index length must match the graph node count')
    if layer_bounds is None:
        layer_min = layer.min()
        layer_max = layer.max()
    else:
        layer_min = torch.as_tensor(layer_bounds[0], device=rhs.device,
                                    dtype=rhs.dtype)
        layer_max = torch.as_tensor(layer_bounds[1], device=rhs.device,
                                    dtype=rhs.dtype)
    if bool(layer_max > layer_min):
        layer = 2.0 * (layer - layer_min) / (layer_max - layer_min) - 1.0
    else:
        layer = torch.zeros_like(layer)

    features = torch.stack([
        rhs.clamp_min(0.0),
        rhs.clamp_max(0.0),
        torch.log1p(diagonal.clamp_min(0.0)),
        torch.log1p(degree),
        layer,
    ], dim=-1)
    if node_stats is not None:
        mean = torch.as_tensor(node_stats['mean'], device=rhs.device,
                               dtype=rhs.dtype)
        std = torch.as_tensor(node_stats['std'], device=rhs.device,
                              dtype=rhs.dtype)
        features[:, :4] = (features[:, :4] - mean) / std.clamp_min(1e-8)
    return features
