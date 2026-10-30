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
