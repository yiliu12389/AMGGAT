"""Ruge-Stüben hierarchy construction and level-wise transfers."""

import numpy as np
import pyamg
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from types import SimpleNamespace

from pyamg.classical.interpolate import classical_interpolation
from pyamg.classical.split import RS
from pyamg.strength import classical_strength_of_connection

from data import build_graph, build_features


def build_amg_hierarchy(matrix, max_levels=4, max_coarse=50, coarse_mask=None,
                        promote_bad_cpts=True, interpolation_tol=1e-8):
    """Build AMG levels and promote non-partitioning F points to C points.

    VDD-influenced nodes are forced C points first. After RS splitting, any
    F-point row whose classical interpolation weights do not sum to one is
    promoted to a C point and interpolation is rebuilt. This keeps the
    absolute-voltage formulation and increases the coarse set only where
    needed.
    """
    current_A = matrix.tocsr().astype(np.float64)
    current_mark = (np.zeros(current_A.shape[0], dtype=bool) if coarse_mask is None
                    else np.asarray(coarse_mask, dtype=bool).copy())
    if current_mark.size != current_A.shape[0]:
        raise ValueError('coarse_mask length must match matrix size')

    levels = []
    for level_index in range(max_levels):
        level = SimpleNamespace(A=current_A, splitting=None, P=None,
                                coarse_nodes=None)
        levels.append(level)
        if level_index == max_levels - 1 or current_A.shape[0] <= max_coarse:
            break
        # First fix every VDD-influenced node as a C-point.  RS then chooses
        # only among the remaining nodes, so the final splitting is defined
        # before interpolation is constructed.
        splitting = current_mark.copy()
        remaining = np.flatnonzero(~splitting)
        if remaining.size == 0:
            break
        strength = classical_strength_of_connection(current_A, theta=0.25)
        rs_remaining = RS(
            strength[remaining, :][:, remaining].tocsr(), second_pass=True)
        splitting[remaining] = np.asarray(rs_remaining, dtype=bool)
        promoted = 0
        for _ in range(8):
            coarse_nodes = np.flatnonzero(splitting)
            if coarse_nodes.size >= current_A.shape[0]:
                break
            P = classical_interpolation(
                current_A, strength, splitting.astype(np.int32)).tocsr()
            P.eliminate_zeros()
            if not promote_bad_cpts:
                break
            row_sum = np.asarray(P.sum(axis=1)).ravel()
            bad = np.flatnonzero((~splitting) &
                                 (row_sum < 1.0 - interpolation_tol))
            if bad.size == 0:
                break
            splitting[bad] = True
            promoted += int(bad.size)
        coarse_nodes = np.flatnonzero(splitting)
        if P.shape[1] >= current_A.shape[0]:
            break
        level.splitting = splitting
        level.P = P
        if coarse_nodes.size != P.shape[1]:
            raise RuntimeError('C-point ordering does not match AMG coarse columns')
        level.coarse_nodes = coarse_nodes
        current_mark = np.asarray(
            abs(P).T @ current_mark.astype(np.float64)).ravel() > 0
        current_A = (P.T @ current_A @ P).tocsr()
        current_A.eliminate_zeros()
    return levels


def solve_coarse_level_systems(amg_levels, rhs_fine):
    """Solve each coarse system A_l x_l = b_l for levels l >= 1.

    The coarse right-hand side is restricted recursively with P.T. Returns
    dictionaries containing the level index, restricted RHS, and solution.
    Solutions have the same units as the original system's unknowns.
    """
    if len(amg_levels) < 2:
        return []

    rhs = np.asarray(rhs_fine, dtype=np.float64).reshape(-1)
    if rhs.size != amg_levels[0].A.shape[0]:
        raise ValueError('rhs_fine length must match the finest AMG matrix')

    results = []
    for level_index in range(1, len(amg_levels)):
        prolongation = amg_levels[level_index - 1].P
        if prolongation is None:
            raise ValueError(f'Missing prolongation P_{level_index - 1}')
        rhs = np.asarray(prolongation.T @ rhs).reshape(-1)
        matrix = amg_levels[level_index].A.tocsc()
        solution = np.asarray(spsolve(matrix, rhs)).reshape(-1)
        if not np.isfinite(solution).all():
            raise RuntimeError(f'Non-finite solution at AMG level {level_index}')
        results.append({
            'level': level_index,
            'rhs': rhs,
            'x': solution,
        })
    return results


def sp_to_torch(matrix, device):
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(
        indices, values, matrix.shape, device=device).coalesce()


def prep_hierarchy(amg_levels, b_vdd_L0, b_isource_L0, layer_idx_L0, device):
    """Build per-level graphs/features and prolongation tensors."""
    n_levels = len(amg_levels)
    b_vdd_levels = [None] * n_levels
    b_isource_levels = [None] * n_levels
    layer_levels = [None] * n_levels
    graphs = [None] * n_levels
    prolongations = []

    b_vdd_levels[0] = b_vdd_L0
    b_isource_levels[0] = b_isource_L0
    layer_levels[0] = layer_idx_L0
    graphs[0] = build_graph(amg_levels[0].A, device)

    for level in range(n_levels - 1):
        prolongation = amg_levels[level].P
        restriction = prolongation.T
        prolongations.append(sp_to_torch(prolongation, device))
        b_isource_levels[level + 1] = np.asarray(
            restriction @ b_isource_levels[level]).ravel()
        b_vdd_levels[level + 1] = np.asarray(
            restriction @ b_vdd_levels[level]).ravel()
        coarse_nodes = amg_levels[level].coarse_nodes
        if coarse_nodes is None:
            raise RuntimeError('AMG level is missing its coarse-node mapping')
        layer_levels[level + 1] = layer_levels[level][coarse_nodes].copy()
        graphs[level + 1] = build_graph(amg_levels[level + 1].A, device)

    coarse_features = [None] * n_levels
    base_layers = np.asarray(layer_idx_L0, dtype=np.float32)
    layer_bounds = ((float(base_layers.min()), float(base_layers.max()))
                    if base_layers.size else (0.0, 0.0))
    for level in range(1, n_levels):
        coarse_features[level] = build_features(
            amg_levels[level].A, b_vdd_levels[level],
            b_isource_levels[level], layer_levels[level], device,
            layer_bounds=layer_bounds)

    return (graphs, coarse_features, prolongations, b_vdd_levels,
            b_isource_levels, layer_levels)
