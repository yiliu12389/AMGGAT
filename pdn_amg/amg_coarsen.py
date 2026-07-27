"""AMG hierarchy construction utilities using pyamg Ruge-Stüben."""

import numpy as np
import torch
import pyamg

from pdn_data import build_graph, build_features


def build_amg_hierarchy(A_sp, max_levels=3, max_coarse=50):
    """RS-AMG on A_sp. Returns pyamg multilevel object's levels list (finest first).
    Each level has: .A (scipy CSR), .P (scipy CSR, N_l × N_{l+1}), absent on coarsest.
    """
    A_csr = A_sp.tocsr().astype(np.float64)
    ml = pyamg.ruge_stuben_solver(
        A_csr, max_levels=max_levels, max_coarse=max_coarse, keep=True)
    return ml.levels


def sp_to_torch(P_sp, device):
    """Scipy sparse → torch sparse COO float32."""
    c = P_sp.tocoo()
    idx = torch.from_numpy(np.vstack([c.row, c.col]).astype(np.int64))
    val = torch.from_numpy(c.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, P_sp.shape,
                                    device=device).coalesce()


def prep_hierarchy(amg_levels, b_vdd_L0, b_isource_L0, layer_idx_L0, device):
    """
    Build per-level graphs, structural features (no GT), and P tensors.

    Returns
    -------
    graphs      : list of graph dicts, one per level (L0 first)
    coarse_feats: list of feature tensors; index 0 is None (fine level built
                  per-step during training), index 1+ are fixed coarse features
    P_tensors   : P_tensors[l] sparse (N_l, N_{l+1}), l = 0 .. L-2
    b_vdd_ls    : per-level b_vdd numpy arrays
    b_iso_ls    : per-level b_isource numpy arrays
    li_ls       : per-level layer_idx numpy arrays
    """
    L = len(amg_levels)
    b_vdd_ls  = [None] * L
    b_iso_ls  = [None] * L
    li_ls     = [None] * L
    graphs    = [None] * L
    P_tensors = []

    b_vdd_ls[0] = b_vdd_L0
    b_iso_ls[0] = b_isource_L0
    li_ls[0]    = layer_idx_L0
    graphs[0]   = build_graph(amg_levels[0].A, b_vdd_L0, device)

    for l in range(L - 1):
        P  = amg_levels[l].P        # scipy sparse (N_l, N_{l+1})
        RT = P.T                    # restriction (N_{l+1}, N_l)
        P_tensors.append(sp_to_torch(P, device))

        bv_c = np.asarray(RT @ b_vdd_ls[l]).ravel()
        bi_c = np.asarray(RT @ b_iso_ls[l]).ravel()
        li_c = np.round(
            np.asarray(RT @ li_ls[l].astype(np.float64)).ravel()
        ).astype(np.int64)

        b_vdd_ls[l + 1] = bv_c
        b_iso_ls[l + 1] = bi_c
        li_ls[l + 1]    = li_c
        graphs[l + 1]   = build_graph(amg_levels[l + 1].A, bv_c, device)

    coarse_feats = [None] * L
    for l in range(1, L):
        coarse_feats[l] = build_features(
            amg_levels[l].A, b_vdd_ls[l], b_iso_ls[l], li_ls[l], device)

    return graphs, coarse_feats, P_tensors, b_vdd_ls, b_iso_ls, li_ls
