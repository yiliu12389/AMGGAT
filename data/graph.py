"""Convert sparse conductance matrices into GAT graph inputs."""

import numpy as np
import torch


def build_graph(matrix, device):
    coo = matrix.tocoo()
    off_diagonal = coo.row != coo.col
    dst, src = coo.row[off_diagonal], coo.col[off_diagonal]
    conductance = -coo.data[off_diagonal].astype(np.float32)
    n_nodes = matrix.shape[0]
    edge_features = np.log1p(np.maximum(conductance, 0.0)).astype(np.float32)
    csr = matrix.tocsr()
    matrix_torch = torch.sparse_csr_tensor(
        torch.from_numpy(csr.indptr.astype(np.int64)),
        torch.from_numpy(csr.indices.astype(np.int64)),
        torch.from_numpy(csr.data.astype(np.float32)),
        size=csr.shape, device=device)
    return {
        'n': n_nodes,
        'row': torch.from_numpy(src.astype(np.int64)).to(device),
        'col': torch.from_numpy(dst.astype(np.int64)).to(device),
        'edge_feat': torch.from_numpy(edge_features).to(device),
        'A': matrix_torch,
    }


def spmv(matrix, vector):
    return torch.mv(matrix, vector)


def rel_res(matrix, voltage, rhs):
    return ((rhs - spmv(matrix, voltage)).norm() / (rhs.norm() + 1e-15)).item()
