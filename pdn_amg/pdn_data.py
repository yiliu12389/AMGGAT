import os, re, glob
import numpy as np
import scipy.sparse as sp
import torch

HERE  = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(HERE)
SOL   = os.path.join(REPO, 'ML-for-IR-drop/solution')
BENCH = os.path.join(REPO, 'ML-for-IR-drop/benchmarks')

G_PIN = 1000.0
_NAME_RE = re.compile(r'n(\d+)_m(\d+)_(\d+)_(\d+)')


# ─────────────────────────────────────────────────────────────────────────────
# SPICE netlist -> sparse conductance system
# ─────────────────────────────────────────────────────────────────────────────
def parse_spice(path):
    node_idx, R_edges, V_pins, I_loads = {}, [], {}, {}
    def gidx(name):
        if name == '0':
            return -1
        if name not in node_idx:
            node_idx[name] = len(node_idx)
        return node_idx[name]
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in '*.':
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            tag = parts[0][0].upper()
            if tag == 'R':
                i, j, R = gidx(parts[1]), gidx(parts[2]), float(parts[3])
                if R > 0:
                    R_edges.append((i, j, 1.0 / R))
            elif tag == 'V':
                i, j, val = gidx(parts[1]), gidx(parts[2]), float(parts[3])
                if j == -1:
                    V_pins[i] = val
                elif i == -1:
                    V_pins[j] = -val
            elif tag == 'I':
                i, j, val = gidx(parts[1]), gidx(parts[2]), float(parts[3])
                if i != -1:
                    I_loads[i] = I_loads.get(i, 0.0) - val
                if j != -1:
                    I_loads[j] = I_loads.get(j, 0.0) + val

    N = len(node_idx)
    names = [None] * N
    for k, v in node_idx.items():
        names[v] = k

    rows, cols, vals = [], [], []
    for i, j, G in R_edges:
        if i == -1 and j == -1:
            continue
        if i == -1:
            rows += [j]; cols += [j]; vals += [G]
        elif j == -1:
            rows += [i]; cols += [i]; vals += [G]
        else:
            rows += [i, j, i, j]
            cols += [i, j, j, i]
            vals += [G, G, -G, -G]

    b_vdd = np.zeros(N, dtype=np.float64)
    b_isource = np.zeros(N, dtype=np.float64)
    for i, I in I_loads.items():
        b_isource[i] += I
    for i, vdd in V_pins.items():
        rows.append(i); cols.append(i); vals.append(G_PIN)
        b_vdd[i] += G_PIN * vdd
    b = b_vdd + b_isource

    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    A.sum_duplicates()
    return A, b, b_vdd, b_isource, names


def load_gt(gt_path, names):
    N = len(names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    x_gt = np.zeros(N, dtype=np.float64)
    mask = np.zeros(N, dtype=bool)
    if not os.path.exists(gt_path):
        return x_gt, mask
    with open(gt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            i = name_to_idx.get(parts[0])
            if i is None:
                continue
            try:
                x_gt[i] = float(parts[1]); mask[i] = True
            except ValueError:
                continue
    return x_gt, mask


def parse_layer(names):
    out = np.zeros(len(names), dtype=np.int64)
    for i, nm in enumerate(names):
        if nm is None:
            continue
        m = _NAME_RE.match(nm)
        if m is not None:
            out[i] = int(m.group(2))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Graph / feature construction
# ─────────────────────────────────────────────────────────────────────────────
def build_graph(A_sp, b_vdd, device):
    A_coo = A_sp.tocoo()
    off = A_coo.row != A_coo.col
    i_idx = A_coo.row[off]
    j_idx = A_coo.col[off]
    w_off = -A_coo.data[off].astype(np.float32)
    N = A_sp.shape[0]
    row = j_idx.astype(np.int64)
    col = i_idx.astype(np.int64)

    w = np.log1p(np.maximum(w_off, 0.0))
    edge_feat = w.astype(np.float32)

    A_csr = A_sp.tocsr()
    A_t = torch.sparse_csr_tensor(
        torch.from_numpy(A_csr.indptr.astype(np.int64)),
        torch.from_numpy(A_csr.indices.astype(np.int64)),
        torch.from_numpy(A_csr.data.astype(np.float32)),
        size=A_csr.shape, device=device)
    is_pin = (b_vdd > 0).astype(np.bool_)
    return {
        'n':         N,
        'row':       torch.from_numpy(row).to(device),
        'col':       torch.from_numpy(col).to(device),
        'edge_feat': torch.from_numpy(edge_feat).to(device),
        'A':         A_t,
        'is_pin':    torch.from_numpy(is_pin).to(device),
    }


def sample_anchor_indices(is_pin, gt_mask, anchor_frac, rng):
    if anchor_frac <= 0 or gt_mask is None:
        return np.array([], dtype=np.int64)
    candidates = np.where(gt_mask & ~is_pin)[0]
    n_anchor = int(round(anchor_frac * candidates.size))
    if n_anchor == 0:
        return np.array([], dtype=np.int64)
    return rng.choice(candidates, size=n_anchor, replace=False)


def load_rw_anchor_cache(ds, name, af, M, seed):
    here = os.path.dirname(os.path.abspath(__file__))
    cp = os.path.join(here, 'data_cache', 'rw_anchors', ds,
                       f'{name}_af{af:g}_M{M}_s{seed}.npz')
    if not os.path.exists(cp):
        return None
    z = np.load(cp)
    return {'anchor_idx': z['anchor_idx'].astype(np.int64),
            'V_hat': z['V_hat'].astype(np.float64),
            'T_rw': float(z.get('T_rw', 0.0))}


def build_features(A_sp, b_vdd, b_isource, layer_idx, device,
                    x_gt=None, gt_mask=None, vdd=None,
                    anchor_frac=0.0, rng=None, anchor_indices=None,
                    encoding='v_ratio'):
    """encoding='v_ratio'    -> 5 features: [v_ratio, b_iso_n, diag_n, deg_n, z_norm]
       encoding='known_drop' -> 6 features: [known_drop_norm, is_known,
                                             b_iso_n, diag_n, deg_n, z_norm]
    """
    N = A_sp.shape[0]
    b_iso_f = b_isource.astype(np.float32)
    diag    = A_sp.diagonal().astype(np.float32)
    is_pin  = (b_vdd > 0)

    diag_log = np.log1p(np.maximum(diag, 0.0))
    diag_n = diag_log / (diag_log.max() + 1e-12)

    b_iso_n = b_iso_f / (np.abs(b_iso_f).max() + 1e-12)

    A_coo = A_sp.tocoo()
    off = A_coo.row != A_coo.col
    degree = np.zeros(N, dtype=np.float32)
    np.add.at(degree, A_coo.row[off], 1.0)
    deg_n = degree / (degree.max() + 1e-12)

    lmax = int(layer_idx.max()) if layer_idx.size else 0
    lmin = int(layer_idx.min()) if layer_idx.size else 0
    denom = max(lmax - lmin, 1)
    z_norm = (2.0 * (layer_idx - lmin) / denom - 1.0).astype(np.float32)

    is_anchor = np.zeros(N, dtype=np.bool_)
    if anchor_indices is not None and len(anchor_indices) > 0:
        is_anchor[anchor_indices] = True
    elif anchor_frac > 0 and x_gt is not None and gt_mask is not None:
        rng = rng if rng is not None else np.random.default_rng()
        chosen = sample_anchor_indices(is_pin, gt_mask, anchor_frac, rng)
        is_anchor[chosen] = True

    if encoding == 'known_drop':
        is_known = (is_pin | is_anchor).astype(np.float32)
        known_drop_mv = np.zeros(N, dtype=np.float32)
        if x_gt is not None and vdd is not None:
            know_mask = (is_pin | is_anchor)
            known_drop_mv[know_mask] = (
                (vdd - x_gt[know_mask]) * 1000.0).astype(np.float32)
        cols = [known_drop_mv, is_known, b_iso_n, diag_n, deg_n, z_norm]
    else:
        v_ratio = np.zeros(N, dtype=np.float32)
        v_ratio[is_pin] = 1.0
        if is_anchor.any() and x_gt is not None and vdd is not None:
            v_ratio[is_anchor] = (x_gt[is_anchor] / vdd).astype(np.float32)
        cols = [v_ratio, b_iso_n, diag_n, deg_n, z_norm]

    feats = np.stack(cols, axis=-1).astype(np.float32)
    return torch.from_numpy(feats).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Linear algebra utils
# ─────────────────────────────────────────────────────────────────────────────
def spmv(A, x):
    return torch.mv(A, x)


def rel_res(A, v, b):
    return ((b - spmv(A, v)).norm() / (b.norm() + 1e-15)).item()


# ─────────────────────────────────────────────────────────────────────────────
# Sample / dataset loading
# ─────────────────────────────────────────────────────────────────────────────
def load_sample(sp_file, sol_subdir, sample_name=None):
    A_sp, b_np, b_vdd, b_isource, names = parse_spice(sp_file)
    name = sample_name or os.path.splitext(os.path.basename(sp_file))[0]
    if name == 'netlist':
        name = os.path.basename(os.path.dirname(sp_file))
    gt_path = os.path.join(SOL, sol_subdir, f'{name}.solution')
    if not os.path.exists(gt_path):
        return None
    x_gt, mask = load_gt(gt_path, names)
    if mask.sum() == 0:
        return None
    layer_idx = parse_layer(names)
    vdd = float(b_vdd.max() / G_PIN) if (b_vdd > 0).any() else 1.0
    return (name, A_sp, b_np, b_vdd, b_isource, layer_idx, x_gt, mask, vdd)


def load_group(subdir):
    base = os.path.join(BENCH, subdir)
    out = []
    if not os.path.isdir(base):
        return out
    for tc in sorted(os.listdir(base)):
        spf = os.path.join(base, tc, 'netlist.sp')
        if not os.path.exists(spf):
            continue
        s = load_sample(spf, subdir, sample_name=tc)
        if s is not None:
            out.append(s)
    return out


def load_group_flat(subdir):
    base = os.path.join(BENCH, subdir)
    out = []
    for spf in sorted(glob.glob(os.path.join(base, '*.sp'))):
        s = load_sample(spf, subdir)
        if s is not None:
            out.append(s)
    return out