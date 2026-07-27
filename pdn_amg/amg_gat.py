"""
amg_gat.py — Hierarchical AMG+GAT for PDN IR-drop prediction.

Architecture (top-down fusion):
  Each AMG level runs an independent GAT → hidden state h_l.
  After all levels finish, coarse h is prolonged and added (with learnable
  projection + LN) to the finer h:
      h1 ← LN(h1 + proj(P_12 @ h2))
      h0 ← LN(h0 + proj(P_01 @ h1))
  Decoder runs on h0 → IR-drop prediction.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pdn_data import (build_features, spmv, load_group, load_group_flat,
                       sample_anchor_indices)
from gat_baseline import GATLayer
from amg_coarsen import build_amg_hierarchy, prep_hierarchy
from metrics import print_group_table, format_epoch_line
from metrics import metrics_A, metrics_B


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class LevelGAT(nn.Module):
    """Independent GAT for one AMG level."""
    def __init__(self, in_dim, head_dim, heads, n_layers):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers - 1):
            layers.append(GATLayer(d, head_dim, heads=heads,
                                   concat=True, activation=F.elu))
            d = head_dim * heads
        layers.append(GATLayer(d, head_dim, heads=heads,
                               concat=False, activation=None))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, graph):
        h = x
        row, col, N = graph['row'], graph['col'], graph['n']
        ef = graph['edge_feat']
        for layer in self.layers:
            h = layer(h, row, col, N, ef)
        return h  # (N, head_dim)


class AMGGAT(nn.Module):
    def __init__(self, in_dim=5, head_dim=32, heads=4,
                 layers_per_level=(4, 3, 2)):
        super().__init__()
        self.n_levels = len(layers_per_level)
        self.gnns = nn.ModuleList([
            LevelGAT(in_dim, head_dim, heads, layers_per_level[l])
            for l in range(self.n_levels)
        ])
        # one proj + LN per coarse→fine fusion step
        self.projs = nn.ModuleList([
            nn.Linear(head_dim, head_dim, bias=False)
            for _ in range(self.n_levels - 1)
        ])
        self.lns = nn.ModuleList([
            nn.LayerNorm(head_dim) for _ in range(self.n_levels - 1)
        ])
        dec_h = head_dim * 2
        self.dec = nn.Sequential(
            nn.Linear(head_dim, dec_h), nn.ReLU(inplace=True),
            nn.Linear(dec_h, 1), nn.Softplus(),
        )
        nn.init.xavier_uniform_(self.dec[0].weight)
        nn.init.zeros_(self.dec[0].bias)
        nn.init.normal_(self.dec[2].weight, std=0.01)
        nn.init.constant_(self.dec[2].bias, -6.5)
        for p in self.projs:
            nn.init.xavier_uniform_(p.weight)

    def forward(self, graphs, features, P_tensors):
        """
        graphs    : list of graph dicts [L0, L1, L2, ...]
        features  : list of feature tensors per level
        P_tensors : P_tensors[l] sparse (N_l, N_{l+1})
        """
        hs = [self.gnns[l](features[l], graphs[l])
              for l in range(self.n_levels)]

        # top-down: coarsest → finest
        for l in range(self.n_levels - 2, -1, -1):
            h_up = torch.sparse.mm(P_tensors[l], hs[l + 1])  # (N_l, head_dim)
            hs[l] = self.lns[l](hs[l] + self.projs[l](h_up))

        return self.dec(hs[0]).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing: build AMG hierarchy for every sample once
# ─────────────────────────────────────────────────────────────────────────────

def precompute(samples, device, max_levels=3):
    out = []
    for tc, A_sp, b_np, b_vdd, b_isource, li, x_gt, mask, vdd in samples:
        try:
            amg_lvls = build_amg_hierarchy(A_sp, max_levels=max_levels)
            n_actual = len(amg_lvls)
            graphs, cfeats, Ps, bv_ls, bi_ls, li_ls = prep_hierarchy(
                amg_lvls, b_vdd, b_isource, li, device)
            out.append(dict(
                tc=tc, A_sp=A_sp, b_np=b_np,
                b_vdd=b_vdd, b_isource=b_isource,
                layer_idx=li, x_gt=x_gt, mask=mask, vdd=vdd,
                graphs=graphs, cfeats=cfeats, Ps=Ps,
                bv_ls=bv_ls, bi_ls=bi_ls, li_ls=li_ls,
                n_levels=n_actual,
            ))
        except Exception as e:
            print(f'  precompute skip {tc}: {e}', flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────────────────────

def eval_amg(model, precomp, device, anchor_frac=0.0, anchor_seed=42,
             encoding='v_ratio', n_model_levels=3):
    model.eval()
    per_A, per_B = [], []
    rng = np.random.default_rng(anchor_seed)
    with torch.no_grad():
        for s in precomp:
            if s['n_levels'] < n_model_levels:
                continue
            try:
                is_pin = (s['b_vdd'] > 0)
                anc = sample_anchor_indices(
                    is_pin, s['mask'].astype(bool), anchor_frac, rng)
                x0 = build_features(
                    s['A_sp'], s['b_vdd'], s['b_isource'],
                    s['layer_idx'], device,
                    x_gt=s['x_gt'], gt_mask=s['mask'], vdd=s['vdd'],
                    anchor_indices=anc, encoding=encoding)
                feats = [x0] + s['cfeats'][1:n_model_levels]
                drop = model(s['graphs'][:n_model_levels],
                             feats, s['Ps'][:n_model_levels - 1])
                drop_gt_n = torch.from_numpy(
                    ((s['vdd'] - s['x_gt']) / s['vdd']).astype(np.float32)
                ).to(device)
                m  = torch.from_numpy(s['mask']).to(device)
                m1 = torch.from_numpy((s['layer_idx'] == 1) & s['mask']).to(device)
                mae_b, r2_b = metrics_B(drop[m], drop_gt_n[m], s['vdd'])
                per_B.append((mae_b, r2_b))
                if m1.sum() > 0:
                    mae_a, f1_a, cc_a = metrics_A(drop[m1], drop_gt_n[m1], s['vdd'])
                    per_A.append((mae_a, f1_a, cc_a))
            except Exception as e:
                print(f'  eval skip {s["tc"]}: {e}', flush=True)

    def _mean(xs):
        if not xs:
            return None
        a = np.array(xs, dtype=float)
        return tuple(a.mean(axis=0).tolist())

    return {'mean_A': _mean(per_A), 'mean_B': _mean(per_B),
            'tag': '', 'per_tc': []}


def format_line(r, label):
    b = r['mean_B']
    a = r['mean_A']
    s = f'{label}  MAE={b[0]:.4f}mV  R²={b[1]:.4f}'
    if a:
        s += f'  (L1) MAE={a[0]:.4f}  F1={a[1]:.3f}  CC={a[2]:.4f}'
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    HERE = os.path.dirname(os.path.abspath(__file__))
    os.chdir(HERE)

    device      = os.environ.get('DEVICE', 'cuda:2' if torch.cuda.is_available() else 'cpu')
    N_EPOCHS    = int(os.environ.get('EPOCHS', 50))
    LR          = float(os.environ.get('LR', 1e-4))
    HEAD_DIM    = int(os.environ.get('HEAD_DIM', 32))
    HEADS       = int(os.environ.get('HEADS', 4))
    LAYERS      = os.environ.get('LAYERS', '4,3,2')   # layers per AMG level
    MAX_LEVELS  = int(os.environ.get('MAX_LEVELS', 3))
    CKPT_NAME   = os.environ.get('CKPT_NAME', 'ckpt_amg_gat')
    ANCHOR_FRAC = float(os.environ.get('ANCHOR_FRAC', 0.0))
    ENCODING    = os.environ.get('ENCODING', 'v_ratio')
    WARMUP      = int(os.environ.get('WARMUP', 5))

    layers_per_level = tuple(int(x) for x in LAYERS.split(','))
    n_levels = len(layers_per_level)
    in_dim = 6 if ENCODING == 'known_drop' else 5

    print(f'Device: {device}  ANCHOR_FRAC={ANCHOR_FRAC}  ENCODING={ENCODING}', flush=True)

    t0 = time.time()
    train_raw    = load_group_flat('fake-circuit-data')
    real_raw     = load_group('real-circuit-data')
    hidden_raw   = load_group('hidden-real-circuit-data')
    print(f'Loaded {len(train_raw)} train / {len(real_raw)} real / '
          f'{len(hidden_raw)} hidden  ({time.time()-t0:.0f}s)', flush=True)

    print('Precomputing AMG hierarchies...', flush=True)
    t1 = time.time()
    train_data  = precompute(train_raw,  device, MAX_LEVELS)
    real_data   = precompute(real_raw,   device, MAX_LEVELS)
    hidden_data = precompute(hidden_raw, device, MAX_LEVELS)
    print(f'  train={len(train_data)} real={len(real_data)} '
          f'hidden={len(hidden_data)}  ({time.time()-t1:.0f}s)', flush=True)

    model = AMGGAT(in_dim=in_dim, head_dim=HEAD_DIM, heads=HEADS,
                   layers_per_level=layers_per_level).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'AMGGAT  layers={layers_per_level}  head_dim={HEAD_DIM}  '
          f'heads={HEADS}  in_dim={in_dim}  params={n_params:,}', flush=True)

    def eval_on(data, label):
        r = eval_amg(model, data, device, ANCHOR_FRAC,
                     encoding=ENCODING, n_model_levels=n_levels)
        print(format_line(r, label), flush=True)
        return r

    eval_on(real_data,   'untrained real  ')
    eval_on(hidden_data, 'untrained hidden')

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    warmup_sch = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, end_factor=1.0, total_iters=WARMUP)
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(N_EPOCHS - WARMUP, 1))
    sch = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sch, cosine_sch], milestones=[WARMUP])

    best_mae = float('inf')
    t_train = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        ep_sup, n_ok = 0.0, 0
        for idx, s in enumerate(train_data):
            if s['n_levels'] < n_levels:
                continue
            try:
                is_pin = (s['b_vdd'] > 0)
                anc = sample_anchor_indices(
                    is_pin, s['mask'].astype(bool),
                    ANCHOR_FRAC, np.random.default_rng(idx))
                x0 = build_features(
                    s['A_sp'], s['b_vdd'], s['b_isource'],
                    s['layer_idx'], device,
                    x_gt=s['x_gt'], gt_mask=s['mask'], vdd=s['vdd'],
                    anchor_indices=anc, encoding=ENCODING)
                feats = [x0] + s['cfeats'][1:n_levels]
                m = torch.from_numpy(s['mask']).to(device)
                drop_gt_n = torch.from_numpy(
                    ((s['vdd'] - s['x_gt']) / s['vdd']).astype(np.float32)
                ).to(device)
            except Exception:
                continue

            opt.zero_grad()
            drop = model(s['graphs'][:n_levels], feats, s['Ps'][:n_levels - 1])
            pred_mv = drop[m] * s['vdd'] * 1000
            gt_mv   = drop_gt_n[m] * s['vdd'] * 1000
            loss = ((pred_mv - gt_mv) ** 2).mean()
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_sup += loss.item()
            n_ok   += 1
        sch.step()

        d = max(n_ok, 1)
        lr_now = opt.param_groups[0]['lr']
        rmse = (ep_sup / d) ** 0.5
        msg = (f'epoch {epoch:3d}  lr={lr_now:.2e}  '
               f'train_RMS={rmse:.4f}mV  ({time.time()-t_train:.0f}s)')
        if epoch % 5 == 0 or epoch == 1:
            r = eval_amg(model, real_data, device, ANCHOR_FRAC,
                         encoding=ENCODING, n_model_levels=n_levels)
            mae_r = r['mean_B'][0]
            msg += f'  ||  {format_line(r, "real")}'
            if mae_r < best_mae:
                best_mae = mae_r
                torch.save(model.state_dict(), CKPT_NAME + '_best.pt')
        print(msg, flush=True)

    torch.save(model.state_dict(), CKPT_NAME + '_final.pt')
    print('=' * 64, flush=True)
    print(f'best real MAE: {best_mae:.4f} mV', flush=True)

    if os.path.exists(CKPT_NAME + '_best.pt'):
        model.load_state_dict(torch.load(CKPT_NAME + '_best.pt',
                                          map_location=device))
        eval_on(real_data,   'REAL   (best)')
        eval_on(hidden_data, 'HIDDEN (best)')
    print('Done.', flush=True)
