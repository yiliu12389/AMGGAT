import os, time
import numpy as np
import torch
import torch.nn as nn

from data import (build_features, load_group, load_group_flat,
                  )
from data.amg_dataset import precompute
from evaluation.metrics import (
    format_epoch_line, metrics_A, metrics_B, print_group_table,
)
from models.amg_gat import AMGGAT
from amg.hierarchy import build_amg_hierarchy, prep_hierarchy
from data.normalization import (
    apply_node_zscore, fit_stats, normalize_precomputed_levels,
    restore_zscore,
)


# ─────────────────────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────────────────────

def eval_amg(model, precomp, device, n_model_levels=4, node_stats=None,
             target_stats=None):
    model.eval()
    per_A, per_B = [], []
    with torch.no_grad():
        for s in precomp:
            if s['n_levels'] < n_model_levels:
                continue
            try:
                x0 = build_features(
                    s['A_sp'], s['b_vdd'], s['b_isource'],
                    s['layer_idx'], device)
                if node_stats is not None:
                    x0 = apply_node_zscore(x0, node_stats[0]['node'])
                feats = [x0] + s['cfeats'][1:n_model_levels]
                drop = model(s['graphs'][:n_model_levels],
                             feats, s['Ps'][:n_model_levels - 1])
                if target_stats is not None:
                    drop = restore_zscore(drop, target_stats) / (s['vdd'] * 1000.0)
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
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(HERE)

    device      = os.environ.get('DEVICE', 'cuda:2' if torch.cuda.is_available() else 'cpu')
    N_EPOCHS    = int(os.environ.get('EPOCHS', 50))
    LR          = float(os.environ.get('LR', 1e-4))
    HEAD_DIM    = int(os.environ.get('HEAD_DIM', 32))
    HEADS       = int(os.environ.get('HEADS', 4))
    LAYERS      = os.environ.get('LAYERS', '3,3,3,3')
    MAX_LEVELS  = int(os.environ.get('MAX_LEVELS', 4))
    CKPT_NAME   = os.environ.get('CKPT_NAME', 'ckpt_amg_gat')
    WARMUP      = int(os.environ.get('WARMUP', 5))

    layers_per_level = tuple(int(x) for x in LAYERS.split(','))
    n_levels = len(layers_per_level)
    in_dim = 5

    print(f'Device: {device}', flush=True)

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

    print('Fitting per-level node/edge z-score statistics on training circuits...',
          flush=True)
    level_stats = []
    for level in range(n_levels):
        if level == 0:
            node_batches = [build_features(
                s['A_sp'], s['b_vdd'], s['b_isource'], s['layer_idx'], device)[:, :4]
                for s in train_data]
        else:
            node_batches = [s['cfeats'][level][:, :4] for s in train_data
                            if s['n_levels'] > level and s['cfeats'][level] is not None]
        edge_batches = [s['graphs'][level]['edge_feat'].unsqueeze(-1)
                        for s in train_data if s['n_levels'] > level]
        if not node_batches or not edge_batches:
            raise ValueError(f'No training examples available for AMG level {level}')
        level_stats.append({'node': fit_stats(node_batches, 4),
                            'edge': fit_stats(edge_batches, 1)})
    target_batches = []
    for sample in train_data:
        valid = sample['mask'].astype(bool)
        target_mv = torch.from_numpy(
            ((sample['vdd'] - sample['x_gt'][valid]) * 1000.0)
            .astype(np.float32)).reshape(-1, 1)
        target_batches.append(target_mv)
    target_stats = fit_stats(target_batches, 1)
    normalize_precomputed_levels(train_data, level_stats)
    normalize_precomputed_levels(real_data, level_stats)
    normalize_precomputed_levels(hidden_data, level_stats)
    print('  fitted per-level stats by equal-weight averaging across train circuits; '
          'test uses train stats', flush=True)

    model = AMGGAT(in_dim=in_dim, head_dim=HEAD_DIM, heads=HEADS,
                   layers_per_level=layers_per_level).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'AMGGAT  layers={layers_per_level}  head_dim={HEAD_DIM}  '
          f'heads={HEADS}  in_dim={in_dim}  params={n_params:,}', flush=True)

    def eval_on(data, label):
        r = eval_amg(model, data, device, n_model_levels=n_levels,
                     node_stats=level_stats, target_stats=target_stats)
        print(format_line(r, label), flush=True)
        return r

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    warmup_sch = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, end_factor=1.0, total_iters=WARMUP)
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(N_EPOCHS - WARMUP, 1))
    sch = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sch, cosine_sch], milestones=[WARMUP])

    best_train_loss = float('inf')
    t_train = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        ep_sup, n_ok = 0.0, 0
        for idx, s in enumerate(train_data):
            if s['n_levels'] < n_levels:
                continue
            try:
                x0 = build_features(
                    s['A_sp'], s['b_vdd'], s['b_isource'],
                    s['layer_idx'], device)
                x0 = apply_node_zscore(x0, level_stats[0]['node'])
                feats = [x0] + s['cfeats'][1:n_levels]
                m = torch.from_numpy(s['mask']).to(device)
                drop_gt_n = torch.from_numpy(
                    ((s['vdd'] - s['x_gt']) / s['vdd']).astype(np.float32)
                ).to(device)
            except Exception:
                continue

            opt.zero_grad()
            drop = model(s['graphs'][:n_levels], feats, s['Ps'][:n_levels - 1])
            pred_mv = restore_zscore(drop, target_stats)[m]
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
        train_loss = ep_sup / d
        msg = (f'epoch {epoch:3d}  lr={lr_now:.2e}  '
               f'train_RMS={rmse:.4f}mV  ({time.time()-t_train:.0f}s)')
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save({'model': model.state_dict(),
                        'level_stats': level_stats,
                        'target_stats': target_stats}, CKPT_NAME + '_best.pt')
        print(msg, flush=True)

    torch.save({'model': model.state_dict(),
                'level_stats': level_stats,
                'target_stats': target_stats}, CKPT_NAME + '_final.pt')
    print('=' * 64, flush=True)
    print(f'best train loss: {best_train_loss:.6g}', flush=True)

    if os.path.exists(CKPT_NAME + '_best.pt'):
        checkpoint = torch.load(CKPT_NAME + '_best.pt', map_location=device)
        model.load_state_dict(checkpoint['model'])
        eval_on(real_data,   'REAL TEST   (best train ckpt)')
        eval_on(hidden_data, 'HIDDEN TEST (best train ckpt)')
    print('Done.', flush=True)
