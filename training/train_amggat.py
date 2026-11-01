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
from models.coarsetofine import AMGGAT
from models.total import TotalAMGGAT
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
    per_A, per_B, per_relative, per_tc = [], [], [], []
    relative_sum, relative_count, within10_count = 0.0, 0, 0
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
                pred_mv = drop[m] * s['vdd'] * 1000.0
                gt_mv = drop_gt_n[m] * s['vdd'] * 1000.0
                pred_c = pred_mv - pred_mv.mean()
                gt_c = gt_mv - gt_mv.mean()
                cc = float((pred_c * gt_c).sum() /
                           (pred_c.norm() * gt_c.norm()).clamp_min(1e-30))
                relative = ((pred_mv - gt_mv).abs() /
                            gt_mv.abs().clamp_min(1e-8))
                relative_metrics = (
                    relative.mean().item(),
                    relative.median().item(),
                    (relative <= 0.10).float().mean().item(),
                )
                per_relative.append(relative_metrics)
                relative_sum += relative.sum().item()
                relative_count += int(relative.numel())
                within10_count += int((relative <= 0.10).sum().item())
                if m1.sum() > 0:
                    mae_a, f1_a, cc_a = metrics_A(drop[m1], drop_gt_n[m1], s['vdd'])
                    per_A.append((mae_a, f1_a, cc_a))
                per_tc.append((s['tc'], mae_b, r2_b, cc, *relative_metrics))
            except Exception as e:
                print(f'  eval skip {s["tc"]}: {e}', flush=True)

    def _mean(xs):
        if not xs:
            return None
        a = np.array(xs, dtype=float)
        return tuple(a.mean(axis=0).tolist())

    return {'mean_A': _mean(per_A), 'mean_B': _mean(per_B),
            'mean_relative': _mean(per_relative),
            'node_mape': (relative_sum / relative_count
                          if relative_count else float('nan')),
            'node_within10': (within10_count / relative_count
                              if relative_count else float('nan')),
            'tag': '', 'per_tc': per_tc}


def print_amg_group_table(result, label):
    print(f'\n===== {label} ({len(result["per_tc"])} testcases) =====',
          flush=True)
    print(f'  {"testcase":12s} | {"MAE(mV)":>10s} {"R2":>9s} '
          f'{"CC":>8s} {"MAPE":>8s} {"median":>8s} {"<=10%":>8s}',
          flush=True)
    print('  ' + '-' * 80, flush=True)
    for tc, mae, r2, cc, mape, median, within10 in result['per_tc']:
        print(f'  {tc:12s} | {mae:10.4f} {r2:+9.4f} {cc:+8.4f} '
              f'{100*mape:7.2f}% {100*median:7.2f}% {100*within10:7.2f}%',
              flush=True)
    print(f'  ALL NODES    | relative MAPE={100*result["node_mape"]:.2f}% '
          f'within10={100*result["node_within10"]:.2f}%', flush=True)


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
    MODEL_NAME  = os.environ.get('MODEL', 'coarsetofine').lower()
    default_ckpt = ('checkpoints/total4_global_zscore'
                    if MODEL_NAME == 'total'
                    else 'checkpoints/coarsetofine4_global_zscore')
    CKPT_NAME   = os.environ.get('CKPT_NAME', default_ckpt)
    WARMUP      = int(os.environ.get('WARMUP', 5))
    EVAL_EVERY  = int(os.environ.get('EVAL_EVERY', 10))

    layers_per_level = tuple(int(x) for x in LAYERS.split(','))
    n_levels = len(layers_per_level)
    in_dim = 5
    if MODEL_NAME not in ('coarsetofine', 'total'):
        raise ValueError('MODEL must be coarsetofine or total')
    if MODEL_NAME == 'total' and len(set(layers_per_level)) != 1:
        raise ValueError('total requires the same GAT depth at every level')

    print(f'Device: {device}', flush=True)

    t0 = time.time()
    train_raw    = load_group_flat('fake-circuit-data')
    real_raw     = load_group('real-circuit-data')
    hidden_raw   = load_group('hidden-real-circuit-data')
    print(f'Loaded {len(train_raw)} train / {len(real_raw)} real / '
          f'{len(hidden_raw)} hidden  ({time.time()-t0:.0f}s)', flush=True)

    print('Precomputing AMG hierarchies...', flush=True)
    t1 = time.time()
    solve_targets = os.environ.get('SOLVE_COARSE_TARGETS', '0') == '1'
    train_data  = precompute(train_raw,  device, MAX_LEVELS, solve_targets)
    real_data   = precompute(real_raw,   device, MAX_LEVELS, solve_targets)
    hidden_data = precompute(hidden_raw, device, MAX_LEVELS, solve_targets)
    print(f'  train={len(train_data)} real={len(real_data)} '
          f'hidden={len(hidden_data)}  ({time.time()-t1:.0f}s)', flush=True)

    print('Fitting per-level node z-score statistics; keeping log1p edges...',
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
        if not node_batches:
            raise ValueError(f'No training examples available for AMG level {level}')
        level_stats.append({'node': fit_stats(node_batches, 4)})
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

    if MODEL_NAME == 'total':
        model = TotalAMGGAT(
            n_levels=n_levels, feature_dim=in_dim, head_dim=HEAD_DIM,
            heads=HEADS, gat_layers=layers_per_level[0]).to(device)
    else:
        model = AMGGAT(in_dim=in_dim, head_dim=HEAD_DIM, heads=HEADS,
                       layers_per_level=layers_per_level).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{MODEL_NAME}  layers={layers_per_level}  head_dim={HEAD_DIM}  '
          f'heads={HEADS}  in_dim={in_dim}  params={n_params:,}', flush=True)

    def eval_on(data, label):
        r = eval_amg(model, data, device, n_model_levels=n_levels,
                     node_stats=level_stats, target_stats=target_stats)
        if label is not None:
            print(format_line(r, label), flush=True)
        return r

    if os.environ.get('EVAL_ONLY', '0') == '1':
        checkpoint_path = os.environ.get('EVAL_CKPT', CKPT_NAME + '_best.pt')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        train_result = eval_on(train_data, 'TRAIN (best checkpoint)')
        real_result = eval_on(real_data, 'REAL TEST (best checkpoint)')
        hidden_result = eval_on(hidden_data, 'HIDDEN TEST (best checkpoint)')
        print_amg_group_table(train_result, 'TRAIN per-circuit')
        print_amg_group_table(real_result, 'REAL per-circuit')
        print_amg_group_table(hidden_result, 'HIDDEN per-circuit')
        raise SystemExit(0)

    def compact_eval(r):
        if r['mean_B'] is None:
            return 'MAE=nanmV R2=nan CC=nan MAPE=nan% <=10=nan%'
        mae, r2 = r['mean_B']
        if r['mean_A'] is None:
            return f'MAE={mae:.5f}mV R2={r2:+.4f} CC=nan MAPE=nan% <=10=nan%'
        _, _, cc = r['mean_A']
        rel = r['mean_relative']
        return (f'MAE={mae:.5f}mV R2={r2:+.4f} CC={cc:+.4f} '
                f'MAPE={100.0 * rel[0]:.2f}% '
                f'median={100.0 * rel[1]:.2f}% '
                f'<=10={100.0 * rel[2]:.2f}%')

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    warmup_sch = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, end_factor=1.0, total_iters=WARMUP)
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(N_EPOCHS - WARMUP, 1))
    sch = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sch, cosine_sch], milestones=[WARMUP])

    best_real_mae = float('inf')
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
        if epoch % EVAL_EVERY == 0:
            real_result = eval_on(real_data, None)
            hidden_result = eval_on(hidden_data, None)
            real_mae = (real_result['mean_B'][0]
                        if real_result['mean_B'] is not None else float('inf'))
            msg += (f' || REAL {compact_eval(real_result)}; '
                    f'HIDDEN {compact_eval(hidden_result)}')
            if real_mae < best_real_mae:
                best_real_mae = real_mae
                torch.save({'model': model.state_dict(),
                            'level_stats': level_stats,
                            'target_stats': target_stats,
                            'best_real_mae': best_real_mae},
                           CKPT_NAME + '_best.pt')
        print(msg, flush=True)

    torch.save({'model': model.state_dict(),
                'level_stats': level_stats,
                'target_stats': target_stats}, CKPT_NAME + '_final.pt')
    print('=' * 64, flush=True)
    print(f'best REAL MAE: {best_real_mae:.6g}mV', flush=True)

    if os.path.exists(CKPT_NAME + '_best.pt'):
        checkpoint = torch.load(CKPT_NAME + '_best.pt', map_location=device)
        model.load_state_dict(checkpoint['model'])
        eval_on(real_data,   'REAL TEST   (best train ckpt)')
        eval_on(hidden_data, 'HIDDEN TEST (best train ckpt)')
    print('Done.', flush=True)
