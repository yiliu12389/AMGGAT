"""Train a three-layer GAT decoder on one standalone AMG level."""

import os
import time

import numpy as np
import torch
import torch.nn as nn

from data import load_group, load_group_flat
from data.amg_dataset import precompute
from data.normalization import (
    apply_node_zscore, fit_stats, normalize_graph, restore_zscore,
)
from models.coarsetofine import A3GAT


LEVEL = int(os.environ.get('AMG_LEVEL', '3'))


def fit_level_stats(samples, level):
    """Fit train-only node, edge, and exact-level voltage z-score statistics."""
    node_batches = [s['cfeats'][level][:, :4] for s in samples
                    if s['n_levels'] > level]
    edge_batches = [s['graphs'][level]['edge_feat'].unsqueeze(-1)
                    for s in samples if s['n_levels'] > level]
    target_batches = [torch.from_numpy(s['x_coarse'][level].astype(np.float32))
                      .reshape(-1, 1) for s in samples if s['n_levels'] > level]
    if not node_batches or not edge_batches or not target_batches:
        raise ValueError(f'No samples reached AMG level A{LEVEL}')
    return {
        'node': fit_stats(node_batches, 4),
        'edge': fit_stats(edge_batches, 1),
        'target': fit_stats(target_batches, 1),
    }


def normalize_level(samples, stats, level):
    for sample in samples:
        if sample['n_levels'] <= level:
            continue
        sample['cfeats'][level] = apply_node_zscore(
            sample['cfeats'][level], stats['node'])
        normalize_graph(sample['graphs'][level], stats['edge'])


def evaluate(model, samples, device, target_stats):
    model.eval()
    per_mae, per_mse, per_r2, per_cc = [], [], [], []
    with torch.no_grad():
        for sample in samples:
            if sample['n_levels'] <= LEVEL:
                continue
            target_v = torch.from_numpy(
                sample['x_coarse'][LEVEL].astype(np.float32)).to(device)
            pred_z = model(sample['graphs'][LEVEL], sample['cfeats'][LEVEL])
            pred_v = restore_zscore(pred_z, target_stats)
            error_mv = (pred_v - target_v) * 1000.0
            per_mae.append(error_mv.abs().mean().item())
            per_mse.append(error_mv.square().mean().item())
            target_centered = target_v - target_v.mean()
            per_r2.append(float(1.0 - error_mv.square().sum() /
                                (target_centered.square().sum() * 1e6 + 1e-30)))
            pred_centered = pred_v - pred_v.mean()
            denom = pred_centered.norm() * target_centered.norm()
            per_cc.append(float((pred_centered * target_centered).sum() /
                                denom.clamp_min(1e-30)))
    return {
        'mae': float(np.mean(per_mae)),
        'mse': float(np.mean(per_mse)),
        'rmse': float(np.sqrt(np.mean(per_mse))),
        'r2': float(np.mean(per_r2)),
        'cc': float(np.mean(per_cc)),
        'circuits': len(per_mse),
    }


def evaluate_train_rms(model, samples, device, target_stats):
    """Evaluate one fixed end-of-epoch model on all training circuits."""
    model.eval()
    per_circuit_mse = []
    with torch.no_grad():
        for sample in samples:
            if sample['n_levels'] <= LEVEL:
                continue
            target_v = torch.from_numpy(
                sample['x_coarse'][LEVEL].astype(np.float32)).to(device)
            pred_z = model(sample['graphs'][LEVEL], sample['cfeats'][LEVEL])
            pred_v = restore_zscore(pred_z, target_stats)
            error_mv = (pred_v - target_v) * 1000.0
            per_circuit_mse.append(error_mv.square().mean().item())
    return float(np.sqrt(np.mean(per_circuit_mse)))


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    torch.manual_seed(0)
    np.random.seed(0)
    device = os.environ.get('DEVICE', 'cuda:8' if torch.cuda.is_available() else 'cpu')
    epochs = int(os.environ.get('EPOCHS', 50))
    lr = float(os.environ.get('LR', 1e-4))
    head_dim = int(os.environ.get('HEAD_DIM', 32))
    heads = int(os.environ.get('HEADS', 4))
    eval_every = int(os.environ.get('EVAL_EVERY', 10))
    ckpt_name = os.environ.get(
        'CKPT_NAME', f'checkpoints/amg_level{LEVEL}_gat3_zscore')
    os.makedirs(os.path.dirname(ckpt_name) or '.', exist_ok=True)

    t0 = time.time()
    train_raw = load_group_flat('fake-circuit-data')
    real_raw = load_group('real-circuit-data')
    hidden_raw = load_group('hidden-real-circuit-data')
    print(f'Device: {device}; A{LEVEL}-only: GAT3(F{LEVEL}) -> '
          f'decoder{LEVEL} -> x{LEVEL}', flush=True)
    print(f'Loaded train={len(train_raw)} real={len(real_raw)} hidden={len(hidden_raw)}', flush=True)
    n_levels = LEVEL + 1
    train = precompute(train_raw, device, max_levels=n_levels)
    real = precompute(real_raw, device, max_levels=n_levels)
    hidden = precompute(hidden_raw, device, max_levels=n_levels)
    print(f'Precomputed A0..A{LEVEL} in {time.time() - t0:.0f}s; '
          f'train={len(train)} real={len(real)} hidden={len(hidden)}', flush=True)

    stats = fit_level_stats(train, LEVEL)
    normalize_level(train, stats, LEVEL)
    normalize_level(real, stats, LEVEL)
    normalize_level(hidden, stats, LEVEL)
    print(f'A{LEVEL} node/edge/x{LEVEL} statistics: train circuits only, '
          'equal circuit weight', flush=True)
    print(f'A{LEVEL} target mean={stats["target"]["mean"].item()*1000:.6f}mV '
          f'std={stats["target"]["std"].item()*1000:.6f}mV '
          '(direct Galerkin RHS restriction)', flush=True)

    model = A3GAT(in_dim=5, head_dim=head_dim, heads=heads, n_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = int(os.environ.get('WARMUP', 5))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01,
                                               end_factor=1.0, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs - warmup, 1)),
        ], milestones=[warmup])
    best_real_mse = float('inf')
    best_epoch = 0
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for sample in train:
            if sample['n_levels'] <= LEVEL:
                continue
            target_v = torch.from_numpy(
                sample['x_coarse'][LEVEL].astype(np.float32)).to(device)
            optimizer.zero_grad()
            pred_z = model(sample['graphs'][LEVEL], sample['cfeats'][LEVEL])
            pred_v = restore_zscore(pred_z, stats['target'])
            loss = (((pred_v - target_v) * 1000.0) ** 2).mean()
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        train_rms = evaluate_train_rms(model, train, device, stats['target'])
        line = (f'epoch {epoch:3d} lr={optimizer.param_groups[0]["lr"]:.2e} '
                f'train_RMS={train_rms:.5f}mV ({time.time() - train_start:.0f}s)')
        if epoch % eval_every == 0:
            real_result = evaluate(model, real, device, stats['target'])
            hidden_result = evaluate(model, hidden, device, stats['target'])
            line += (f' || REAL A{LEVEL} MAE={real_result["mae"]:.5f}mV '
                     f'RMS={real_result["rmse"]:.5f}mV R2={real_result["r2"]:+.4f} '
                     f'CC={real_result["cc"]:+.4f}; '
                     f'HIDDEN A{LEVEL} MAE={hidden_result["mae"]:.5f}mV '
                     f'RMS={hidden_result["rmse"]:.5f}mV '
                     f'R2={hidden_result["r2"]:+.4f} CC={hidden_result["cc"]:+.4f}')
            if real_result['mse'] < best_real_mse:
                best_real_mse = real_result['mse']
                best_epoch = epoch
                torch.save({'model': model.state_dict(), 'stats': stats,
                            'amg_level': LEVEL,
                            'target_kind': 'direct_galerkin_voltage'},
                           ckpt_name + '_best.pt')
        print(line, flush=True)

    torch.save({'model': model.state_dict(), 'stats': stats,
                'amg_level': LEVEL,
                'target_kind': 'direct_galerkin_voltage'},
               ckpt_name + '_final.pt')
    print(f'best REAL A{LEVEL} MSE={best_real_mse:.6f} mV^2 at epoch {best_epoch}', flush=True)
    best = torch.load(ckpt_name + '_best.pt', map_location=device)
    model.load_state_dict(best['model'])
    real_result = evaluate(model, real, device, stats['target'])
    hidden_result = evaluate(model, hidden, device, stats['target'])
    for label, result in (('REAL', real_result), ('HIDDEN', hidden_result)):
        print(f'{label} A{LEVEL} BEST: MAE={result["mae"]:.5f}mV '
              f'RMS={result["rmse"]:.5f}mV R2={result["r2"]:+.4f} '
              f'CC={result["cc"]:+.4f} circuits={result["circuits"]}', flush=True)
