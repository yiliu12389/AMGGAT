import os
import time

import numpy as np
import torch
import torch.nn as nn

from data import load_group, load_group_flat
from data.amg_dataset import precompute
from data.features import build_features_from_rhs
from data.normalization import fit_stats, normalize_graph
from models.residual import ResidualVCycle


def level_rhs(sample, level, device):
    if level == 0:
        return torch.from_numpy(sample['b_np'].astype(np.float32)).to(device)
    rhs = sample['bv_ls'][level] + sample['bi_ls'][level]
    return torch.from_numpy(rhs.astype(np.float32)).to(device)


def sample_layer_bounds(sample):
    layers = sample['layer_idx']
    return float(layers.min()), float(layers.max())


def fit_all_stats(samples, n_levels, device):
    node_stats, edge_stats, target_stats = [], [], []
    for level in range(n_levels):
        edge_batches = []
        target_batches = []
        for sample in samples:
            if sample['n_levels'] <= level:
                continue
            rhs = level_rhs(sample, level, device)
            edge_batches.append(sample['graphs'][level]['edge_feat'].unsqueeze(-1))
            if level == 0:
                target = sample['x_gt'][sample['mask'].astype(bool)]
            else:
                target = sample['x_coarse'][level]
            target_batches.append(torch.from_numpy(
                target.astype(np.float32)).reshape(-1, 1))
        edge_stats.append(fit_stats(edge_batches, 1))
        target_stats.append(fit_stats(target_batches, 1))
    return edge_stats, target_stats


def fit_l0_node_stats(samples, device):
    batches = []
    for sample in samples:
        if sample['n_levels'] < 1:
            continue
        features = build_features_from_rhs(
            sample['graphs'][0], level_rhs(sample, 0, device),
            sample['li_ls'][0], layer_bounds=sample_layer_bounds(sample))
        batches.append(features[:, :4])
    return fit_stats(batches, 4)


def fit_runtime_node_stats(samples, model, n_levels, device, l0_stats):
    down_stats = [l0_stats] + [None] * (n_levels - 1)
    model.eval()
    with torch.no_grad():
        for level in range(1, n_levels):
            batches = []
            print(f'Fitting downward L{level} feature stats...', flush=True)
            for index, sample in enumerate(samples, 1):
                if sample['n_levels'] < n_levels:
                    continue
                _, trace = model(
                    sample['graphs'][:n_levels],
                    sample['li_ls'][:n_levels],
                    level_rhs(sample, 0, device),
                    sample['Ps'][:n_levels - 1],
                    node_stats=down_stats,
                    up_node_stats=[None] * n_levels,
                    layer_bounds=sample_layer_bounds(sample),
                    return_trace=True,
                )
                batches.append(
                    trace['downward_features'][level][:, :4].detach())
                if index % 10 == 0:
                    print(f'  processed {index}/{len(samples)} circuits',
                          flush=True)
            down_stats[level] = fit_stats(batches, 4)
            print(f'  L{level} mean={down_stats[level]["mean"].tolist()} '
                  f'std={down_stats[level]["std"].tolist()}', flush=True)

        up_stats = [None] * n_levels
        for level in range(n_levels - 2, -1, -1):
            batches = []
            print(f'Fitting upward L{level} feature stats...', flush=True)
            for index, sample in enumerate(samples, 1):
                if sample['n_levels'] < n_levels:
                    continue
                _, trace = model(
                    sample['graphs'][:n_levels],
                    sample['li_ls'][:n_levels],
                    level_rhs(sample, 0, device),
                    sample['Ps'][:n_levels - 1],
                    node_stats=down_stats,
                    up_node_stats=up_stats,
                    layer_bounds=sample_layer_bounds(sample),
                    return_trace=True,
                )
                batches.append(
                    trace['upward_features'][level][:, :4].detach())
                if index % 10 == 0:
                    print(f'  processed {index}/{len(samples)} circuits',
                          flush=True)
            up_stats[level] = fit_stats(batches, 4)
            print(f'  L{level} mean={up_stats[level]["mean"].tolist()} '
                  f'std={up_stats[level]["std"].tolist()}', flush=True)

    return {'down': down_stats, 'up': up_stats}


def normalize_edges(samples, edge_stats):
    for sample in samples:
        for level in range(min(len(edge_stats), sample['n_levels'])):
            normalize_graph(sample['graphs'][level], edge_stats[level])


def predict(model, sample, n_levels, node_stats):
    if isinstance(node_stats, dict):
        down_stats = node_stats['down']
        up_stats = node_stats['up']
    else:
        down_stats = node_stats
        up_stats = node_stats
    return model(
        sample['graphs'][:n_levels],
        sample['li_ls'][:n_levels],
        level_rhs(sample, 0, sample['graphs'][0]['A'].device),
        sample['Ps'][:n_levels - 1],
        node_stats=down_stats,
        up_node_stats=up_stats,
        layer_bounds=sample_layer_bounds(sample),
    )


def evaluate(model, samples, n_levels, node_stats):
    model.eval()
    per_mae, per_mse, per_r2, per_cc = [], [], [], []
    within10, total = 0, 0
    with torch.no_grad():
        for sample in samples:
            if sample['n_levels'] < n_levels:
                continue
            pred_v = predict(model, sample, n_levels, node_stats)
            mask = torch.from_numpy(sample['mask'].astype(bool)).to(pred_v.device)
            gt_v = torch.from_numpy(sample['x_gt'].astype(np.float32)).to(pred_v.device)
            error_mv = (pred_v[mask] - gt_v[mask]) * 1000.0
            gt_drop_mv = (sample['vdd'] - gt_v[mask]) * 1000.0
            per_mae.append(error_mv.abs().mean().item())
            per_mse.append(error_mv.square().mean().item())
            centered = gt_v[mask] - gt_v[mask].mean()
            per_r2.append(float(1.0 - error_mv.square().sum() /
                               (centered.square().sum() * 1e6 + 1e-30)))
            pred_centered = pred_v[mask] - pred_v[mask].mean()
            denom = pred_centered.norm() * centered.norm()
            per_cc.append(float((pred_centered * centered).sum() /
                                denom.clamp_min(1e-30)))
            within10 += int((error_mv.abs() <= 0.1 * gt_drop_mv.abs()).sum())
            total += int(mask.sum())
    if not per_mse:
        return None
    return {
        'mae': float(np.mean(per_mae)),
        'mse': float(np.mean(per_mse)),
        'r2': float(np.mean(per_r2)),
        'cc': float(np.mean(per_cc)),
        'within10': 100.0 * within10 / max(total, 1),
        'circuits': len(per_mse),
    }


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    torch.manual_seed(0)
    np.random.seed(0)

    device = os.environ.get('DEVICE', 'cuda:8' if torch.cuda.is_available() else 'cpu')
    n_levels = int(os.environ.get('N_LEVELS', 4))
    epochs = int(os.environ.get('EPOCHS', 50))
    eval_every = int(os.environ.get('EVAL_EVERY', 10))
    lr = float(os.environ.get('LR', 1e-4))
    head_dim = int(os.environ.get('HEAD_DIM', 32))
    heads = int(os.environ.get('HEADS', 4))
    gat_layers = int(os.environ.get('GAT_LAYERS', 3))
    ckpt_name = os.environ.get(
        'CKPT_NAME', 'checkpoints/residual_vcycle4_splitdecoder_zscore')
    os.makedirs(os.path.dirname(ckpt_name) or '.', exist_ok=True)

    start = time.time()
    train_raw = load_group_flat('fake-circuit-data')
    real_raw = load_group('real-circuit-data')
    hidden_raw = load_group('hidden-real-circuit-data')
    print(f'Device={device} levels={n_levels} GAT={gat_layers} '
          f'heads={heads} dim={head_dim} decoders=separate-down-up',
          flush=True)
    print(f'Loaded train={len(train_raw)} real={len(real_raw)} '
          f'hidden={len(hidden_raw)}', flush=True)

    train = precompute(train_raw, device, max_levels=n_levels)
    real = precompute(real_raw, device, max_levels=n_levels)
    hidden = precompute(hidden_raw, device, max_levels=n_levels)
    print(f'Precomputed AMG in {time.time() - start:.0f}s; '
          f'train={len(train)} real={len(real)} hidden={len(hidden)}', flush=True)

    edge_stats, target_stats = fit_all_stats(
        train, n_levels, device)
    normalize_edges(train, edge_stats)
    normalize_edges(real, edge_stats)
    normalize_edges(hidden, edge_stats)

    l0_stats = fit_l0_node_stats(train, device)
    model = ResidualVCycle(
        n_levels=n_levels, feature_dim=5, head_dim=head_dim,
        heads=heads, gat_layers=gat_layers).to(device)
    node_stats = fit_runtime_node_stats(
        train, model, n_levels, device, l0_stats)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = int(os.environ.get('WARMUP', 5))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs - warmup, 1)),
        ],
        milestones=[warmup],
    )

    best_real_mse = float('inf')
    train_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for sample in train:
            if sample['n_levels'] < n_levels:
                continue
            optimizer.zero_grad()
            prediction = predict(model, sample, n_levels, node_stats)
            mask = torch.from_numpy(sample['mask'].astype(bool)).to(device)
            target = torch.from_numpy(sample['x_gt'].astype(np.float32)).to(device)
            loss = (((prediction[mask] - target[mask]) * 1000.0) ** 2).mean()
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        train_rms = float(np.sqrt(np.mean(losses))) if losses else float('nan')
        line = (f'epoch {epoch:3d} lr={optimizer.param_groups[0]["lr"]:.2e} '
                f'train_RMS={train_rms:.5f}mV ({time.time() - train_start:.0f}s)')
        if epoch % eval_every == 0:
            real_result = evaluate(model, real, n_levels, node_stats)
            hidden_result = evaluate(model, hidden, n_levels, node_stats)
            line += (f' || REAL MAE={real_result["mae"]:.5f}mV '
                     f'R2={real_result["r2"]:+.4f} CC={real_result["cc"]:+.4f} '
                     f'within10={real_result["within10"]:.2f}%; '
                     f'HIDDEN MAE={hidden_result["mae"]:.5f}mV '
                     f'R2={hidden_result["r2"]:+.4f} CC={hidden_result["cc"]:+.4f} '
                     f'within10={hidden_result["within10"]:.2f}%')
            if real_result['mse'] < best_real_mse:
                best_real_mse = real_result['mse']
                torch.save({
                    'model': model.state_dict(),
                    'node_stats': node_stats,
                    'edge_stats': edge_stats,
                    'target_stats': target_stats,
                    'decoder_heads': 'separate_down_up',
                    'n_levels': n_levels,
                }, ckpt_name + '_best.pt')
        print(line, flush=True)

    torch.save({
        'model': model.state_dict(),
        'node_stats': node_stats,
        'edge_stats': edge_stats,
        'target_stats': target_stats,
        'decoder_heads': 'separate_down_up',

        'n_levels': n_levels,
    }, ckpt_name + '_final.pt')
    print(f'best REAL MSE={best_real_mse:.6f}mV^2', flush=True)
