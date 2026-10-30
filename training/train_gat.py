
import os, time
import numpy as np
import torch
import torch.nn as nn

from data import (build_graph, build_features, load_group,
                  load_group_flat)
from evaluation.metrics import (
    eval_samples, format_epoch_line, metrics_A, metrics_B, print_group_table,
)
from models.gat import GAT
from models.layers import GATLayer
from data.normalization import (
    apply_node_zscore, fit_stats, normalize_graph, restore_zscore,
)

if __name__ == '__main__':
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(HERE)
    device = os.environ.get('DEVICE', 'cuda:8' if torch.cuda.is_available() else 'cpu')
    N_EPOCHS = int(os.environ.get('EPOCHS', 50))
    LR = float(os.environ.get('LR', 1e-4))
    HEAD_DIM = int(os.environ.get('HEAD_DIM', 32))
    HEADS = int(os.environ.get('HEADS', 4))
    N_LAYERS = int(os.environ.get('N_LAYERS', 6))
    CKPT_NAME = os.environ.get(
        'CKPT_NAME', 'checkpoints/vanilla_gat6_equal_circuit_zscore')
    EVAL_EVERY = int(os.environ.get('EVAL_EVERY', 10))
    LOSS = os.environ.get('LOSS', 'mse').lower()
    LOG_EPS = float(os.environ.get('LOG_EPS', 0.01))

    feat_fn = build_features
    in_dim = 5
    os.makedirs(os.path.dirname(CKPT_NAME) or '.', exist_ok=True)

    print(f'Device: {device}', flush=True)
    t0 = time.time()
    train_samples = load_group_flat('fake-circuit-data')
    real_samples = load_group('real-circuit-data')
    hidden_samples = load_group('hidden-real-circuit-data')
    print(f'Train={len(train_samples)}  '
          f'Test={len(real_samples)} real + {len(hidden_samples)} hidden  '
          f'({time.time()-t0:.0f}s)', flush=True)

    print('Fitting node/edge z-score statistics on training circuits...', flush=True)
    node_batches, edge_batches = [], []
    for _, matrix, _, b_vdd, b_isource, layers, _, _, _ in train_samples:
        node_batches.append(
            feat_fn(matrix, b_vdd, b_isource, layers, device)[:, :4])
        edge_batches.append(build_graph(matrix, device)['edge_feat'].unsqueeze(-1))
    node_stats = fit_stats(node_batches, 4)
    edge_stats = fit_stats(edge_batches, 1)
    target_batches = []
    for *_, x_gt, mask, vdd in train_samples:
        target_mv = torch.from_numpy(((vdd - x_gt[mask.astype(bool)]) * 1000.0)
                                     .astype(np.float32)).reshape(-1, 1)
        target_batches.append(target_mv)
    target_stats = fit_stats(target_batches, 1)
    print('  per-circuit node/edge/target mean/std averaged equally over train circuits',
          flush=True)

    def eval_on(model, samples):
        return eval_samples(model, samples, device, feature_fn=feat_fn,
                            node_stats=node_stats,
                            edge_stats=edge_stats, target_stats=target_stats)

    print('=' * 64, flush=True)
    model = GAT(in_dim=in_dim, head_dim=HEAD_DIM, heads=HEADS,
                n_layers=N_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'GAT: head_dim={HEAD_DIM} heads={HEADS} layers={N_LAYERS} '
          f'in_dim={in_dim}  params={n_params:,}', flush=True)

    WARMUP = int(os.environ.get('WARMUP', 5))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    warmup_sch = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, end_factor=1.0, total_iters=WARMUP)
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(N_EPOCHS - WARMUP, 1))
    sch = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sch, cosine_sch], milestones=[WARMUP])
    best_real_loss = float('inf')
    best_real_epoch = 0
    best_train_rms = float('nan')
    t_train = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        ep_sup, ep_tot, n_ok = 0.0, 0.0, 0
        for idx, (name, A_sp, b_np, b_vdd, b_isource, layer_idx, x_gt, mask, vdd) in enumerate(train_samples):
            try:
                g = build_graph(A_sp, device)
                normalize_graph(g, edge_stats)
                x_in = feat_fn(A_sp, b_vdd, b_isource, layer_idx, device)
                x_in = apply_node_zscore(x_in, node_stats)
                m = torch.from_numpy(mask).to(device)
                gt_mv_all = torch.from_numpy(
                    ((vdd - x_gt) * 1000.0).astype(np.float32)).to(device)
            except Exception:
                continue
            opt.zero_grad()
            pred_z = model(g, x_in)
            pred_mv_all = restore_zscore(pred_z, target_stats)
            pred_mv = pred_mv_all[m]
            gt_mv   = gt_mv_all[m]
            if LOSS == 'log_mse':
                lp = torch.log(pred_mv.clamp(min=LOG_EPS))
                lg = torch.log(gt_mv.clamp(min=LOG_EPS))
                loss_sup = ((lp - lg) ** 2).mean()
            else:
                loss_sup = ((pred_mv - gt_mv) ** 2).mean()

            loss = loss_sup
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_sup += loss_sup.item()
            ep_tot += loss.item()
            n_ok   += 1
        sch.step()

        d = max(n_ok, 1)
        lr_now = opt.param_groups[0]['lr']
        train_rmse = (ep_sup / d) ** 0.5
        train_loss = ep_tot / d
        unit = 'log' if LOSS == 'log_mse' else 'mV'
        msg = (f'epoch {epoch:3d}  lr={lr_now:.2e}  '
               f'train_RMS={train_rmse:.4f} {unit}  '
               f'loss={LOSS}  '
               f'Loss={ep_tot/d:.4f}')
        msg += f'  ({time.time()-t_train:.0f}s)'
        if epoch % EVAL_EVERY == 0:
            real_result = eval_on(model, real_samples)
            hidden_result = eval_on(model, hidden_samples)
            real_loss = real_result['mean_mse']
            msg += (f'  ||  REAL test MSE={real_loss:.6g} mV²; '
                    f'{format_epoch_line(real_result, "real")}\n'
                    f'      HIDDEN test {format_epoch_line(hidden_result, "hidden")}')
            if real_loss < best_real_loss:
                best_real_loss = real_loss
                best_real_epoch = epoch
                best_train_rms = train_rmse
                torch.save({'model': model.state_dict(), 'node_stats': node_stats,
                    'edge_stats': edge_stats, 'target_stats': target_stats},
                           CKPT_NAME + '_best.pt')
        print(msg, flush=True)

    torch.save({'model': model.state_dict(), 'node_stats': node_stats,
                'edge_stats': edge_stats, 'target_stats': target_stats},
               CKPT_NAME + '_final.pt')
    print('=' * 64, flush=True)
    print(f'best real test MSE: {best_real_loss:.6g} mV²', flush=True)

    if os.path.exists(CKPT_NAME + '_best.pt'):
        checkpoint = torch.load(CKPT_NAME + '_best.pt', map_location=device)
        model.load_state_dict(checkpoint['model'])
        real_best = eval_on(model, real_samples)
        hidden_best = eval_on(model, hidden_samples)
        print_group_table(real_best, label='REAL TEST (lowest real-loss ckpt)')
        print_group_table(hidden_best, label='HIDDEN TEST (lowest real-loss ckpt)')
        print('\nVanilla GAT 6-layer summary (best real-loss checkpoint)', flush=True)
        print('| Train RMS (mV) | REAL MAE (mV) | REAL nodes ≤10% | REAL CC | '
              'HIDDEN MAE (mV) | HIDDEN R² | HIDDEN CC |', flush=True)
        print(f'| {best_train_rms:.4f} | {real_best["mean_B"][0]:.4f} | '
              f'{real_best["within10_pct"]:.2f}% | {real_best["mean_CC"]:+.4f} | '
              f'{hidden_best["mean_B"][0]:.4f} | '
              f'{hidden_best["mean_B"][1]:+.4f} | '
              f'{hidden_best["mean_CC"]:+.4f} |', flush=True)
    print('Done.', flush=True)
