import os, time
import numpy as np
import torch

from data import build_graph, build_features, load_sample, load_group, load_group_flat
from data.normalization import (
    apply_node_zscore, normalize_graph, restore_zscore,
)


# ============================================================================
# Per-testcase metrics
# ============================================================================
def metrics_A(drop_p_norm, drop_g_norm, vdd):
    """A-tier: MAE, F1, CC on IR drop (mV). Inputs = normalized drop."""
    drop_p = drop_p_norm * vdd * 1000.0
    drop_g = drop_g_norm * vdd * 1000.0
    mae = (drop_p - drop_g).abs().mean().item()
    maxg = drop_g.max().item()
    th = 0.9 * maxg
    gt_pos, pd_pos = drop_g > th, drop_p > th
    tp = int((gt_pos & pd_pos).sum())
    fp = int((~gt_pos & pd_pos).sum())
    fn = int((gt_pos & ~pd_pos).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    vp_c = drop_p - drop_p.mean()
    vg_c = drop_g - drop_g.mean()
    denom = (vp_c.norm() * vg_c.norm()).clamp_min(1e-30)
    cc = float((vp_c * vg_c).sum() / denom)
    return mae, f1, cc


def metrics_B(drop_p_norm, drop_g_norm, vdd):
    """B-tier: MAE (mV), R2. All-node vs golden .solution. Inputs = normalized drop."""
    drop_p = drop_p_norm * vdd * 1000.0
    drop_g = drop_g_norm * vdd * 1000.0
    mae = (drop_p - drop_g).abs().mean().item()
    ss_res = ((drop_p - drop_g) ** 2).sum()
    ss_tot = ((drop_g - drop_g.mean()) ** 2).sum()
    r2 = float(1.0 - ss_res / (ss_tot + 1e-30))
    return mae, r2


# ============================================================================
# Batch evaluation
# ============================================================================
def eval_samples(model, samples, device, tag='', feature_fn=None,
                 node_stats=None, edge_stats=None, target_stats=None):
    if feature_fn is None:
        feature_fn = build_features
    model.eval()
    per_tc = []
    per_A, per_B, per_MSE, per_CC, per_T = [], [], [], [], []
    within10_nodes = 0
    valid_nodes = 0
    for tc, A_sp, b_np, b_vdd, b_iso, li, x_gt, mask, vdd in samples:
        try:
            g   = build_graph(A_sp, device)
            if edge_stats is not None:
                normalize_graph(g, edge_stats)
            xin = feature_fn(A_sp, b_vdd, b_iso, li, device)
            if node_stats is not None:
                xin = apply_node_zscore(xin, node_stats)
            gt_mv_all = torch.from_numpy(
                ((vdd - x_gt) * 1000.0).astype(np.float32)).to(device)

            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                pred_raw = model(g, xin)
                if target_stats is not None:
                    pred_mv_all = restore_zscore(pred_raw, target_stats)
                else:
                    pred_mv_all = pred_raw * (vdd * 1000.0)
            if str(device).startswith('cuda'):
                torch.cuda.synchronize()
            runtime = time.time() - t0
            per_T.append(runtime)

            mm_b = torch.from_numpy(mask).to(device)
            pred_mv = pred_mv_all[mm_b]
            gt_mv = gt_mv_all[mm_b]
            error_mv = pred_mv - gt_mv
            per_MSE.append(float(error_mv.square().mean().item()))
            within10_nodes += int((error_mv.abs() <= 0.1 * gt_mv.abs()).sum().item())
            valid_nodes += int(gt_mv.numel())
            pred_centered = pred_mv - pred_mv.mean()
            gt_centered = gt_mv - gt_mv.mean()
            cc_denom = pred_centered.norm() * gt_centered.norm()
            per_CC.append(float((pred_centered * gt_centered).sum() /
                                cc_denom.clamp_min(1e-30)))
            mae_b = error_mv.abs().mean().item()
            ss_res = error_mv.square().sum()
            ss_tot = (gt_mv - gt_mv.mean()).square().sum()
            r2_b = float(1.0 - ss_res / (ss_tot + 1e-30))
            per_B.append((mae_b, r2_b))

            m1 = (li == 1) & mask
            if m1.sum() == 0:
                mae_a, f1_a, cc_a = float('nan'), float('nan'), float('nan')
            else:
                mm_a = torch.from_numpy(m1).to(device)
                pred_a, gt_a = pred_mv_all[mm_a], gt_mv_all[mm_a]
                mae_a = (pred_a - gt_a).abs().mean().item()
                threshold = 0.9 * gt_a.max()
                gt_pos, pred_pos = gt_a > threshold, pred_a > threshold
                tp = int((gt_pos & pred_pos).sum())
                fp = int((~gt_pos & pred_pos).sum())
                fn = int((gt_pos & ~pred_pos).sum())
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1_a = (2 * precision * recall / (precision + recall)
                        if precision + recall else 0.0)
                pred_a_c = pred_a - pred_a.mean()
                gt_a_c = gt_a - gt_a.mean()
                cc_denom = pred_a_c.norm() * gt_a_c.norm()
                cc_a = float((pred_a_c * gt_a_c).sum() /
                             cc_denom.clamp_min(1e-30))
            per_A.append((mae_a, f1_a, cc_a))
            per_tc.append((tc, mae_a, f1_a, cc_a, mae_b, r2_b, runtime))
        except Exception as ex:
            print(f'  eval skip {tc}: {type(ex).__name__}: {ex}', flush=True)

    def _mean(xs, mask_nan=False):
        if not xs:
            return None
        a = np.array(xs, dtype=float)
        if mask_nan:
            a = a[~np.isnan(a[:, 0])]
        if a.size == 0:
            return tuple(float('nan') for _ in xs[0])
        return tuple(a.mean(axis=0).tolist())

    return {
        'tag'      : tag,
        'per_tc'   : per_tc,
        'mean_A'   : _mean(per_A, mask_nan=True),
        'mean_B'   : _mean(per_B),
        'mean_mse' : float(np.mean(per_MSE)) if per_MSE else float('nan'),
        'within10_pct': (100.0 * within10_nodes / valid_nodes
                         if valid_nodes else float('nan')),
        'mean_CC': float(np.nanmean(per_CC)) if per_CC else float('nan'),
        'mean_time': float(np.mean(per_T)) if per_T else float('nan'),
    }


# ============================================================================
# Reporting helpers
# ============================================================================
def print_group_table(result, label):
    print(f'\n===== {label} ({len(result["per_tc"])} testcases) =====',
          flush=True)
    print(f'  {"testcase":12s} | {"A: MAE(M1)":>10s} {"F1":>5s} {"CC":>7s} | '
          f'{"B: MAE(all)":>11s} {"R2":>8s} {"time(s)":>8s}', flush=True)
    print('  ' + '-' * 82, flush=True)
    for tc, mae_a, f1_a, cc_a, mae_b, r2_b, rt in result['per_tc']:
        print(f'  {tc:12s} | {mae_a:10.4f} {f1_a:5.3f} {cc_a:+7.3f} | '
              f'{mae_b:11.4f} {r2_b:+8.4f} {rt:8.3f}', flush=True)
    if result['mean_A'] and result['mean_B']:
        mA, mB = result['mean_A'], result['mean_B']
        print('  ' + '-' * 82, flush=True)
        print(f'  {"MEAN":12s} | {mA[0]:10.4f} {mA[1]:5.3f} {mA[2]:+7.3f} | '
              f'{mB[0]:11.4f} {mB[1]:+8.4f} {result["mean_time"]:8.3f}',
              flush=True)


def format_epoch_line(result, prefix='real'):
    """Compact one-line summary for per-epoch training logs."""
    mB = result.get('mean_B') or (float('nan'),)*2
    return (f'{prefix}[MAE={mB[0]:.4f}mV '
            f'within10={result.get("within10_pct", float("nan")):.2f}% '
            f'R2={mB[1]:+.3f} CC={result.get("mean_CC", float("nan")):+.3f}]')


