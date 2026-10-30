"""Plot per-node AMG-GAT predictions for one REAL circuit."""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from amg.hierarchy import build_amg_hierarchy, prep_hierarchy
from data import BENCH, build_features, load_gt, parse_layer, parse_spice
from models.amg_gat import AMGGAT
from data.normalization import (
    apply_node_zscore, normalize_graph, restore_zscore,
)

_NAME_RE = re.compile(r'n(\d+)_m(\d+)_(\d+)_(\d+)')


def parse_xy(names):
    x = np.full(len(names), np.nan)
    y = np.full(len(names), np.nan)
    for index, name in enumerate(names):
        match = _NAME_RE.match(name or '')
        if match:
            x[index], y[index] = float(match.group(3)), float(match.group(4))
    return x, y


def scatter(ax, x, y, values, title, cmap='hot_r', vmin=None, vmax=None):
    artist = ax.scatter(x, y, c=values, s=4, cmap=cmap, vmin=vmin,
                        vmax=vmax, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.colorbar(artist, ax=ax, shrink=0.8, pad=0.02)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('circuit', nargs='?', default='testcase3')
    parser.add_argument('--checkpoint', default='ckpt_amg_L5_rev_best.pt')
    parser.add_argument('--device', default='cuda:2' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--layers', default='2,2,2,3,4',
                        help='GAT depth per AMG level, e.g. 2,2,2,3,4')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    layers = tuple(int(value) for value in args.layers.split(','))
    n_levels = len(layers)
    device = torch.device(args.device)
    spice_path = Path(BENCH) / 'real-circuit-data' / args.circuit / 'netlist.sp'
    if not spice_path.is_file():
        raise FileNotFoundError(spice_path)

    matrix, _, b_vdd, b_isource, names, vdd = parse_spice(str(spice_path))
    solution_path = Path(BENCH).parent / 'solution' / 'real-circuit-data' / f'{args.circuit}.solution'
    x_gt, mask = load_gt(str(solution_path), names)
    layer_idx = parse_layer(names)
    model = AMGGAT(in_dim=5, head_dim=32, heads=4,
                   layers_per_level=layers).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        level_stats = checkpoint.get('level_stats')
        target_stats = checkpoint.get('target_stats')
    else:
        model.load_state_dict(checkpoint)
        level_stats = None
        target_stats = None
    model.eval()

    levels = build_amg_hierarchy(
        matrix, max_levels=n_levels, coarse_mask=(b_vdd != 0))
    graphs, coarse_features, prolongations, *_ = prep_hierarchy(
        levels, b_vdd, b_isource, layer_idx, device)
    features = [build_features(matrix, b_vdd, b_isource, layer_idx, device)]
    features.extend(coarse_features[1:n_levels])
    if level_stats is not None:
        for level in range(n_levels):
            normalize_graph(graphs[level], level_stats[level]['edge'])
            if level > 0:
                features[level] = apply_node_zscore(
                    features[level], level_stats[level]['node'])
        features[0] = apply_node_zscore(features[0], level_stats[0]['node'])
    with torch.no_grad():
        prediction = model(graphs[:n_levels], features,
                          prolongations[:n_levels - 1]).cpu().numpy()

    if target_stats is None:
        pred_mv = prediction * vdd * 1000.0
    else:
        pred_mv = restore_zscore(torch.from_numpy(prediction), target_stats).numpy()
    gt_mv = (vdd - x_gt) * 1000.0
    error_mv = pred_mv - gt_mv
    x, y = parse_xy(names)
    selected = (layer_idx == 1) & mask & np.isfinite(x) & np.isfinite(y)
    if not selected.any():
        raise ValueError(f'{args.circuit} has no valid layer-1 nodes to plot')

    x_plot, y_plot = x[selected], y[selected]
    gt, pred, error = gt_mv[selected], pred_mv[selected], error_mv[selected]
    current = np.abs(b_isource[selected])
    print(f'{args.circuit}: n={selected.sum():,} VDD={vdd:.3f}V '
          f'GT mean={gt.mean():.3f}mV pred mean={pred.mean():.3f}mV '
          f'MAE={np.abs(error).mean():.3f}mV')

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f'{args.circuit} | layers={layers} | MAE={np.abs(error).mean():.3f}mV')
    vmax_drop = max(gt.max(), pred.max())
    scatter(axes[0, 0], x_plot, y_plot, gt, 'GT drop (mV)', vmin=0, vmax=vmax_drop)
    scatter(axes[0, 1], x_plot, y_plot, pred, 'Pred drop (mV)', vmin=0, vmax=vmax_drop)
    error_max = np.abs(error).max()
    scatter(axes[0, 2], x_plot, y_plot, error, 'Prediction error (mV)',
            cmap='RdBu_r', vmin=-error_max, vmax=error_max)
    scatter(axes[1, 0], x_plot, y_plot, np.abs(error), '|Error| (mV)',
            cmap='Reds', vmin=0)
    scatter(axes[1, 1], x_plot, y_plot, current * 1e6,
            'Current source magnitude (μA)', cmap='YlOrRd', vmin=0)

    axis = axes[1, 2]
    axis.scatter(gt, pred, c=np.abs(error), cmap='Reds', s=6, alpha=0.5,
                 vmin=0, vmax=np.abs(error).max(), rasterized=True)
    axis.plot([0, vmax_drop * 1.05], [0, vmax_drop * 1.05], 'k--', lw=1)
    axis.set_xlabel('GT drop (mV)')
    axis.set_ylabel('Predicted drop (mV)')
    axis.set_title('Prediction vs GT')
    output = args.output or f'diag_viz_{args.circuit}.png'
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
