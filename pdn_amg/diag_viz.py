"""
Visualize per-node prediction quality on one real circuit.
Outputs: diag_viz_<tc>.png  (2x3 subplots)
"""
import os, re, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pdn_data import load_group, build_features, parse_spice, load_gt, parse_layer
from amg_gat import AMGGAT, precompute

DEVICE   = 'cuda:2'
CKPT     = 'ckpt_amg_L5_rev_best.pt'
LAYERS   = (2, 2, 2, 3, 4)
N_LEVELS = len(LAYERS)
MAX_LV   = 5

# ── parse x,y from node name ──────────────────────────────────────────────────
_NAME_RE = re.compile(r'n(\d+)_m(\d+)_(\d+)_(\d+)')
def parse_xy(names):
    x = np.full(len(names), np.nan)
    y = np.full(len(names), np.nan)
    for i, nm in enumerate(names):
        if nm is None: continue
        m = _NAME_RE.match(nm)
        if m:
            x[i] = float(m.group(3))
            y[i] = float(m.group(4))
    return x, y

# ── load model ────────────────────────────────────────────────────────────────
model = AMGGAT(in_dim=5, head_dim=32, heads=4,
               layers_per_level=LAYERS).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

# ── pick testcase ─────────────────────────────────────────────────────────────
TC = sys.argv[1] if len(sys.argv) > 1 else 'testcase3'

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'ML-for-IR-drop/benchmarks')
SOL   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'ML-for-IR-drop/solution')

sp_file = os.path.join(BENCH, 'real-circuit-data', TC, 'netlist.sp')
A_sp, b_np, b_vdd, b_isource, names = parse_spice(sp_file)
gt_path = os.path.join(SOL, 'real-circuit-data', f'{TC}.solution')
x_gt, mask = load_gt(gt_path, names)
layer_idx = parse_layer(names)
vdd = float(b_vdd.max() / 1000.0)

px, py = parse_xy(names)

# ── run model ─────────────────────────────────────────────────────────────────
from amg_coarsen import build_amg_hierarchy, prep_hierarchy
amg_lvls = build_amg_hierarchy(A_sp, max_levels=MAX_LV)
graphs, cfeats, Ps, bv_ls, bi_ls, li_ls = prep_hierarchy(
    amg_lvls, b_vdd, b_isource, layer_idx, DEVICE)
x0 = build_features(A_sp, b_vdd, b_isource, layer_idx, DEVICE)
feats = [x0] + cfeats[1:N_LEVELS]

with torch.no_grad():
    pred_n = model(graphs[:N_LEVELS], feats, Ps[:N_LEVELS-1])

pred_mv = pred_n.cpu().numpy() * vdd * 1000.0
gt_mv   = ((vdd - x_gt) / vdd) * vdd * 1000.0   # = (vdd - x_gt)*1000
err_mv  = pred_mv - gt_mv

# ── select layer-1 nodes with solution ───────────────────────────────────────
m1 = (layer_idx == 1) & mask & ~np.isnan(px) & ~np.isnan(py)

xs  = px[m1];  ys  = py[m1]
gt  = gt_mv[m1]
pr  = pred_mv[m1]
er  = err_mv[m1]
I   = np.abs(b_isource[m1])

print(f'{TC}  n_nodes={m1.sum()}  vdd={vdd:.3f}V')
print(f'  gt  min={gt.min():.3f}  max={gt.max():.3f}  mean={gt.mean():.3f} mV')
print(f'  pred min={pr.min():.3f}  max={pr.max():.3f}  mean={pr.mean():.3f} mV')
print(f'  MAE={np.abs(er).mean():.3f} mV')

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(f'{TC}  –  AMG-GAT L5-rev {LAYERS}  |  MAE={np.abs(er).mean():.3f}mV  '
             f'max_gt={gt.max():.2f}mV  max_pred={pr.max():.2f}mV', fontsize=13)

s = 4

def scatter(ax, xv, yv, cv, title, cmap='hot_r', vmin=None, vmax=None):
    sc = ax.scatter(xv, yv, c=cv, s=s, cmap=cmap, vmin=vmin, vmax=vmax,
                    rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)

vmax_drop = max(gt.max(), pr.max())
scatter(axes[0,0], xs, ys, gt, 'GT drop (mV)',    vmin=0, vmax=vmax_drop)
scatter(axes[0,1], xs, ys, pr, 'Pred drop (mV)',  vmin=0, vmax=vmax_drop)

er_max = np.abs(er).max()
scatter(axes[0,2], xs, ys, er,
        'Error = pred-GT (mV)\n(red=over, blue=under)',
        cmap='RdBu_r', vmin=-er_max, vmax=er_max)

scatter(axes[1,0], xs, ys, np.abs(er),
        '|Error| (mV)',  cmap='Reds', vmin=0)

scatter(axes[1,1], xs, ys, I*1e6,
        'b_isource (μA)',  cmap='YlOrRd', vmin=0)

ax = axes[1,2]
vmax_sc = max(gt.max(), pr.max()) * 1.05
ax.scatter(gt, pr, c=np.abs(er), cmap='Reds', s=6, alpha=0.5,
           vmin=0, vmax=np.abs(er).max(), rasterized=True)
ax.plot([0, vmax_sc], [0, vmax_sc], 'k--', lw=1, label='y=x')
ax.set_xlabel('GT drop (mV)');  ax.set_ylabel('Pred drop (mV)')
ax.set_title('Pred vs GT  (color=|error|)', fontsize=10)
ax.legend(fontsize=8)
cc = np.corrcoef(gt, pr)[0,1]
ss_res = ((pr - gt)**2).sum()
ss_tot = ((gt - gt.mean())**2).sum()
r2 = 1 - ss_res/(ss_tot+1e-30)
ax.text(0.05, 0.92, f'CC={cc:.3f}  R2={r2:.3f}', transform=ax.transAxes,
        fontsize=9, bbox=dict(fc='white', alpha=0.7))

plt.tight_layout()
out = f'diag_viz_{TC}.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
