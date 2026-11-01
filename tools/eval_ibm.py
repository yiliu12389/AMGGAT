import gc
import glob
import os
import subprocess
import sys

import torch

from data import build_features, iter_group_flat
from data.amg_dataset import precompute
from data.normalization import apply_node_zscore, normalize_precomputed_levels
from evaluation.metrics import eval_samples, print_group_table
from models.coarsetofine import AMGGAT
from models.gat import GAT
from models.total import TotalAMGGAT


def run_vanilla(sample, checkpoint, device):
    node_stats = checkpoint['node_stats']
    target_stats = checkpoint['target_stats']
    model = GAT(in_dim=5, head_dim=32, heads=4, n_layers=6).to(device)
    model.load_state_dict(checkpoint['model'])

    def feature_fn(A_sp, b_vdd, b_isource, layers, run_device):
        features = build_features(A_sp, b_vdd, b_isource, layers, run_device)
        return apply_node_zscore(features, node_stats)

    result = eval_samples(model, [sample], device, feature_fn=feature_fn,
                          node_stats=None, edge_stats=None,
                          target_stats=target_stats)
    print_group_table(result, 'IBM Vanilla GAT')
    print(f'ALL_NODES MAE={result["mean_B"][0]:.4f}mV '
          f'R2={result["mean_B"][1]:+.4f} CC={result["mean_CC"]:+.4f} '
          f'<=10={result["within10_pct"]:.2f}%', flush=True)


def run_amg(sample, checkpoint, device, model_name):
    level_stats = checkpoint['level_stats']
    target_stats = checkpoint['target_stats']
    if model_name == 'total':
        model = TotalAMGGAT(n_levels=4, feature_dim=5, head_dim=32,
                            heads=4, gat_layers=3).to(device)
    else:
        model = AMGGAT(in_dim=5, head_dim=32, heads=4,
                       layers_per_level=(3, 3, 3, 3)).to(device)
    model.load_state_dict(checkpoint['model'])
    data = precompute([sample], device, max_levels=4, solve_targets=False)
    print(f'AMG precompute: {len(data)} sample(s)', flush=True)
    normalize_precomputed_levels(data, level_stats)
    if len(data) != 1 or data[0]['n_levels'] < 4:
        raise RuntimeError('IBM AMG preparation did not produce four levels')
    prepared = data[0]
    model.eval()

    class PreparedModel(torch.nn.Module):
        def forward(self, graph, features):
            return model(prepared['graphs'][:4],
                         [features] + prepared['cfeats'][1:4],
                         prepared['Ps'][:3])

    result = eval_samples(
        PreparedModel(), [sample], device, node_stats=level_stats[0]['node'],
        target_stats=target_stats)
    print_group_table(result, f'IBM {model_name}')
    print(f'ALL_NODES MAE={result["mean_B"][0]:.4f}mV '
          f'R2={result["mean_B"][1]:+.4f} CC={result["mean_CC"]:+.4f} '
          f'<=10={result["within10_pct"]:.2f}%', flush=True)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(here)
    device = os.environ.get(
        'DEVICE', 'cuda:8' if torch.cuda.is_available() else 'cpu')
    model_name = os.environ.get('MODEL', 'vanilla').lower()
    default_ckpt = {
        'vanilla': 'checkpoints/vanilla_gat6_logedge_best.pt',
        'total': 'checkpoints/total4_layer_shared_logedge_best.pt',
        'coarsetofine': 'checkpoints/coarsetofine4_cat_global_zscore_best.pt',
    }
    if model_name not in default_ckpt:
        raise ValueError('MODEL must be vanilla, total, or coarsetofine')
    ckpt_path = os.environ.get('CKPT', default_ckpt[model_name])
    print(f'Device: {device}', flush=True)
    print(f'Checkpoint: {ckpt_path}', flush=True)

    only = os.environ.get('IBM_ONLY', '').strip()
    if not only:
        bench = os.path.join(os.path.dirname(here), 'ML-for-IR-drop',
                             'benchmarks', 'ibm-circuit-data')
        names = [os.path.splitext(os.path.basename(path))[0]
                 for path in sorted(glob.glob(os.path.join(bench, '*.spice')))]
        for name in names:
            child_env = os.environ.copy()
            child_env['IBM_ONLY'] = name
            print(f'\n===== {name} =====', flush=True)
            completed = subprocess.run(
                [sys.executable, __file__], env=child_env, check=False)
            if completed.returncode:
                print(f'FAILED {name}: exit code {completed.returncode}',
                      flush=True)
        return

    print(f'Model: {model_name}', flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device)
    n_tested = 0
    for sample in iter_group_flat('ibm-circuit-data', names=[only]):
        print(f'\n===== {sample[0]} =====', flush=True)
        try:
            if model_name == 'vanilla':
                run_vanilla(sample, checkpoint, device)
            else:
                run_amg(sample, checkpoint, device, model_name)
        finally:
            del sample
            gc.collect()
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
        n_tested += 1
    if not n_tested:
        raise RuntimeError('No IBM samples found')


if __name__ == '__main__':
    main()
