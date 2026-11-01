"""Precompute circuit samples on an AMG hierarchy."""

from amg.hierarchy import (
    build_amg_hierarchy, prep_hierarchy, solve_coarse_level_systems,
)


def precompute(samples, device, max_levels=4, solve_targets=True):
    prepared = []
    for tc, matrix, rhs, b_vdd, b_isource, layers, x_gt, mask, vdd in samples:
        try:
            levels = build_amg_hierarchy(
                matrix, max_levels=max_levels, coarse_mask=(b_vdd != 0))
            # Exact voltage solutions x1, x2, ... used as optional local
            # supervision targets.  x_coarse[l] lives on A_l.
            coarse_results = (solve_coarse_level_systems(levels, rhs)
                              if solve_targets else [])
            x_coarse = [None] + [result['x'] for result in coarse_results]
            graphs, coarse_features, prolongations, bv, bi, li = prep_hierarchy(
                levels, b_vdd, b_isource, layers, device)
            prepared.append({
                'tc': tc, 'A_sp': matrix, 'b_np': rhs,
                'b_vdd': b_vdd, 'b_isource': b_isource,
                'layer_idx': layers, 'x_gt': x_gt, 'mask': mask, 'vdd': vdd,
                'graphs': graphs, 'cfeats': coarse_features,
                'Ps': prolongations, 'bv_ls': bv, 'bi_ls': bi, 'li_ls': li,
                'x_coarse': x_coarse,
                'n_levels': len(levels),
            })
        except Exception as error:
            print(f'  precompute skip {tc}: {error}', flush=True)
    return prepared
