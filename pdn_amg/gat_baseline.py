
import os, time
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pdn_data import (build_graph, build_features, spmv, load_group,
                       load_group_flat, sample_anchor_indices,
                       load_rw_anchor_cache)


# ─────────────────────────────────────────────────────────────────────────────
# Vanilla GAT
# ─────────────────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    def __init__(self, in_dim, head_dim, heads=4,
                 concat=True, slope=0.2, activation=None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.concat = concat
        self.slope = slope
        self.activation = activation
        self.W_neigh = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.W_self  = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.a = nn.Parameter(torch.empty(heads, 2 * head_dim + 1))
        nn.init.xavier_uniform_(self.W_neigh.weight)
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, row, col, N, edge_feat):
        H, D = self.heads, self.head_dim
        E = edge_feat.shape[0]
        hn = self.W_neigh(h).view(N, H, D)
        hs = self.W_self(h).view(N, H, D)
        h_dst = hn[col]
        h_src = hn[row]
        ef = edge_feat.view(E, 1, 1).expand(E, H, 1)
        x_e = torch.cat([h_dst, h_src, ef], dim=-1)
        a_logit = F.leaky_relu((x_e * self.a).sum(-1), self.slope)

        col_h = col.unsqueeze(-1).expand(-1, H)
        a_max = torch.full((N, H), -1e30, device=h.device, dtype=h.dtype)
        a_max.scatter_reduce_(0, col_h, a_logit, reduce='amax', include_self=True)
        alpha = (a_logit - a_max[col]).exp()
        norm = torch.zeros(N, H, device=h.device, dtype=h.dtype)
        norm.scatter_add_(0, col_h, alpha)
        alpha = alpha / (norm[col] + 1e-16)

        msg = alpha.unsqueeze(-1) * h_src
        agg = torch.zeros(N, H, D, device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, col.view(-1, 1, 1).expand(-1, H, D), msg)

        out = hs + agg
        if self.concat:
            out = out.view(N, H * D)
        else:
            out = out.mean(dim=1)
        if self.activation is not None:
            out = self.activation(out)
        return out


class GAT(nn.Module):
    def __init__(self, in_dim=5, head_dim=32, heads=4, n_layers=6, use_bc=False):
        super().__init__()
        self.use_bc = use_bc
        effective_in_dim = in_dim - 2 if use_bc else in_dim
        layers = []
        d = effective_in_dim
        for l in range(n_layers - 1):
            layers.append(GATLayer(d, head_dim, heads=heads,
                                   concat=True, activation=F.elu))
            d = head_dim * heads
        layers.append(GATLayer(d, head_dim, heads=heads,
                               concat=False, activation=None))
        self.layers = nn.ModuleList(layers)
        if use_bc:
            hidden_dim = head_dim * heads
            self.bc_proj = nn.Linear(1, hidden_dim)
            nn.init.xavier_uniform_(self.bc_proj.weight)
            nn.init.zeros_(self.bc_proj.bias)
        dec_hidden = head_dim * 2
        self.dec = nn.Sequential(
            nn.Linear(head_dim, dec_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dec_hidden, 1),
            nn.Softplus(),
        )
        nn.init.xavier_uniform_(self.dec[0].weight)
        nn.init.zeros_(self.dec[0].bias)
        nn.init.normal_(self.dec[2].weight, std=0.01)
        nn.init.constant_(self.dec[2].bias, -6.5)

    def forward(self, graph, x):
        row, col, N = graph['row'], graph['col'], graph['n']
        ef = graph['edge_feat']
        if self.use_bc:
            bc_val   = x[:, 0:1]    
            is_known = x[:, 1:2]   
            h = x[:, 2:]          
            bc_embed = self.bc_proj(bc_val)   
        else:
            h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, row, col, N, ef)
            if self.use_bc and i < len(self.layers) - 1:
                h = h + is_known * bc_embed
        return self.dec(h).squeeze(-1)                               


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    HERE = os.path.dirname(os.path.abspath(__file__))
    REPO = os.path.dirname(HERE)
    os.chdir(HERE)
    device = os.environ.get('DEVICE', 'cuda:2' if torch.cuda.is_available() else 'cpu')
    N_EPOCHS = int(os.environ.get('EPOCHS', 50))
    LR = float(os.environ.get('LR', 1e-4))
    HEAD_DIM = int(os.environ.get('HEAD_DIM', 32))
    HEADS = int(os.environ.get('HEADS', 4))
    N_LAYERS = int(os.environ.get('N_LAYERS', 6))
    LAM = 0.1
    CKPT_NAME = os.environ.get('CKPT_NAME', 'ckpt_gat')
    ANCHOR_FRAC = float(os.environ.get('ANCHOR_FRAC', 0.0))
    ANCHOR_MAX  = float(os.environ.get('ANCHOR_MAX', 0.0))   # >0 => per-graph uniform [0, ANCHOR_MAX]
    LOSS = os.environ.get('LOSS', 'mse').lower()          # 'mse' | 'log_mse'
    LOG_EPS = float(os.environ.get('LOG_EPS', 0.01))       # mV floor for log
    ENCODING = os.environ.get('ENCODING', 'v_ratio')       # 'v_ratio' | 'known_drop'
    USE_BC = os.environ.get('USE_BC', '0') == '1'          # if 1, GAT uses boundary-condition injection
    USE_RW_ANCHORS = os.environ.get('USE_RW_ANCHORS', '0') == '1'
    RW_M = int(os.environ.get('RW_M', 30))
    RW_SEED = int(os.environ.get('RW_SEED', 42))

    from metrics import eval_samples, print_group_table, format_epoch_line

    feat_fn = partial(build_features, encoding=ENCODING)
    in_dim = 6 if ENCODING == 'known_drop' else 5

    print(f'Device: {device}  ANCHOR_FRAC={ANCHOR_FRAC}', flush=True)
    t0 = time.time()
    train_samples = load_group_flat('fake-circuit-data')
    real_samples = load_group('real-circuit-data')
    hidden_samples = load_group('hidden-real-circuit-data')
    print(f'Train={len(train_samples)} (fake)  '
          f'Eval={len(real_samples)} real + {len(hidden_samples)} hidden  '
          f'({time.time()-t0:.0f}s)', flush=True)

    from metrics import metrics_A, metrics_B

    def _rw_eval(model, samples, ds):
        model.eval()
        per_tc, per_A, per_B, per_T = [], [], [], []
        for tc, A_sp, b_np, b_vdd, b_isource, layer_idx, x_gt, mask, vdd in samples:
            cache = load_rw_anchor_cache(ds, tc, ANCHOR_FRAC, RW_M, RW_SEED)
            if cache is None:
                continue
            anchor_idx = cache['anchor_idx']
            x_gt_use = x_gt.copy()
            x_gt_use[anchor_idx] = cache['V_hat']
            try:
                g = build_graph(A_sp, b_vdd, device)
                xin = feat_fn(A_sp, b_vdd, b_isource, layer_idx, device,
                              x_gt=x_gt_use, gt_mask=mask, vdd=vdd,
                              anchor_indices=anchor_idx)
                drop_gt_n = torch.from_numpy(
                    ((vdd - x_gt) / vdd).astype(np.float32)).to(device)
                t0 = time.time()
                with torch.no_grad():
                    drop_pred = model(g, xin)
                per_T.append(time.time() - t0)
                mm_b = torch.from_numpy(mask).to(device)
                mae_b, r2_b = metrics_B(drop_pred[mm_b], drop_gt_n[mm_b], vdd)
                per_B.append((mae_b, r2_b))
                m1 = (layer_idx == 1) & mask
                if m1.sum() == 0:
                    mae_a, f1_a, cc_a = float('nan'), float('nan'), float('nan')
                else:
                    mm_a = torch.from_numpy(m1).to(device)
                    mae_a, f1_a, cc_a = metrics_A(drop_pred[mm_a], drop_gt_n[mm_a], vdd)
                per_A.append((mae_a, f1_a, cc_a))
                per_tc.append((tc, mae_a, f1_a, cc_a, mae_b, r2_b, per_T[-1]))
            except Exception as ex:
                print(f'  eval skip {tc}: {type(ex).__name__}: {ex}', flush=True)

        def _mean(xs, mask_nan=False):
            if not xs: return None
            a = np.array(xs, dtype=float)
            if mask_nan:
                a = a[~np.isnan(a[:, 0])]
            if a.size == 0:
                return tuple(float('nan') for _ in xs[0])
            return tuple(a.mean(axis=0).tolist())
        return {'tag': '', 'per_tc': per_tc,
                'mean_A': _mean(per_A, mask_nan=True),
                'mean_B': _mean(per_B),
                'mean_time': float(np.mean(per_T)) if per_T else float('nan')}

    def eval_on(model, samples, ds='real-circuit-data'):
        if USE_RW_ANCHORS:
            return _rw_eval(model, samples, ds)
        return eval_samples(model, samples, device, feature_fn=feat_fn,
                            anchor_frac=ANCHOR_FRAC)

    print('=' * 64, flush=True)
    model = GAT(in_dim=in_dim, head_dim=HEAD_DIM, heads=HEADS,
                n_layers=N_LAYERS, use_bc=USE_BC).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'GAT: head_dim={HEAD_DIM} heads={HEADS} layers={N_LAYERS} '
          f'in_dim={in_dim}  params={n_params:,}', flush=True)

    r0 = eval_on(model, real_samples, 'real-circuit-data')
    h0 = eval_on(model, hidden_samples, 'hidden-real-circuit-data')
    print(f'untrained  {format_epoch_line(r0, "real")}', flush=True)
    print(f'untrained  {format_epoch_line(h0, "hidden")}', flush=True)

    WARMUP = int(os.environ.get('WARMUP', 5))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    warmup_sch = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, end_factor=1.0, total_iters=WARMUP)
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(N_EPOCHS - WARMUP, 1))
    sch = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sch, cosine_sch], milestones=[WARMUP])
    best_mae = float('inf')
    history = []
    t_train = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        ep_sup, ep_res, ep_tot, n_ok = 0.0, 0.0, 0.0, 0
        for idx, (name, A_sp, b_np, b_vdd, b_isource, layer_idx, x_gt, mask, vdd) in enumerate(train_samples):
            try:
                g = build_graph(A_sp, b_vdd, device)
                if ANCHOR_MAX > 0:
                    af = float(np.random.default_rng(epoch * 100000 + idx).uniform(0, ANCHOR_MAX))
                else:
                    af = ANCHOR_FRAC
                is_pin_np = (b_vdd > 0)
                if USE_RW_ANCHORS:
                    cache = load_rw_anchor_cache('fake-circuit-data', name,
                                                  af, RW_M, RW_SEED)
                    if cache is None:
                        raise RuntimeError(f'RW anchor cache missing for {name}')
                    anchor_idx = cache['anchor_idx']
                    x_gt_use = x_gt.copy()
                    x_gt_use[anchor_idx] = cache['V_hat']
                else:
                    anchor_rng = np.random.default_rng(idx)
                    anchor_idx = sample_anchor_indices(
                        is_pin_np, mask.astype(bool), af, anchor_rng)
                    x_gt_use = x_gt
                x_in = feat_fn(A_sp, b_vdd, b_isource, layer_idx, device,
                              x_gt=x_gt_use, gt_mask=mask, vdd=vdd,
                              anchor_indices=anchor_idx)
                m = torch.from_numpy(mask).to(device)
                b_r = torch.from_numpy(b_np.astype(np.float32)).to(device)
                drop_gt_n = torch.from_numpy(
                    ((vdd - x_gt) / vdd).astype(np.float32)).to(device)
            except Exception:
                continue
            opt.zero_grad()
            drop = model(g, x_in)
            pred_mv = drop[m] * vdd * 1000
            gt_mv   = drop_gt_n[m] * vdd * 1000
            if LOSS == 'log_mse':
                lp = torch.log(pred_mv.clamp(min=LOG_EPS))
                lg = torch.log(gt_mv.clamp(min=LOG_EPS))
                loss_sup = ((lp - lg) ** 2).mean()
            else:
                loss_sup = ((pred_mv - gt_mv) ** 2).mean()

            with torch.no_grad():
                v = (1.0 - drop) * vdd
                r = spmv(g['A'], v) - b_r
                loss_res = (r * r).sum() / (b_r * b_r).sum().clamp_min(1e-12)

            loss = loss_sup                      
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_sup += loss_sup.item()
            ep_res += loss_res.item()
            ep_tot += loss.item()
            n_ok   += 1
        sch.step()

        d = max(n_ok, 1)
        lr_now = opt.param_groups[0]['lr']
        train_rmse = (ep_sup / d) ** 0.5
        unit = 'log' if LOSS == 'log_mse' else 'mV'
        msg = (f'epoch {epoch:3d}  lr={lr_now:.2e}  '
               f'train_RMS={train_rmse:.4f} {unit}  '
               f'loss={LOSS}  '
               f'Loss={ep_tot/d:.4f}  physics_res(unused)={ep_res/d:.4f}')
        if epoch % 5 == 0 or epoch == 1:
            r = eval_on(model, real_samples, 'real-circuit-data')
            mae_r = r['mean_B'][0]
            msg += f'  ||  {format_epoch_line(r, "real")}  ({time.time()-t_train:.0f}s)'
            history.append((epoch, r))
            if mae_r < best_mae:
                best_mae = mae_r
                torch.save(model.state_dict(), CKPT_NAME + '_best.pt')
        else:
            msg += f'  ({time.time()-t_train:.0f}s)'
        print(msg, flush=True)

    torch.save(model.state_dict(), CKPT_NAME + '_final.pt')
    print('=' * 64, flush=True)
    if history:
        mae_bs = [h[1]['mean_B'][0] for h in history]
        print(f'best real (all-node) MAE: {min(mae_bs):.4f} mV', flush=True)

    if os.path.exists(CKPT_NAME + '_best.pt'):
        model.load_state_dict(torch.load(CKPT_NAME + '_best.pt',
                                         map_location=device))
        print_group_table(eval_on(model, real_samples, 'real-circuit-data'),
                          label='REAL   (best ckpt)')
        print_group_table(eval_on(model, hidden_samples, 'hidden-real-circuit-data'),
                          label='HIDDEN (best ckpt)')
    print('Done.', flush=True)
