"""Training-set z-score statistics for node, edge, and target values."""

import torch


class RunningStats:
    """Numerically stable feature-wise statistics accumulated by batches."""

    def __init__(self, width):
        self.count = 0
        self.mean = torch.zeros(width, dtype=torch.float64)
        self.m2 = torch.zeros(width, dtype=torch.float64)

    def update(self, values, mask=None):
        values = values.detach().reshape(-1, self.mean.numel()).to(
            device='cpu', dtype=torch.float64)
        if mask is not None:
            values = values[mask.detach().reshape(-1).cpu().bool()]
        if values.numel() == 0:
            return
        n = values.shape[0]
        batch_mean = values.mean(dim=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(dim=0)
        if self.count == 0:
            self.count, self.mean, self.m2 = n, batch_mean, batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * (n / total)
        self.m2 += batch_m2 + delta.square() * (self.count * n / total)
        self.count = total

    def result(self):
        if not self.count:
            raise ValueError('Cannot fit normalization stats on empty data')
        std = torch.sqrt(self.m2 / self.count).clamp_min(1e-8)
        return {'mean': self.mean.float(), 'std': std.float()}


def fit_stats(batches, width, masks=None):
    """Average per-circuit feature statistics with equal circuit weights.

    Each tensor is one circuit (or one circuit at a particular AMG level).
    Its mean/std are computed over that circuit's nodes/edges, then circuit
    statistics are averaged. This matches CFIRSTNET's get_min_max_mean_std.
    """
    if masks is None:
        masks = [None] * len(batches)
    circuit_means = []
    circuit_stds = []
    for values, mask in zip(batches, masks):
        values = values.detach().reshape(-1, width).to(
            device='cpu', dtype=torch.float64)
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
            values = values[mask]
        if values.numel() == 0:
            continue
        circuit_means.append(values.mean(dim=0))
        # Match torch.std's default sample-standard-deviation convention.
        if values.shape[0] > 1:
            circuit_stds.append(values.std(dim=0, unbiased=True))
        else:
            circuit_stds.append(torch.zeros(width, dtype=values.dtype))
    if not circuit_means:
        raise ValueError('Cannot fit normalization stats on empty data')
    mean = torch.stack(circuit_means).mean(dim=0)
    std = torch.stack(circuit_stds).mean(dim=0).clamp_min(1e-8)
    return {'mean': mean.float(), 'std': std.float()}


def apply_zscore(values, stats):
    mean = torch.as_tensor(stats['mean'], device=values.device,
                           dtype=values.dtype)
    std = torch.as_tensor(stats['std'], device=values.device,
                          dtype=values.dtype)
    return (values - mean) / std


def apply_node_zscore(features, stats):
    """Z-score electrical features while preserving layer's [-1, 1] encoding."""
    result = features.clone()
    result[:, :4] = apply_zscore(features[:, :4], stats)
    return result


def restore_zscore(values, stats):
    """Convert model z-scores back to the original physical units."""
    mean = torch.as_tensor(stats['mean'], device=values.device,
                           dtype=values.dtype)
    std = torch.as_tensor(stats['std'], device=values.device,
                          dtype=values.dtype)
    return values * std + mean


def normalize_graph(graph, stats):
    """Keep edge features as log1p conductances, matching the old pipeline."""
    return graph


def normalize_precomputed_levels(samples, stats_by_level):
    """Normalize node features; keep log1p edge features in physical graph scale."""
    for sample in samples:
        for level, stats in enumerate(stats_by_level):
            if level >= sample['n_levels']:
                break
            if level > 0 and sample['cfeats'][level] is not None:
                sample['cfeats'][level] = apply_node_zscore(
                    sample['cfeats'][level], stats['node'])
    return samples
