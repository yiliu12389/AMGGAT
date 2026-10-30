"""Reusable graph-attention layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim, head_dim, heads=4, concat=True,
                 slope=0.2, activation=None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.concat = concat
        self.slope = slope
        self.activation = activation
        self.W = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.W_e = nn.Linear(1, heads * head_dim, bias=False)
        self.a = nn.Parameter(torch.empty(heads, 3 * head_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.W_e.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, row, col, n_nodes, edge_feat):
        n_heads, head_dim = self.heads, self.head_dim
        n_edges = edge_feat.shape[0]
        h_transformed = self.W(h).view(n_nodes, n_heads, head_dim)
        h_dst, h_src = h_transformed[col], h_transformed[row]
        edge_transformed = self.W_e(edge_feat.reshape(n_edges, 1)).view(
            n_edges, n_heads, head_dim)
        edge_input = torch.cat([h_dst, h_src, edge_transformed], dim=-1)
        logits = F.leaky_relu((edge_input * self.a).sum(-1), self.slope)

        dst_by_head = col.unsqueeze(-1).expand(-1, n_heads)
        maxima = torch.full((n_nodes, n_heads), -1e30,
                            device=h.device, dtype=h.dtype)
        maxima.scatter_reduce_(0, dst_by_head, logits, reduce='amax',
                               include_self=True)
        exp_logits = (logits - maxima[col]).exp()
        normalizers = torch.zeros(n_nodes, n_heads, device=h.device, dtype=h.dtype)
        normalizers.scatter_add_(0, dst_by_head, exp_logits)
        attention = exp_logits / (normalizers[col] + 1e-16)

        messages = attention.unsqueeze(-1) * h_src
        aggregate = torch.zeros(n_nodes, n_heads, head_dim,
                                device=h.device, dtype=h.dtype)
        aggregate.scatter_add_(
            0, col.view(-1, 1, 1).expand(-1, n_heads, head_dim), messages)
        output = h_transformed + aggregate
        output = (output.reshape(n_nodes, n_heads * head_dim)
                  if self.concat else output.mean(dim=1))
        return self.activation(output) if self.activation is not None else output
