"""Temporal encoder, graph convolution and forecast head (Q4b, Q4c, Q5)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import T_IN, T_OUT


def row_normalise_with_self_loops(adj):
    """Directed normalisation for the fixed graph: A_hat = D^-1 (A + I), D = row sums of A + I.

    The LargeST SD road-distance matrix is not symmetric, so the symmetric D^-1/2 (A+I) D^-1/2 form
    does not apply. Row normalisation makes each node take a weighted average over itself and the
    sensors j with A_ij != 0, which keeps the scale of Z unchanged.
    """
    a = np.array(adj, dtype=np.float64)
    np.fill_diagonal(a, 0.0)
    a = a + np.eye(a.shape[0])
    a = a / a.sum(1, keepdims=True)
    return torch.tensor(a, dtype=torch.float32)


class TemporalEncoder(nn.Module):
    """Per-sensor temporal convolution: (B, 8, N) -> (B, N, d). Weights are shared across sensors."""

    def __init__(self, d=32, k=3):
        super().__init__()
        self.c1 = nn.Conv2d(1, d, (1, k))                 # over time, one sensor at a time
        self.c2 = nn.Conv2d(d, d, (1, k))
        t_left = T_IN - 2 * (k - 1)
        self.proj = nn.Linear(d * t_left, d)

    def forward(self, x):
        h = x.transpose(1, 2).unsqueeze(1)                 # (B, 1, N, 8)
        h = F.relu(self.c1(h))
        h = F.relu(self.c2(h))                             # (B, d, N, 4)
        B, d, N, t = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B, N, d * t)
        return F.relu(self.proj(h))                        # (B, N, d)


class AdaptiveAdjacency(nn.Module):
    """A_adaptive = row-softmax(ReLU(E1 E2^T)), E1, E2 in R^{N x de}, randomly initialised."""

    def __init__(self, n, de=10):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(n, de))
        self.E2 = nn.Parameter(torch.randn(n, de))

    def forward(self):
        return torch.softmax(F.relu(self.E1 @ self.E2.T), dim=1)


class Forecaster(nn.Module):
    """graph in {"none", "fixed", "adaptive", "hybrid"}.

    Graph models: Z = f_temporal(X), H = ReLU(A_hat Z W_g), Y_hat = f_head(H).
    Temporal-only ("none"): the same encoder and head without the graph layer, Y_hat = f_head(Z).
    skip=True gives the head Z + H instead of H (used only for the comparison in Q4(c)).
    """

    def __init__(self, n, graph="fixed", adj=None, d=32, de=10, skip=False):
        super().__init__()
        self.graph = graph
        self.skip = skip
        self.enc = TemporalEncoder(d)
        if graph != "none":
            self.Wg = nn.Linear(d, d, bias=False)
        if graph in ("fixed", "hybrid"):
            self.register_buffer("A_fixed", row_normalise_with_self_loops(adj))
        if graph in ("adaptive", "hybrid"):
            self.adapt = AdaptiveAdjacency(n, de)
        if graph == "hybrid":
            self.alpha_logit = nn.Parameter(torch.zeros(()))   # alpha = sigmoid(logit), learned, starts at 0.5
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, T_OUT))

    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def adjacency(self):
        if self.graph == "fixed":
            return self.A_fixed
        if self.graph == "adaptive":
            return self.adapt()
        if self.graph == "hybrid":
            a = self.alpha()
            return a * self.A_fixed + (1 - a) * self.adapt()   # both row-stochastic
        return None

    def forward(self, x):
        z = self.enc(x)                                     # (B, N, d)
        if self.graph != "none":
            A = self.adjacency()
            h = F.relu(torch.einsum("ij,bjd->bid", A, self.Wg(z)))   # H, graph product over nodes per batch item
            z = z + h if self.skip else h
        return self.head(z).transpose(1, 2)                 # (B, 4, N)
