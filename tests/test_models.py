"""Model checks: shapes, adjacency normalisation, fixed graph frozen, temporal-only equivalence, metrics."""
import numpy as np
import torch

from pa3.models import AdaptiveAdjacency, Forecaster, row_normalise_with_self_loops
from pa3.train import evaluate

N = 30


def rand_adj(seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((N, N)) < 0.2) * rng.uniform(0.01, 1, (N, N))
    np.fill_diagonal(A, 1.0)
    A[5] = 0; A[5, 5] = 1.0                                        # isolated node
    return A.astype(np.float32)


def test_fixed_normalisation_is_row_stochastic_and_respects_edges():
    A = rand_adj()
    Ah = row_normalise_with_self_loops(A).numpy()
    assert np.allclose(Ah.sum(1), 1, atol=1e-6)
    off = ~np.eye(N, dtype=bool)
    assert ((Ah > 0) & off <= (A > 0) & off).all()                 # no edge where A has none
    assert np.isclose(Ah[5, 5], 1.0)                               # isolated node keeps itself only
    assert np.allclose(np.diag(Ah), 1 / (A.sum(1) - np.diag(A) + 1), atol=1e-6)


def test_shapes_and_row_sums_all_graphs():
    x = torch.randn(4, 8, N)
    for g in ["none", "fixed", "adaptive", "hybrid"]:
        torch.manual_seed(0)
        m = Forecaster(N, g, rand_adj(), d=16, de=4)
        assert m(x).shape == (4, 4, N)
        A = m.adjacency()
        if g != "none":
            assert A.shape == (N, N) and torch.allclose(A.sum(1), torch.ones(N), atol=1e-5)
            assert (A >= 0).all()


def test_topk_sparsification_rows():
    torch.manual_seed(0)
    a = AdaptiveAdjacency(N, 4, topk=5)
    A = a()
    assert torch.allclose(A.sum(1), torch.ones(N), atol=1e-5)
    assert ((A > 0).sum(1) >= 5).all()                             # ties at the k-th logit can keep a few more


def test_fixed_graph_not_trained_and_adaptive_is():
    torch.manual_seed(0)
    A = rand_adj()
    for g in ["fixed", "adaptive", "hybrid"]:
        m = Forecaster(N, g, A, d=16, de=4)
        names = {n for n, _ in m.named_parameters()}
        assert "A_fixed" not in names
        before = m.adjacency().detach().clone()
        opt = torch.optim.Adam(m.parameters(), lr=0.05)
        for _ in range(3):
            loss = m(torch.randn(8, 8, N)).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        after = m.adjacency().detach()
        if g == "fixed":
            assert torch.equal(before, after)
        else:
            assert not torch.allclose(before, after)
            assert any(n.startswith("adapt.E") for n in names)
        if g == "hybrid":
            assert 0 < float(m.alpha().detach()) < 1 and "alpha_logit" in names


def test_temporal_only_equals_graph_model_with_graph_term_removed():
    torch.manual_seed(0)
    base = Forecaster(N, "none", None, d=16)
    full = Forecaster(N, "fixed", rand_adj(), d=16)
    full.load_state_dict({**full.state_dict(), **base.state_dict()}, strict=False)
    with torch.no_grad():
        full.Wg.weight.zero_()
    x = torch.randn(3, 8, N)
    assert torch.allclose(base(x), full(x), atol=1e-6)


def test_graph_term_matches_formula():
    torch.manual_seed(0)
    m = Forecaster(N, "fixed", rand_adj(), d=16)
    x = torch.randn(2, 8, N)
    z = m.enc(x)
    h = torch.relu(torch.stack([m.A_fixed @ z[b] @ m.Wg.weight.T for b in range(2)]))
    expected = m.head(z + h).transpose(1, 2)
    assert torch.allclose(m(x), expected, atol=1e-5)


def test_sensors_do_not_mix_without_graph():
    torch.manual_seed(0)
    m = Forecaster(N, "none", None, d=16)
    x = torch.randn(1, 8, N); x2 = x.clone(); x2[0, :, 7] += 10
    diff = (m(x) - m(x2)).abs().sum(1)[0]
    assert diff[7] > 0 and torch.allclose(diff[torch.arange(N) != 7], torch.zeros(N - 1))


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x[:, -4:, :]                                         # "predict" the last 4 inputs


def test_evaluate_matches_manual_masked_metrics():
    rng = np.random.default_rng(0)
    xs = [torch.tensor(rng.normal(size=(5, 8, N)), dtype=torch.float32) for _ in range(3)]
    ys = [torch.tensor(rng.normal(size=(5, 4, N)), dtype=torch.float32) * 50 + 300 for _ in range(3)]
    ms = [torch.tensor(rng.random((5, 4, N)) > 0.3) for _ in range(3)]
    mean = torch.full((N,), 300.0); std = torch.full((N,), 50.0)
    loader = [(x, y, m, torch.arange(5)) for x, y, m in zip(xs, ys, ms)]
    res = evaluate(_Identity(), loader, mean, std)
    P = np.concatenate([(x[:, -4:] * 50 + 300).numpy() for x in xs]); Y = np.concatenate([y.numpy() for y in ys])
    M = np.concatenate([m.numpy() for m in ms])
    e = (P - Y)[M]
    assert np.isclose(res["MAE"], np.abs(e).mean(), rtol=1e-6)
    assert np.isclose(res["RMSE"], np.sqrt((e ** 2).mean()), rtol=1e-6)
    for h in range(4):
        eh = (P - Y)[:, h][M[:, h]]
        assert np.isclose(res["MAE_h"][h], np.abs(eh).mean(), rtol=1e-6)
