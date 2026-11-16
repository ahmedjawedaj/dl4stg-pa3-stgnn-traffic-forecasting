"""Independent checks of the Q1 and Q2 answers (does not reuse scripts/q1_q2.py)."""
import itertools

import numpy as np
import torch


def test_q1a_equal_n_and_S_collide():
    A, B = [+1, -1, 0, 0, -1], [-1, -1, -1, +1, +1]
    assert (len(A), sum(A)) == (len(B), sum(B)) and A.count(0) != B.count(0)
    for w, c in np.random.default_rng(1).normal(size=(100, 2)):
        assert np.isclose(sum(w * x + c for x in A), sum(w * x + c for x in B))


def test_q1b_base_K_plus_1_is_uniquely_decodable_for_every_K():
    for K in range(1, 13):
        B = K + 1
        phi = {-1: 1, 0: B, 1: B * B}
        seen = {}
        for a, s, b in itertools.product(range(K + 1), repeat=3):
            if 1 <= a + s + b <= K:
                M = a * phi[-1] + s * phi[0] + b * phi[1]
                assert (M % B, (M // B) % B, M // B ** 2) == (a, s, b)
                assert M not in seen
                seen[M] = (a, s, b)


def test_q1b_base_K_collides():
    K = 8
    phi = {-1: 1, 0: K, 1: K * K}
    assert K * phi[-1] == 1 * phi[0]


def test_q1b_nonsteady_count():
    rng = np.random.default_rng(2)
    for n in range(1, 30):
        x = rng.integers(-1, 2, n)
        assert (x ** 2).sum() == (x != 0).sum()


def test_q1c_table():
    onehot = {-1: (1, 0, 0), 0: (0, 1, 0), 1: (0, 0, 1)}
    hoods = [h for n in range(1, 9) for h in itertools.combinations_with_replacement([-1, 0, 1], n)]
    assert len(hoods) == 164
    aggs = {"sum": lambda M: tuple(M.sum(0)), "mean": lambda M: tuple(np.round(M.mean(0), 9)),
            "max": lambda M: tuple(M.max(0))}
    tasks = {"i": lambda h: h.count(0), "ii": lambda h: h.count(-1) > len(h) / 2, "iii": lambda h: -1 in h}
    expected = {("i", "sum"): True, ("i", "mean"): False, ("i", "max"): False,
                ("ii", "sum"): True, ("ii", "mean"): True, ("ii", "max"): False,
                ("iii", "sum"): True, ("iii", "mean"): True, ("iii", "max"): True}
    for (t, a), possible in expected.items():
        answers = {}
        ok = True
        for h in hoods:
            key = aggs[a](np.array([onehot[x] for x in h]))
            if key in answers and answers[key] != tasks[t](h):
                ok = False
            answers.setdefault(key, tasks[t](h))
        assert ok == possible, (t, a)
    # the counterexamples quoted in the report
    mx = lambda h: tuple(np.array([onehot[x] for x in h]).max(0))
    mean = lambda h: tuple(np.array([onehot[x] for x in h]).mean(0))
    assert mean((0,)) == mean((0, 0)) and mx((0,)) == mx((0, 0))
    assert mx((-1, 0)) == mx((-1, -1, 0)) == (1, 1, 0)


def test_q2a_least_squares():
    x = np.zeros(40)
    x[20] = 1
    y = np.roll(x, -1)
    F = np.stack([x, np.roll(x, 1) + np.roll(x, -1)], 1)
    (a, b), *_ = np.linalg.lstsq(F, y, rcond=None)
    assert np.isclose(a, 0) and np.isclose(b, 0.5)
    assert np.isclose(((F @ [a, b] - y) ** 2).sum(), 0.5)


def test_q2a_rollout_does_not_move():
    i = np.arange(40)
    p = np.exp(-((i - 20) ** 2) / 8)
    m0 = p.sum()
    for _ in range(20):
        p = 0.5 * (np.roll(p, 1) + np.roll(p, -1))
    assert abs((p * i).sum() / p.sum() - 20) < 1e-3 and np.isclose(p.sum(), m0)   # tiny tail wraps the ring


def test_q2b_conv_kernel():
    conv = torch.nn.Conv1d(1, 1, 3, padding=1, padding_mode="circular", bias=False)
    with torch.no_grad():
        conv.weight[:] = torch.tensor([[[0.0, 0.0, 1.0]]])
    x = np.random.default_rng(3).normal(size=40)
    out = conv(torch.tensor(x, dtype=torch.float32)[None, None]).detach().numpy()[0, 0]
    assert np.allclose(out, np.roll(x, -1), atol=1e-6)


def test_q2c_history_both_directions():
    rng = np.random.default_rng(4)
    f = lambda xt, xp: np.roll(xt, 1) + np.roll(xt, -1) - xp      # a=0, b=1, c=-1
    for _ in range(50):
        xp = rng.normal(size=40)
        xt = np.roll(xp, -1)
        assert np.allclose(f(xt, xp), np.roll(xt, -1))   # backward
        xt = np.roll(xp, 1)
        assert np.allclose(f(xt, xp), np.roll(xt, 1))     # forward
