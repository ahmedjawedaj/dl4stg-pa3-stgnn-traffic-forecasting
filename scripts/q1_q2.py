"""Q1 and Q2: numerical checks and figures.

Run: python scripts/q1_q2.py
Writes results/q1_q2.json and figures/q2a_heatmaps.png
"""
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}

# ---------------- Q1(a) ----------------
# Sum of affine messages: sum(w*x + c) = w*S + c*n. Any two neighbourhoods with
# the same (n, S) give the same aggregate for every w, c, so rho sees the same input.
nbr_fig = [+1, -1, 0, 0, -1]          # neighbourhood drawn in the handout
nbr_alt = [-1, -1, -1, +1, +1]
rng = np.random.default_rng(0)
same = True
for _ in range(1000):
    w, c = rng.normal(size=2)
    agg1 = sum(w * x + c for x in nbr_fig)
    agg2 = sum(w * x + c for x in nbr_alt)
    same &= bool(np.isclose(agg1, agg2))
out["q1a"] = {
    "A": nbr_fig, "B": nbr_alt,
    "n_S_A": [len(nbr_fig), sum(nbr_fig)], "n_S_B": [len(nbr_alt), sum(nbr_alt)],
    "steady_A": nbr_fig.count(0), "steady_B": nbr_alt.count(0),
    "aggregates_equal_for_1000_random_w_c": same,
}

# ---------------- Q1(b) ----------------
def phi_b(K):
    B = K + 1
    return {-1: 1, 0: B, +1: B * B}, B


def decode(M, B):
    return M % B, (M // B) % B, M // (B * B)   # (slowing, steady, speeding)


K = 8
phi, B = phi_b(K)
triples = [(a, s, b) for a, s, b in itertools.product(range(K + 1), repeat=3) if 1 <= a + s + b <= K]
sums = {}
ok_decode = True
for a, s, b in triples:
    M = a * phi[-1] + s * phi[0] + b * phi[+1]
    sums.setdefault(M, []).append((a, s, b))
    ok_decode &= decode(M, B) == (a, s, b)
collisions = [v for v in sums.values() if len(v) > 1]

# Base K is too small: a=K slowing and one steady neighbour collide.
phi_bad = {-1: 1, 0: K, +1: K * K}
bad = (K * phi_bad[-1], 1 * phi_bad[0])

# Non-steady count with phi(x) = x^2, any neighbourhood size up to 50.
ok_nonsteady = True
for n in range(1, 51):
    for _ in range(200):
        xs = rng.integers(-1, 2, size=n)
        ok_nonsteady &= int((xs ** 2).sum()) == int((xs != 0).sum())

out["q1b"] = {
    "K": K, "base": B, "phi": {"-1": phi[-1], "0": phi[0], "+1": phi[+1]},
    "num_triples_checked": len(triples), "num_distinct_sums": len(sums),
    "collisions": len(collisions), "all_decoded_correctly": ok_decode,
    "max_sum": max(sums),
    "base_K_counterexample": {"a=K,s=0,b=0": bad[0], "a=0,s=1,b=0": bad[1]},
    "nonsteady_phi_x_squared_ok": ok_nonsteady,
}

# ---------------- Q1(c) ----------------
# Brute force over every allowed neighbourhood (multiset of 1..8 trends) to confirm the table.
onehot = {-1: np.array([1, 0, 0]), 0: np.array([0, 1, 0]), +1: np.array([0, 0, 1])}
hoods = []
for n in range(1, 9):
    for combo in itertools.combinations_with_replacement([-1, 0, 1], n):
        hoods.append(combo)

tasks = {
    "i_count_steady": lambda h: h.count(0),
    "ii_majority_slowing": lambda h: h.count(-1) > len(h) / 2,
    "iii_any_slowing": lambda h: h.count(-1) > 0,
}
aggs = {
    "sum": lambda M: tuple(M.sum(0)),
    "mean": lambda M: tuple(np.round(M.mean(0), 10)),
    "max": lambda M: tuple(M.max(0)),
}
table = {}
for tname, tfun in tasks.items():
    for aname, afun in aggs.items():
        seen = {}
        witness = None
        for h in hoods:
            key = afun(np.stack([onehot[x] for x in h]))
            ans = tfun(h)
            if key in seen and seen[key][1] != ans and witness is None:
                witness = {"vector": [float(v) for v in key], "hood1": list(seen[key][0]),
                           "answer1": seen[key][1], "hood2": list(h), "answer2": ans}
            seen.setdefault(key, (h, ans))
        table[f"{tname}/{aname}"] = {"possible": witness is None, "witness": witness}
out["q1c"] = {"num_neighbourhoods": len(hoods), "table": table}

# ---------------- Q2 ----------------
N = 40


def true_step(x):
    return np.roll(x, -1)          # x_i(t+1) = x_{i+1}(t)


# (a)i least squares on the isolated slowdown, solved numerically as a check
x0 = np.zeros(N); x0[20] = 1
y = true_step(x0)
feats = np.stack([x0, np.roll(x0, 1) + np.roll(x0, -1)], 1)   # [x_i, x_{i-1}+x_{i+1}]
coef, *_ = np.linalg.lstsq(feats, y, rcond=None)
a_ls, b_ls = coef
sse = float(((feats @ coef - y) ** 2).sum())
out["q2a_i"] = {"a": float(a_ls), "b": float(b_ls), "min_sse": sse}

# (a)ii roll out fitted layer vs truth on a smooth bump
idx = np.arange(N)
bump = np.exp(-((idx - 20) ** 2) / 8)
pred = [bump.copy()]; true = [bump.copy()]
for _ in range(20):
    p = pred[-1]
    pred.append(a_ls * p + b_ls * (np.roll(p, 1) + np.roll(p, -1)))
    true.append(true_step(true[-1]))
pred = np.array(pred); true = np.array(true)
vmax = max(pred.max(), true.max())
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, M, title in [(axs[0], true, "True motion (np.roll(x, -1))"),
                     (axs[1], pred, f"Fitted symmetric layer (a={a_ls:.2f}, b={b_ls:.2f})")]:
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                   extent=[-0.5, N - 0.5, -0.5, 20.5])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("sensor index i")
    ax.set_yticks(range(0, 21, 5))
axs[0].set_ylabel("time step t")
fig.colorbar(im, ax=axs, shrink=0.9, label="slowdown")
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "figures", "q2a_heatmaps.png"), dpi=160, bbox_inches="tight")
plt.close(fig)
centre = lambda v: float((v * idx).sum() / v.sum())
out["q2a_ii"] = {
    "pred_peak_t0": float(pred[0].max()), "pred_peak_t20": float(pred[20].max()),
    "pred_centre_t20": centre(pred[20]), "pred_mass_t0": float(pred[0].sum()),
    "pred_mass_t20": float(pred[20].sum()), "true_peak_t20": float(true[20].max()),
    "true_argmax_t20": int(true[20].argmax()),
}

# (b) directional weights a=0, w-=0, w+=1, also as a circular conv1d
try:
    import torch
    conv = torch.nn.Conv1d(1, 1, 3, padding=1, padding_mode="circular", bias=False)
    with torch.no_grad():
        conv.weight[:] = torch.tensor([[[0.0, 0.0, 1.0]]])   # multiplies [x_{i-1}, x_i, x_{i+1}]
    have_torch = True
except ImportError:
    have_torch = False
ok_b = True; ok_conv = True
for _ in range(200):
    x = rng.normal(size=N)
    pred_b = 0 * x + 0 * np.roll(x, 1) + 1 * np.roll(x, -1)
    ok_b &= np.allclose(pred_b, true_step(x))
    if have_torch:
        yc = conv(torch.tensor(x, dtype=torch.float32)[None, None]).detach().numpy()[0, 0]
        ok_conv &= np.allclose(yc, true_step(x), atol=1e-5)
out["q2b"] = {"a": 0, "w_minus": 0, "w_plus": 1, "exact_on_200_random": bool(ok_b),
              "conv1d_kernel": [0, 0, 1], "padding": "circular",
              "conv1d_exact": bool(ok_conv) if have_torch else None}

# (c) history: a=0, b=1, c=-1
a_c, b_c, c_c = 0.0, 1.0, -1.0
ok_back = True; ok_fwd = True
for _ in range(200):
    xp = rng.normal(size=N)
    # backward motion
    xt = np.roll(xp, -1); xn = np.roll(xt, -1)
    pr = a_c * xt + b_c * (np.roll(xt, 1) + np.roll(xt, -1)) + c_c * xp
    ok_back &= np.allclose(pr, xn)
    # forward motion x_i(t+1) = x_{i-1}(t)
    xt = np.roll(xp, 1); xn = np.roll(xt, 1)
    pr = a_c * xt + b_c * (np.roll(xt, 1) + np.roll(xt, -1)) + c_c * xp
    ok_fwd &= np.allclose(pr, xn)
out["q2c"] = {"a": a_c, "b": b_c, "c": c_c, "exact_backward": bool(ok_back),
              "exact_forward_same_coeffs": bool(ok_fwd)}

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "q1_q2.json"), "w") as f:
    json.dump(out, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
