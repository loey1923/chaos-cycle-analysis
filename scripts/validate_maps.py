import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.maps import logistic, tent, chebyshev, sine, henon, WARMUP

FIGS = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGS, exist_ok=True)

N = 10000
SEED = 0.3
SEED_HENON = (-0.5, 0.2)

maps = [
    ("Logistic", logistic, SEED, {"mu": 3.99}),
    ("Tent", tent, SEED, {"r": 1.99}),
    ("Chebyshev", chebyshev, 0.2, {}),
    ("Sine", sine, SEED, {"r": 0.99}),
    ("Henon", henon, SEED_HENON, {"a": 1.4, "b": 0.3}),
]

for name, func, seed, params in maps:
    print(f"Generating orbit for {name} ...")
    if func is henon:
        sx, sy = seed
        seq = func(sx, sy, N, **params)
    else:
        seq = func(seed, N, **params)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 1]})

    ax = axes[0]
    ax.plot(seq[:500], linewidth=0.5, color="steelblue")
    ax.set_title(f"{name} — First 500 iterations")
    ax.set_xlabel("n")
    ax.set_ylabel("x_n")

    ax = axes[1]
    ax.plot(seq[-500:], linewidth=0.5, color="steelblue")
    ax.set_title(f"{name} — Last 500 of {N} iterations")
    ax.set_xlabel("n")
    ax.set_ylabel("x_n")

    fig.tight_layout()
    path = os.path.join(FIGS, f"validation_orbit_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

print("=== Henon x-distribution histogram ===")
sx, sy = SEED_HENON
seq = henon(sx, sy, N)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(seq, bins=80, density=True, color="steelblue", edgecolor="none")
ax.set_title("Henon — x-component distribution (N=10000)")
ax.set_xlabel("x")
ax.set_ylabel("Density")
fig.tight_layout()
path = os.path.join(FIGS, "validation_henon_histogram.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  Saved {path}")

print("=== Sine extra confirmation: orbit + histogram ===")
seq = sine(SEED, N)
fig, axes = plt.subplots(2, 2, figsize=(12, 6))

axes[0, 0].plot(seq[:500], linewidth=0.5, color="coral")
axes[0, 0].set_title("Sine — First 500")
axes[0, 0].set_xlabel("n")
axes[0, 0].set_ylabel("x_n")

axes[0, 1].plot(seq[-500:], linewidth=0.5, color="coral")
axes[0, 1].set_title(f"Sine — Last 500 of {N}")
axes[0, 1].set_xlabel("n")
axes[0, 1].set_ylabel("x_n")

axes[1, 0].hist(seq, bins=80, density=True, color="coral", edgecolor="none")
axes[1, 0].set_title("Sine — Distribution")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("Density")

axes[1, 1].plot(seq[:-1], seq[1:], ",", alpha=0.3, color="coral", markersize=0.3)
axes[1, 1].set_title("Sine — Return map")
axes[1, 1].set_xlabel("x_n")
axes[1, 1].set_ylabel("x_{n+1}")
axes[1, 1].set_aspect("equal")

fig.tight_layout()
path = os.path.join(FIGS, "validation_sine_extra.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  Saved {path}")

print("Done. All validation figures generated.")
