import numpy as np


WARMUP = 1000


def logistic(seed: float, n: int, mu: float = 3.99) -> np.ndarray:
    xs = np.empty(n + WARMUP)
    x = seed
    for i in range(n + WARMUP):
        x = mu * x * (1.0 - x)
        xs[i] = x
    return xs[WARMUP:]


def tent(seed: float, n: int, r: float = 1.99) -> np.ndarray:
    xs = np.empty(n + WARMUP)
    x = seed
    for i in range(n + WARMUP):
        x = r * x if x < 0.5 else r * (1.0 - x)
        xs[i] = x
    return xs[WARMUP:]


def chebyshev(seed: float, n: int, k: int = 3) -> np.ndarray:
    xs = np.empty(n + WARMUP)
    x = seed
    for i in range(n + WARMUP):
        x = 4.0 * x ** 3 - 3.0 * x
        xs[i] = x
    return xs[WARMUP:]


def cubic(seed: float, n: int, r: float = 0.99) -> np.ndarray:
    xs = np.empty(n + WARMUP)
    x = seed
    for i in range(n + WARMUP):
        x = r * np.sin(np.pi * x)
        xs[i] = x
    return xs[WARMUP:]


def henon(seed_x: float, seed_y: float, n: int, a: float = 1.4, b: float = 0.3) -> np.ndarray:
    xs = np.empty(n + WARMUP)
    x, y = seed_x, seed_y
    for i in range(n + WARMUP):
        xn = 1.0 - a * x * x + y
        yn = b * x
        x, y = xn, yn
        xs[i] = x
    return xs[WARMUP:]
