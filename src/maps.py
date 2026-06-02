import numpy as np


WARMUP = 1000


def logistic(seed: float, n: int, mu: float = 3.99, dtype: type = np.float64) -> np.ndarray:
    mu = dtype(mu)
    x = dtype(seed)
    xs = np.empty(n + WARMUP, dtype=dtype)
    for i in range(n + WARMUP):
        x = mu * x * (dtype(1.0) - x)
        xs[i] = x
    return xs[WARMUP:]


def tent(seed: float, n: int, r: float = 1.99, dtype: type = np.float64) -> np.ndarray:
    r = dtype(r)
    half = dtype(0.5)
    x = dtype(seed)
    xs = np.empty(n + WARMUP, dtype=dtype)
    for i in range(n + WARMUP):
        x = r * x if x < half else r * (dtype(1.0) - x)
        xs[i] = x
    return xs[WARMUP:]


def chebyshev(seed: float, n: int, k: int = 3, dtype: type = np.float64) -> np.ndarray:
    x = dtype(seed)
    xs = np.empty(n + WARMUP, dtype=dtype)
    for i in range(n + WARMUP):
        x = dtype(4.0) * x ** 3 - dtype(3.0) * x
        xs[i] = x
    return xs[WARMUP:]


def sine(seed: float, n: int, r: float = 0.99, dtype: type = np.float64) -> np.ndarray:
    r = dtype(r)
    x = dtype(seed)
    xs = np.empty(n + WARMUP, dtype=dtype)
    for i in range(n + WARMUP):
        x = r * np.sin(dtype(np.pi) * x)
        xs[i] = x
    return xs[WARMUP:]


def henon(seed_x: float, seed_y: float, n: int, a: float = 1.4, b: float = 0.3, dtype: type = np.float64) -> np.ndarray:
    a = dtype(a)
    b = dtype(b)
    x = dtype(seed_x)
    y = dtype(seed_y)
    xs = np.empty(n + WARMUP, dtype=dtype)
    for i in range(n + WARMUP):
        xn = dtype(1.0) - a * x * x + y
        yn = b * x
        x, y = xn, yn
        xs[i] = x
    return xs[WARMUP:]
