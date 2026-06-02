import math
import numpy as np


def generate_permutation(seq: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(seq)):
        raise ValueError("Sequence contains NaN or Inf; map may have diverged")
    return np.argsort(seq)


def cycle_decomposition(perm: np.ndarray) -> dict[int, int]:
    n = len(perm)
    visited = np.zeros(n, dtype=bool)
    counts: dict[int, int] = {}
    for i in range(n):
        if visited[i]:
            continue
        length = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            length += 1
        counts[length] = counts.get(length, 0) + 1
    return counts


def permutation_order(cycle_lengths: dict[int, int]) -> int:
    order = 1
    for length in cycle_lengths:
        order = math.lcm(order, length)
    return order
