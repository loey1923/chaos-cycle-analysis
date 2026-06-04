import math
import random
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
from concurrent.futures import ProcessPoolExecutor

from src.maps import logistic, tent, chebyshev, sine, henon, WARMUP
from src.permutation import generate_permutation, cycle_decomposition, permutation_order


def _single_run(args):
    map_func, params, N, seed_pair = args

    if map_func is henon:
        sx, sy = seed_pair
        seq = map_func(sx, sy, N, **params)
    else:
        seq = map_func(seed_pair[0], N, **params)

    if not np.all(np.isfinite(seq)):
        return None

    perm = generate_permutation(seq)
    cycles = cycle_decomposition(perm)
    order = permutation_order(cycles)
    ln_order = math.log(order) if order > 0 else 0.0

    max_cycle_len = max(cycles.keys(), default=0)
    max_cycle_ratio = max_cycle_len / N if N > 0 else 0.0
    total_cycles = sum(cycles.values())

    fixed_point_count = cycles.get(1, 0)
    fixed_point_ratio = fixed_point_count / N if N > 0 else 0.0
    short_count = sum(c for length, c in cycles.items() if length <= 3)
    short_cycle_ratio = short_count / N if N > 0 else 0.0

    return ln_order, max_cycle_ratio, total_cycles, fixed_point_ratio, short_cycle_ratio


def _make_seed(map_func):
    if map_func is chebyshev:
        return random.random() * 2.0 - 1.0, 0
    elif map_func is henon:
        return (random.uniform(-1.5, 1.5), random.uniform(-0.5, 0.5)), None
    else:
        return random.random(), 0


def batch_avg_ln_order(map_func, params, N_list, num_seeds, warmup=1000):
    records = []
    for N in N_list:
        valid = []
        attempts = 0
        max_attempts = num_seeds * 5
        while len(valid) < num_seeds and attempts < max_attempts:
            batch_size = num_seeds - len(valid)
            seeds = []
            for _ in range(batch_size):
                seed_val, _ = _make_seed(map_func)
                if map_func is henon:
                    seeds.append(seed_val)
                else:
                    seeds.append((seed_val, 0))
            args_list = [(map_func, params, N, s) for s in seeds]
            with ProcessPoolExecutor() as ex:
                batch_results = list(ex.map(_single_run, args_list))
            valid.extend(r for r in batch_results if r is not None)
            attempts += batch_size

        if len(valid) < num_seeds:
            raise RuntimeError(f"Only got {len(valid)}/{num_seeds} valid samples for N={N} after {max_attempts} attempts")

        valid = valid[:num_seeds]
        ln_orders = [r[0] for r in valid]
        max_ratios = [r[1] for r in valid]
        cycle_counts = [r[2] for r in valid]
        fixed_ratios = [r[3] for r in valid]
        short_ratios = [r[4] for r in valid]

        records.append({
            "N": N,
            "avg_ln_order": float(np.mean(ln_orders)),
            "std_ln_order": float(np.std(ln_orders)),
            "avg_max_cycle_ratio": float(np.mean(max_ratios)),
            "avg_cycle_count": float(np.mean(cycle_counts)),
            "avg_fixed_point_ratio": float(np.mean(fixed_ratios)),
            "avg_short_cycle_ratio": float(np.mean(short_ratios)),
        })
    return pd.DataFrame(records)


def logistic_param_scan(N=1024, num_seeds=30, mu_min=3.6, mu_max=4.0, mu_step=0.01):
    records = []
    mu_vals = np.arange(mu_min, mu_max + mu_step / 2, mu_step)
    for mu in mu_vals:
        params = {"mu": mu}
        df = batch_avg_ln_order(logistic, params, [N], num_seeds)
        row = df.iloc[0]
        records.append({
            "mu": round(mu, 4),
            "avg_ln_order": row["avg_ln_order"],
            "std_ln_order": row["std_ln_order"],
            "avg_max_cycle_ratio": row["avg_max_cycle_ratio"],
            "avg_cycle_count": row["avg_cycle_count"],
        })
    return pd.DataFrame(records)


def landau_ln_order(N: np.ndarray) -> np.ndarray:
    return np.sqrt(N * np.log(N) / 2.0)


def golomb_dickman_max_cycle(N: np.ndarray) -> np.ndarray:
    # Golomb-Dickman constant λ ≈ 0.6243.
    # Reference: Shepp L A, Lloyd S P. Ordered cycle lengths in a
    # random permutation. Transactions of the AMS, 1966.
    return 0.6243 * N


def seed_avalanche(map_func, params, N, seed, eps=1e-14):
    if map_func is henon:
        sx, sy = seed
        seq1 = map_func(sx, sy, N, **params)
        seq2 = map_func(sx + eps, sy, N, **params)
    else:
        seq1 = map_func(seed, N, **params)
        seq2 = map_func(seed + eps, N, **params)

    perm1 = generate_permutation(seq1)
    perm2 = generate_permutation(seq2)

    n = len(perm1)
    diff_count = np.sum(perm1 != perm2)
    diff_rate = diff_count / n

    footrule = np.sum(np.abs(perm1 - perm2)) / (n * n / 2.0)
    return {"diff_rate": float(diff_rate), "footrule": float(footrule)}


def _precision_single(args):
    map_func, params, N, seed, precision = args

    if "float32" in precision:
        dtype = np.float32
    elif "float64" in precision:
        dtype = np.float64
    else:
        dtype = np.float64

    if "Decimal" in precision:
        prec = int(precision.replace("Decimal(", "").replace(")", ""))
        getcontext().prec = prec
        x = Decimal(str(seed))
        n_total = N + WARMUP
        xs = []
        if map_func is logistic:
            mu = Decimal(str(params.get("mu", 3.99)))
            for _ in range(n_total):
                x = mu * x * (1 - x)
                xs.append(float(x))
        elif map_func is tent:
            r = Decimal(str(params.get("r", 1.99)))
            for _ in range(n_total):
                half = Decimal("0.5")
                x = r * x if x < half else r * (1 - x)
                xs.append(float(x))
        elif map_func is sine:
            r = Decimal(str(params.get("r", 0.99)))
            for _ in range(n_total):
                xf = float(x)
                sin_val = math.sin(xf * math.pi)
                x = r * Decimal(str(sin_val))
                xs.append(float(x))
        else:
            raise ValueError(f"Decimal unsupported for {map_func}")
        seq = np.array(xs[WARMUP:], dtype=float)
    elif map_func is henon:
        seq = map_func(seed[0], seed[1], N, **params, dtype=dtype)
    else:
        seq = map_func(dtype(seed), N, **params, dtype=dtype)

    perm = generate_permutation(seq)
    cycles = cycle_decomposition(perm)
    order = permutation_order(cycles)
    ln_order = math.log(order) if order > 0 else 0.0
    max_cycle_len = max(cycles.keys(), default=0)
    cycle_count = sum(cycles.values())

    return {
        "precision": precision,
        "order": order,
        "ln_order": ln_order,
        "cycle_counts": cycle_count,
        "max_cycle_len": max_cycle_len,
    }


def precision_comparison(map_func, base_params, N, seed):
    result = {}
    for precision in ["float32", "float64", "Decimal(50)"]:
        args = (map_func, base_params, N, seed, precision)
        result[precision] = _precision_single(args)
    return result


def sorting_bias_experiment(N, num_seeds):
    records = []
    map_configs = [
        ("Logistic", logistic, {"mu": 3.99}),
        ("Tent", tent, {"r": 1.99}),
        ("Chebyshev", chebyshev, {}),
        ("Sine", sine, {"r": 0.99}),
        ("Henon", henon, {"a": 1.4, "b": 0.3}),
    ]

    for name, map_func, params in map_configs:
        collected = 0
        attempts = 0
        while collected < num_seeds and attempts < num_seeds * 10:
            attempts += 1
            if map_func is henon:
                sx, sy = random.uniform(-1.5, 1.5), random.uniform(-0.5, 0.5)
                seq = map_func(sx, sy, N, **params)
            else:
                seed = random.random()
                if map_func is chebyshev:
                    seed = random.uniform(-1, 1)
                seq = map_func(seed, N, **params)
            if not np.all(np.isfinite(seq)):
                continue
            collected += 1
            perm = generate_permutation(seq)
            cycles = cycle_decomposition(perm)
            order = permutation_order(cycles)
            ln_order = math.log(order) if order > 0 else 0.0
            max_ratio = max(cycles.keys(), default=0) / N
            total_cycles = sum(cycles.values())
            records.append({
                "group": f"Chaos-{name}", "N": N,
                "ln_order": ln_order, "max_cycle_ratio": max_ratio,
                "cycle_count": total_cycles,
            })

    for _ in range(num_seeds):
        seq = np.array([random.random() for _ in range(N)])
        perm = generate_permutation(seq)
        cycles = cycle_decomposition(perm)
        order = permutation_order(cycles)
        ln_order = math.log(order) if order > 0 else 0.0
        max_ratio = max(cycles.keys(), default=0) / N
        total_cycles = sum(cycles.values())
        records.append({
            "group": "Random+argsort", "N": N,
            "ln_order": ln_order, "max_cycle_ratio": max_ratio,
            "cycle_count": total_cycles,
        })

    for _ in range(num_seeds):
        arr = list(range(N))
        for i in range(N - 1, 0, -1):
            j = random.randint(0, i)
            arr[i], arr[j] = arr[j], arr[i]
        perm = np.array(arr)
        cycles = cycle_decomposition(perm)
        order = permutation_order(cycles)
        ln_order = math.log(order) if order > 0 else 0.0
        max_ratio = max(cycles.keys(), default=0) / N
        total_cycles = sum(cycles.values())
        records.append({
            "group": "Fisher-Yates", "N": N,
            "ln_order": ln_order, "max_cycle_ratio": max_ratio,
            "cycle_count": total_cycles,
        })

    return pd.DataFrame(records)
