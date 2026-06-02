import os
import math
import random
import numpy as np
import pandas as pd

from src.maps import logistic, tent, chebyshev, sine, henon
from src.permutation import generate_permutation, cycle_decomposition, permutation_order
from src.analysis import (
    batch_avg_ln_order, landau_ln_order, golomb_dickman_max_cycle,
    seed_avalanche, precision_comparison, sorting_bias_experiment,
)
from src.visualize import (
    fig1_avg_ln_order, fig2_cycle_distribution, fig3_seed_avalanche,
    fig4_precision_degradation, fig5_sorting_bias, fig6_summary_metrics,
)

DATA_DIR = "data"
FIGURES_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MAP_CONFIGS = {
    "Logistic": (logistic, {"mu": 3.99}),
    "Tent": (tent, {"r": 1.99}),
    "Chebyshev": (chebyshev, {"k": 3}),
    "Sine": (sine, {"r": 0.99}),
    "Henon": (henon, {"a": 1.4, "b": 0.3}),
}

N_LIST = [64, 128, 256, 512, 1024, 2048, 4096]
NUM_SEEDS = 100


def main():
    random.seed(42)
    np.random.seed(42)

    print("[1/8] Running batch_avg_ln_order ...")
    results = {}
    all_records = []
    for name, (func, params) in MAP_CONFIGS.items():
        df = batch_avg_ln_order(func, params, N_LIST, NUM_SEEDS)
        results[name] = df
        df.insert(0, "mapping", name)
        all_records.append(df)
    pd.concat(all_records, ignore_index=True).to_csv(
        os.path.join(DATA_DIR, "avg_ln_order.csv"), index=False
    )

    landau_ref = pd.DataFrame({
        "N": N_LIST,
        "ln_order": landau_ln_order(np.array(N_LIST, dtype=float)),
    })

    fy_records = []
    for N in N_LIST:
        orders = []
        for _ in range(NUM_SEEDS):
            arr = list(range(N))
            for i in range(N - 1, 0, -1):
                j = random.randint(0, i)
                arr[i], arr[j] = arr[j], arr[i]
            perm = np.array(arr)
            cycles = cycle_decomposition(perm)
            order = permutation_order(cycles)
            orders.append(math.log(order) if order > 0 else 0.0)
        fy_records.append({"N": N, "avg_ln_order": np.mean(orders)})
    fisher_yates_ref = pd.DataFrame(fy_records)

    print("[2/8] Generating fig1 (avg ln order) ...")
    fig1_avg_ln_order(results, landau_ref, fisher_yates_ref,
                      os.path.join(FIGURES_DIR, "fig1_avg_ln_order.png"))

    print("[3/8] Generating fig2 (cycle distribution) ...")
    all_perms = {}
    for name, (func, params) in MAP_CONFIGS.items():
        if func is henon:
            seq = func(-0.5, 0.2, 1024, **params)
        elif func is chebyshev:
            seq = func(0.5, 1024, **params)
        else:
            seq = func(0.3, 1024, **params)
        perm = generate_permutation(seq)
        all_perms[name] = perm
    fig2_cycle_distribution(all_perms, 1024,
                            os.path.join(FIGURES_DIR, "fig2_cycle_distribution.png"))

    print("[4/8] Running seed_avalanche ...")
    avalanche_records = []
    for name, (func, params) in MAP_CONFIGS.items():
        seed = (-0.5, 0.2) if func is henon else (0.3 if func is chebyshev else 0.3)
        result = seed_avalanche(func, params, 1024, seed)
        result["mapping"] = name
        avalanche_records.append(result)
    avalanche_df = pd.DataFrame(avalanche_records)
    avalanche_df.to_csv(os.path.join(DATA_DIR, "avalanche.csv"), index=False)
    fig3_seed_avalanche(avalanche_df,
                        os.path.join(FIGURES_DIR, "fig3_seed_avalanche.png"))

    print("[5/8] Running precision comparison ...")
    precision_records = []
    for name in ["Logistic", "Tent", "Sine"]:
        func, params = MAP_CONFIGS[name]
        for N in [256, 512, 1024]:
            for _ in range(30):
                seed = random.random()
                result = precision_comparison(func, params, N, seed)
                for prec, data in result.items():
                    precision_records.append({
                        "mapping": name, "N": N, "precision": prec,
                        "ln_order": data["ln_order"],
                        "cycle_counts": data["cycle_counts"],
                        "max_cycle_len": data["max_cycle_len"],
                    })
    precision_df = pd.DataFrame(precision_records)
    precision_df.to_csv(os.path.join(DATA_DIR, "precision.csv"), index=False)
    fig4_precision_degradation(precision_df,
                               os.path.join(FIGURES_DIR, "fig4_precision_degradation.png"))

    print("[6/8] Running sorting bias experiment ...")
    sorting_df = sorting_bias_experiment(1024, 60)
    sorting_df.to_csv(os.path.join(DATA_DIR, "sorting_bias.csv"), index=False)
    fig5_sorting_bias(sorting_df,
                      os.path.join(FIGURES_DIR, "fig5_sorting_bias.png"))

    print("[7/8] Generating fig6 (summary metrics) ...")
    safety_records = []
    for name in MAP_CONFIGS:
        avg_ln = results[name].set_index("N")["avg_ln_order"]
        landau_val = landau_ln_order(np.array([2048], dtype=float))[0]
        ratio = avg_ln.get(2048, 0) / landau_val if landau_val > 0 else 0
        safety_records.append({"mapping": name, "metric": "Q1_LandauRatio", "value": ratio})

        av_row = avalanche_df[avalanche_df["mapping"] == name]
        if not av_row.empty:
            safety_records.append({"mapping": name, "metric": "Q2_DiffRate", "value": av_row.iloc[0]["diff_rate"]})
            safety_records.append({"mapping": name, "metric": "Q2_Footrule", "value": av_row.iloc[0]["footrule"]})

    for name in MAP_CONFIGS:
        sub = precision_df[(precision_df["mapping"] == name) & (precision_df["N"] == 256)]
        if not sub.empty:
            f32 = sub[sub["precision"] == "float32"]["ln_order"].mean()
            dec = sub[sub["precision"] == "Decimal(50)"]["ln_order"].mean()
            deg = (dec - f32) / dec if dec > 0 else 0
            safety_records.append({"mapping": name, "metric": "Q4_Degradation", "value": deg})

    for name in MAP_CONFIGS:
        sub = sorting_df[sorting_df["group"] == f"Chaos-{name}"]
        fy = sorting_df[sorting_df["group"] == "Fisher-Yates"]
        if not sub.empty and not fy.empty:
            lr = sub["max_cycle_ratio"].mean() / fy["max_cycle_ratio"].mean() if fy["max_cycle_ratio"].mean() > 0 else 1
            safety_records.append({"mapping": name, "metric": "Q8_MaxCycleRatio", "value": lr})
            cc = sub["cycle_count"].mean() / math.log(1024)
            safety_records.append({"mapping": name, "metric": "Q9_CycleCount_norm", "value": cc})

    safety_df = pd.DataFrame(safety_records)
    fig6_summary_metrics(safety_df,
                         os.path.join(FIGURES_DIR, "fig6_summary_metrics.png"))

    print("[8/8] Done. All figures saved to figures/, data to data/.")


if __name__ == "__main__":
    main()
