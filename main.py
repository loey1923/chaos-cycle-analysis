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
    logistic_param_scan,
)
from src.visualize import (
    fig1_avg_ln_order, fig2_cycle_distribution, fig3_seed_avalanche,
    fig4_precision_degradation, fig5_sorting_bias, fig6_summary_metrics,
    fig_param_scan_logistic,
)

DATA_DIR = "data"
FIGURES_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

MAP_CONFIGS = {
    "Logistic": (logistic, {"mu": 3.99}),
    "Tent": (tent, {"r": 1.99}),
    "Chebyshev": (chebyshev, {}),
    "Sine": (sine, {"r": 0.99}),
    "Henon": (henon, {"a": 1.4, "b": 0.3}),
}

N_LIST = [64, 128, 256, 512, 1024, 2048, 4096]
NUM_SEEDS = 100


def main():
    random.seed(42)
    np.random.seed(42)

    print("[1/9] Running batch_avg_ln_order ...")
    results = {}
    all_records = []
    for name, (func, params) in MAP_CONFIGS.items():
        df = batch_avg_ln_order(func, params, N_LIST, NUM_SEEDS)
        results[name] = df
        df.insert(0, "mapping", name)
        all_records.append(df)

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
        fy_records.append({
            "N": N,
            "avg_ln_order": float(np.mean(orders)),
            "std_ln_order": float(np.std(orders)),
        })

    fy_df = pd.DataFrame(fy_records)
    fy_df.insert(0, "mapping", "Fisher-Yates")
    all_records.append(fy_df)
    pd.concat(all_records, ignore_index=True).to_csv(
        os.path.join(DATA_DIR, "avg_ln_order.csv"), index=False
    )

    landau_ref = pd.DataFrame({
        "N": N_LIST,
        "ln_order": landau_ln_order(np.array(N_LIST, dtype=float)),
    })

    fisher_yates_ref = fy_df

    print("[2/9] Generating fig1 (avg ln order) ...")
    fig1_avg_ln_order(results, landau_ref, fisher_yates_ref,
                      os.path.join(FIGURES_DIR, "fig1_avg_ln_order.png"))

    print("[3/9] Generating fig2 (cycle distribution) ...")
    all_perms = {}
    for name, (func, params) in MAP_CONFIGS.items():
        if func is henon:
            seq = func(-0.5, 0.2, 1024, **params)
        elif func is chebyshev:
            seq = func(0.2, 1024, **params)
        else:
            seq = func(0.3, 1024, **params)
        perm = generate_permutation(seq)
        all_perms[name] = perm
    fig2_cycle_distribution(all_perms, 1024,
                            os.path.join(FIGURES_DIR, "fig2_cycle_distribution.png"))

    print("[4/9] Running seed_avalanche ...")
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

    print("[5/9] Running precision comparison ...")
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

    print("[6/9] Running sorting bias experiment ...")
    sorting_df = sorting_bias_experiment(1024, 60)
    sorting_df.to_csv(os.path.join(DATA_DIR, "sorting_bias.csv"), index=False)
    fig5_sorting_bias(sorting_df,
                      os.path.join(FIGURES_DIR, "fig5_sorting_bias.png"))

    print("[7/9] Running logistic parameter scan (E4) ...")
    param_scan_df = logistic_param_scan(N=1024, num_seeds=30)
    param_scan_df.to_csv(os.path.join(DATA_DIR, "param_scan_logistic.csv"), index=False)
    fig_param_scan_logistic(param_scan_df,
                            os.path.join(FIGURES_DIR, "fig_param_scan_logistic.png"))

    print("[8/9] Generating fig6 (summary metrics) ...")
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

        sub_prec = precision_df[(precision_df["mapping"] == name) & (precision_df["N"] == 256)]
        if not sub_prec.empty:
            f32 = sub_prec[sub_prec["precision"] == "float32"]["ln_order"].mean()
            dec = sub_prec[sub_prec["precision"] == "Decimal(50)"]["ln_order"].mean()
            deg = (dec - f32) / dec if dec > 0 else 0
            safety_records.append({"mapping": name, "metric": "Q4_Degradation", "value": deg})

        sub_sort = sorting_df[sorting_df["group"] == f"Chaos-{name}"]
        fy_row = sorting_df[sorting_df["group"] == "Fisher-Yates"]
        if not sub_sort.empty and not fy_row.empty:
            fy_ln = fy_row["ln_order"].mean()
            chaos_ln = sub_sort["ln_order"].mean()
            q3 = chaos_ln / fy_ln if fy_ln > 0 else 1
            safety_records.append({"mapping": name, "metric": "Q3_SortBias", "value": q3})

            lr = sub_sort["max_cycle_ratio"].mean() / fy_row["max_cycle_ratio"].mean() if fy_row["max_cycle_ratio"].mean() > 0 else 1
            safety_records.append({"mapping": name, "metric": "Q8_MaxCycleRatio", "value": lr})
            cc = sub_sort["cycle_count"].mean() / math.log(1024)
            safety_records.append({"mapping": name, "metric": "Q9_CycleCount_norm", "value": cc})

        row = results[name].set_index("N")
        if 1024 in row.index:
            safety_records.append({"mapping": name, "metric": "Q6_ShortCycle", "value": row.loc[1024, "avg_short_cycle_ratio"]})
            safety_records.append({"mapping": name, "metric": "Q7_FixedPoint", "value": row.loc[1024, "avg_fixed_point_ratio"]})

    safety_df = pd.DataFrame(safety_records)
    fig6_summary_metrics(safety_df,
                         os.path.join(FIGURES_DIR, "fig6_summary_metrics.png"))

    print("[9/9] Done. All figures saved to figures/, data to data/.")


if __name__ == "__main__":
    main()
