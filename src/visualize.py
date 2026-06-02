import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def fig1_avg_ln_order(results: dict, landau_ref, fisher_yates_ref, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, df in results.items():
        ax.plot(df["N"], df["avg_ln_order"], marker="o", label=name)
    ax.plot(landau_ref["N"], landau_ref["ln_order"], "k--", label="Landau (ref)")
    ax.plot(fisher_yates_ref["N"], fisher_yates_ref["avg_ln_order"], "k:", label="Fisher-Yates")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("E[ln(order)]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def fig2_cycle_distribution(all_perms: dict, N: int, output_path: str):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3), sharey=True)
    for ax, (name, perm) in zip(axes, all_perms.items()):
        cycles = {}
        visited = np.zeros(N, dtype=bool)
        for i in range(N):
            if visited[i]:
                continue
            length = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                length += 1
            cycles[length] = cycles.get(length, 0) + 1
        lengths = list(cycles.keys())
        counts = list(cycles.values())
        ax.bar(lengths, counts, width=0.8)
        ax.set_title(name)
        ax.set_xlabel("Cycle length")
    axes[0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def fig3_seed_avalanche(avalanche_df: pd.DataFrame, output_path: str):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(avalanche_df))
    ax1.bar(x - 0.2, avalanche_df["diff_rate"], 0.4, label="DiffRate", color="steelblue")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, avalanche_df["footrule"], 0.4, label="Footrule", color="coral")
    ax1.set_xticks(x)
    ax1.set_xticklabels(avalanche_df["mapping"], rotation=30)
    ax1.set_ylabel("DiffRate")
    ax2.set_ylabel("Footrule")
    fig.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def fig4_precision_degradation(precision_df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=precision_df, x="mapping", y="ln_order", hue="precision", ax=ax)
    ax.set_xlabel("Mapping")
    ax.set_ylabel("ln(order)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def fig5_sorting_bias(sorting_df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [("ln_order", "ln(order)"), ("max_cycle_ratio", "Max cycle ratio"), ("cycle_count", "Cycle count")]
    for ax, (col, label) in zip(axes, metrics):
        sns.boxplot(data=sorting_df, x="group", y=col, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def fig6_summary_metrics(safety_df: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    pivot = safety_df.pivot_table(index="mapping", columns="metric", values="value")
    sns.heatmap(pivot, annot=True, cmap="RdYlGn_r", center=0.5, ax=ax)
    ax.set_title("Comprehensive safety metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
