import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def flow_plot():
    df = pd.DataFrame({
        "Mix": ["N-0-M1", "N-0-M2", "SH-2-M1", "SH-2-M2",
                "SS-2-M1", "SS-2-M2", "MH-2-M1", "LH-2-M1"],
        "Flow": [236, 195, 195, 157, 202, 163, 185, 205],
    })
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, df["Flow"], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Mix"], rotation=45, ha="right")
    ax.set_xlabel("Mix")
    ax.set_ylabel("Flow (mm)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    fig.tight_layout()


def compressive_age_plot():
    df = pd.DataFrame({
        "Specimen": ["N-0-M1", "LH-2-M1", "MH-2-M1", "SH-2-M1", "SS-2-M1"],
        "7-day": [75.11, 92.22, 95.01, 89.36, 89.83],
        "28-day": [77.65, 105.51, 106.53, 104.10, 105.25],
        "7-low": [72.85, 87.74, 91.78, 86.17, 83.56],
        "7-high": [77.06, 98.09, 99.24, 93.45, 96.43],
        "28-low": [76.90, 101.74, 104.69, 101.90, 99.59],
        "28-high": [78.65, 111.56, 107.87, 105.95, 110.37],
    })

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x-0.15, df["7-day"], width=0.25, label="7-day")
    ax.bar(x+0.15, df["28-day"], width=0.25, label="28-day")

    ax.errorbar(
        x-0.15, df["7-day"],
        yerr=[df["7-day"]-df["7-low"], df["7-high"]-df["7-day"]],
        fmt="none", capsize=3
    )
    ax.errorbar(
        x+0.15, df["28-day"],
        yerr=[df["28-day"]-df["28-low"], df["28-high"]-df["28-day"]],
        fmt="none", capsize=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["Specimen"])
    ax.set_xlabel("Specimen")
    ax.set_ylabel("Compressive Strength (MPa)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    fig.tight_layout()


def mix_comparison_plot():
    df = pd.DataFrame({
        "Type": ["N-0", "SH-2", "SS-2"],
        "MIX 1": [77.65, 104.10, 105.25],
        "MIX 2": [89.26, 119.47, 118.22],
        "M1-low": [76.90, 101.90, 99.59],
        "M1-high": [78.65, 105.95, 110.37],
        "M2-low": [87.60, 116.71, 115.86],
        "M2-high": [90.26, 121.95, 120.91],
    })

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x-0.15, df["MIX 1"], width=0.25, label="MIX 1")
    ax.bar(x+0.15, df["MIX 2"], width=0.25, label="MIX 2")
    ax.errorbar(
        x-0.15, df["MIX 1"],
        yerr=[df["MIX 1"]-df["M1-low"], df["M1-high"]-df["MIX 1"]],
        fmt="none", capsize=3
    )
    ax.errorbar(
        x+0.15, df["MIX 2"],
        yerr=[df["MIX 2"]-df["M2-low"], df["M2-high"]-df["MIX 2"]],
        fmt="none", capsize=3
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["Type"])
    ax.set_ylabel("Compressive Strength (MPa)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    fig.tight_layout()


def split_tensile_plot():
    df = pd.DataFrame({
        "Mix": ["N-0-M1", "LH-2-M1", "MH-2-M1", "SH-2-M1", "SS-2-M1"],
        "Mean": [3.12, 7.01, 11.27, 12.93, 12.74],
        "Low": [2.76, 6.70, 10.10, 12.28, 12.06],
        "High": [3.48, 7.28, 12.52, 13.56, 13.34],
    })

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x, df["Mean"], width=0.45, edgecolor="black")
    ax.errorbar(
        x, df["Mean"],
        yerr=[df["Mean"]-df["Low"], df["High"]-df["Mean"]],
        fmt="none", capsize=3
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["Mix"], rotation=45, ha="right")
    ax.set_xlabel("Mix")
    ax.set_ylabel("Split Tensile Strength (MPa)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    fig.tight_layout()


if __name__ == "__main__":
    flow_plot()
    compressive_age_plot()
    mix_comparison_plot()
    split_tensile_plot()
    plt.show()
