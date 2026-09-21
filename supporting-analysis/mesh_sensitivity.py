import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from structural_analytics.response import response_points


def clean(load, disp):
    load = pd.to_numeric(load, errors="coerce")
    disp = pd.to_numeric(disp, errors="coerce")
    mask = load.notna() & disp.notna()
    return load[mask].to_numpy(), disp[mask].to_numpy()


def sheet1_analysis(path):
    df = pd.read_excel(path, sheet_name="Sheet1", header=0)

    specs = {
        "50 mm": (0, 1),
        "40 mm": (2, 3),
        "30 mm": (4, 5),
        "25 mm": (6, 7),
    }

    rows = []
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, (lcol, dcol) in specs.items():
        load, disp = clean(df.iloc[:, lcol], df.iloc[:, dcol])
        ax.plot(disp, load, label=label)
        rows.append({"mesh": label, **response_points(load, disp)})

    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Load (kN)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    return pd.DataFrame(rows), fig


def accuracy_plot():
    df = pd.DataFrame({
        "Mesh size (mm)": [50, 40, 30, 25],
        "Peak load": [77.9, 87.2, 91.2, 93.6],
        "Deflection at peak": [99.6, 99.8, 96.9, 97.2],
        "Failure load": [55.0, 73.3, 85.0, 83.7],
        "Deflection at failure": [91.8, 94.7, 94.3, 93.8],
        "Stiffness": [62.7, 81.1, 85.2, 88.0],
    })

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 6))

    for col in df.columns[1:]:
        ax.plot(x, df[col], marker="o", label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(df["Mesh size (mm)"])
    ax.set_xlabel("Mesh size / (mm)")
    ax.set_ylabel("Accuracy / (%)")
    ax.set_ylim(40, 105)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("excel")
    args = parser.parse_args()

    summary, fig = sheet1_analysis(args.excel)
    print(summary.to_string(index=False))
    accuracy_plot()
    plt.show()
