import argparse
import pandas as pd
import matplotlib.pyplot as plt
from structural_analytics.response import response_points


def clean(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    return x[mask].to_numpy(), y[mask].to_numpy()


def main(path, skiprows=2):
    df = pd.read_excel(path, header=None, skiprows=skiprows)

    curves = [
        (1, 0, "CB - experimental", "-"),
        (4, 3, "CB - numerical", "--"),
        (7, 6, "RE-40 - experimental", "-"),
        (10, 9, "RE-40 - numerical", "--"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    for xcol, ycol, label, linestyle in curves:
        disp, load = clean(df.iloc[:, xcol], df.iloc[:, ycol])
        ax.plot(disp, load, linestyle=linestyle, label=label)

        pts = response_points(load, disp)
        print(f"\n{label}")
        for key, value in pts.items():
            print(f"{key:22s}: {value:.4f}")

    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Load (kN)")
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("excel")
    parser.add_argument("--skiprows", type=int, default=2)
    args = parser.parse_args()
    main(args.excel, args.skiprows)
    plt.show()
