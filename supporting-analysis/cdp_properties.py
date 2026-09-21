import argparse
import pandas as pd
import matplotlib.pyplot as plt


def clean(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    return x[mask], y[mask]


def main(path, sheet="Sheet3"):
    df = pd.read_excel(path, sheet_name=sheet, header=1)

    plots = [
        (1, 0, "Inelastic Strain", "Damage", "cdp_damage_1.png"),
        (3, 2, "Inelastic Strain", "Damage", "cdp_damage_2.png"),
        (5, 4, "Strain", "Stress / (MPa)", "cdp_stress_1.png"),
        (7, 6, "Strain", "Stress / (MPa)", "cdp_stress_2.png"),
    ]

    for xcol, ycol, xlabel, ylabel, filename in plots:
        x, y = clean(df.iloc[:, xcol], df.iloc[:, ycol])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(x, y, "k-", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("excel")
    parser.add_argument("--sheet", default="Sheet3")
    args = parser.parse_args()
    main(args.excel, args.sheet)
