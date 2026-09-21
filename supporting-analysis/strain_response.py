import argparse
import pandas as pd
import matplotlib.pyplot as plt

COLUMN_MAP = {
    0: "Control Load",
    1: "Control Strain stirrup 1",
    2: "Control Strain longitudinal bar",
    3: "Control Strain stirrup 2",
    5: "RE-2S-40 Load",
    6: "RE-2S-40 Strain",
    8: "RE-2S-40R Load",
    9: "RE-2S-40R Strain long",
    11: "RE-2S-40G Load",
    12: "RE-2S-40G Strain long",
}


def plot_pairs(df, pairs, title):
    fig, ax = plt.subplots(figsize=(5, 4))

    for strain_col, load_col, label in pairs:
        x = pd.to_numeric(df[strain_col], errors="coerce")
        y = pd.to_numeric(df[load_col], errors="coerce")
        mask = x.notna() & y.notna()
        ax.plot(x[mask], y[mask], linewidth=1, label=label)

    ax.set_xlabel("Strain / (με)")
    ax.set_ylabel("Load / (kN)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("excel")
    parser.add_argument("--sheet", default="Sheet2")
    args = parser.parse_args()

    df = pd.read_excel(args.excel, sheet_name=args.sheet, header=None).rename(columns=COLUMN_MAP)

    plot_pairs(df, [
        ("Control Strain stirrup 1", "Control Load", "CB – Stirrup 1"),
        ("Control Strain longitudinal bar", "Control Load", "CB – Longitudinal"),
        ("Control Strain stirrup 2", "Control Load", "CB – Stirrup 2"),
    ], "Control Beam")

    plot_pairs(df, [
        ("Control Strain longitudinal bar", "Control Load", "CB – Longitudinal"),
        ("RE-2S-40R Strain long", "RE-2S-40R Load", "RE-2S-40R – Longitudinal"),
        ("RE-2S-40G Strain long", "RE-2S-40G Load", "RE-2S-40G – Longitudinal"),
    ], "Longitudinal Strain Comparison")

    plot_pairs(df, [
        ("Control Strain stirrup 1", "Control Load", "CB – Stirrup 1"),
        ("Control Strain stirrup 2", "Control Load", "CB – Stirrup 2"),
        ("RE-2S-40 Strain", "RE-2S-40 Load", "RE-2S-40 – Stirrup"),
    ], "Stirrup Strain Comparison")

    plt.show()
