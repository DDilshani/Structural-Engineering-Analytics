import argparse
import pandas as pd
import matplotlib.pyplot as plt

COLUMN_MAP = {
    1: "CB Moment", 0: "CB Curvature",
    4: "RE-2S-10 Moment", 3: "RE-2S-10 Curvature",
    7: "RE-2S-20 Moment", 6: "RE-2S-20 Curvature",
    10: "RE-2S-30 Moment", 9: "RE-2S-30 Curvature",
    13: "RE-2S-40 Moment", 12: "RE-2S-40 Curvature",
}

CURVES = [
    ("RE-2S-10 Moment", "RE-2S-10 Curvature", "RE-2S-10"),
    ("RE-2S-20 Moment", "RE-2S-20 Curvature", "RE-2S-20"),
    ("RE-2S-30 Moment", "RE-2S-30 Curvature", "RE-2S-30"),
    ("RE-2S-40 Moment", "RE-2S-40 Curvature", "RE-2S-40"),
    ("CB Moment", "CB Curvature", "Control Beam"),
]


def plot_sheet(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet, header=None).rename(columns=COLUMN_MAP)
    fig, ax = plt.subplots(figsize=(6, 4))

    for mcol, ccol, label in CURVES:
        m = pd.to_numeric(df[mcol], errors="coerce")
        c = pd.to_numeric(df[ccol], errors="coerce")
        mask = m.notna() & c.notna()
        ax.plot(c[mask] / 1e-4, m[mask], linewidth=1, label=label)

    ax.set_xlabel(r"Curvature / ($10^{-4}$ rad/mm)")
    ax.set_ylabel("Moment / (kNm)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, linewidth=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("excel")
    parser.add_argument("--sheet", default="Sheet3")
    args = parser.parse_args()
    plot_sheet(args.excel, args.sheet)
    plt.show()
