import argparse
import pandas as pd
import matplotlib.pyplot as plt

from structural_analytics.response import response_points


def clean(load, disp):
    load = pd.to_numeric(load, errors="coerce")
    disp = pd.to_numeric(disp, errors="coerce")
    mask = load.notna() & disp.notna()
    return load[mask].to_numpy(), disp[mask].to_numpy()


def analyse(load, disp, label):
    result = response_points(load, disp)
    print(f"\n{label}")
    for key, value in result.items():
        print(f"{key:22s}: {value:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(disp, load, label="Load-Displacement Curve")

    ax.axhline(result["peak_load"], linestyle="--", label="Peak load")
    ax.axvline(result["yield_displacement"], linestyle=":", label="Yield displacement")

    if pd.notna(result["yield_load"]):
        ax.scatter(
            result["yield_displacement"],
            result["yield_load"],
            label="Yield point",
        )

    ax.scatter(
        result["peak_displacement"],
        result["peak_load"],
        label="Peak point",
    )

    ax.set_xlabel("Displacement")
    ax.set_ylabel("Load")
    ax.set_title(label)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return result, fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("excel")
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--load-col", type=int, default=0)
    parser.add_argument("--disp-col", type=int, default=1)
    parser.add_argument("--skiprows", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_excel(
        args.excel,
        sheet_name=args.sheet,
        header=None,
        skiprows=args.skiprows,
    )
    load, disp = clean(df.iloc[:, args.load_col], df.iloc[:, args.disp_col])
    analyse(load, disp, "Load vs Displacement")
    plt.show()
