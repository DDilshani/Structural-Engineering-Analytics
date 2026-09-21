import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main(path):
    df = pd.read_excel(path, header=None)
    n = df.shape[1] // 2

    for i in range(n):
        ycol = i * 2
        xcol = ycol + 1

        title = df.iloc[0, ycol]
        ylabel = df.iloc[1, ycol]
        xlabel = df.iloc[1, xcol]

        x = pd.to_numeric(df.iloc[2:, xcol], errors="coerce")
        y = pd.to_numeric(df.iloc[2:, ycol], errors="coerce")
        mask = x.notna() & y.notna()

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(x[mask], y[mask], linewidth=1.5, label=title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("excel")
    args = parser.parse_args()
    main(args.excel)
