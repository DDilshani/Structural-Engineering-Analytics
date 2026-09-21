import argparse
import pandas as pd
import matplotlib.pyplot as plt


def read_files(file1, file2):
    df1 = pd.read_csv(file1, skiprows=16)
    df2 = pd.read_csv(file2, skiprows=16)
    return df1, df2


def combine(df1, df2, drop_first=40000):
    df1 = df1.iloc[drop_first:].reset_index(drop=True)
    df2 = df2.iloc[drop_first:].reset_index(drop=True)
    n = min(len(df1), len(df2))

    return pd.DataFrame({
        "CH_1": df1.loc[:n-1, "CH_1"],
        "CH_2": df1.loc[:n-1, "CH_2"],
        "CH_3": df1.loc[:n-1, "CH_3"],
        "CH_4": df1.loc[:n-1, "CH_4"],
        "Load": df2.loc[:n-1, "CH_1"],
        "Strain": df2.loc[:n-1, "CH_2"],
    })


def plot_channels(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for channel in ["CH_1", "CH_2", "CH_3", "CH_4"]:
        ax.plot(df[channel], df["Load"], label=f"{channel} vs Load")
    ax.set_xlabel("Channel value")
    ax.set_ylabel("Load")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(df["Strain"], df["Load"])
    ax2.set_xlabel("Strain")
    ax2.set_ylabel("Load")
    ax2.grid(True)
    fig2.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file1")
    parser.add_argument("file2")
    parser.add_argument("--drop-first", type=int, default=40000)
    parser.add_argument("--mode", choices=["all", "unique", "drop_every_10th"], default="all")
    parser.add_argument("--output", default="processed_channels.csv")
    args = parser.parse_args()

    df1, df2 = read_files(args.file1, args.file2)
    data = combine(df1, df2, args.drop_first)

    if args.mode == "unique":
        data = data[~data.duplicated(keep=False)]
    elif args.mode == "drop_every_10th":
        data = data.drop(index=data.index[::10]).reset_index(drop=True)

    data.to_csv(args.output, index=False)
    plot_channels(data)
    plt.show()
