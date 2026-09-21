import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import load_dataset, PROJECT_ROOT
from config import FEATURES, TARGET


def main():
    df = load_dataset()

    out = PROJECT_ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)

    df.describe().T.to_csv(out / "descriptive_statistics.csv")

    corr = df[FEATURES + [TARGET]].corr(numeric_only=True)
    corr.to_csv(out / "correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pair = sns.pairplot(df[FEATURES + [TARGET]], diag_kind="kde")
    pair.savefig(out / "pairplot.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    for feature in FEATURES:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(df[feature], df[TARGET], s=15)
        ax.set_xlabel(feature)
        ax.set_ylabel("Compressive Strength / (MPa)")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out / f"{feature}_vs_compstrength.png", dpi=250, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = ax.scatter(
        df["wb"],
        df["superplasticizer"],
        c=df[TARGET],
        s=15,
        cmap="cool",
    )
    fig.colorbar(scatter, ax=ax, label="Compressive Strength")
    ax.set_xlabel("w/b")
    ax.set_ylabel("Superplasticizer")
    ax.set_title("Superplasticizer vs w/b")
    fig.tight_layout()
    fig.savefig(out / "superplasticizer_vs_wb.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(df.describe().T)


if __name__ == "__main__":
    main()
