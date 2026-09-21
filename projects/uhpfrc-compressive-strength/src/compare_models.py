import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (
    load_dataset,
    split_dataset,
    load_saved_model,
    PROJECT_ROOT,
)
from structural_analytics.metrics import regression_metrics


MODELS = [
    "linear_regression",
    "svr",
    "random_forest",
    "gradient_boosting",
    "decision_tree",
    "xgboost",
    "lightgbm",
    "mlp",
]


def main():
    df = load_dataset()
    _, X_test, _, y_test = split_dataset(df)

    rows = []
    predictions = {}

    fig, ax = plt.subplots(figsize=(9, 7))

    for name in MODELS:
        model = load_saved_model(name)
        pred = model.predict(X_test)
        predictions[name] = pred

        rows.append({
            "model": name,
            **regression_metrics(y_test, pred),
        })

        ax.scatter(
            y_test,
            pred,
            alpha=0.55,
            label=name,
        )

    lo = min(y_test.min(), *(p.min() for p in predictions.values()))
    hi = max(y_test.max(), *(p.max() for p in predictions.values()))

    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Actual Compressive Strength")
    ax.set_ylabel("Predicted Compressive Strength")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        PROJECT_ROOT / "results" / "all_models_actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    table = pd.DataFrame(rows).sort_values("r2", ascending=False)
    table.to_csv(
        PROJECT_ROOT / "results" / "model_comparison.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(table["model"], table["r2"])
    ax.set_ylabel("Test R²")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(
        PROJECT_ROOT / "results" / "model_r2_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
