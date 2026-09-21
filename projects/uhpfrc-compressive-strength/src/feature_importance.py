import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from common import (
    load_dataset,
    split_dataset,
    load_saved_model,
    PROJECT_ROOT,
)
from config import FEATURES


def main(model_name):
    model = load_saved_model(model_name)

    df = load_dataset()
    _, X_test, _, y_test = split_dataset(df)

    out = PROJECT_ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)

    raw_model = (
        model.named_steps["model"]
        if hasattr(model, "named_steps")
        else model
    )

    if hasattr(raw_model, "feature_importances_"):
        values = np.asarray(raw_model.feature_importances_, dtype=float)
        values = 100 * values / values.sum()

        table = pd.DataFrame({
            "feature": FEATURES,
            "importance_percent": values,
        }).sort_values("importance_percent", ascending=False)

    else:
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=30,
            random_state=42,
            n_jobs=-1,
        )

        table = pd.DataFrame({
            "feature": FEATURES,
            "importance": result.importances_mean,
            "std": result.importances_std,
        }).sort_values("importance", ascending=False)

    table.to_csv(
        out / f"{model_name}_feature_importance.csv",
        index=False,
    )

    value_column = (
        "importance_percent"
        if "importance_percent" in table.columns
        else "importance"
    )

    ordered = table.sort_values(value_column)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(ordered["feature"], ordered[value_column])
    ax.set_xlabel(value_column.replace("_", " ").title())
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(
        out / f"{model_name}_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args.model)
