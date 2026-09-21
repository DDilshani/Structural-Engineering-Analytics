import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline

from common import (
    load_dataset,
    split_dataset,
    load_saved_model,
    PROJECT_ROOT,
)
from config import FEATURES


def explain(model, X_train, X_explain):
    if isinstance(model, Pipeline):
        background = shap.sample(
            X_train,
            min(100, len(X_train)),
            random_state=42,
        )
        explainer = shap.KernelExplainer(model.predict, background)
        values = explainer.shap_values(X_explain)

        return shap.Explanation(
            values=np.asarray(values),
            data=np.asarray(X_explain),
            feature_names=list(X_explain.columns),
        )

    try:
        explainer = shap.Explainer(model, X_train)
        return explainer(X_explain)
    except Exception:
        background = shap.sample(
            X_train,
            min(100, len(X_train)),
            random_state=42,
        )
        explainer = shap.KernelExplainer(model.predict, background)
        values = explainer.shap_values(X_explain)

        return shap.Explanation(
            values=np.asarray(values),
            data=np.asarray(X_explain),
            feature_names=list(X_explain.columns),
        )


def main(model_name):
    model = load_saved_model(model_name)

    df = load_dataset()
    X_train, X_test, _, _ = split_dataset(df)
    X_all = df[FEATURES]
    y_all = df["compstrength"]

    out = PROJECT_ROOT / "results" / f"{model_name}_shap"
    out.mkdir(parents=True, exist_ok=True)

    test_exp = explain(model, X_train, X_test)

    shap.summary_plot(test_exp, X_test, show=False)
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    full_exp = explain(model, X_train, X_all)

    table = pd.DataFrame({
        "Data Point Index": X_all.index,
        "Compressive Strength": y_all.values,
    })

    for i, feature in enumerate(FEATURES):
        table[f"{feature} (SHAP Value)"] = full_exp.values[:, i]
        table[f"{feature} (Feature Value)"] = X_all[feature].values

        shap.dependence_plot(
            feature,
            full_exp.values,
            X_all,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            out / f"shap_dependence_{feature}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(
            X_all[feature],
            full_exp.values[:, i],
            c=y_all,
            s=12,
            cmap="cool",
        )
        fig.colorbar(scatter, ax=ax, label="Compressive Strength")
        ax.set_xlabel(feature)
        ax.set_ylabel(f"{feature} SHAP Value")
        fig.tight_layout()
        fig.savefig(
            out / f"shap_vs_feature_{feature}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    table.to_csv(out / "shap_values_and_features.csv", index=False)

    target = "cement"
    target_index = FEATURES.index(target)
    interactions = []

    for i, feature in enumerate(FEATURES):
        if i == target_index:
            continue

        score = np.mean(
            full_exp.values[:, i] *
            full_exp.values[:, target_index]
        )
        interactions.append((feature, abs(float(score))))

    interactions.sort(key=lambda item: item[1], reverse=True)

    with open(out / "cement_top_interactions.txt", "w", encoding="utf-8") as f:
        for feature, score in interactions[:3]:
            f.write(f"{feature}: {score:.8f}\n")

    row_index = min(10, len(X_all) - 1)
    single = explain(model, X_train, X_all.iloc[[row_index]])

    force = shap.force_plot(
        single.base_values[0],
        single.values[0],
        X_all.iloc[row_index],
        matplotlib=False,
    )
    shap.save_html(str(out / "shap_force_example.html"), force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args.model)
