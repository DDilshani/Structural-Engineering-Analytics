from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

from structural_analytics.metrics import regression_metrics
from structural_analytics.plotting import actual_vs_predicted


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = regression_metrics(y_train, train_pred)
    test_metrics = regression_metrics(y_test, test_pred)

    result = {
        "model": name,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }

    return result, train_pred, test_pred


def save_evaluation_outputs(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result, train_pred, test_pred = evaluate_model(
        name,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    pd.DataFrame([result]).to_csv(
        output_dir / f"{name}_metrics.csv",
        index=False,
    )

    fig, _ = actual_vs_predicted(
        y_train,
        train_pred,
        y_test,
        test_pred,
        title=name,
    )
    fig.savefig(
        output_dir / f"{name}_actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    save_learning_curve(
        model,
        X_train,
        y_train,
        output_dir / f"{name}_learning_curve.png",
    )

    return result, train_pred, test_pred


def save_learning_curve(model, X_train, y_train, path):
    train_sizes, train_scores, validation_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )

    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    validation_mean = -np.mean(validation_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
    )
    ax.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.1,
    )

    ax.plot(train_sizes, train_mean, "o-", label="Training MSE")
    ax.plot(
        train_sizes,
        validation_mean,
        "o-",
        label="Cross-validation MSE",
    )

    ax.set_xlabel("Training examples")
    ax.set_ylabel("MSE")
    ax.set_title("Learning Curve")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_actual_predicted_series(
    y_train,
    train_pred,
    y_test,
    test_pred,
    path,
):
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    actual = pd.concat([y_train, y_test]).sort_index()

    predicted = pd.concat([
        pd.Series(train_pred, index=y_train.index),
        pd.Series(test_pred, index=y_test.index),
    ]).sort_index()

    table = pd.DataFrame({
        "Actual": actual,
        "Predicted": predicted,
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(table.index, table["Actual"], marker="o", label="Actual")
    ax.plot(table.index, table["Predicted"], marker="x", label="Predicted")

    ax.set_xlabel("Data Points")
    ax.set_ylabel("Compressive Strength / (MPa)")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return table
