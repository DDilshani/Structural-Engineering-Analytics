import numpy as np
import matplotlib.pyplot as plt


def actual_vs_predicted(y_train, train_pred, y_test, test_pred, title=None):
    values = np.concatenate([
        np.asarray(y_train, dtype=float),
        np.asarray(train_pred, dtype=float),
        np.asarray(y_test, dtype=float),
        np.asarray(test_pred, dtype=float),
    ])

    lower = np.floor(values.min() / 10) * 10
    upper = np.ceil(values.max() / 10) * 10
    axis = np.array([lower, upper])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_train, train_pred, alpha=0.5, label="Train")
    ax.scatter(y_test, test_pred, alpha=0.7, label="Test")

    ax.plot(axis, axis, linestyle="--", label="x = y")
    ax.fill_between(axis, 0.9 * axis, 1.1 * axis, alpha=0.10, label="±10%")
    ax.fill_between(axis, 0.8 * axis, 1.2 * axis, alpha=0.07, label="±20%")

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Actual Compressive Strength / (MPa)")
    ax.set_ylabel("Predicted Compressive Strength / (MPa)")

    if title:
        ax.set_title(title)

    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig, ax
