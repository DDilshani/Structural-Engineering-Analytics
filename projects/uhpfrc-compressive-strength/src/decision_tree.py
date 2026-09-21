from pathlib import Path
import pandas as pd

from common import (
    load_dataset,
    split_dataset,
    PROJECT_ROOT,
    save_model,
)
from model_helpers import save_evaluation_outputs

from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


MODEL_NAME = "decision_tree"


def build_model():
    return DecisionTreeRegressor(
        min_samples_split=16,
        min_samples_leaf=10,
        max_features=None,
        max_depth=4,
        random_state=42,
    )


def tune_model(X_train, y_train, quick=False):
    parameter_space = {
        "max_depth": [None] + list(range(1, 11)),
        "min_samples_split": [2, 3, 4, 8, 16],
        "min_samples_leaf": [1, 2, 3, 4, 10],
        "max_features": [None, "sqrt", "log2"],
    }

    search = RandomizedSearchCV(
        DecisionTreeRegressor(random_state=42),
        parameter_space,
        n_iter=15 if quick else 50,
        scoring="neg_mean_squared_error",
        cv=5 if quick else 10,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    return search


def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)

    model = build_model()
    model.fit(X_train, y_train)

    save_model(model, MODEL_NAME)

    result, _, _ = save_evaluation_outputs(
        MODEL_NAME,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        PROJECT_ROOT / "results",
    )

    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
