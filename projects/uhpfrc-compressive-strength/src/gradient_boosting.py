from pathlib import Path
import pandas as pd

from common import (
    load_dataset,
    split_dataset,
    PROJECT_ROOT,
    save_model,
)
from model_helpers import save_evaluation_outputs

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV


MODEL_NAME = "gradient_boosting"


def build_model():
    return GradientBoostingRegressor(
        n_estimators=150,
        min_samples_split=6,
        min_samples_leaf=4,
        max_features=None,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )


def tune_model(X_train, y_train, quick=False):
    parameter_space = {
        "n_estimators": [150],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4, 5, 7],
        "min_samples_split": [2, 3, 4, 5, 6],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": [None, "sqrt", "log2"],
    }

    search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        parameter_space,
        n_iter=15 if quick else 60,
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
