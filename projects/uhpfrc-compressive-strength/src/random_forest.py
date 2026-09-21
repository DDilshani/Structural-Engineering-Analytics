from pathlib import Path
import pandas as pd

from common import (
    load_dataset,
    split_dataset,
    PROJECT_ROOT,
    save_model,
)
from model_helpers import save_evaluation_outputs

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


MODEL_NAME = "random_forest"


def build_model():
    return RandomForestRegressor(
        n_estimators=1555,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        max_depth=11,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )


def tune_model(X_train, y_train, quick=False):
    parameter_space = {
        "n_estimators": [int(x) for x in np.linspace(1000, 2000, 10)],
        "max_features": [1.0, "sqrt", "log2", None, 0.2],
        "max_depth": [int(x) for x in np.linspace(1, 20, 10)] + [None],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 3],
        "bootstrap": [True, False],
    }

    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        parameter_space,
        n_iter=20 if quick else 100,
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
