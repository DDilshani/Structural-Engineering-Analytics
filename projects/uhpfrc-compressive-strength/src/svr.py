from pathlib import Path
import pandas as pd

from common import (
    load_dataset,
    split_dataset,
    PROJECT_ROOT,
    save_model,
)
from model_helpers import save_evaluation_outputs

import json
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


MODEL_NAME = "svr"


def build_model():
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", SVR(
            kernel="poly",
            gamma=100.0,
            degree=2,
            C=0.01,
        )),
    ])


def tune_model(X_train, y_train, quick=False):
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", SVR()),
    ])

    parameter_space = {
        "model__kernel": ["poly", "rbf", "sigmoid"],
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", "auto", 0.01, 0.1, 1],
        "model__epsilon": [0.1, 0.2, 0.5, 1.0],
        "model__degree": [2, 3, 4],
        "model__coef0": [0.0, 0.1, 1.0],
        "model__tol": [1e-4, 1e-3, 1e-2],
        "model__shrinking": [True, False],
    }

    search = RandomizedSearchCV(
        pipeline,
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
