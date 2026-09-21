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
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


MODEL_NAME = "mlp"


def build_model():
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
        )),
    ])


def tune_model(X_train, y_train, quick=False):
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", MLPRegressor(
            max_iter=500,
            random_state=42,
        )),
    ])

    parameter_space = {
        "model__hidden_layer_sizes": [(20,), (50,), (50, 50), (100, 50)],
        "model__activation": ["relu", "tanh"],
        "model__solver": ["adam", "lbfgs"],
        "model__alpha": [0.05],
        "model__learning_rate": ["constant", "adaptive"],
    }

    search = RandomizedSearchCV(
        pipeline,
        parameter_space,
        n_iter=10 if quick else 30,
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
