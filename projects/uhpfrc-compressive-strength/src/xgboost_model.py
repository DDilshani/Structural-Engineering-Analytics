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
from xgboost import XGBRegressor


MODEL_NAME = "xgboost"


def build_model():
    return XGBRegressor(
        subsample=0.8,
        reg_lambda=10,
        reg_alpha=0.1,
        n_estimators=1520,
        min_child_weight=1,
        max_depth=3,
        learning_rate=0.05,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )


def tune_model(X_train, y_train, quick=False):
    parameter_space = {
        "n_estimators": list(range(1600, 1650, 5)),
        "max_depth": [9],
        "learning_rate": [0.05],
        "subsample": [0.7],
        "colsample_bytree": [0.7],
        "reg_alpha": [20],
        "reg_lambda": [10],
        "min_child_weight": [1, 2],
    }

    search = RandomizedSearchCV(
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
        parameter_space,
        n_iter=10 if quick else 20,
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
