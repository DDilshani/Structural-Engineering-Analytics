import joblib

from common import load_dataset, split_dataset, PROJECT_ROOT
from model_helpers import save_actual_predicted_series


MODELS = [
    "svr",
    "random_forest",
    "gradient_boosting",
]


def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(df)
    out = PROJECT_ROOT / "results"

    for name in MODELS:
        model = joblib.load(PROJECT_ROOT / "models" / f"{name}.joblib")
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        table = save_actual_predicted_series(
            y_train,
            train_pred,
            y_test,
            test_pred,
            out / f"{name}_series.png",
        )
        table.to_excel(out / f"{name}_sorted_values.xlsx", index=False)


if __name__ == "__main__":
    main()
