import argparse

from common import (
    load_dataset,
    split_dataset,
    load_saved_model,
    PROJECT_ROOT,
)
from model_helpers import save_learning_curve


def main(model_name):
    model = load_saved_model(model_name)

    df = load_dataset()
    X_train, _, y_train, _ = split_dataset(df)

    save_learning_curve(
        model,
        X_train,
        y_train,
        PROJECT_ROOT / "results" / f"{model_name}_learning_curve.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args.model)
