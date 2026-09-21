import argparse
import numpy as np
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

from common import (
    load_dataset,
    split_dataset,
    load_saved_model,
    PROJECT_ROOT,
)
from config import FEATURES


def main(model_name, row_index=10):
    model = load_saved_model(model_name)

    df = load_dataset()
    X_train, _, _, _ = split_dataset(df)

    row_index = min(row_index, len(X_train) - 1)

    explainer = LimeTabularExplainer(
        np.asarray(X_train),
        feature_names=FEATURES,
        mode="regression",
        verbose=False,
    )

    explanation = explainer.explain_instance(
        np.asarray(X_train.iloc[row_index]),
        model.predict,
    )

    out = PROJECT_ROOT / "results" / f"{model_name}_lime"
    out.mkdir(parents=True, exist_ok=True)

    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(7, 5)
    fig.tight_layout()
    fig.savefig(out / "lime_example.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--row", type=int, default=10)
    args = parser.parse_args()
    main(args.model, args.row)
