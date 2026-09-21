import argparse
import joblib
import numpy as np
import pandas as pd

from common import load_dataset, PROJECT_ROOT
from config import FEATURES


def generate_candidates(X, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    data = {}

    for column in FEATURES:
        if column == "fibertype":
            data[column] = rng.choice(X[column].unique(), size=n)
        else:
            data[column] = rng.uniform(X[column].min(), X[column].max(), size=n)

    return pd.DataFrame(data)


def apply_original_example_constraints(df):
    df = df.copy()

    df["fillers"] = 0
    df["aspectratio"] = 65
    df["volume"] = 0.02
    df["tensilestrength"] = 1500

    return df[
        (df["supcem"] < 0.125)
        & (df["superplasticizer"] < 0.012)
    ]


def main(target, samples, constraints):
    model = joblib.load(PROJECT_ROOT / "models" / "xgboost.joblib")
    data = load_dataset()
    X = data[FEATURES]

    candidates = generate_candidates(X, samples)

    if constraints:
        candidates = apply_original_example_constraints(candidates)

    candidates["prediction"] = model.predict(candidates[FEATURES])
    candidates["error"] = candidates["prediction"] - target
    candidates["abs_error"] = candidates["error"].abs()
    candidates = candidates.sort_values("abs_error")

    # Deviation from nearest existing mixture
    best = candidates[(candidates["error"] >= 0) & (candidates["error"] <= 5)].head(200).copy()

    deviations = []
    base = X.replace(0, 1e-11)

    for _, row in best[FEATURES].iterrows():
        relative = ((X - row) / base).abs()
        deviations.append(relative.max(axis=1).min())

    best["deviation_from_existing"] = deviations
    best = best.sort_values(["deviation_from_existing", "abs_error"])

    out = PROJECT_ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)

    candidates.head(100).to_csv(
        out / f"candidate_mixes_target_{target:g}.csv",
        index=False,
    )
    best.to_csv(
        out / f"candidate_mixes_target_{target:g}_nearest_existing.csv",
        index=False,
    )

    print(best.head(20).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--constraints", action="store_true")
    args = parser.parse_args()

    main(args.target, args.samples, args.constraints)
