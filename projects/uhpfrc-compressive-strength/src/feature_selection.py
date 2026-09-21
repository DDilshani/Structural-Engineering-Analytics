import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

from common import load_dataset, PROJECT_ROOT
from config import FEATURES, TARGET


def main(k=5):
    df = load_dataset()
    X = df[FEATURES]
    y = df[TARGET]

    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X, y)

    results = pd.DataFrame({
        "feature": FEATURES,
        "score": selector.scores_,
        "p_value": selector.pvalues_,
        "selected": selector.get_support(),
    }).sort_values("score", ascending=False)

    out = PROJECT_ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "feature_selection.csv", index=False)

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
