import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PATH, FEATURES, TARGET, TEST_SIZE, RANDOM_STATE


def load_dataset(path=DATA_PATH):
    df = pd.read_excel(path)

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    return df[required].copy().dropna()


def split_dataset(df):
    return train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )


def ensure_output_dirs():
    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)


def save_model(model, name):
    ensure_output_dirs()
    path = PROJECT_ROOT / "models" / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_saved_model(name):
    return joblib.load(PROJECT_ROOT / "models" / f"{name}.joblib")
