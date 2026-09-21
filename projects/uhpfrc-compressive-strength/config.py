from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT.parents[1] / "data" / "ml" / "uhpfrc_compressive_strength_database.xlsx"

TARGET = "compstrength"

FEATURES = [
    "fineagg",
    "cement",
    "fibertype",
    "superplasticizer",
    "fillers",
    "volume",
    "wb",
    "tensilestrength",
    "aspectratio",
    "supcem",
]

FEATURE_LABELS = {
    "fineagg": "Fine Aggregate",
    "cement": "Cement",
    "fibertype": "Fiber Type",
    "superplasticizer": "Superplasticizer",
    "fillers": "Fillers",
    "volume": "Volume of Fibers",
    "wb": "w/b",
    "tensilestrength": "Fiber Tensile Strength",
    "aspectratio": "Fiber Aspect Ratio",
    "supcem": "Supplementary Cementitious Materials",
}

TEST_SIZE = 0.20
RANDOM_STATE = 42
