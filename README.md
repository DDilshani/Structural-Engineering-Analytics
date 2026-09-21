# Structural Engineering Analytics

A code-focused repository containing machine-learning, experimental post-processing, numerical-validation and plotting work from structural-engineering research.

The main ML project predicts UHPFRC compressive strength. Each machine-learning model has its **own Python file**, so the implementation is easy to browse on GitHub.

## UHPFRC Dataset

Place the Excel dataset here:

```text
projects/uhpfrc-compressive-strength/data/raw/uhpfrc_compressive_strength_database.xlsx
```

Target:

```text
compstrength
```

Features:

```text
fineagg
cement
fibertype
superplasticizer
fillers
volume
wb
tensilestrength
aspectratio
supcem
```

## Machine-Learning Models

Each model can be run independently.

### Multiple Linear Regression

```bash
python projects/uhpfrc-compressive-strength/src/linear_regression.py
```

### Support Vector Regression

```bash
python projects/uhpfrc-compressive-strength/src/svr.py
```

### Random Forest

```bash
python projects/uhpfrc-compressive-strength/src/random_forest.py
```

### Gradient Boosting

```bash
python projects/uhpfrc-compressive-strength/src/gradient_boosting.py
```

### Decision Tree

```bash
python projects/uhpfrc-compressive-strength/src/decision_tree.py
```

### XGBoost

```bash
python projects/uhpfrc-compressive-strength/src/xgboost_model.py
```

### LightGBM

```bash
python projects/uhpfrc-compressive-strength/src/lightgbm_model.py
```

### MLP Regressor

```bash
python projects/uhpfrc-compressive-strength/src/mlp.py
```

Each model file includes:

- model definition;
- training;
- prediction;
- regression metrics;
- actual-vs-predicted graph;
- learning curve;
- model saving; and
- hyperparameter tuning when applicable.

## Additional ML Analysis

The repository also includes:

- exploratory analysis;
- correlation heatmaps;
- pair plots;
- feature selection;
- SHAP summary plots;
- SHAP dependence plots;
- SHAP force plots;
- LIME explanations;
- feature importance;
- permutation importance for MLP;
- actual-vs-predicted series plots;
- multi-model comparison;
- Taylor diagram; and
- target compressive-strength candidate mix search.

## Supporting Structural Analysis

The `supporting-analysis/` folder contains the non-ML research coding:

- moment-curvature diagrams;
- jacket-thickness capacity comparison;
- load-displacement processing;
- 2/3-P yield-point estimation;
- ductility and stiffness calculations;
- beam-response comparison;
- strain-response plots;
- raw test-channel processing;
- experimental-vs-numerical validation;
- material-test graphs;
- CDP-property plots;
- mesh sensitivity; and
- generic paired-column Excel plotting.


## Run All Analyses

After placing the datasets in the `data/` folders, execute:

```bash
python run_all_analyses.py
```

The master runner checks whether each input file exists, executes the matching analysis scripts, and skips analyses whose input files are missing.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
pip install -e .
```

Run tests:

```bash
pytest -q
```
