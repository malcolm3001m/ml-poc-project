from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for dir in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

MODELS = {
    "decision_tree": {
        "name": "Decision Tree",
        "description": "Single tree baseline; log1p-transformed target via TransformedTargetRegressor.",
        "path": MODELS_DIR / "decision_tree.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Tree ensemble; log1p-transformed target via TransformedTargetRegressor.",
        "path": MODELS_DIR / "random_forest.joblib",
    },
    "xgboost_regressor": {
        "name": "XGBoost Regressor",
        "description": "Gradient-boosted trees; log1p-transformed target via TransformedTargetRegressor.",
        "path": MODELS_DIR / "xgboost_regressor.joblib",
    },
    "catboost_regressor": {
        "name": "CatBoost Regressor",
        "description": "Gradient boosting with NATIVE categorical handling (no label-encoding hack). "
                       "Manual log1p wrap because sklearn's TransformedTargetRegressor cannot clone "
                       "CatBoost. Winner of scripts/experiment.py.",
        "path": MODELS_DIR / "catboost_regressor.joblib",
    },
}
