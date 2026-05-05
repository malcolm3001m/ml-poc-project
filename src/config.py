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
        "description": "Single tree baseline trained on a log transformed peak value target.",
        "path": MODELS_DIR / "decision_tree.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Tree ensemble trained to capture non linear player value patterns.",
        "path": MODELS_DIR / "random_forest.joblib",
    },
    "xgboost_regressor": {
        "name": "XGBoost Regressor",
        "description": "Gradient boosted tree model trained for peak market value prediction.",
        "path": MODELS_DIR / "xgboost_regressor.joblib",
    },
}
