from pathlib import Path
import os
# Suppress unwanted TensorFlow loggings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Define paths
ROOT_PATH      = Path(__file__).resolve().parents[2]
RAW_PATH       = ROOT_PATH   / "data"        / "raw"
PROCESSED_PATH = ROOT_PATH   / "data"        / "processed"
LOGS_PATH      = ROOT_PATH   / "logs"
MODEL_PATH     = ROOT_PATH   / "model"
OPTUNA_PATH    = ROOT_PATH   / "optuna"
PARAMS_PATH    = MODEL_PATH  / "params.json"
STUDY_PATH     = OPTUNA_PATH / "hp.log"
# Create directories if they do not exist
RAW_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)
OPTUNA_PATH.mkdir(parents=True, exist_ok=True)
