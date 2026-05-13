# Paths, columns, seeds, bins

from pathlib import Path

# mlflow issues
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow.vanotole-lab.com"
os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "false"

# Paths
def train_data_path() -> Path:
    """
    Returns the location of train data directory, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the train data directory
    """
    cwd = Path.cwd()
    for folder in (cwd, cwd / "..", cwd / ".." / ".."):
        data_folder = folder / "data" / "train.csv"
        if data_folder.exists() and data_folder.is_file():
            print("Train data directory found in ", data_folder)
            return data_folder
    raise Exception("Train data not found")
        
def test_data_path() -> Path:
    """
    Returns the location of test data directory, allowing for script executions in subfolders without worrying about the
    relative location of the data

    :return: the path to the test data directory
    """
    cwd = Path.cwd()
    for folder in (cwd, cwd / "..", cwd / ".." / ".."):
        data_folder = folder / "data" / "test.csv"
        if data_folder.exists() and data_folder.is_file():
            print("Test data directory found in ", data_folder)
            return data_folder
    raise Exception("Test data not found")
        
# Feature columns
COL_EVENT_ID = "event_id"

# Target columns
COL_TIME_TO_HIT_HOURS = "time_to_hit_hours"
COL_EVENT = "event"

# Target parameters
TIME_BINS_HOURS = [0, 12, 24, 48, 72]   # ejemplo
N_CLASSES = 4
CENSORED_CLASS = 3

# MLFlow
MLFLOW_TRACKING_URL = "http://mlflow.vanotole-lab.com"
MLFLOW_EXPERIMENT_NAME = "WiDS2026"