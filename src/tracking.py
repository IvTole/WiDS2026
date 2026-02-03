# MlFlow tracking
# For opening experiments
import functools
import traceback
import mlflow
from src.config import MLFLOW_TRACKING_URL, MLFLOW_EXPERIMENT_NAME

def mlflow_logger(func):
    """Decorator to automatically start and close an mlflow run"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)  # nombre estable
       #try:
       #     exp_id = mlflow.create_experiment(name=MLFLOW_EXPERIMENT_NAME)
       # except Exception as e:
        #    exp_id = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME).experiment_id

        with mlflow.start_run():
            print("run_id:", mlflow.active_run().info.run_id)
            print("artifact_uri:", mlflow.get_artifact_uri())
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print("\n[MLFLOW LOGGER] Exception raised inside decorated function:")
                traceback.print_exc()
                mlflow.set_tag("exception", repr(e)) # to see error log inside UI
                raise
    return wrapper
