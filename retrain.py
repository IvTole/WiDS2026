import argparse #permite pasar argumentos al script desde la línea de comandos
import mlflow
import mlflow.sklearn

from src.config import MLFLOW_TRACKING_URL, MLFLOW_EXPERIMENT_NAME
from src.io import Dataset

def main(model_id: str, tag: str):
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)

    # Model load
    model = mlflow.sklearn.load_model(f"models:/{model_id}")
    print(f"Loaded: {type(model).__name__} from run {model_id}")

    # Load data
    data = Dataset()
    X_train, y_train, _ = data.load_data_xy()
    print(f"Retraining on: {X_train.shape[0]} samples.")
    model.fit(X_train, y_train)
    print(f"Retrained on {X_train.shape[0]} samples.")

    # Model registry
    model_type = type(model.named_steps["model"]).__name__
    run_name = f"production_{tag}"
    artifact_name = f"production_{tag}"

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("stage", "production")
        mlflow.set_tag("source_model_id", model_id)
        mlflow.sklearn.log_model(model, name=artifact_name)
        print(f"Production model logged: {run.info.run_id}")

if __name__ == "__main__":

    # cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True) # e.g. m-0d9ed811fd3546c69228f636ffd1504b
    parser.add_argument("--tag", required=True) # e.g. lr_stdscaler, algo de contexto, incluido apellido de estudiante
    args = parser.parse_args()

    main(args.model_id, args.tag)