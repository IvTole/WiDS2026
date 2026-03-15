import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from src.config import MLFLOW_TRACKING_URL, COL_EVENT_ID
from src.io import Dataset
from src.config import test_data_path

def main(model_id: str, output_name: str):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)

    # Load production model
    model = mlflow.sklearn.load_model(f"models:/{model_id}")
    print(f"Loaded model {type(model).__name__} from model id {model_id}")

    # Load test data
    data = Dataset()
    _, _, X_test = Dataset().load_data_xy()
    event_ids = pd.read_csv(test_data_path())[COL_EVENT_ID]
    
    # Predict probabilities
    probs = model.predict_proba(X_test)

    # Build submission file
    submission = pd.DataFrame(
        {
            COL_EVENT_ID: event_ids,
            "prob_12h": probs[:, 0],
            "prob_24h": probs[:, 1],
            "prob_48h": probs[:, 2],
            "prob_72h": probs[:, 3]
        }
    )

    submission.to_csv(path_or_buf=f"submissions/{output_name}.csv",
                      index=False)
    print(f"Submission saved to submissions/{output_name} ({len(submission)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-name", default="submission")
    args = parser.parse_args()

    main(args.model_id, args.output_name)