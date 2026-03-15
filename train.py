from datetime import datetime
from src.io import Dataset
from src.evaluation import ModelEvaluation
from src.preprocessing import build_preprocessor

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def main():
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start Date and Time: ", start_datetime)

    data = Dataset()
    X_train, y_train, _ = data.load_data_xy() # Loads train and test dataset (X feature matrix and y target matrix)
    print(f"X_train shape: {X_train.shape}")

    # Model pipeline
    pipeline_lr = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(solver="lbfgs", max_iter=5000))
        ]
    )

    # Evaluate Models
    ev = ModelEvaluation(X=X_train, y=y_train, tag='lr')
    ev.evaluate_model(pipeline_lr)

if __name__ == "__main__":
    main()
