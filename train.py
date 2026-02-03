from datetime import datetime
from src.io import Dataset
from src.evaluation import ModelEvaluation

from sklearn.linear_model import LogisticRegression

def main():
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start date & time : ", start_datetime)

    data = Dataset()
    X_train, y_train, X_test = data.load_data_xy() # Loads train and test dataset (X feature matrix and y target matrix)
    print(X_train)
    print(y_train)
    print(X_test)

    # Evaluate Models
    ev = ModelEvaluation(X=X_train, y=y_train, tag='lr')
    ev.evaluate_model(LogisticRegression(solver='lbfgs', max_iter=5000))

if __name__ == "__main__":
    main()
