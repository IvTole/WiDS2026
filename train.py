from datetime import datetime
from src.io import Dataset
from src.evaluation import ModelEvaluation
from src.preprocessing import build_preprocessor

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier # <-- Importamos XGBoost

def main():
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start Date and Time: ", start_datetime)

    data = Dataset()
    # Loads train and test dataset (X feature matrix and y target matrix)
    X_train, y_train, _ = data.load_data_xy() 
    print(f"X_train shape: {X_train.shape}")

    
    # 1. Modelo Base: Regresión Logística
    
    pipeline_lr = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(solver="saga", max_iter=5000, penalty='elasticnet', l1_ratio=0.5))
        ]
    )

    print("\n--- Entrenando Logistic Regression ---")
    ev_lr = ModelEvaluation(X=X_train, y=y_train, tag='lr')
    ev_lr.evaluate_model(pipeline_lr)


    
    # 2. TU CONTRIBUCIÓN: Modelo XGBoost
    
    pipeline_xgb = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            # Parámetros iniciales estándar para empezar
            ("model", XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)) 
        ]
    )

    print("\n--- Entrenando XGBoost ---")
    ev_xgb = ModelEvaluation(X=X_train, y=y_train, tag='xgb_fajardo') # Tu tag personalizado
    ev_xgb.evaluate_model(pipeline_xgb)

if __name__ == "__main__":
    main()