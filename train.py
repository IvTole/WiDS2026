from datetime import datetime
from src.io import Dataset
from src.evaluation import ModelEvaluation
from src.preprocessing import build_preprocessor
from src.config import SELECTED_MODEL, MODEL_PARAMS # Importamos tu configuración

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier # Importamos el nuevo modelo

def main():
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Start Date and Time: ", start_datetime)

    data = Dataset()
    X_train, y_train, _ = data.load_data_xy()
    print(f"X_train shape: {X_train.shape}")

    # --- Lógica de Selección de Modelo (Contribución de Erick) ---
    if SELECTED_MODEL == "knn":
        model_instance = KNeighborsClassifier(**MODEL_PARAMS["knn"])
        tag_name = 'knn'
    else:
        model_instance = LogisticRegression(**MODEL_PARAMS["lr"])
        tag_name = 'lr'

    # Pipeline unificado
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", model_instance)
        ]
    )

    # Evaluación y registro en MLflow (usando el tag dinámico)
    ev = ModelEvaluation(X=X_train, y=y_train, tag=tag_name)
    ev.evaluate_model(pipeline)

if __name__ == "__main__":
    main()