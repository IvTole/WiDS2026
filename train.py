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

    # --- CONTRIBUCIÓN CECILIA: VALIDACIÓN CRUZADA ---
    print("\n--- Iniciando Validación Cruzada Estratificada ---")
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import numpy as np
    
    # 1. Usamos StratifiedKFold para mantener la proporción de las clases
    # Bajamos a cv=3 porque tienes clases con muy pocos miembros (solo 3)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # 2. Cambiamos la métrica a 'roc_auc_ovr' (One-vs-Rest) para soportar multiclase
    try:
        cv_scores = cross_val_score(
            pipeline_lr, 
            X_train, 
            y_train, 
            cv=skf, 
            scoring='roc_auc_ovr' # 'ovr' soluciona el error multi_class
        )
        
        print(f"Scores ROC-AUC (OVR) por fold: {cv_scores}")
        print(f"Promedio ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    except Exception as e:
        print(f"Nota: No se pudo calcular ROC-AUC debido al desbalance extremo. Error: {e}")
        # Si falla el AUC por las clases pequeñas, usamos Accuracy como respaldo
        cv_scores = cross_val_score(pipeline_lr, X_train, y_train, cv=skf, scoring='accuracy')
        print(f"Scores Accuracy por fold: {cv_scores}")
        print(f"Promedio Accuracy: {np.mean(cv_scores):.4f}")
    
    print("-------------------------------------------\n")

    # Evaluate Models
    ev = ModelEvaluation(X=X_train, y=y_train, tag='lr')
    ev.evaluate_model(pipeline_lr)

if __name__ == "__main__":
    main()
