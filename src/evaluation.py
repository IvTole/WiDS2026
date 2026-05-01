# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import functools

# Sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# MLFlow
from src.tracking import mlflow_logger
from mlflow import log_param, log_metric
import mlflow.sklearn
from mlflow.models.signature import infer_signature




class ModelEvaluation:
    """
    Supports the evaluation of classification models (multinomial), collecting the results.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, tag: str, test_size: float = 0.2, shuffle: bool = True, random_state: int = 42):
        """
        :param X: the inputs
        :param y: the prediction targets
        :param test_size: the fraction of the data to reserve for testing
        :param shuffle: whether to shuffle the data prior to splitting
        :param random_state: the random seed
        :param tag: target name for logging
        """

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y,
            random_state=random_state, test_size=test_size, shuffle=shuffle)
        
        self.tag = tag

    @mlflow_logger
    def evaluate_model(self, model) -> float:
        """
        :param model: the model to evaluate
        :return: the accuracy score
        """
        
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        model_type = type(model.named_steps['model']).__name__
        print(f"\n=============================================")
        print(f"Evaluando Modelo: {model_type}")
        print(f"=============================================")
        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_valid)
        
        # 1. Calculamos el accuracy básico
        acc = accuracy_score(y_true=self.y_valid, y_pred=y_pred)
        print(f"Accuracy General: {acc:.4f}\n")

        # 2. MEJORA: Reporte de Clasificación Detallado
        print("--- Reporte de Clasificación ---")
        reporte = classification_report(self.y_valid, y_pred)
        print(reporte)

        # 3. MEJORA: Matriz de Confusión Visual (Opcional, guardada como imagen)
        
        cm = confusion_matrix(self.y_valid, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {model_type}')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción')
        #plt.savefig(f'confusion_matrix_{model_type}.png') # Guarda la imagen en tu carpeta
        plt.show()

        
        # log parameters and metrics in MLFlow
        model_name = type(model).__name__ + '_' + self.tag
        # ...