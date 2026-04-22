# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import functools

# Sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score
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
        #print("Model Type", type(model).__name__ + '_' + self.tag)
        model_type = type(model.named_steps['model']).__name__
        print(f"Model Type: {model_type}")
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_valid)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(self.X_valid)
        else:
            y_prob = None
        acc = accuracy_score(y_true=self.y_valid, y_pred=y_pred)
        f1 = f1_score(self.y_valid, y_pred, average="weighted")
        roc_auc = None
        if y_prob is not None:
            try:
                roc_auc = roc_auc_score(self.y_valid, y_prob, multi_class="ovr")
            except:
                roc_auc = None
        print(f"{model_type}:")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  F1-score: {f1:.3f}")
        if roc_auc is not None:
            print(f"  ROC-AUC: {roc_auc:.3f}")

        # log parameters and metrics in MLFlow
        model_name = type(model).__name__ + '_' + self.tag
        log_param(f"Model Type", type(model).__name__ + '_' + self.tag)
        for hyperparameter, value in model.get_params().items():
            log_param(hyperparameter, value)
        log_metric("accuracy_score", acc)
        log_metric("f1_score", f1)
        if roc_auc is not None:
            log_metric("roc_auc", roc_auc)

        # Signature (avoid int for signature warning)
        X_sig = self.X_train.copy()
        int_cols = X_sig.select_dtypes(include=["int64", "int32"]).columns
        if len(int_cols) > 0:
            X_sig[int_cols] = X_sig[int_cols].astype("float64")
        signature = infer_signature(X_sig, model.predict(X_sig))
        mlflow.sklearn.log_model(model, name=model_name, signature=signature, registered_model_name=None)

        return {
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc_auc
        }