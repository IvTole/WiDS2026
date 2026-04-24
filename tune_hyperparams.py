"""
Hyperparameter tuning for multiple models.
GridSearchCV for RandomForest, XGBoost, SVM, LogisticRegression, and KNN.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import mlflow
import mlflow.sklearn
from src.io import Dataset
from src.preprocessing import build_preprocessor
from src.config import MLFLOW_TRACKING_URL, MLFLOW_EXPERIMENT_NAME
from sklearn.pipeline import Pipeline
import json
from datetime import datetime
import os
import time

RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = 'accuracy'  # Can be: 'accuracy', 'f1_weighted', 'roc_auc', etc.
RESULTS_DIR = 'tuning_results'

os.makedirs(RESULTS_DIR, exist_ok=True)


PARAM_GRIDS = {
    'LogisticRegression': {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__solver': ['lbfgs', 'saga'],
        'model__penalty': ['l2', 'elasticnet'],
        'model__max_iter': [1000, 5000]
    },
    
    'RandomForest': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [5, 10, 15, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2']
    },
    
    'XGBoost': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [3, 5, 7, 9],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__subsample': [0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'model__reg_lambda': [0, 0.1, 1.0, 10.0]
    },
    
    'SVM': {
        'model__C': [0.1, 1, 10, 100],
        'model__kernel': ['linear', 'rbf', 'poly'],
        'model__gamma': ['scale', 'auto', 0.001, 0.01],
        'model__degree': [2, 3, 4]
    },
    
    'KNN': {
        'model__n_neighbors': [3, 5, 7, 9, 11, 15],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan', 'minkowski']
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_pipeline(model_name, model_instance):
    """Build a pipeline with preprocessing plus the model."""
    pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', model_instance)
    ])
    return pipeline

def log_results_to_mlflow(model_name, best_params, best_score, cv_results, elapsed_time):
    """Log results in MLflow."""
    
    with mlflow.start_run(run_name=f"{model_name}_tuning"):
        # Log parameters
        for key, value in best_params.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                mlflow.log_param(key, value)
        
        # Log metrics
        mlflow.log_metric("best_score", best_score)
        mlflow.log_metric("elapsed_time_seconds", elapsed_time)
        
        # Log full results as an artifact
        results_df = pd.DataFrame(cv_results)
        results_path = f"{RESULTS_DIR}/{model_name}_cv_results.csv"
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)
        
        print(f"  ✓ Results logged in MLflow (run_id: {mlflow.active_run().info.run_id})")

def tune_model(model_name, X_train, y_train):
    """
    Tune hyperparameters for a single model.

    Args:
        model_name: 'LogisticRegression', 'RandomForest', 'XGBoost', 'SVM', 'KNN'
        X_train: Training features
        y_train: Training target
    Returns:
        dict with tuning results
    """
    
    print(f"\n{'='*80}")
    print(f"  TUNING: {model_name}")
    print(f"{'='*80}")
    
    # Select model
    models = {
        'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, solver='saga', max_iter=5000),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss', verbose=0),
        'SVM': SVC(random_state=RANDOM_STATE, probability=True),
        'KNN': KNeighborsClassifier()
    }
    
    model_instance = models[model_name]
    pipeline = build_pipeline(model_name, model_instance)
    param_grid = PARAM_GRIDS[model_name]
    
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=-1,
        verbose=1
    )
    search_type = "GridSearchCV"
    
    # Run search
    start_time = time.time()
    print(f"\n  Iniciando {search_type} con {CV_FOLDS} folds...")
    search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    # Results
    best_params = search.best_params_
    best_score = search.best_score_
    
    print(f"\n  ✓ {search_type} completado en {elapsed_time:.2f}s")
    print(f"  ✓ Mejor score ({SCORING}): {best_score:.4f}")
    print(f"\n  Best parameters:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")
    
    # Log in MLflow
    log_results_to_mlflow(model_name, best_params, best_score, search.cv_results_, elapsed_time)
    
    return {
        'model_name': model_name,
        'search_type': search_type,
        'best_params': best_params,
        'best_score': best_score,
        'best_model': search.best_estimator_,
        'elapsed_time': elapsed_time,
        'cv_results': search.cv_results_
    }


def compare_all_models(results_list):
    """Create a comparison table for all models."""
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Best Score': f"{r['best_score']:.4f}",
            'Tiempo (s)': f"{r['elapsed_time']:.2f}",
            'Parameters': len(r['best_params'])
        }
        for r in results_list
    ])
    
    # Sort by score (convert to float temporarily)
    comparison_df_sort = comparison_df.copy()
    comparison_df_sort['Best Score'] = comparison_df_sort['Best Score'].astype(float)
    comparison_df_sort = comparison_df_sort.sort_values('Best Score', ascending=False)
    comparison_df = comparison_df.loc[comparison_df_sort.index]
    
    print(f"\n{'='*80}")
    print("  MODEL COMPARISON")
    print(f"{'='*80}")
    print(comparison_df.to_string(index=False))
    
    # Save table
    comparison_path = f"{RESULTS_DIR}/comparison_summary.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n  ✓ Resumen guardado en: {comparison_path}")
    
    return comparison_df

def save_best_models(results_list):
    """Save tuned models for later reference."""
    best_model_info = {
        'timestamp': datetime.now().isoformat(),
        'models': []
    }
    
    for r in results_list:
        model_info = {
            'name': r['model_name'],
            'best_score': float(r['best_score']),
            'elapsed_time': float(r['elapsed_time']),
            'best_params': {str(k).replace('model__', ''): str(v) for k, v in r['best_params'].items()}
        }
        best_model_info['models'].append(model_info)
        
        # Save model in MLflow
        mlflow.sklearn.log_model(r['best_model'], f"{r['model_name']}_best_model")
    
    # Save JSON summary
    json_path = f"{RESULTS_DIR}/best_models_summary.json"
    with open(json_path, 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    print(f"  ✓ Models saved to: {json_path}")

def main():
    print("\n" + "="*80)
    print("  HYPERPARAMETER TUNING - MULTIPLE MODELS")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    data = Dataset()
    X_train, y_train, _ = data.load_data_xy()
    print(f"   ✓ X_train shape: {X_train.shape}")
    print(f"   ✓ y_train shape: {y_train.shape}")
    print(f"   ✓ Classes: {np.unique(y_train)}")
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # List of models to tune
    models_to_tune = ['LogisticRegression', 'RandomForest', 'XGBoost', 'SVM', 'KNN']
    
    results = []
    
    print("\n2. Starting hyperparameter tuning...")
    for model_name in models_to_tune:
        result = tune_model(model_name, X_train, y_train)
        results.append(result)
    
    # Final comparison
    print("\n3. Generating comparison summary...")
    comparison = compare_all_models(results)
    
    # Save complete results
    print("\n4. Saving results...")
    save_best_models(results)
    
    print(f"\n✓ IMPLEMENTATION COMPLETED")
    print(f"\nResults saved in: {RESULTS_DIR}/")
    print(f"  - comparison_summary.csv")
    print(f"  - best_models_summary.json")
    print(f"  - *_cv_results.csv (per model)")
    print(f"\n✓ Experiments logged in MLflow at: {MLFLOW_TRACKING_URL}")
    print(f"  Experiment: {MLFLOW_EXPERIMENT_NAME}")
    
    return results, comparison

if __name__ == "__main__":
    results, comparison = main()
