"""
Hyperparameter tuning for multiple models.
HalvingGridSearchCV for RandomForest, XGBoost, SVM, LogisticRegression, and KNN.
"""

import warnings
import pandas as pd
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import get_scorer
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
import argparse

RANDOM_STATE = 42
CV_FOLDS = 5
RESULTS_DIR = 'tuning_results'

os.makedirs(RESULTS_DIR, exist_ok=True)


PARAM_GRIDS = {
    'LogisticRegression': {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__solver': ['lbfgs', 'saga'],
        'model__penalty': ['l2']
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

def build_pipeline(model_instance):
    """Build a pipeline with preprocessing plus the model."""
    pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('model', model_instance)
    ])
    return pipeline

def tune_model(model_name, X_train, y_train, X_val, y_val, scoring):
    """
    Tune hyperparameters for a single model using HalvingGridSearchCV.

    Args:
        model_name: 'LogisticRegression', 'RandomForest', 'XGBoost', 'SVM', 'KNN'
        X_train: Training features
        y_train: Training target
        X_val: Validation features for holdout evaluation
        y_val: Validation target for holdout evaluation
        scoring: Scoring metric
    Returns:
        dict with tuning results
    """
    
    print(f"\n{'='*80}")
    print(f"  TUNING: {model_name}")
    print(f"{'='*80}")
    
    # Select model
    models = {
        'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=5000),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss', verbosity=0),
        'SVM': SVC(random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(n_jobs=-1)
    }
    
    model_instance = models[model_name]
    pipeline = build_pipeline(model_instance)
    param_grid = PARAM_GRIDS[model_name]
    
    cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    search = HalvingGridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        factor=3
    )
    search_type = "HalvingGridSearchCV"
    
    with mlflow.start_run(run_name=f"{model_name}_tuning"):
        # Log basic config
        mlflow.log_param("search_type", search_type)
        mlflow.log_param("scoring", scoring)
        
        # Run search
        start_time = time.time()
        print(f"\n  Starting {search_type} with {CV_FOLDS} folds...")
        search.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        # Results
        best_params = search.best_params_
        best_score = search.best_score_
        best_model = search.best_estimator_

        print(f"Best CV score ({scoring}): {best_score:.4f}")
        print(f"\nBest parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
            if isinstance(value, (int, float, str, bool, type(None))):
                mlflow.log_param(param, value)
                
        # Evaluate on holdout set
        scorer = get_scorer(scoring)
        val_score = scorer(best_model, X_val, y_val)
        print(f"Holdout Validation Score ({scoring}): {val_score:.4f}")
        
        # Log metrics safely
        import math
        if not math.isnan(best_score):
            mlflow.log_metric("final_best_cv_score", best_score)
        if not math.isnan(val_score):
            mlflow.log_metric("final_holdout_score", val_score)
        mlflow.log_metric("final_elapsed_time_sec", elapsed_time)
        
        # Log full results as an artifact
        results_df = pd.DataFrame(search.cv_results_)
        results_path = f"{RESULTS_DIR}/{model_name}_cv_results.csv"
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)
        
        # Log model inside the correct MLflow run
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Saving scikit-learn models in the pickle or cloudpickle format",
                category=FutureWarning,
            )
            mlflow.sklearn.log_model(best_model, name=f"{model_name}_best_model")
        
        print(f"  Results and model logged in MLflow (run_id: {mlflow.active_run().info.run_id})")
        
    return {
        'model_name': model_name,
        'search_type': search_type,
        'best_params': best_params,
        'best_cv_score': best_score,
        'holdout_val_score': val_score,
        'best_model': best_model,
        'elapsed_time': elapsed_time,
        'cv_results': search.cv_results_
    }


def compare_all_models(results_list, scoring):
    """Create a comparison table for all models."""
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            f'Best CV Score ({scoring})': f"{r['best_cv_score']:.4f}",
            f'Holdout Score ({scoring})': f"{r['holdout_val_score']:.4f}",
            'Time (s)': f"{r['elapsed_time']:.2f}",
            'Parameters': len(r['best_params'])
        }
        for r in results_list
    ])
    
    # Sort by holdout score
    comparison_df_sort = comparison_df.copy()
    comparison_df_sort[f'Holdout Score ({scoring})'] = comparison_df_sort[f'Holdout Score ({scoring})'].astype(float)
    comparison_df_sort = comparison_df_sort.sort_values(f'Holdout Score ({scoring})', ascending=False)
    comparison_df = comparison_df.loc[comparison_df_sort.index]
    
    print(f"\n{'='*80}")
    print("  MODEL COMPARISON")
    print(f"{'='*80}")
    print(comparison_df.to_string(index=False))
    
    # Save table
    comparison_path = f"{RESULTS_DIR}/comparison_summary.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n  ✓ Summary saved to: {comparison_path}")
    
    return comparison_df

def save_best_models_summary(results_list):
    """Save tuned models summary for later reference."""
    best_model_info = {
        'timestamp': datetime.now().isoformat(),
        'models': []
    }
    
    for r in results_list:
        model_info = {
            'name': r['model_name'],
            'best_cv_score': float(r['best_cv_score']),
            'holdout_val_score': float(r['holdout_val_score']),
            'elapsed_time': float(r['elapsed_time']),
            'best_params': {str(k).replace('model__', ''): str(v) for k, v in r['best_params'].items()}
        }
        best_model_info['models'].append(model_info)
    
    # Save JSON summary
    json_path = f"{RESULTS_DIR}/best_models_summary.json"
    with open(json_path, 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    print(f"  Models summary saved to: {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with HalvingGridSearchCV")
    parser.add_argument("--models", nargs="+", 
                        default=['LogisticRegression', 'RandomForest', 'XGBoost', 'SVM', 'KNN'],
                        help="List of models to tune (e.g., --models XGBoost RandomForest)")
    parser.add_argument("--scoring", type=str, default="f1_macro",
                        help="Scoring metric to use for evaluation (default: f1_macro)")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("  HYPERPARAMETER TUNING - MULTIPLE MODELS")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    data = Dataset()
    X, y, _ = data.load_data_xy()
    
    # Remove classes with fewer than CV_FOLDS samples to prevent CV and XGBoost failures
    class_counts = pd.Series(y).value_counts()
    valid_classes = class_counts[class_counts >= CV_FOLDS].index
    if len(valid_classes) < len(class_counts):
        print(f"   Warning: Dropping classes with fewer than {CV_FOLDS} samples: {set(class_counts.index) - set(valid_classes)}")
        mask = np.isin(y, valid_classes)
        X = X[mask]
        y = y[mask]
        
    # Remap remaining classes to be strictly sequential (0, 1, 2...) for XGBoost compatibility
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"   Remapped valid classes to sequential indices: {np.unique(y)}")
        
    # Create holdout validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   X_val shape: {X_val.shape}")
    print(f"   y_val shape: {y_val.shape}")
    print(f"   Classes: {np.unique(y_train)}")
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    results = []
    
    print(f"\n2. Starting hyperparameter tuning for models: {args.models}...")
    for model_name in args.models:
        if model_name not in PARAM_GRIDS:
            print(f"Warning: Model '{model_name}' not found in PARAM_GRIDS. Skipping.")
            continue
        result = tune_model(model_name, X_train, y_train, X_val, y_val, args.scoring)
        results.append(result)
    
    if not results:
        print("No models were tuned. Exiting.")
        return
        
    # Final comparison
    print("\n3. Generating comparison summary...")
    comparison = compare_all_models(results, args.scoring)
    
    # Save complete results
    print("\n4. Saving results...")
    save_best_models_summary(results)
    
    print(f"\n IMPLEMENTATION COMPLETED")
    print(f"\nResults saved in: {RESULTS_DIR}/")
    print(f"  - comparison_summary.csv")
    print(f"  - best_models_summary.json")
    print(f"  - *_cv_results.csv (per model)")
    print(f"\n Experiments logged in MLflow at: {MLFLOW_TRACKING_URL}")
    print(f"  Experiment: {MLFLOW_EXPERIMENT_NAME}")
    
    return results, comparison

if __name__ == "__main__":
    results, comparison = main()
