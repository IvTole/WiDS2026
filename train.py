"""
=============================================================
  Random Forest
  
=============================================================
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

#RUTAS DE ARCHIVOS
BASE_URL   = "https://raw.githubusercontent.com/HectorDelgado9997/WiDs/refs/heads/main"
RUTA_TRAIN = f"{BASE_URL}/train.csv"
RUTA_TEST  = f"{BASE_URL}/test.csv"


#CARGA DE DATOS
def cargar_datos(ruta: str) -> pd.DataFrame:
    
    df = pd.read_csv(ruta)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


#CREACION VARIABLES OBJETIVO (solo para train)
def crear_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las 4 variables objetivo.
    Un incendio 'alcanza' una zona si event==1 y llega antes del horizonte.
    Solo se aplica sobre train.csv.
    """
    y = pd.DataFrame()
    y['prob_12h'] = ((df['event'] == 1) & (df['time_to_hit_hours'] <= 12)).astype(int)
    y['prob_24h'] = ((df['event'] == 1) & (df['time_to_hit_hours'] <= 24)).astype(int)
    y['prob_48h'] = ((df['event'] == 1) & (df['time_to_hit_hours'] <= 48)).astype(int)
    y['prob_72h'] = ((df['event'] == 1) & (df['time_to_hit_hours'] <= 72)).astype(int)

    print("Distribucion de targets (positivos / total):")
    for col in y.columns:
        print(f"   {col}: {y[col].sum()} positivos ({y[col].mean()*100:.1f}%)")
    print()
    return y


#CONSTRUIR PIPELINE PARA EVALUACION (un target a la vez)
def construir_pipeline_simple() -> Pipeline:
    """
    Pipeline con RandomForestClassifier simple (sin MultiOutputClassifier).
    Se usa SOLO para la validacion cruzada, evaluando un target a la vez.
    Esto evita el error de compatibilidad con cross_val_score.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])


#CONSTRUIR PIPELINE PARA ENTRENAMIENTO FINAL (todos los targets juntos)
def construir_pipeline_multioutput() -> Pipeline:
    """
    Pipeline con MultiOutputClassifier para entrenar los 4 targets simultaneamente.
    Se usa SOLO para el entrenamiento final.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        ))
    ])


#EVALUAR CON VALIDACION CRUZADA
def evaluar_modelo(X: pd.DataFrame, y: pd.DataFrame):
    """
    Evalua cada target por separado con validacion cruzada de 5 folds.
    Usa un pipeline simple (sin MultiOutputClassifier) .
    Metrica: ROC-AUC.
    """
    print("Evaluando con validacion cruzada (5 folds)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for columna in y.columns:
        pipeline_simple = construir_pipeline_simple()
        scores = cross_val_score(
            pipeline_simple, X, y[columna],
            cv=cv, scoring='roc_auc', n_jobs=-1
        )
        print(f"   {columna} - ROC-AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
    print()


#ENTRENAR MODELO FINAL Y GRAFICAR IMPORTANCIA
def entrenar_y_graficar_importancia(pipeline: Pipeline, X: pd.DataFrame, y: pd.DataFrame):
    """
    Entrena el modelo final con todos los datos de train usando MultiOutputClassifier.
    Grafica y guarda las 15 variables mas importantes segun prob_12h.
    """
    pipeline.fit(X, y)
    print("Modelo entrenado con todos los datos de entrenamiento.\n")

    primer_rf = pipeline.named_steps['model'].estimators_[0]
    importancias = pd.Series(primer_rf.feature_importances_, index=X.columns)
    top15 = importancias.sort_values(ascending=False).head(15)

    print("Top 10 variables mas importantes (segun prob_12h):")
    print(top15.head(10).to_string())
    print()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top15.values, y=top15.index, palette='viridis')
    plt.title("Feature Importance - Random Forest")
    plt.xlabel("Importancia media")
    plt.tight_layout()
    plt.show()
    
    return pipeline


# 9. GENERAR SUBMISSION (usando test.csv)
def generar_submission(pipeline: Pipeline, X_test: pd.DataFrame, event_ids: pd.Series):
    """
    Genera y muestra las predicciones sobre test.csv
    RECORDAR: siempre predecir sobre test.csv, nunca sobre train.csv.
    """
    probabilidades = pipeline.predict_proba(X_test)

    submission = pd.DataFrame({'event_id': event_ids})
    nombres = ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']

    for i, nombre in enumerate(nombres):
        submission[nombre] = probabilidades[i][:, 1]

    # Se muestra en pantalla
    print("Predicciones generadas:")
    print(submission)
    return submission


# 10. FUNCION PRINCIPAL
def main():
    print("=" * 60)
    print("  Random Forest MultiOutput")
    print("=" * 60 + "\n")

    # Paso 1: Cargar train y test
    df_train = cargar_datos(RUTA_TRAIN)
    df_test  = cargar_datos(RUTA_TEST)

    # Paso 2: Crear targets (solo desde train)
    y = crear_targets(df_train)

    # Paso 3: Preparar features
    X_train = df_train.drop(columns=['event_id', 'event', 'time_to_hit_hours'])
    X_test  = df_test.drop(columns=['event_id'])

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}\n")

    # Paso 4: Evaluar con validacion cruzada (pipeline simple, un target a la vez)
    evaluar_modelo(X_train, y)

    # Paso 5: Entrenar modelo final (pipeline multioutput, todos los targets juntos)
    pipeline_final = construir_pipeline_multioutput()
    pipeline_final = entrenar_y_graficar_importancia(pipeline_final, X_train, y)

    # Paso 6: Generar submission con test.csv
    generar_submission(pipeline_final, X_test, df_test['event_id'])

    print("\nProceso completado!")


if __name__ == "__main__":
    main()